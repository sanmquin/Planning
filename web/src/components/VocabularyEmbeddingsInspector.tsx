import { useState, useMemo } from 'react';
import modelsWeightsData from '../data/models_weights.json';
import { Layers, Sparkles } from 'lucide-react';

interface VocabularyEmbeddingsInspectorProps {
  selectedEpoch?: "300" | "400" | "500";
  onSelectEpoch?: (epoch: "300" | "400" | "500") => void;
}

export default function VocabularyEmbeddingsInspector({
  selectedEpoch = "500",
  onSelectEpoch
}: VocabularyEmbeddingsInspectorProps) {
  const [activeCheckpoint, setActiveCheckpoint] = useState<"300" | "400" | "500">(selectedEpoch);
  const currentEpoch = onSelectEpoch ? selectedEpoch : activeCheckpoint;

  const handleEpochChange = (epoch: "300" | "400" | "500") => {
    setActiveCheckpoint(epoch);
    if (onSelectEpoch) onSelectEpoch(epoch);
  };

  const [selectedToken, setSelectedToken] = useState<number>(0);
  const [hoveredSimCell, setHoveredSimCell] = useState<{ t1: number; t2: number; sim: number } | null>(null);

  const rawWeights = (modelsWeightsData as any)[currentEpoch] || {};
  const tokenEmbeddings: number[][] = rawWeights["token_embedding.weight"] || []; // [42 x 16]

  const vocabSize = tokenEmbeddings.length || 42;

  // Compute 42x42 Cosine Similarity Matrix
  const similarityMatrix: number[][] = useMemo(() => {
    if (!tokenEmbeddings || tokenEmbeddings.length === 0) return [];

    const matrix: number[][] = [];
    const norms = tokenEmbeddings.map(vec =>
      Math.sqrt(vec.reduce((a, b) => a + b * b, 0)) || 1.0
    );

    for (let i = 0; i < vocabSize; i++) {
      const row: number[] = [];
      const vecI = tokenEmbeddings[i];
      const normI = norms[i];

      for (let j = 0; j < vocabSize; j++) {
        const vecJ = tokenEmbeddings[j];
        const normJ = norms[j];

        let dot = 0;
        for (let d = 0; d < 16; d++) {
          dot += vecI[d] * vecJ[d];
        }
        const sim = dot / (normI * normJ);
        row.push(sim);
      }
      matrix.push(row);
    }
    return matrix;
  }, [tokenEmbeddings, vocabSize]);

  // Compute 2D PCA Projection of 16-dim vectors for 2D Scatter plot
  const pcaPoints = useMemo(() => {
    if (!tokenEmbeddings || tokenEmbeddings.length === 0) return [];

    const numVecs = tokenEmbeddings.length;
    const dim = 16;

    // Mean-center data
    const mean = new Array(dim).fill(0);
    for (let i = 0; i < numVecs; i++) {
      for (let d = 0; d < dim; d++) mean[d] += tokenEmbeddings[i][d];
    }
    for (let d = 0; d < dim; d++) mean[d] /= numVecs;

    const centered = tokenEmbeddings.map(vec => vec.map((val, d) => val - mean[d]));

    // Power Iteration to get top 2 principal components
    const getTopEigenvector = (data: number[][], iterations = 20): number[] => {
      let vec: number[] = new Array(dim).fill(0).map((_, idx) => (idx % 2 === 0 ? 1 : -1));
      let norm = Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
      vec = vec.map(x => x / norm);

      for (let iter = 0; iter < iterations; iter++) {
        const w = new Array(dim).fill(0);
        for (let i = 0; i < numVecs; i++) {
          let dot = 0;
          for (let d = 0; d < dim; d++) dot += data[i][d] * vec[d];
          for (let d = 0; d < dim; d++) w[d] += data[i][d] * dot;
        }
        norm = Math.sqrt(w.reduce((a, b) => a + b * b, 0)) || 1.0;
        vec = w.map(x => x / norm);
      }
      return vec;
    };

    const pc1 = getTopEigenvector(centered, 20);

    // Deflate centered data for PC2
    const centered2 = centered.map(v => {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += v[d] * pc1[d];
      return v.map((val, d) => val - dot * pc1[d]);
    });

    const pc2 = getTopEigenvector(centered2, 20);

    // Project centered vectors onto PC1 and PC2
    const points = centered.map((vec, idx) => {
      let x = 0;
      let y = 0;
      for (let d = 0; d < dim; d++) {
        x += vec[d] * pc1[d];
        y += vec[d] * pc2[d];
      }
      return { token: idx, x, y };
    });

    // Normalize x and y to [30, 470] canvas bounds
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs) || 1;
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys) || 1;

    return points.map(p => ({
      token: p.token,
      cx: 40 + ((p.x - minX) / (maxX - minX || 1)) * 420,
      cy: 40 + ((p.y - minY) / (maxY - minY || 1)) * 320
    }));
  }, [tokenEmbeddings]);

  const getTokenLabel = (tok: number) => {
    if (tok === 40) return 'PAD (40)';
    if (tok === 41) return 'STOP (41)';
    if (tok < 20) return `Node ${tok}`;
    return `Tok ${tok}`;
  };

  const getTokenBadgeColor = (tok: number) => {
    if (tok === 40) return 'bg-zinc-800 text-zinc-400 border-zinc-700';
    if (tok === 41) return 'bg-rose-950 text-rose-300 border-rose-600/50';
    if (tok < 20) return 'bg-indigo-950 text-indigo-300 border-indigo-500/40';
    return 'bg-amber-950 text-amber-300 border-amber-500/40';
  };

  const activeVector = tokenEmbeddings[selectedToken] || [];
  const activeNorm = Math.sqrt(activeVector.reduce((a, b) => a + b * b, 0));

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-5 font-sans">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            Vocabulary Token Embeddings &amp; Geometric Representation Space
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            42 Vocabulary Embeddings (16-dim vectors). Graph Nodes (0–19), PAD (40), STOP (41).
          </p>
        </div>

        {/* Checkpoint Selector */}
        <div className="flex items-center gap-2 bg-zinc-950 p-1.5 rounded-lg border border-zinc-800 text-xs font-mono">
          <span className="text-zinc-500 px-1 text-[11px] flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-indigo-400" /> Checkpoint:
          </span>
          {(["300", "400", "500"] as const).map(ep => (
            <button
              key={ep}
              onClick={() => handleEpochChange(ep)}
              className={`px-3 py-1 rounded transition-all font-bold ${
                currentEpoch === ep
                  ? 'bg-indigo-600 text-white shadow-[0_0_10px_rgba(99,102,241,0.4)]'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Epoch {ep}
            </button>
          ))}
        </div>
      </div>

      {/* Main 2-column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left Column: PCA 2D Scatter Plot */}
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-3 font-mono">
          <div className="flex items-center justify-between text-xs text-zinc-300 border-b border-zinc-850 pb-2">
            <span className="font-bold text-indigo-300">
              2D PCA Representation Space (Top Principal Components)
            </span>
            <span className="text-[10px] text-zinc-500">
              Vocab Size = {vocabSize}
            </span>
          </div>

          <div className="relative bg-zinc-900/80 rounded-lg border border-zinc-800 aspect-[5/4] max-h-[380px] overflow-hidden flex items-center justify-center p-2">
            <svg viewBox="0 0 500 400" className="w-full h-full">
              {/* Axis grid lines */}
              <line x1="250" y1="20" x2="250" y2="380" stroke="#3f3f46" strokeWidth="1" strokeDasharray="3,3" />
              <line x1="20" y1="200" x2="480" y2="200" stroke="#3f3f46" strokeWidth="1" strokeDasharray="3,3" />

              {/* Scatter Plot Points */}
              {pcaPoints.map(pt => {
                const isSelected = selectedToken === pt.token;
                const isPad = pt.token === 40;
                const isStop = pt.token === 41;
                const isNode = pt.token < 20;

                let fill = '#a1a1aa';
                let radius = 10;
                if (isNode) fill = '#818cf8';
                else if (isPad) fill = '#52525b';
                else if (isStop) fill = '#f43f5e';
                else fill = '#fbbf24';

                if (isSelected) radius = 14;

                return (
                  <g
                    key={pt.token}
                    onClick={() => setSelectedToken(pt.token)}
                    className="cursor-pointer transition-transform duration-150"
                  >
                    {isSelected && (
                      <circle
                        cx={pt.cx}
                        cy={pt.cy}
                        r={radius + 6}
                        fill={fill}
                        opacity="0.3"
                        className="animate-pulse"
                      />
                    )}

                    <circle
                      cx={pt.cx}
                      cy={pt.cy}
                      r={radius}
                      fill={fill}
                      stroke={isSelected ? '#ffffff' : '#18181b'}
                      strokeWidth={isSelected ? 2.5 : 1.5}
                    />

                    <text
                      x={pt.cx}
                      y={pt.cy + 3.5}
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize={isSelected ? "11" : "9"}
                      fontWeight="bold"
                      pointerEvents="none"
                    >
                      {pt.token}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Legend Overlay */}
            <div className="absolute bottom-2 left-2 bg-zinc-950/90 border border-zinc-800 rounded p-2 text-[10px] space-y-1 font-mono">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
                <span className="text-zinc-300">Graph Nodes (0–19)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                <span className="text-zinc-300">STOP Token (41)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-zinc-600"></span>
                <span className="text-zinc-300">PAD Token (40)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Selected Token Vector & Details */}
        <div className="space-y-4 font-mono">
          {/* Token Selector Toolbar */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-300">
              <span className="font-bold">Select Token to Inspect Vector:</span>
              <span className={`px-2 py-0.5 rounded border text-[11px] font-bold ${getTokenBadgeColor(selectedToken)}`}>
                {getTokenLabel(selectedToken)}
              </span>
            </div>

            <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto custom-scrollbar p-1">
              {Array.from({ length: vocabSize }).map((_, tok) => {
                const isSelected = selectedToken === tok;
                return (
                  <button
                    key={tok}
                    onClick={() => setSelectedToken(tok)}
                    className={`px-2 py-0.5 rounded text-[10px] transition-all ${
                      isSelected
                        ? 'bg-indigo-600 text-white font-bold ring-1 ring-white'
                        : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {tok}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 16-dim Vector Bars */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between text-xs border-b border-zinc-850 pb-2">
              <span className="font-bold text-indigo-300">
                Embedding Vector E[Token #{selectedToken}] (16-dim):
              </span>
              <span className="text-[10px] text-zinc-400">
                L2 Norm = {activeNorm.toFixed(4)}
              </span>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5">
              {activeVector.map((val, idx) => {
                const absNorm = Math.min(Math.abs(val) / 1.5, 1.0);
                const isPos = val >= 0;

                return (
                  <div key={idx} className="flex flex-col items-center bg-zinc-900 p-1.5 rounded border border-zinc-800">
                    <span className="text-[9px] text-zinc-500 mb-1">d={idx}</span>
                    <div className="w-full bg-zinc-950 h-10 rounded overflow-hidden flex flex-col justify-end p-0.5 border border-zinc-850">
                      <div
                        style={{ height: `${Math.max(12, absNorm * 100)}%` }}
                        className={`w-full rounded-sm transition-all ${
                          isPos ? 'bg-indigo-500' : 'bg-rose-500'
                        }`}
                      />
                    </div>
                    <span className="text-[9px] font-bold text-zinc-200 mt-1">{val.toFixed(3)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* 42x42 Cosine Similarity Heatmap Section */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-3 font-mono">
        <div className="flex items-center justify-between text-xs text-zinc-300 border-b border-zinc-850 pb-2">
          <span className="font-bold text-indigo-300">
            42 × 42 Token Embedding Cosine Similarity Matrix
          </span>
          {hoveredSimCell ? (
            <span className="text-violet-300 font-bold text-[11px]">
              Sim({getTokenLabel(hoveredSimCell.t1)}, {getTokenLabel(hoveredSimCell.t2)}) = {(hoveredSimCell.sim * 100).toFixed(2)}%
            </span>
          ) : (
            <span className="text-[10px] text-zinc-500">
              Hover over matrix cells to inspect pairwise token similarity scores
            </span>
          )}
        </div>

        <div className="overflow-auto max-h-[320px] custom-scrollbar p-1">
          {similarityMatrix.length > 0 && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${vocabSize}, minmax(10px, 1fr))`
              }}
              className="gap-0.5"
            >
              {similarityMatrix.map((row, rIdx) =>
                row.map((val, cIdx) => {
                  const normSim = (val + 1) / 2; // Map [-1, 1] to [0, 1]
                  const color = val >= 0
                    ? `rgba(99, 102, 241, ${0.1 + normSim * 0.85})`
                    : `rgba(244, 63, 94, ${0.1 + (1 - normSim) * 0.85})`;

                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      onMouseEnter={() => setHoveredSimCell({ t1: rIdx, t2: cIdx, sim: val })}
                      onMouseLeave={() => setHoveredSimCell(null)}
                      style={{ backgroundColor: color }}
                      className="h-3.5 rounded-[1px] cursor-pointer hover:scale-125 transition-transform duration-100"
                      title={`Sim(${rIdx}, ${cIdx}): ${val.toFixed(4)}`}
                    />
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
