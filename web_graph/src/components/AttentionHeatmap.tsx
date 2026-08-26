import { useState } from 'react';
import { AutoregressiveStepTrace, GraphSample } from '../model/graph_transformer';
import { Flame } from 'lucide-react';

interface AttentionHeatmapProps {
  stepTrace: AutoregressiveStepTrace;
  sample: GraphSample;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  selectedHead: number;
  onSelectHead: (head: number) => void;
  attentionType: 'cross' | 'decoder_self' | 'encoder_self';
  onSelectType: (type: 'cross' | 'decoder_self' | 'encoder_self') => void;
}

export default function AttentionHeatmap({
  stepTrace,
  sample,
  selectedLayer,
  onSelectLayer,
  selectedHead,
  onSelectHead,
  attentionType,
  onSelectType
}: AttentionHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    q: number;
    k: number;
    val: number;
  } | null>(null);

  let matrix: number[][] = [];
  if (attentionType === 'cross') {
    const layerData = stepTrace.decoderCrossAttn[selectedLayer];
    matrix = layerData?.crossAttnHeads?.[selectedHead]?.attnWeights || [];
  } else if (attentionType === 'decoder_self') {
    const layerData = stepTrace.decoderSelfAttn[selectedLayer];
    matrix = layerData?.selfAttnHeads?.[selectedHead]?.attnWeights || [];
  } else {
    const layerData = stepTrace.encoderSelfAttn[selectedLayer];
    matrix = layerData?.selfAttnHeads?.[selectedHead]?.attnWeights || [];
  }

  const qLen = matrix.length;
  const kLen = matrix[0]?.length || 0;

  const srcTokens = sample.trace;
  const currSeq = stepTrace.currTgtSeq;

  const getQTokenLabel = (idx: number) => {
    if (attentionType === 'encoder_self') {
      return idx < srcTokens.length ? `T[${idx}]=${srcTokens[idx]}` : `PAD[${idx}]`;
    }
    return idx < currSeq.length ? `P[${idx}]=${currSeq[idx]}` : `P[${idx}]`;
  };

  const getKTokenLabel = (idx: number) => {
    if (attentionType === 'decoder_self') {
      return idx < currSeq.length ? `P[${idx}]=${currSeq[idx]}` : `P[${idx}]`;
    }
    return idx < srcTokens.length ? `T[${idx}]=${srcTokens[idx]}` : `PAD[${idx}]`;
  };

  const getCellColor = (val: number) => {
    const opacity = Math.min(Math.max(val, 0), 1);
    if (opacity < 0.05) return 'rgba(30, 27, 75, 0.2)';
    if (opacity < 0.2) return `rgba(99, 102, 241, ${0.2 + opacity * 0.4})`;
    if (opacity < 0.5) return `rgba(139, 92, 246, ${0.5 + opacity * 0.4})`;
    return `rgba(236, 72, 153, ${0.8 + opacity * 0.2})`;
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-5">
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Flame className="w-5 h-5 text-amber-400" />
            Attention Matrix Heatmap Visualizer
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Softmax attention score A = softmax(QK^T / √d_k) routing predictions step-by-step.
          </p>
        </div>

        {/* Controls Toggles */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Attention Type Selector */}
          <div className="flex items-center bg-zinc-950 p-1 rounded-lg border border-zinc-800">
            <button
              onClick={() => onSelectType('cross')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                attentionType === 'cross'
                  ? 'bg-indigo-600 text-white font-bold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Cross-Attention (Decoder &rarr; DFS Trace)
            </button>
            <button
              onClick={() => onSelectType('decoder_self')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                attentionType === 'decoder_self'
                  ? 'bg-indigo-600 text-white font-bold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Decoder Self-Attention
            </button>
            <button
              onClick={() => onSelectType('encoder_self')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                attentionType === 'encoder_self'
                  ? 'bg-indigo-600 text-white font-bold shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Encoder Self-Attention
            </button>
          </div>

          {/* Layer Selector */}
          <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 font-mono">
            <span className="text-zinc-500 px-1 text-[11px]">Layer:</span>
            {[0, 1].map(l => (
              <button
                key={l}
                onClick={() => onSelectLayer(l)}
                className={`px-2.5 py-0.5 rounded transition-all ${
                  selectedLayer === l
                    ? 'bg-violet-600 text-white font-bold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                L{l + 1}
              </button>
            ))}
          </div>

          {/* Head Selector */}
          <div className="flex items-center gap-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 font-mono">
            <span className="text-zinc-500 px-1 text-[11px]">Head:</span>
            {[0, 1, 2, 3].map(h => (
              <button
                key={h}
                onClick={() => onSelectHead(h)}
                className={`px-2 py-0.5 rounded transition-all ${
                  selectedHead === h
                    ? 'bg-amber-500 text-zinc-950 font-bold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                H{h}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap Grid Section */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs font-mono text-zinc-400">
          <span>Query Pos (Q) &darr; vs Key Tokens (K) &rarr;</span>
          <span className="text-[11px] text-zinc-500">
            Matrix Dimensions: {qLen} x {kLen}
          </span>
        </div>

        {/* Scrollable Container with Explicit CSS Inline Grid */}
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 overflow-x-auto custom-scrollbar">
          {matrix.length > 0 && kLen > 0 ? (
            <div className="min-w-[500px] space-y-1">
              {/* Key Column Labels Header */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: `80px repeat(${kLen}, minmax(0, 1fr))`
                }}
                className="gap-1 text-[9px] font-mono text-zinc-500 text-center pb-1 border-b border-zinc-850"
              >
                <div className="text-left pl-1">Q \ K</div>
                {Array.from({ length: kLen }).map((_, kIdx) => (
                  <div key={kIdx} className="truncate" title={getKTokenLabel(kIdx)}>
                    {kIdx < srcTokens.length ? srcTokens[kIdx] : '.'}
                  </div>
                ))}
              </div>

              {/* Matrix Rows */}
              {matrix.map((row, qIdx) => (
                <div
                  key={qIdx}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: `80px repeat(${kLen}, minmax(0, 1fr))`
                  }}
                  className="gap-1 items-center"
                >
                  {/* Row Query Title */}
                  <div className="text-[10px] font-mono text-zinc-400 truncate pr-1" title={getQTokenLabel(qIdx)}>
                    {getQTokenLabel(qIdx)}
                  </div>

                  {/* Row Cell Values */}
                  {row.map((val, kIdx) => {
                    const isHovered = hoveredCell?.q === qIdx && hoveredCell?.k === kIdx;

                    return (
                      <div
                        key={kIdx}
                        onMouseEnter={() => setHoveredCell({ q: qIdx, k: kIdx, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{
                          backgroundColor: getCellColor(val)
                        }}
                        className={`aspect-square rounded-[2px] cursor-pointer transition-all duration-100 ${
                          isHovered
                            ? 'ring-2 ring-white scale-125 z-20 shadow-[0_0_10px_rgba(255,255,255,0.8)]'
                            : 'hover:opacity-90'
                        }`}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          ) : (
            <div className="p-8 text-center text-zinc-500 text-xs italic">
              No attention trace available for current configuration.
            </div>
          )}
        </div>
      </div>

      {/* Dynamic Hover Details Bar */}
      <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-3 min-h-[56px] flex items-center justify-between transition-all font-mono text-xs">
        {hoveredCell ? (
          <div className="flex items-center justify-between w-full">
            <div className="space-y-0.5">
              <p className="text-zinc-200 font-bold">
                Query {getQTokenLabel(hoveredCell.q)} &rarr; Key {getKTokenLabel(hoveredCell.k)}
              </p>
              <p className="text-[11px] text-zinc-400">
                Layer {selectedLayer + 1}, Head {selectedHead} ({attentionType})
              </p>
            </div>
            <div className="text-right">
              <span className="text-base font-bold text-violet-400">
                {(hoveredCell.val * 100).toFixed(2)}%
              </span>
              <p className="text-[9px] text-zinc-500 uppercase tracking-wider">Softmax Attention</p>
            </div>
          </div>
        ) : (
          <div className="text-zinc-500 text-xs italic flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse"></span>
            Hover over any cell in the attention matrix to inspect query-key softmax scores.
          </div>
        )}
      </div>
    </div>
  );
}
