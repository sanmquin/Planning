import { useState } from 'react';
import { GraphSample } from '../model/graph_transformer';
import { Network, Activity } from 'lucide-react';

interface GraphVisualizerProps {
  sample: GraphSample;
  currentStep: number;
  predictedSP: number[];
  groundTruthSP: number[];
  modelEpoch: "300" | "400" | "500";
}

export default function GraphVisualizer({
  sample,
  currentStep,
  predictedSP,
  groundTruthSP,
  modelEpoch
}: GraphVisualizerProps) {
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  const startNode = groundTruthSP[0];
  const goalNode = groundTruthSP[groundTruthSP.length - 1];

  const activeRollout = predictedSP.slice(0, currentStep + 1);
  const activeNodeSet = new Set(activeRollout);
  const gtNodeSet = new Set(groundTruthSP);

  const mapCoords = (node: number) => {
    const raw = sample.node_coords[String(node)] || [0, 0];
    const x = 250 + raw[0] * 180;
    const y = 200 + raw[1] * 140;
    return { x, y };
  };

  const edgeSet = new Set(
    sample.edges.flatMap(([u, v]) => [`${u}-${v}`, `${v}-${u}`])
  );

  const predEdges: [number, number][] = [];
  for (let i = 0; i < activeRollout.length - 1; i++) {
    predEdges.push([activeRollout[i], activeRollout[i + 1]]);
  }

  const gtEdges: [number, number][] = [];
  for (let i = 0; i < groundTruthSP.length - 1; i++) {
    gtEdges.push([groundTruthSP[i], groundTruthSP[i + 1]]);
  }

  const isExact = predictedSP.length === groundTruthSP.length &&
    predictedSP.every((val, idx) => val === groundTruthSP[idx]);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Network className="w-5 h-5 text-cyan-400" />
            2D Graph Shortest Path Layout
            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 font-mono">
              Sample #{sample.id}
            </span>
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Visualization of graph network G=(V, E) with DFS trace and step-by-step model predictions.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className={`px-2.5 py-1 rounded-md border font-bold ${
            isExact
              ? 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300'
              : 'bg-amber-950/80 border-amber-500/50 text-amber-300'
          }`}>
            Epoch {modelEpoch}: {isExact ? 'Exact Match ✓' : 'Rollout Deviation ⚠'}
          </span>
        </div>
      </div>

      {/* SVG Canvas Container */}
      <div className="relative bg-zinc-950 rounded-lg border border-zinc-850 p-2 overflow-hidden aspect-[16/9] max-h-[420px] flex items-center justify-center">
        <svg viewBox="0 0 500 400" className="w-full h-full">
          <defs>
            <filter id="glow-start" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-goal" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* 1. Base Graph Edges */}
          {sample.edges.map(([u, v], idx) => {
            const p1 = mapCoords(u);
            const p2 = mapCoords(v);
            return (
              <line
                key={`edge-${idx}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke="#3f3f46"
                strokeWidth="1.5"
                strokeOpacity="0.6"
              />
            );
          })}

          {/* 2. Ground-Truth Shortest Path (Dashed Blue Line) */}
          {gtEdges.map(([u, v], idx) => {
            const p1 = mapCoords(u);
            const p2 = mapCoords(v);
            return (
              <line
                key={`gt-edge-${idx}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke="#38bdf8"
                strokeWidth="3.5"
                strokeDasharray="5,4"
                strokeOpacity="0.8"
              />
            );
          })}

          {/* 3. Active Predicted Path (Solid Lime/Cyan Line) */}
          {predEdges.map(([u, v], idx) => {
            const p1 = mapCoords(u);
            const p2 = mapCoords(v);
            const isValidEdge = edgeSet.has(`${u}-${v}`);
            return (
              <line
                key={`pred-edge-${idx}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke={isValidEdge ? '#a3e635' : '#f43f5e'}
                strokeWidth="4"
                strokeLinecap="round"
              />
            );
          })}

          {/* 4. Graph Nodes */}
          {sample.nodes.map(node => {
            const pos = mapCoords(node);
            const isStart = node === startNode;
            const isGoal = node === goalNode;
            const isActiveInRollout = activeNodeSet.has(node);
            const isCurrentHead = activeRollout[activeRollout.length - 1] === node;
            const isHovered = hoveredNode === node;

            let fillColor = '#27272a';
            let strokeColor = '#52525b';
            let radius = 14;

            if (isStart) {
              fillColor = '#22c55e';
              strokeColor = '#4ade80';
              radius = 18;
            } else if (isGoal) {
              fillColor = '#e11d48';
              strokeColor = '#fb7185';
              radius = 18;
            } else if (isCurrentHead) {
              fillColor = '#a3e635';
              strokeColor = '#facc15';
              radius = 18;
            } else if (isActiveInRollout) {
              fillColor = '#06b6d4';
              strokeColor = '#67e8f9';
              radius = 15;
            } else if (gtNodeSet.has(node)) {
              fillColor = '#1e293b';
              strokeColor = '#0284c7';
            }

            return (
              <g
                key={`node-${node}`}
                onMouseEnter={() => setHoveredNode(node)}
                onMouseLeave={() => setHoveredNode(null)}
                className="cursor-pointer transition-transform duration-150"
              >
                {(isStart || isGoal || isCurrentHead) && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={radius + 5}
                    fill={isStart ? '#22c55e' : isGoal ? '#e11d48' : '#a3e635'}
                    opacity="0.35"
                    className="animate-pulse"
                  />
                )}

                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={radius}
                  fill={fillColor}
                  stroke={strokeColor}
                  strokeWidth={isHovered ? 3 : 2}
                />

                <text
                  x={pos.x}
                  y={pos.y + 4}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize={radius > 15 ? "12" : "10"}
                  fontWeight="bold"
                  pointerEvents="none"
                >
                  {node}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend Overlay */}
        <div className="absolute top-3 left-3 bg-zinc-950/90 border border-zinc-800 rounded-lg p-2.5 text-[10px] space-y-1.5 font-mono">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span>
            <span className="text-zinc-300">Start Node ({startNode})</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-rose-600 inline-block"></span>
            <span className="text-zinc-300">Goal Node ({goalNode})</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-1 bg-lime-400 inline-block"></span>
            <span className="text-zinc-300">Model Predicted Path</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-1 border-b-2 border-dashed border-sky-400 inline-block"></span>
            <span className="text-zinc-300">Ground-Truth Path</span>
          </div>
        </div>

        {/* Hover Node Tooltip */}
        {hoveredNode !== null && (
          <div className="absolute bottom-3 right-3 bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-xs space-y-1 shadow-lg max-w-xs">
            <div className="font-bold text-zinc-100 flex items-center justify-between">
              <span>Node ID #{hoveredNode}</span>
              <span className="text-[10px] text-indigo-400 font-mono">
                Original ID: {sample.mapping[String(hoveredNode)] ?? 'N/A'}
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Regressions Induced: <strong className="text-violet-300">{sample.node_backtraces[String(hoveredNode)] ?? 0} times</strong>
            </p>
            <p className="text-[10px] text-zinc-500">
              In Ground-Truth Path: {gtNodeSet.has(hoveredNode) ? 'Yes ✓' : 'No ✗'}
            </p>
          </div>
        )}
      </div>

      {/* Traversal Trace Details */}
      <div className="bg-zinc-950/80 border border-zinc-850 rounded-lg p-3.5 text-xs font-mono space-y-2">
        <div className="flex items-center justify-between text-zinc-400">
          <span className="font-bold text-zinc-300 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-indigo-400" />
            Input Goal-Terminated DFS Exploration Trace (K={sample.trace.length}):
          </span>
          <span className="text-[11px] text-zinc-500">
            Backtracks: <strong className="text-amber-400">{sample.backtracks}</strong>
          </span>
        </div>

        <div className="flex flex-wrap gap-1 bg-zinc-900 p-2 rounded border border-zinc-800 text-[11px] max-h-20 overflow-y-auto custom-scrollbar">
          {sample.trace.map((tok, idx) => {
            const isTargetGoal = tok === goalNode;
            const isStartTok = idx === 0;

            return (
              <span
                key={idx}
                className={`px-1.5 py-0.5 rounded ${
                  isStartTok
                    ? 'bg-emerald-950 border border-emerald-500/50 text-emerald-300 font-bold'
                    : isTargetGoal
                    ? 'bg-rose-950 border border-rose-500/50 text-rose-300 font-bold'
                    : 'bg-zinc-800/80 text-zinc-300'
                }`}
              >
                {tok}
              </span>
            );
          })}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] pt-1 border-t border-zinc-900">
          <div>
            <span className="text-zinc-500">Target Shortest Path (M={groundTruthSP.length}): </span>
            <span className="text-sky-300 font-bold">[{groundTruthSP.join(', ')}]</span>
          </div>
          <div>
            <span className="text-zinc-500">Predicted Rollout: </span>
            <span className={isExact ? 'text-lime-300 font-bold' : 'text-amber-300 font-bold'}>
              [{predictedSP.join(', ')}]
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
