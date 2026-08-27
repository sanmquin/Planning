import { useState } from 'react';
import modelsWeightsData from '../data/models_weights.json';
import { Database, Lock, Eye, BarChart2 } from 'lucide-react';
import { AutoregressiveStepTrace } from '../model/graph_transformer';

interface WeightsInspectorProps {
  selectedEpoch?: "300" | "400" | "500";
  onSelectEpoch?: (epoch: "300" | "400" | "500") => void;
  activeStepTrace?: AutoregressiveStepTrace;
}

export default function WeightsInspector({
  selectedEpoch = "500",
  onSelectEpoch,
  activeStepTrace
}: WeightsInspectorProps) {
  const [activeCheckpoint, setActiveCheckpoint] = useState<"300" | "400" | "500">(selectedEpoch);

  const currentEpoch = onSelectEpoch ? selectedEpoch : activeCheckpoint;

  const handleEpochChange = (epoch: "300" | "400" | "500") => {
    setActiveCheckpoint(epoch);
    if (onSelectEpoch) onSelectEpoch(epoch);
  };

  const weightsData = (modelsWeightsData as any)[currentEpoch] || {};
  const paramKeys = Object.keys(weightsData);

  const [selectedKey, setSelectedKey] = useState<string>(paramKeys[0] || 'token_embedding.weight');
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; val: number } | null>(null);

  const paramData = weightsData[selectedKey];

  const flatten = (arr: any): number[] => {
    if (!Array.isArray(arr)) return [Number(arr)];
    return arr.flatMap(flatten);
  };

  const totalParams = paramKeys.reduce((acc, key) => {
    const tensor = weightsData[key];
    return acc + flatten(tensor).length;
  }, 0);

  const trainableParams = paramKeys.reduce((acc, key) => {
    if (key === 'pos_encoder.pe') return acc;
    const tensor = weightsData[key];
    return acc + flatten(tensor).length;
  }, 0);

  const flatValues = paramData ? flatten(paramData) : [];
  const count = flatValues.length;
  const mean = count > 0 ? flatValues.reduce((a, b) => a + b, 0) / count : 0;
  const min = count > 0 ? Math.min(...flatValues) : 0;
  const max = count > 0 ? Math.max(...flatValues) : 0;
  const variance = count > 0 ? flatValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / count : 0;
  const std = Math.sqrt(variance);
  const l2Norm = Math.sqrt(flatValues.reduce((a, b) => a + b * b, 0));

  // Determine shape
  const getShape = (tensor: any): number[] => {
    if (!Array.isArray(tensor)) return [];
    if (!Array.isArray(tensor[0])) return [tensor.length];
    if (!Array.isArray(tensor[0][0])) return [tensor.length, tensor[0].length];
    return [tensor.length, tensor[0].length, tensor[0][0].length];
  };

  const shape = getShape(paramData);
  const is2D = shape.length === 2;

  const getHeatmapColor = (val: number) => {
    const norm = Math.min(Math.abs(val) / (max || 1.0), 1.0);
    if (val >= 0) {
      return `rgba(99, 102, 241, ${0.15 + norm * 0.8})`;
    } else {
      return `rgba(244, 63, 94, ${0.15 + norm * 0.8})`;
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-5">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Lock className="w-5 h-5 text-cyan-400" />
            Frozen Model Weights &amp; Parameter Matrix Inspector
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Static model weights frozen after training. View tensor values, weight heatmaps, and stats before &amp; during inference.
          </p>
        </div>

        {/* Checkpoint Toggle Button Group */}
        <div className="flex items-center gap-2 bg-zinc-950 p-1.5 rounded-lg border border-zinc-800 text-xs font-mono">
          <span className="text-zinc-500 px-1 text-[11px] flex items-center gap-1">
            <Database className="w-3 h-3 text-cyan-400" /> Checkpoint:
          </span>
          {(["300", "400", "500"] as const).map(ep => (
            <button
              key={ep}
              onClick={() => handleEpochChange(ep)}
              className={`px-3 py-1 rounded transition-all font-bold ${
                currentEpoch === ep
                  ? 'bg-cyan-500 text-zinc-950 shadow-[0_0_10px_rgba(6,182,212,0.4)]'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Epoch {ep}
            </button>
          ))}
        </div>
      </div>

      {/* Overview Stat Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
        <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-850 space-y-0.5">
          <p className="text-[10px] text-zinc-500 uppercase">Total Parameters</p>
          <p className="text-base font-bold text-cyan-300">{totalParams.toLocaleString()}</p>
        </div>

        <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-850 space-y-0.5">
          <p className="text-[10px] text-zinc-500 uppercase">Trainable Tensors</p>
          <p className="text-base font-bold text-emerald-400">{trainableParams.toLocaleString()}</p>
        </div>

        <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-850 space-y-0.5">
          <p className="text-[10px] text-zinc-500 uppercase">Selected Tensor</p>
          <p className="text-xs font-bold text-violet-300 truncate" title={selectedKey}>{selectedKey}</p>
        </div>

        <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-850 space-y-0.5">
          <p className="text-[10px] text-zinc-500 uppercase">Tensor Shape</p>
          <p className="text-xs font-bold text-amber-300">[{shape.join(' × ')}]</p>
        </div>
      </div>

      {/* Main Inspector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Parameter List Tree */}
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-1.5 max-h-[420px] overflow-y-auto custom-scrollbar font-mono text-xs">
          <p className="text-[10px] uppercase font-bold text-zinc-500 mb-2 px-1 flex items-center justify-between">
            <span>Frozen Tensors ({paramKeys.length}):</span>
          </p>
          {paramKeys.map(key => {
            const isSelected = selectedKey === key;
            const kShape = getShape(weightsData[key]);

            return (
              <button
                key={key}
                onClick={() => setSelectedKey(key)}
                className={`w-full text-left px-2.5 py-1.5 rounded transition-all flex items-center justify-between text-[11px] ${
                  isSelected
                    ? 'bg-cyan-600 text-white font-bold shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
                }`}
              >
                <span className="truncate pr-2">{key}</span>
                <span className="text-[9px] text-zinc-500 font-normal shrink-0">
                  [{kShape.join('×')}]
                </span>
              </button>
            );
          })}
        </div>

        {/* Selected Tensor Detailed View */}
        <div className="md:col-span-2 space-y-4 font-mono">
          {/* Summary Stats Box */}
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3.5 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-850 pb-2 text-xs">
              <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                {selectedKey}
              </span>
              <span className="text-[11px] text-zinc-400">
                Elements: <strong className="text-zinc-200">{count}</strong>
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs">
              <div>
                <p className="text-[10px] text-zinc-500">Min:</p>
                <p className="font-bold text-rose-400">{min.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">Max:</p>
                <p className="font-bold text-indigo-400">{max.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">Mean:</p>
                <p className="font-bold text-zinc-200">{mean.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">Std Dev:</p>
                <p className="font-bold text-amber-300">{std.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-[10px] text-zinc-500">L2 Norm:</p>
                <p className="font-bold text-emerald-400">{l2Norm.toFixed(4)}</p>
              </div>
            </div>
          </div>

          {/* Active Rollout Dynamic Context Card (If available during inference) */}
          {activeStepTrace && (
            <div className="bg-cyan-950/30 border border-cyan-500/40 rounded-lg p-3 text-xs space-y-1">
              <div className="flex items-center justify-between font-bold text-cyan-300">
                <span className="flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-cyan-400" />
                  Live Inference State Alignment (Step #{activeStepTrace.step + 1}):
                </span>
                <span className="text-[10px] text-zinc-400">
                  Input Token: {activeStepTrace.computationTrace.inputToken}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400">
                Viewing frozen weights alongside active decoder representation vector h_dec (d=16) during rollout.
              </p>
            </div>
          )}

          {/* Matrix Visual Heatmap Grid / Raw Values */}
          {is2D && shape[0] <= 42 && shape[1] <= 48 ? (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Weight Matrix 2D Heatmap Grid ({shape[0]} × {shape[1]}):</span>
                {hoveredCell && (
                  <span className="text-cyan-300 font-bold text-[11px]">
                    [{hoveredCell.r}, {hoveredCell.c}] = {hoveredCell.val.toFixed(5)}
                  </span>
                )}
              </div>

              <div className="overflow-auto max-h-[260px] custom-scrollbar p-1">
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: `repeat(${shape[1]}, minmax(8px, 1fr))`
                  }}
                  className="gap-0.5"
                >
                  {(paramData as number[][]).map((row, rIdx) =>
                    row.map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{ backgroundColor: getHeatmapColor(val) }}
                        className="h-3 rounded-[1px] cursor-pointer hover:scale-125 transition-transform duration-100"
                        title={`[${rIdx}, ${cIdx}]: ${val.toFixed(5)}`}
                      />
                    ))
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-[260px] overflow-auto custom-scrollbar text-[10px]">
              <p className="text-zinc-500 mb-2">Weight Value Tensor Snippet:</p>
              <pre className="text-cyan-300 whitespace-pre-wrap leading-relaxed">
                {JSON.stringify(paramData, null, 2)?.slice(0, 1800)}...
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
