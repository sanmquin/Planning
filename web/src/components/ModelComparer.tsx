import { GraphSample, FullInferenceTrace, TRAINING_HISTORY } from '../model/graph_transformer';
import { GitCompare, TrendingUp } from 'lucide-react';

interface ModelComparerProps {
  sample: GraphSample;
  trace300: FullInferenceTrace;
  trace400: FullInferenceTrace;
  trace500?: FullInferenceTrace;
  activeModelEpoch: "300" | "400" | "500";
  onSelectModelEpoch: (epoch: "300" | "400" | "500") => void;
}

export default function ModelComparer({
  sample,
  trace300,
  trace400,
  trace500,
  activeModelEpoch,
  onSelectModelEpoch
}: ModelComparerProps) {
  const h500 = TRAINING_HISTORY["500"] || TRAINING_HISTORY["400"];

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-indigo-400" />
            Model Checkpoint Comparison: Epochs 300, 400 &amp; 500 on Sample #{sample.id}
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Benchmarking intermediate training (Epoch 300) against converged states (Epochs 400 &amp; 500).
          </p>
        </div>

        {/* Model Epoch Selector Toggle */}
        <div className="flex items-center bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-mono">
          {(["300", "400", "500"] as const).map(ep => (
            <button
              key={ep}
              onClick={() => onSelectModelEpoch(ep)}
              className={`px-3 py-1.5 rounded font-bold transition-all flex items-center gap-1.5 ${
                activeModelEpoch === ep
                  ? ep === "300"
                    ? 'bg-amber-500 text-zinc-950 shadow-sm'
                    : ep === "400"
                    ? 'bg-emerald-500 text-zinc-950 shadow-sm'
                    : 'bg-indigo-500 text-white shadow-sm'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Epoch {ep}
            </button>
          ))}
        </div>
      </div>

      {/* Side-by-Side Rollout Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Epoch 300 Card */}
        <div
          onClick={() => onSelectModelEpoch("300")}
          className={`p-4 rounded-xl border transition-all cursor-pointer space-y-3 ${
            activeModelEpoch === "300"
              ? 'bg-amber-950/20 border-amber-500/60 shadow-[0_0_15px_rgba(245,158,11,0.15)]'
              : 'bg-zinc-950/60 border-zinc-850 hover:border-zinc-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
              <h3 className="text-xs font-bold text-zinc-200">Epoch 300</h3>
            </div>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-950 border border-amber-500/30 text-amber-300">
              Intermediate
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
            <div>
              <p className="text-[9px] text-zinc-500">TF Acc:</p>
              <p className="text-xs font-bold text-amber-300">79.22%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Exact Match:</p>
              <p className="text-xs font-bold text-rose-400">13.40%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Validity:</p>
              <p className="text-xs font-bold text-amber-400">13.80%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Val Loss:</p>
              <p className="text-xs font-bold text-zinc-300">0.8603</p>
            </div>
          </div>

          <div className="space-y-1 text-xs font-mono pt-1">
            <div className="flex items-center justify-between text-[10px] text-zinc-400">
              <span>Rollout:</span>
              <span className={trace300.isExactMatch ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                {trace300.isExactMatch ? 'Exact ✓' : 'Drift ⚠'}
              </span>
            </div>
            <div className="p-1.5 bg-zinc-900 rounded border border-zinc-800 text-[10px] text-amber-200 truncate">
              [{trace300.predictedSP.join(', ')}]
            </div>
          </div>
        </div>

        {/* Epoch 400 Card */}
        <div
          onClick={() => onSelectModelEpoch("400")}
          className={`p-4 rounded-xl border transition-all cursor-pointer space-y-3 ${
            activeModelEpoch === "400"
              ? 'bg-emerald-950/20 border-emerald-500/60 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
              : 'bg-zinc-950/60 border-zinc-850 hover:border-zinc-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              <h3 className="text-xs font-bold text-zinc-200">Epoch 400</h3>
            </div>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300">
              Converged
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
            <div>
              <p className="text-[9px] text-zinc-500">TF Acc:</p>
              <p className="text-xs font-bold text-emerald-400">98.18%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Exact Match:</p>
              <p className="text-xs font-bold text-emerald-300">80.00%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Validity:</p>
              <p className="text-xs font-bold text-emerald-300">80.60%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Val Loss:</p>
              <p className="text-xs font-bold text-zinc-300">0.0973</p>
            </div>
          </div>

          <div className="space-y-1 text-xs font-mono pt-1">
            <div className="flex items-center justify-between text-[10px] text-zinc-400">
              <span>Rollout:</span>
              <span className={trace400.isExactMatch ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                {trace400.isExactMatch ? 'Exact ✓' : 'Drift ⚠'}
              </span>
            </div>
            <div className="p-1.5 bg-zinc-900 rounded border border-zinc-800 text-[10px] text-emerald-200 truncate">
              [{trace400.predictedSP.join(', ')}]
            </div>
          </div>
        </div>

        {/* Epoch 500 Card */}
        <div
          onClick={() => onSelectModelEpoch("500")}
          className={`p-4 rounded-xl border transition-all cursor-pointer space-y-3 ${
            activeModelEpoch === "500"
              ? 'bg-indigo-950/30 border-indigo-500/60 shadow-[0_0_15px_rgba(99,102,241,0.2)]'
              : 'bg-zinc-950/60 border-zinc-850 hover:border-zinc-700'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
              <h3 className="text-xs font-bold text-zinc-200">Epoch 500</h3>
            </div>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-indigo-950 border border-indigo-500/30 text-indigo-300">
              Optimal Peak
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2 rounded-lg border border-zinc-800">
            <div>
              <p className="text-[9px] text-zinc-500">TF Acc:</p>
              <p className="text-xs font-bold text-indigo-300">99.45%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Exact Match:</p>
              <p className="text-xs font-bold text-indigo-300">96.40%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Validity:</p>
              <p className="text-xs font-bold text-indigo-300">96.80%</p>
            </div>
            <div>
              <p className="text-[9px] text-zinc-500">Val Loss:</p>
              <p className="text-xs font-bold text-zinc-300">0.0412</p>
            </div>
          </div>

          <div className="space-y-1 text-xs font-mono pt-1">
            <div className="flex items-center justify-between text-[10px] text-zinc-400">
              <span>Rollout:</span>
              <span className={trace500?.isExactMatch ? 'text-indigo-400 font-bold' : 'text-amber-400 font-bold'}>
                {trace500?.isExactMatch ? 'Exact ✓' : 'Drift ⚠'}
              </span>
            </div>
            <div className="p-1.5 bg-zinc-900 rounded border border-zinc-800 text-[10px] text-indigo-200 truncate">
              [{trace500?.predictedSP.join(', ') ?? 'N/A'}]
            </div>
          </div>
        </div>
      </div>

      {/* Historical Training Trajectories Visual Table */}
      <div className="bg-zinc-950/80 border border-zinc-800 rounded-lg p-4 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
          <TrendingUp className="w-4 h-4 text-indigo-400" />
          Validation Metric Trajectories over Epochs (50 to 400)
        </h3>

        <div className="overflow-x-auto custom-scrollbar">
          <table className="w-full text-xs font-mono text-left text-zinc-300">
            <thead className="bg-zinc-900 text-zinc-400 text-[10px] uppercase border-b border-zinc-800">
              <tr>
                <th className="p-2">Epoch</th>
                <th className="p-2">Val Loss</th>
                <th className="p-2">Teacher-Forcing Acc (%)</th>
                <th className="p-2">Rollout Exact Match (%)</th>
                <th className="p-2">Path Validity (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-850">
              {h500.val_epochs?.map((ep, idx) => {
                const is300 = ep === 300;
                const is400 = ep === 400;
                const is500 = ep === 500;

                return (
                  <tr
                    key={ep}
                    className={`transition-colors ${
                      is300
                        ? 'bg-amber-950/30 text-amber-200 font-bold'
                        : is400
                        ? 'bg-emerald-950/30 text-emerald-200 font-bold'
                        : is500
                        ? 'bg-indigo-950/30 text-indigo-200 font-bold'
                        : 'hover:bg-zinc-900/50'
                    }`}
                  >
                    <td className="p-2 flex items-center gap-1.5">
                      <span>{ep}</span>
                      {is300 && <span className="text-[9px] px-1 bg-amber-500/20 text-amber-300 rounded">Epoch 300</span>}
                      {is400 && <span className="text-[9px] px-1 bg-emerald-500/20 text-emerald-300 rounded">Epoch 400</span>}
                      {is500 && <span className="text-[9px] px-1 bg-indigo-500/20 text-indigo-300 rounded">Epoch 500</span>}
                    </td>
                    <td className="p-2">{h500.val_loss[idx]?.toFixed(4)}</td>
                    <td className="p-2">{h500.val_tf_acc[idx]?.toFixed(2)}%</td>
                    <td className="p-2">{h500.val_exact_match[idx]?.toFixed(2)}%</td>
                    <td className="p-2">{h500.val_path_validity[idx]?.toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
