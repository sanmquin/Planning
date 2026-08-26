import { GraphSample, FullInferenceTrace, TRAINING_HISTORY } from '../model/graph_transformer';
import { GitCompare, TrendingUp } from 'lucide-react';

interface ModelComparerProps {
  sample: GraphSample;
  trace300: FullInferenceTrace;
  trace400: FullInferenceTrace;
  activeModelEpoch: "300" | "400";
  onSelectModelEpoch: (epoch: "300" | "400") => void;
}

export default function ModelComparer({
  sample,
  trace300,
  trace400,
  activeModelEpoch,
  onSelectModelEpoch
}: ModelComparerProps) {
  const h400 = TRAINING_HISTORY["400"];

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-indigo-400" />
            Model Checkpoint Comparison: Epoch 300 vs Epoch 400
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Benchmarking intermediate training (Epoch 300) against converged state (Epoch 400) on compounding rollout errors.
          </p>
        </div>

        {/* Model Epoch Selector Toggle */}
        <div className="flex items-center bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs font-mono">
          <button
            onClick={() => onSelectModelEpoch("300")}
            className={`px-3 py-1.5 rounded font-bold transition-all flex items-center gap-1.5 ${
              activeModelEpoch === "300"
                ? 'bg-amber-500 text-zinc-950 shadow-[0_0_10px_rgba(245,158,11,0.3)]'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Epoch 300 Model
          </button>

          <button
            onClick={() => onSelectModelEpoch("400")}
            className={`px-3 py-1.5 rounded font-bold transition-all flex items-center gap-1.5 ${
              activeModelEpoch === "400"
                ? 'bg-emerald-500 text-zinc-950 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Epoch 400 Model
          </button>
        </div>
      </div>

      {/* Side-by-Side Rollout Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              <h3 className="text-sm font-bold text-zinc-200">Epoch 300 Checkpoint</h3>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 border border-amber-500/30 text-amber-300">
              Intermediate Phase
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2.5 rounded-lg border border-zinc-800">
            <div>
              <p className="text-[10px] text-zinc-500">TF Token Acc:</p>
              <p className="text-sm font-bold text-amber-300">79.22%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Rollout Exact Match:</p>
              <p className="text-sm font-bold text-rose-400">13.40%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Path Validity:</p>
              <p className="text-sm font-bold text-amber-400">13.80%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Val Loss:</p>
              <p className="text-sm font-bold text-zinc-300">0.8603</p>
            </div>
          </div>

          {/* Rollout Trace on Active Sample */}
          <div className="space-y-1 text-xs font-mono pt-1">
            <div className="flex items-center justify-between text-[11px] text-zinc-400">
              <span>Sample #{sample.id} Rollout:</span>
              <span className={trace300.isExactMatch ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                {trace300.isExactMatch ? 'Exact Match ✓' : 'Compounding Drift ⚠'}
              </span>
            </div>
            <div className="p-2 bg-zinc-900 rounded border border-zinc-800 text-[11px] text-amber-200 truncate">
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
              <h3 className="text-sm font-bold text-zinc-200">Epoch 400 Checkpoint</h3>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300">
              Converged Phase
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-zinc-900/80 p-2.5 rounded-lg border border-zinc-800">
            <div>
              <p className="text-[10px] text-zinc-500">TF Token Acc:</p>
              <p className="text-sm font-bold text-emerald-400">98.18%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Rollout Exact Match:</p>
              <p className="text-sm font-bold text-emerald-300">80.00%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Path Validity:</p>
              <p className="text-sm font-bold text-emerald-300">80.60%</p>
            </div>
            <div>
              <p className="text-[10px] text-zinc-500">Val Loss:</p>
              <p className="text-sm font-bold text-zinc-300">0.0973</p>
            </div>
          </div>

          {/* Rollout Trace on Active Sample */}
          <div className="space-y-1 text-xs font-mono pt-1">
            <div className="flex items-center justify-between text-[11px] text-zinc-400">
              <span>Sample #{sample.id} Rollout:</span>
              <span className={trace400.isExactMatch ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold'}>
                {trace400.isExactMatch ? 'Exact Match ✓' : 'Compounding Drift ⚠'}
              </span>
            </div>
            <div className="p-2 bg-zinc-900 rounded border border-zinc-800 text-[11px] text-emerald-200 truncate">
              [{trace400.predictedSP.join(', ')}]
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
              {h400.val_epochs?.map((ep, idx) => {
                const is300 = ep === 300;
                const is400 = ep === 400;

                return (
                  <tr
                    key={ep}
                    className={`transition-colors ${
                      is300
                        ? 'bg-amber-950/30 text-amber-200 font-bold'
                        : is400
                        ? 'bg-emerald-950/30 text-emerald-200 font-bold'
                        : 'hover:bg-zinc-900/50'
                    }`}
                  >
                    <td className="p-2 flex items-center gap-1.5">
                      <span>{ep}</span>
                      {is300 && <span className="text-[9px] px-1 bg-amber-500/20 text-amber-300 rounded">Checkpoint 1</span>}
                      {is400 && <span className="text-[9px] px-1 bg-emerald-500/20 text-emerald-300 rounded">Checkpoint 2</span>}
                    </td>
                    <td className="p-2">{h400.val_loss[idx]?.toFixed(4)}</td>
                    <td className="p-2">{h400.val_tf_acc[idx]?.toFixed(2)}%</td>
                    <td className="p-2">{h400.val_exact_match[idx]?.toFixed(2)}%</td>
                    <td className="p-2">{h400.val_path_validity[idx]?.toFixed(2)}%</td>
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
