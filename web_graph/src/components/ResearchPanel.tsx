import { Brain, AlertOctagon, CheckCircle, Compass } from 'lucide-react';

export default function ResearchPanel() {
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-md space-y-6">
      <div className="border-b border-zinc-800 pb-4">
        <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
          <Brain className="w-5 h-5 text-amber-400" />
          Research Analytics &amp; Algorithmic Path Extraction Dynamics
        </h2>
        <p className="text-xs text-zinc-400 mt-0.5">
          Theoretical derivations, compounding rollout error analysis, and structural cross-attention routing mechanics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs text-zinc-300 leading-relaxed font-sans">
        {/* Research Question 1 */}
        <div className="bg-zinc-950/80 border border-zinc-800 rounded-xl p-4 space-y-2.5">
          <div className="flex items-center gap-2 text-amber-400 font-bold">
            <AlertOctagon className="w-4 h-4 shrink-0" />
            <span>1. Compounding Rollout Errors</span>
          </div>
          <p className="text-zinc-400 leading-normal">
            Why does rollout exact match drop precipitously from teacher-forcing accuracy when sequences grow to M &ge; 10?
          </p>
          <div className="p-2.5 bg-zinc-900 rounded border border-zinc-850 font-mono text-[11px] text-amber-200">
            {"P(Rollout Match) = ∏(1 - ε_m) ≈ (1 - ε)^M"}
          </div>
          <p className="text-zinc-400 text-[11px]">
            In Epoch 300, a minor 20.8% teacher-forcing token error rate compounds over 15 steps into a 97% sequence failure rate (yielding only 13.4% exact rollout match).
          </p>
        </div>

        {/* Research Question 2 */}
        <div className="bg-zinc-950/80 border border-zinc-800 rounded-xl p-4 space-y-2.5">
          <div className="flex items-center gap-2 text-cyan-400 font-bold">
            <Compass className="w-4 h-4 shrink-0" />
            <span>2. DFS Trace Compression Ratio (M / K)</span>
          </div>
          <p className="text-zinc-400 leading-normal">
            How does cross-attention filter out return steps (t_k = t_k-2) and dead-end exploration subtrees embedded in 1D traces?
          </p>
          <div className="p-2.5 bg-zinc-900 rounded border border-zinc-850 font-mono text-[11px] text-cyan-200">
            {"η = (Path Length M) / (Trace Length K) ≈ 0.31"}
          </div>
          <p className="text-zinc-400 text-[11px]">
            The decoder cross-attention layers attend selectively to forward transition edges while dampening key representations corresponding to dead-end backtracks.
          </p>
        </div>

        {/* Research Question 3 */}
        <div className="bg-zinc-950/80 border border-zinc-800 rounded-xl p-4 space-y-2.5">
          <div className="flex items-center gap-2 text-emerald-400 font-bold">
            <CheckCircle className="w-4 h-4 shrink-0" />
            <span>3. Converged Alignment (Epoch 400)</span>
          </div>
          <p className="text-zinc-400 leading-normal">
            What structural phase transition occurs between Epoch 300 and Epoch 400?
          </p>
          <div className="p-2.5 bg-zinc-900 rounded border border-zinc-850 font-mono text-[11px] text-emerald-200">
            {"Val Loss: 0.8603 → 0.0973 (8.8x drop)"}
          </div>
          <p className="text-zinc-400 text-[11px]">
            By Epoch 400, teacher forcing token accuracy climbs to 98.18%, suppressing early prefix errors and driving rollout exact match up to 80.0%.
          </p>
        </div>
      </div>
    </div>
  );
}
