import { AutoregressiveStepTrace, STOP_TOKEN, PAD_TOKEN } from '../model/graph_transformer';
import { BarChart3 } from 'lucide-react';

interface LogitInspectorProps {
  stepTrace: AutoregressiveStepTrace;
  groundTruthSP: number[];
  currentStep: number;
}

export default function LogitInspector({
  stepTrace,
  groundTruthSP,
  currentStep
}: LogitInspectorProps) {
  const targetToken = currentStep + 1 < groundTruthSP.length
    ? groundTruthSP[currentStep + 1]
    : currentStep + 1 === groundTruthSP.length
    ? STOP_TOKEN
    : PAD_TOKEN;

  const isCorrect = stepTrace.predictedToken === targetToken;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-3">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            Classifier Logit &amp; Next-Step Probability Inspector
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Output FC layer projection z = h_dec W_out + b_out Softmax probabilities over 42 vocabulary tokens.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-zinc-400">Logit Margin Δz:</span>
          <span className="px-2 py-0.5 rounded bg-zinc-950 border border-zinc-800 text-amber-300 font-bold">
            {stepTrace.computationTrace?.logitMargin.toFixed(3) ?? 'N/A'}
          </span>
          <span className="text-zinc-400">Target Token:</span>
          <span className="px-2 py-0.5 rounded bg-zinc-800 text-sky-300 font-bold">
            {targetToken === STOP_TOKEN ? 'STOP (41)' : targetToken === PAD_TOKEN ? 'PAD (40)' : targetToken}
          </span>
          <span className={`px-2 py-0.5 rounded font-bold ${
            isCorrect
              ? 'bg-emerald-950/80 border border-emerald-500/50 text-emerald-300'
              : 'bg-rose-950/80 border border-rose-500/50 text-rose-300'
          }`}>
            {isCorrect ? 'Correct Token ✓' : 'Token Error ⚠'}
          </span>
        </div>
      </div>

      {/* Top 5 Probability Bars */}
      <div className="space-y-3 font-mono">
        <p className="text-xs font-bold text-zinc-300 uppercase tracking-wider">
          Top-5 Candidate Predictions for Step #{currentStep + 1}:
        </p>

        <div className="space-y-2">
          {stepTrace.topK.map((item, idx) => {
            const isPredicted = idx === 0;
            const isTarget = item.token === targetToken;
            const pct = (item.prob * 100).toFixed(2);

            let barColor = 'bg-zinc-700';
            if (isPredicted && isTarget) {
              barColor = 'bg-emerald-500';
            } else if (isPredicted) {
              barColor = 'bg-rose-500';
            } else if (isTarget) {
              barColor = 'bg-sky-500';
            }

            return (
              <div key={item.token} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-5 text-zinc-500 font-bold">#{idx + 1}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                      isPredicted
                        ? 'bg-indigo-600 text-white'
                        : 'bg-zinc-800 text-zinc-300'
                    }`}>
                      Token {item.token === STOP_TOKEN ? 'STOP (41)' : item.token === PAD_TOKEN ? 'PAD (40)' : item.token}
                    </span>

                    {isTarget && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-sky-950 border border-sky-500/40 text-sky-300">
                        Target Goal Token
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="text-zinc-500 text-[11px]">Logit: {item.logit.toFixed(3)}</span>
                    <span className="font-bold text-zinc-200">{pct}%</span>
                  </div>
                </div>

                <div className="w-full bg-zinc-950 h-2 rounded-full overflow-hidden border border-zinc-850">
                  <div
                    style={{ width: `${Math.max(1, item.prob * 100)}%` }}
                    className={`h-full transition-all duration-300 ${barColor}`}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
