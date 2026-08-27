import { useState } from 'react';
import { AutoregressiveStepTrace, STOP_TOKEN, PAD_TOKEN } from '../model/graph_transformer';
import { Cpu, ArrowRight, Layers, Zap, CheckCircle2, AlertTriangle } from 'lucide-react';

interface InferencePipelineInspectorProps {
  stepTrace: AutoregressiveStepTrace;
  groundTruthSP: number[];
  currentStep: number;
}

type PipelineStage = 'embeddings' | 'decoder_l1' | 'decoder_l2' | 'classifier';

export default function InferencePipelineInspector({
  stepTrace,
  groundTruthSP,
  currentStep
}: InferencePipelineInspectorProps) {
  const [activeStage, setActiveStage] = useState<PipelineStage>('classifier');
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0); // Index in topK / snippet

  const targetToken = currentStep + 1 < groundTruthSP.length
    ? groundTruthSP[currentStep + 1]
    : currentStep + 1 === groundTruthSP.length
    ? STOP_TOKEN
    : PAD_TOKEN;

  const isCorrect = stepTrace.predictedToken === targetToken;
  const comp = stepTrace.computationTrace;

  const renderVectorBar = (vector: number[], label: string, color = 'bg-indigo-500', maxVal = 2.0) => {
    return (
      <div className="space-y-1 font-mono text-[10px]">
        <div className="flex items-center justify-between text-zinc-400">
          <span>{label} (d={vector.length}):</span>
          <span className="text-[9px] text-zinc-500">
            Norm={Math.sqrt(vector.reduce((a, b) => a + b * b, 0)).toFixed(3)}
          </span>
        </div>
        <div className="grid grid-cols-8 sm:grid-cols-16 gap-1 bg-zinc-950 p-2 rounded border border-zinc-850">
          {vector.map((val, idx) => {
            const absNorm = Math.min(Math.abs(val) / maxVal, 1.0);
            const isPos = val >= 0;
            return (
              <div key={idx} className="flex flex-col items-center group relative cursor-pointer">
                <div className="w-full bg-zinc-900 h-10 rounded flex flex-col justify-end overflow-hidden p-0.5 border border-zinc-800">
                  <div
                    style={{ height: `${Math.max(10, absNorm * 100)}%` }}
                    className={`w-full rounded-sm transition-all ${
                      isPos ? color : 'bg-rose-500/80'
                    }`}
                  />
                </div>
                <span className="text-[8px] text-zinc-500 mt-0.5">{idx}</span>

                {/* Tooltip */}
                <div className="absolute bottom-full mb-1 hidden group-hover:block z-30 bg-zinc-900 border border-zinc-700 text-zinc-100 text-[10px] p-1.5 rounded shadow-xl whitespace-nowrap">
                  Dim [{idx}]: {val.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 backdrop-blur-md space-y-5">
      {/* Header & Status */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-800 pb-3">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Mechanistic Inference Computation Flow &amp; Circuit Inspector
          </h2>
          <p className="text-xs text-zinc-400 mt-0.5">
            Step #{currentStep + 1} Forward Pass: <span className="font-mono text-indigo-300">Token {comp.inputToken} &rarr; Emb+PE &rarr; Dec L1 &rarr; Dec L2 &rarr; FC Logits &rarr; Softmax</span>
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-zinc-400">Target Goal:</span>
          <span className="px-2 py-0.5 rounded bg-zinc-800 text-sky-300 font-bold">
            {targetToken === STOP_TOKEN ? 'STOP (41)' : targetToken === PAD_TOKEN ? 'PAD (40)' : targetToken}
          </span>
          <span className={`px-2 py-0.5 rounded font-bold flex items-center gap-1 ${
            isCorrect
              ? 'bg-emerald-950/80 border border-emerald-500/50 text-emerald-300'
              : 'bg-rose-950/80 border border-rose-500/50 text-rose-300'
          }`}>
            {isCorrect ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
            {isCorrect ? 'Correct ✓' : 'Error ⚠'}
          </span>
        </div>
      </div>

      {/* Interactive Pipeline Stage Navigation Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
        <button
          onClick={() => setActiveStage('embeddings')}
          className={`p-3 rounded-lg border text-left transition-all ${
            activeStage === 'embeddings'
              ? 'bg-indigo-950/80 border-indigo-500 text-indigo-200 shadow-[0_0_12px_rgba(99,102,241,0.25)] font-bold'
              : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5">Stage 1</div>
          <div className="flex items-center justify-between">
            <span>1. Embeddings &amp; PE</span>
            <ArrowRight className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">E(x) + PE(t) &rarr; R^16</div>
        </button>

        <button
          onClick={() => setActiveStage('decoder_l1')}
          className={`p-3 rounded-lg border text-left transition-all ${
            activeStage === 'decoder_l1'
              ? 'bg-violet-950/80 border-violet-500 text-violet-200 shadow-[0_0_12px_rgba(139,92,246,0.25)] font-bold'
              : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5">Stage 2</div>
          <div className="flex items-center justify-between">
            <span>2. Decoder Layer 1</span>
            <ArrowRight className="w-3.5 h-3.5 text-violet-400" />
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">Self/Cross-Attn + FFN1</div>
        </button>

        <button
          onClick={() => setActiveStage('decoder_l2')}
          className={`p-3 rounded-lg border text-left transition-all ${
            activeStage === 'decoder_l2'
              ? 'bg-cyan-950/80 border-cyan-500 text-cyan-200 shadow-[0_0_12px_rgba(6,182,212,0.25)] font-bold'
              : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5">Stage 3</div>
          <div className="flex items-center justify-between">
            <span>3. Decoder Layer 2</span>
            <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">Final Residual Stream h_dec</div>
        </button>

        <button
          onClick={() => setActiveStage('classifier')}
          className={`p-3 rounded-lg border text-left transition-all ${
            activeStage === 'classifier'
              ? 'bg-emerald-950/80 border-emerald-500 text-emerald-200 shadow-[0_0_12px_rgba(16,185,129,0.25)] font-bold'
              : 'bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-0.5">Stage 4</div>
          <div className="flex items-center justify-between">
            <span>4. Logit Classifier</span>
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">h W_out + b &rarr; Δz Margin</div>
        </button>
      </div>

      {/* Stage Detail Panels */}

      {/* Stage 1: Embeddings & PE */}
      {activeStage === 'embeddings' && (
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
            <h3 className="text-xs font-bold text-indigo-300 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              Stage 1: Input Representation Construction (Token Embedding + Sinusoidal PE)
            </h3>
            <span className="text-[11px] font-mono text-zinc-400">
              Token ID = <strong className="text-indigo-400">{comp.inputToken}</strong>
            </span>
          </div>

          <p className="text-xs text-zinc-400 leading-relaxed">
            The input target sequence token y_t = {comp.inputToken} is looked up in token embedding matrix E (42x16) and summed with sinusoidal positional encoding PE(pos={currentStep}) (16-dim) at target rollout step t = {currentStep + 1}.
          </p>

          <div className="space-y-4">
            {renderVectorBar(comp.inputTokenEmbedding, `1. Token Embedding Vector E(${comp.inputToken})`, 'bg-indigo-500')}
            {renderVectorBar(comp.positionalEncoding, `2. Sinusoidal Positional Encoding PE(pos=${currentStep})`, 'bg-violet-500')}
            {renderVectorBar(comp.combinedEmbedding, `3. Combined Input Vector x_dec^(0) = E + PE`, 'bg-emerald-500')}
          </div>
        </div>
      )}

      {/* Stage 2 & 3: Decoder Layer 1 / Decoder Layer 2 */}
      {(activeStage === 'decoder_l1' || activeStage === 'decoder_l2') && (() => {
        const layerIdx = activeStage === 'decoder_l1' ? 0 : 1;
        const layerComp = comp.decoderLayers[layerIdx];
        const themeColor = layerIdx === 0 ? 'bg-violet-500' : 'bg-cyan-500';

        return (
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
              <h3 className="text-xs font-bold text-zinc-100 flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                Decoder Layer {layerIdx + 1} Transformer Block Transformation
              </h3>
              <span className="text-[11px] font-mono text-zinc-400">
                Residual Stream Vector h_dec Layer {layerIdx + 1} (16-dim)
              </span>
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed">
              Decoder Layer {layerIdx + 1} processes the active position vector through Causal Self-Attention (2 heads), Cross-Attention over the 1D DFS trace (2 heads), and GELU Feed-Forward Network (16 &rarr; 32 &rarr; 16).
            </p>

            <div className="space-y-4">
              {renderVectorBar(layerComp.inputState, `Input State h_dec^(${layerIdx})`, 'bg-zinc-400')}
              {renderVectorBar(layerComp.selfAttnOut, `Causal Self-Attention Output (2 Heads)`, themeColor)}
              {renderVectorBar(layerComp.postSelfNorm, `Post-Self-Attn LayerNorm1 & Residual`, themeColor)}
              {renderVectorBar(layerComp.crossAttnOut, `Cross-Attention Output over DFS Trace (2 Heads)`, 'bg-indigo-500')}
              {renderVectorBar(layerComp.postCrossNorm, `Post-Cross-Attn LayerNorm2 & Residual`, 'bg-indigo-400')}
              {renderVectorBar(layerComp.ffnHiddenGelu, `FFN Intermediate GELU Hidden Activations (d=32)`, 'bg-amber-500', 3.0)}
              {renderVectorBar(layerComp.outputState, `Layer ${layerIdx + 1} Output Vector h_dec^(${layerIdx + 1})`, 'bg-emerald-500')}
            </div>
          </div>
        );
      })()}

      {/* Stage 4: FC Projection & Logit Classification */}
      {activeStage === 'classifier' && (
        <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-4 space-y-5">
          <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
            <h3 className="text-xs font-bold text-emerald-300 flex items-center gap-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              Stage 4: Linear Classifier z = h_dec W_out + b_out &amp; Logit Margin Δz
            </h3>
            <span className="text-[11px] font-mono text-emerald-400 font-bold">
              Logit Margin Δz = {comp.logitMargin.toFixed(3)}
            </span>
          </div>

          <p className="text-xs text-zinc-400 leading-relaxed">
            The final decoder representation h_dec (16-dim) is projected onto vocabulary size 42 via linear classification weights W_out (42x16) and bias b_out. Logit margin Δz = z_top1 - z_top2 quantifies decision certainty.
          </p>

          {renderVectorBar(comp.finalDecoderState, `Final Decoder Representation Vector h_dec`, 'bg-cyan-500')}

          {/* Top Candidates Dot-Product Breakdown */}
          <div className="space-y-3 font-mono">
            <div className="flex items-center justify-between text-xs font-bold text-zinc-300">
              <span>Classifier Matrix Multiplication Breakdown for Candidate Tokens:</span>
              <span className="text-[10px] text-zinc-500">
                Formula: z_v = &sum; (h_d &middot; W_v,d) + b_v
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
              {comp.fcWeightsSnippet.map((item, idx) => {
                const isSelected = selectedTokenIndex === idx;
                const isPredicted = idx === 0;

                return (
                  <button
                    key={item.token}
                    onClick={() => setSelectedTokenIndex(idx)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      isSelected
                        ? 'bg-emerald-950/80 border-emerald-500 text-emerald-200 shadow-md ring-1 ring-emerald-400'
                        : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700 text-zinc-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-bold mb-1">
                      <span>Token #{item.token}</span>
                      {isPredicted && (
                        <span className="px-1.5 py-0.2 rounded text-[9px] bg-emerald-500 text-zinc-950">Top-1</span>
                      )}
                    </div>
                    <div className="text-[10px] text-zinc-400 space-y-0.5">
                      <p>Dot Prod: {item.dotProduct.toFixed(3)}</p>
                      <p>Bias b_v: {item.bias.toFixed(3)}</p>
                      <p className="font-bold text-zinc-100">Logit z_v: {item.logit.toFixed(3)}</p>
                      <p className="text-emerald-400 font-bold">Prob: {(item.prob * 100).toFixed(2)}%</p>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected Token FC Weight Vector Dot Product Inspector */}
            {comp.fcWeightsSnippet[selectedTokenIndex] && (() => {
              const item = comp.fcWeightsSnippet[selectedTokenIndex];
              return (
                <div className="bg-zinc-900/90 border border-zinc-800 rounded-lg p-3 space-y-3">
                  <div className="flex items-center justify-between text-xs text-zinc-200">
                    <span className="font-bold text-emerald-300">
                      Classifier Weights W_out[Token #{item.token}] Vector (16-dim):
                    </span>
                    <span className="text-[11px] text-zinc-400 font-mono">
                      Dot Product ({item.dotProduct.toFixed(3)}) + Bias ({item.bias.toFixed(3)}) = Logit ({item.logit.toFixed(3)})
                    </span>
                  </div>

                  {renderVectorBar(item.weightVec, `Weight Vector W_out[Token ${item.token}]`, 'bg-amber-500', 1.5)}
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
