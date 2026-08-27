import { X, Calculator, ArrowRight, Layers, Zap, Cpu } from 'lucide-react';

interface ComputationFormulaModalProps {
  isOpen: boolean;
  onClose: () => void;
  stage: 'embeddings' | 'decoder_l1' | 'decoder_l2' | 'classifier';
}

export default function ComputationFormulaModal({
  isOpen,
  onClose,
  stage
}: ComputationFormulaModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl max-w-2xl w-full p-6 space-y-5 text-zinc-100 font-sans shadow-2xl overflow-y-auto max-h-[90vh] custom-scrollbar">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-zinc-850 pb-3">
          <div className="flex items-center gap-2 text-indigo-400 font-bold text-base">
            <Calculator className="w-5 h-5 text-indigo-400" />
            <span>Mathematical Formulas &amp; Matrix Operations Breakdown</span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-zinc-900 text-zinc-400 hover:text-zinc-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Stage Content */}

        {/* Stage 1: Embeddings */}
        {stage === 'embeddings' && (
          <div className="space-y-4 text-xs leading-relaxed font-mono">
            <div className="flex items-center gap-2 text-indigo-300 font-bold text-sm">
              <Layers className="w-4 h-4 text-indigo-400" />
              Stage 1: Token Embedding &amp; Positional Encoding Lookup
            </div>

            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="text-zinc-400 text-[11px]">Mathematical Formula:</p>
              <div className="text-indigo-200 font-bold text-sm">
                {"x_t^{(0)} = E[y_t] + PE(t)"}
              </div>
              <p className="text-zinc-400 text-[11px]">
                Where E ∈ ℝ^(42×16) is the frozen token embedding table and PE(t) ∈ ℝ^16 is the sinusoidal positional encoding at step position t.
              </p>
            </div>

            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="text-zinc-400 text-[11px]">Matrix Dimension Operations:</p>
              <div className="flex items-center gap-2 text-emerald-300 font-bold">
                <span>[1 × 16]</span>
                <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                <span>E[y_t] [1 × 16] + PE(t) [1 × 16]</span>
                <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                <span>[1 × 16]</span>
              </div>
            </div>

            <div className="p-3 bg-zinc-900/50 rounded-lg border border-zinc-850 text-zinc-400 space-y-1">
              <p className="font-bold text-zinc-300">Sinusoidal PE Formula:</p>
              <p>{"PE(pos, 2i) = sin(pos / 10000^(2i / d_model))"}</p>
              <p>{"PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))"}</p>
            </div>
          </div>
        )}

        {/* Stage 2 & 3: Decoder Layer 1 / 2 */}
        {(stage === 'decoder_l1' || stage === 'decoder_l2') && (
          <div className="space-y-4 text-xs leading-relaxed font-mono">
            <div className="flex items-center gap-2 text-violet-300 font-bold text-sm">
              <Cpu className="w-4 h-4 text-violet-400" />
              {stage === 'decoder_l1' ? 'Stage 2: Decoder Layer 1' : 'Stage 3: Decoder Layer 2'} Multi-Head Attention &amp; FFN Circuit
            </div>

            {/* QKV Projections */}
            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">1. Query, Key, Value Projections (Linear Linear Transformation):</p>
              <div className="text-violet-200 text-xs space-y-1">
                <p>{"Q = X W_Q + b_Q,   W_Q ∈ ℝ^(16×16)"}</p>
                <p>{"K = X_src W_K + b_K,   W_K ∈ ℝ^(16×16)"}</p>
                <p>{"V = X_src W_V + b_V,   W_V ∈ ℝ^(16×16)"}</p>
              </div>
              <p className="text-[11px] text-zinc-400">
                Split into 2 attention heads of head dimension d_head = d_model / num_heads = 16 / 2 = 4.
              </p>
            </div>

            {/* Softmax Scaled Dot-Product */}
            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">2. Scaled Dot-Product Attention Score Matrix A:</p>
              <div className="text-amber-300 text-sm font-bold">
                {"A = softmax( (Q K^T) / √d_head )"}
              </div>
              <div className="flex items-center gap-2 text-cyan-300 text-[11px]">
                <span>Matrix Dims:</span>
                <span>Q [T_dec × 4] × K^T [4 × T_src] → QK^T [T_dec × T_src]</span>
              </div>
            </div>

            {/* LayerNorm & Residual */}
            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">3. LayerNorm &amp; Residual Stream Connection:</p>
              <div className="text-emerald-300 text-xs">
                {"LN(x) = ( (x - μ) / √(σ² + ε) ) ⊙ γ + β"}
              </div>
              <p className="text-[11px] text-zinc-400">
                {"Residual addition: x_out = LN(x_in + MHA(x_in))"}
              </p>
            </div>

            {/* GELU FFN */}
            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">4. Feed-Forward Network (FFN) with GELU Activation:</p>
              <div className="text-indigo-300 text-xs">
                {"FFN(x) = GELU(x W_1 + b_1) W_2 + b_2"}
              </div>
              <div className="flex items-center gap-2 text-emerald-300 text-[11px]">
                <span>Dims: [1 × 16] × [16 × 32] → GELU([1 × 32]) × [32 × 16] → [1 × 16]</span>
              </div>
            </div>
          </div>
        )}

        {/* Stage 4: Logit Classifier */}
        {stage === 'classifier' && (
          <div className="space-y-4 text-xs leading-relaxed font-mono">
            <div className="flex items-center gap-2 text-emerald-300 font-bold text-sm">
              <Zap className="w-4 h-4 text-emerald-400" />
              Stage 4: Logit Output Classifier &amp; Softmax Distribution
            </div>

            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">1. Linear Classification Projection z:</p>
              <div className="text-emerald-200 text-sm font-bold">
                {"z = h_dec W_out^T + b_out"}
              </div>
              <p className="text-[11px] text-zinc-400">
                Where h_dec ∈ ℝ^16 is the final decoder state vector, W_out ∈ ℝ^(42×16) is the classifier weight matrix, and b_out ∈ ℝ^42 is the bias vector.
              </p>
            </div>

            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">2. Matrix Operation Dimensions:</p>
              <div className="flex items-center gap-2 text-cyan-300 text-xs">
                <span>h_dec [1 × 16]</span>
                <span>×</span>
                <span>W_out^T [16 × 42]</span>
                <span>+ b_out [1 × 42]</span>
                <ArrowRight className="w-3.5 h-3.5 text-zinc-500" />
                <span className="font-bold text-emerald-300">Logits z [1 × 42]</span>
              </div>
            </div>

            <div className="p-3 bg-zinc-900 rounded-lg border border-zinc-800 space-y-2">
              <p className="font-bold text-zinc-300">3. Softmax Probabilities &amp; Logit Margin Δz:</p>
              <div className="text-amber-300 text-xs">
                {"P(y_t = v | y_<t, X) = exp(z_v) / ∑_{u=0}^{41} exp(z_u)"}
              </div>
              <div className="text-violet-300 text-xs">
                {"Logit Margin Δz = z_top1 - z_top2"}
              </div>
            </div>
          </div>
        )}

        {/* Modal Footer */}
        <div className="border-t border-zinc-850 pt-3 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition-colors"
          >
            Close Formula Modal
          </button>
        </div>
      </div>
    </div>
  );
}
