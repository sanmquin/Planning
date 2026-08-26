import modelsWeightsData from '../data/models_weights.json';
import graphSamplesData from '../data/graph_samples.json';
import trainingHistoryData from '../data/training_history.json';

export interface GraphSample {
  id: number;
  trace: number[];
  sp: number[];
  nodes: number[];
  edges: number[][]; // [u, v] pairs
  node_coords: Record<string, [number, number]>;
  mapping: Record<string, number>;
  backtracks: number;
  node_backtraces: Record<string, number>;
}

export interface TrainingHistory {
  train_loss: number[];
  val_epochs: number[];
  val_loss: number[];
  val_tf_acc: number[];
  val_exact_match: number[];
  val_path_validity: number[];
}

export interface AttentionHeadWeights {
  attnWeights: number[][]; // [Q_len x K_len]
}

export interface LayerAttentionTrace {
  selfAttnHeads: AttentionHeadWeights[]; // 4 heads
  crossAttnHeads?: AttentionHeadWeights[]; // 4 heads (for decoder layers)
}

export interface AutoregressiveStepTrace {
  step: number;
  currTgtSeq: number[];
  encoderSelfAttn: LayerAttentionTrace[]; // 2 layers
  decoderSelfAttn: LayerAttentionTrace[]; // 2 layers
  decoderCrossAttn: LayerAttentionTrace[]; // 2 layers
  logits: number[]; // [42]
  probabilities: number[]; // [42]
  topK: { token: number; prob: number; logit: number }[];
  predictedToken: number;
}

export interface FullInferenceTrace {
  modelEpoch: "300" | "400";
  sample: GraphSample;
  srcTokens: number[];
  paddedSrcTokens: number[];
  groundTruthSP: number[];
  steps: AutoregressiveStepTrace[];
  predictedSP: number[];
  isExactMatch: boolean;
  isValidConnectivity: boolean;
}

// Global data exports
export const GRAPH_SAMPLES: GraphSample[] = graphSamplesData.samples as unknown as GraphSample[];
export const TRAINING_HISTORY: Record<string, TrainingHistory> = trainingHistoryData as unknown as Record<string, TrainingHistory>;
export const VOCAB_SIZE = graphSamplesData.vocab_size; // 42
export const PAD_TOKEN = graphSamplesData.pad_token; // 40
export const STOP_TOKEN = graphSamplesData.stop_token; // 41
export const MAX_SRC_LEN = graphSamplesData.max_src_len; // 50
export const MAX_TGT_LEN = graphSamplesData.max_tgt_len; // 21

type RawWeights = Record<string, any>;
const WEIGHTS_MAP: Record<string, RawWeights> = modelsWeightsData as unknown as Record<string, RawWeights>;

// Helper GELU activation
function gelu(x: number): number {
  return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

// Softmax helper over 1D array with masking support
function softmax(logits: number[], mask?: boolean[]): number[] {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (mask && mask[i]) continue;
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  if (maxLogit === -Infinity) maxLogit = 0;

  const exps = new Array(logits.length).fill(0);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    if (mask && mask[i]) {
      exps[i] = 0;
    } else {
      exps[i] = Math.exp(logits[i] - maxLogit);
      sum += exps[i];
    }
  }

  if (sum === 0) sum = 1;
  return exps.map(e => e / sum);
}

// LayerNorm helper
function layerNorm(
  x: number[],
  weight: number[],
  bias: number[],
  eps = 1e-5
): number[] {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;

  let variance = 0;
  for (let i = 0; i < d; i++) {
    const diff = x[i] - mean;
    variance += diff * diff;
  }
  variance /= d;

  const std = Math.sqrt(variance + eps);
  const out = new Array(d);
  for (let i = 0; i < d; i++) {
    out[i] = ((x[i] - mean) / std) * weight[i] + bias[i];
  }
  return out;
}

// Compute Sinusoidal Positional Encoding
function getPositionalEncoding(seqLen: number, dModel = 16): number[][] {
  const pe: number[][] = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = new Array(dModel).fill(0);
    for (let i = 0; i < dModel; i += 2) {
      const divTerm = Math.exp(-Math.log(10000.0) * (i / dModel));
      row[i] = Math.sin(pos * divTerm);
      if (i + 1 < dModel) {
        row[i + 1] = Math.cos(pos * divTerm);
      }
    }
    pe.push(row);
  }
  return pe;
}

// Single Head/Multi Head Attention calculation
interface MHAOutput {
  output: number[][]; // [SeqLen x d_model]
  attnWeights: number[][][]; // [numHeads x Q_len x K_len]
}

function computeMultiHeadAttention(
  query: number[][], // [Q_len x d_model]
  key: number[][],   // [K_len x d_model]
  value: number[][], // [K_len x d_model]
  inProjWeight: number[][], // [48 x 16]
  inProjBias: number[],     // [48]
  outProjWeight: number[][], // [16 x 16]
  outProjBias: number[],     // [16]
  numHeads = 4,
  dModel = 16,
  keyPaddingMask?: boolean[], // length K_len
  causalMask?: boolean[][]   // [Q_len x K_len]
): MHAOutput {
  const qLen = query.length;
  const kLen = key.length;
  const dHead = dModel / numHeads; // 4

  // Extract Q, K, V projections from in_proj
  const Wq = inProjWeight.slice(0, dModel);
  const Bq = inProjBias.slice(0, dModel);

  const Wk = inProjWeight.slice(dModel, 2 * dModel);
  const Bk = inProjBias.slice(dModel, 2 * dModel);

  const Wv = inProjWeight.slice(2 * dModel, 3 * dModel);
  const Bv = inProjBias.slice(2 * dModel, 3 * dModel);

  // Project Query
  const Q: number[][] = [];
  for (let i = 0; i < qLen; i++) {
    const row = new Array(dModel).fill(0);
    for (let d = 0; d < dModel; d++) {
      let sum = Bq[d];
      for (let k = 0; k < dModel; k++) sum += query[i][k] * Wq[d][k];
      row[d] = sum;
    }
    Q.push(row);
  }

  // Project Key
  const K: number[][] = [];
  for (let j = 0; j < kLen; j++) {
    const row = new Array(dModel).fill(0);
    for (let d = 0; d < dModel; d++) {
      let sum = Bk[d];
      for (let k = 0; k < dModel; k++) sum += key[j][k] * Wk[d][k];
      row[d] = sum;
    }
    K.push(row);
  }

  // Project Value
  const V: number[][] = [];
  for (let j = 0; j < kLen; j++) {
    const row = new Array(dModel).fill(0);
    for (let d = 0; d < dModel; d++) {
      let sum = Bv[d];
      for (let k = 0; k < dModel; k++) sum += value[j][k] * Wv[d][k];
      row[d] = sum;
    }
    V.push(row);
  }

  const allHeadAttn: number[][][] = []; // [numHeads x Q_len x K_len]
  const headOutputs: number[][][] = []; // [numHeads x Q_len x dHead]

  const scale = Math.sqrt(dHead);

  for (let h = 0; h < numHeads; h++) {
    const startIdx = h * dHead;
    const endIdx = (h + 1) * dHead;

    const headAttnMatrix: number[][] = [];
    const headOutMatrix: number[][] = [];

    for (let qIdx = 0; qIdx < qLen; qIdx++) {
      const logits = new Array(kLen).fill(0);
      const mask = new Array(kLen).fill(false);

      for (let kIdx = 0; kIdx < kLen; kIdx++) {
        if (keyPaddingMask && keyPaddingMask[kIdx]) {
          mask[kIdx] = true;
        }
        if (causalMask && causalMask[qIdx] && !causalMask[qIdx][kIdx]) {
          mask[kIdx] = true;
        }

        let dot = 0;
        for (let d = startIdx; d < endIdx; d++) {
          dot += Q[qIdx][d] * K[kIdx][d];
        }
        logits[kIdx] = dot / scale;
      }

      const probs = softmax(logits, mask);
      headAttnMatrix.push(probs);

      // Weighted sum over V
      const vOut = new Array(dHead).fill(0);
      for (let kIdx = 0; kIdx < kLen; kIdx++) {
        const p = probs[kIdx];
        for (let d = 0; d < dHead; d++) {
          vOut[d] += p * V[kIdx][startIdx + d];
        }
      }
      headOutMatrix.push(vOut);
    }

    allHeadAttn.push(headAttnMatrix);
    headOutputs.push(headOutMatrix);
  }

  // Concatenate head outputs -> [Q_len x dModel]
  const concatHeads: number[][] = [];
  for (let i = 0; i < qLen; i++) {
    const row = new Array(dModel).fill(0);
    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < dHead; d++) {
        row[h * dHead + d] = headOutputs[h][i][d];
      }
    }
    concatHeads.push(row);
  }

  // Final linear output projection
  const finalOut: number[][] = [];
  for (let i = 0; i < qLen; i++) {
    const row = new Array(dModel).fill(0);
    for (let d = 0; d < dModel; d++) {
      let sum = outProjBias[d];
      for (let k = 0; k < dModel; k++) {
        sum += concatHeads[i][k] * outProjWeight[d][k];
      }
      row[d] = sum;
    }
    finalOut.push(row);
  }

  return {
    output: finalOut,
    attnWeights: allHeadAttn
  };
}

// Forward Pass Engine for AutoregressiveGraphTransformer
export function runSingleStepInference(
  modelEpoch: "300" | "400",
  sample: GraphSample,
  currTgtSeq: number[]
): AutoregressiveStepTrace {
  const rawWeights = WEIGHTS_MAP[modelEpoch];

  // 1. Prepare Input Trace (Padded to MAX_SRC_LEN=50)
  const src = [...sample.trace];
  while (src.length < MAX_SRC_LEN) {
    src.push(PAD_TOKEN);
  }
  const srcMask = src.map(t => t === PAD_TOKEN);

  // 2. Token Embedding + PE for Encoder
  const tokenEmbed = rawWeights["token_embedding.weight"] as number[][];
  const pe = getPositionalEncoding(MAX_SRC_LEN, 16);

  let encMem: number[][] = [];
  for (let i = 0; i < MAX_SRC_LEN; i++) {
    const tok = src[i];
    const emb = tokenEmbed[tok];
    const row = new Array(16);
    for (let d = 0; d < 16; d++) {
      row[d] = emb[d] + pe[i][d];
    }
    encMem.push(row);
  }

  const encoderSelfAttnTraces: LayerAttentionTrace[] = [];

  // 3. Encoder Layers (2 Layers)
  for (let l = 0; l < 2; l++) {
    const prefix = `encoder.layers.${l}.`;
    const inProjW = rawWeights[prefix + "self_attn.in_proj_weight"];
    const inProjB = rawWeights[prefix + "self_attn.in_proj_bias"];
    const outProjW = rawWeights[prefix + "self_attn.out_proj.weight"];
    const outProjB = rawWeights[prefix + "self_attn.out_proj.bias"];

    const mha = computeMultiHeadAttention(
      encMem,
      encMem,
      encMem,
      inProjW,
      inProjB,
      outProjW,
      outProjB,
      4,
      16,
      srcMask
    );

    encoderSelfAttnTraces.push({
      selfAttnHeads: mha.attnWeights.map(w => ({ attnWeights: w }))
    });

    // Residual + Norm1
    const norm1W = rawWeights[prefix + "norm1.weight"];
    const norm1B = rawWeights[prefix + "norm1.bias"];
    let x1: number[][] = [];
    for (let i = 0; i < MAX_SRC_LEN; i++) {
      const res = encMem[i].map((v, d) => v + mha.output[i][d]);
      x1.push(layerNorm(res, norm1W, norm1B));
    }

    // FFN (linear1 -> gelu -> linear2) + Residual + Norm2
    const l1W = rawWeights[prefix + "linear1.weight"]; // [32 x 16]
    const l1B = rawWeights[prefix + "linear1.bias"];   // [32]
    const l2W = rawWeights[prefix + "linear2.weight"]; // [16 x 32]
    const l2B = rawWeights[prefix + "linear2.bias"];   // [16]

    const norm2W = rawWeights[prefix + "norm2.weight"];
    const norm2B = rawWeights[prefix + "norm2.bias"];

    let x2: number[][] = [];
    for (let i = 0; i < MAX_SRC_LEN; i++) {
      const ffnHidden = new Array(32).fill(0);
      for (let h = 0; h < 32; h++) {
        let sum = l1B[h];
        for (let d = 0; d < 16; d++) sum += x1[i][d] * l1W[h][d];
        ffnHidden[h] = gelu(sum);
      }

      const ffnOut = new Array(16).fill(0);
      for (let d = 0; d < 16; d++) {
        let sum = l2B[d];
        for (let h = 0; h < 32; h++) sum += ffnHidden[h] * l2W[d][h];
        ffnOut[d] = sum;
      }

      const res = x1[i].map((v, d) => v + ffnOut[d]);
      x2.push(layerNorm(res, norm2W, norm2B));
    }

    encMem = x2;
  }

  // 4. Decoder Input (currTgtSeq) + PE
  const tgtLen = currTgtSeq.length;
  const tgtPE = getPositionalEncoding(tgtLen, 16);

  let decState: number[][] = [];
  for (let i = 0; i < tgtLen; i++) {
    const tok = currTgtSeq[i];
    const emb = tokenEmbed[tok];
    const row = new Array(16);
    for (let d = 0; d < 16; d++) {
      row[d] = emb[d] + tgtPE[i][d];
    }
    decState.push(row);
  }

  // Construct Causal Triangular Mask
  const causalMask: boolean[][] = [];
  for (let q = 0; q < tgtLen; q++) {
    const row = new Array(tgtLen).fill(false);
    for (let k = 0; k <= q; k++) {
      row[k] = true;
    }
    causalMask.push(row);
  }

  const decoderSelfAttnTraces: LayerAttentionTrace[] = [];
  const decoderCrossAttnTraces: LayerAttentionTrace[] = [];

  // 5. Decoder Layers (2 Layers)
  for (let l = 0; l < 2; l++) {
    const prefix = `decoder.layers.${l}.`;

    // Causal Self-Attention
    const selfInProjW = rawWeights[prefix + "self_attn.in_proj_weight"];
    const selfInProjB = rawWeights[prefix + "self_attn.in_proj_bias"];
    const selfOutProjW = rawWeights[prefix + "self_attn.out_proj.weight"];
    const selfOutProjB = rawWeights[prefix + "self_attn.out_proj.bias"];

    const selfMHA = computeMultiHeadAttention(
      decState,
      decState,
      decState,
      selfInProjW,
      selfInProjB,
      selfOutProjW,
      selfOutProjB,
      4,
      16,
      undefined,
      causalMask
    );

    // Residual + Norm1
    const norm1W = rawWeights[prefix + "norm1.weight"];
    const norm1B = rawWeights[prefix + "norm1.bias"];
    let y1: number[][] = [];
    for (let i = 0; i < tgtLen; i++) {
      const res = decState[i].map((v, d) => v + selfMHA.output[i][d]);
      y1.push(layerNorm(res, norm1W, norm1B));
    }

    // Cross-Attention
    const crossInProjW = rawWeights[prefix + "multihead_attn.in_proj_weight"];
    const crossInProjB = rawWeights[prefix + "multihead_attn.in_proj_bias"];
    const crossOutProjW = rawWeights[prefix + "multihead_attn.out_proj.weight"];
    const crossOutProjB = rawWeights[prefix + "multihead_attn.out_proj.bias"];

    const crossMHA = computeMultiHeadAttention(
      y1,       // Query from decoder
      encMem,   // Key from encoder
      encMem,   // Value from encoder
      crossInProjW,
      crossInProjB,
      crossOutProjW,
      crossOutProjB,
      4,
      16,
      srcMask
    );

    decoderSelfAttnTraces.push({
      selfAttnHeads: selfMHA.attnWeights.map(w => ({ attnWeights: w }))
    });

    decoderCrossAttnTraces.push({
      selfAttnHeads: [],
      crossAttnHeads: crossMHA.attnWeights.map(w => ({ attnWeights: w }))
    });

    // Residual + Norm2
    const norm2W = rawWeights[prefix + "norm2.weight"];
    const norm2B = rawWeights[prefix + "norm2.bias"];
    let y2: number[][] = [];
    for (let i = 0; i < tgtLen; i++) {
      const res = y1[i].map((v, d) => v + crossMHA.output[i][d]);
      y2.push(layerNorm(res, norm2W, norm2B));
    }

    // FFN + Residual + Norm3
    const l1W = rawWeights[prefix + "linear1.weight"]; // [32 x 16]
    const l1B = rawWeights[prefix + "linear1.bias"];
    const l2W = rawWeights[prefix + "linear2.weight"]; // [16 x 32]
    const l2B = rawWeights[prefix + "linear2.bias"];

    const norm3W = rawWeights[prefix + "norm3.weight"];
    const norm3B = rawWeights[prefix + "norm3.bias"];

    let y3: number[][] = [];
    for (let i = 0; i < tgtLen; i++) {
      const ffnHidden = new Array(32).fill(0);
      for (let h = 0; h < 32; h++) {
        let sum = l1B[h];
        for (let d = 0; d < 16; d++) sum += y2[i][d] * l1W[h][d];
        ffnHidden[h] = gelu(sum);
      }

      const ffnOut = new Array(16).fill(0);
      for (let d = 0; d < 16; d++) {
        let sum = l2B[d];
        for (let h = 0; h < 32; h++) sum += ffnHidden[h] * l2W[d][h];
        ffnOut[d] = sum;
      }

      const res = y2[i].map((v, d) => v + ffnOut[d]);
      y3.push(layerNorm(res, norm3W, norm3B));
    }

    decState = y3;
  }

  // 6. Output Classifier FC Projection for Last Token Position
  const fcW = rawWeights["fc_out.weight"] as number[][]; // [42 x 16]
  const fcB = rawWeights["fc_out.bias"] as number[];    // [42]

  const lastVec = decState[tgtLen - 1];
  const logits = new Array(VOCAB_SIZE).fill(0);
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let sum = fcB[v];
    for (let d = 0; d < 16; d++) {
      sum += lastVec[d] * fcW[v][d];
    }
    logits[v] = sum;
  }

  const probs = softmax(logits);

  // Top-5 Token Probabilities
  const tokenPairs = probs.map((prob, tok) => ({ token: tok, prob, logit: logits[tok] }));
  tokenPairs.sort((a, b) => b.prob - a.prob);
  const topK = tokenPairs.slice(0, 5);

  const predictedToken = topK[0].token;

  return {
    step: tgtLen - 1,
    currTgtSeq: [...currTgtSeq],
    encoderSelfAttn: encoderSelfAttnTraces,
    decoderSelfAttn: decoderSelfAttnTraces,
    decoderCrossAttn: decoderCrossAttnTraces,
    logits,
    probabilities: probs,
    topK,
    predictedToken
  };
}

// Helper to run full rollout trajectory on a sample
export function runFullRolloutInference(
  modelEpoch: "300" | "400",
  sample: GraphSample
): FullInferenceTrace {
  const srcTokens = [...sample.trace];
  const paddedSrcTokens = [...srcTokens];
  while (paddedSrcTokens.length < MAX_SRC_LEN) {
    paddedSrcTokens.push(PAD_TOKEN);
  }

  const groundTruthSP = [...sample.sp];
  const steps: AutoregressiveStepTrace[] = [];

  const currSeq = [srcTokens[0]]; // Start token

  for (let s = 0; s < MAX_TGT_LEN - 1; s++) {
    const stepTrace = runSingleStepInference(modelEpoch, sample, currSeq);
    steps.push(stepTrace);

    const nextTok = stepTrace.predictedToken;
    if (nextTok === STOP_TOKEN || nextTok === PAD_TOKEN) {
      break;
    }
    currSeq.push(nextTok);
  }

  const predictedSP = [...currSeq];

  // Check Exact Match
  const isExactMatch =
    predictedSP.length === groundTruthSP.length &&
    predictedSP.every((val, idx) => val === groundTruthSP[idx]);

  // Check Path Connectivity Validity
  let isValidConnectivity = false;
  if (
    predictedSP.length >= 2 &&
    predictedSP[0] === groundTruthSP[0] &&
    predictedSP[predictedSP.length - 1] === groundTruthSP[groundTruthSP.length - 1]
  ) {
    isValidConnectivity = true;
    const edgeSet = new Set(
      sample.edges.flatMap(([u, v]) => [`${u}-${v}`, `${v}-${u}`])
    );
    for (let k = 0; k < predictedSP.length - 1; k++) {
      const u = predictedSP[k];
      const v = predictedSP[k + 1];
      if (!edgeSet.has(`${u}-${v}`)) {
        isValidConnectivity = false;
        break;
      }
    }
  }

  return {
    modelEpoch,
    sample,
    srcTokens,
    paddedSrcTokens,
    groundTruthSP,
    steps,
    predictedSP,
    isExactMatch,
    isValidConnectivity
  };
}
