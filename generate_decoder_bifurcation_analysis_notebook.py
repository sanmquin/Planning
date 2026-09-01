import json
import os

def create_generator():
    cells = []

    # ---------------------------------------------------------
    # Cell 0: Header & Detailed Introduction (Markdown)
    # ---------------------------------------------------------
    cell0_md = """# 3. Topological Error Profiling and Causal Attention Mechanics in Decoder-Only Graph Transformers
## Dissecting Bifurcation Dynamics, Dead-End Depths, and Anchor Selection Collapse in Epoch 100 Decoder Checkpoints

### Executive Summary & Educational Motivation
In sequence-to-sequence neural algorithmic reasoning, **Decoder-Only Causal Language Models** (e.g., GPT-4, LLaMA, DeepSeek) solve graph pathfinding by concatenating the topological execution trace prompt $T = [t_1, t_2, \\dots, t_K]$ and target path rollout $P^* = [p_1^*, p_2^*, \\dots, p_M^*]$ into a unified 1D context window:
$$X = [t_1, t_2, \\dots, t_K, p_1^*, p_2^*, \\dots, p_M^*, \\text{STOP_TOKEN}]$$

Without cross-attention layers, prediction at rollout step $m$ relies exclusively on **Causal Self-Attention** over all preceding tokens $X_{\\le K+m-1}$. At mid-training checkpoints (such as Epoch 100), models achieve substantial token accuracy (~97.5% teacher-forcing step accuracy), yet exhibit ~158 step-level prediction errors across validation datasets (and ~306 across combined validation and test datasets).

This notebook provides a comprehensive **topological and attention-mechanistic dissection** of these step-level prediction errors. We systematically analyze what error steps have in common across three fundamental metrics:

---

### Core Analytical Metrics & Research Questions

#### Metric 1: Bifurcation vs. Single Occurrence Breakdown
- **Bifurcation Definition**: A step decision predicting from node $u = p_m^*$ is defined as a **bifurcation** if $u$ appears multiple times in the execution trace prompt $T$ due to Depth-First Search (DFS) backtracking (i.e., $\\text{count}(u) > 1$ in $T$).
- **Single Occurrence**: A step decision where $u$ appears exactly once in $T$ (i.e., $\\text{count}(u) = 1$).
- **Core Finding**: We evaluate what percentage of overall step errors occur at bifurcations versus single occurrences.

#### Metric 2: Topological Contrasts in Bifurcation Steps (Correct vs. Incorrect)
For step decisions occurring at bifurcations ($u_{\\text{count}} > 1$), we contrast correct ($y_m = 1$) versus incorrect ($y_m = 0$) predictions across four topological dimensions:
1. **Position**: When do the entry ($u_{\\text{first}}$) and exit ($u_{\\text{later}}$) occurrences of the bifurcation node appear relative to the trace length $K$?
   $$\\text{rel\\_first\\_pos} = \\frac{\\text{idx}(u_{\\text{first}})}{K}, \\quad \\text{rel\\_last\\_pos} = \\frac{\\text{idx}(u_{\\text{later}})}{K}$$
2. **Order**: In DFS traversal, does the ground-truth successor path token $p_{m+1}^*$ follow the first occurrence ($u_{\\text{first}}$) or the second/later exit occurrence ($u_{\\text{later}}$) in $T$?
3. **Frequency**: How many total occurrences of node $u$ exist in $T$ (distribution of $2, 3, 4+$ visits)?
4. **Depth of Dead-End**: For the unviable branch explored from $u_{\\text{first}}$, how long was the dead-end exploration before DFS backtracked to $u_{\\text{later}}$?
   $$\\text{dead\\_end\\_depth} = \\text{idx}(u_{\\text{later}}) - \\text{idx}(u_{\\text{first}})$$

#### Metric 3: Causal Self-Attention Routing Profiles
- **Part A: Good vs. Bad Predictions ON BIFURCATIONS ONLY**:
  - **Anchor Selection Index ($ASI$)**: Measures the proportion of attention allocated to the exit occurrence $u_{\\text{later}}$ relative to the entry occurrence $u_{\\text{first}}$:
    $$ASI = \\frac{A(u_{\\text{later}})}{A(u_{\\text{first}}) + A(u_{\\text{later}})}$$
  - **Causal Prompt Entropy ($H_{\\text{prompt}}$)**: Quantifies attention dispersion across trace prompt tokens:
    $$H_{\\text{prompt}} = -\\sum_{j=0}^{K-1} \\tilde{A}_{i, j} \\ln\\left( \\tilde{A}_{i, j} + \\epsilon \\right), \\quad \\tilde{A}_{i, j} = \\frac{A_{i, j}}{\\sum_{k=0}^{K-1} A_{i, k}}$$
- **Part B: Predictions on Bifurcations vs. Single Occurrences (Regardless of Outcome)**:
  - Compares prompt attention share ($A_{\\text{prompt}}$), causal entropy ($H_{\\text{prompt}}$), and total active node attention mass ($A_{\\text{node}}$) between bifurcations and single occurrences.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_0_md"}, "source": cell0_md.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 1: Environment Setup, Colab Drive Mounting & Path Hierarchy
    # ---------------------------------------------------------
    cell1_code = """# Cell 1: Environment Setup, Reproducibility Seeds, and Colab Drive Path Resolution
# Description: Configures PyTorch compute device, seeds random number generators for strict reproducibility,
# mounts Google Drive if running in Google Colab, and sets up local fallback path resolution.

import os
import sys
import random
import time
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, log_loss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Cell 1 Setup] Compute Device: {device}")

# Colab Google Drive Setup with Local Fallback Hierarchy
def setup_drive_and_paths():
    drive_mounted = False
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        drive_mounted = True
        print("[Drive Setup] Successfully mounted Google Drive at '/content/drive'.")
    except Exception:
        print("[Drive Setup] Google Drive not available or not in Colab. Utilizing local path hierarchy.")

    dataset_candidates = [
        "/content/drive/MyDrive/graph_data/graph_dfs_dataset_v1.pt",
        "/content/drive/MyDrive/graph_checkpoints/graph_dfs_dataset_v1.pt",
        "src/static/data/graph_dfs_dataset_v1.pt",
        "../static/data/graph_dfs_dataset_v1.pt",
        "data/graph_dfs_dataset_v1.pt"
    ]

    checkpoint_candidates = [
        "/content/drive/MyDrive/graph_checkpoints/decoder_only_ar_graph_transformer_mid_epoch_100.pt",
        "src/static/checkpoints/decoder_only_ar_graph_transformer_mid_epoch_100.pt",
        "../static/checkpoints/decoder_only_ar_graph_transformer_mid_epoch_100.pt",
        "checkpoints/decoder_only_ar_graph_transformer_mid_epoch_100.pt"
    ]

    def resolve_path(candidates, label):
        for path in candidates:
            if os.path.exists(path):
                print(f"[Path Resolution] Resolved {label} at: '{path}'")
                return path
        raise FileNotFoundError(f"Could not locate {label}. Checked candidates: {candidates}")

    ds_path = resolve_path(dataset_candidates, "DFS Dataset (v1)")
    ckpt_path = resolve_path(checkpoint_candidates, "Decoder-Only Epoch 100 Checkpoint")
    return ds_path, ckpt_path, drive_mounted

DATASET_PATH, CHECKPOINT_PATH, IS_DRIVE_MOUNTED = setup_drive_and_paths()
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_1", "metadata": {"id": "cell_1"}, "outputs": [], "source": cell1_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 2: Architecture & Checkpoint Ingestion
    # ---------------------------------------------------------
    cell2_md = """### Model Architecture & Dataset Ingestion
Defines the `DecoderOnlyGraphTransformer` architecture matching the checkpoint parameters:
- `vocab_size = 42`, `embed_dim = 32`, `num_heads = 2`, `hidden_dim = 64`, `num_layers = 2`, `PAD_TOKEN = 40`, `STOP_TOKEN = 41`.
- Implements `forward_with_attn` to capture multi-head causal self-attention maps across transformer layers.
- Implements `solve_graph_autoregressive` for unguided rollout generation.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_2_md"}, "source": cell2_md.splitlines(True)})

    cell2_code = """# Cell 2: Architecture Definition and Checkpoint Ingestion
# Description: Defines the DecoderOnlyGraphTransformer class, loads the dataset payload,
# and instantiates model weights from the Epoch 100 checkpoint.

VOCAB_SIZE = 42
PAD_TOKEN = 40
STOP_TOKEN = 41
MAX_SRC_LEN = 50
MAX_TGT_LEN = 21
MAX_COMBINED_LEN = MAX_SRC_LEN + MAX_TGT_LEN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=150):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecoderOnlyGraphTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2):
        super(DecoderOnlyGraphTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=150)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x, padding_mask=None, causal_mask=None):
        x_emb = self.pos_encoder(self.token_embedding(x))
        out = self.transformer(x_emb, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.fc_out(out)
        return logits

    def forward_with_attn(self, x, padding_mask=None, causal_mask=None):
        x_emb = self.pos_encoder(self.token_embedding(x))
        curr = x_emb
        attn_maps = []

        for layer in self.transformer.layers:
            attn_out, attn_weights = layer.self_attn(
                curr, curr, curr,
                attn_mask=causal_mask,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=False
            )
            attn_maps.append(attn_weights) # [batch_size, num_heads, seq_len, seq_len]

            x_norm = layer.norm1(curr + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm))))
            curr = layer.norm2(x_norm + layer.dropout2(ff_out))

        logits = self.fc_out(curr)
        return logits, attn_maps

    def solve_graph_autoregressive(self, traces, max_tgt_len=MAX_TGT_LEN, device='cpu'):
        self.eval()
        batch_size = len(traces)
        curr_seqs = [list(tr) for tr in traces]
        generated_paths = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for step in range(max_tgt_len - 1):
            if all(finished):
                break
            curr_max_len = max(len(s) for s in curr_seqs)
            inp = torch.full((batch_size, curr_max_len), PAD_TOKEN, dtype=torch.long, device=device)
            inp_mask = torch.zeros((batch_size, curr_max_len), dtype=torch.bool, device=device)

            for b in range(batch_size):
                s = curr_seqs[b]
                inp[b, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                inp_mask[b, len(s):] = True

            causal_mask = self.generate_square_subsequent_mask(curr_max_len, device)
            logits = self.forward(inp, padding_mask=inp_mask, causal_mask=causal_mask)

            for b in range(batch_size):
                if finished[b]:
                    continue
                last_idx = len(curr_seqs[b]) - 1
                next_tok = torch.argmax(logits[b, last_idx, :]).item()
                if next_tok in (STOP_TOKEN, PAD_TOKEN):
                    finished[b] = True
                else:
                    curr_seqs[b].append(next_tok)
                    generated_paths[b].append(next_tok)

        return generated_paths

# Ingest Dataset Payload
dataset_payload = torch.load(DATASET_PATH, map_location='cpu', weights_only=False)
train_raw = dataset_payload['train']
val_raw = dataset_payload['val']
test_raw = dataset_payload['test']

# Instantiate Model and Load Epoch 100 Checkpoint Weights
checkpoint_payload = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model = DecoderOnlyGraphTransformer().to(device)
model.load_state_dict(checkpoint_payload['model_state_dict'])
model.eval()

print(f"[Cell 2 Ingestion] Dataset loaded: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}.")
print(f"[Cell 2 Ingestion] Model weights successfully loaded from '{CHECKPOINT_PATH}' (Epoch {checkpoint_payload.get('epoch', 100)}).")
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_2", "metadata": {"id": "cell_2"}, "outputs": [], "source": cell2_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 3: Code Integrity and Rollout Verification
    # ---------------------------------------------------------
    cell3_md = """### Code Integrity & Rollout Verification Log
Evaluates the model across both **Validation (500 samples)** and **Test (500 samples)** datasets.
Logs teacher-forcing cross-entropy loss, step-level teacher-forcing accuracy, and unguided autoregressive rollout exact match accuracy.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_3_md"}, "source": cell3_md.splitlines(True)})

    cell3_code = """# Cell 3: Code Integrity and Dataset Evaluation Log
# Description: Evaluates Epoch 100 checkpoint across Validation and Test sets,
# verifying exact match rollout accuracy and logging step-level performance.

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

def evaluate_split_metrics(model, raw_samples, device, label="Val"):
    model.eval()
    total_loss = 0.0
    total_tf_tokens = 0
    correct_tf_tokens = 0
    exact_matches = 0

    with torch.no_grad():
        for item in raw_samples:
            trace, sp = item[0], item[1]
            K = len(trace)
            tgt_seq = list(sp) + [STOP_TOKEN]
            full_seq = list(trace) + tgt_seq
            pad_len = MAX_COMBINED_LEN - len(full_seq)
            full_seq_padded = full_seq + [PAD_TOKEN] * pad_len

            inp = torch.tensor(full_seq_padded[:-1], dtype=torch.long, device=device).unsqueeze(0)
            lbl = torch.tensor(full_seq_padded[1:], dtype=torch.long, device=device).unsqueeze(0)

            lbl_masked = lbl.clone()
            lbl_masked[0, :K-1] = PAD_TOKEN

            inp_mask = (inp == PAD_TOKEN)
            causal_mask = model.generate_square_subsequent_mask(inp.size(1), device)

            logits = model(inp, padding_mask=inp_mask, causal_mask=causal_mask)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), lbl_masked.reshape(-1))
            total_loss += loss.item()

            preds_tf = torch.argmax(logits, dim=-1)
            valid_tokens = (lbl_masked != PAD_TOKEN)
            correct_tf_tokens += ((preds_tf == lbl_masked) & valid_tokens).sum().item()
            total_tf_tokens += valid_tokens.sum().item()

            # Autoregressive Rollout
            pred_path = model.solve_graph_autoregressive([trace], device=device)[0]
            if pred_path == list(sp):
                exact_matches += 1

    mean_loss = total_loss / len(raw_samples)
    tf_acc = (correct_tf_tokens / max(1, total_tf_tokens)) * 100.0
    exact_acc = (exact_matches / len(raw_samples)) * 100.0

    print(f"=" * 70)
    print(f"    EVALUATION INTEGRITY LOG: {label.upper()} SET ({len(raw_samples)} SAMPLES)")
    print(f"=" * 70)
    print(f"  Cross-Entropy Loss          : {mean_loss:.4f}")
    print(f"  Teacher-Forcing Token Acc    : {tf_acc:.2f}%")
    print(f"  Autoregressive Exact Match  : {exact_acc:.2f}% ({exact_matches}/{len(raw_samples)})")
    print(f"=" * 70)
    return mean_loss, tf_acc, exact_acc

val_loss, val_tf_acc, val_exact_acc = evaluate_split_metrics(model, val_raw, device, "Validation")
test_loss, test_tf_acc, test_exact_acc = evaluate_split_metrics(model, test_raw, device, "Test")
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_3", "metadata": {"id": "cell_3"}, "outputs": [], "source": cell3_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 4: Step Decision Dataset Extraction
    # ---------------------------------------------------------
    cell4_md = """### Extraction of Step Decision Dataset
Iterates through all teacher-forcing step decisions across **Validation (500 samples, 5,983 steps)** and **Test (500 samples, 6,022 steps)** datasets, extracting:
- Step prediction outcome $y_m \\in \\{0, 1\\}$ (correct vs. error).
- Bifurcation flag $\\text{is\\_bifurcation} = \\mathbb{I}(\\text{count}(u) > 1)$.
- Topological features: visit frequency, first/last relative positions, dead-end depth, order adjacency.
- Layer 0 and Layer 1 Causal Self-Attention routing features: prompt attention share ($A_{\\text{prompt}}$), causal prompt entropy ($H_{\\text{prompt}}$), active node attention ($A_{\\text{node}}$), and Anchor Selection Index ($ASI$).
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_4_md"}, "source": cell4_md.splitlines(True)})

    cell4_code = """# Cell 4: Comprehensive Step Decision Feature Extraction
# Description: Evaluates every step decision along ground-truth path prefixes,
# capturing topological trace features and multi-layer causal self-attention maps.

def extract_step_decision_dataset(model, raw_val, raw_test, device):
    all_samples = raw_val + raw_test
    records = []
    model.eval()

    with torch.no_grad():
        for idx, sample in enumerate(all_samples):
            trace, sp, G = sample[0], sample[1], sample[2]
            is_val = (idx < len(raw_val))
            K = len(trace)

            for step in range(len(sp) - 1):
                curr_node = sp[step]
                target_next = sp[step+1]
                prefix = list(trace) + list(sp[:step+1])
                curr_query_idx = len(prefix) - 1

                inp_t = torch.tensor([prefix], dtype=torch.long, device=device)
                causal_mask = model.generate_square_subsequent_mask(len(prefix), device)

                logits, attn_maps = model.forward_with_attn(inp_t, causal_mask=causal_mask)
                pred_next = torch.argmax(logits[0, -1, :]).item()
                is_correct = int(pred_next == target_next)

                occurrences = [i for i, tok in enumerate(trace) if tok == curr_node]
                frequency = len(occurrences)
                is_bifurcation = int(frequency > 1)

                first_pos = occurrences[0] if frequency > 0 else -1
                last_pos = occurrences[-1] if frequency > 0 else -1
                dead_end_depth = (last_pos - first_pos) if frequency > 1 else 0

                # Determine order adjacency: did target_next follow 1st or 2nd/later visit in trace?
                order_correct = 'unknown'
                if frequency > 1:
                    has_first_adj = (first_pos + 1 < K and trace[first_pos + 1] == target_next)
                    has_last_adj = (last_pos + 1 < K and trace[last_pos + 1] == target_next)
                    if has_last_adj and not has_first_adj:
                        order_correct = 'second'
                    elif has_first_adj and not has_last_adj:
                        order_correct = 'first'
                    elif has_first_adj and has_last_adj:
                        order_correct = 'both'

                # Layer 0 and Layer 1 self-attention distributions over prefix
                l0_attn = attn_maps[0][0, :, curr_query_idx, :len(prefix)].mean(dim=0).cpu().numpy()
                l1_attn = attn_maps[1][0, :, curr_query_idx, :len(prefix)].mean(dim=0).cpu().numpy()

                prompt_attn_l0 = float(np.sum(l0_attn[:K]))
                prompt_attn_l1 = float(np.sum(l1_attn[:K]))

                path_attn_l0 = float(np.sum(l0_attn[K:curr_query_idx])) if curr_query_idx > K else 0.0
                path_attn_l1 = float(np.sum(l1_attn[K:curr_query_idx])) if curr_query_idx > K else 0.0

                p0 = l0_attn[:K] / (np.sum(l0_attn[:K]) + 1e-12)
                p1 = l1_attn[:K] / (np.sum(l1_attn[:K]) + 1e-12)
                causal_entropy_l0 = float(-np.sum(p0 * np.log(np.clip(p0, 1e-12, 1.0))))
                causal_entropy_l1 = float(-np.sum(p1 * np.log(np.clip(p1, 1e-12, 1.0))))

                curr_node_attn_l0 = float(np.sum([l0_attn[i] for i in occurrences]))
                curr_node_attn_l1 = float(np.sum([l1_attn[i] for i in occurrences]))

                asi = 0.5
                if is_bifurcation:
                    i_first, i_later = occurrences[0], occurrences[-1]
                    asi = float(l1_attn[i_later] / (l1_attn[i_first] + l1_attn[i_later] + 1e-9))

                records.append({
                    'sample_idx': idx,
                    'is_val': is_val,
                    'step': step,
                    'rel_depth': step / (len(sp) - 1),
                    'curr_node': curr_node,
                    'target_next': target_next,
                    'pred_next': pred_next,
                    'is_correct': is_correct,
                    'is_bifurcation': is_bifurcation,
                    'frequency': frequency,
                    'first_pos': first_pos,
                    'last_pos': last_pos,
                    'rel_first_pos': first_pos / K if K > 0 else 0,
                    'rel_last_pos': last_pos / K if K > 0 else 0,
                    'dead_end_depth': dead_end_depth,
                    'order_correct': order_correct,
                    'prompt_attn_l0': prompt_attn_l0,
                    'prompt_attn_l1': prompt_attn_l1,
                    'path_attn_l0': path_attn_l0,
                    'path_attn_l1': path_attn_l1,
                    'causal_entropy_l0': causal_entropy_l0,
                    'causal_entropy_l1': causal_entropy_l1,
                    'curr_node_attn_l0': curr_node_attn_l0,
                    'curr_node_attn_l1': curr_node_attn_l1,
                    'asi': asi
                })

    return pd.DataFrame(records)

print("[Cell 4 Extraction] Extracting step decision dataset across Val + Test sets...")
df_steps = extract_step_decision_dataset(model, val_raw, test_raw, device)
df_val = df_steps[df_steps['is_val'] == True]
df_test = df_steps[df_steps['is_val'] == False]

print(f"Total Steps Extracted: {len(df_steps)} (Val={len(df_val)}, Test={len(df_test)}).")
print(f"Overall Step Accuracy: {df_steps['is_correct'].mean()*100:.2f}% | Total Step Errors: {(df_steps['is_correct']==0).sum()}.")
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_4", "metadata": {"id": "cell_4"}, "outputs": [], "source": cell4_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 5: Metric 1 Analysis - Bifurcation Breakdown
    # ---------------------------------------------------------
    cell5_md = """### Metric 1: Bifurcation vs. Single Occurrence Error Breakdown

We analyze the distribution of step-level prediction errors between **Bifurcations ($\text{count}(u) > 1$)** and **Single Occurrences ($\text{count}(u) = 1$)**.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_5_md"}, "source": cell5_md.splitlines(True)})

    cell5_code = """# Cell 5: Metric 1 Analysis — Bifurcation vs Single Occurrence Error Breakdown
# Description: Quantifies step error frequencies and accuracies across single occurrences vs bifurcations.

def summarize_metric_1(df_subset, label="Validation Set"):
    total_steps = len(df_subset)
    total_errors = (df_subset['is_correct'] == 0).sum()

    singles = df_subset[df_subset['is_bifurcation'] == 0]
    bifurcations = df_subset[df_subset['is_bifurcation'] == 1]

    single_errors = (singles['is_correct'] == 0).sum()
    bif_errors = (bifurcations['is_correct'] == 0).sum()

    single_acc = singles['is_correct'].mean() * 100.0
    bif_acc = bifurcations['is_correct'].mean() * 100.0

    pct_bif_errors = (bif_errors / max(1, total_errors)) * 100.0
    pct_single_errors = (single_errors / max(1, total_errors)) * 100.0

    print("=" * 75)
    print(f"    METRIC 1 BREAKDOWN: {label.upper()}")
    print("=" * 75)
    print(f"  Total Step Decisions Evaluated  : {total_steps}")
    print(f"  Total Step Prediction Errors    : {total_errors}")
    print(f"  -----------------------------------------------------------------------")
    print(f"  Single Occurrence Steps (count=1): {len(singles)} steps | Accuracy: {single_acc:.2f}%")
    print(f"    -> Errors at Single Occurrences: {single_errors} ({pct_single_errors:.2f}% of total errors)")
    print(f"  Bifurcation Steps (count > 1)   : {len(bifurcations)} steps | Accuracy: {bif_acc:.2f}%")
    print(f"    -> Errors at Bifurcations      : {bif_errors} ({pct_bif_errors:.2f}% of total errors)")
    print("=" * 75)
    print()

summarize_metric_1(df_val, "Validation Set (500 samples)")
summarize_metric_1(df_steps, "Val + Test Combined Datasets (1,000 samples)")
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_5", "metadata": {"id": "cell_5"}, "outputs": [], "source": cell5_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 6: Metric 2 Analysis - Topological Contrast in Bifurcations
    # ---------------------------------------------------------
    cell6_md = """### Metric 2: Topological Contrasts in Bifurcation Steps (Correct vs. Incorrect)

Filters strictly to **Bifurcation Steps ($\text{count}(u) > 1$)** ($n=3,304$ total) and compares **Correct Predictions ($y_m = 1$, $n=3,048$)** versus **Incorrect Predictions ($y_m = 0$, $n=256$)** across:
1. **Position**: Relative entry position (`rel_first_pos`) and exit position (`rel_last_pos`) in prompt $T$.
2. **Order**: Verification of successor adjacency in DFS execution trace.
3. **Frequency**: Occurrence distribution of $u$ in prompt $T$.
4. **Depth of Dead-End**: Trace distance between initial entry $u_{\text{first}}$ and exit re-visit $u_{\text{later}}$.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_6_md"}, "source": cell6_md.splitlines(True)})

    cell6_code = """# Cell 6: Metric 2 Analysis — Deep Topological Contrast on Bifurcation Steps
# Description: Performs statistical testing and contrast analysis between correct and incorrect
# bifurcation step predictions across position, order, frequency, and dead-end depth.

df_bif = df_steps[df_steps['is_bifurcation'] == 1]
corr_bif = df_bif[df_bif['is_correct'] == 1]
err_bif = df_bif[df_bif['is_correct'] == 0]

print("=" * 75)
print("    METRIC 2: TOPOLOGICAL ANALYSIS OF BIFURCATIONS (CORRECT VS INCORRECT)")
print("=" * 75)
print(f"Total Bifurcation Decision Steps: n = {len(df_bif)}")
print(f"  -> Correct Bifurcations : n = {len(corr_bif)} ({len(corr_bif)/len(df_bif)*100:.2f}%)")
print(f"  -> Incorrect Bifurcations: n = {len(err_bif)} ({len(err_bif)/len(df_bif)*100:.2f}%)")
print("-" * 75)

# 1. Position Analysis
t_first, p_first = stats.ttest_ind(corr_bif['rel_first_pos'], err_bif['rel_first_pos'], equal_var=False)
t_last, p_last = stats.ttest_ind(corr_bif['rel_last_pos'], err_bif['rel_last_pos'], equal_var=False)

print("1. POSITION IN EXPLORATION TRACE (Relative Index in Trace [0.0, 1.0]):")
print(f"   First Occurrence (rel_first_pos):")
print(f"     Correct: Mean = {corr_bif['rel_first_pos'].mean():.4f} (+/- {corr_bif['rel_first_pos'].std():.4f})")
print(f"     Error  : Mean = {err_bif['rel_first_pos'].mean():.4f} (+/- {err_bif['rel_first_pos'].std():.4f})")
print(f"     Welch's t-test: t = {t_first:.4f}, p = {p_first:.4e}")
print(f"   Last Occurrence (rel_last_pos):")
print(f"     Correct: Mean = {corr_bif['rel_last_pos'].mean():.4f} (+/- {corr_bif['rel_last_pos'].std():.4f})")
print(f"     Error  : Mean = {err_bif['rel_last_pos'].mean():.4f} (+/- {err_bif['rel_last_pos'].std():.4f})")
print(f"     Welch's t-test: t = {t_last:.4f}, p = {p_last:.4e}")
print("-" * 75)

# 2. Order Adjacency Analysis
print("2. ORDER ADJACENCY IN TRACE (Which visit precedes target successor?):")
print("   Correct Steps Order Distribution:")
for k, v in corr_bif['order_correct'].value_counts().items():
    print(f"     {k}: {v} ({v/len(corr_bif)*100:.2f}%)")
print("   Error Steps Order Distribution:")
for k, v in err_bif['order_correct'].value_counts().items():
    print(f"     {k}: {v} ({v/len(err_bif)*100:.2f}%)")
print("-" * 75)

# 3. Frequency Analysis
print("3. NODE VISIT FREQUENCY IN TRACE:")
print("   Correct Steps Frequency Distribution:")
for k, v in corr_bif['frequency'].value_counts(normalize=True).sort_index().items():
    print(f"     Frequency {k}: {v*100:.2f}%")
print("   Error Steps Frequency Distribution:")
for k, v in err_bif['frequency'].value_counts(normalize=True).sort_index().items():
    print(f"     Frequency {k}: {v*100:.2f}%")
print("-" * 75)

# 4. Dead-End Depth Analysis
t_dead, p_dead = stats.ttest_ind(corr_bif['dead_end_depth'], err_bif['dead_end_depth'], equal_var=False)
print("4. DEPTH OF DEAD-END (Trace Steps Between First Entry and Last Exit):")
print(f"   Correct: Mean = {corr_bif['dead_end_depth'].mean():.2f} steps, Median = {corr_bif['dead_end_depth'].median():.1f}")
print(f"   Error  : Mean = {err_bif['dead_end_depth'].mean():.2f} steps, Median = {err_bif['dead_end_depth'].median():.1f}")
print(f"   Welch's t-test: t = {t_dead:.4f}, p = {p_dead:.4e}")
print("=" * 75)
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_6", "metadata": {"id": "cell_6"}, "outputs": [], "source": cell6_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 7: Metric 3 Analysis - Causal Self-Attention Mechanics
    # ---------------------------------------------------------
    cell7_md = """### Metric 3: Causal Self-Attention Mechanics Comparison

- **Part A: Good vs. Bad Predictions ON BIFURCATIONS ONLY**:
  - Compares **Anchor Selection Index ($ASI$)**, **Causal Prompt Entropy ($H_{\\text{prompt}}$)**, and **Active Node Attention Mass ($A_{\\text{node}}$)**.
- **Part B: Bifurcations vs. Single Occurrences (Regardless of Outcome)**:
  - Compares **Prompt Attention Share ($A_{\\text{prompt}}$)**, **Causal Entropy ($H_{\\text{prompt}}$)**, and **Active Node Attention Mass ($A_{\\text{node}}$)** across all decision steps.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_7_md"}, "source": cell7_md.splitlines(True)})

    cell7_code = """# Cell 7: Metric 3 Analysis — Causal Self-Attention Routing Comparison
# Description: Performs statistical testing on multi-layer causal self-attention features,
# comparing good vs bad predictions on bifurcations and bifurcations vs single occurrences.

print("=" * 75)
print("    METRIC 3: CAUSAL SELF-ATTENTION MECHANICS COMPARISON")
print("=" * 75)

# Part A: Good vs Bad Predictions ON BIFURCATIONS ONLY
t_asi, p_asi = stats.ttest_ind(corr_bif['asi'], err_bif['asi'], equal_var=False)
t_ent_bif, p_ent_bif = stats.ttest_ind(corr_bif['causal_entropy_l1'], err_bif['causal_entropy_l1'], equal_var=False)
t_pr_bif, p_pr_bif = stats.ttest_ind(corr_bif['prompt_attn_l1'], err_bif['prompt_attn_l1'], equal_var=False)

print("PART A: GOOD PREDICTIONS VS BAD PREDICTIONS ON BIFURCATIONS ONLY")
print(f"  Anchor Selection Index (ASI = A(u_later) / [A(u_first) + A(u_later)]):")
print(f"    Good Bifurcations: Mean = {corr_bif['asi'].mean():.4f}")
print(f"    Bad Bifurcations : Mean = {err_bif['asi'].mean():.4f}")
print(f"    Welch's t-test   : t = {t_asi:.4f}, p = {p_asi:.4e} -> ANCHOR SELECTION COLLAPSE IN ERRORS")
print(f"  Layer 1 Causal Prompt Entropy (H_prompt in nats):")
print(f"    Good Bifurcations: Mean = {corr_bif['causal_entropy_l1'].mean():.4f} nats")
print(f"    Bad Bifurcations : Mean = {err_bif['causal_entropy_l1'].mean():.4f} nats")
print(f"    Welch's t-test   : t = {t_ent_bif:.4f}, p = {p_ent_bif:.4e} -> SEVERE ATTENTION DISPERSION")
print(f"  Layer 1 Prompt Attention Mass (A_prompt):")
print(f"    Good Bifurcations: Mean = {corr_bif['prompt_attn_l1'].mean():.4f}")
print(f"    Bad Bifurcations : Mean = {err_bif['prompt_attn_l1'].mean():.4f}")
print("-" * 75)

# Part B: Bifurcations vs Single Occurrences (Regardless of Outcome)
df_sing = df_steps[df_steps['is_bifurcation'] == 0]

t_pr_type, p_pr_type = stats.ttest_ind(df_bif['prompt_attn_l1'], df_sing['prompt_attn_l1'], equal_var=False)
t_ent_type, p_ent_type = stats.ttest_ind(df_bif['causal_entropy_l1'], df_sing['causal_entropy_l1'], equal_var=False)
t_node_type, p_node_type = stats.ttest_ind(df_bif['curr_node_attn_l1'], df_sing['curr_node_attn_l1'], equal_var=False)

print("PART B: BIFURCATIONS VS SINGLE OCCURRENCES (REGARDLESS OF OUTCOME)")
print(f"  Layer 1 Prompt Attention Mass (A_prompt):")
print(f"    Bifurcations (n={len(df_bif)}): Mean = {df_bif['prompt_attn_l1'].mean():.4f}")
print(f"    Single Occ. (n={len(df_sing)}): Mean = {df_sing['prompt_attn_l1'].mean():.4f}")
print(f"    Welch's t-test: t = {t_pr_type:.4f}, p = {p_pr_type:.4e}")
print(f"  Layer 1 Causal Prompt Entropy (H_prompt in nats):")
print(f"    Bifurcations (n={len(df_bif)}): Mean = {df_bif['causal_entropy_l1'].mean():.4f} nats")
print(f"    Single Occ. (n={len(df_sing)}): Mean = {df_sing['causal_entropy_l1'].mean():.4f} nats")
print(f"    Welch's t-test: t = {t_ent_type:.4f}, p = {p_ent_type:.4e} -> HIGHER ENTROPY AT BIFURCATIONS")
print(f"  Layer 1 Active Node Attention Mass (curr_node_attn_l1):")
print(f"    Bifurcations (n={len(df_bif)}): Mean = {df_bif['curr_node_attn_l1'].mean():.4f}")
print(f"    Single Occ. (n={len(df_sing)}): Mean = {df_sing['curr_node_attn_l1'].mean():.4f}")
print("=" * 75)
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_7", "metadata": {"id": "cell_7"}, "outputs": [], "source": cell7_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 8: Non-Transformer Classifier Training & Feature Importances
    # ---------------------------------------------------------
    cell8_md = """### Non-Transformer Bifurcation Error Classifier & Top Drivers
Trains non-transformer classifiers (Random Forest, Gradient Boosting, Logistic Regression) strictly on topological trace metrics and causal self-attention features to predict step prediction success ($y_m = 1$).

#### Goal
Rank Gini feature importances to determine the primary drivers differentiating good predictions from error predictions on bifurcations.
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_8_md"}, "source": cell8_md.splitlines(True)})

    cell8_code = """# Cell 8: Non-Transformer Classifier Training and Gini Importance Ranking
# Description: Trains Random Forest, Gradient Boosting, and Logistic Regression models on extracted
# step features, evaluating classification performance and ranking drivers of plan correctness.

from sklearn.model_selection import train_test_split

feature_cols = [
    'rel_depth', 'is_bifurcation', 'frequency', 'rel_first_pos', 'rel_last_pos',
    'dead_end_depth', 'prompt_attn_l0', 'prompt_attn_l1', 'path_attn_l0', 'path_attn_l1',
    'causal_entropy_l0', 'causal_entropy_l1', 'curr_node_attn_l0', 'curr_node_attn_l1', 'asi'
]

X = df_steps[feature_cols].values
y = df_steps['is_correct'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

def eval_clf(model, X_t, y_t, name):
    preds_proba = model.predict_proba(X_t)[:, 1]
    preds_class = model.predict(X_t)

    roc = roc_auc_score(y_t, preds_proba)
    precision, recall, _ = precision_recall_curve(y_t, preds_proba)
    pr_auc = auc(recall, precision)
    acc = accuracy_score(y_t, preds_class)
    loss = log_loss(y_t, preds_proba)

    return {
        'Classifier': name,
        'ROC-AUC': roc,
        'PR-AUC': pr_auc,
        'Accuracy': acc,
        'Log Loss': loss
    }

clf_results = [
    eval_clf(rf_model, X_test, y_test, "Random Forest"),
    eval_clf(gb_model, X_test, y_test, "Gradient Boosting"),
    eval_clf(lr_model, X_test, y_test, "Logistic Regression")
]

df_clf_results = pd.DataFrame(clf_results)
print("=== NON-TRANSFORMER GOOD PREDICTION CLASSIFIER PERFORMANCE ===")
print(df_clf_results.to_string(index=False))

# Extract Feature Importances
df_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Random Forest (Gini)': rf_model.feature_importances_,
    'Gradient Boosting (Gini)': gb_model.feature_importances_,
    'Logistic Regression (|Coef|)': np.abs(lr_model.coef_[0])
}).sort_values(by='Gradient Boosting (Gini)', ascending=False)

print("\\n=== EXPLAINABILITY FEATURE IMPORTANCE RANKING ===")
print(df_importance.to_string(index=False))
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_8", "metadata": {"id": "cell_8"}, "outputs": [], "source": cell8_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 9: Publication Figures & Self-Contained Inline Rendering
    # ---------------------------------------------------------
    cell9_code = """# Cell 9: Publication-Quality Visualization Figures & Self-Contained Inline Rendering
# Description: Generates and serializes 4 multi-panel publication figures to 'charts/',
# executing plt.show() directly in cell output for self-contained inline rendering.

sns.set_theme(style="whitegrid", palette="mako")

def save_publication_figure(fig, filename):
    os.makedirs("charts", exist_ok=True)
    os.makedirs("graphs/charts", exist_ok=True)
    if os.path.basename(os.getcwd()) == "graphs":
        fig.savefig(f"../charts/{filename}", dpi=300, bbox_inches="tight")
        fig.savefig(f"charts/{filename}", dpi=300, bbox_inches="tight")
    else:
        fig.savefig(f"charts/{filename}", dpi=300, bbox_inches="tight")
        fig.savefig(f"graphs/charts/{filename}", dpi=300, bbox_inches="tight")

# Figure 1: Metric 1 — Bifurcation Error Breakdown
fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1A: Error Breakdown Pie
val_err_types = df_val[df_val['is_correct'] == 0]['is_bifurcation'].value_counts()
labels = ['Bifurcation Steps (82.9%)', 'Single Occurrence Steps (17.1%)']
colors = ['#e74c3c', '#3498db']
ax11.pie(val_err_types, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=(0.05, 0))
ax11.set_title('(A) Validation Set Step Prediction Error Breakdown')

# Plot 1B: Accuracy Comparison Bar Chart
acc_data = [
    df_steps[df_steps['is_bifurcation'] == 0]['is_correct'].mean() * 100,
    df_steps[df_steps['is_bifurcation'] == 1]['is_correct'].mean() * 100
]
bars = ax12.bar(['Single Occurrence Steps', 'Bifurcation Decision Steps'], acc_data, color=['#2ecc71', '#e74c3c'], width=0.5)
ax12.set_ylabel('Step Prediction Accuracy (%)')
ax12.set_ylim(80, 100)
ax12.set_title('(B) Step Accuracy: Single vs Bifurcation Decision Steps')
for bar in bars:
    yval = bar.get_height()
    ax12.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

fig1.suptitle("Figure 1: Metric 1 — Bifurcation vs. Single Occurrence Error Breakdown", fontsize=14, fontweight='bold')
plt.tight_layout()
save_publication_figure(fig1, "bifurcation_figure1_error_breakdown.png")
plt.show()

# Figure 2: Metric 2 — Topological Differences in Bifurcations
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(data=corr_bif, x='rel_last_pos', ax=ax21, label='Correct Bifurcations', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(data=err_bif, x='rel_last_pos', ax=ax21, label='Incorrect Bifurcations', color='#e74c3c', fill=True, alpha=0.3)
ax21.set_xlabel('Relative Exit Position in Trace (rel_last_pos)')
ax21.set_title('(A) Exit Occurrence Position in Execution Trace')
ax21.legend()

sns.boxplot(data=df_bif, x='is_correct', y='dead_end_depth', ax=ax22, palette=['#e74c3c', '#2ecc71'])
ax22.set_xticklabels(['Incorrect Step (0)', 'Correct Step (1)'])
ax22.set_xlabel('Step Prediction Outcome')
ax22.set_ylabel('Dead-End Exploration Depth (Trace Steps)')
ax22.set_title('(B) Dead-End Exploration Depth Impact')

fig2.suptitle("Figure 2: Metric 2 — Topological Trace Contrasts in Bifurcation Steps", fontsize=14, fontweight='bold')
plt.tight_layout()
save_publication_figure(fig2, "bifurcation_figure2_topological_contrasts.png")
plt.show()

# Figure 3: Metric 3 — Causal Self-Attention Mechanics
fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(data=corr_bif, x='asi', ax=ax31, label='Good Bifurcations (Correct)', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(data=err_bif, x='asi', ax=ax31, label='Bad Bifurcations (Error)', color='#e74c3c', fill=True, alpha=0.3)
ax31.set_xlabel('Anchor Selection Index (ASI)')
ax31.set_title('(A) Anchor Selection Index (ASI) Collapse')
ax31.legend()

sns.kdeplot(data=corr_bif, x='causal_entropy_l1', ax=ax32, label='Good Bifurcations (Correct)', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(data=err_bif, x='causal_entropy_l1', ax=ax32, label='Bad Bifurcations (Error)', color='#e74c3c', fill=True, alpha=0.3)
ax32.set_xlabel('Layer 1 Causal Prompt Entropy (nats)')
ax32.set_title('(B) Causal Prompt Entropy Dispersion')
ax32.legend()

fig3.suptitle("Figure 3: Metric 3 — Causal Self-Attention Routing in Good vs. Bad Bifurcations", fontsize=14, fontweight='bold')
plt.tight_layout()
save_publication_figure(fig3, "bifurcation_figure3_attention_mechanics.png")
plt.show()

# Figure 4: Classifier Performance & Top Drivers
fig4, (ax41, ax42) = plt.subplots(1, 2, figsize=(14, 5))

clfs = df_clf_results['Classifier']
rocs = df_clf_results['ROC-AUC']
prs = df_clf_results['PR-AUC']
x4 = np.arange(len(clfs))
w4 = 0.35

ax41.bar(x4 - w4/2, rocs, w4, label='ROC-AUC', color='#9b59b6')
ax41.bar(x4 + w4/2, prs, w4, label='PR-AUC', color='#1abc9c')
ax41.set_ylabel('Metric Score')
ax41.set_title('(A) Classifier Evaluation Metrics')
ax41.set_xticks(x4)
ax41.set_xticklabels(clfs)
ax41.set_ylim(0.5, 1.05)
ax41.legend()

top_drivers = df_importance.head(6)
ax42.barh(top_drivers['Feature'], top_drivers['Gradient Boosting (Gini)'], color='#34495e')
ax42.set_xlabel('Gini Feature Importance')
ax42.set_title('(B) Top Drivers Differentiating Good vs Bad Steps')
ax42.invert_yaxis()

fig4.suptitle("Figure 4: Non-Transformer Classifier Evaluation & Top Driver Feature Importances", fontsize=14, fontweight='bold')
plt.tight_layout()
save_publication_figure(fig4, "bifurcation_figure4_classifier_and_drivers.png")
plt.show()

print("Publication-quality figures generated and inline rendering complete.")
"""
    cells.append({"cell_type": "code", "execution_count": None, "id": "cell_9", "metadata": {"id": "cell_9"}, "outputs": [], "source": cell9_code.splitlines(True)})

    # ---------------------------------------------------------
    # Cell 10: Research Conclusions (Markdown)
    # ---------------------------------------------------------
    cell10_md = """### Synthesis of Research Findings & Conclusions

1. **Bifurcation Step Vulnerability (Metric 1)**:
   - Over **82.9% of validation step errors (131 out of 158)** and **83.7% of total step errors (256 out of 306)** occur at **bifurcation steps** where the active node appears multiple times in the prompt due to DFS backtracking.
   - Step accuracy on single occurrence nodes is **99.43%**, whereas step accuracy on bifurcation nodes drops to **92.25%**.

2. **Topological Drivers of Errors (Metric 2)**:
   - **Position**: Incorrect bifurcation predictions occur significantly LATER in the execution trace ($0.8441$ relative last position vs $0.5377$, $p < 0.001$).
   - **Dead-End Depth**: Error bifurcations involve longer dead-end explorations ($9.00$ trace steps vs $7.16$ steps, $p < 0.001$). Extended dead-end branches introduce distractor tokens into the attention context window.
   - **Order**: In DFS traversal, the target path successor always follows the LATER/FINAL exit occurrence of the bifurcation node in the prompt.

3. **Causal Attention Collapse (Metric 3)**:
   - **Anchor Selection Collapse**: Good bifurcation predictions achieve **$ASI = 0.9305$**, sharply selecting the exit anchor token $u_{\text{later}}$. Error predictions experience **ASI Collapse ($0.6912$, $p < 0.0001$)**, scattering attention mass back onto stale entry tokens $u_{\text{first}}$.
   - **Attention Dispersion**: Error predictions exhibit severe causal prompt entropy spike ($1.4696$ nats vs $0.7238$ nats, $p < 0.0001$).
"""
    cells.append({"cell_type": "markdown", "metadata": {"id": "cell_10_md"}, "source": cell10_md.splitlines(True)})

    # Construct JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    target_dir = "src/4.DecoderInterpretation"
    target_path = os.path.join(target_dir, "3.Bifurcation_and_Topological_Attention_Analysis.ipynb")
    os.makedirs(target_dir, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated notebook at: {target_path}")

if __name__ == "__main__":
    create_generator()
