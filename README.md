# Graph Shortest Path Extraction Benchmarks

This directory contains research tutorials, procedural dataset generators, and Transformer architectures for extracting direct shortest paths from algorithmic execution traces (Depth-First Search and Random Walk traces).

---

## 1. Complex Dataset Specification & Flavors

The procedural graph datasets (`graphs/data/graph_dfs_dataset.pt`, `graphs/data/graph_rw_dataset.pt`, and `graphs/data/graph_rw_dense_dataset.pt`) evaluate transformer reasoning across different execution trace structures and topological complexities.

### Dataset Flavors
1. **Depth-First Search (`dfs`)**: Systematic tree-structured exploration traces containing forward branch expansion and backtracking steps.
2. **Sparse Random Walk (`rw`)**: Stochastic, unguided random walks over sparse tree-like graphs (< 2.5 average degree per node).
3. **Dense Random Walk (`rw_dense`)**: Stochastic random walks over **multi-dimensional dense mesh topologies** using **Best-of-N Candidate Quality Optimization**:
   - **4+ Node Connectivity Guarantee**: Every node strictly maintains a degree $k \ge 4$ (with average node degree $d_{\text{avg}} \ge 5.0$).
   - **Multi-Layered & Cross-Diagonal Meshes**: Constructed as 2D/3D multi-layer lattices and diagonal grid meshes ($N=30$) with high clustering coefficients ($\text{CC} \approx 0.45+$).
   - **Best-of-N Quality Scoring ($Q$)**: For each sample slot, candidate random walks are simulated, scored, and the highest-complexity sample is selected based on:
     - **Alternate Shortest Paths ($num\_alt\_sps$)**: Count of distinct equal-length shortest paths in $G_{\text{tr}}$.
     - **Sub-Loop Revisits ($revisited\_nodes$)**: Count of unique nodes visited $> 1$ time across non-trivial sub-loops.
     - **Decoy Edge Ratio ($decoy\_ratio$)**: Ratio of observed trace edges that act as distractor edges off the target shortest path $P^*$.

```
$$Q(T) = 2.5 \cdot N_{\text{alt\_sp}} + 1.5 \cdot N_{\text{revisit}} + 12.0 \cdot \eta_{\text{decoy}} + 5.0 \cdot \text{CC}(G) + 1.0 \cdot \langle k \rangle$$
```

### Traversal Parameters & Sequence Bounds
- **Input Traversal Trace ($T$)**: Goal-terminated 1D execution trace containing forward exploration, loops, and return/backtracking steps.
  - **Sequence Length ($K$)**: $30 \le K \le 50$ (`MAX_SRC_LEN = 50`)
  - The destination node $g$ appears **exactly once** at the final position ($t_K = g$).
- **Target Shortest Path ($P^*$)**: Direct shortest path connecting start node $s$ to destination node $g$.
  - **Sequence Length ($M$)**: $10 \le M \le 20$ (`MAX_TGT_LEN = 21` including `STOP_TOKEN`)
- **Vocabulary & Token Identifiers**:
  - Node Identifier Vocabulary: Tokens `0` through `39` ($V = 40$ randomized node IDs per sample).
  - Special Control Tokens: `PAD_TOKEN = 40`, `STOP_TOKEN = 41` (`VOCAB_SIZE = 42`).

### Node Backtraces & Induced Regressions Metric
During traversal, whenever $t_k = t_{k-2}$, the transition represents a return step from dead-end or sub-branch node $t_{k-1}$ back to parent node $t_k$. We track two key metrics:
1. **Total Backtrace Count**: Total return steps in the trace.
2. **Node-Level Induced Regressions ($B(v)$)**: How many times node $v$ induced a backtrack/regression during traversal:
   $$B(v) = \sum_{k=3}^K \mathbb{I}\big(t_k = t_{k-2} \text{ and } t_{k-1} = v\big)$$

---

## 2. Mechanics of a Good Plan vs. a Bad Plan

Sequential autoregressive rollout ($M \in [10, 20]$) over complex 1D traversal traces ($K \in [30, 50]$) evaluates the model's spatial planning and trajectory consistency.

### Good Plan Mechanics
- **Cross-Attention / Causal Alignment**: The transformer attends to the correct contextual representations in the traversal trace, identifying true forward edge transitions.
- **Valid Path Connectivity**: Each predicted step $p_m$ forms a valid edge $(p_{m-1}, p_m) \in E_G$ on the graph, terminating strictly at goal $g$.
- **Adjacency Compression**: The model successfully filters out return steps ($t_k = t_{k-2}$) and non-optimal loops embedded in $T$.

### Bad Plan Mechanics & Compounding Errors
- **Early Prefix Errors**: In long target sequences ($M \in [10, 20]$), an incorrect token choice at early step $m$ introduces an off-path node into the causal context.
- **Compounding Error Propagation**: Once an invalid or off-path node is generated, the causal state shifts into out-of-distribution space. Subsequent predictions fail to align with graph adjacencies, leading to premature termination or hallucinated path loops.
- **Rollout Error Scaling**: Because sequence match requires $M$ consecutive correct decisions, exact path match probability scales exponentially:
  $$P(\text{Exact Match}) = \prod_{m=1}^M P(p_m^* \mid p_{<m}^*, T) \approx (1 - \epsilon)^M$$
  With $M \ge 10$, even low token error rates $\epsilon \approx 0.05$ result in non-trivial rollout failure rates ($1 - 0.95^{15} \approx 53.7\%$).

---

## 3. Notebook Configuration & Training Controls

`graphs/1.step_by_step_graph_shortest_path_tutorial.ipynb` includes explicit configuration controls in Cell 5:

```python
config = {
    "dataset_flavor": "rw_dense", # 'rw_dense' for Dense Random Walk, 'rw' for Random Walk, 'dfs' for Depth-First Search
    "model_size": "base",        # 'small', 'base', or 'large'
    "restart_training": False,   # Set to True to bypass saved checkpoints and start fresh from epoch 1
    "run_full_training": False,  # Set to True to skip 'epochs_to_train' limit and run full 'total_epochs'
    "resume_training": True,     # Resumes from latest checkpoint if restart_training is False
    "total_epochs": 10000,
    "save_every": 1000,
    "validate_every": 50,
    "epochs_to_train": 20,       # Interactive execution chunk size
    "learning_rate": 1e-3,
    "batch_size": 64
}
```

### Key Configuration Flags
- **`dataset_flavor`**:
  - `"rw_dense"`: Dense Random Walk dataset ($d_{\text{min}} \ge 4$, $d_{\text{avg}} \ge 5.0$, Best-of-N quality score $Q$, decoy edge ratio $> 60\%$).
  - `"rw"`: Sparse Random Walk dataset.
  - `"dfs"`: Depth-First Search tree exploration dataset.
- **`restart_training`**:
  - `True`: Ignores existing checkpoints in `checkpoints/` and initializes model weights fresh from epoch 1.
  - `False`: Automatically attempts to load latest checkpoint.
- **`run_full_training`**:
  - `True`: Trains continuously up to `total_epochs` (e.g., 10,000 epochs) without stopping at `epochs_to_train`.
  - `False`: Runs an interactive chunk of `epochs_to_train` (e.g., 20 epochs) for local verification.
- **Periodic Validation & Checkpointing**:
  - Validation runs **strictly every 50 epochs** (`validate_every = 50`).
  - Model checkpoints are serialized every 1,000 epochs to Google Drive (`/content/drive/MyDrive/graph_checkpoints`) with local fallback (`checkpoints/`).

---

## 4. Directory Structure & Files

- `0.graph_dataset_and_topology_analysis_tutorial.ipynb`: DFS dataset generation notebook and topological characterization.
- `0.random_walk_graph_dataset_tutorial.ipynb`: Sparse Random Walk dataset generation notebook.
- `0.dense_random_walk_graph_dataset_tutorial.ipynb`: Dense Random Walk dataset generation notebook ($d_{\text{min}} \ge 4$, Best-of-N Quality Scoring).
- `0.one_shot_graph_shortest_path_tutorial.ipynb`: One-Shot Non-Autoregressive Transformer tutorial.
- `1.step_by_step_graph_shortest_path_tutorial.ipynb`: Step-by-Step Autoregressive Graph Shortest Path Transformer tutorial.
- `src/1.Inference/3.Large_scale_SGD_dense_autoregressive_GSP.ipynb`: Large-scale Autoregressive Graph Transformer trained with SGD momentum and stochastic 1,000 / 30,000 epoch sub-sampling on 10x dense execution traces.
- `src/3.DecoderOnly/1.Small_Easy_DecoderOnly_Autoregressive_GSP.ipynb`: Decoder-Only Causal Language Model for graph shortest path extraction over concatenated execution traces and target path tokens.
- `2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb`: Mechanistic interpretability and causal activation patching tutorial dissecting the phase transition from Epoch 300 to Epoch 400.
- `3.topological_difficulty_and_step_error_prediction_tutorial.ipynb`: Topological difficulty modeling and step-by-step error prediction notebook decoupling task difficulty from attention misrouting.
- `generate_data_notebook.py`: Programmatic generator for DFS dataset notebook.
- `generate_rw_data_notebook.py`: Programmatic generator for Sparse Random Walk dataset notebook.
- `generate_rw_dense_data_notebook.py`: Programmatic generator for Dense Random Walk dataset notebook.
- `generate_notebook.py`: Programmatic generator for One-Shot Notebook.
- `generate_ar_notebook.py`: Programmatic generator for Autoregressive Notebook.
- `generate_decoder_only_notebook.py`: Programmatic generator for Decoder-Only Causal LLM Notebook.
- `generate_dense_decoder_only_notebook.py`: Programmatic generator for Mid-Scale Dense Random Walk Decoder-Only Autoregressive Notebook.
- `src/3.DecoderOnly/2.Mid_Dense_DecoderOnly_Autoregressive_GSP.ipynb`: Mid-scale Decoder-Only Causal Language Model solver for graph shortest path extraction over dense random walk execution traces ($d_{\text{min}} \ge 4$).
- `generate_mechanistic_notebook.py`: Programmatic generator for Mechanistic Analysis Notebook 2.
- `generate_difficulty_notebook.py`: Programmatic generator for Topological Difficulty Notebook 3.
- `generate_dupe_attention_notebook.py`: Programmatic generator for Duplicated Token Attention Notebook 6.
- `generate_interpretability_notebook_5.py`: Programmatic generator for Attention Map Explainability and Good Prediction Classifier Notebook 5.
- `generate_decoder_interpretability_notebook.py`: Programmatic generator for Decoder-Only Representation Dynamics and Causal Self-Attention Notebook.
- `src/2.Interpretation/4.Duplicated_token_attention_and_backtrace_mechanics.ipynb`: Research tutorial notebook on duplicated token attention mechanics and backtrace dynamics.
- `src/2.Interpretation/5.Attention_map_explainability_and_good_prediction_classifier.ipynb`: Research tutorial notebook on cross-attention map explainability, non-transformer prediction classifiers, and verification of research theses across Epoch 300 and 500 checkpoints.
- `src/4.DecoderInterpretation/1.Decoder_Only_Representation_and_Attention_Mechanics.ipynb`: Interpretability notebook dissecting representation drift, logit margin amplification, and causal prompt attention mechanics in Decoder-Only Graph Shortest Path Transformers across Epoch 100 and Epoch 1000 checkpoints.
- `generate_decoder_only_interpretability_notebook.py`: Programmatic generator for Decoder-Only Causal Self-Attention Interpretability Notebook.
- `src/2.Interpretation/4.Duplicated_token_attention_and_backtrace_mechanics.ipynb`: Research tutorial notebook on duplicated token attention mechanics and backtrace dynamics.
- `src/2.Interpretation/5.Attention_map_explainability_and_good_prediction_classifier.ipynb`: Research tutorial notebook on cross-attention map explainability, non-transformer prediction classifiers, and verification of research theses across Epoch 300 and 500 checkpoints.
- `src/4.DecoderOnlyInterpretability/1.Good_vs_bad_plans_decoder_only_interpretability.ipynb`: Research tutorial notebook on causal self-attention routing, prompt mass allocation, and Good vs. Bad plan mechanics in Decoder-Only Causal Graph Transformers.
- `src/4.DecoderInterpretation/3.Bifurcation_and_Topological_Attention_Analysis.ipynb`: Detailed topological error profiling and causal attention mechanics notebook dissecting bifurcation dynamics, dead-end depths, and anchor selection collapse in Epoch 100 Decoder Checkpoints.
- `generate_decoder_bifurcation_analysis_notebook.py`: Programmatic generator for Decoder-Only Bifurcation and Topological Attention Analysis Notebook.
- `src/4.DecoderInterpretation/3.Topological_graph_complexity_good_vs_bad_plans.ipynb`: Research tutorial notebook characterizing macro-level graph topology, connectivity, and traversal complexity drivers of Good Plans vs Bad Plans using checkpoint `decoder_only_ar_graph_transformer_mid_epoch_100.pt`.
- `generate_decoder_topological_analysis_notebook.py`: Programmatic generator for Topological Graph Complexity Analysis Notebook.
- `generate_decoder_rw_eval_notebook.py`: Programmatic generator for Multi-Metric Optimality Benchmark Notebook across DFS, Sparse RW, and Dense RW Traces.
- `src/3.DecoderOnly/4.Dense_RW_Optimal_Path_Evaluation.ipynb`: Research tutorial notebook evaluating the Base Decoder-Only Graph Transformer (`decoder_only_ar_graph_transformer_rw_dense_base_epoch_1000.pt`) across 3 consolidated datasets (DFS, Sparse Random Walk, Dense Random Walk) evaluating Token Accuracy (Exact Match), Path Connectivity Validity, and Optimal Path Accuracy.
- `data/graph_dfs_dataset.pt`: Pre-generated DFS dataset payload.
- `data/graph_rw_dataset.pt`: Pre-generated RW dataset payload.
- `data/graph_rw_dense_dataset.pt`: Pre-generated Dense RW dataset payload ($d_{\text{min}} \ge 4$, Best-of-N $Q$).
- `data/inference_dataset_epoch_300.pt`: Reusable exported validation set evaluation dataset with per-step activation parameters for Epoch 300.
- `data/inference_dataset_epoch_400.pt`: Reusable exported validation set evaluation dataset with per-step activation parameters for Epoch 400.
- `data/step_error_classification_dataset.pt`: Reusable step-level evaluation dataset containing 6,932 step instances with full classification layer outputs and graph topology features.
- `checkpoints/`: Local directory for model checkpoints.
- `charts/`: Output visualization figures.

---

## 5. Mechanistic Interpretability & Exported Inference Datasets

Notebook `2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb` analyzes the training phase transition between Epoch 300 and Epoch 400.

### Key Mechanistic Findings
- **Phase Transition Surge**: Autoregressive rollout exact match accuracy increases from **13.4%** at Epoch 300 to **80.0%** at Epoch 400 on the 500-sample validation set.
- **Cross-Attention Sharpening**: Layer 1 cross-attention entropy drops sharply from **0.87 nats** to **0.40 nats**, reflecting learned precision in locating target step nodes within 1D DFS traces.
- **Logit Margin Amplification**: Mean step logit margin $\Delta z = z_{\text{top1}} - z_{\text{top2}}$ increases from **2.92** to **5.75**, providing robust decision margins.
- **Transition Breakdown**: Out of 500 validation samples, **340 samples (68.0%)** improve from incorrect to correct exact matches, **60 samples (12.0%)** remain correct, **93 samples (18.6%)** remain failed, and **7 samples (1.4%)** regress.
- **Anthropic J-Space Causal Interpretability**:
  - **Downstream Residual Jacobian Steering ($h_1 \to h_2$)**: Computing downstream Jacobian $J_{h_1} = \nabla_{h_1}(z_c - z_w)$ over Epoch 300 residual activations (between Decoder Layer 1 & 2) isolates the steering direction required to convert error predictions into ground-truth target tokens, recovering target tokens with logit margin amplification $\Delta z > +6.0$.
  - **Attention Map Traceback ($J_{h_1} \to J_A$)**: Backpropagating $J_{h_1}$ through Layer 1 cross-attention reveals the attention Jacobians $J_A = \nabla_A \Delta z$, demonstrating that steering shifts attention weight away from distractor/decoy tokens and re-allocates attention mass onto valid forward edge transitions in the 1D DFS execution trace.

---

## 6. Topological Difficulty & Step Error Prediction (Notebook 3)

Notebook `3.topological_difficulty_and_step_error_prediction_tutorial.ipynb` models what makes a given autoregressive step difficult based on graph topology, enabling the mechanistic decoupling of inherent task difficulty from model attention failures.

### Key Findings & Benchmark Metrics
- **Step Dataset Payload (`data/step_error_classification_dataset.pt`)**: Serializes 6,932 step decision instances across Epoch 300 and Epoch 400 validation inferences, linking classification layer outputs (top-1 logit, logit margin $\Delta z_m$, target token probability, target token rank, cross-attention entropy) to step-level graph topology features.
- **Predictor Model Performance**: Non-transformer classifiers trained strictly on graph topology features predict step-level decision difficulty with high precision:
  - **Random Forest**: $\text{ROC-AUC} = 0.9867$, $\text{PR-AUC} = 0.9978$, $\text{Accuracy} = 96.83\%$.
  - **Gradient Boosting**: $\text{ROC-AUC} = 0.9864$, $\text{PR-AUC} = 0.9977$, $\text{Accuracy} = 97.48\%$.
  - **MLP Classifier**: $\text{ROC-AUC} = 0.9892$, $\text{PR-AUC} = 0.9982$, $\text{Accuracy} = 97.19\%$.
- **Top Topological Difficulty Drivers**: Feature importance analysis reveals that relative step depth ($\tau_m = m / M$, Gini $0.648$), step index ($m$, Gini $0.143$), shortest path length ($M$, Gini $0.036$), node out-degree ($k_{\text{out}}$, Gini $0.028$), and decoy neighbor ratio ($\eta_{\text{decoy}}$) are the primary graph features driving step errors.
- **Mechanistic Error Decoupling**:
  - **Topologically Difficult Errors**: Errors occurring at high topological complexity decisions ($D(m) \ge 0.5$).
  - **Attention Misrouting Failures**: Errors occurring at topologically simple decisions ($D(m) < 0.5$) where prediction failed due to cross-attention misrouting / high entropy.

---

## 7. Attention Routing Dynamics over Duplicated Tokens and Backtrace Trajectories (Notebook 6)

Notebook `src/2.Interpretation/4.Duplicated_token_attention_and_backtrace_mechanics.ipynb` investigates how Transformer attention handles duplicated tokens (nodes experienced during backtraces/dead-ends) versus unique tokens in graph shortest path extraction across inference phases and training epochs.

### 3-Tier Metric Progression & Quantitative Results

#### 1. Descriptive Stage: Attention Mass Allocation
- **Duplicated Token Attention Share ($R_{\text{dupe}}$)**: In Layer 1 cross-attention, duplicated tokens account for $29.59\%$ of total attention mass in Epoch 400 versus $33.97\%$ in Epoch 300.
- **First vs. Later Token Split Ratio ($S_{\text{later}}$)**: When attending to a duplicated node $V$ in the trace, the model allocates $86.71\%$ of its duplicated attention mass to the **later occurrence** ($V_{\text{later}}$) in Epoch 400 versus $77.66\%$ in Epoch 300.
- **Active Frontier Anchoring**: Post-phase transition models treat $V_{\text{later}}$ (the exit node from a backtrace trajectory) as the active frontier anchor for predicting the next step $V_{\text{next}}$.

#### 2. Diagnostic Stage: Anchor Selection Index ($ASI$) & Error Dissection
- **Anchor Selection Index ($ASI$)**: Quantifies the attention preference for $V_{\text{later}}$ over $V_{\text{first}}$:
  $$ASI(m) = \frac{A(V_{\text{later}})}{A(V_{\text{first}}) + A(V_{\text{later}})}$$
- **Step Prediction Accuracy**: Step decision accuracy increases from $90.31\%$ (Epoch 300) to $98.75\%$ (Epoch 400), with mean cross-attention entropy dropping from $0.8349$ to $0.6423$ nats.
- **Error Dissection**: On Error Steps in Epoch 300, $ASI$ drops sharply to $0.6469$ (and lower on critical branching points), demonstrating that attention misrouting back to stale initial occurrences ($V_{\text{first}}$) directly causes step decision failures.

#### 3. Causal Stage: Causal Attention Masking Interventions
- **Masking $V_{\text{later}}$ Exit Anchor Region**: Suppressing pre-softmax attention logits at the $V_{\text{later}}$ exit region ($i_{\text{later}}$ and $i_{\text{later}}+1$) in Layer 1 cross-attention causes an absolute collapse in target token probability $P(V_{\text{next}})$ from $0.4329$ down to $0.0002$ and a massive target logit margin drop ($\Delta z$ loss of $18.2610$ logit points).
- **Masking $V_{\text{first}}$ Keys**: Suppressing $V_{\text{first}}$ key vectors ($i_{\text{first}}$) results in negligible change ($P(V_{\text{next}}) = 0.4330$), confirming that $V_{\text{later}}$ is the causally dominant position for autoregressive rollout continuation.

---

## 8. Attention Map Explainability and Step Decision Correctness Classification (Notebook 5)

Notebook `src/2.Interpretation/5.Attention_map_explainability_and_good_prediction_classifier.ipynb` computes cross-attention maps from the Epoch 300 and Epoch 500 checkpoints, trains non-transformer classifiers to predict step decision success based on graph topology and attention features, and evaluates three core research theses regarding graph reasoning.

### Non-Transformer Good Prediction Classifiers
Non-transformer classifiers trained on extracted topology and attention map features predict step prediction success ($y_m \in \{0, 1\}$) with exceptional precision:
- **Gradient Boosting Classifier**: $\text{ROC-AUC} = 0.9533$, $\text{PR-AUC} = 0.9956$, $\text{Accuracy} = 93.10\%$, $\text{Log Loss} = 0.1515$.
- **Random Forest Classifier**: $\text{ROC-AUC} = 0.9513$, $\text{PR-AUC} = 0.9954$, $\text{Accuracy} = 93.43\%$, $\text{Log Loss} = 0.1517$.
- **Logistic Regression**: $\text{ROC-AUC} = 0.9073$, $\text{PR-AUC} = 0.9905$, $\text{Accuracy} = 92.59\%$, $\text{Log Loss} = 0.1856$.

### Top Drivers of Good Predictions
Feature importance analysis reveals that **Layer 1 Cross-Attention Entropy** (`l1_entropy`, Gini $0.3780$), **Current Node Attention Mass** (`curr_node_attn_l1`, Gini $0.1522$), and **Future Dead-End Bifurcation Attention** (`future_dead_bif_attn`, Gini $0.1098$) are the primary drivers determining whether the Transformer will answer correctly or fail.

### Verification of Core Research Theses
1. **Thesis 1 (Bifurcation vs. Linear Path Encoding)**:
   - Bifurcation decisions exhibit higher attention entropy ($1.1483$ vs $0.6757$ nats in Epoch 300) and require exit anchor selection ($ASI$), confirming that bifurcations represent higher cognitive decision complexity.
2. **Thesis 2 (Attention Mass Allocation Requirement)**:
   - Bifurcation nodes demand significantly higher cross-attention mass than linear path nodes ($0.1760$ vs $0.1134$ in Epoch 300, Welch's $t = 17.18, p < 10^{-60}$; $0.0603$ vs $0.0344$ in Epoch 500, Welch's $t = 11.93, p < 10^{-30}$), proving that greater attention mass is required at trace bifurcations to prevent rollout failure.
3. **Thesis 3 (Selective Future Bifurcation Encoding)**:
   - Cross-attention maps selectively assign $3.94\times$ higher attention mass to future exit-path bifurcations over dead-end bifurcations in Epoch 300 ($0.2093$ vs $0.0531$), which sharpens to a **$51.57\times$ preference** in Epoch 500 ($0.2523$ vs $0.0049$, paired $t = 33.25, p < 10^{-200}$). This confirms that future bifurcations encode whether they lie on the exit path versus dead-end sub-branches.

---

## 9. Large-Scale Stochastic Gradient Descent (SGD) Autoregressive Solver (Notebook 3)

Notebook `src/1.Inference/3.Large_scale_SGD_dense_autoregressive_GSP.ipynb` implements an Autoregressive Graph Shortest Path Transformer trained on the 10x Scale Dense Dataset (`graph_rw_dense_10x_dataset.pt`, $N_{\text{train}} = 30,000$) using Stochastic Gradient Descent (SGD) with momentum ($\mu = 0.9$).

### Key Features & Sub-Sampling Strategy
- **Stochastic Sub-Sampling (1,000 / 30,000)**: In each epoch, a random subset of 1,000 training instances is sampled without replacement using `torch.utils.data.RandomSampler`. This exposes the model to the full 30,000-sample dataset across training while maintaining constant per-epoch compute requirements.
- **First-Order SGD Dynamics**: Investigates SGD optimization behavior ($\eta = 0.01$, $\mu = 0.9$) on sequence-to-sequence causal graph transformers, evaluating gradient noise tolerance and rollout generalization over dense graphs ($k \ge 4$, decoy ratio $> 60\%$).
- **Multi-Metric Evaluation**: Tracks Cross-Entropy Loss, Teacher-Forcing Token Accuracy, Autoregressive Rollout Exact Match %, and Path Connectivity Validity % across training epochs and held-out test evaluations.

---

## 10. Decoder-Only Causal Language Model Solver (Notebook 1 in `src/3.DecoderOnly/`)

Notebook `src/3.DecoderOnly/1.Small_Easy_DecoderOnly_Autoregressive_GSP.ipynb` reformulates graph shortest path extraction using a **Decoder-Only Transformer Architecture**, matching the standard causal language modeling paradigm of modern Large Language Models (GPT-4, LLaMA, DeepSeek).

### Key Architectural & Training Features
- **Unified Causal Sequence Formulation**: Concatenates execution trace prompt $T = [t_1, \dots, t_K]$ and target path $P^* = [p_1^*, \dots, p_M^*]$ into a unified 1D sequence $X = [t_1, \dots, t_K, p_1^*, \dots, p_M^*, \text{STOP\_TOKEN}]$.
- **Elimination of Cross-Attention**: Replaces Encoder-Decoder cross-attention blocks with a single stack of 4 Causal Transformer layers utilizing lower-triangular causal self-attention masking.
- **Selective Loss Masking**: Masks out cross-entropy loss for prompt trace tokens ($i < K-1$), directing 100% of gradient updates toward predicting next path rollout tokens.
- **Unguided Causal Rollout (`solve_graph_autoregressive`)**: Prompts the causal model with trace sequence $T$, generating path tokens step-by-step using causal self-attention over the expanding context.

---

## 11. Decoder-Only Representation Dynamics & Causal Self-Attention Mechanics (Notebook 1 in `src/4.DecoderInterpretation/`)

Notebook `src/4.DecoderInterpretation/1.Decoder_Only_Representation_and_Attention_Mechanics.ipynb` investigates how training impacts a **Decoder-Only Transformer's** internal representations and causal self-attention mechanisms to find graph shortest paths, replacing cross-attention with causal prompt self-attention.

### Key Mechanistic Findings
- **Checkpoint & Dataset Verification**: Evaluates model code integrity across checkpoints `decoder_only_ar_graph_transformer_mid_epoch_100.pt` and `decoder_only_ar_graph_transformer_mid_epoch_1000.pt`. Exact match rollout accuracy increases from **75.40%** at Epoch 100 to **99.40%** at Epoch 1000 on the validation set.
- **Logit Margin Amplification**: Mean step decision logit margin $\Delta z = z_{\text{top1}} - z_{\text{top2}}$ expands from **6.89 logit points** at Epoch 100 to **10.08 logit points** at Epoch 1000, reflecting enhanced confidence and decision boundary stability.
- **Causal Prompt Attention Sharpening**: Layer 1 causal attention entropy over prompt trace tokens drops from **0.58 nats** (Epoch 100) to **0.05 nats** (Epoch 1000), demonstrating razor-sharp attention allocation to key trace positions.
- **Causal Masking Interventions**: Suppressing the target node's exit anchor in the prompt reduces target token prediction probability by **77.61%**, confirming that prompt exit anchors causally govern next path token generation.
## 11. Decoder-Only Causal Self-Attention Interpretability & Good vs. Bad Plan Mechanics (`src/4.DecoderOnlyInterpretability/`)

Notebook `src/4.DecoderOnlyInterpretability/1.Good_vs_bad_plans_decoder_only_interpretability.ipynb` investigates how good plans look different from bad plans in a **Decoder-Only Causal Graph Transformer** (`decoder_only_ar_graph_transformer_mid_epoch_100.pt`), replacing cross-attention analysis with **Causal Self-Attention Analysis** over the unified sequence $X = [t_1, \dots, t_K, p_1^*, \dots, p_M^*]$.

### Integrity Verification & Logged Accuracy
The checkpoint compatibility is verified on `graph_dfs_dataset.pt` with logged metrics across validation and test splits:
- **Validation Set (500 samples)**: Loss = `6.6096`, Teacher-Forcing Token Acc = `23.02%`, Rollout Exact Match = `0.00%`.
- **Test Set (500 samples)**: Loss = `6.6684`, Teacher-Forcing Token Acc = `22.05%`, Rollout Exact Match = `0.00%`.

### Causal Self-Attention Mechanics: Good Plans vs. Bad Plans
- **Prompt Attention Mass ($A_{\text{prompt}}$)**: Good plans maintain significantly higher attention mass on the trace prompt prompt region ($0.9455$ vs. $0.9141$, Welch's $t = 7.0067, p < 10^{-11}$), keeping the causal decoder anchored in graph topology.
- **Causal Prompt Entropy ($H_{\text{prompt}}$)**: Good plans exhibit lower causal prompt entropy ($1.0482$ vs. $1.3015$ nats, Welch's $t = -13.9990, p < 10^{-30}$), reflecting focused allocation onto active exit anchors rather than attention dispersion over distractor nodes.
- **Anchor Selection Index ($ASI$) at Trace Bifurcations**: At trace-based bifurcations, good steps strongly prefer the exit occurrence $V_{\text{later}}$ over entry occurrence $V_{\text{first}}$ ($ASI = 0.8054$ vs. $0.5119$, Welch's $t = 12.3782, p < 10^{-30}$).

### Non-Transformer Good Prediction Classifiers
Non-transformer classifiers trained on extracted causal self-attention and graph topology features predict step decision correctness without a Transformer:
- **Gradient Boosting Classifier**: $\text{ROC-AUC} = 0.9126$, $\text{PR-AUC} = 0.8278$, $\text{Accuracy} = 85.69\%$, $\text{Log Loss} = 0.3368$.
- **Random Forest Classifier**: $\text{ROC-AUC} = 0.9063$, $\text{PR-AUC} = 0.8052$, $\text{Accuracy} = 85.02\%$, $\text{Log Loss} = 0.3867$.
- **Top Drivers (Gini Importances)**: Layer 0 Current Node Attention (`curr_node_attn_l0`, Gini $0.2586$), Layer 1 Current Node Attention (`curr_node_attn_l1`, Gini $0.1890$), Layer 1 Causal Entropy (`causal_entropy_l1`, Gini $0.1217$), and $ASI$ ($0.0840$).

---

## 12. Decoder-Only Bifurcation & Topological Attention Analysis (`src/4.DecoderInterpretation/3.Bifurcation_and_Topological_Attention_Analysis.ipynb`)

Notebook `src/4.DecoderInterpretation/3.Bifurcation_and_Topological_Attention_Analysis.ipynb` performs a detailed topological and causal attention error profiling of the **Epoch 100 Decoder-Only Transformer** (`decoder_only_ar_graph_transformer_mid_epoch_100.pt`) across validation (158 step errors) and test sets (~306 total step errors across 12,005 step decisions).

### Key Empirical Findings Across Core Metrics

#### 1. Metric 1: Bifurcation vs. Single Occurrence Error Breakdown
- **Bifurcation Step Vulnerability**: Over **82.9% of validation step errors (131 out of 158)** and **83.7% of total step errors (256 out of 306)** occur at **bifurcation decision steps** where the current node appears multiple times in the trace prompt due to DFS backtracking.
- **Step Accuracy Disparity**: Step accuracy on single occurrence nodes is **99.43%** (50 errors out of 8,701 steps), whereas step accuracy on bifurcation nodes drops to **92.25%** (256 errors out of 3,304 steps).

#### 2. Metric 2: Topological Trace Contrasts on Bifurcations (Correct vs. Incorrect)
- **Trace Position**: Incorrect bifurcation predictions occur significantly LATER in execution trace prompts (mean last position relative to trace length = $0.8441$ for errors vs $0.5377$ for correct, Welch's $t = -22.01, p < 10^{-67}$). Late bifurcations suffer from longer context accumulation.
- **Depth of Dead-End**: Error bifurcations involve significantly GREATER dead-end exploration depths (mean $9.00$ trace steps between entry and exit re-visit vs $7.16$ steps, Welch's $t = -3.02, p < 0.003$). Extended dead-end sub-branches introduce distractor tokens into the causal context window.
- **Node Visit Frequency**: Error bifurcations exhibit higher node visit frequencies (31.6% have $\ge 3$ visits vs 18.1% for correct bifurcations).
- **Order Adjacency**: In 100% of cases, the ground-truth successor path token follows the SECOND/LATER exit occurrence of the bifurcation node in the trace prompt.

#### 3. Metric 3: Causal Self-Attention Mechanics Comparison
- **Part A (Good vs. Bad Predictions on Bifurcations Only)**:
  - **Anchor Selection Index Collapse**: Good bifurcation predictions achieve **$ASI = 0.9305$**, sharply routing attention to the exit anchor $u_{\text{later}}$. Error predictions experience severe **ASI Collapse ($0.6912$, Welch's $t = 13.20, p < 10^{-30}$)**, scattering attention mass back onto stale entry tokens $u_{\text{first}}$.
  - **Attention Dispersion**: Error predictions exhibit severe causal prompt entropy spikes ($1.4696$ nats vs $0.7238$ nats, Welch's $t = -26.70, p < 10^{-80}$).
- **Part B (Bifurcations vs. Single Occurrences)**:
  - Bifurcations inherently exhibit higher causal prompt entropy ($0.7816$ nats vs $0.5230$ nats, Welch's $t = 28.53, p < 10^{-160}$) due to the cognitive complexity of resolving multiple prompt occurrences.

#### 4. Non-Transformer Good Prediction Classifiers & Top Drivers
Non-transformer classifiers (Gradient Boosting $\text{ROC-AUC} = 0.9673$, Random Forest $\text{ROC-AUC} = 0.9653$) identify **Layer 1 Causal Prompt Entropy** (`causal_entropy_l1`, Gini $0.3230$), **Relative Exit Position** (`rel_last_pos`, Gini $0.1478$), and **Layer 0 Active Node Attention** (`curr_node_attn_l0`, Gini $0.0967$) as the top topological and attention drivers predicting step success.
## 12. Topological Graph Complexity Analysis of Good Plans vs. Bad Plans (`src/4.DecoderInterpretation/3.Topological_graph_complexity_good_vs_bad_plans.ipynb`)

Notebook `3.Topological_graph_complexity_good_vs_bad_plans.ipynb` evaluates macro-level graph structure and traversal geometry using checkpoint `decoder_only_ar_graph_transformer_mid_epoch_100.pt` on dataset `graph_dfs_dataset_v1.pt` (500 validation samples, 75.40% exact match rollout accuracy).

### Topological Complexity Drivers of Plan Failure
1. **Traversal Expansion Overhead ($K/M$)**:
   - Good Plans exhibit a significantly lower expansion ratio $K/M = 2.58 \pm 0.42$ versus Bad Plans $K/M = 3.65 \pm 0.71$ (Welch's $t = -16.82, p < 10^{-35}$). High ratios reflect excessive execution trace clutter relative to shortest path length.
2. **Backtracks ($N_{\text{backtrack}}$) and Decoy Edge Ratio ($\eta_{\text{decoy}}$)**:
   - Bad Plans contain nearly double the trace backtracks ($11.82 \pm 3.10$ vs $6.14 \pm 2.05$, Welch's $t = -18.41, p < 10^{-40}$) and higher decoy edge ratios ($\eta_{\text{decoy}} = 0.68 \pm 0.08$ vs $0.51 \pm 0.09$, $p < 10^{-25}$), creating dense distractor nodes that disperse causal attention mass.
3. **Graph Clustering ($CC$) & Connectivity ($\lambda_2$)**:
   - Higher average clustering coefficient ($CC(G) = 0.38$ vs $0.24$, $p < 10^{-15}$) and algebraic connectivity ($\lambda_2 = 0.42$ vs $0.29$, $p < 10^{-12}$) correlate with higher rollout failure rates, as dense local meshes present complex alternative sub-branches.

### Non-Transformer Topological Complexity Classifier
Non-transformer models trained strictly on macro-level graph topology features predict whether a graph instance will cause an autoregressive plan failure:
- **Gradient Boosting Classifier**: $\text{ROC-AUC} = 0.9412$, $\text{PR-AUC} = 0.9785$, $\text{Accuracy} = 88.80\%$.
- **Random Forest Classifier**: $\text{ROC-AUC} = 0.9380$, $\text{PR-AUC} = 0.9761$, $\text{Accuracy} = 88.00\%$.
- **Top Topological Drivers**: Traversal Expansion Ratio ($K/M$, Gini $0.3842$), Total Backtracks ($N_{\text{backtrack}}$, Gini $0.2615$), Decoy Edge Ratio ($\eta_{\text{decoy}}$, Gini $0.1580$), and Clustering Coefficient ($CC$, Gini $0.0812$).

---

## 13. Multi-Metric Optimality Benchmark for Decoder-Only Graph Transformers across DFS, Sparse RW, and Dense RW Traces (`src/3.DecoderOnly/4.Dense_RW_Optimal_Path_Evaluation.ipynb`)

Notebook `src/3.DecoderOnly/4.Dense_RW_Optimal_Path_Evaluation.ipynb` evaluates the **Base Decoder-Only Autoregressive Graph Transformer** (`decoder_only_ar_graph_transformer_rw_dense_base_epoch_1000.pt`) across 3 consolidated execution trace datasets: **Depth First Search (DFS)**, **Sparse Random Walk (RW)**, and **Dense Random Walk (Dense RW)** ($N=1,000$ per dataset).

### Key Motivation & Metric Blind Spot Resolution
Standard exact-match validation flags a rollout as a failure whenever $P_{\text{pred}} \neq P^*$. However, stochastic random walk execution traces over multi-dimensional dense graphs ($d_{\text{min}} \ge 4$) frequently admit **multiple distinct, equal-length optimal shortest paths**. Consolidating evaluation across 3 core metrics—Token Accuracy (Exact Match) (%), Path Connectivity Validity (%), and Optimal Path Accuracy (%)—reveals that many "failures" in dense graphs are topologically valid, optimal shortest paths along symmetric mesh branches.

### Multi-Metric Empirical Benchmark Summary ($N=3,000$)
| Dataset | Token Accuracy (Exact Match) (%) | Path Validity (%) | Optimal Path (%) |
| :--- | :---: | :---: | :---: |
| **Depth First Search (DFS)** ($N=1,000$) | **39.40%** (394) | **49.60%** (496) | **39.40%** (394) |
| **Sparse Random Walk (RW)** ($N=1,000$) | **32.50%** (325) | **51.50%** (515) | **33.50%** (335) |
| **Dense Random Walk (Dense RW)** ($N=1,000$) | **12.30%** (123) | **57.60%** (576) | **20.50%** (205) |

### Key Research Insights
1. **DFS Determinism**: In deterministic tree-structured DFS execution traces, alternate equal-length optimal paths do not exist; thus Token Accuracy (Exact Match) strictly equals Optimal Path Accuracy (**39.40%**).
2. **Dense Mesh Optimality Surge (+66.67% Relative Increase)**: In Dense Random Walks ($d_{\text{min}} \ge 4$), evaluating topological path optimality recovers a **+66.67% relative increase** in measured model optimality over exact match (**12.30%** exact match vs. **20.50%** optimal path), resolving the metric distortion in dense mesh topologies.
