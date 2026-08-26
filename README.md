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
- **Cross-Attention Alignment**: The decoder attends to the correct contextual representations in the encoded memory $H_{src}$, identifying true forward edge transitions.
- **Valid Path Connectivity**: Each predicted step $p_m$ forms a valid edge $(p_{m-1}, p_m) \in E_G$ on the graph, terminating strictly at goal $g$.
- **Adjacency Compression**: The model successfully filters out return steps ($t_k = t_{k-2}$) and non-optimal loops embedded in $T$.

### Bad Plan Mechanics & Compounding Errors
- **Early Prefix Errors**: In long target sequences ($M \in [10, 20]$), an incorrect token choice at early step $m$ introduces an off-path node into the causal decoder context.
- **Compounding Error Propagation**: Once an invalid or off-path node is generated, the causal decoder state shifts into out-of-distribution space. Subsequent predictions fail to align with graph adjacencies, leading to premature termination or hallucinated path loops.
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
- `2.mechanistic_interpretability_and_causal_analysis_tutorial.ipynb`: Mechanistic interpretability and causal activation patching tutorial dissecting the phase transition from Epoch 300 to Epoch 400.
- `3.topological_difficulty_and_step_error_prediction_tutorial.ipynb`: Topological difficulty modeling and step-by-step error prediction notebook decoupling task difficulty from attention misrouting.
- `generate_data_notebook.py`: Programmatic generator for DFS dataset notebook.
- `generate_rw_data_notebook.py`: Programmatic generator for Sparse Random Walk dataset notebook.
- `generate_rw_dense_data_notebook.py`: Programmatic generator for Dense Random Walk dataset notebook.
- `generate_notebook.py`: Programmatic generator for One-Shot Notebook.
- `generate_ar_notebook.py`: Programmatic generator for Autoregressive Notebook.
- `generate_mechanistic_notebook.py`: Programmatic generator for Mechanistic Analysis Notebook 2.
- `generate_difficulty_notebook.py`: Programmatic generator for Topological Difficulty Notebook 3.
- `generate_dupe_attention_notebook.py`: Programmatic generator for Duplicated Token Attention Notebook 6.
- `src/2.Interpretation/4.Duplicated_token_attention_and_backtrace_mechanics.ipynb`: Research tutorial notebook on duplicated token attention mechanics and backtrace dynamics.
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
