# PROMPT: Generate a complete Neural Computing Project from scratch

You are an expert machine learning engineer and academic writing assistant. You will produce a **complete, submission-ready** project. The project compares an MLP and an SVM on a credit card fraud detection dataset and produces all required deliverables.

---

## CONTEXT: WHAT THIS COURSEWORK IS

This is a Master's-level individual project. You must:
1. Implement, train, and critically compare two Neural Computing methods using Python and Jupyter notebooks.
2. Submit a code bundle, a 6-page research paper, a 2-page supplementary, and a 5–10 minute video.

**Allowed libraries:** NumPy, PyTorch, Scikit-Learn, Skorch. **NOT allowed:** TensorFlow, Keras, Matlab.

---

## MARKING SCHEME (out of 75)

Optimise every deliverable against these exact mark allocations:

| Category | Marks | What the marker looks for |
|---|---|---|
| Code organisation and comments | 5 | Clean, well-commented code. Every function has a docstring. |
| Sophistication of implementation | 10 | Not just calling sklearn defaults — show you understand what's under the hood. Proper CV, early stopping, ablation. |
| Method summary, pros/cons, hypothesis | 5 | Brief, accurate description of both methods with a testable hypothesis. |
| Training/evaluation methodology | 5 | Cross-validation, parameter selection, train/val/test split, leakage avoidance. |
| **Critical evaluation of results** | **20** | **The single biggest mark bucket.** Fair, accurate analysis. Discuss limitations. Do NOT overstate findings. Compare merits AND limitations. |
| References, lessons learned, future work | 5 | ~6 Harvard-style references. Honest "what went wrong" section. Concrete future directions. |
| Glossary and intermediate results | 5 | Define key terms. Show failed experiments, negative results, intermediate architectures. |
| Implementation details | 5 | Main implementation choices explained (why this architecture, why this loss, why this optimiser). |
| Video explanation | 10 | (Not generated here — but the code and paper should be structured to present well in a 5–10 min video.) |
| Overall clarity of presentation | 5 | Paper is well-structured, figures are clear, writing is concise. |

**Critical rule:** "You will NOT be marked on how good or state-of-the-art your results are. What matters is that you follow a sound methodology and present clearly the problem, your method and the results, with a critical and fair comparative evaluation."

---

## FIXED CHOICES (do NOT change these)

### Dataset: Credit Card Fraud Detection

- **Source:** Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015) 'Calibrating probability with undersampling for unbalanced classification', *2015 IEEE Symposium Series on Computational Intelligence (SSCI)*, Cape Town, South Africa, 7–10 December. IEEE, pp. 159–166.
- **Download URL:** https://zenodo.org/records/7395559 (also available at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Raw size:** 284,807 transactions, 492 frauds (0.172% positive class).
- **Features:** 30 numerical — V1–V28 are PCA-transformed components (original features anonymised for confidentiality), plus Time (seconds elapsed since first transaction) and Amount (transaction value). Target: Class (0 = legitimate, 1 = fraud).
- **No missing values.** All features are numeric. No categorical encoding needed.

**Use a stratified subsample:** Keep ALL 492 fraud cases + randomly sample 10,000 legitimate transactions → **10,492 rows, ~4.69% positive class.** This makes SVM training tractable (kernel SVM is O(n²)–O(n³) on full 284k rows) while preserving every minority-class example. Note this subsampling in the paper and discuss the impact on generalisability in the Discussion/limitations section.

**Why this dataset is a good choice:**
- Class imbalance forces use of PR-AUC over accuracy, enables SMOTE ablation, and produces interesting precision-recall trade-offs between the two models.
- All-numerical PCA features make preprocessing simple and eliminate confounding factors from encoding choices.
- The dataset is well-cited with extensive prior work to reference.
- At 10,492 rows both methods train in seconds, enabling proper cross-validation, multi-seed runs, and scaling experiments within a 10-minute CPU budget.

### Method pair: MLP vs SVM

**Method 1: Multilayer Perceptron (MLP)**
- Fully-connected feed-forward network trained with backpropagation (Rumelhart, Hinton and Williams, 1986)
- Implement in PyTorch
- Architecture: configurable hidden layers, ReLU activations, dropout regularisation, single logit output
- Loss: `BCEWithLogitsLoss` (numerically stable binary cross-entropy)
- Optimiser: Adam with weight decay
- Early stopping on validation PR-AUC (not validation loss — validation loss can look fine while the model ignores the minority class)
- Strengths: learns non-linear feature representations end-to-end, scales linearly with training size, flexible capacity
- Weaknesses: non-convex optimisation (sensitive to initialisation), many hyperparameters, no closed-form solution

**Method 2: Support Vector Machine (SVM)**
- Maximum-margin classifier (Cortes and Vapnik, 1995)
- Implement via scikit-learn's `SVC` with `probability=True` (Platt scaling for calibrated probabilities)
- Compare kernels: linear, RBF, polynomial
- Strengths: convex optimisation (global optimum guaranteed), strong theoretical grounding via structural risk minimisation, effective in high-dimensional spaces, robust with small datasets
- Weaknesses: O(n²)–O(n³) training complexity, kernel choice is a manual design decision, no end-to-end feature learning

**Why these two are a good pair:** Both are supervised discriminative classifiers solving the same task, so they are directly comparable. They differ fundamentally in loss geometry (non-convex cross-entropy vs convex quadratic program), representation (learned features vs fixed kernel mapping), and regularisation (early stopping + weight decay vs C parameter + structural risk framing). This gives rich material for critical evaluation — the biggest mark bucket.

### Hypothesis (state verbatim in the paper)

*On this dataset, a kernel SVM and an MLP will achieve comparable ranking performance (ROC-AUC), but will differ meaningfully in (a) the precision–recall trade-off at operating thresholds, (b) sensitivity to class imbalance handling (no reweighting vs class weights vs SMOTE), and (c) computational cost as a function of training set size.*

---

## YOUR TASK: PRODUCE ALL OF THE FOLLOWING

### A. Main training notebook (`neco_starter.ipynb`)

A single Jupyter notebook that runs end-to-end and produces all experimental results. Structure it with these **12 sections:**

1. **Environment & reproducibility**
   - Imports: numpy, pandas, matplotlib, seaborn, sklearn (train_test_split, StratifiedKFold, GridSearchCV, StandardScaler, SVC, Pipeline, all metrics), imblearn (SMOTE, Pipeline), torch (nn, optim, DataLoader, TensorDataset), time, joblib, json, random, pathlib
   - Set `SEED = 42` everywhere: random, numpy, torch, torch.cuda
   - `torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False`
   - Auto-detect device: `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
   - Create output directory: `OUT_DIR = Path('outputs'); OUT_DIR.mkdir(exist_ok=True)`
   - Consistent plotting style: `sns.set_style('whitegrid'); plt.rcParams['figure.dpi'] = 100`

2. **Data loading & EDA**
   - Load `creditcard_reduced.csv`
   - Print shape, column names, missing values count, class distribution, fraud rate
   - `.describe()` on Time, Amount, Class
   - **Figure: class balance** — two-panel: (left) bar chart of class counts, (right) Amount distribution by class on log scale → save as `fig01_class_balance.png`
   - Top 10 features by absolute correlation with Class → print

3. **Train / validation / test split**
   - Three-way stratified split: 60% train, 20% val, 20% test
   - Use `train_test_split` twice with `stratify=y, random_state=SEED`
   - Print sample counts and fraud rate for each split
   - Verify each split has proportional fraud cases

4. **Preprocessing**
   - `StandardScaler` fit on X_train ONLY → transform X_train, X_val, X_test
   - Explain in markdown: scaling matters for MLP (gradient magnitudes) and SVM (RBF kernel width is scale-dependent)
   - Sanity check: print first 3 feature means (~0) and stds (~1) after scaling

5. **MLP: implementation, training, hyperparameter search**
   - **5.1 Model class:** `MLP(nn.Module)` with configurable `hidden_sizes`, `dropout`, single logit output. Include `predict_proba` convenience method. Full docstrings.
   - **5.2 Training loop:** `train_mlp()` function with early stopping on validation PR-AUC (NOT loss). Returns history dict, best PR-AUC, epochs trained. Supports `pos_weight` for class weighting. Full docstring.
   - **5.3 Baseline:** Train one default config `(64, 32), dropout=0.2, lr=1e-3` as a sanity check. Print metrics. **Figure: learning curves** — training loss + validation PR-AUC/ROC-AUC → save as `fig02_mlp_learning_curves.png`
   - **5.4 Hyperparameter search with 5-fold stratified CV:** Write a `cv_score_mlp()` helper that runs k-fold CV with per-fold StandardScaler. Grid: hidden_sizes in {(32,), (64,), (64,32), (128,64), (128,64,32)}, dropout in {0.0, 0.2, 0.3}, lr in {1e-4, 1e-3, 3e-3}. Score by mean PR-AUC. Display sorted results table. **This matches the SVM's GridSearchCV methodology for a fair comparison.**
   - **5.5 Retrain best config** on X_train split, report validation PR-AUC.

6. **SVM: implementation, hyperparameter search**
   - Use scikit-learn `Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=SEED))])` — the pipeline's internal scaler is separate from the MLP scaler and handles its own per-fold fitting during GridSearchCV.
   - `GridSearchCV` with `StratifiedKFold(n_splits=5)`, scored by `average_precision`, `n_jobs=-1`, `refit=True`
   - **Grid:** kernels {linear, RBF, poly}, C in {0.1, 1, 10}, gamma in {scale, 0.01, 0.1} (for RBF/poly), degree in {2, 3} (for poly)
   - **Important: fit on X_trainval (the combined train+val set) to match the MLP's CV search scope**
   - Print best params, best CV PR-AUC
   - Display full CV results table (for supplementary material)
   - Refit best config on X_train only, report validation PR-AUC
   - **Note:** `probability=True` enables Platt scaling (Platt, 1999) — needed for PR curves. Slower but necessary.

7. **Class imbalance ablation**
   - Three strategies × two models = 6 cells:
     - **None:** standard BCE loss / `class_weight=None`
     - **Class weights:** `pos_weight` in BCEWithLogitsLoss (= n_neg/n_pos ≈ 20.34) / `class_weight='balanced'`
     - **SMOTE:** oversample training data only, via imblearn `SMOTE(random_state=SEED)`. Use imblearn `Pipeline` for SVM so SMOTE is applied inside CV folds correctly. **CRITICAL: SMOTE must NEVER touch validation or test data.**
   - Display results as a DataFrame table
   - **Figure: grouped bar chart** of validation PR-AUC by strategy × model → save as `fig03_imbalance_ablation.png`
   - Pick the best strategy per model for downstream evaluation

8. **Final evaluation on held-out test set (with threshold tuning)**
   - **8.1 Threshold tuning:** Write a `tune_threshold()` function that scans 200 thresholds in (0.01, 0.99) on VALIDATION data and returns the F1-maximising one. Apply to both models.
   - **Figure: F1 vs threshold** for both models, with vertical lines at tuned thresholds and at 0.5 → save as `fig04_threshold_tuning.png`
   - **8.2 Results table:** Report ROC-AUC, PR-AUC, F1, precision, recall for both models at BOTH 0.5 and tuned thresholds (4 rows total).
   - **Figure: ROC + PR curves** side by side on test set → save as `fig05_test_curves.png`

9. **Multi-seed robustness**
   - Write a `run_seed()` function that re-runs the entire pipeline (split → scale → train → tune threshold → test) for a given seed
   - Run 5 seeds: `[42, 123, 2024, 7, 31415]`
   - Print per-seed PR-AUC for both models
   - Aggregate: mean ± std for ROC-AUC, PR-AUC, F1, precision, recall
   - **Figure: per-seed PR-AUC dots with mean bars** → save as `fig06_multi_seed.png`

10. **Training-size scaling**
    - Train both models at fractions [0.1, 0.25, 0.5, 0.75, 1.0] of X_train
    - Use stratified subsampling to maintain class balance at each fraction
    - Record wall-clock training time and test PR-AUC for each
    - Display as a table
    - **Figure: two-panel** — (left) training time vs n_train, (right) test PR-AUC vs n_train → save as `fig07_training_size_scaling.png`

11. **Error analysis & disagreement**
    - Hard predictions at tuned thresholds
    - **Figure: side-by-side confusion matrices** at tuned thresholds → save as `fig08_confusion_matrices.png`
    - Agreement matrix: both correct, both wrong, only MLP correct, only SVM correct → print counts and percentages
    - **Figure: probability-space scatter** (SVM prob vs MLP prob, coloured by true class, with threshold lines) → save as `fig09_disagreement.png`

12. **Save best models**
    - `joblib.dump(scaler, OUT_DIR / 'scaler.joblib')`
    - `torch.save({'state_dict': ..., 'hidden_sizes': ..., 'dropout': ..., 'in_dim': ..., 'tuned_threshold': ...}, OUT_DIR / 'best_mlp.pt')`
    - `joblib.dump(final_svm, OUT_DIR / 'best_svm.joblib')` — the full Pipeline (scaler + SVC)
    - Write `config.json` with: seed, feature_cols, mlp_config, svm_config, expected test metrics
    - **Reload sanity check:** load everything back and verify test metrics match (this becomes the basis for test.ipynb)
    - Print list of saved files

**Code quality requirements:**
- Every function must have a docstring explaining what it does, its parameters, and what it returns
- Add inline comments explaining **WHY** (not WHAT) for non-obvious choices
- Use consistent variable naming: `X_train_s` for scaled, `X_trainval` for combined, etc.
- Print progress throughout so the marker can see what's happening
- All figures saved to `outputs/` with descriptive filenames
- Total runtime: **under 10 minutes on CPU**

### B. Test notebook (`test.ipynb`)

A separate notebook that the marker runs to verify results WITHOUT retraining:

1. **Imports and reproducibility** — same SEED=42, same device detection
2. **Load data and reconstruct test split** — load `creditcard_reduced.csv`, run the same two `train_test_split` calls with same `random_state=SEED` → get identical X_test, y_test
3. **MLP class definition** — duplicate from main notebook (needed to deserialise the state_dict)
4. **Load saved artefacts** — scaler, MLP checkpoint, SVM pipeline, config.json. Print loaded configs.
5. **Inference** — scale X_test with loaded scaler, run MLP forward pass + sigmoid, run SVM predict_proba
6. **Metrics table** — same `evaluate()` function, same 4-row table (MLP/SVM × default/tuned thresholds)
7. **Reproduce Figure 5 from the paper: ROC + PR curves** (required: at least 2 figures)
8. **Reproduce Figure 8 from the paper: confusion matrices at tuned thresholds**
9. **Pass/fail check** — compare reproduced metrics against config.json expected values with tolerance 1e-6. Print "Reproduction check: PASS ✓ (10/10 metrics match)"
10. **Summary paragraph** — no retraining happened, metrics match, see main notebook for full pipeline

Runtime: **under 30 seconds on CPU.**

### C. `readme.txt`

Must contain these blocks (the brief specifies this exact format):

```
requirements:
numpy==<version>
pandas==<version>
scikit-learn==<version>
imbalanced-learn==<version>
torch==<version>
matplotlib==<version>
seaborn==<version>
joblib==<version>
jupyter==1.0.0

setup_instructions:
1. Extract all the files from the zip file
2. Change directory (cd) to the extracted folder, and save any test data into this root folder
3. Create a new virtualenv and install packages from requirements:
   virtualenv Student_env
   cd Student_env
   source bin/activate
   cd -
   pip install -r requirements.txt
4. Copy and paste the following code to open the IPYNB:
   jupyter notebook test.ipynb
```

Also include:
- Author name and student ID placeholders
- File manifest describing every file in the zip
- Dataset source: https://zenodo.org/records/7395559 with the Dal Pozzolo et al. (2015) citation
- Headline results summary table (multi-seed mean ± std)
- Code attribution section stating all code is original work using standard library APIs


### D. `requirements.txt`

Pin exact versions. Add a comment block at the top:
```
# Install PyTorch matching your CUDA version first:
#   pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
#   pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
# For CPU-only: pip install torch
# Then install the rest: pip install -r requirements.txt
```

### E. 6-page research paper (as Markdown, ready to paste into Word)

Format: **single column, font Arial 11, max 6 pages** including all figures and references.

**Structure:**

| Section | ~Pages | Content |
|---|---|---|
| I. Introduction | 0.5 | Credit card fraud as imbalanced classification. Why accuracy is misleading. State hypothesis verbatim. Contribution: controlled MLP-vs-SVM comparison under matched protocol. |
| II. Data and Preprocessing | 0.5 | Dal Pozzolo et al. (2015) citation. PCA features V1–V28, Time, Amount. Stratified subsample (10,492 rows, 4.69%). Three-way stratified split. StandardScaler on train only. |
| III. Methods | 1.5 | **A:** MLP — architecture, BCEWithLogitsLoss, Adam, early stopping on val PR-AUC, 5-fold CV search (cite Rumelhart et al. 1986). **B:** SVM — maximum-margin, kernels, Platt scaling, 5-fold CV GridSearchCV (cite Cortes & Vapnik 1995, Platt 1999). **C:** Class imbalance — three strategies, SMOTE only on training (cite Chawla et al. 2002). PR-AUC as primary metric (cite Saito & Rehmsmeier 2015). **D:** Evaluation protocol — threshold tuning on val F1, multi-seed robustness, training-size scaling. |
| IV. Results | 2.0 | HP search winners (brief). Imbalance ablation table. Threshold tuning results. Single-seed test table (4 rows). **Figure 1: ROC + PR curves.** Multi-seed mean±std table. Error analysis. **Figure 2: Training-size scaling.** |
| V. Discussion | 1.0 | **INVEST HERE — 20 marks.** (1) Headline: comparable ranking, different PR trade-off. (2) Why linear SVM wins over RBF (PCA features already linearly structured). (3) Why SMOTE didn't help at 4.69% (would likely help at 0.17%). (4) SVM super-linear scaling vs MLP near-linear. (5) **Critical limitations** (≥5): subsampled imbalance not representative; 5 seeds is small n; threshold is another hyperparameter adding variance; Platt scaling compresses SVM probabilities making 0.5 default meaningless; no ensembling/cost-sensitive learning explored; single dataset limits generalisability. |
| VI. Conclusion & Future Work | 0.25 | Hypothesis largely supported. Practical choice depends on cost ratio. Future: native 0.17% imbalance; MLP-SVM ensemble; cost-sensitive objectives; calibration diagnostics. |
| References | 0.25 | ~6–7 Harvard-style references (see list below). |

**References to include (all verified to exist):**
1. Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357.
2. Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', *Machine Learning*, 20(3), pp. 273–297.
3. Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015) 'Calibrating probability with undersampling for unbalanced classification', *2015 IEEE SSCI*, pp. 159–166.
4. Esenogho, E., Mienye, I.D., Swart, T.G., Aruleba, K. and Obaido, G. (2022) 'A neural network ensemble with feature engineering for improved credit card fraud detection', *IEEE Access*, 10, pp. 16400–16407.
5. Platt, J.C. (1999) 'Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods', in *Advances in Large Margin Classifiers*. MIT Press, pp. 61–74.
6. Rumelhart, D.E., Hinton, G.E. and Williams, R.J. (1986) 'Learning representations by back-propagating errors', *Nature*, 323(6088), pp. 533–536.
7. Saito, T. and Rehmsmeier, M. (2015) 'The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets', *PLOS ONE*, 10(3), e0118432.

**Style rules:**
- Do NOT overstate findings — the marker penalises this explicitly
- Use "we" (standard academic first-person plural)
- Include figure captions: "Figure 1: ..." below each figure placeholder
- Include at least one results table
- Do NOT claim state-of-the-art
- Be honest about limitations — this earns marks, not loses them
- Do NOT fabricate any citations

### F. 2-page supplementary material

Contains:
- **Glossary** of all key terms: PR-AUC (average precision), ROC-AUC, stratified k-fold cross-validation, SMOTE, early stopping, class weighting, Platt scaling, BCEWithLogitsLoss, Adam optimiser, RBF kernel, StandardScaler, precision, recall, F1 score
- **Full MLP CV results table** (all 7 configurations with CV mean ± std PR-AUC and fold-level scores)
- **Full SVM grid search results table** (all 16 configurations ranked by mean test score)
- **Intermediate results:** negative findings (e.g., "SMOTE marginally hurt both models"), failed or suboptimal architectures tried, implementation issues encountered and how they were resolved
- **Additional figures** not in the main paper: imbalance ablation bar chart (fig03), F1-vs-threshold plot (fig04), disagreement scatter (fig09)

### G. HPC compatibility (OPTIONAL — include only if student runs on a university Linux cluster)

If the project needs to run on a university HPC cluster with GPU nodes via SLURM:

1. **Audit the code** for HPC-breaking issues:
   - Hardcoded `'cuda'` or `'cpu'` device → replace with `get_device()` helper
   - matplotlib without `Agg` backend → add headless guard: `if not os.environ.get('DISPLAY'): matplotlib.use('Agg')`
   - `n_jobs=-1` in scikit-learn → replace with SLURM-aware: `int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))`
   - Windows path separators → use `pathlib.Path` throughout
   - Interactive tqdm bars → add `disable=not sys.stdout.isatty()`

2. **Create SLURM job scripts** in `scripts/hpc/`:
   - `run_mlp.slurm`: 1 GPU + 4 CPUs + 16 GB RAM + 2 hours
   - `run_svm.slurm`: 16 CPUs + 16 GB RAM + 4 hours (CPU-parallel grid search)
   - Export `PYTHONUNBUFFERED=1` and `MPLBACKEND=Agg`
   - `mkdir -p outputs/logs outputs models` before the Python call
   - Include submit/monitor/cancel commands as comments

3. **Create `scripts/hpc/run_local.sh`** — plain bash fallback for interactive/non-SLURM use.

4. **Patch code minimally** — surgical fixes only, do NOT rewrite working training logic.

5. **Add "Submitting to HPC" section to readme.txt** with submit, monitor, cancel, and 4-line troubleshooting (CUDA mismatch, OOM, time-limit, wrong scheduler).

---

## DELIVERABLE CHECKLIST

When you are done, verify every item exists and is complete:

```
submission/
├── readme.txt                    # With requirements: and setup_instructions: blocks
├── requirements.txt              # Pinned versions with PyTorch CUDA comment block
├── neco_starter.ipynb            # Main training notebook (12 sections, fully commented)
├── neco_starter.html             # Executed HTML of the main notebook (with saved outputs)
├── test.ipynb                    # Test notebook (loads models, reproduces 2 figures, pass/fail check)
├── test.html                     # Executed HTML of test notebook
├── creditcard_reduced.csv        # Stratified subsample: 10,492 rows (492 fraud + 10,000 legit)
├── models/
│   ├── best_mlp.pt               # MLP state_dict + architecture + tuned threshold
│   ├── best_svm.joblib           # Full sklearn Pipeline (StandardScaler + linear SVC)
│   ├── scaler.joblib             # StandardScaler fitted on training split
│   └── config.json               # Hyperparams, thresholds, expected test metrics
├── outputs/
│   ├── fig01_class_balance.png
│   ├── fig02_mlp_learning_curves.png
│   ├── fig03_imbalance_ablation.png
│   ├── fig04_threshold_tuning.png
│   ├── fig05_test_curves.png
│   ├── fig06_multi_seed.png
│   ├── fig07_training_size_scaling.png
│   ├── fig08_confusion_matrices.png
│   └── fig09_disagreement.png
├── paper.md                      # 6-page paper as Markdown (ready to paste into Word)
├── supplementary.md              # 2-page supplementary as Markdown
└── scripts/hpc/                  # (OPTIONAL) SLURM scripts
    ├── run_mlp.slurm
    ├── run_svm.slurm
    └── run_local.sh
```

## QUALITY GATES — CHECK BEFORE FINISHING

1. [ ] `test.ipynb` loads saved models and reproduces test metrics WITHOUT retraining
2. [ ] `test.ipynb` reproduces at least 2 figures from the paper (ROC/PR curves + confusion matrices)
3. [ ] All code has docstrings and inline comments explaining WHY
4. [ ] The paper states the hypothesis verbatim in the Introduction
5. [ ] The paper includes at least 2 quantitative figures with captions
6. [ ] The Discussion section has at least 5 concrete, specific limitations
7. [ ] The paper does NOT overstate findings or claim state-of-the-art
8. [ ] References are Harvard style with exactly 6–7 verified citations (no fabricated refs)
9. [ ] The supplementary contains a glossary and full HP search tables
10. [ ] `readme.txt` has both `requirements:` and `setup_instructions:` blocks in the brief's format
11. [ ] All random seeds are pinned (`SEED = 42`) for reproducibility
12. [ ] SMOTE is NEVER applied to validation or test data
13. [ ] StandardScaler is fit ONLY on training data (separate instances for MLP and SVM pipelines)
14. [ ] Total notebook runtime is under 10 minutes on CPU
15. [ ] The 6-page paper fits in 6 pages (roughly 3000–3500 words + 2 figures + references)
16. [ ] The MLP HP search uses 5-fold stratified CV (matching the SVM's GridSearchCV methodology)
17. [ ] Threshold tuning is done on VALIDATION set, never on test set
18. [ ] The test notebook ends with a clear PASS/FAIL check
