# HPC-Ready NECO Project: Prompt for Code Generation

Use this prompt with Claude (or any capable LLM) to transform the NECO coursework project into an HPC-compatible codebase that can be cloned and run on a university SLURM cluster with GPU nodes.

---

## THE PROMPT

```
Transform the code of this neural computing / machine learning project so it runs correctly on a university Linux HPC cluster with GPU nodes, while preserving the ability to run it locally (Windows, macOS, or Linux). Do not rewrite working code — add HPC-compatible infrastructure around it, and patch only what is genuinely broken for HPC use.

Assume the user already knows how to:
- SSH into the cluster.
- Clone the repo and create a Python virtualenv.
- Install `requirements.txt` and load any required system modules.
- Transfer data files to the cluster.

Your job is to produce the **code changes** and **job submission scripts** needed, not environment bootstrapping.

---

## Project Context

This is an INM427 Neural Computing (NECO) individual coursework at City, University of London. It compares an MLP (PyTorch) vs an SVM (scikit-learn) on a credit card fraud detection dataset.

### Current project structure:

```
neco_project/
├── creditcard_reduced.csv          # 10,492 rows, 31 columns, 5.3 MB
├── neco_starter.ipynb              # Main training notebook (55 cells):
│                                   #   - EDA, stratified 60/20/20 split
│                                   #   - MLP: PyTorch, BCEWithLogitsLoss, Adam,
│                                   #     early stopping on val PR-AUC, 5-fold CV HP search
│                                   #   - SVM: scikit-learn SVC, GridSearchCV with 5-fold CV,
│                                   #     kernels {linear, RBF, poly}, probability=True (Platt scaling)
│                                   #   - Class imbalance ablation: none / class_weight / SMOTE
│                                   #   - Threshold tuning on validation F1
│                                   #   - Multi-seed robustness (5 seeds)
│                                   #   - Training-size scaling experiment
│                                   #   - Error analysis & disagreement
│                                   #   - Saves best models to models/
├── test.ipynb                      # Test-time evaluation notebook (20 cells):
│                                   #   - Loads saved models, reconstructs test split
│                                   #   - Evaluates on test set WITHOUT retraining
│                                   #   - Reproduces 2 paper figures
│                                   #   - Pass/fail check against config.json
├── test.html                       # Executed HTML snapshot of test.ipynb
├── readme.txt                      # Submission readme with requirements & setup blocks
├── requirements.txt                # Pinned package versions
└── models/
    ├── best_mlp.pt                 # MLP state_dict + arch metadata + tuned threshold
    ├── best_svm.joblib             # Full sklearn Pipeline: StandardScaler + linear SVC
    ├── scaler.joblib               # StandardScaler fitted on training split
    └── config.json                 # Hyperparameters, thresholds, expected test metrics
```

### Key technical details:

**Framework:** PyTorch (MLP) + scikit-learn (SVM). Mixed GPU/CPU workload.
- MLP training is GPU-beneficial but runs fine on CPU (~2-3 min on CPU for full pipeline).
- SVM GridSearchCV is CPU-parallel (`n_jobs=-1` currently). Runs ~40s on 8 cores.
- SMOTE (imbalanced-learn) is CPU-only.

**Entry points:** Currently the project is entirely notebook-based (`neco_starter.ipynb` runs top-to-bottom). For HPC, the training logic needs to be extracted into standalone `.py` scripts that SLURM can call.

**Saved artifacts:** 
- `best_mlp.pt` saved with `torch.save()` — contains state_dict, hidden_sizes tuple, dropout float, in_dim int, tuned_threshold float.
- `best_svm.joblib` saved with `joblib.dump()` — full sklearn Pipeline.
- `scaler.joblib` — StandardScaler.

**Package versions used:**
- python 3.12, numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0
- imbalanced-learn 0.14.1, torch 2.11.0, matplotlib 3.10.8
- seaborn 0.13.2, joblib 1.5.3

**Known HPC issues in the current code:**
1. `matplotlib` is imported without a headless backend guard — will crash on GPU nodes without X11.
2. `n_jobs=-1` in GridSearchCV will grab ALL node cores, not just the SLURM allocation.
3. `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` is fine but is set once in a notebook cell — needs to be a function in standalone scripts.
4. No CLI argument parsing — everything is hardcoded in notebook cells.
5. Plots use `plt.show()` which blocks on headless nodes.
6. No SLURM job scripts exist.
7. The notebook is the only entry point — no `.py` training scripts.

---

## What to produce

### Step 1 — Audit
Produce a short audit confirming the issues above and identifying any others.

### Step 2 — Extract training scripts from the notebook

Create two standalone Python scripts that replicate the notebook's training pipeline:

**`src/train_mlp.py`** — Extracts the MLP training pipeline from the notebook:
- Loads `creditcard_reduced.csv`
- Performs the stratified 60/20/20 split (same `random_state=42`)
- Scales features with StandardScaler
- Runs the 5-fold CV hyperparameter search (the `cv_score_mlp` function)
- Trains the best MLP config on the training split with early stopping
- Runs the class imbalance ablation
- Tunes threshold on validation
- Evaluates on test set
- Saves the best model to `models/best_mlp.pt`, scaler to `models/scaler.joblib`
- Saves all figures to `outputs/figures/`
- Accepts CLI args: `--seed`, `--epochs`, `--patience`, `--data-path`, `--output-dir`

**`src/train_svm.py`** — Extracts the SVM training pipeline:
- Same data loading and split as above (must produce identical splits)
- Runs GridSearchCV with the same parameter grid
- Runs the class imbalance ablation for SVM
- Tunes threshold on validation
- Evaluates on test set
- Saves the best model to `models/best_svm.joblib`
- Saves all figures to `outputs/figures/`
- Accepts CLI args: `--seed`, `--n-jobs`, `--data-path`, `--output-dir`

**`src/evaluate.py`** — Combined evaluation script:
- Loads both saved models
- Reconstructs the test split
- Produces the combined ROC/PR curves, confusion matrices, disagreement analysis
- Runs the multi-seed robustness experiment
- Runs the training-size scaling experiment
- Saves all figures to `outputs/figures/`
- Accepts CLI args: `--seeds` (list), `--data-path`, `--output-dir`

**`src/utils.py`** — Shared utilities:
- `get_device()` function
- `get_cpu_count()` SLURM-aware function
- `MLP` class definition (shared between train and evaluate)
- `train_mlp()` function
- `tune_threshold()` function
- `evaluate()` metrics function
- Matplotlib headless backend guard
- Data loading and splitting function (to guarantee identical splits everywhere)

### Step 3 — SLURM job scripts

Create `scripts/hpc/` with:

**`scripts/hpc/run_mlp.slurm`** — GPU job for MLP training:
- 1 GPU, 4 CPUs, 16 GB RAM, 1 hour
- Calls `python src/train_mlp.py`

**`scripts/hpc/run_svm.slurm`** — CPU-parallel job for SVM training:
- 0 GPU, 16 CPUs, 16 GB RAM, 2 hours
- Calls `python src/train_svm.py --n-jobs 16`

**`scripts/hpc/run_eval.slurm`** — GPU job for combined evaluation:
- 1 GPU, 4 CPUs, 16 GB RAM, 1 hour
- Calls `python src/evaluate.py`

**`scripts/hpc/run_all.slurm`** — Orchestrator that submits MLP and SVM in parallel, then eval after both complete (using SLURM `--dependency=afterok`)

**`scripts/hpc/run_local.sh`** — Plain bash fallback for local runs or interactive HPC sessions

Each SLURM script must include the header comments for submit/monitor/cancel commands.

### Step 4 — Patch requirements.txt

Add PyTorch install hint at the top. Pin versions with comments explaining why.

### Step 5 — Update readme.txt

Add a "Submitting to HPC" section with:
1. How to submit jobs
2. How to monitor
3. How to cancel
4. Troubleshooting (CUDA mismatch, OOM, time limit, wrong scheduler)

### Step 6 — Do NOT modify

- `test.ipynb` — markers run this, leave it exactly as-is
- `test.html` — leave as-is
- `neco_starter.ipynb` — leave as-is (it's the submitted notebook)
- `models/` contents — these are the saved artifacts from the submitted version
- Working training logic, loss functions, metrics, model architecture

---

## Target project structure after transformation:

```
neco_project/
├── creditcard_reduced.csv
├── neco_starter.ipynb              # UNCHANGED — submitted notebook
├── test.ipynb                      # UNCHANGED — marker evaluation
├── test.html                       # UNCHANGED
├── readme.txt                      # MODIFIED — added HPC section only
├── requirements.txt                # MODIFIED — added PyTorch install hint
├── src/
│   ├── utils.py                    # NEW — shared utilities
│   ├── train_mlp.py                # NEW — standalone MLP training
│   ├── train_svm.py                # NEW — standalone SVM training
│   └── evaluate.py                 # NEW — combined evaluation + figures
├── scripts/
│   └── hpc/
│       ├── run_mlp.slurm           # NEW — GPU job
│       ├── run_svm.slurm           # NEW — CPU job
│       ├── run_eval.slurm          # NEW — evaluation job
│       ├── run_all.slurm           # NEW — orchestrator
│       └── run_local.sh            # NEW — local fallback
├── models/                         # Saved artifacts (existing + regenerated by scripts)
│   ├── best_mlp.pt
│   ├── best_svm.joblib
│   ├── scaler.joblib
│   └── config.json
└── outputs/
    ├── logs/                       # SLURM logs go here
    └── figures/                    # All saved figures go here
```

## Deliverable format

1. The audit (Step 1) as a numbered markdown list.
2. Every new file created, fully written out (no placeholders, no "...").
3. Every modified file shown as a unified diff or clearly marked before/after.
4. A final list grouped as: **Created** / **Modified** / **Unchanged but reviewed**.
```

---

## HOW TO USE THIS PROMPT

1. Copy everything between the ``` markers above.
2. Paste it into a new Claude conversation (or any LLM that can handle long prompts).
3. Attach your current project files: `neco_starter.ipynb`, `test.ipynb`, `requirements.txt`, `readme.txt`, and the `models/` folder contents.
4. The LLM will produce all the new files and patches.
5. Create a Git repo, add all the files, push to GitHub/GitLab.
6. SSH into the HPC, clone the repo, create a venv, install requirements, and run `sbatch scripts/hpc/run_all.slurm`.
