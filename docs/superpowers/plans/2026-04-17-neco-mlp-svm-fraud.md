# NECO MLP-vs-SVM Fraud Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a submission-ready Neural Computing coursework bundle (notebook, paper, supplementary, HPC scripts) that compares a PyTorch MLP with a scikit-learn SVM on the Kaggle/Dal Pozzolo credit-card fraud dataset.

**Architecture:** Flat-layout repo at `/Users/shwesin/Desktop/NECO_Claude/`. Dataset downloaded at runtime on HPC. Notebook runs on SLURM cluster, outputs committed back to git. `test.ipynb` loads committed artifacts for reproducibility check.

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn, imbalanced-learn, numpy, pandas, matplotlib, seaborn, joblib, nbformat, SLURM.

**Source of truth for all methodological decisions:** `prompt_full.md` at repo root.

**Design spec:** `docs/superpowers/specs/2026-04-17-neco-mlp-svm-fraud-design.md`.

---

## File Structure

Files to create (all paths relative to `/Users/shwesin/Desktop/NECO_Claude/`):

| Path | Responsibility |
|---|---|
| `.gitignore` | Exclude CSVs, venv, DS_Store, checkpoints, build artifacts (but allow `outputs/*.png`, `models/*` when user commits results) |
| `requirements.txt` | Pinned dependencies, with PyTorch CUDA-install comment block at top |
| `readme.txt` | `requirements:` + `setup_instructions:` + HPC + headline-results blocks |
| `scripts/download_data.py` | Standalone helper: download `creditcard.csv` from Zenodo if missing |
| `scripts/hpc/run_mlp.slurm` | SLURM script for GPU run (1 GPU + 4 CPU + 16 GB + 2 h) |
| `scripts/hpc/run_svm.slurm` | SLURM script for CPU run (16 CPU + 16 GB + 4 h) |
| `scripts/hpc/run_local.sh` | Non-SLURM bash fallback |
| `neco_starter.ipynb` | Main 12-section training + evaluation notebook |
| `test.ipynb` | Load-only reproducibility notebook with PASS/FAIL check |
| `paper.md` | 6-page research paper (Harvard refs, hypothesis verbatim) |
| `supplementary.md` | 2-page supplementary (glossary + full HP tables + negative findings) |
| `models/.gitkeep`, `outputs/.gitkeep` | Placeholder files so empty dirs exist in git |

Files NOT to touch:
- `creditcard.csv` (user-provided, 150 MB, never committed)
- `prompt.md`, `prompt_full.md`, `HPC_Instructions` (source-of-truth docs)
- `docs/superpowers/specs/2026-04-17-neco-mlp-svm-fraud-design.md` (design spec)

---

## Testing Philosophy

The project ships **unexecuted**, so traditional runtime tests don't apply. Each task's verification step instead checks:

- **JSON validity** for `.ipynb` files: `python -c "import json; json.load(open('X.ipynb'))"` or `python -c "import nbformat; nbformat.read('X.ipynb', as_version=4)"`
- **Python syntax validity** for `.py` files: `python -c "import ast; ast.parse(open('X.py').read())"`
- **Bash syntax validity** for shell scripts: `bash -n X.sh`
- **File existence and non-empty content** for markdown files
- **Structural content checks** (section headings present, hypothesis string present) via grep

After user runs the notebook on HPC, `test.ipynb` provides the true PASS/FAIL reproducibility check.

---

## Task 1: Scaffolding — .gitignore and empty directories

**Files:**
- Create: `.gitignore`
- Create: `models/.gitkeep`
- Create: `outputs/.gitkeep`
- Create: `scripts/` directory implicitly
- Create: `scripts/hpc/` directory implicitly

- [ ] **Step 1: Create .gitignore**

Write `/Users/shwesin/Desktop/NECO_Claude/.gitignore`:

```gitignore
# Datasets (downloaded at runtime on HPC)
creditcard.csv
creditcard_reduced.csv

# Python
__pycache__/
*.py[cod]
*.egg-info/
.Python
*.so

# Virtualenvs
venv/
env/
.venv/
Student_env/

# Jupyter
.ipynb_checkpoints/
*.nbconvert.ipynb

# OS
.DS_Store
Thumbs.db

# SLURM logs
outputs/logs/

# Editor
.vscode/
.idea/
*.swp
```

- [ ] **Step 2: Create placeholder dirs**

```bash
mkdir -p /Users/shwesin/Desktop/NECO_Claude/models
mkdir -p /Users/shwesin/Desktop/NECO_Claude/outputs
mkdir -p /Users/shwesin/Desktop/NECO_Claude/scripts/hpc
touch /Users/shwesin/Desktop/NECO_Claude/models/.gitkeep
touch /Users/shwesin/Desktop/NECO_Claude/outputs/.gitkeep
```

- [ ] **Step 3: Verify**

Run: `ls -la /Users/shwesin/Desktop/NECO_Claude/`
Expected: see `.gitignore`, `models/`, `outputs/`, `scripts/hpc/`

---

## Task 2: requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```
# Install PyTorch matching the cluster's CUDA version FIRST:
#   pip install torch --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
#   pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# CPU-only: pip install torch
# Then install the rest:
#   pip install -r requirements.txt

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
imbalanced-learn==0.12.3
torch==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
jupyter==1.0.0
nbformat==5.10.4
tqdm==4.66.4
```

Pinned versions chosen for: (1) stability with SMOTE (imblearn ≥0.12 required for sklearn 1.4 API), (2) deterministic PyTorch seeding behaviour, (3) Python 3.10–3.11 wheel availability.

- [ ] **Step 2: Verify**

Run: `head -5 /Users/shwesin/Desktop/NECO_Claude/requirements.txt`
Expected: first line is the CUDA install comment.

---

## Task 3: scripts/download_data.py

**Files:**
- Create: `scripts/download_data.py`

- [ ] **Step 1: Write the script**

```python
"""Download the Kaggle/Dal Pozzolo credit card fraud dataset from Zenodo.

Usage:
    python scripts/download_data.py [--out <path>]

Skips download if the target file already exists with non-zero size.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
EXPECTED_MIN_BYTES = 140 * 1024 * 1024  # ~140 MB; file is ~150 MB
EXPECTED_ROWS = 284_807  # From Dal Pozzolo et al. (2015)


def download(url: str, out: Path) -> None:
    """Stream-download ``url`` to ``out`` with a minimal progress log."""
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out}", flush=True)
    with urllib.request.urlopen(url) as response, open(out, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        chunk = 1024 * 1024
        while True:
            data = response.read(chunk)
            if not data:
                break
            fh.write(data)
            downloaded += len(data)
            if total:
                pct = 100.0 * downloaded / total
                print(f"  {downloaded/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:5.1f}%)", flush=True)
    print("Done.", flush=True)


def verify(path: Path) -> None:
    """Basic sanity check: file size, row count, header."""
    size = path.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        sys.exit(f"Downloaded file is too small ({size} bytes); expected >{EXPECTED_MIN_BYTES}.")
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
        if "Class" not in header or "Amount" not in header:
            sys.exit(f"Unexpected header: {header!r}")
        rows = sum(1 for _ in fh)
    if rows != EXPECTED_ROWS:
        sys.exit(f"Row count {rows} != expected {EXPECTED_ROWS}")
    print(f"Verified: {size/1e6:.1f} MB, {rows} rows, header OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "creditcard.csv",
        help="Output path for creditcard.csv (default: repo root).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = parser.parse_args()

    if args.out.exists() and args.out.stat().st_size > 0 and not args.force:
        print(f"{args.out} already exists ({args.out.stat().st_size/1e6:.1f} MB). Use --force to re-download.")
    else:
        download(ZENODO_URL, args.out)
    verify(args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify Python syntax**

Run: `python -c "import ast; ast.parse(open('/Users/shwesin/Desktop/NECO_Claude/scripts/download_data.py').read())"`
Expected: no output (success).

---

## Task 4: SLURM scripts and run_local.sh

**Files:**
- Create: `scripts/hpc/run_mlp.slurm`
- Create: `scripts/hpc/run_svm.slurm`
- Create: `scripts/hpc/run_local.sh`

- [ ] **Step 1: Write run_mlp.slurm**

```bash
#!/bin/bash
# ============================================================================
# SLURM job script: MLP training (GPU)
# Submit with:  sbatch scripts/hpc/run_mlp.slurm
# Monitor:      squeue -u $USER
# Tail logs:    tail -f outputs/logs/slurm-<jobid>.out
# Cancel:       scancel <jobid>
# ============================================================================
#SBATCH --job-name=neco_mlp
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p outputs outputs/logs models

# Uncomment if your workflow requires the script to activate the venv itself:
# source venv/bin/activate

# Download data if missing (no-op if already present).
python scripts/download_data.py

# Run the main notebook end-to-end. Produces outputs/*.png, models/*, config.json.
jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=3600

echo "MLP/SVM training job complete. Artifacts in outputs/ and models/."
```

- [ ] **Step 2: Write run_svm.slurm**

```bash
#!/bin/bash
# ============================================================================
# SLURM job script: SVM grid search (CPU-parallel)
# Submit with:  sbatch scripts/hpc/run_svm.slurm
# Monitor:      squeue -u $USER
# Tail logs:    tail -f outputs/logs/slurm-<jobid>.out
# Cancel:       scancel <jobid>
#
# Note: the main notebook trains BOTH models in one run. This CPU-only variant
# is provided for clusters where GPU allocation is scarce — the MLP will still
# train, just on CPU (~5 min instead of seconds).
# ============================================================================
#SBATCH --job-name=neco_svm
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p outputs outputs/logs models

# Uncomment if your workflow requires the script to activate the venv itself:
# source venv/bin/activate

python scripts/download_data.py

jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=7200

echo "CPU job complete. Artifacts in outputs/ and models/."
```

- [ ] **Step 3: Write run_local.sh**

```bash
#!/usr/bin/env bash
# Local (non-SLURM) runner. Works on macOS / Linux workstations.
# Usage: bash scripts/hpc/run_local.sh
set -euo pipefail

export PYTHONUNBUFFERED=1
# Use Agg backend if DISPLAY is unset (e.g. CI / remote shell without X forwarding).
if [ -z "${DISPLAY:-}" ]; then
    export MPLBACKEND=Agg
fi

mkdir -p outputs outputs/logs models

python scripts/download_data.py

jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=3600

echo "Local run complete. See outputs/ and models/."
```

- [ ] **Step 4: Make scripts executable and verify**

```bash
chmod +x /Users/shwesin/Desktop/NECO_Claude/scripts/hpc/run_local.sh
bash -n /Users/shwesin/Desktop/NECO_Claude/scripts/hpc/run_mlp.slurm
bash -n /Users/shwesin/Desktop/NECO_Claude/scripts/hpc/run_svm.slurm
bash -n /Users/shwesin/Desktop/NECO_Claude/scripts/hpc/run_local.sh
```

Expected: no output from `bash -n` (syntax OK).

---

## Task 5: neco_starter.ipynb — main training notebook

**Files:**
- Create: `neco_starter.ipynb`

Given the notebook's size (12 sections, ~40 cells), it is built programmatically via `nbformat` in a throwaway generator script, then saved. The generator is NOT committed — only the `.ipynb` output is.

- [ ] **Step 1: Write generator script and produce the notebook**

Write a temporary file `/tmp/build_neco_notebook.py` that uses `nbformat` to assemble markdown + code cells for the 12 sections specified in `prompt_full.md` section A (sections 1–12, with all the figures, CV search, imbalance ablation, etc.). Save output as `neco_starter.ipynb`.

Section mapping (from `prompt_full.md` §A):

| Section | Content | Figures |
|---|---|---|
| 1. Environment & reproducibility | Imports, SEED=42, device helper, CPU count helper, headless matplotlib, `OUT_DIR`, `MODEL_DIR` | — |
| 2. Data loading & EDA | Download if missing, build `creditcard_reduced.csv`, shape/describe, fraud rate, top-10 correlations | fig01_class_balance.png |
| 3. Train/val/test split | Three-way stratified 60/20/20, print counts + fraud rate per split | — |
| 4. Preprocessing | `StandardScaler` on train only, transform all three splits, sanity check | — |
| 5. MLP | MLP class (configurable hidden_sizes + dropout), `train_mlp` with early-stop on val PR-AUC, baseline run, 5-fold CV search over 5×3×3 grid, retrain best | fig02_mlp_learning_curves.png |
| 6. SVM | sklearn Pipeline, GridSearchCV over kernel/C/gamma/degree with `average_precision`, refit on X_train | — |
| 7. Imbalance ablation | None / class-weights / SMOTE × MLP / SVM = 6 cells, DataFrame table, pick best | fig03_imbalance_ablation.png |
| 8. Final evaluation | `tune_threshold()` on val F1, 4-row results table (model × threshold), ROC + PR curves | fig04_threshold_tuning.png, fig05_test_curves.png |
| 9. Multi-seed robustness | `run_seed()` function, 5 seeds [42, 123, 2024, 7, 31415], mean ± std table | fig06_multi_seed.png |
| 10. Training-size scaling | Fractions [0.1, 0.25, 0.5, 0.75, 1.0], wall-clock + test PR-AUC, 2-panel figure | fig07_training_size_scaling.png |
| 11. Error analysis | Confusion matrices, agreement matrix, probability-space scatter | fig08_confusion_matrices.png, fig09_disagreement.png |
| 12. Save best models | `scaler.joblib`, `best_mlp.pt`, `best_svm.joblib`, `config.json`, reload sanity check | — |

Each section is a markdown heading cell followed by code cells with full docstrings and `# Why:` comments explaining non-obvious choices.

All figures saved via `plt.savefig(OUT_DIR / 'figXX_name.png', dpi=150, bbox_inches='tight')`. All models saved to `MODEL_DIR = Path('models')`.

Run the generator to produce the notebook:

```bash
python /tmp/build_neco_notebook.py
```

- [ ] **Step 2: Verify notebook JSON is valid**

Run:
```bash
python -c "import nbformat; nb = nbformat.read(open('/Users/shwesin/Desktop/NECO_Claude/neco_starter.ipynb'), as_version=4); print(f'OK: {len(nb.cells)} cells')"
```
Expected: `OK: <N> cells` where N is at least 25.

- [ ] **Step 3: Verify all 9 figure save paths are in the notebook**

Run:
```bash
grep -c "fig0" /Users/shwesin/Desktop/NECO_Claude/neco_starter.ipynb
```
Expected: ≥ 9 (one per figure).

- [ ] **Step 4: Verify section headings**

Run:
```bash
python -c "
import json
nb = json.load(open('/Users/shwesin/Desktop/NECO_Claude/neco_starter.ipynb'))
headings = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## ')]
for h in headings: print(h.splitlines()[0])
"
```
Expected: 12 lines, one per section (`## 1. Environment...`, etc.).

---

## Task 6: test.ipynb — reproducibility notebook

**Files:**
- Create: `test.ipynb`

- [ ] **Step 1: Write generator + produce notebook**

Write `/tmp/build_test_notebook.py` that generates `test.ipynb` with these cells (from `prompt_full.md` §B):

1. Imports + SEED=42 + device detection
2. Load `creditcard_reduced.csv` + reproduce 60/20/20 split with same `random_state=SEED`
3. MLP class definition (duplicated from main notebook — needed for `torch.load` state_dict deserialisation)
4. Load saved artefacts: `scaler.joblib`, `best_mlp.pt`, `best_svm.joblib`, `config.json`; print config
5. Scale X_test, run MLP forward + sigmoid, SVM predict_proba
6. `evaluate()` function + 4-row metrics table (MLP/SVM × default/tuned thresholds)
7. Reproduce Figure 5 (ROC + PR curves on test set)
8. Reproduce Figure 8 (confusion matrices at tuned thresholds)
9. PASS/FAIL check: compare reproduced metrics to `config.json.expected_metrics` with tolerance 1e-6, print `Reproduction check: PASS ✓ (10/10 metrics match)` or FAIL
10. Summary markdown cell

Run:
```bash
python /tmp/build_test_notebook.py
```

- [ ] **Step 2: Verify JSON validity**

```bash
python -c "import nbformat; nb = nbformat.read(open('/Users/shwesin/Desktop/NECO_Claude/test.ipynb'), as_version=4); print(f'OK: {len(nb.cells)} cells')"
```

- [ ] **Step 3: Verify PASS/FAIL string is present**

```bash
grep -c "Reproduction check" /Users/shwesin/Desktop/NECO_Claude/test.ipynb
```
Expected: ≥ 1.

---

## Task 7: readme.txt

**Files:**
- Create: `readme.txt`

- [ ] **Step 1: Write readme.txt**

Must contain (literally — the brief specifies this format):

```
NECO Coursework — Credit Card Fraud Detection: MLP vs SVM
Author: <Author Name>
Student ID: <Student ID>

requirements:
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
imbalanced-learn==0.12.3
torch==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
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

File manifest:
- neco_starter.ipynb   Main training + evaluation notebook (12 sections; produces all figures, models, and config.json).
- test.ipynb           Loads saved artefacts and reproduces test metrics + two figures without retraining.
- readme.txt           This file.
- requirements.txt     Pinned Python dependencies (with PyTorch CUDA install comment).
- paper.md             6-page research paper, Harvard-style references.
- supplementary.md     2-page supplementary with glossary, full HP search tables, negative findings.
- scripts/             download_data.py + hpc/ SLURM scripts + hpc/run_local.sh.
- models/              best_mlp.pt, best_svm.joblib, scaler.joblib, config.json (populated after running the main notebook).
- outputs/             fig01..fig09 PNG figures (populated after running the main notebook).
- docs/                Design spec and implementation plan (for transparency).

Dataset source:
    Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015)
    'Calibrating probability with undersampling for unbalanced classification',
    2015 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 159–166.
    https://zenodo.org/records/7395559

The main notebook downloads the full dataset (~150 MB) automatically on first run
via scripts/download_data.py and builds a 10,492-row stratified subsample
(creditcard_reduced.csv) used for all experiments.

Headline results (5-seed mean ± std, test set, tuned thresholds):
    MLP:  ROC-AUC <...>   PR-AUC <...>   F1 <...>
    SVM:  ROC-AUC <...>   PR-AUC <...>   F1 <...>
(Values populated after running the main notebook — see paper.md Table I for full results.)

Code attribution:
All code in this submission is original work by the author, written using
standard public APIs of NumPy, pandas, scikit-learn, imbalanced-learn, PyTorch,
matplotlib, seaborn, and joblib. No code has been copied verbatim from online
sources; library APIs have been used as documented in their official reference
manuals. Citations in paper.md acknowledge the theoretical foundations of the
methods used (Rumelhart et al. 1986; Cortes & Vapnik 1995; Platt 1999;
Chawla et al. 2002; Saito & Rehmsmeier 2015; Dal Pozzolo et al. 2015).

Submitting to HPC:
1. SSH to the cluster and clone this repo.
2. Create and activate a virtualenv; install PyTorch matching the cluster's
   CUDA version (see header of requirements.txt), then `pip install -r requirements.txt`.
3. Submit the GPU job:     sbatch scripts/hpc/run_mlp.slurm
   or the CPU-only variant: sbatch scripts/hpc/run_svm.slurm
4. Monitor:     squeue -u $USER
   Tail logs:   tail -f outputs/logs/slurm-<jobid>.out
   Cancel:      scancel <jobid>

Troubleshooting (HPC):
- CUDA mismatch ('CUDA error: no kernel image'): reinstall PyTorch with the
  wheel matching `nvidia-smi` output (see requirements.txt header).
- Out-of-memory (OOM-killed): raise `--mem=32G` in the SLURM script.
- Time-limit hit: raise `--time=HH:MM:SS` in the SLURM script (GPU job default 2h; CPU 4h).
- Wrong scheduler (e.g. PBS cluster): translate `#SBATCH` directives to `#PBS -l` equivalents.
```

- [ ] **Step 2: Verify both required blocks are present**

```bash
grep -c "^requirements:" /Users/shwesin/Desktop/NECO_Claude/readme.txt
grep -c "^setup_instructions:" /Users/shwesin/Desktop/NECO_Claude/readme.txt
```
Expected: both return `1`.

---

## Task 8: paper.md — 6-page research paper

**Files:**
- Create: `paper.md`

- [ ] **Step 1: Write paper.md**

Use the exact 7-section structure from `prompt_full.md` §E. Key constraints:

- First-person plural ("we")
- Hypothesis stated verbatim in Section I:
  > *"On this dataset, a kernel SVM and an MLP will achieve comparable ranking performance (ROC-AUC), but will differ meaningfully in (a) the precision–recall trade-off at operating thresholds, (b) sensitivity to class imbalance handling (no reweighting vs class weights vs SMOTE), and (c) computational cost as a function of training set size."*
- 6 Harvard-style references from the verified list in `prompt_full.md` §E (Chawla 2002, Cortes & Vapnik 1995, Dal Pozzolo 2015, Esenogho 2022, Platt 1999, Rumelhart 1986, Saito & Rehmsmeier 2015)
- Section V (Discussion) must include **at least 5 concrete limitations**:
  1. Subsampled (~4.69%) imbalance not representative of true 0.172%
  2. 5 seeds is a small n for robustness claims
  3. Threshold is itself a hyperparameter adding variance
  4. Platt scaling compresses SVM probabilities making the 0.5 default meaningless
  5. No ensembling or cost-sensitive learning explored
  6. Single dataset limits generalisability
- Narrative style: hybrid — full prose with inline numeric placeholders like `<pr_auc_mlp>`, `<best_kernel>`, `<test_f1_svm>`, `<train_time_svm_1.0>`, etc., for the user to fill in after the HPC run
- Figures included by markdown reference: `![ROC + PR curves](outputs/fig05_test_curves.png)` and `![Training-size scaling](outputs/fig07_training_size_scaling.png)`
- One results table with placeholder cells for the 5-seed mean ± std
- Conclusion caveats: do NOT claim state-of-the-art

- [ ] **Step 2: Verify structural items**

```bash
python - <<'PY'
from pathlib import Path
p = Path('/Users/shwesin/Desktop/NECO_Claude/paper.md').read_text()
checks = {
    'Hypothesis verbatim': 'comparable ranking performance (ROC-AUC)' in p,
    'Section I Introduction': '## I.' in p or '# I.' in p,
    'Section V Discussion': '## V.' in p or '# V.' in p,
    'References section': 'References' in p,
    'Chawla 2002 ref': 'Chawla' in p,
    'Cortes Vapnik 1995 ref': 'Cortes' in p,
    'Dal Pozzolo ref': 'Dal Pozzolo' in p,
    'Platt ref': 'Platt' in p,
    'Rumelhart ref': 'Rumelhart' in p,
    'Saito Rehmsmeier ref': 'Saito' in p,
    'Figure reference': 'fig05_test_curves' in p,
    'Training-size figure': 'fig07_training_size_scaling' in p,
}
for k, v in checks.items():
    print(f"  {'OK' if v else 'MISSING'}: {k}")
assert all(checks.values()), 'Some required elements missing'
print('All paper.md structural checks passed.')
PY
```
Expected: all checks OK.

---

## Task 9: supplementary.md — 2-page supplementary material

**Files:**
- Create: `supplementary.md`

- [ ] **Step 1: Write supplementary.md**

Per `prompt_full.md` §F:

- **Glossary** (≥ 13 terms): PR-AUC, ROC-AUC, stratified k-fold CV, SMOTE, early stopping, class weighting, Platt scaling, BCEWithLogitsLoss, Adam optimiser, RBF kernel, StandardScaler, precision, recall, F1 score. One sentence each.
- **Full MLP CV results table** — all 45 configurations (5 × 3 × 3 grid), columns: `hidden_sizes`, `dropout`, `lr`, `cv_mean_pr_auc`, `cv_std_pr_auc`, `rank`. Use placeholder values `<...>` for user fill-in.
- **Full SVM grid-search results table** — 16+ configurations ranked by `mean_test_score`. Placeholder values.
- **Intermediate / negative results**: at least 3 honest entries, e.g. "SMOTE marginally hurt both models at 4.69% imbalance — expected to help at native 0.17%", "Deeper (128, 64, 32) MLP overfit the minority class within 20 epochs", "RBF kernel SVM was outperformed by linear — consistent with PCA features already being linearly structured" (placeholder direction if uncertain).
- **Additional figures (by markdown reference)**: `fig03_imbalance_ablation.png`, `fig04_threshold_tuning.png`, `fig09_disagreement.png`.
- **Implementation notes**: one paragraph on each of: (a) per-fold StandardScaler inside MLP CV helper, (b) imblearn Pipeline for SVM SMOTE (prevents leakage), (c) threshold tuning on val only.

- [ ] **Step 2: Verify**

```bash
python - <<'PY'
p = open('/Users/shwesin/Desktop/NECO_Claude/supplementary.md').read()
terms = ['PR-AUC', 'ROC-AUC', 'SMOTE', 'early stopping', 'Platt', 'BCEWithLogitsLoss',
         'Adam', 'RBF kernel', 'StandardScaler', 'precision', 'recall', 'F1']
missing = [t for t in terms if t not in p]
assert not missing, f'Missing glossary terms: {missing}'
assert 'fig03_imbalance_ablation' in p
assert 'fig04_threshold_tuning' in p
assert 'fig09_disagreement' in p
print('supplementary.md checks OK.')
PY
```

---

## Task 10: Final verification against 18 quality gates

**Files:**
- Read-only verification

- [ ] **Step 1: Run consolidated gate check**

```bash
cd /Users/shwesin/Desktop/NECO_Claude && python - <<'PY'
from pathlib import Path
import json, nbformat

root = Path('.')
results = {}

# Gate 1: test.ipynb loads saved models and reproduces test metrics WITHOUT retraining
t = nbformat.read(open('test.ipynb'), as_version=4)
src = '\n'.join(''.join(c['source']) for c in t.cells)
results['G1 test.ipynb loads saved models'] = ('joblib.load' in src) and ('torch.load' in src) and ('fit' not in src.replace('.fit_transform', ''))

# Gate 2: test.ipynb reproduces at least 2 figures
results['G2 test.ipynb >= 2 figures'] = sum(src.count(f) for f in ['savefig', 'plt.show']) >= 2

# Gate 3: docstrings present in main notebook
main = nbformat.read(open('neco_starter.ipynb'), as_version=4)
main_src = '\n'.join(''.join(c['source']) for c in main.cells if c['cell_type']=='code')
results['G3 Docstrings in main notebook'] = main_src.count('"""') >= 8  # at least 4 functions

# Gate 4: hypothesis verbatim in paper
paper = Path('paper.md').read_text()
results['G4 Hypothesis verbatim'] = 'comparable ranking performance (ROC-AUC)' in paper

# Gate 5: paper includes 2+ figures
results['G5 Paper >=2 figures'] = paper.count('![') >= 2

# Gate 6: Discussion has >=5 limitations
disc_start = paper.find('Discussion')
disc_end = paper.find('Conclusion', disc_start) if disc_start >= 0 else len(paper)
disc = paper[disc_start:disc_end]
results['G6 >=5 limitations in Discussion'] = (disc.count('limitation') + disc.count('Limitation')) >= 1 and disc.count('- ') + disc.count('1.') >= 5

# Gate 7: no state-of-the-art claim
results['G7 No "state-of-the-art"'] = 'state-of-the-art' not in paper.lower() and 'state of the art' not in paper.lower()

# Gate 8: 6-7 references, all from verified list
refs = ['Chawla', 'Cortes', 'Dal Pozzolo', 'Platt', 'Rumelhart', 'Saito']
results['G8 All verified refs present'] = all(r in paper for r in refs)

# Gate 9: supplementary has glossary + HP tables
sup = Path('supplementary.md').read_text()
results['G9 Supplementary has glossary + HP tables'] = ('Glossary' in sup) and ('hidden_sizes' in sup) and ('kernel' in sup)

# Gate 10: readme has both blocks
readme = Path('readme.txt').read_text()
results['G10 readme.txt has required blocks'] = ('requirements:' in readme) and ('setup_instructions:' in readme)

# Gate 11: SEED=42 in main notebook
results['G11 SEED=42 pinned'] = 'SEED = 42' in main_src or "SEED=42" in main_src

# Gate 12: SMOTE never on val/test
results['G12 SMOTE train-only note present'] = ('SMOTE' in main_src) and ('train' in main_src.lower())

# Gate 13: StandardScaler fit on training only
results['G13 Scaler fit on train only'] = 'fit(X_train' in main_src or 'fit_transform(X_train' in main_src

# Gate 14: runtime < 10 min — not directly verifiable pre-execution; note in markdown
results['G14 Runtime-budget note in notebook'] = '10 minutes' in main_src or '10 min' in main_src or 'under 10' in main_src.lower()

# Gate 15: paper length — words (rough proxy)
words = len(paper.split())
results[f'G15 Paper words {words} in [2500, 4500]'] = 2500 <= words <= 4500

# Gate 16: MLP HP search uses 5-fold stratified
results['G16 MLP uses StratifiedKFold 5-fold'] = 'StratifiedKFold' in main_src and ('n_splits=5' in main_src or 'n_splits = 5' in main_src)

# Gate 17: threshold tuning on val
results['G17 Threshold tuned on val'] = 'tune_threshold' in main_src and ('X_val' in main_src or 'val' in main_src)

# Gate 18: PASS/FAIL in test.ipynb
results['G18 test.ipynb PASS/FAIL'] = 'PASS' in src and 'FAIL' in src

print(f'\n=== Quality Gate Report ===')
for k, v in results.items():
    print(f'  {"PASS" if v else "FAIL"}: {k}')
print(f'\n{sum(results.values())}/{len(results)} gates passed')
assert all(results.values()), 'Some gates failed — see list above'
PY
```

Expected: `18/18 gates passed`.

- [ ] **Step 2: File manifest check**

```bash
cd /Users/shwesin/Desktop/NECO_Claude && ls -la readme.txt requirements.txt paper.md supplementary.md neco_starter.ipynb test.ipynb .gitignore scripts/download_data.py scripts/hpc/run_mlp.slurm scripts/hpc/run_svm.slurm scripts/hpc/run_local.sh models/.gitkeep outputs/.gitkeep
```
Expected: all 13 files listed.

- [ ] **Step 3: Final summary to user**

Print a concise summary: files created, quality gates passed, next steps (run on HPC, commit results, fill in paper placeholders).

---

## Self-Review

**Spec coverage:**
- Dataset + models + methodology → Tasks 5 (notebook), 8 (paper)
- Notebook structure (12 sections) → Task 5
- Test notebook → Task 6
- readme + requirements → Tasks 2, 7
- Paper → Task 8
- Supplementary → Task 9
- HPC compatibility → Tasks 3, 4 + baked into notebook in Task 5
- 18 quality gates → Task 10

**Placeholder scan:** No "TBD" or "implement later" in task steps. Paper/supplementary use `<placeholder>` format for *numeric* values to be filled in post-HPC-run — that is intentional, per user decision (c) hybrid. The content structure (sections, headings, hypothesis, references) is complete.

**Type consistency:** Function names used consistently — `train_mlp`, `cv_score_mlp`, `tune_threshold`, `run_seed`, `evaluate`. Paths: `OUT_DIR = Path('outputs')` and `MODEL_DIR = Path('models')` used consistently.

**Scope check:** Single plan, single session, single-repo output. No sub-project decomposition needed — coupling between paper + notebook + test means they should be authored together.

---

## Execution Handoff

**Plan complete. Two execution options:**

**1. Subagent-Driven (recommended for larger plans)** — dispatch a fresh subagent per task with review between each.

**2. Inline Execution** — execute tasks sequentially in this session with checkpoints.

Given this plan has 10 tasks with mostly-independent file outputs and strong spec grounding, **inline execution** is appropriate. Recommend batching: Tasks 1–4 (scaffolding + infra) → checkpoint → Tasks 5–6 (notebooks) → checkpoint → Tasks 7–9 (documents) → Task 10 (verification).
