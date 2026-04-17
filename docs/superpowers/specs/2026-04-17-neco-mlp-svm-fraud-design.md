# NECO Coursework: MLP vs SVM on Credit Card Fraud — Design Spec

**Date:** 2026-04-17
**Status:** Approved (ready for implementation planning)
**Source requirements:** `prompt_full.md` in repo root (exhaustive task specification)

## 1. Purpose

Produce a complete, submission-ready Master's-level Neural Computing coursework bundle comparing a PyTorch MLP and a scikit-learn SVM on the Kaggle/Dal Pozzolo credit-card fraud dataset. Target: the 75-mark marking scheme in `prompt_full.md`, with specific emphasis on the 20-mark "Critical evaluation of results" bucket.

The spec in `prompt_full.md` is authoritative for all methodological choices (dataset, models, loss functions, grid search, imbalance strategies, metrics, paper structure, references). This document records only the workflow-level decisions agreed during brainstorming.

## 2. Agreed workflow

1. User commits the repo (no dataset files).
2. `git clone` on a university Linux HPC cluster with GPU nodes (SLURM scheduler).
3. SLURM job (or interactive run) downloads `creditcard.csv` from Zenodo, builds the stratified subsample `creditcard_reduced.csv`, runs the full training + evaluation pipeline, saves figures/models/config.
4. User commits outputs back to the repo and pushes.
5. Markers (or user) clone the repo and run `test.ipynb` — which loads the committed artifacts and produces a PASS/FAIL reproducibility check without retraining.

## 3. Key decisions (confirmed with user)

| Decision | Choice | Rationale |
|---|---|---|
| Notebook execution | Ship unexecuted | User runs on HPC, reviews results locally |
| HTML renders | Not produced | User reviews via Jupyter directly |
| HPC compatibility | Full — SLURM scripts + code helpers | HPC_Instructions file in repo confirms HPC is primary target |
| Author identity | `<Author Name>` / `<Student ID>` placeholders | User fills in at submission time |
| Repo layout | Flat at root (no `submission/` wrapper) | User git-clones the root, so root == submission |
| Dataset handling | Download at runtime on HPC, gitignore CSVs | 150 MB full CSV too large for git; download URL `https://zenodo.org/records/7395559/files/creditcard.csv?download=1` |
| Paper narrative | Hybrid — full prose with inline numeric placeholders (e.g. `<pr_auc_mlp>`) | Near-final paper, user fills in concrete numbers after HPC run |

## 4. Deliverables

Per `prompt_full.md` section "DELIVERABLE CHECKLIST", with the following adjustments from the agreed workflow:

**Shipped in git (authored by me):**
- `readme.txt` — requirements: + setup_instructions: + HPC section + headline-results placeholder
- `requirements.txt` — pinned versions, PyTorch CUDA comment block
- `neco_starter.ipynb` — 12 sections per §A of `prompt_full.md`, with HPC helpers baked in
- `test.ipynb` — loads artifacts, reproduces 2 figures, PASS/FAIL check
- `paper.md` — 6-page research paper, Harvard refs, hypothesis verbatim
- `supplementary.md` — 2-page supplementary with glossary + HP tables
- `scripts/download_data.py` — standalone Zenodo downloader
- `scripts/hpc/run_mlp.slurm`, `scripts/hpc/run_svm.slurm`, `scripts/hpc/run_local.sh`
- `.gitignore` — excludes CSVs, venv, DS_Store, notebook checkpoints, output artifacts (user uncomments/removes before committing results)
- `models/.gitkeep`, `outputs/.gitkeep`
- This design spec

**Generated on HPC by the user (committed after run):**
- `creditcard_reduced.csv` (or kept out of git — user's choice)
- `outputs/fig01–fig09.png`
- `models/best_mlp.pt`, `models/best_svm.joblib`, `models/scaler.joblib`, `models/config.json`

## 5. Architectural guarantees (from `prompt_full.md`)

These are repeated here because they gate correctness and the 20-mark critical-evaluation bucket:

- `StandardScaler` fit on training split only — never on val, test, or combined data before splitting.
- SMOTE applied only to training data, inside CV folds via `imblearn.Pipeline` for the SVM path.
- MLP early stopping on validation PR-AUC, not validation loss.
- Threshold tuning on validation set only, never on test.
- Seed pinning: `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- MLP CV search uses 5-fold stratified to match SVM's GridSearchCV methodology.
- Both models use `average_precision` / PR-AUC as primary metric.
- SVM `probability=True` (Platt scaling) required for PR curves.
- Stratified subsample: all 492 frauds + 10,000 legit = 10,492 rows, seed=42.
- Test notebook runtime < 30 s on CPU; main notebook runtime < 10 min on CPU (HPC GPU expected to be faster).

## 6. HPC-specific additions

- `get_device()` helper — autodetect CUDA
- `get_cpu_count()` — respects `SLURM_CPUS_PER_TASK`, falls back to `os.cpu_count()`
- Headless matplotlib guard — `matplotlib.use('Agg')` when no `DISPLAY`
- All paths via `pathlib.Path`
- `tqdm(..., disable=not sys.stdout.isatty())` to quiet progress bars in SLURM logs
- `PYTHONUNBUFFERED=1` and `MPLBACKEND=Agg` exported in SLURM scripts
- `mkdir -p outputs outputs/logs models` at the top of each SLURM script
- Troubleshooting block in readme covering CUDA mismatch / OOM / time-limit / wrong scheduler

## 7. Explicit non-goals

- Not executing the notebooks.
- Not exporting HTML renders.
- Not committing the dataset files.
- Not filling in author identity.
- Not creating a `submission/` wrapper folder.
- Not committing to git (user's responsibility).
- Not claiming state-of-the-art or inflating findings in the paper.
- Not refactoring / improving anything beyond what the prompt specifies.

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Empirical claims in paper contradicted by HPC run | Hybrid placeholders; user verifies every number before submission |
| HPC CUDA / PyTorch version mismatch | PyTorch install comment at top of requirements.txt; pinned versions; troubleshooting block |
| 6-page limit breached after Word conversion | Paper aims for ~3000–3500 words + 2 figures + refs; user trims in Word if needed |
| Markers cannot run test.ipynb (no CSV on their machine) | test.ipynb reconstructs the test split from `creditcard_reduced.csv` — which user commits after running — so markers only need `git clone` + pip install, no dataset download |
| SMOTE / imblearn CV leakage | Use imblearn Pipeline (not sklearn Pipeline) for SVM SMOTE path; write explicit markdown note in notebook |

## 9. Out-of-scope for this spec

Anything not listed in `prompt_full.md`'s 12-section notebook structure, marking scheme, or deliverable checklist. If something comes up during implementation that wasn't in either document, flag it and ask before adding.
