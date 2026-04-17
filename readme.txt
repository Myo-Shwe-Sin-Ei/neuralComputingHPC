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
- neco_starter.ipynb   Main training + evaluation notebook (12 sections; produces all
                       figures, models, and config.json).
- test.ipynb           Loads saved artefacts and reproduces test metrics and two
                       figures without retraining.
- readme.txt           This file.
- requirements.txt     Pinned Python dependencies (with PyTorch CUDA install comment).
- paper.md             6-page research paper, Harvard-style references.
- supplementary.md     2-page supplementary with glossary, full HP search tables,
                       and negative findings.
- scripts/             download_data.py + hpc/ SLURM scripts + hpc/run_local.sh.
- models/              best_mlp.pt, best_svm.joblib, scaler.joblib, config.json
                       (populated after running the main notebook).
- outputs/             fig01..fig09 PNG figures (populated after running the
                       main notebook).
- docs/                Design spec and implementation plan (for transparency).

Dataset source:
    Dal Pozzolo, A., Caelen, O., Johnson, R.A. and Bontempi, G. (2015)
    'Calibrating probability with undersampling for unbalanced classification',
    2015 IEEE Symposium Series on Computational Intelligence (SSCI),
    Cape Town, South Africa, 7-10 December. IEEE, pp. 159-166.
    https://zenodo.org/records/7395559

The main notebook downloads the full dataset (~150 MB) automatically on first
run via scripts/download_data.py, then builds a 10,492-row stratified
subsample (creditcard_reduced.csv) used for all experiments. The markers only
need creditcard_reduced.csv to run test.ipynb; it is produced by the main
notebook and can be committed or regenerated on a fresh clone.

Headline results (5-seed mean +/- std, test set, tuned thresholds):
    MLP:  ROC-AUC <mlp_roc_auc_mean> +/- <mlp_roc_auc_std>
          PR-AUC  <mlp_pr_auc_mean>  +/- <mlp_pr_auc_std>
          F1      <mlp_f1_mean>      +/- <mlp_f1_std>
    SVM:  ROC-AUC <svm_roc_auc_mean> +/- <svm_roc_auc_std>
          PR-AUC  <svm_pr_auc_mean>  +/- <svm_pr_auc_std>
          F1      <svm_f1_mean>      +/- <svm_f1_std>

(Values populated after running neco_starter.ipynb Section 9 — see paper.md
Table I for the full result table.)

Code attribution:
All code in this submission is original work by the author, written using
standard public APIs of NumPy, pandas, scikit-learn, imbalanced-learn,
PyTorch, matplotlib, seaborn, and joblib. No code has been copied verbatim
from online sources; library APIs have been used as documented in their
official reference manuals. Citations in paper.md acknowledge the theoretical
foundations of the methods used (Rumelhart, Hinton and Williams 1986;
Cortes and Vapnik 1995; Platt 1999; Chawla et al. 2002; Saito and
Rehmsmeier 2015; Dal Pozzolo et al. 2015).

Submitting to HPC:
1. SSH to the cluster and git clone this repository.
2. Create and activate a virtualenv. Install PyTorch matching the cluster's
   CUDA version (see header of requirements.txt), then:
       pip install -r requirements.txt
3. Submit the GPU job:
       sbatch scripts/hpc/run_mlp.slurm
   Or the CPU-only variant (if the GPU queue is busy):
       sbatch scripts/hpc/run_svm.slurm
4. Monitor:
       squeue -u $USER
   Tail logs:
       tail -f outputs/logs/slurm-<jobid>.out
   Cancel:
       scancel <jobid>

Troubleshooting (HPC):
- CUDA mismatch ("CUDA error: no kernel image"): reinstall PyTorch with the
  wheel matching `nvidia-smi` output (see the comment block at the top of
  requirements.txt).
- OOM-killed by the scheduler: raise `--mem=32G` in the SLURM script.
- Time-limit hit: raise `--time=HH:MM:SS` (GPU default 2h; CPU 4h).
- Wrong scheduler (e.g. a PBS cluster): translate the `#SBATCH` directives to
  their `#PBS -l` equivalents and rename the file extension to .pbs.
