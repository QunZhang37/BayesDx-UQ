# BayesDx-UQ: Bayesian Deep Learning for Uncertainty Quantification in Medical Diagnosis

**Short description:** End-to-end research codebase for *Bayesian and statistically principled* uncertainty quantification in medical diagnosis, including MC Dropout, Deep Ensembles, Evidential Deep Learning, Calibration (ECE/ACE), Conformal Prediction, and uncertainty-aware triage.

## Highlights
- **Bayesian approximations**: MC Dropout, Deep Ensembles, Laplace approx., Evidential DL
- **Uncertainty types**: Aleatoric vs Epistemic; decompositions and visualizations
- **Calibration**: ECE/MCE/ACE, temperature scaling, isotonic regression
- **Conformal prediction**: class-conditional sets and *risk-controlling prediction sets (RCPS)*
- **Evaluation**: AUC/F1, risk-coverage curves, selective prediction, OOD detection (MSP, ODIN)
- **Reproducible pipelines**: Hydra configs, deterministic seeds, experiment tracking hooks
- **Data**: Example synthetic medical images + loaders for open datasets (e.g., HAM10000, CheXpert*)

> *Note*: Dataset downloaders reference public sources; user must accept their licenses.

---

## Quickstart
```bash
# (Option A) Minimal install
pip install -r requirements.txt

# (Option B) Conda
conda env create -f environment.yml && conda activate bayesdx

# Train a MC Dropout model on synthetic dermoscopy-like data
python -m bayesdx.train model=mc_dropout data=synthetic exp.name=mcdo_synth

# Evaluate with uncertainty metrics and calibration
python -m bayesdx.eval ckpt=outputs/mcdo_synth/checkpoints/best.pt data=synthetic eval.uq=all

# Generate risk-coverage and reliability diagrams
python -m bayesdx.viz ckpt=outputs/mcdo_synth/checkpoints/best.pt plots=[reliability,risk_coverage]
```

---

## Repository layout
```
BayesDx-UQ/
 ├─ bayesdx/                 # Source package
 │   ├─ __init__.py
 │   ├─ data.py              # Data loaders (synthetic + hooks to open datasets)
 │   ├─ models.py            # CNN backbones and Bayesian variants
 │   ├─ train.py             # Training loop with Hydra config
 │   ├─ eval.py              # Evaluation and metrics
 │   ├─ uncertainty.py       # MC Dropout, Ensembles, Laplace, Evidential
 │   ├─ calibration.py       # ECE/ACE, temp scaling, isotonic
 │   ├─ conformal.py         # Split-conformal + RCPS utilities
 │   ├─ metrics.py           # AUROC, F1, Brier, NLL, OOD
 │   ├─ viz.py               # Reliability/risk-coverage plots
 │   ├─ utils.py             # Seeds, checkpoints, logging helpers
 │   └─ hydra_main.py        # Entrypoint for Hydra-based CLI
 ├─ configs/                 # Hydra configs (YAML)
 ├─ notebooks/               # Jupyter demos
 ├─ data/                    # (empty) Placeholder with README
 ├─ scripts/                 # CLI helpers (shell & python)
 ├─ tests/                   # Pytest unit tests
 ├─ docs/                    # Paper outline + LaTeX skeleton
 ├─ ci/                      # CI configs (ruff/mypy/pytest)
 ├─ .gitignore
 ├─ LICENSE
 ├─ CITATION.cff
 ├─ CODE_OF_CONDUCT.md
 ├─ CONTRIBUTING.md
 ├─ SECURITY.md
 ├─ pyproject.toml
 ├─ setup.cfg
 ├─ requirements.txt
 ├─ environment.yml
 ├─ Makefile
 └─ Dockerfile
```
---

## Datasets (examples)
- **SyntheticDerm** (default): small synthetic patch images with injected lesions and noise for rapid iteration.
- **HAM10000** (dermoscopy)**: Add download path in `configs/data/ham10000.yaml` then run `scripts/fetch_ham10000.py`.
- **CheXpert** (chest X-rays)**: Update `configs/data/chexpert.yaml`, accept terms, and run `scripts/fetch_chexpert.py`.

> **Compliance**: Ensure proper data use and cite the original datasets.

---

## Reproducibility
- Deterministic seeds for PyTorch/CUDA (where supported)
- Config-driven experiments via **Hydra**
- Hash-stamped checkpoints and YAML snapshots of effective configs
- Complete metrics JSON + plots under `outputs/`

## Citing
If you use this repository, please cite via `CITATION.cff`.
