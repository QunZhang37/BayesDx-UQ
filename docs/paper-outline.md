# Paper Outline: Bayesian Deep Learning for Uncertainty Quantification in Medical Diagnosis

## 1. Introduction
- Clinical motivation for calibrated uncertainty
- Aleatoric vs epistemic: why both matter for triage and safety

## 2. Related Work
- Bayesian NNs, MC Dropout, Deep Ensembles, Laplace, EDL
- Calibration: temperature scaling, isotonic; conformal prediction; OOD detection

## 3. Methods
- Base CNN architecture (or Vision Transformer variants)
- Bayesian approximations
- Calibration + Conformal pipeline
- Uncertainty-aware triage strategy and risk-coverage objectives

## 4. Data
- SyntheticDerm (ours) + optional public datasets (HAM10000, CheXpert)
- Pre-processing, splits, ethical considerations

## 5. Experiments
- Metrics: AUROC, Brier, NLL, ECE, ACE
- Risk–coverage and selective prediction
- OOD detection with MSP/ODIN as baseline

## 6. Results
- Tables and plots
- Statistical tests and confidence intervals

## 7. Discussion
- Clinical implications, failure modes, limitations

## 8. Conclusion
- Summary, future work

## Appendix
- Hyperparameters, configs
