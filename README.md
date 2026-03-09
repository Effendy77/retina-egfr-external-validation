# Retinal Deep Learning for External Validation of eGFR Prediction
![Python](https://img.shields.io/badge/python-3.9-blue)
![Framework](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18926496.svg)](https://doi.org/10.5281/zenodo.18926496)

Deep learning framework for predicting kidney function (eGFR) from retinal fundus images with external validation using UK Biobank data.

---

### Paper
Effendy Bin Hashim et al.  
University of Liverpool  

### Repository
Implementation of the external validation pipeline used in the study.

---
<p align="center">
  <img src="docs/Retinal DL pipeline.png" width="800">
</p>

# Retina eGFR External Validation

This repository contains the code and results for external validation of deep learning models predicting kidney function (eGFR) from retinal fundus images and clinical variables.

## Models evaluated

Four ablation models were evaluated:

| Model | Inputs |
|------|------|
| T1_base | Baseline clinical variables |
| I1_base | Retinal image |
| IM1_base | Image + baseline clinical variables |
| IM4_qrisk_retfeat | Image + clinical variables + retinal vascular features |

## External dataset

Instance-1 cohort  
n = 6298 participants

## Evaluation metrics

Models were evaluated using:

- RMSE
- MAE
- R²
- Pearson correlation

## External validation results

| Model | RMSE | MAE | R² | Correlation |
|------|------|------|------|------|
| IM4_qrisk_retfeat | 12.19 | 9.60 | 0.077 | 0.388 |
| T1_base | 12.23 | 9.88 | 0.071 | 0.420 |
| IM1_base | 12.45 | 9.71 | 0.038 | 0.422 |
| I1_base | 12.64 | 9.75 | 0.008 | 0.422 |

## Repository structure
scripts/ analysis scripts
results/ model predictions and evaluation outputs


## Reproducing the evaluation

Run:
python scripts/evaluate_external_models.py
python scripts/plot_external_validation_figure.py
python scripts/ckd_stage_analysis.py


## Author

Effendy Bin Hashim  
University of Liverpool


