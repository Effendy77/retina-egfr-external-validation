# Reproducibility Checklist

This document summarizes how to reproduce the external validation results reported in this repository.

## Code availability

All scripts required for the evaluation are available in the `scripts/` directory.

Key scripts:

- run_external_inference_clean.py
- evaluate_external_models.py
- plot_external_validation_figure.py
- ckd_stage_analysis.py
- ensemble_external_models.py

## Environment

Python environment used for this experiment:

egfr_v2_training_env.yml

Create environment:

conda env create -f egfr_v2_training_env.yml


## Dataset

The external validation dataset contains retinal fundus images and clinical variables.

Due to dataset governance restrictions, raw data cannot be shared publicly.

Researchers with appropriate access can reproduce the results using the provided scripts.

## Model types evaluated

Four model configurations were evaluated:

| Model | Inputs |
|------|------|
| T1_base | Clinical variables |
| I1_base | Retinal image |
| IM1_base | Image + baseline clinical variables |
| IM4_qrisk_retfeat | Image + clinical variables + retinal vascular features |

## Evaluation metrics

Model performance was evaluated using:

- RMSE
- MAE
- R²
- Pearson correlation

Clinical interpretation was further assessed using CKD stage classification.

## Hardware

Experiments were executed on a GPU-enabled high-performance computing cluster.

## Results

All evaluation outputs are available in the `results/` directory.

This includes:

- prediction outputs
- evaluation metrics
- scatter plots
- CKD stage agreement plots


