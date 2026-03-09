import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# CONFIG
# ============================================================

RESULT_ROOT = "../results"

MODELS = {
    "T1_base": "T1_base",
    "I1_base": "I1_base",
    "IM1_base": "IM1_base",
    "IM4_qrisk_retfeat": "IM4_qrisk_retfeat"
}

TARGET_COLUMN = "egfr"
PRED_COLUMN = "pred_mean"

# ============================================================
# METRIC FUNCTION
# ============================================================

def compute_metrics(df):

    y_true = df[TARGET_COLUMN].values
    y_pred = df[PRED_COLUMN].values   # ← NO SCALING

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


# ============================================================
# LOOP THROUGH MODELS
# ============================================================

results = []

for folder, model_name in MODELS.items():

    file_path = os.path.join(
        RESULT_ROOT,
        folder,
        "instance1_predictions_ensemble.csv"
    )

    if not os.path.exists(file_path):
        print(f"Skipping {model_name} (file not found)")
        continue

    df = pd.read_csv(file_path)

    rmse, mae, r2 = compute_metrics(df)

    results.append({
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })


# ============================================================
# SAVE RESULTS
# ============================================================

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("rmse")

print("\nExternal Validation Results\n")
print(df_results)

df_results.to_csv(
    os.path.join(RESULT_ROOT, "external_validation_metrics.csv"),
    index=False
)

print("\nSaved to:")
print(os.path.join(RESULT_ROOT, "external_validation_metrics.csv"))