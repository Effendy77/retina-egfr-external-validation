import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# CONFIG
# ============================================

RESULT_ROOT = "../results"

TRAIN_MEAN = 97.04167134460015
SCALE = 63.135428286714365

MODELS = {
    "T1_base": "T1",
    "I1_base": "I1",
    "IM1_base": "IM1",
    "IM4_qrisk_retfeat": "IM4"
}

# ============================================
# METRIC FUNCTION
# ============================================

def compute_metrics(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]

    return rmse, mae, r2, corr


# ============================================
# STORAGE
# ============================================

results = []

# ============================================
# LOOP THROUGH MODELS
# ============================================

for folder, model_type in MODELS.items():

    path = os.path.join(
        RESULT_ROOT,
        folder,
        "instance1_predictions_ensemble.csv"
    )

    print("\nLoading:", path)

    df = pd.read_csv(path)

    y_true = df.egfr.values

    # -------------------------
    # Reconstruction
    # -------------------------

    if model_type == "T1":

        y_pred = TRAIN_MEAN - df.pred_mean.values / SCALE

    else:

        y_pred = TRAIN_MEAN + df.pred_mean.values

    # -------------------------
    # Metrics
    # -------------------------

    rmse, mae, r2, corr = compute_metrics(y_true, y_pred)

    results.append({
        "model": folder,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Correlation": corr
    })

    # -------------------------
    # Scatter plot
    # -------------------------

    plt.figure(figsize=(6,6))

    plt.scatter(y_true, y_pred, alpha=0.4)

    plt.xlabel("True eGFR")
    plt.ylabel("Predicted eGFR")

    plt.title(f"{folder} : True vs Predicted")

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--"
    )

    plt.savefig(
        os.path.join(RESULT_ROOT, f"{folder}_scatter.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------
    # Bland–Altman
    # -------------------------

    mean = (y_true + y_pred) / 2
    diff = y_pred - y_true

    md = np.mean(diff)
    sd = np.std(diff)

    plt.figure(figsize=(6,6))

    plt.scatter(mean, diff, alpha=0.4)

    plt.axhline(md, linestyle="--")
    plt.axhline(md + 1.96*sd, linestyle="--")
    plt.axhline(md - 1.96*sd, linestyle="--")

    plt.xlabel("Mean eGFR")
    plt.ylabel("Prediction Error")

    plt.title(f"{folder} Bland–Altman")

    plt.savefig(
        os.path.join(RESULT_ROOT, f"{folder}_bland_altman.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================
# SAVE RESULTS
# ============================================

df_results = pd.DataFrame(results)

df_results = df_results.sort_values("RMSE")

print("\n==============================")
print("External Validation Results")
print("==============================")

print(df_results)

df_results.to_csv(
    os.path.join(RESULT_ROOT, "external_validation_metrics.csv"),
    index=False
)

print("\nSaved results to:")
print(os.path.join(RESULT_ROOT, "external_validation_metrics.csv"))