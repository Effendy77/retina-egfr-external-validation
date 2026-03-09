import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULT_ROOT = "../results"

TRAIN_MEAN = 97.04167134460015
SCALE = 63.135428286714365

# --------------------------------
# Load predictions
# --------------------------------

def load_model(folder):

    path = os.path.join(
        RESULT_ROOT,
        folder,
        "instance1_predictions_ensemble.csv"
    )

    df = pd.read_csv(path)

    if folder == "T1_base":
        pred = TRAIN_MEAN - df.pred_mean / SCALE
    else:
        pred = TRAIN_MEAN + df.pred_mean

    return df.egfr.values, pred.values


# --------------------------------
# Load all models
# --------------------------------

y_true, pred_T1 = load_model("T1_base")
_, pred_I1 = load_model("I1_base")
_, pred_IM1 = load_model("IM1_base")
_, pred_IM4 = load_model("IM4_qrisk_retfeat")


# --------------------------------
# Ensemble prediction
# --------------------------------

pred_ensemble = (
    pred_T1 +
    pred_I1 +
    pred_IM1 +
    pred_IM4
) / 4


# --------------------------------
# Metrics
# --------------------------------

rmse = np.sqrt(mean_squared_error(y_true, pred_ensemble))
mae = mean_absolute_error(y_true, pred_ensemble)
r2 = r2_score(y_true, pred_ensemble)
corr = np.corrcoef(y_true, pred_ensemble)[0,1]

print("\n==============================")
print("ENSEMBLE MODEL")
print("==============================")

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)
print("Correlation:", corr)


# --------------------------------
# Save predictions
# --------------------------------

df_out = pd.DataFrame({
    "true_egfr": y_true,
    "pred_ensemble": pred_ensemble
})

output = os.path.join(
    RESULT_ROOT,
    "ensemble_predictions.csv"
)

df_out.to_csv(output, index=False)

print("\nSaved:", output)