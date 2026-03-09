import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

RESULT_ROOT = "../results"

MODELS = {
    "T1_base": "Tabular",
    "I1_base": "Image",
    "IM1_base": "Image + Baseline",
    "IM4_qrisk_retfeat": "Full Multimodal"
}

for folder, label in MODELS.items():

    file_path = os.path.join(
        RESULT_ROOT,
        folder,
        "instance1_predictions_ensemble.csv"
    )

    if not os.path.exists(file_path):
        print("Skipping", folder)
        continue

    df = pd.read_csv(file_path)

    y_true = df["egfr"].values
    y_pred = df["pred_mean"].values / 20   # convert to real eGFR

    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.4
    )

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )

    plt.xlabel("True eGFR")
    plt.ylabel("Predicted eGFR")

    plt.title(f"{label}\nR² = {r2:.3f}")

    plt.tight_layout()

    output_file = os.path.join(
        RESULT_ROOT,
        f"scatter_{folder}.png"
    )

    plt.savefig(output_file, dpi=300)

    plt.close()

    print("Saved:", output_file)