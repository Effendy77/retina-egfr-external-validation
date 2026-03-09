import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

RESULT_ROOT = "../results"

TRAIN_MEAN = 97.04167134460015
SCALE = 63.135428286714365

MODELS = {
    "T1_base": "Tabular",
    "I1_base": "Image",
    "IM1_base": "Image + Baseline",
    "IM4_qrisk_retfeat": "Full Multimodal"
}

def get_ckd_stage(e):

    if e >= 90:
        return "G1"
    elif e >= 60:
        return "G2"
    elif e >= 30:
        return "G3"
    else:
        return "G4"


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

    # --------------------------------
    # Correct reconstruction
    # --------------------------------
    if folder == "T1_base":
        y_pred = TRAIN_MEAN - df["pred_mean"].values / SCALE
    else:
        y_pred = TRAIN_MEAN + df["pred_mean"].values

    true_stage = [get_ckd_stage(x) for x in y_true]
    pred_stage = [get_ckd_stage(x) for x in y_pred]

    acc = accuracy_score(true_stage, pred_stage)
    kappa = cohen_kappa_score(true_stage, pred_stage)

    print("\n==============================")
    print(label)
    print("Accuracy:", acc)
    print("Cohen kappa:", kappa)

    stages = ["G1","G2","G3","G4"]

    cm = confusion_matrix(true_stage, pred_stage, labels=stages)

    fig = plt.figure(figsize=(6,5))

    plt.imshow(cm)

    plt.xticks(range(len(stages)), stages)
    plt.yticks(range(len(stages)), stages)

    plt.xlabel("Predicted Stage")
    plt.ylabel("True Stage")

    plt.title(f"{label}\nCKD Stage Agreement")

    for i in range(len(stages)):
        for j in range(len(stages)):
            plt.text(j,i,cm[i,j],
                     ha="center",
                     va="center")

    plt.tight_layout()

    output = os.path.join(
        RESULT_ROOT,
        f"ckd_stage_{folder}.png"
    )

    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)