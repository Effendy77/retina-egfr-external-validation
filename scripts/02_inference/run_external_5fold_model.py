import sys
import os
import argparse

import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(description="External 5-fold inference")

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--instance_csv", type=str, required=True)

parser.add_argument("--image_root", type=str, required=True)
parser.add_argument("--mask_root", type=str, required=True)

parser.add_argument("--retfound_weights", type=str, required=True)

parser.add_argument("--output_root", type=str, default="../results")
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = "/mnt/fastscratch/users/ebh77/kidney/code/retina-kidney-AI"
sys.path.insert(0, PROJECT_ROOT)

from egfr_ablation_v2.src.model.multimodal_fusion_ablation import MultimodalKidneyModelV2
from egfr_ablation_v2.src.datasets.multimodal_dataset_ablation import MultimodalKidneyDatasetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = args.model_path
MODEL_NAME = args.model_name

INSTANCE_CSV = args.instance_csv
IMAGE_ROOT = args.image_root
MASK_ROOT = args.mask_root

RETFOUND_WEIGHT_PATH = args.retfound_weights

OUTPUT_DIR = os.path.join(args.output_root, MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n===================================================")
print("External validation")
print("Model:", MODEL_NAME)
print("Model path:", MODEL_PATH)
print("Device:", DEVICE)
print("===================================================\n")

# ============================================================
# DATASET (MATCH TRAINING CONFIGURATION)
# ============================================================

dataset = MultimodalKidneyDatasetV2(
    csv_path=INSTANCE_CSV,
    image_root=IMAGE_ROOT,
    mask_root=MASK_ROOT,
    tabular_mode="baseline_plus_qrisk",
    use_retinal_features=True,
)

print("Dataset size:", len(dataset))
print("Tabular features used:", dataset.tabular_features)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

# ============================================================
# RUN 5-FOLD INFERENCE
# ============================================================

all_fold_predictions = []

for fold in range(5):

    print(f"\nRunning fold {fold}")

    # Initialize model each fold
    model = MultimodalKidneyModelV2(
        weight_path=RETFOUND_WEIGHT_PATH,
        num_tabular_features=10
    )

    model.to(DEVICE)
    model.eval()

    fold_weight_path = os.path.join(
        MODEL_PATH,
        f"fold{fold}",
        "best_model.pth"
    )

    print("Loading:", fold_weight_path)

    state_dict = torch.load(
        fold_weight_path,
        map_location=DEVICE
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    fold_preds = []

    with torch.no_grad():

        for batch in tqdm(loader):

            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)

            output = model(image, mask, tabular)

            fold_preds.extend(
                output.squeeze().cpu().numpy()
            )

    all_fold_predictions.append(fold_preds)

# ============================================================
# ENSEMBLE PREDICTIONS
# ============================================================

df_final = pd.read_csv(INSTANCE_CSV)

for fold in range(len(all_fold_predictions)):
    df_final[f"pred_fold{fold}"] = all_fold_predictions[fold]

df_final["pred_mean"] = np.mean(
    np.column_stack(all_fold_predictions),
    axis=1
)

# ============================================================
# RESCALE PREDICTIONS BACK TO REAL eGFR
# (training used egfr × 20)
# ============================================================

for fold in range(len(all_fold_predictions)):
    df_final[f"pred_fold{fold}"] = df_final[f"pred_fold{fold}"] / 20.0

df_final["pred_mean"] = df_final["pred_mean"] / 20.0

# ============================================================
# SAVE OUTPUT
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "instance1_predictions_ensemble.csv"
)

df_final.to_csv(
    output_file,
    index=False
)

print("\n===================================================")
print("External inference completed")
print("Saved to:", output_file)
print("===================================================\n")