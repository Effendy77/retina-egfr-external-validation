import sys
import os
import argparse
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Clean external inference")

parser.add_argument("--model_path", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--instance_csv", required=True)

parser.add_argument("--image_root", required=True)
parser.add_argument("--mask_root", required=True)

parser.add_argument("--retfound_weights", required=True)

parser.add_argument("--tabular_mode", default="baseline")
parser.add_argument("--use_retinal_features", action="store_true")

parser.add_argument("--output_root", default="../results")
parser.add_argument("--batch_size", type=int, default=8)

args = parser.parse_args()

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = "/mnt/fastscratch/users/ebh77/kidney/code/retina-kidney-AI"
sys.path.append(PROJECT_ROOT)

from egfr_ablation_v2.src.model.multimodal_fusion_ablation import MultimodalKidneyModelV2
from egfr_ablation_v2.src.datasets.multimodal_dataset_ablation import MultimodalKidneyDatasetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = os.path.join(args.output_root, args.model_name)
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n========================================")
print("External Inference")
print("Model:", args.model_name)
print("========================================")

# ============================================================
# DATASET
# ============================================================

dataset = MultimodalKidneyDatasetV2(
    csv_path=args.instance_csv,
    image_root=args.image_root,
    mask_root=args.mask_root,
    tabular_mode=args.tabular_mode,
    use_retinal_features=args.use_retinal_features
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

print("Dataset size:", len(dataset))
print("Tabular features:", dataset.tabular_features)

# ============================================================
# LOAD MASTER CSV
# ============================================================

df_master = pd.read_csv(args.instance_csv)

# ============================================================
# DETECT TABULAR SIZE FROM CHECKPOINT
# ============================================================

def detect_tabular_features(checkpoint_path):

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "tabular_encoder.mlp.0.weight" in ckpt:
        return ckpt["tabular_encoder.mlp.0.weight"].shape[1]

    raise RuntimeError("Cannot detect tabular feature size")

# ============================================================
# RUN 5-FOLD INFERENCE
# ============================================================

predictions_all_folds = []

for fold in range(5):

    print(f"\nRunning fold {fold}")

    weight_path = os.path.join(
        args.model_path,
        f"fold{fold}",
        "best_model.pth"
    )

    print("Loading checkpoint:", weight_path)

    num_tabular_features = detect_tabular_features(weight_path)

    print("Detected tabular features:", num_tabular_features)

    model = MultimodalKidneyModelV2(
        weight_path=args.retfound_weights,
        num_tabular_features=num_tabular_features
    )

    state_dict = torch.load(weight_path, map_location=DEVICE)

    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    fold_preds = []

    with torch.no_grad():

        for batch in tqdm(loader):

            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)

            output = model(image, mask, tabular)

            preds = output.squeeze().cpu().numpy()

            fold_preds.extend(preds)

    # Save fold predictions
    df_fold = pd.DataFrame({
        "pred": fold_preds
    })

    df_fold.to_csv(
        os.path.join(MODEL_DIR, f"fold{fold}_predictions.csv"),
        index=False
    )

    predictions_all_folds.append(fold_preds)

# ============================================================
# BUILD ENSEMBLE (SAFE VERSION)
# ============================================================

predictions_all_folds = np.column_stack(predictions_all_folds)

df_master["pred_mean"] = predictions_all_folds.mean(axis=1)

# also save individual fold columns
for i in range(5):
    df_master[f"pred_fold{i}"] = predictions_all_folds[:, i]

# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

output_path = os.path.join(
    MODEL_DIR,
    "instance1_predictions_ensemble.csv"
)

df_master.to_csv(output_path, index=False)

print("\n========================================")
print("Saved ensemble predictions")
print(output_path)
print("========================================")