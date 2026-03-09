import sys
import os

PROJECT_ROOT = "/mnt/fastscratch/users/ebh77/kidney/code/retina-kidney-AI/egfr_v2"
sys.path.append(PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.multimodal_fusion_v2 import MultimodalKidneyModelV2
from src.datasets.multimodal_dataset_v2 import MultimodalKidneyDatasetV2

# ============================================================
# CONFIG
# ============================================================

CANONICAL_MODEL_PATH = "/mnt/fastscratch/users/ebh77/kidney/outputs/egfr_ablation_v2/20251224_005309_1131860/IM4_qrisk_retfeat/seed_123"

RETFOUND_WEIGHT_PATH = "/mnt/fastscratch/users/ebh77/kidney/weights/retfound/RETFound_mae_natureCFP.pth"

INSTANCE1_CSV = "../data/instance1_external_v2_ready_CLEAN_MASK.csv"

IMAGE_ROOT = "/mnt/fastscratch/users/ebh77/instance1_automorph/raw"
MASK_ROOT  = "/mnt/fastscratch/users/ebh77/instance1_automorph/automorph_output/Results/M2/raw_binary"

OUTPUT_DIR = "../results"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD TRAINING TARGET MEAN (for inverse centering)
# ============================================================

TRAIN_CSV = "/mnt/fastscratch/users/ebh77/kidney/code/retina-kidney-AI/egfr_v2/data/multimodal_master_CLEANv2.csv"
train_df = pd.read_csv(TRAIN_CSV)
TRAIN_MEAN = train_df["egfr"].mean()

print(f"\nTraining eGFR mean (used for inverse transform): {TRAIN_MEAN:.4f}\n")



os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

dataset = MultimodalKidneyDatasetV2(
    csv_path=INSTANCE1_CSV,
    image_root=IMAGE_ROOT,
    mask_root=MASK_ROOT
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_fold_predictions = []

# ============================================================
# 5-FOLD INFERENCE
# ============================================================

for fold in range(5):

    print(f"\nRunning fold {fold} inference...")

    model = MultimodalKidneyModelV2(
        weight_path=RETFOUND_WEIGHT_PATH,
        num_tabular_features=10
    )

    fold_weight_path = os.path.join(
        CANONICAL_MODEL_PATH,
        f"fold{fold}",
        "best_model.pth"
    )
    print("Loading weights from:", fold_weight_path)

    state_dict = torch.load(fold_weight_path, map_location=DEVICE)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n=== STATE_DICT CHECK ===")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    
    print("Head weight mean:", model.head.weight.mean().item())
    print("Head weight std:", model.head.weight.std().item())
    print("Head bias:", model.head.bias.mean().item())
    
    
    
    print("========================\n")
    
       
    model.to(DEVICE)
    model.eval()

    fold_preds = []

    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            tabular = batch["tabular"].to(DEVICE)

            output = model(image, mask, tabular)
            fold_preds.extend(output.squeeze().cpu().numpy())

    all_fold_predictions.append(fold_preds)

# ============================================================
# ENSEMBLE
# ============================================================

df_final = pd.read_csv(INSTANCE1_CSV)

# Attach fold predictions
for fold in range(len(all_fold_predictions)):
    df_final[f"pred_fold{fold}"] = all_fold_predictions[fold]

# Compute ensemble mean (centered output)
df_final["pred_mean"] = np.mean(
    np.column_stack(all_fold_predictions),
    axis=1
)

# ============================================================
# INVERSE CENTERING (ADD BACK TRAINING MEAN)
# ============================================================

for fold in range(len(all_fold_predictions)):
    df_final[f"pred_fold{fold}"] += TRAIN_MEAN

df_final["pred_mean"] += TRAIN_MEAN

# Save
df_final.to_csv(
    os.path.join(OUTPUT_DIR, "instance1_predictions_ensemble.csv"),
    index=False
)

print("\nExternal inference complete.")