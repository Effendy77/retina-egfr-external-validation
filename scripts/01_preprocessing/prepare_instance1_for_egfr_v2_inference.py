import pandas as pd
from pathlib import Path

BASE_DIR = Path("/mnt/fastscratch/users/ebh77/instance1_automorph")
RAW_DIR = BASE_DIR / "raw"
MASK_DIR = BASE_DIR / "automorph_output/Results/M3"

INPUT_CSV = Path("../data/instance1_multimodal_ready.csv")
OUTPUT_CSV = Path("../data/instance1_external_v2_ready.csv")

df = pd.read_csv(INPUT_CSV)

# Rename columns to match training schema
df = df.rename(columns={
    "age_inst1": "age",
    "eGFR_true": "egfr",
    "New_Image_name": "clean_filename"
})

# Required training columns not present in external cohort
df["diabetes"] = 0
df["hypertension"] = 0
df["qrisk3"] = 0
df["dm_htn_combined"] = 0
df["mace"] = None
df["event_occurred"] = None

# Construct image and mask paths
df["fundus_image"] = df["clean_filename"].apply(lambda x: str(RAW_DIR / x))
df["vessel_mask"] = df["clean_filename"].apply(lambda x: str(MASK_DIR / x))
df["image_path"] = df["fundus_image"]

# Keep only columns used during training
final_cols = [
    "eid",
    "age",
    "sex",
    "diabetes",
    "hypertension",
    "egfr",
    "qrisk3",
    "mace",
    "event_occurred",
    "image_path",
    "dm_htn_combined",
    "clean_filename",
    "fundus_image",
    "vessel_mask",
    "fractal_dim",
    "vessel_density",
    "eccentricity",
    "mean_width_px",
]

df_final = df[final_cols]

df_final.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows:", len(df_final))
