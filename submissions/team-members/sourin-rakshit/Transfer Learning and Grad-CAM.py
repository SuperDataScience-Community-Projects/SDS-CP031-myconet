!pip install mlflow captum --quiet
import os
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import optuna
import mlflow
import mlflow.pytorch


# optional: TorchCam for Grad-CAM
try:
    from torchcam.methods import GradCAM
    from torchcam.utils import overlay_mask
except ImportError:
    GradCAM = None
    overlay_mask = None



# ───────────────────────────────────────────────────────────────
# 1. ARGS & CONFIG
# ───────────────────────────────────────────────────────────────
DATA_DIR          = "/kaggle/input/defungi"
CKPT_DIR          = "checkpoints"
IMG_SIZE          = 224
EPOCHS            = 30
PATIENCE          = 5
N_TRIALS          = 20        # set to 0 to skip Optuna
FREEZE_BACKBONE   = False
NUM_CAMS          = 4         # Grad-CAM imgs per category
BATCH_SIZE_DEFAULT= 16
LR_DEFAULT        = 0.05
DROPOUT_DEFAULT   = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

print("Using device:", DEVICE)
mlflow.set_experiment("defungi_efficientnet_optuna")

# ───────────────────────────────────────────────────────────────
# 2. TRANSFORMS & SPLITS
# ───────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2,0.2,0.2,0.2),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

splits_fp = Path(CKPT_DIR) / "splits.joblib"
if splits_fp.exists():
    train_df, val_df, test_df = joblib.load(splits_fp)
    print("Loaded cached splits.")
else:
    records = []
    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        for fn in os.listdir(cls_dir):
            records.append((os.path.join(cls_dir, fn), cls))
    df = pd.DataFrame(records, columns=["path","class"])
    df["label"] = pd.Categorical(df["class"]).codes

    trv, test_df = train_test_split(
        df, test_size=0.1,
        stratify=df["label"], random_state=42
    )
    val_frac = 0.1 / 0.9
    train_df, val_df = train_test_split(
        trv, test_size=val_frac,
        stratify=trv["label"], random_state=42
    )
    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)
    joblib.dump((train_df, val_df, test_df), splits_fp)
    print("Saved splits cache.")

num_classes = train_df["label"].nunique()


# ───────────────────────────────────────────────────────────────
# 3. CLASS WEIGHTS
# ───────────────────────────────────────────────────────────────
cw = compute_class_weight(
    "balanced", classes=np.arange(num_classes),
    y=train_df["label"]
)
weight_tensor = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

# ───────────────────────────────────────────────────────────────
# 4. DATASET
# ───────────────────────────────────────────────────────────────
class FungiDS(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.df = df
        self.tf = tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), int(row["label"])



