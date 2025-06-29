import os
import json
import joblib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import optuna
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt


# 1. ARGS & CONFIG

DATA_DIR = Path("/kaggle/input/defungi")
assert DATA_DIR.exists(), f"{DATA_DIR} not found!"
print("Classes:", [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir()])

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="checkpoints", help="Where to save outputs")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--epochs",   type=int, default=20)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n_trials", type=int, default=50)
args, _ = parser.parse_known_args()

args.data_dir = str(DATA_DIR)
os.makedirs(args.ckpt_dir, exist_ok=True)

splits_fp      = os.path.join(args.ckpt_dir, "splits.joblib")
best_params_fp = os.path.join(args.ckpt_dir, "best_params.json")
final_model_fp = os.path.join(args.ckpt_dir, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

mlflow.set_experiment("defungi_pytorch_optuna")


# 2. GLOBAL TRANSFORMS

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2,0.2),
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# 3. STRATIFIED SPLITS (cache)

if os.path.exists(splits_fp):
    train_df, val_df, test_df = joblib.load(splits_fp)
    print("Loaded splits.")
else:
    records = []
    for cls in sorted(os.listdir(args.data_dir)):
        for fn in os.listdir(os.path.join(args.data_dir, cls)):
            records.append((os.path.join(args.data_dir, cls, fn), cls))
    df = pd.DataFrame(records, columns=["path","class"])
    classes = sorted(df["class"].unique())
    df["label"] = df["class"].map({c:i for i,c in enumerate(classes)})
    trv, test_df = train_test_split(df, test_size=0.1,
                                    stratify=df["label"], random_state=42)
    val_frac = 0.1/0.9
    train_df, val_df = train_test_split(trv, test_size=val_frac,
                                        stratify=trv["label"], random_state=42)
    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)
    joblib.dump((train_df, val_df, test_df), splits_fp)
    print("Saved splits.")

num_classes = len(train_df["class"].unique())


# 4. CLASS & SAMPLE WEIGHTS

y = train_df["label"].values
cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
weight_tensor = torch.tensor(cw, dtype=torch.float, device=DEVICE)
sample_weights = [cw[label] for label in y]


# 5. DATASET

class SplitDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.tf = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.loc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.tf:
            img = self.tf(img)
        return img, row["label"]


# 6. MODEL TEMPLATE

class CNN(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 128*(args.img_size//8)**2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat,256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 7. OPTUNA OBJECTIVE

def objective(trial):
    # suggest hyperparams
    lr      = trial.suggest_float("lr",1e-5,1e-2,log=True)
    bs      = trial.suggest_categorical("batch_size",[32,64,128])
    dropout = trial.suggest_float("dropout",0.2,0.6)

    # data loaders
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(SplitDataset(train_df, train_tf),
                              batch_size=bs, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(SplitDataset(val_df,   eval_tf),
                              batch_size=bs, shuffle=False, num_workers=4)

    # model, loss, optimizer, scheduler
    model     = CNN(dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        # training
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                preds.extend(out.argmax(1).cpu().tolist())
                trues.extend(labels.cpu().tolist())
        val_acc = accuracy_score(trues, preds)
        sched.step(1 - val_acc)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_val = max(best_val, val_acc)

    return best_val


# 8. HYPERPARAM SEARCH

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.n_trials)

with open(best_params_fp, "w") as f:
    json.dump(study.best_params, f, indent=2)
print("Best params:", study.best_params)


# 9. FINAL RETRAIN & SAVE (w/ Early Stopping & MLflow Logging)

with mlflow.start_run():
    mlflow.log_params(study.best_params)

    # rebuild model & optimizer & scheduler
    model     = CNN(study.best_params["dropout"]).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=study.best_params["lr"])
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # data loaders
    train_loader = DataLoader(SplitDataset(train_df, train_tf),
                              batch_size=study.best_params["batch_size"],
                              sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True),
                              num_workers=4)
    val_loader = DataLoader(SplitDataset(val_df, eval_tf),
                            batch_size=study.best_params["batch_size"],
                            shuffle=False, num_workers=4)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs+1):
        # train epoch
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # val epoch
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                running_val_loss += criterion(out, labels).item()
        avg_val_loss = running_val_loss / len(val_loader)

        # scheduler step
        sched.step(avg_val_loss)

        # log to MLflow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss",   avg_val_loss,   step=epoch)

        # early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), final_model_fp)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best model
    model.load_state_dict(torch.load(final_model_fp))

    # log the model artifact
    mlflow.pytorch.log_model(model, "model")


# 10. EVALUATE & PLOT

test_loader = DataLoader(SplitDataset(test_df, eval_tf),
                         batch_size=study.best_params["batch_size"],
                         shuffle=False, num_workers=4)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        preds.extend(out.argmax(1).cpu().tolist())
        trues.extend(labels.cpu().tolist())

acc = accuracy_score(trues, preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    trues, preds, average="weighted", zero_division=0
)

print(f"Test → acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
mlflow.log_metrics({
    "test_acc": acc,
    "test_precision": prec,
    "test_recall": rec,
    "test_f1": f1
})
