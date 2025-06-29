# Block 1: Locate & Inspect Dataset
from pathlib import Path
import os

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 1) See which inputs are available in this session:
print("Available inputs under /kaggle/input:", os.listdir("/kaggle/input"))

# 2) Point to the Defungi folder (adjust name if necessary)
DATA_DIR = Path("/kaggle/input/defungi")      # or "/kaggle/input/joebeachcapital-defungi"
assert DATA_DIR.exists(), f"{DATA_DIR} not found!"
CLASS_DIRS = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
print("Found classes:", [d.name for d in CLASS_DIRS])

# Block 2: Build DataFrame of file paths & labels
records = []
for cls_dir in CLASS_DIRS:
    cls_name = cls_dir.name
    for img_path in cls_dir.glob("*"):
        records.append({
            "filepath": str(img_path),
            "class": cls_name
        })

df = pd.DataFrame.from_records(records)
print(f"Total images: {len(df)}")
df.head()

# Block 3: Class Distribution & Imbalance Check
class_counts = df["class"].value_counts().sort_index()
print(class_counts)

plt.figure(figsize=(8,5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Distribution (H1–H6)")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.show()

# Block 4: Missing / Broken Files Check
broken = []
for idx, row in df.iterrows():
    try:
        Image.open(row["filepath"]).verify()
    except Exception:
        broken.append(row["filepath"])

print(f"Broken or unreadable images: {len(broken)}")
if broken:
    print(broken[:10])

# Block 5: Visualize Sample Images per Class
def plot_samples(df, cls, n=6):
    imgs = df[df["class"]==cls]["filepath"].sample(n, random_state=42).tolist()
    fig, axes = plt.subplots(1, n, figsize=(n*2,2))
    for ax, img_p in zip(axes, imgs):
        ax.imshow(Image.open(img_p))
        ax.axis("off")
    fig.suptitle(f"Samples from class {cls}", y=1.1)
    plt.tight_layout()
    plt.show()

for cls in sorted(df["class"].unique()):
    plot_samples(df, cls, n=5)


# Block 6:Proportion by Class 
# Assuming `class_counts` from before
total = class_counts.sum()
imbalance = (class_counts / total).sort_index()
print(imbalance)

plt.figure(figsize=(6,4))
sns.barplot(x=imbalance.index, y=imbalance.values)
plt.title("Class Proportions (Imbalance)")
plt.ylabel("Proportion of dataset")
plt.show()

from sklearn.utils.class_weight import compute_class_weight
classes = sorted(df["class"].unique())
y = df["class"].values
cw = compute_class_weight("balanced", classes=classes, y=y)
class_weights = dict(zip(classes, cw))
print(class_weights)
