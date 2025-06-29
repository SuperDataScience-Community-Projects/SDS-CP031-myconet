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
