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
