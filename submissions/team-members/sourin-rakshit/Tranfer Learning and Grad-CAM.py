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
