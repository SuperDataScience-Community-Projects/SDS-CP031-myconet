# Welcome to the SuperDataScience Community Project!
Welcome to the **MycoNet - CNN-Based Microscopy Classifier for Dermatophyte & Yeast Detection** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**MycoNet** guides students through building a _from-scratch_ convolutional neural network (CNN) that classifies 9,114 pre-cropped fungal microscopy images (six classes H1 – H6) from the **DeFungi** dataset.  

The project emphasizes core deep-learning skills—data handling, augmentation, training loops, and evaluation—while offering an **optional** extension to experiment with transfer-learning backbones for those who wish to explore further.

Link to dataset: https://www.kaggle.com/datasets/joebeachcapital/defungi

## Project Objectives
### Dataset Exploration & Augmentation
- Load images, confirm class counts, and visualize a few samples per class.
- Create an **80 / 10 / 10** stratified split (train / validation / test).
- Implement lightweight augmentations (random flips, 90° rotations, brightness/contrast jitter) to boost generalization, using `torchvision.transforms` or `tf.keras.preprocessing`.

### Model Development & Training
1. **Baseline CNN (required)**
    - Build a simple architecture (e.g., Conv → ReLU → MaxPool × 3 – 4, followed by fully-connected layers and Softmax).
    - Use cross-entropy loss, Adam optimiser, learning-rate scheduling, early stopping.

2. **Evaluation**
    - Track training/validation accuracy and loss.
    - Report final **Top-1 Accuracy**, **Precision**, **Recall**, and **F1-score** on the held-out test set; include a confusion matrix.

3. **Explainability**
    - Generate **Grad-CAM** heat-maps for a few correctly and incorrectly classified images to illustrate model focus.

4. **Experiment Tracking using ML Flow**
	-  Conduct experiments with different models and track experiments using ML Flow

5. **(Optional) Transfer Learning**
    - Replace the baseline with a frozen backbone such as ResNet-18 or EfficientNet-B0 and fine-tune the head.
    - Compare metrics with the scratch CNN.

### Model Deployment
- **Streamlit Web App**
    - Drag-and-drop upload for a single microscope image.
    - Display predicted class, probability bar chart, and optional Grad-CAM overlay.

- Deploy to **Streamlit Community Cloud** (CPU-only is fine for inference).


## Technical Requirements

- **Core framework**: _either_ **PyTorch 2.x** _or_ **TensorFlow/Keras 2.x**
- **Data & augmentation**: `torchvision` or `tf.keras.preprocessing`, plus `Pillow`
- **Visualisation & evaluation**: `matplotlib`, `seaborn`, `scikit-learn` for confusion matrix
- **Explainability**: simple Grad-CAM utility (e.g., `torchcam` or custom Keras function)
- **Deployment**: `streamlit`
- **Environment**: Python 3.9+; GPU optional but recommended for training


## Workflow & Timeline

| Phase                                           | Core Tasks                                                                                                                                                   | Duration        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| **1 · Setup & EDA**                             | GitHub repo, virtual environment, class balance analysis, data split, augmentation pipeline                                                                  | **Week 1**      |
| **3 · Feature Engineering & Model Development** | Implement baseline CNN, train & validate, compute metrics, Grad-CAM visualisations; optional transfer-learning experiment, experiment tracking using ML Flow | **Weeks 2 – 4** |
| **4 · Deployment**                              | Build Streamlit app (upload → prediction + heat-map), deploy to cloud                                                                                        | **Week 5**      |
