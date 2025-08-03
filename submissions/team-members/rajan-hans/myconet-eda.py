import os
from collections import Counter
from PIL import Image

import matplotlib.pyplot as plt

# Set your dataset directory
dataset_dir = 'C:/MyWork/Tech-Work/SDS/Data/myconet'  # Update this path

""" Visualize the dataset   
This script visualizes the dataset by displaying sample images from each class and plotting the class distribution.
It assumes the dataset is organized in subdirectories named after the classes.
"""


# List of classes
classes = ['h1', 'h2', 'h3', 'h5', 'h6']

# Gather image file paths and class labels
image_paths = []
labels = []

for cls in classes:
    class_dir = os.path.join(dataset_dir, cls)
    if not os.path.isdir(class_dir):
        print(f"Warning: Directory {class_dir} does not exist.")
        continue
    img_count = 0
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(class_dir, fname))
            labels.append(cls)
            img_count += 1
    print(f"Found {img_count} images for class '{cls}'")


# Check if we have images
if not image_paths:
    print("No images found in the dataset.")
    exit()          

# Visualize a few samples per class with class names on the left
samples_per_class = 5
fig, axes = plt.subplots(len(classes), samples_per_class + 1, figsize=(2 * (samples_per_class + 1), 2 * len(classes)))
fig.suptitle("Sample Images per Class", fontsize=16)

for row_idx, cls in enumerate(classes):
    # Get image paths for this class
    cls_imgs = [p for p, l in zip(image_paths, labels) if l == cls][:samples_per_class]
    # Show class name on the left
    axes[row_idx, 0].text(0.5, 0.5, cls, fontsize=12, ha='center', va='center')
    axes[row_idx, 0].axis('off')
    # Show images
    for col_idx, img_path in enumerate(cls_imgs):
        ax = axes[row_idx, col_idx + 1]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    # Hide unused axes if not enough images
    for col_idx in range(len(cls_imgs) + 1, samples_per_class + 1):
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.show()

# Draw bar chart of class distribution
class_counts = Counter(labels)
plt.figure(figsize=(10, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()


""" Perform stratified split (train / validation / test).
This section splits the dataset into training, validation, and test sets while maintaining the class distribution.
"""
import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Gather image paths and labels into a dataframe
print
split_base_dir = os.path.join(dataset_dir, 'fungi_split')
data = []
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        images = glob.glob(os.path.join(class_path, '*.jpg'))  # Adjust ext if needed
        for img_path in images:
            data.append({'filepath': img_path, 'label': class_name})

df = pd.DataFrame(data)

# Step 2: Stratified split: train/val/test (e.g., 70/20/10)
print("Splitting dataset into train, validation, and test sets...")
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=1/3, stratify=temp_df['label'], random_state=42
)
print('Train:', len(train_df), 'Val:', len(val_df), 'Test:', len(test_df))

# Step 3: Function to copy images to new folder structure
print ("Copying images to new folder structure...")
def copy_images(df, split_name, base_out_dir):
    for _, row in df.iterrows():
        label = row['label']
        img_path = row['filepath']
        out_dir = os.path.join(base_out_dir, split_name, label)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))

# Step 4: Perform the copying
copy_images(train_df, 'train', split_base_dir)
copy_images(val_df, 'val', split_base_dir)
copy_images(test_df, 'test', split_base_dir)

print("Done! Check the 'fungi_split' directory.")

# Optional: Display the distribution
print("\nClass distribution in each split:")
for split, df_ in zip(['Train', 'Val', 'Test'], [train_df, val_df, test_df]):
    print(f"{split}:\n", df_['label'].value_counts())

# Draw bar chart of class distribution in each split
splits = {'Train': train_df, 'Val': val_df, 'Test': test_df}
plt.figure(figsize=(15, 4))
for i, (split_name, split_df) in enumerate(splits.items()):
    plt.subplot(1, 3, i + 1)
    split_df['label'].value_counts().plot(kind='bar')
    plt.title(f'{split_name} Split')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





