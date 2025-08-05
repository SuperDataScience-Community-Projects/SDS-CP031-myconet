# Data Augmentation
# This section implements data augmentation techniques to artificially expand the training dataset.

import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, load_img, array_to_img
)

# Original and augmented directories
# ========== USER SETTINGS ==========
dataset_dir = 'C:/MyWork/Tech-Work/SDS/Data/myconet' 
split_dir = os.path.join(dataset_dir, 'fungi_split')
aug_dir = os.path.join(dataset_dir, 'fungi_split_aug')
AUGS_PER_IMAGE = 2                          # Number of augmented images per original in train
img_format = 'jpg'                          # Save format: 'jpg' or 'png'
img_height, img_width = 224, 224            # Image size (adjust as needed)
# ===================================

# Augmentation pipeline for training
def contrast_jitter(image):
    factor = np.random.uniform(0.8, 1.2)
    mean = np.mean(image, axis=(0,1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 1)  # images are rescaled to 0-1

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    brightness_range=[0.8, 1.2],
    preprocessing_function=contrast_jitter
)

# For val/test: only rescale (no augmentation)
plain_datagen = ImageDataGenerator(rescale=1./255)

# Helper to ensure directory exists
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ---------- TRAIN: AUGMENT AND SAVE ----------
for class_name in os.listdir(os.path.join(split_dir, 'train')):
    class_in = os.path.join(split_dir, 'train', class_name)
    class_out = os.path.join(aug_dir, 'train', class_name)
    makedir(class_out)
    image_files = [f for f in os.listdir(class_in) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        img_path = os.path.join(class_in, image_file)
        img = load_img(img_path, target_size=(img_height, img_width))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        # Generate and save N augmentations
        aug_iter = train_datagen.flow(x, batch_size=1)
        for i in range(AUGS_PER_IMAGE):
            batch = next(aug_iter)
            aug_img = array_to_img(batch[0])
            aug_filename = f'aug_{os.path.splitext(image_file)[0]}_{i}.{img_format}'
            aug_img.save(os.path.join(class_out, aug_filename))
        # Optionally: Copy original as well (uncomment next line if desired)
        # shutil.copy2(img_path, os.path.join(class_out, image_file))

print('Train split: Augmentation and rescaling completed.')

# ---------- VAL/TEST: RESCALE & COPY ONLY ----------
for split in ['val', 'test']:
    split_in = os.path.join(split_dir, split)
    split_out = os.path.join(aug_dir, split)
    makedir(split_out)
    for class_name in os.listdir(split_in):
        class_in = os.path.join(split_in, class_name)
        class_out = os.path.join(split_out, class_name)
        makedir(class_out)
        image_files = [f for f in os.listdir(class_in) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            img_path = os.path.join(class_in, image_file)
            img = load_img(img_path, target_size=(img_height, img_width))
            x = img_to_array(img)
            x = x / 255.0  # Manual rescaling to 0-1
            img_rescaled = array_to_img(x)
            img_rescaled.save(os.path.join(class_out, image_file))

print('Validation/Test splits: Rescaling and copying originals completed.')
print(f"\nAll done. Augmented and rescaled dataset is in '{aug_dir}'")
