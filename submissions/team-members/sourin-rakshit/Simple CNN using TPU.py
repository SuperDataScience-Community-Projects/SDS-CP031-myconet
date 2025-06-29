# 0. Install dependencies

!pip install mlflow keras-tuner --quiet


# 1. Imports & Config

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import mlflow
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Configure MLflow experiment
mlflow.set_experiment("defungi_tpu_keras")

# Hyperparameters / paths
DATA_DIR         = Path("/kaggle/input/defungi")
IMG_SIZE         = 224
EPOCHS           = 20
PATIENCE         = 5
MAX_TRIALS       = 20
EXECUTIONS_PER_TRIAL = 3
CHECKPOINT_DIR   = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# 2. TPU SETUP with graceful fallback

try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"✅ Running on TPU: {resolver.master()}")
    # Optional: XLA JIT
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print(f"⚠️ TPU init failed: {e}")
    strategy = tf.distribute.get_strategy()
    print(f"✅ Running on default strategy: {strategy.__class__.__name__}")


# 3. Prepare Filepaths & Labels

records = []
for cls_dir in sorted(DATA_DIR.iterdir()):
    if cls_dir.is_dir():
        for img_path in cls_dir.glob("*"):
            records.append({
                "filepath": str(img_path),
                "label": cls_dir.name
            })
df = pd.DataFrame.from_records(records)
df["label_id"] = df["label"].astype("category").cat.codes
class_names = list(df["label"].astype("category").cat.categories)

# Stratified split: train (80%), val (10%), test (10%)
train_val, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label_id"], random_state=42
)
train_df, val_df = train_test_split(
    train_val, test_size=0.1111, stratify=train_val["label_id"], random_state=42
)
train_df, val_df, test_df = [
    d.reset_index(drop=True) for d in (train_df, val_df, test_df)
]

train_paths  = train_df["filepath"].values
train_labels = train_df["label_id"].values
val_paths    = val_df["filepath"].values
val_labels   = val_df["label_id"].values
test_paths   = test_df["filepath"].values
test_labels  = test_df["label_id"].values


# 4. Data Pipeline

def parse_fn(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_fn(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

def make_ds(paths, labels, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths))
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# 5. Model Builder for Keras Tuner

def build_model(hp):
    model = keras.Sequential()
    # <<< add this so model.variables exist before fit() >>>
    model.add(keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    
    # Tune number of convolutional blocks (2–4)
    for i in range(hp.Int("conv_blocks", 2, 4, default=3)):
        filters = hp.Choice(f"filters_{i}", [32, 64, 128], default=64)
        model.add(keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(keras.layers.MaxPooling2D())
    
    model.add(keras.layers.Flatten())
    # Tune dense units and dropout
    units = hp.Choice("dense_units", [128, 256, 512], default=256)
    model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dropout(
        hp.Float("dropout", 0.2, 0.6, step=0.1, default=0.4)
    ))
    model.add(keras.layers.Dense(len(class_names), activation="softmax"))

    # Log-uniform learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



# 6. Hyperparameter Search with Hyperband (using TPU strategy)

with strategy.scope():
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=EPOCHS,
        factor=3,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory="kt_tuner",
        project_name="defungi_cnn",
        distribution_strategy=strategy,
    )

stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

tuner.search(
    make_ds(train_paths, train_labels, batch_size=32, training=True),
    validation_data=make_ds(val_paths, val_labels, batch_size=32),
    epochs=EPOCHS,
    callbacks=[stop_early],
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# 7. Final Training & MLflow Logging

with mlflow.start_run():
    # Log chosen hyperparameters
    mlflow.log_params(best_hps.values)

    # Build & compile model with best_hps
    with strategy.scope():
        model = build_model(best_hps)

    # Callbacks
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    reducelr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2
    )
    cp_cb = keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "best_model.h5"),
        monitor="val_loss", save_best_only=True
    )

    # Train
    history = model.fit(
        make_ds(train_paths, train_labels, batch_size=128, training=True),
        validation_data=make_ds(val_paths, val_labels, batch_size=128),
        epochs=EPOCHS,
        callbacks=[earlystop_cb, reducelr_cb, cp_cb]
    )

    # Log per-epoch metrics
    for epoch in range(len(history.history["loss"])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(make_ds(test_paths, test_labels, batch_size=128))
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
