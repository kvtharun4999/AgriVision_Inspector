"""
AgriVision Turmeric Inspector - Model Training Pipeline
-------------------------------------------------------
Trains MobileNetV2 transfer learning model by combining:
1. Dataset A: "Turmeric Plant Disease Augmented Dataset/" (Local augmented dataset)
2. Dataset B: "dataset_hf/" (Hugging Face Project-AgML/turmeric_leaf_disease_classification)
3. Optional Negatives: "extra_negatives/" (Non-turmeric objects / leaves)

Workflow:
- Automatically scans all available datasets and merges corresponding classes.
- Stratified 70% Train, 15% Validation, and 15% Test split.
- Applies real-time data augmentation (flip, rotation, brightness).
- Computes class weights to handle dataset imbalance.
- Trains MobileNetV2 with frozen base (~12 epochs), then fine-tunes top layers.
- Evaluates on untouched test set and saves confusion matrix plot.
- Saves model/turmeric_model.keras, model/class_names.json, and model/turmeric_model.tflite.
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5

TARGET_CLASSES = {
    "Aphids Disease",
    "Dry Leaf",
    "Healthy Leaf",
    "Leaf Blotch",
    "Leaf Spot",
    "Rhizome Disease Root",
    "Rhizome Healthy Root",
    "Not turmeric leaf",
}

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_A_PATHS = [
    os.path.join(BASE_DIR, "Turmeric Plant Disease Augmented Dataset"),
    "/home/tom/Projects/mini_project/Turmeric Plant Disease Augmented Dataset",
]

DATASET_C_PATHS = [
    os.path.join(BASE_DIR, "Image Dataset for Turmeric Plant Leaf Disease Detection"),
    "/home/tom/Projects/mini_project/Image Dataset for Turmeric Plant Leaf Disease Detection",
]

DATASET_B_DIR = os.path.join(BASE_DIR, "dataset_hf")
EXTRA_NEGATIVES_DIR = os.path.join(BASE_DIR, "extra_negatives")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "turmeric_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
TFLITE_SAVE_PATH = os.path.join(MODEL_DIR, "turmeric_model.tflite")
ROOT_TFLITE_PATH = os.path.join(BASE_DIR, "model.tflite")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")


def normalize_class_name(name):
    """Normalize class folder variations into standardized target classes."""
    clean = name.replace("_", " ").strip()
    low = clean.lower()

    if "rhizome" in low and "disease" in low:
        return "Rhizome Disease Root"
    if "rhizome" in low and ("healthy" in low or "root" in low):
        return "Rhizome Healthy Root"
    if "dry" in low:
        return "Dry Leaf"
    if "aphid" in low:
        return "Aphids Disease"
    if "blotch" in low:
        return "Leaf Blotch"
    if "spot" in low:
        return "Leaf Spot"
    if "healthy" in low:
        return "Healthy Leaf"
    if "not turmeric" in low:
        return "Not turmeric leaf"

    return clean if clean in TARGET_CLASSES else None


def find_class_subdirs(root_dir):
    """Find all subdirectories mapping to target classes, supporting nested structures."""
    matched = []
    if not os.path.exists(root_dir):
        return matched

    root_norm = normalize_class_name(os.path.basename(root_dir))
    if root_norm:
        return [(root_norm, root_dir)]

    for current_root, dirs, _ in os.walk(root_dir):
        dirname = os.path.basename(current_root)
        norm = normalize_class_name(dirname)
        if norm and norm in TARGET_CLASSES and current_root != root_dir:
            matched.append((norm, current_root))
            dirs.clear()

    return matched


def collect_images_from_folder(folder_path, source_name, seen_hashes=None):
    """Recursively scan a directory and map unique images to normalized classes."""
    import hashlib
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images_by_class = {}

    if not os.path.exists(folder_path):
        return images_by_class

    class_subdirs = find_class_subdirs(folder_path)
    print(f"\nScanning {source_name} at: '{folder_path}'")

    for norm_class, full_sub_path in sorted(class_subdirs, key=lambda x: (x[0], x[1])):
        imgs = []
        for root, _, files in os.walk(full_sub_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    img_path = os.path.join(root, f)
                    if seen_hashes is not None:
                        try:
                            with open(img_path, "rb") as fh:
                                h = hashlib.md5(fh.read()).hexdigest()
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)
                        except Exception:
                            pass
                    imgs.append(img_path)

        if imgs:
            images_by_class.setdefault(norm_class, []).extend(imgs)
            rel_folder = os.path.relpath(full_sub_path, folder_path)
            print(f"  [{norm_class}] from '{rel_folder}': {len(imgs)} unique images")

    return images_by_class


def aggregate_all_datasets(custom_dirs=None):
    """Combines Dataset A + Dataset C (User dataset) + Dataset B (Hugging Face) + Custom dirs + Extra Negatives."""
    combined_by_class = {}
    seen_hashes = set()

    # 1. Dataset A: Local Augmented Dataset
    dataset_a_dir = next((p for p in DATASET_A_PATHS if os.path.exists(p) and os.path.isdir(p)), None)
    if dataset_a_dir:
        data_a = collect_images_from_folder(dataset_a_dir, "Dataset A (Turmeric Plant Disease Augmented)", seen_hashes)
        for cls, imgs in data_a.items():
            combined_by_class.setdefault(cls, []).extend(imgs)
    else:
        print("Notice: Primary Dataset A directory not found.")

    # 2. Dataset C: User's new dataset ("Image Dataset for Turmeric Plant Leaf Disease Detection")
    dataset_c_dir = next((p for p in DATASET_C_PATHS if os.path.exists(p) and os.path.isdir(p)), None)
    if dataset_c_dir:
        data_c = collect_images_from_folder(dataset_c_dir, "Dataset C (Turmeric Plant Leaf Disease Detection)", seen_hashes)
        for cls, imgs in data_c.items():
            combined_by_class.setdefault(cls, []).extend(imgs)
    else:
        print("Notice: Dataset C directory not found.")

    # 3. Dataset B: Hugging Face Dataset (dataset_hf/)
    if os.path.exists(DATASET_B_DIR):
        data_b = collect_images_from_folder(DATASET_B_DIR, "Dataset B (Hugging Face)", seen_hashes)
        for cls, imgs in data_b.items():
            combined_by_class.setdefault(cls, []).extend(imgs)

    # 4. Any custom dataset directories provided by user
    if custom_dirs:
        for idx, cdir in enumerate(custom_dirs):
            clean_dir = os.path.normpath(cdir)
            if os.path.exists(clean_dir):
                data_custom = collect_images_from_folder(clean_dir, f"Custom Dataset {idx+1} ({os.path.basename(clean_dir)})", seen_hashes)
                for cls, imgs in data_custom.items():
                    combined_by_class.setdefault(cls, []).extend(imgs)
            else:
                print(f"Warning: Provided custom dataset directory not found: '{cdir}'")

    # 5. Optional Extra Negatives (extra_negatives/)
    if os.path.exists(EXTRA_NEGATIVES_DIR):
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        neg_imgs = []
        for root, _, files in os.walk(EXTRA_NEGATIVES_DIR):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    neg_imgs.append(os.path.join(root, f))
        if neg_imgs:
            combined_by_class.setdefault("Not turmeric leaf", []).extend(neg_imgs)
            print(f"\nExtra Negatives: {len(neg_imgs)} images mapped to 'Not turmeric leaf'")

    if not combined_by_class:
        print("\n" + "=" * 60)
        print("ERROR: No images found across any dataset directories!")
        print("=" * 60)
        sys.exit(1)

    # Build final flat arrays
    sorted_classes = sorted(combined_by_class.keys())
    image_paths = []
    labels = []

    print("\n" + "=" * 60)
    print("FINAL COMBINED DATASET SUMMARY:")
    print("=" * 60)
    for idx, cls in enumerate(sorted_classes):
        imgs = combined_by_class[cls]
        print(f"  Class [{idx}] '{cls}': {len(imgs)} total images")
        for p in imgs:
            image_paths.append(p)
            labels.append(idx)

    print(f"\nTotal dataset size: {len(image_paths)} images across {len(sorted_classes)} classes.")
    return np.array(image_paths), np.array(labels), sorted_classes


def split_data(image_paths, labels):
    """Split 70% Train, 15% Validation, 15% Test with stratified distribution."""
    print("\nSplitting dataset into 70% Train / 15% Validation / 15% Test...")
    paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=SEED
    )

    val_fraction = 0.15 / 0.85
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_train_val, labels_train_val, test_size=val_fraction, stratify=labels_train_val, random_state=SEED
    )

    print(f"  Train:      {len(paths_train)} ({len(paths_train)/len(image_paths)*100:.1f}%)")
    print(f"  Validation: {len(paths_val)} ({len(paths_val)/len(image_paths)*100:.1f}%)")
    print(f"  Test:       {len(paths_test)} ({len(paths_test)/len(image_paths)*100:.1f}%)")

    return (paths_train, labels_train), (paths_val, labels_val), (paths_test, labels_test)


def load_and_preprocess_image(path, label, num_classes):
    """Decodes image, resizes to 224x224, and one-hot encodes label."""
    img_raw = tf.io.read_file(path)
    img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    img = tf.cast(img, tf.float32)
    one_hot = tf.one_hot(label, depth=num_classes)
    return img, one_hot


def build_pipeline(paths, labels, num_classes, is_training=False):
    """Builds optimized tf.data dataset pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(
        lambda p, l: load_and_preprocess_image(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def build_mobilenet_model(num_classes):
    """MobileNetV2 model architecture with data augmentation layers."""
    print("\nBuilding MobileNetV2 Transfer Learning Architecture...")

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.15, seed=SEED),
        layers.RandomBrightness(0.2, seed=SEED),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="turmeric_leaf_input")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="AgriVision_Turmeric_MobileNetV2")
    return model, base_model


def run_training(custom_datasets=None, initial_epochs=INITIAL_EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS, batch_size=BATCH_SIZE, dry_run=False):
    """Main training workflow."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    image_paths, labels, class_names = aggregate_all_datasets(custom_dirs=custom_datasets)
    num_classes = len(class_names)

    if dry_run:
        print("\n[DRY RUN] Dataset scan and validation complete. Skipping model training.")
        return

    (train_p, train_l), (val_p, val_l), (test_p, test_l) = split_data(image_paths, labels)

    # Balanced class weights for imbalance
    unique_cls = np.unique(train_l)
    weights = compute_class_weight(class_weight="balanced", classes=unique_cls, y=train_l)
    class_weights = {int(c): float(w) for c, w in zip(unique_cls, weights)}
    print(f"\nClass weights (imbalance handling): {class_weights}")

    train_ds = build_pipeline(train_p, train_l, num_classes, is_training=True)
    val_ds = build_pipeline(val_p, val_l, num_classes, is_training=False)
    test_ds = build_pipeline(test_p, test_l, num_classes, is_training=False)

    model, base_model = build_mobilenet_model(num_classes)

    # --- Phase 1: Train Classification Head ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"\n--- Phase 1: Training Classification Head ({initial_epochs} epochs) ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        ]
    )

    # --- Phase 2: Fine-Tuning Top Layers ---
    print(f"\n--- Phase 2: Fine-Tuning Top Layers ({fine_tune_epochs} epochs) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    total_epochs = initial_epochs + fine_tune_epochs
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]
    )

    # --- Evaluate on Untouched Test Set ---
    print("\n--- Evaluating Model on Untouched Test Set ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n>>> Final Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(test_l, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(test_l, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.title("AgriVision Turmeric Disease - Test Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved: {CONFUSION_MATRIX_PATH}")

    # Save artifacts
    print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print(f"Saving class names to {CLASS_NAMES_PATH}...")
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # Export TFLite
    print(f"\nConverting to TFLite ({TFLITE_SAVE_PATH})...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_bytes)
    print(f"Saved TFLite model ({len(tflite_bytes)/(1024*1024):.2f} MB)")

    import shutil
    shutil.copyfile(TFLITE_SAVE_PATH, ROOT_TFLITE_PATH)
    print(f"Copied TFLite model to root: {ROOT_TFLITE_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING FINISHED! Web app will now automatically use the trained model.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgriVision Turmeric Disease Detection Model Training")
    parser.add_argument("--dataset", "--extra-dataset", dest="extra_datasets", action="append", default=[],
                        help="Path(s) to additional dataset folders to include in training.")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS, help="Number of initial head training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and summarize all datasets without training.")
    args = parser.parse_args()

    run_training(
        custom_datasets=args.extra_datasets,
        initial_epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

