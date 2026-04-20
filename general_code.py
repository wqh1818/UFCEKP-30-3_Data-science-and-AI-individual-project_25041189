# This code is to run everytime when starting to train a model

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
zip_path = "/content/drive/MyDrive/individual projects/datasets/datasets_normalandmasks.zip"
extract_dir = "/content/local_dataset/"
base_path = "/content/local_dataset/datasets"

# Check if the dataset folder already exists to avoid re-extracting
if os.path.exists(base_path) and len(os.listdir(base_path)) > 0:
    print("Dataset already extracted. Skipping unzip process.")
else:
    print("Dataset not found locally. Preparing to extract...")

    # Create the local directory
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip the file directly into Colab's fast local storage
    print("Extracting normal and masks dataset....")
    !unzip -q "{zip_path}" -d "{extract_dir}"
    print("Extraction complete!")

print(f"Data is ready at: {base_path}")

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import seaborn as sns

import cv2


# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

IMG_SIZE = 224
BATCH_SIZE = 32

print("Done importing everything needed!!")

# Custom callback to print explicit epoch timings
class TimeHistory(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_time_start
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f" - Time: {epoch_time:.2f} seconds")
        print(f" - Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f}")
        print(f" - Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}\n")


# showcasing before training
def show_one_image_per_class(df, showcase_title):
    classes = df['label'].unique()

    plt.figure(figsize=(12,4))

    for i, class_name in enumerate(classes):
        # Get one sample from each class
        sample = df[df['label'] == class_name].sample(1, random_state=42).iloc[0]

        img = cv2.imread(sample['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(classes), i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

    plt.suptitle(f"{showcase_title}", fontsize=14)
    plt.tight_layout()
    plt.show()


# Show the prediction with image after training
def show_predictions(generator, model, class_labels, n=9):
    generator.reset()
    preds = model.predict(generator)

    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    paths = generator.filepaths

    plt.figure(figsize=(12,12))

    indices = np.random.choice(len(paths), n, replace=False)

    for i, idx in enumerate(indices):
        img = plt.imread(paths[idx])

        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.axis('off')

        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]

        color = "green" if true_label == pred_label else "red"

        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.suptitle("Model Predictions on Test Images", fontsize=14)
    plt.show()


def segment_image(img_path):
    class_name = img_path.split("/")[-2]
    # Map to mask folder
    mask_class = mask_classes[class_name]
    img_name = os.path.basename(img_path)
    mask_path = os.path.join(base_path, mask_class, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Image not found {img_path}")
        return None

    # Read mask in grayscale
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"ERROR: Mask not found {mask_path}")
        return None

    # Resize mask to match image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    # Normalize mask to 0–1
    mask = mask / 255.0
    # Expand mask to 3 channels
    mask = np.expand_dims(mask, axis=-1)
    segmented = img * mask

    return segmented.astype(np.uint8)


def process_split(df, split_name):
    seg_paths = []
    seg_labels = []
    seg_ids = []

    print(f"\nProcessing {split_name} set...")

    for i, row in df.iterrows():
        img_path = row["path"]
        label = row["label"]

        # Call function to segment image
        seg_img = segment_image(img_path)
        if seg_img is None:
            continue

        # Create output directory and save segmented image
        save_dir = os.path.join(segmented_base, split_name, label)
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, seg_img)

        seg_paths.append(save_path)
        seg_labels.append(label)
        seg_ids.append(i)

    # Create dataframe and save into csv file
    seg_df = pd.DataFrame({
        "id": seg_ids,
        "path": seg_paths,
        "label": seg_labels
    }).set_index("id")

    csv_name = f"{split_name}_split_segmented.csv"
    seg_df.to_csv(csv_name, index=False)

    print(f"Segmented {split_name}_df completed. Total {len(seg_df)} images saved.")
    print(f"Segmented {split_name}_df saved successfully!")
    return seg_df


# Augmentation for training (normal lung image)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

# No augmentation for validation and testing (normal lung image)
val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)


print("Done defining functions!!")

# Data preparation for normal lung images
print("Preparing normal lung images...")
classes = {
    "covid19_images": 0,
    "lung_opacity_images": 1,
    "normal_images": 2
}

# Collect all normal lungs image paths and labels from the fast local directory
image_paths = []
labels = []

for class_name in classes.keys():
    class_folder = os.path.join(base_path, class_name)

    # Check if the directory exists to catch zip extraction path issues
    if not os.path.exists(class_folder):
        print(f"WARNING: Path not found - {class_folder}")
        continue

    for img in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, img))
        labels.append(class_name)

df = pd.DataFrame({
    "path": image_paths,
    "label": labels
})

# Split: 70% Train, 15% Validation, 15% Test
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

print(f"Normal train samples: {len(train_df)}")
print(f"Normal validation samples: {len(val_df)}")
print(f"Normal test samples: {len(test_df)}")

# Save normal train_df to csv
train_df.to_csv("train_split_norm.csv", index=False)
print("Normal train_df saved successfully!")
# Save normal val_df to csv
val_df.to_csv("val_split_norm.csv", index=False)
print("Normal val_df saved successfully!")
# Save normal test_df to csv
test_df.to_csv("test_split_norm.csv", index=False)
print("Normal test_df saved successfully!")

# Call function to showcase the image
showcase_title = "Representative for Normal Chest X-ray Images per Class"
show_one_image_per_class(train_df, showcase_title)


# Data preparation for segmented lung images
print("\nPreparing segmented lung images...")
segmented_base = "/content/local_dataset/segmented"
os.makedirs(segmented_base, exist_ok=True)

# Mapping between image folders and mask folders
mask_classes = {
    "covid19_images": "covid19_mask_images",
    "lung_opacity_images": "lung_opacity_mask_images",
    "normal_images": "normal_mask_images"
}

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()
train_df.index.name = "id"
val_df.index.name = "id"
test_df.index.name = "id"

train_seg_df = process_split(train_df, "train")
val_seg_df   = process_split(val_df, "val")
test_seg_df  = process_split(test_df, "test")

# Produce the data efficiency training data
train_75_df = train_df.groupby("label", group_keys=False).sample(frac=0.75, random_state=SEED)
train_50_df = train_df.groupby("label", group_keys=False).sample(frac=0.50, random_state=SEED)
print("Preparing segmented 75% datasets...")
train_seg75_df = train_seg_df.loc[train_75_df.index]
print(f"Segmented train_seg75_df completed. Total {len(train_seg75_df)} images saved")
print("Preparing segmented 50% datasets...")
train_seg50_df = train_seg_df.loc[train_50_df.index]
print(f"Segmented train_seg50_df completed. Total {len(train_seg50_df)} images saved")

# Save reduced segmented dataset to csv
train_seg75_df.to_csv("train_split_segmented75.csv")
train_seg50_df.to_csv("train_split_segmented50.csv")
print("Segmented train_seg75_df saved successfully!")
print("Segmented train_seg50_df saved successfully!")
print(train_df.index.equals(train_seg_df.index))

# Call function to showcase the image
showcase_title = "Representative for Segmented Chest X-ray Images per Class"
show_one_image_per_class(train_seg_df, showcase_title)
print(train_75_df["label"].value_counts(normalize=True))
print(train_50_df["label"].value_counts(normalize=True))
