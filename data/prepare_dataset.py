"""
prepare_dataset.py
Downloads KITTI subset from Hugging Face, converts to YOLO format,
uploads to GCS bucket for Vertex AI training.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from google.cloud import storage
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────
PROJECT_ID = "mlops-edge-perception"
BUCKET_NAME = "mlops-edge-perception-bucket"
GCS_DATA_PATH = "kitti"
LOCAL_DATA_DIR = Path("data/kitti_processed")
NUM_TRAIN = 400
NUM_VAL = 100

# Only these 3 classes matter for our project
CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# ── Label Conversion ─────────────────────────────────────────
def convert_to_yolo(sample, img_width, img_height):
    """
    Convert KITTI bounding boxes to YOLO format.
    KITTI: [x_min, y_min, x_max, y_max] in pixels
    YOLO:  [class, center_x, center_y, width, height] normalized 0-1
    """
    yolo_lines = []

    for obj in sample["label"]:
        # Get class name and skip if not in our 3 classes
        class_name = obj["type"]
        if class_name not in CLASSES:
            continue

        class_id = CLASSES[class_name]

        # Get bounding box in pixel coordinates
        x_min, y_min, x_max, y_max = obj["bbox"]

        # Convert to YOLO normalized format
        center_x = (x_min + x_max) / 2 / img_width
        center_y = (y_min + y_max) / 2 / img_height
        width    = (x_max - x_min) / img_width
        height   = (y_max - y_min) / img_height

        yolo_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return yolo_lines

# ── Folder Setup ─────────────────────────────────────────────
def create_local_dirs():
    """Create local directory structure for processed dataset."""
    dirs = [
        LOCAL_DATA_DIR / "images" / "train",
        LOCAL_DATA_DIR / "images" / "val",
        LOCAL_DATA_DIR / "labels" / "train",
        LOCAL_DATA_DIR / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure at {LOCAL_DATA_DIR}")

    # ── GCS Upload ───────────────────────────────────────────────
def upload_to_gcs(local_path, gcs_path):
    """Upload a single file to GCS bucket."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

    # ── Main Download Function ───────────────────────────────────
def download_and_process(split, num_samples, dataset):
    """
    Download images and convert labels for a given split.
    split: 'train' or 'val'
    num_samples: how many images to process
    dataset: the Hugging Face dataset object
    """
    print(f"\nProcessing {num_samples} samples for {split} split...")

    skipped = 0
    saved = 0

    for idx in tqdm(range(num_samples)):
        sample = dataset[idx]

        # Get image
        img = sample["image"]
        img_width, img_height = img.size

        # Convert labels to YOLO format
        yolo_lines = convert_to_yolo(sample, img_width, img_height)

        # Skip images with no relevant objects
        if len(yolo_lines) == 0:
            skipped += 1
            continue

        # Save image locally
        img_filename = f"{idx:06d}.png"
        img_path = LOCAL_DATA_DIR / "images" / split / img_filename
        img.save(img_path)

        # Save label file locally
        label_filename = f"{idx:06d}.txt"
        label_path = LOCAL_DATA_DIR / "labels" / split / label_filename
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        saved += 1

    print(f"  Saved: {saved} | Skipped (no objects): {skipped}")

    # ── Upload All Files to GCS ──────────────────────────────────
def upload_all_to_gcs():
    """Upload all processed files to GCS bucket."""
    print(f"\nUploading to gs://{BUCKET_NAME}/{GCS_DATA_PATH}/...")

    all_files = list(LOCAL_DATA_DIR.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]

    for local_path in tqdm(files_only):
        # Convert local path to GCS path
        # e.g. data/kitti_processed/images/train/000001.png
        #   -> kitti/images/train/000001.png
        relative = local_path.relative_to(LOCAL_DATA_DIR)
        gcs_path = f"{GCS_DATA_PATH}/{relative}"

        # Fix Windows backslashes → forward slashes for GCS
        gcs_path = gcs_path.replace("\\", "/")

        upload_to_gcs(str(local_path), gcs_path)

    print(f"Upload complete.")

    # ── Entry Point ──────────────────────────────────────────────
def main():
    print("=== KITTI Dataset Preparation ===")
    print(f"Train samples: {NUM_TRAIN}")
    print(f"Val samples:   {NUM_VAL}")
    print(f"Classes:       {list(CLASSES.keys())}")

    # Step 1 — Create local folders
    create_local_dirs()

    # Step 2 — Load dataset from Hugging Face
    print("\nLoading KITTI dataset from Hugging Face...")
    print("(This may take a few minutes on first download)")
    dataset = load_dataset("nateraw/kitti", split="train")

    # Step 3 — Process train split
    download_and_process("train", NUM_TRAIN, dataset)

    # Step 4 — Process val split (use images after train set)
    val_dataset = load_dataset("nateraw/kitti", split="train")
    val_subset = val_dataset.select(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL))
    download_and_process("val", NUM_VAL, val_subset)

    # Step 5 — Save dataset config file
    config = {
        "classes": list(CLASSES.keys()),
        "num_classes": len(CLASSES),
        "train_images": str(LOCAL_DATA_DIR / "images" / "train"),
        "val_images": str(LOCAL_DATA_DIR / "images" / "val"),
        "gcs_bucket": BUCKET_NAME,
        "gcs_path": GCS_DATA_PATH
    }
    config_path = LOCAL_DATA_DIR / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDataset config saved to {config_path}")

    # Step 6 — Upload everything to GCS
    upload_all_to_gcs()

    print("\n=== Done! ===")
    print(f"Data available at: gs://{BUCKET_NAME}/{GCS_DATA_PATH}/")


if __name__ == "__main__":
    main()