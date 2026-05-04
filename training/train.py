"""
train.py — RT-DETR fine-tuning script for Vertex AI
Runs on Vertex AI GPU, reads data from GCS, saves model to GCS.
"""

import os
import json
import time
import torch
import argparse
import psutil
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)
from PIL import Image
from google.cloud import storage
import numpy as np

os.environ["PJRT_DEVICE"] = "GPU"
os.environ["XLA_FLAGS"] = "--xla_force_disable_all_pjrt_local_device_queries"
os.environ["DISABLE_XLA"] = "1"
os.environ["USE_TORCH_XLA"] = "0"


def log_memory(label=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {label}: {ram_mb:.0f} MB RAM used")

# ── Argument Parser ───────────────────────────────────────────
# Vertex AI passes hyperparameters as command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",        type=int,   default=20)
parser.add_argument("--batch_size",    type=int,   default=8)
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--bucket_name",   type=str,   default="mlops-edge-perception-bucket")
parser.add_argument("--gcs_data_path", type=str,   default="kitti")
parser.add_argument("--output_dir",    type=str,   default="/tmp/model_output")
args = parser.parse_args()

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID   = "mlops-edge-perception"
BUCKET_NAME  = args.bucket_name
#MODEL_NAME   = "PekingU/rtdetr_r50vd_coco_o365"
#MODEL_NAME = "facebook/detr-resnet-50"
MODEL_NAME = "hustvl/yolos-tiny"
CLASSES      = ["Car", "Pedestrian", "Cyclist"]
NUM_CLASSES  = len(CLASSES)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

# ── GCS Helper ────────────────────────────────────────────────
def download_from_gcs(bucket_name, gcs_prefix, local_dir):
    """Download all files from a GCS prefix to local directory."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blobs  = bucket.list_blobs(prefix=gcs_prefix)

    local_dir = Path(local_dir)
    downloaded = 0

    for blob in blobs:
        # Skip "directory" blobs
        if blob.name.endswith("/"):
            continue
        local_path = local_dir / Path(blob.name).relative_to(gcs_prefix)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded += 1

    print(f"Downloaded {downloaded} files from gs://{bucket_name}/{gcs_prefix}")


def upload_to_gcs(local_path, bucket_name, gcs_path):
    """Upload a single file to GCS."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded {local_path} → gs://{bucket_name}/{gcs_path}")


# ── Dataset Class ─────────────────────────────────────────────
class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI object detection.
    Reads images and YOLO-format label files from local disk.
    """
    def __init__(self, images_dir, labels_dir, processor):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.processor  = processor

        # Get all image files that have matching label files
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*.png")):
            label_path = self.labels_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                self.samples.append((img_path, label_path))

        print(f"Found {len(self.samples)} samples in {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h  = image.size

        # Load YOLO labels → convert to COCO format for RT-DETR
        boxes      = []
        class_ids  = []

        with open(label_path) as f:
            for line in f.readlines():
                parts     = line.strip().split()
                class_id  = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                # Convert YOLO (normalized) → absolute pixel coords
                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                x_max = (cx + bw / 2) * w
                y_max = (cy + bh / 2) * h

                boxes.append([x_min, y_min, x_max, y_max])
                class_ids.append(class_id)

        # Format for HuggingFace processor
        target = {
            "image_id":    idx,
            "annotations": [
                {
                    "image_id":    idx,
                    "category_id": class_ids[i],
                    "bbox": [
                        boxes[i][0],
                        boxes[i][1],
                        boxes[i][2] - boxes[i][0],  # width
                        boxes[i][3] - boxes[i][1],  # height
                    ],
                    "area": (boxes[i][2] - boxes[i][0]) *
                            (boxes[i][3] - boxes[i][1]),
                    "iscrowd": 0,
                }
                for i in range(len(boxes))
            ],
        }

        return image, target


_processor = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return _processor

def collate_fn(batch):
    """Custom collate to handle variable number of boxes per image."""
    processor = get_processor()
    images    = [item[0] for item in batch]
    targets   = [item[1] for item in batch]
    encoding  = processor(images=images, annotations=targets, return_tensors="pt")
    return encoding


# ── Training Loop ─────────────────────────────────────────────
def train():
    # Step 1 — Download data from GCS to local /tmp
    print("\n=== Downloading data from GCS ===")
    local_data = "/tmp/kitti"
    download_from_gcs(BUCKET_NAME, args.gcs_data_path, local_data)
    log_memory("after GCS download")

    # Step 2 — Create datasets
    print("\n=== Creating datasets ===")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_dataset = KITTIDataset(
        images_dir = f"{local_data}/images/train",
        labels_dir = f"{local_data}/labels/train",
        processor  = processor,
    )
    val_dataset = KITTIDataset(
        images_dir = f"{local_data}/images/val",
        labels_dir = f"{local_data}/labels/val",
        processor  = processor,
    )

    log_memory("after dataset creation")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    log_memory("after dataloader creation")

    # Step 3 — Load pretrained RT-DETR model
    print(f"\n=== Loading YOLOS model ({MODEL_NAME}) ===")
    id2label = {i: c for i, c in enumerate(CLASSES)}
    label2id = {c: i for i, c in enumerate(CLASSES)}

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label             = id2label,
        label2id             = label2id,
        ignore_mismatched_sizes = True,  # allows head replacement
    )
    model.to(DEVICE)
    log_memory("after model load")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Step 4 — Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Step 5 — Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")
    best_val_loss = float("inf")
    metrics_log   = []
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # — Train —
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            if i == 0:
                log_memory("after first batch")
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE) if "pixel_mask" in batch else None
            labels = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["labels"]]
            batch = {"pixel_values": pixel_values, "labels": labels}
            if pixel_mask is not None:
                batch["pixel_mask"] = pixel_mask
            outputs  = model(**batch)
            loss     = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_loss += loss.item()

            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # — Validate —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask = batch["pixel_mask"].to(DEVICE) if "pixel_mask" in batch else None
                labels = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch["labels"]]
                batch = {"pixel_values": pixel_values, "labels": labels}
                if pixel_mask is not None:
                    batch["pixel_mask"] = pixel_mask
                outputs  = model(**batch)
                val_loss += outputs.loss.item()

        val_loss  /= len(val_loader)
        epoch_time = time.time() - start_time
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.2e}")

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Log metrics
        metrics_log.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir / "best_model")
            processor.save_pretrained(output_dir / "best_model")
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

    # Step 6 — Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_loss": best_val_loss,
            "epochs":        args.epochs,
            "metrics_log":   metrics_log,
        }, f, indent=2)

    # Step 7 — Upload model to GCS
    print("\n=== Uploading model to GCS ===")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # ← move outside loop
    for file in (output_dir / "best_model").rglob("*"):
        if file.is_file():
            # Versioned path
            gcs_path = f"models/runs/{run_id}/{file.relative_to(output_dir)}"
            gcs_path = gcs_path.replace("\\", "/")
            upload_to_gcs(file, BUCKET_NAME, gcs_path)

            # Fixed path for optimize_component
            fixed_path = f"models/yolos_kitti/{file.relative_to(output_dir)}"
            fixed_path = fixed_path.replace("\\", "/")
            upload_to_gcs(file, BUCKET_NAME, fixed_path)

    #upload_to_gcs(metrics_path, BUCKET_NAME, "models/rtdetr_kitti/metrics.json") 
    upload_to_gcs(metrics_path, BUCKET_NAME, "models/yolos_kitti/metrics.json")  #changed model, hence changing path names

    print(f"\n=== Training Complete ===")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to gs://{BUCKET_NAME}/models/yolos_kitti/") #changed model, hence changing path names


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        print(f"\n=== TRAINING FAILED ===")
        print(f"Error: {e}")
        traceback.print_exc()
        raise