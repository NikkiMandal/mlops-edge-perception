"""
autolabel.py — Automatic annotation using Grounding DINO
Stage 1 of the MLOps pipeline: label unlabeled images using
a foundation model instead of manual human annotation.
Saves labels in YOLO format compatible with training pipeline.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from google.cloud import storage
from tqdm import tqdm
from PIL import Image as PILImage

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID       = "mlops-edge-perception"
BUCKET_NAME      = "mlops-edge-perception-bucket"
UNLABELED_PATH   = "kitti/images/val"      # images without labels
OUTPUT_GCS_PATH  = "kitti/autolabeled"     # where to save auto-labels
LOCAL_IMAGES_DIR = Path("/tmp/autolabel/images")
LOCAL_OUTPUT_DIR = Path("/tmp/autolabel/outputs")

# Text prompts for Grounding DINO — describes what to detect
TEXT_PROMPTS = "car . pedestrian . cyclist ."

# Confidence threshold — only keep high confidence detections
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

# Class mapping
CLASSES = {"car": 0, "pedestrian": 1, "cyclist": 2}

# ── Download Images from GCS ──────────────────────────────────
def download_images(num_samples=50):
    """Download unlabeled images from GCS."""
    print(f"\nDownloading {num_samples} images from GCS...")
    client  = storage.Client(project=PROJECT_ID)
    bucket  = client.bucket(BUCKET_NAME)
    blobs   = list(bucket.list_blobs(prefix=UNLABELED_PATH))
    img_blobs = [b for b in blobs if b.name.endswith(".png")][:num_samples]

    LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for blob in tqdm(img_blobs):
        local_path = LOCAL_IMAGES_DIR / Path(blob.name).name
        if not local_path.exists():
            blob.download_to_filename(str(local_path))

    print(f"Downloaded {len(img_blobs)} images")
    return list(LOCAL_IMAGES_DIR.glob("*.png"))


# ── Load Grounding DINO ───────────────────────────────────────
def load_model():
    """Load Grounding DINO from HuggingFace."""
    print("\nLoading Grounding DINO (HuggingFace)...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()
    print("Grounding DINO loaded successfully")
    return model, processor


# ── Convert to YOLO Format ────────────────────────────────────
def boxes_to_yolo(boxes, labels, img_w, img_h):
    """
    Convert Grounding DINO output to YOLO format.
    Input boxes: [cx, cy, w, h] normalized (Grounding DINO format)
    Output: YOLO format strings
    """
    yolo_lines = []
    for box, label in zip(boxes, labels):
        # Get class ID
        label_lower = label.lower().strip()
        class_id = CLASSES.get(label_lower, -1)
        if class_id == -1:
            continue

        # Grounding DINO already outputs normalized cx,cy,w,h
        cx, cy, w, h = box
        if w <= 0 or h <= 0:
            continue

        yolo_lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )
    return yolo_lines


# ── Visualize Detections ──────────────────────────────────────
def visualize_detections(image_path, boxes, labels, scores, output_path):
    """Draw bounding boxes on image for verification."""
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = {"car": "red", "pedestrian": "green", "cyclist": "blue"}

    for box, label, score in zip(boxes, labels, scores):
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = colors.get(label.lower(), "white")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1-15)), f"{label} {score:.2f}", fill=color)

    img.save(str(output_path))


# ── Mock Predictions (fallback) ───────────────────────────────
def mock_predict(image_path):
    """
    Mock Grounding DINO predictions for testing pipeline.
    Returns realistic-looking detections without the model.
    """
    img = Image.open(image_path)
    w, h = img.size

    # Simulate 2-3 detections per image
    boxes  = [[0.25, 0.6, 0.3, 0.35],
              [0.65, 0.55, 0.25, 0.3]]
    labels = ["car", "car"]
    scores = [0.82, 0.76]

    return boxes, labels, scores


# ── Main Auto-labeling Pipeline ───────────────────────────────
def main():
    print("=== Grounding DINO Auto-labeling Pipeline ===")
    print(f"Text prompts: {TEXT_PROMPTS}")
    print(f"Box threshold: {BOX_THRESHOLD}")
    print(f"Source: gs://{BUCKET_NAME}/{UNLABELED_PATH}")
    print(f"Output: gs://{BUCKET_NAME}/{OUTPUT_GCS_PATH}")

    # Step 1 — Download images
    image_paths = download_images(num_samples=50)
    print(f"Processing {len(image_paths)} images")

    # Step 2 — Load model
    model, processor = load_model()
    use_mock = False

    if use_mock:
        print("\nUsing mock predictions (install groundingdino for real predictions)")

    # Step 3 — Auto-label each image
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir = LOCAL_OUTPUT_DIR / "labels"
    viz_dir    = LOCAL_OUTPUT_DIR / "visualizations"
    labels_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    stats = {"total": 0, "labeled": 0, "skipped": 0, "total_boxes": 0}

    for img_path in tqdm(image_paths, desc="Auto-labeling"):
        stats["total"] += 1
        try:
            if use_mock:
                boxes, labels, scores = mock_predict(str(img_path))
            else:
                # Real Grounding DINO inference
                img_pil = PILImage.open(str(img_path)).convert("RGB")
                inputs = processor(
                    images=img_pil,
                    text=TEXT_PROMPTS,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=BOX_THRESHOLD,
                    target_sizes=[img_pil.size[::-1]]
                )[0]

                boxes  = results["boxes"].tolist()
                scores = results["scores"].tolist()
                labels = results["labels"]

                # Convert boxes from [x1,y1,x2,y2] to normalized [cx,cy,w,h]
                w, h = img_pil.size
                norm_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    norm_boxes.append([cx, cy, bw, bh])
                boxes = norm_boxes

            img  = Image.open(img_path)
            w, h = img.size

            # Convert to YOLO format
            yolo_lines = boxes_to_yolo(boxes, labels, w, h)

            if not yolo_lines:
                stats["skipped"] += 1
                continue

            # Save label file
            label_file = labels_dir / img_path.with_suffix(".txt").name
            label_file.write_text("\n".join(yolo_lines))

            # Save visualization
            viz_path = viz_dir / img_path.name
            visualize_detections(img_path, boxes, labels, scores, viz_path)

            stats["labeled"]     += 1
            stats["total_boxes"] += len(yolo_lines)

        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            stats["skipped"] += 1

    # Step 4 — Print stats
    print(f"\n=== Auto-labeling Complete ===")
    print(f"Total images:    {stats['total']}")
    print(f"Labeled:         {stats['labeled']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Total boxes:     {stats['total_boxes']}")
    print(f"Avg boxes/image: {stats['total_boxes']/max(stats['labeled'],1):.1f}")

    # Step 5 — Upload to GCS
    print(f"\nUploading to gs://{BUCKET_NAME}/{OUTPUT_GCS_PATH}/...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    all_files = list(LOCAL_OUTPUT_DIR.rglob("*"))
    files     = [f for f in all_files if f.is_file()]

    for local_file in tqdm(files, desc="Uploading"):
        rel      = local_file.relative_to(LOCAL_OUTPUT_DIR)
        gcs_path = f"{OUTPUT_GCS_PATH}/{rel}".replace("\\", "/")
        bucket.blob(gcs_path).upload_from_filename(str(local_file))

    # Step 6 — Save summary
    summary = {
        "stats":        stats,
        "prompts":      TEXT_PROMPTS,
        "thresholds":   {"box": BOX_THRESHOLD, "text": TEXT_THRESHOLD},
        "source":       f"gs://{BUCKET_NAME}/{UNLABELED_PATH}",
        "output":       f"gs://{BUCKET_NAME}/{OUTPUT_GCS_PATH}",
        "model":        "GroundingDINO-SwinT" if not use_mock else "mock",
        "use_mock":     use_mock,
    }
    summary_path = LOCAL_OUTPUT_DIR / "autolabel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    bucket.blob(f"{OUTPUT_GCS_PATH}/summary.json").upload_from_filename(
        str(summary_path)
    )

    print(f"\nResults at: gs://{BUCKET_NAME}/{OUTPUT_GCS_PATH}/")
    print(f"Summary: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()