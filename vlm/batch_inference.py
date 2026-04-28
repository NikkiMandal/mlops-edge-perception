"""
batch_inference.py — Runs YOLOS + VLM on a folder of images.
Automatically triggers VLM when anomaly conditions are met.
"""
import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from google.cloud import storage
from anomaly_layer import analyze_scene, should_trigger_vlm

PROJECT_ID  = "mlops-edge-perception"
BUCKET_NAME = "mlops-edge-perception-bucket"
MODEL_PATH  = "/tmp/rtdetr_model"
CLASSES     = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
THRESHOLD   = 0.5

def download_model():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blobs  = [b for b in bucket.list_blobs(
        prefix="models/rtdetr_kitti/best_model"
    ) if not b.name.endswith("/")]
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel  = blob.name.split("best_model/")[-1]
        dest = Path(MODEL_PATH) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Model downloaded to {MODEL_PATH}")

def run_inference(image_path, model, processor):
    """Run YOLOS on one image, return detections."""
    img    = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        threshold    = THRESHOLD,
        target_sizes = target_sizes
    )[0]

    w, h = img.size
    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "label": CLASSES.get(label.item(), "Unknown"),
            "score": round(score.item(), 3),
            "box":   [x1/w, y1/h, x2/w, y2/h],
        })
    return detections

def main(images_dir="data/kitti_processed/images/val", max_images=10):
    print("=== Automatic YOLOS + VLM Inference Pipeline ===")

    # Load model
    print("Loading YOLOS model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model     = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
    model.eval()

    image_paths = sorted(Path(images_dir).glob("*.png"))[:max_images]
    print(f"Processing {len(image_paths)} images")

    results_log = []
    vlm_triggered_count = 0

    for img_path in image_paths:
        # Step 1 — YOLOS inference
        detections = run_inference(str(img_path), model, processor)

        # Step 2 — Check if VLM should trigger
        trigger, reason = should_trigger_vlm(detections)

        result = {
            "image":      img_path.name,
            "detections": len(detections),
            "vlm_triggered": trigger,
        }

        if trigger:
            vlm_triggered_count += 1
            # Step 3 — VLM reasoning (automatic)
            vlm_result = analyze_scene(str(img_path), detections)
            result["risk_level"]       = vlm_result.get("risk_level", "N/A")
            result["scene_description"] = vlm_result.get("scene_description", "")
            print(f"  {img_path.name}: VLM triggered ({reason}) "
                  f"→ Risk: {vlm_result.get('risk_level', 'N/A')}")
        else:
            print(f"  {img_path.name}: {len(detections)} detections, "
                  f"no anomaly")

        results_log.append(result)

    # Save log
    output_path = Path("vlm/outputs/batch_inference_log.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "total_images":    len(image_paths),
            "vlm_triggered":   vlm_triggered_count,
            "trigger_rate":    f"{vlm_triggered_count/len(image_paths)*100:.1f}%",
            "results":         results_log,
        }, f, indent=2)

    print(f"\n=== Done ===")
    print(f"VLM triggered on {vlm_triggered_count}/{len(image_paths)} images "
          f"({vlm_triggered_count/len(image_paths)*100:.1f}%)")
    print(f"Log saved to {output_path}")


if __name__ == "__main__":
    import sys
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "data/kitti_processed/images/val"
    main(images_dir)