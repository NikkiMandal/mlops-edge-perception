"""
simulate_drift.py — Simulates data drift for demonstration
Applies image transformations to val images to simulate
real-world distribution shift (lighting change, camera shift etc.)
Then re-uploads to GCS as new production data.
Run drift_detect.py after this to see drift triggered.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from google.cloud import storage
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID         = "mlops-edge-perception"
BUCKET_NAME        = "mlops-edge-perception-bucket"
ORIGINAL_GCS_PATH  = "kitti/images/val"
DRIFTED_GCS_PATH   = "kitti/images/val_drifted"
LOCAL_ORIGINAL     = Path("/tmp/drift_sim/original")
LOCAL_DRIFTED      = Path("/tmp/drift_sim/drifted")
NUM_SAMPLES        = 100

# ── Drift Simulation Types ────────────────────────────────────
def apply_lighting_drift(img):
    """
    Simulate lighting change — overexposure.
    ELI5: Like someone turned up the brightness on the camera.
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(2.5)  # 2.5x brighter


def apply_fog_drift(img):
    """
    Simulate foggy weather conditions.
    ELI5: Add a white haze over the image like fog.
    """
    fog = Image.new("RGB", img.size, (200, 200, 200))
    return Image.blend(img, fog, alpha=0.5)


def apply_blur_drift(img):
    """
    Simulate camera focus loss or motion blur.
    ELI5: Like taking a photo while the car is moving fast.
    """
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def apply_combined_drift(img):
    """
    Apply multiple drift types together for stronger signal.
    This ensures Evidently detects drift reliably.
    """
    img = apply_lighting_drift(img)
    img = apply_fog_drift(img)
    img = apply_blur_drift(img)
    return img


# ── GCS Helpers ───────────────────────────────────────────────
def download_originals():
    """Download original val images from GCS."""
    print(f"\nDownloading {NUM_SAMPLES} original val images...")
    client  = storage.Client(project=PROJECT_ID)
    bucket  = client.bucket(BUCKET_NAME)
    blobs   = list(bucket.list_blobs(prefix=ORIGINAL_GCS_PATH))
    img_blobs = [b for b in blobs if b.name.endswith(".png")][:NUM_SAMPLES]

    LOCAL_ORIGINAL.mkdir(parents=True, exist_ok=True)
    for blob in tqdm(img_blobs):
        filename   = Path(blob.name).name
        local_path = LOCAL_ORIGINAL / filename
        if not local_path.exists():
            blob.download_to_filename(str(local_path))

    print(f"Downloaded {len(img_blobs)} images to {LOCAL_ORIGINAL}")
    return len(img_blobs)


def upload_drifted():
    """Upload drifted images to GCS as new production data."""
    print(f"\nUploading drifted images to gs://{BUCKET_NAME}/{DRIFTED_GCS_PATH}/...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    drifted_images = list(LOCAL_DRIFTED.glob("*.png"))
    for img_path in tqdm(drifted_images):
        gcs_path = f"{DRIFTED_GCS_PATH}/{img_path.name}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(img_path))

    print(f"Uploaded {len(drifted_images)} drifted images")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=== Data Drift Simulation ===")
    print("Simulating: lighting change + fog + blur")
    print(f"Original:   gs://{BUCKET_NAME}/{ORIGINAL_GCS_PATH}")
    print(f"Drifted:    gs://{BUCKET_NAME}/{DRIFTED_GCS_PATH}")

    # Step 1 — Download originals
    num_downloaded = download_originals()

    # Step 2 — Apply drift transformations
    print(f"\nApplying drift transformations to {num_downloaded} images...")
    LOCAL_DRIFTED.mkdir(parents=True, exist_ok=True)

    original_images = sorted(LOCAL_ORIGINAL.glob("*.png"))
    for img_path in tqdm(original_images):
        img     = Image.open(img_path).convert("RGB")
        drifted = apply_combined_drift(img)
        drifted.save(LOCAL_DRIFTED / img_path.name)

    print(f"Transformed {len(original_images)} images")

    # Step 3 — Show sample comparison stats
    sample_orig    = np.array(Image.open(original_images[0])) / 255.0
    sample_drifted = np.array(
        Image.open(LOCAL_DRIFTED / original_images[0].name)
    ) / 255.0

    print(f"\nSample image stats comparison:")
    print(f"  Original  — mean: {sample_orig.mean():.3f}, "
          f"std: {sample_orig.std():.3f}")
    print(f"  Drifted   — mean: {sample_drifted.mean():.3f}, "
          f"std: {sample_drifted.std():.3f}")
    print(f"  Mean shift: {abs(sample_drifted.mean() - sample_orig.mean()):.3f}")

    # Step 4 — Upload drifted images
    upload_drifted()

    # Step 5 — Save drift simulation config
    config = {
        "drift_type":       "combined (lighting + fog + blur)",
        "num_images":       len(original_images),
        "original_path":    f"gs://{BUCKET_NAME}/{ORIGINAL_GCS_PATH}",
        "drifted_path":     f"gs://{BUCKET_NAME}/{DRIFTED_GCS_PATH}",
        "transformations": [
            "brightness x2.5 (lighting change)",
            "fog blend alpha=0.5 (weather)",
            "gaussian blur radius=3 (motion/focus)"
        ]
    }
    config_path = Path("monitoring/drift_simulation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {config_path}")
    print("\n=== Done ===")
    print("Now run: python monitoring/drift_detect.py")
    print("Update NEW_DATA_GCS_PATH to 'kitti/images/val_drifted'")
    print("You should see drift score > 30% and retrain triggered")


if __name__ == "__main__":
    main()