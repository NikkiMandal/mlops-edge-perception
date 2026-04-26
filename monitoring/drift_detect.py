"""
drift_detect.py — Data drift detection using Evidently AI
Compares new inference data against training baseline.
Triggers retraining if drift score exceeds threshold.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset
import pandas as pd
from google.cloud import storage
from PIL import Image

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID        = "mlops-edge-perception"
BUCKET_NAME       = "mlops-edge-perception-bucket"
BASELINE_GCS_PATH = "kitti/images/train"   # training images
NEW_DATA_GCS_PATH = "kitti/images/val"     # simulates new production data
LOCAL_BASELINE    = Path("/tmp/drift/baseline")
LOCAL_NEW_DATA    = Path("/tmp/drift/new_data")
REPORTS_DIR       = Path("monitoring/reports")
DRIFT_THRESHOLD   = 0.3   # if drift score > 30%, trigger retraining
NUM_SAMPLES       = 100    # images to sample for drift check

# ── GCS Download ──────────────────────────────────────────────
def download_sample_from_gcs(gcs_prefix, local_dir, num_samples):
    """Download a sample of images from GCS for drift analysis."""
    print(f"\nDownloading {num_samples} samples from gs://{BUCKET_NAME}/{gcs_prefix}")

    client  = storage.Client(project=PROJECT_ID)
    bucket  = client.bucket(BUCKET_NAME)
    blobs   = list(bucket.list_blobs(prefix=gcs_prefix))

    # Filter to image files only
    img_blobs = [b for b in blobs if b.name.endswith(".png")][:num_samples]

    if not img_blobs:
        raise FileNotFoundError(
            f"No images found at gs://{BUCKET_NAME}/{gcs_prefix}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    for blob in img_blobs:
        filename   = Path(blob.name).name
        local_path = local_dir / filename
        if not local_path.exists():
            blob.download_to_filename(str(local_path))

    print(f"Downloaded {len(img_blobs)} images to {local_dir}")
    return len(img_blobs)


# ── Feature Extraction ────────────────────────────────────────
def extract_image_features(image_dir):
    """
    Extract simple statistical features from images.
    ELI5: Instead of comparing raw pixels, we compare
    summary statistics — brightness, contrast, color balance.
    These are fast to compute and good drift indicators.
    """
    print(f"\nExtracting features from {image_dir}...")

    features = []
    image_paths = sorted(Path(image_dir).glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Extract per-channel statistics
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        features.append({
            # Brightness features
            "mean_brightness":  float(img_array.mean()),
            "std_brightness":   float(img_array.std()),

            # Per-channel means (color balance)
            "mean_r": float(r.mean()),
            "mean_g": float(g.mean()),
            "mean_b": float(b.mean()),

            # Per-channel std (contrast per channel)
            "std_r":  float(r.std()),
            "std_g":  float(g.std()),
            "std_b":  float(b.std()),

            # Image texture proxy
            "max_brightness":  float(img_array.max()),
            "min_brightness":  float(img_array.min()),

            # Histogram-based features
            "p25_brightness":  float(np.percentile(img_array, 25)),
            "p75_brightness":  float(np.percentile(img_array, 75)),
            "iqr_brightness":  float(
                np.percentile(img_array, 75) -
                np.percentile(img_array, 25)
            ),
        })

    df = pd.DataFrame(features)
    print(f"Extracted {len(df)} feature vectors "
          f"({len(df.columns)} features each)")
    return df


# ── Drift Detection ───────────────────────────────────────────
def detect_drift(baseline_df, new_data_df):
    """
    Run Evidently drift detection.
    Compares new data distribution against baseline.
    """
    print("\n=== Running Evidently Drift Detection ===")

    # Wrap dataframes in Evidently Dataset format
    baseline_dataset = Dataset.from_pandas(baseline_df)
    new_data_dataset  = Dataset.from_pandas(new_data_df)

    # Build and run report
    report = Report([DataDriftPreset()])
    my_eval = report.run(
        reference_data = baseline_dataset,
        current_data   = new_data_dataset,
    )

    # Extract results
    results       = my_eval.dict()
    drift_share   = results.get("drift_share", 0.0)
    drift_detected = drift_share > DRIFT_THRESHOLD

    print(f"Drift share:     {drift_share*100:.1f}%")
    print(f"Drift detected:  {drift_detected}")
    print(f"Threshold:       {DRIFT_THRESHOLD*100:.0f}%")

    return my_eval, drift_detected, drift_share


# ── Save Reports ──────────────────────────────────────────────
def save_reports(report, drift_detected, drift_share):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON summary
    summary = {
        "timestamp":      timestamp,
        "drift_detected": drift_detected,
        "drift_share":    drift_share,
        "threshold":      DRIFT_THRESHOLD,
        "action":         "retrain" if drift_share > DRIFT_THRESHOLD else "none",
    }
    json_path = REPORTS_DIR / f"drift_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved: {json_path}")

    return summary


# ── Upload Reports to GCS ─────────────────────────────────────
def upload_reports_to_gcs(summary):
    """Upload drift reports to GCS for logging."""
    print("\nUploading reports to GCS...")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    for report_file in REPORTS_DIR.glob("*"):
        if report_file.is_file():
            gcs_path = f"monitoring/drift_reports/{report_file.name}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(report_file))
            print(f"Uploaded: {report_file.name}")


# ── Retrain Trigger ───────────────────────────────────────────
def trigger_retraining(summary):
    """
    Trigger retraining pipeline if drift exceeds threshold.
    ELI5: If the alarm goes off, this calls the fire department.
    In production this would kick off a Vertex AI pipeline run.
    """
    if summary["action"] != "retrain":
        print("\n✓ No retraining needed — drift within acceptable range")
        return False

    print("\nDRIFT THRESHOLD EXCEEDED — Triggering retraining")
    print(f"   Drift share: {summary['drift_share']*100:.1f}% "
          f"> threshold {DRIFT_THRESHOLD*100:.0f}%")

    # Write trigger file — CI/CD pipeline watches for this
    trigger_path = Path("monitoring/retrain_trigger.json")
    trigger_data = {
        "trigger":    True,
        "timestamp":  summary["timestamp"],
        "reason":     "data_drift",
        "drift_share": summary["drift_share"],
    }
    with open(trigger_path, "w") as f:
        json.dump(trigger_data, f, indent=2)

    print(f"   Trigger file written: {trigger_path}")
    print("   In production: this would submit a new Vertex AI training job")
    return True


# ── Entry Point ───────────────────────────────────────────────
def main():
    print("=== Drift Detection Pipeline ===")
    print(f"Baseline: gs://{BUCKET_NAME}/{BASELINE_GCS_PATH}")
    print(f"New data: gs://{BUCKET_NAME}/{NEW_DATA_GCS_PATH}")
    print(f"Threshold: {DRIFT_THRESHOLD*100:.0f}%")

    # Step 1 — Download samples
    download_sample_from_gcs(
        BASELINE_GCS_PATH, LOCAL_BASELINE, NUM_SAMPLES
    )
    download_sample_from_gcs(
        NEW_DATA_GCS_PATH, LOCAL_NEW_DATA, NUM_SAMPLES
    )

    # Step 2 — Extract features
    baseline_df = extract_image_features(LOCAL_BASELINE)
    new_data_df = extract_image_features(LOCAL_NEW_DATA)

    # Step 3 — Detect drift
    report, drift_detected, drift_share = detect_drift(
        baseline_df, new_data_df
    )

    # Step 4 — Save reports
    summary = save_reports(report, drift_detected, drift_share)

    # Step 5 — Upload to GCS
    upload_reports_to_gcs(summary)

    # Step 6 — Trigger retraining if needed
    triggered = trigger_retraining(summary)

    # Final summary
    print("\n=== Drift Detection Complete ===")
    print(f"Drift detected:  {drift_detected}")
    print(f"Drift share:     {drift_share*100:.1f}%")
    print(f"Retrain triggered: {triggered}")


if __name__ == "__main__":
    main()