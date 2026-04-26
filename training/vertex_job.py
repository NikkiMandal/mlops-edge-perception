"""
vertex_job.py — Submits RT-DETR training job to Vertex AI
Run this from your laptop to kick off cloud training.
"""

import google.cloud.aiplatform as aip
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID   = "mlops-edge-perception"
REGION       = "us-central1"
BUCKET_NAME  = "mlops-edge-perception-bucket"
BUCKET_URI   = f"gs://{BUCKET_NAME}"

# Job settings
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
JOB_NAME     = f"rtdetr-kitti-{TIMESTAMP}"

# Machine settings — T4 GPU is cheapest on Vertex AI (~$0.35/hr)
MACHINE_TYPE      = "n1-standard-4"
ACCELERATOR_TYPE  = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

# Pre-built PyTorch container from Google
# This has PyTorch + CUDA already installed — no Docker needed
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"

# Training hyperparameters
ARGS = [
    "--epochs",        "20",
    "--batch_size",    "8",
    "--lr",            "1e-4",
    "--bucket_name",   BUCKET_NAME,
    "--gcs_data_path", "kitti",
    "--output_dir",    "/tmp/model_output",
]

def submit_training_job():
    # Initialize Vertex AI
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    print(f"Project:      {PROJECT_ID}")
    print(f"Region:       {REGION}")
    print(f"Bucket:       {BUCKET_URI}")
    print(f"Timestamp:    {TIMESTAMP}")
    print(f"Estimated cost: ~${0.35 * 1:.2f}/hr (T4 GPU)")

    print(f"=== Submitting Vertex AI Training Job ===")
    print(f"Job name:     {JOB_NAME}")
    print(f"Machine:      {MACHINE_TYPE} + {ACCELERATOR_TYPE}")
    print(f"Training for: 20 epochs on KITTI subset")

    # Create custom training job
    job = aip.CustomTrainingJob(
        display_name   = JOB_NAME,
        script_path    = "training/train.py",
        container_uri  = TRAIN_IMAGE,
        requirements   = [
            "transformers>=4.30.0",
            "google-cloud-storage",
            "Pillow",
            "tqdm",
        ],
    )

    # Submit job — this is non-blocking by default
    model = job.run(
        args             = ARGS,
        replica_count    = 1,
        machine_type     = MACHINE_TYPE,
        accelerator_type = ACCELERATOR_TYPE,
        accelerator_count= ACCELERATOR_COUNT,
        sync             = True,  # wait for job to complete
    )

    print(f"Job state:    {job.state}")
    print(f"Job resource: {job.resource_name}")
    print(f"Monitor at:   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")

    print(f"\n=== Job Complete ===")
    print(f"Check results at: gs://{BUCKET_NAME}/models/rtdetr_kitti/")


if __name__ == "__main__":
    submit_training_job()