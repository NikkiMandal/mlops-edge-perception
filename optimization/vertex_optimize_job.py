"""
vertex_optimize_job.py — Runs optimization benchmark on Vertex AI T4 GPU
Produces FP32 vs FP16 vs INT8 benchmark with real GPU numbers.
"""

import google.cloud.aiplatform as aip
from datetime import datetime

PROJECT_ID   = "mlops-edge-perception"
REGION       = "us-central1"
BUCKET_NAME  = "mlops-edge-perception-bucket"
BUCKET_URI   = f"gs://{BUCKET_NAME}"
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
JOB_NAME     = f"rtdetr-optimize-{TIMESTAMP}"
MACHINE_TYPE      = "n1-standard-4"
ACCELERATOR_TYPE  = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1
TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"

def submit_optimize_job():
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

    print(f"=== Submitting Optimization Job ===")
    print(f"Job: {JOB_NAME}")

    job = aip.CustomTrainingJob(
        display_name  = JOB_NAME,
        script_path   = "optimization/export_onnx.py",
        container_uri = TRAIN_IMAGE,
        requirements  = [
            "transformers==4.40.0",
            "torchvision>=0.15.0",
            "timm",
            "onnx",
            "onnxruntime-gpu",
            "google-cloud-storage",
            "Pillow",
            "numpy",
        ],
    )

    job.run(
        args              = [],
        replica_count     = 1,
        machine_type      = MACHINE_TYPE,
        accelerator_type  = ACCELERATOR_TYPE,
        accelerator_count = ACCELERATOR_COUNT,
        environment_variables = {
            "PJRT_DEVICE":  "GPU",
            "DISABLE_XLA":  "1",
            "USE_TORCH_XLA": "0",
        },
        sync = True,
    )

    print(f"=== Optimization Complete ===")
    print(f"Results at: gs://{BUCKET_NAME}/optimization/")

if __name__ == "__main__":
    submit_optimize_job()