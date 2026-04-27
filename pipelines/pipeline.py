"""
pipeline.py — Kubeflow Pipelines DAG on Vertex AI
Chain: data_prep → train → optimize → monitor
Compile: python pipelines/pipeline.py
Submit:  python -c "from pipelines.pipeline import submit_pipeline; submit_pipeline()"

Prerequisite — upload training script to GCS once before running:
  gsutil cp training/train.py gs://mlops-edge-perception-bucket/scripts/train.py
"""

from kfp import dsl, compiler
import google.cloud.aiplatform as aip

PROJECT_ID    = "mlops-edge-perception"
REGION        = "us-central1"
BUCKET_NAME   = "mlops-edge-perception-bucket"
BUCKET        = f"gs://{BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET}/pipeline_runs"

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"


# ── Component 1: Data Preparation ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-storage>=2.0.0",
        "datasets>=2.0.0",
        "Pillow>=9.0.0",
        "tqdm>=4.0.0",
    ],
)
def data_prep_component(
    project_id: str,
    bucket_name: str,
    gcs_data_path: str,
    num_train: int,
    num_val: int,
) -> str:
    """
    Download KITTI from HuggingFace, convert to YOLO format, upload to GCS.
    Skips if data already exists in GCS. Returns the GCS dataset URI.
    """
    import json
    from pathlib import Path
    from datasets import load_dataset
    from google.cloud import storage
    from tqdm import tqdm

    LOCAL_DIR = Path("/tmp/kitti_processed")
    CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    def check_exists() -> bool:
        client = storage.Client(project=project_id)
        blobs  = list(client.bucket(bucket_name).list_blobs(
            prefix=f"{gcs_data_path}/images/train", max_results=5
        ))
        return len(blobs) > 0

    def upload_file(local_path: Path, gcs_path: str) -> None:
        client = storage.Client(project=project_id)
        client.bucket(bucket_name).blob(gcs_path).upload_from_filename(
            str(local_path)
        )

    def convert_to_yolo(sample: dict, w: int, h: int) -> list:
        lines = []
        for obj in sample.get("label", []):
            cls_name = obj.get("type", "")
            if cls_name not in CLASSES:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            lines.append(
                f"{CLASSES[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )
        return lines

    if check_exists():
        print(f"Data already at gs://{bucket_name}/{gcs_data_path} — skipping")
        return f"gs://{bucket_name}/{gcs_data_path}"

    print("Loading KITTI from HuggingFace (nateraw/kitti)...")
    dataset = load_dataset("nateraw/kitti", split="train")

    splits = {
        "train": range(0, num_train),
        "val":   range(num_train, num_train + num_val),
    }

    for split, idx_range in splits.items():
        img_dir = LOCAL_DIR / "images" / split
        lbl_dir = LOCAL_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for idx in tqdm(idx_range, desc=f"Processing {split}"):
            sample = dataset[idx]
            img    = sample["image"].convert("RGB")
            w, h   = img.size

            yolo_lines = convert_to_yolo(sample, w, h)
            if not yolo_lines:
                continue

            stem = f"{idx:06d}"
            img.save(str(img_dir / f"{stem}.png"))
            (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))
            saved += 1

        print(f"  {split}: {saved} samples saved")

    config = {
        "classes": list(CLASSES.keys()),
        "num_classes": len(CLASSES),
        "gcs_path": f"gs://{bucket_name}/{gcs_data_path}",
    }
    cfg_path = LOCAL_DIR / "dataset_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))

    all_files = [f for f in LOCAL_DIR.rglob("*") if f.is_file()]
    print(f"Uploading {len(all_files)} files to GCS...")
    for local_file in tqdm(all_files, desc="Uploading"):
        rel      = local_file.relative_to(LOCAL_DIR)
        gcs_path = f"{gcs_data_path}/{rel}".replace("\\", "/")
        upload_file(local_file, gcs_path)

    gcs_uri = f"gs://{bucket_name}/{gcs_data_path}"
    print(f"Dataset ready at {gcs_uri}")
    return gcs_uri


# ── Component 2: Training ──────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
        "google-cloud-storage>=2.0.0",
    ],
)
def train_component(
    project_id:    str,
    region:        str,
    bucket_name:   str,
    gcs_data_path: str,
    epochs:        int,
    batch_size:    int,
    lr:            float,
    dataset_uri:   str,
) -> str:
    """
    Submit Vertex AI Custom Training Job running training/train.py on a T4 GPU.
    The training script is fetched from gs://{bucket_name}/scripts/train.py.
    Returns GCS path to the saved model.
    """
    import google.cloud.aiplatform as aip
    from datetime import datetime

    aip.init(project=project_id, location=region,
             staging_bucket=f"gs://{bucket_name}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    job_name  = f"rtdetr-kitti-{timestamp}"

    train_image = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest"
    )

    train_cmd = (
        "pip install -q transformers>=4.30.0 google-cloud-storage Pillow tqdm && "
        f"gsutil cp gs://{bucket_name}/scripts/train.py /tmp/train.py && "
        f"python /tmp/train.py "
        f"--epochs={epochs} "
        f"--batch_size={batch_size} "
        f"--lr={lr} "
        f"--bucket_name={bucket_name} "
        f"--gcs_data_path={gcs_data_path} "
        "--output_dir=/tmp/model_output"
    )

    worker_pool_specs = [{
        "machine_spec": {
            "machine_type":       "n1-standard-4",
            "accelerator_type":   "NVIDIA_TESLA_T4",
            "accelerator_count":  1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": train_image,
            "command":   ["bash", "-c"],
            "args":      [train_cmd],
        },
    }]

    print(f"Submitting job: {job_name}")
    job = aip.CustomJob(
        display_name      = job_name,
        worker_pool_specs = worker_pool_specs,
        staging_bucket    = f"gs://{bucket_name}",
    )
    job.run(sync=True)
    print(f"Job state: {job.state}")

    model_uri = f"gs://{bucket_name}/models/rtdetr_kitti"
    print(f"Model available at: {model_uri}")
    return model_uri


# ── Component 3: Optimization ──────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
        "google-cloud-storage>=2.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
    ],
)
def optimize_component(
    project_id:  str,
    bucket_name: str,
    model_uri:   str,
) -> str:
    """
    Download trained model, export to ONNX, benchmark FP32/ONNX/INT8-PTQ,
    upload results to GCS. Returns GCS path to optimization outputs.
    """
    import json
    import time
    import torch
    import numpy as np
    import onnxruntime as ort
    from pathlib import Path
    from transformers import AutoModelForObjectDetection
    from google.cloud import storage

    LOCAL_MODEL = Path("/tmp/rtdetr_model")
    OUTPUT_DIR  = Path("/tmp/optimize_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_GCS_PREFIX = "models/rtdetr_kitti/best_model"
    DUMMY_SHAPE      = (1, 3, 640, 640)
    WARMUP           = 10
    BENCH_RUNS       = 50

    # ── Download model ─────────────────────────────────────────────────────────
    print(f"Downloading model from gs://{bucket_name}/{MODEL_GCS_PREFIX}...")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs  = [b for b in bucket.list_blobs(prefix=MODEL_GCS_PREFIX)
              if not b.name.endswith("/")]

    if not blobs:
        raise RuntimeError(f"No model blobs at gs://{bucket_name}/{MODEL_GCS_PREFIX}")

    LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel  = blob.name[len(MODEL_GCS_PREFIX) + 1:]
        dest = LOCAL_MODEL / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Downloaded {len(blobs)} files")

    model = AutoModelForObjectDetection.from_pretrained(str(LOCAL_MODEL))
    model.eval()

    # ── Timing helper ──────────────────────────────────────────────────────────
    def bench_torch(m, inp, label: str):
        with torch.no_grad():
            for _ in range(WARMUP):
                m(pixel_values=inp)
        lats = []
        with torch.no_grad():
            for _ in range(BENCH_RUNS):
                t0 = time.perf_counter()
                m(pixel_values=inp)
                lats.append((time.perf_counter() - t0) * 1000)
        ms   = float(np.mean(lats))
        fps  = 1000.0 / ms
        size = sum(
            p.numel() * p.element_size() for p in m.parameters()
        ) / (1024 * 1024)
        print(f"  {label}: {ms:.1f}ms | {fps:.1f} FPS | {size:.1f}MB")
        return ms, fps, size

    results = {}
    dummy   = torch.randn(DUMMY_SHAPE)

    # FP32 baseline
    ms, fps, size = bench_torch(model, dummy, "PyTorch FP32")
    results["pytorch_fp32"] = {
        "format": "PyTorch FP32",
        "latency_ms": round(ms, 2),
        "fps":        round(fps, 1),
        "size_mb":    round(size, 1),
    }

    # ── Export ONNX ────────────────────────────────────────────────────────────
    onnx_path = OUTPUT_DIR / "rtdetr_fp32.onnx"
    print("\nExporting ONNX (opset 17)...")
    torch.onnx.export(
        model,
        {"pixel_values": dummy},
        str(onnx_path),
        opset_version       = 17,
        input_names         = ["pixel_values"],
        output_names        = ["logits", "pred_boxes"],
        dynamic_axes        = {
            "pixel_values": {0: "batch_size"},
            "logits":       {0: "batch_size"},
            "pred_boxes":   {0: "batch_size"},
        },
        do_constant_folding = True,
    )
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX saved: {onnx_size:.1f}MB")

    # Benchmark ONNX FP32
    sess     = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_np   = np.random.randn(*DUMMY_SHAPE).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"pixel_values": inp_np})
    lats = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"pixel_values": inp_np})
        lats.append((time.perf_counter() - t0) * 1000)
    onnx_ms  = float(np.mean(lats))
    onnx_fps = 1000.0 / onnx_ms
    print(f"  ONNX FP32: {onnx_ms:.1f}ms | {onnx_fps:.1f} FPS | {onnx_size:.1f}MB")
    results["onnx_fp32"] = {
        "format":     "ONNX FP32",
        "latency_ms": round(onnx_ms, 2),
        "fps":        round(onnx_fps, 1),
        "size_mb":    round(onnx_size, 1),
    }

    # ── INT8 PTQ ───────────────────────────────────────────────────────────────
    try:
        import torch.quantization as quant
        model_cpu  = model.cpu()
        model_cpu.eval()
        model_cpu.qconfig = quant.get_default_qconfig("fbgemm")
        prepared   = quant.prepare(model_cpu, inplace=False)
        calibration_dummy = torch.randn(DUMMY_SHAPE)
        with torch.no_grad():
            for _ in range(20):
                try:
                    prepared(pixel_values=calibration_dummy)
                except Exception:
                    pass
        model_ptq = quant.convert(prepared, inplace=False)
        ms, fps, size = bench_torch(model_ptq, dummy, "INT8 PTQ")
        results["int8_ptq"] = {
            "format":     "INT8 PTQ",
            "latency_ms": round(ms, 2),
            "fps":        round(fps, 1),
            "size_mb":    round(size, 1),
        }
    except Exception as e:
        print(f"PTQ skipped: {e}")
        results["int8_ptq"] = {"format": "INT8 PTQ", "error": str(e)}

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'Format':<20} {'Latency(ms)':<14} {'FPS':<10} {'Size(MB)'}")
    print("-" * 56)
    for r in results.values():
        if "error" in r:
            print(f"{r['format']:<20} SKIPPED: {r['error'][:30]}")
        else:
            print(f"{r['format']:<20} {r['latency_ms']:<14} "
                  f"{r['fps']:<10} {r['size_mb']}")

    # ── Upload results ─────────────────────────────────────────────────────────
    bench_path = OUTPUT_DIR / "benchmark_results.json"
    bench_path.write_text(json.dumps(results, indent=2))

    gcs_out_prefix = "models/rtdetr_kitti/optimized"
    for f in [onnx_path, bench_path]:
        blob = client.bucket(bucket_name).blob(f"{gcs_out_prefix}/{f.name}")
        blob.upload_from_filename(str(f))
        print(f"Uploaded gs://{bucket_name}/{gcs_out_prefix}/{f.name}")

    return f"gs://{bucket_name}/{gcs_out_prefix}"


# ── Component 4: Drift Monitoring ─────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "evidently==0.7.21",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "google-cloud-storage>=2.0.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.3.0",
    ],
)
def monitor_component(
    project_id:        str,
    bucket_name:       str,
    baseline_gcs_path: str,
    new_data_gcs_path: str,
    drift_threshold:   float,
    optimize_uri:      str,
) -> bool:
    """
    Detect image feature drift between baseline (train) and new data (val)
    using Evidently. Uploads a drift summary JSON to GCS.
    Returns True if retraining should be triggered.
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    from PIL import Image
    from google.cloud import storage
    from evidently import Report, Dataset
    from evidently.presets import DataDriftPreset

    LOCAL_BASE  = Path("/tmp/drift/baseline")
    LOCAL_NEW   = Path("/tmp/drift/new_data")
    REPORTS_DIR = Path("/tmp/drift/reports")
    NUM_SAMPLES = 100

    def download_images(gcs_prefix: str, local_dir: Path) -> None:
        client = storage.Client(project=project_id)
        blobs  = [b for b in client.bucket(bucket_name).list_blobs(
            prefix=gcs_prefix
        ) if b.name.endswith(".png")][:NUM_SAMPLES]
        if not blobs:
            raise FileNotFoundError(
                f"No images at gs://{bucket_name}/{gcs_prefix}"
            )
        local_dir.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            dest = local_dir / Path(blob.name).name
            if not dest.exists():
                blob.download_to_filename(str(dest))
        print(f"Downloaded {len(blobs)} images → {local_dir}")

    def extract_features(img_dir: Path) -> pd.DataFrame:
        rows = []
        for p in sorted(img_dir.glob("*.png")):
            arr     = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            rows.append({
                "mean_brightness": float(arr.mean()),
                "std_brightness":  float(arr.std()),
                "mean_r":          float(r.mean()),
                "mean_g":          float(g.mean()),
                "mean_b":          float(b.mean()),
                "std_r":           float(r.std()),
                "std_g":           float(g.std()),
                "std_b":           float(b.std()),
                "p25":             float(np.percentile(arr, 25)),
                "p75":             float(np.percentile(arr, 75)),
                "iqr":             float(
                    np.percentile(arr, 75) - np.percentile(arr, 25)
                ),
            })
        df = pd.DataFrame(rows)
        print(f"  Extracted {len(df)} feature vectors from {img_dir.name}")
        return df

    # Download samples
    download_images(baseline_gcs_path, LOCAL_BASE)
    download_images(new_data_gcs_path, LOCAL_NEW)

    # Extract features
    baseline_df = extract_features(LOCAL_BASE)
    new_data_df = extract_features(LOCAL_NEW)

    # Run Evidently drift report
    print("\nRunning Evidently drift detection...")
    report = Report([DataDriftPreset()])
    result = report.run(
        reference_data=Dataset.from_pandas(baseline_df),
        current_data=Dataset.from_pandas(new_data_df),
    )
    res_dict    = result.dict()
    drift_share = float(res_dict.get("drift_share", 0.0))
    retrain     = drift_share > drift_threshold

    print(f"Drift share:    {drift_share * 100:.1f}%")
    print(f"Threshold:      {drift_threshold * 100:.0f}%")
    print(f"Retrain needed: {retrain}")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp":      timestamp,
        "drift_share":    drift_share,
        "drift_detected": retrain,
        "threshold":      drift_threshold,
        "action":         "retrain" if retrain else "none",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / f"drift_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    client     = storage.Client(project=project_id)
    gcs_report = f"monitoring/drift_reports/{summary_path.name}"
    client.bucket(bucket_name).blob(gcs_report).upload_from_filename(
        str(summary_path)
    )
    print(f"Report → gs://{bucket_name}/{gcs_report}")

    return retrain


# ── Pipeline DAG ───────────────────────────────────────────────────────────────
@dsl.pipeline(
    name        = "kitti-rtdetr-pipeline",
    description = "RT-DETR KITTI: data_prep -> train -> optimize -> monitor",
)
def kitti_pipeline(
    project_id:        str   = PROJECT_ID,
    region:            str   = REGION,
    bucket_name:       str   = BUCKET_NAME,
    gcs_data_path:     str   = "kitti",
    num_train:         int   = 400,
    num_val:           int   = 100,
    epochs:            int   = 20,
    batch_size:        int   = 8,
    lr:                float = 1e-4,
    drift_threshold:   float = 0.3,
    baseline_gcs_path: str   = "kitti/images/train",
    new_data_gcs_path: str   = "kitti/images/val",
) -> None:

    data_task = data_prep_component(
        project_id    = project_id,
        bucket_name   = bucket_name,
        gcs_data_path = gcs_data_path,
        num_train     = num_train,
        num_val       = num_val,
    )

    train_task = train_component(
        project_id    = project_id,
        region        = region,
        bucket_name   = bucket_name,
        gcs_data_path = gcs_data_path,
        epochs        = epochs,
        batch_size    = batch_size,
        lr            = lr,
        dataset_uri   = data_task.output,
    )

    optimize_task = optimize_component(
        project_id  = project_id,
        bucket_name = bucket_name,
        model_uri   = train_task.output,
    )

    monitor_component(
        project_id        = project_id,
        bucket_name       = bucket_name,
        baseline_gcs_path = baseline_gcs_path,
        new_data_gcs_path = new_data_gcs_path,
        drift_threshold   = drift_threshold,
        optimize_uri      = optimize_task.output,
    )


# ── Compile ────────────────────────────────────────────────────────────────────
def compile_pipeline(output_path: str = "pipelines/kitti_pipeline.yaml") -> None:
    compiler.Compiler().compile(
        pipeline_func = kitti_pipeline,
        package_path  = output_path,
    )
    print(f"Compiled -> {output_path}")


# ── Submit ─────────────────────────────────────────────────────────────────────
def submit_pipeline(
    compiled_path: str = "pipelines/kitti_pipeline.yaml",
) -> None:
    """Submit compiled pipeline to Vertex AI Pipelines."""
    from datetime import datetime

    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

    job = aip.PipelineJob(
        display_name   = f"kitti-rtdetr-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        template_path  = compiled_path,
        pipeline_root  = PIPELINE_ROOT,
        enable_caching = True,
    )
    job.submit()
    print(f"Submitted: {job.resource_name}")
    print(
        "Monitor: https://console.cloud.google.com/vertex-ai/pipelines"
        f"?project={PROJECT_ID}"
    )


if __name__ == "__main__":
    compile_pipeline()
