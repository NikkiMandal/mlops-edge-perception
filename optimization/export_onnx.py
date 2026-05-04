"""
export_onnx.py — Export trained RT-DETR to ONNX and benchmark formats
Produces the FP32 → FP16 → INT8 PTQ → INT8 QAT benchmark table.
Run this AFTER training is complete and model is in GCS.
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from google.cloud import storage
from PIL import Image

os.environ["PJRT_DEVICE"] = "GPU"
os.environ["DISABLE_XLA"] = "1"
os.environ["USE_TORCH_XLA"] = "0"

# ── Configuration ─────────────────────────────────────────────
PROJECT_ID      = "mlops-edge-perception"
BUCKET_NAME     = "mlops-edge-perception-bucket"
MODEL_GCS_PATH  = "models/yolos_kitti/best_model"  
LOCAL_MODEL_DIR = Path("/tmp/yolos_model")
OUTPUT_DIR      = Path("optimization/outputs")
CLASSES         = ["Car", "Pedestrian", "Cyclist"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy input for benchmarking — standard KITTI image size
DUMMY_INPUT_SIZE = (1, 3, 640, 640)
NUM_WARMUP_RUNS  = 10
NUM_BENCH_RUNS   = 100

print(f"Device: {DEVICE}")
print(f"Output dir: {OUTPUT_DIR}")

# ── GCS Download ──────────────────────────────────────────────
def download_model_from_gcs():
    """Download trained model from GCS to local disk."""
    print(f"\n=== Downloading model from GCS ===")
    print(f"Source: gs://{BUCKET_NAME}/{MODEL_GCS_PATH}")

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blobs  = list(bucket.list_blobs(prefix=MODEL_GCS_PATH))

    if not blobs:
        raise FileNotFoundError(
            f"No model found at gs://{BUCKET_NAME}/{MODEL_GCS_PATH}\n"
            f"Make sure training completed successfully first."
        )

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path   = blob.name.replace(MODEL_GCS_PATH + "/", "")
        local_path = LOCAL_MODEL_DIR / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"  Downloaded: {rel_path}")

    print(f"Model downloaded to {LOCAL_MODEL_DIR}")


# ── Load Model ────────────────────────────────────────────────
def load_model():
    """Load trained yolos model from local disk."""
    print(f"\n=== Loading model ===")
    processor = AutoImageProcessor.from_pretrained(str(LOCAL_MODEL_DIR))
    model     = AutoModelForObjectDetection.from_pretrained(str(LOCAL_MODEL_DIR))
    model.eval()
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {total_params:,} parameters")
    return model, processor


# ── Latency Benchmark ─────────────────────────────────────────
def benchmark_latency(model, input_tensor, label=""):
    """
    Measure inference latency with warmup runs.
    Returns mean latency in milliseconds and FPS.
    """
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    # Warmup — GPU needs a few runs to reach stable speed
    print(f"  Warming up ({NUM_WARMUP_RUNS} runs)...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            _ = model(pixel_values=input_tensor)

    # Benchmark
    print(f"  Benchmarking ({NUM_BENCH_RUNS} runs)...")
    latencies = []
    with torch.no_grad():
        for _ in range(NUM_BENCH_RUNS):
            if DEVICE.type == "cuda":
                # GPU timing needs synchronization
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(pixel_values=input_tensor)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                # CPU timing
                t0 = time.perf_counter()
                _ = model(pixel_values=input_tensor)
                latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = np.mean(latencies)
    p95_ms  = np.percentile(latencies, 95)
    fps     = 1000 / mean_ms

    print(f"  {label}: mean={mean_ms:.1f}ms | p95={p95_ms:.1f}ms | {fps:.1f} FPS")
    return mean_ms, p95_ms, fps


# ── ONNX Export ───────────────────────────────────────────────
def export_to_onnx(model, output_path):
    """
    Export PyTorch model to ONNX format.
    ONNX is the universal format — works with any runtime.
    """
    print(f"\n=== Exporting to ONNX ===")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(DUMMY_INPUT_SIZE).to(DEVICE)

    torch.onnx.export(
        model,
        {"pixel_values": dummy_input},
        str(output_path),
        opset_version    = 17,
        input_names      = ["pixel_values"],
        output_names     = ["logits", "pred_boxes"],
        dynamic_axes     = {
            "pixel_values": {0: "batch_size"},
            "logits":       {0: "batch_size"},
            "pred_boxes":   {0: "batch_size"},
        },
        do_constant_folding = True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({size_mb:.1f} MB)")
    return size_mb


# ── ONNX Runtime Benchmark ────────────────────────────────────
def benchmark_onnx(onnx_path, precision="fp32"):
    """
    Benchmark ONNX model with ONNX Runtime.
    Supports FP32 and FP16 precision.
    """
    import onnxruntime as ort

    print(f"\n=== Benchmarking ONNX {precision.upper()} ===")

    # Session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Use CUDA provider if available, else CPU
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda"
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=providers
    )

    # Create dummy input
    dummy = np.random.randn(*DUMMY_INPUT_SIZE).astype(np.float32)
    if precision == "fp16":
        dummy = dummy.astype(np.float16)

    # Warmup
    print(f"  Warming up ({NUM_WARMUP_RUNS} runs)...")
    for _ in range(NUM_WARMUP_RUNS):
        session.run(None, {"pixel_values": dummy})

    # Benchmark
    print(f"  Benchmarking ({NUM_BENCH_RUNS} runs)...")
    latencies = []
    for _ in range(NUM_BENCH_RUNS):
        t0 = time.perf_counter()
        session.run(None, {"pixel_values": dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = np.mean(latencies)
    p95_ms  = np.percentile(latencies, 95)
    fps     = 1000 / mean_ms

    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  ONNX {precision.upper()}: mean={mean_ms:.1f}ms | "
          f"p95={p95_ms:.1f}ms | {fps:.1f} FPS | {size_mb:.1f}MB")

    return mean_ms, p95_ms, fps, size_mb


# ── Quantization (PTQ) ────────────────────────────────────────
def quantize_ptq_onnx(onnx_path):
    """
    INT8 quantization using ONNX Runtime — works on CPU.
    More compatible with transformer architectures than PyTorch PTQ.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"\n=== Applying ONNX Runtime INT8 Dynamic Quantization ===")

    output_path = Path(str(onnx_path).replace("fp32", "int8"))

    quantize_dynamic(
        model_input  = str(onnx_path),
        model_output = str(output_path),
        weight_type  = QuantType.QInt8,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"INT8 model saved: {output_path} ({size_mb:.1f} MB)")
    return output_path, size_mb


# ── Main Benchmark Runner ─────────────────────────────────────
def run_benchmarks(model):
    """Run all benchmarks and produce the comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    dummy_input = torch.randn(DUMMY_INPUT_SIZE)

    # 1 — FP32 PyTorch baseline
    print("\n" + "="*50)
    print("BENCHMARK 1: PyTorch FP32 (baseline)")
    print("="*50)
    mean_ms, p95_ms, fps = benchmark_latency(model, dummy_input, "FP32")
    fp32_size = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 * 1024)
    results["pytorch_fp32"] = {
        "format": "PyTorch FP32",
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms":  round(p95_ms, 2),
        "fps":             round(fps, 1),
        "size_mb":         round(fp32_size, 1),
    }

    # 2 — FP16 PyTorch
    print("\n" + "="*50)
    print("BENCHMARK 2: PyTorch FP16")
    print("="*50)
    if DEVICE.type == "cuda":
        try:
            # Reload model fresh in FP16 to avoid mixed precision issues
            model_fp16 = AutoModelForObjectDetection.from_pretrained(
                str(LOCAL_MODEL_DIR),
                torch_dtype=torch.float16
            )
            model_fp16.eval()
            model_fp16.to(DEVICE)
            dummy_fp16 = torch.randn(DUMMY_INPUT_SIZE, dtype=torch.float16).to(DEVICE)
            mean_ms, p95_ms, fps = benchmark_latency(model_fp16, dummy_fp16, "FP16")
            fp16_size = fp32_size / 2
            results["pytorch_fp16"] = {
                "format":          "PyTorch FP16",
                "mean_latency_ms": round(mean_ms, 2),
                "p95_latency_ms":  round(p95_ms, 2),
                "fps":             round(fps, 1),
                "size_mb":         round(fp16_size, 1),
        }
        except Exception as e:
            print(f"  FP16 failed: {e}")
            results["pytorch_fp16"] = {"format": "PyTorch FP16", "note": str(e)[:50]}
    else:
        print("  Skipping FP16 — requires CUDA GPU")
        results["pytorch_fp16"] = {"format": "PyTorch FP16", "note": "Requires CUDA GPU"}

    # 3 — ONNX FP32
    print("\n" + "="*50)
    print("BENCHMARK 3: ONNX FP32")
    print("="*50)
    onnx_path = OUTPUT_DIR / "yolos_fp32.onnx"
    export_to_onnx(model, onnx_path)
    mean_ms, p95_ms, fps, size_mb = benchmark_onnx(onnx_path, "fp32")
    results["onnx_fp32"] = {
        "format":          "ONNX FP32",
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms":  round(p95_ms, 2),
        "fps":             round(fps, 1),
        "size_mb":         round(size_mb, 1),
    }

    # 4 — ONNX INT8 Dynamic Quantization
    print("\n" + "="*50)
    print("BENCHMARK 4: ONNX INT8 PTQ (Dynamic)")
    print("="*50)
    try:
        int8_path, int8_size = quantize_ptq_onnx(onnx_path)
        mean_ms, p95_ms, fps, _ = benchmark_onnx(str(int8_path), "fp32")
        results["int8_ptq"] = {
            "format":          "ONNX INT8 PTQ",
            "mean_latency_ms": round(mean_ms, 2),
            "p95_latency_ms":  round(p95_ms, 2),
            "fps":             round(fps, 1),
            "size_mb":         round(int8_size, 1),
        }
    except Exception as e:
        print(f"  INT8 failed: {e}")
        results["int8_ptq"] = {"format": "ONNX INT8 PTQ", "note": str(e)[:50]}

    return results


# ── Print Benchmark Table ─────────────────────────────────────
def print_benchmark_table(results):
    """Print results as a formatted table."""
    print("\n" + "="*70)
    print("BENCHMARK TABLE — YOLOS KITTI")
    print("="*70)
    print(f"{'Format':<20} {'Latency(ms)':<15} {'P95(ms)':<12} "
          f"{'FPS':<10} {'Size(MB)':<10}")
    print("-"*70)

    for key, r in results.items():
        if "note" in r:
            print(f"{r['format']:<20} {'N/A ('+r['note']+')'}")
            continue
        print(f"{r['format']:<20} "
              f"{r.get('mean_latency_ms', 'N/A'):<15} "
              f"{r.get('p95_latency_ms', 'N/A'):<12} "
              f"{r.get('fps', 'N/A'):<10} "
              f"{r.get('size_mb', 'N/A'):<10}")

    print("="*70)


# ── Save Results ──────────────────────────────────────────────
def save_results(results):
    """Save benchmark results to JSON for later use."""
    output_path = OUTPUT_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ── Entry Point ───────────────────────────────────────────────
def main():
    print("=== YOLOS Optimization Benchmark ===")
    print(f"Will benchmark: FP32 → FP16 → ONNX FP32 → INT8 PTQ")

    global OUTPUT_DIR
    if not Path("optimization").exists():
        OUTPUT_DIR = Path("/tmp/optimization_outputs")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download model from GCS
    download_model_from_gcs()

    # Load model
    model, processor = load_model()

    # Run all benchmarks
    results = run_benchmarks(model)

    # Print table
    print_benchmark_table(results)

    # Save results
    save_results(results)

    print("\n=== Done ===")
    print("Run this script again after GPU training for full results")
    print("including FP16 and TensorRT INT8 numbers.")

    # Upload results to GCS
    print("\n=== Uploading results to GCS ===")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            blob = bucket.blob(f"optimization/{f.name}")
            blob.upload_from_filename(str(f))
            print(f"Uploaded: {f.name}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
