# mlops-edge-perception

A production-grade MLOps platform for autonomous driving perception, built on Google Cloud. Combines YOLOS Vision Transformer object detection with automated data labeling, model optimization, drift-triggered retraining, and VLM-based scene reasoning.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATED PIPELINE (Vertex AI)               │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Grounding  │───▶│  Data Prep │───▶│   YOLOS     │          │
│  │  DINO       │    │  + Upload   │    │  Training   │          │
│  │  Autolabel  │    │  to GCS     │    │  T4 GPU     │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                               │                 │
│  ┌─────────────┐    ┌─────────────┐           │                 │
│  │  Evidently  │◀───|   ONNX      |◀─────────┘                 │
│  │  Drift      │    │  Optimize   │                             │
│  │  Monitor    │    │  INT8 PTQ   │                             │
│  └──────┬──────┘    └─────────────┘                             │
│         │                                                       │
└─────────┼────────────────────────────────────────────────────---┘
          │ drift > 30%
          ▼
┌─────────────────────┐      ┌─────────────────────────────────┐
│  GitHub Actions     |      │  Claude VLM Anomaly Layer       │
│  Auto-Retrain       │      │  On-demand scene reasoning      │
│  (every 6 hours)    │      │  + risk assessment              │
└─────────────────────┘      └─────────────────────────────────┘
```

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Model | YOLOS-tiny (Vision Transformer) |
| Dataset | KITTI Autonomous Driving (500 images) |
| Classes | Car, Pedestrian, Cyclist |
| Training | 20 epochs on NVIDIA T4 GPU |
| Best Val Loss | 1.02 |
| Training Infrastructure | Vertex AI Custom Training Job |

### Optimization Benchmark (CPU)

| Format | Latency (ms) | FPS | Size (MB) | vs Baseline |
|--------|-------------|-----|-----------|-------------|
| PyTorch FP32 | 505ms | 2.0 | 24.7MB | baseline |
| ONNX FP32 | 1182ms | 0.8 | 24.9MB | — |
| ONNX INT8 PTQ | 1003ms | 1.0 | **9.0MB** | **63% smaller** |

### Drift Detection

| Metric | Value |
|--------|-------|
| Features extracted | 13 per image |
| Statistical test | Kolmogorov-Smirnov |
| Drift threshold | 30% of features |
| Simulated drift | Brightness x2.5 + Fog + Blur |
| Mean pixel shift | 0.356 → 0.693 (+94%) |
| Detection result | 100% features drifted, retrain triggered |

### Grounding DINO Auto-labeling

| Metric | Value |
|--------|-------|
| Images processed | 49/50 |
| Bounding boxes generated | 247 |
| Avg boxes per image | 5.0 |
| Text prompts | "car . pedestrian . cyclist ." |
| Model | IDEA-Research/grounding-dino-tiny |

---

## Pipeline Components

### Stage 1 - Grounding DINO Auto-labeling (`data/autolabel.py`)
Zero-shot object detection using text prompts to automatically annotate unlabeled KITTI images. Eliminates manual annotation overhead. Outputs YOLO-format labels uploaded to GCS.

### Stage 2 - Data Preparation (`data/prepare_dataset.py`)
Downloads KITTI dataset from HuggingFace, converts bounding box format (KITTI → YOLO), filters to 3 classes, uploads 1001 files to GCS.

### Stage 3 - YOLOS Training (`training/train.py`)
Fine-tunes YOLOS-tiny Vision Transformer on KITTI subset using Vertex AI Custom Training Job with T4 GPU. Key implementation details:
- Singleton processor pattern in DataLoader `collate_fn` to prevent memory leaks
- XLA environment variables to prevent torch_xla crash in GPU containers
- Model versioning with timestamped GCS paths (`models/runs/TIMESTAMP/`)

### Stage 4 - Model Optimization (`optimization/export_onnx.py`)
Exports trained model to ONNX and benchmarks FP32 vs INT8 PTQ quantization. Uses ONNX Runtime `quantize_dynamic` for transformer-compatible INT8 quantization (PyTorch built-in PTQ does not support transformer architectures).

### Stage 5 - Drift Detection (`monitoring/drift_detect.py`)
Extracts 13 statistical features per image (brightness, per-channel RGB stats, percentiles) and runs Kolmogorov-Smirnov tests between training baseline and production data. Writes `retrain_trigger.json` when drift exceeds 30% threshold.

### Stage 6 - Kubeflow Pipeline DAG (`pipelines/pipeline.py`)
Orchestrates all 5 stages as a Vertex AI Pipelines DAG with sequential dependencies and model versioning.

### Stage 7 - Claude VLM Anomaly Layer (`vlm/anomaly_layer.py`)
Triggers Claude Vision API on anomalous frames (low confidence, high object count, pedestrian-vehicle proximity). Returns structured risk assessment (LOW/MEDIUM/HIGH) with natural language scene description.

---

## Automated Retraining Loop

```
GitHub Actions (every 6 hours)
        ↓
drift_detect.py runs automatically
        ↓
Drift > 30%? → writes retrain_trigger.json
        ↓
GitHub Actions submits new Vertex AI Pipeline
        ↓
Full 5-stage pipeline reruns with new data
```

Implemented in `.github/workflows/drift_monitor.yml`.

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Cloud Platform | Google Cloud (Vertex AI, GCS) |
| Pipeline Orchestration | Kubeflow Pipelines (KFP v2) |
| Object Detection | YOLOS-tiny (HuggingFace Transformers) |
| Auto-labeling | Grounding DINO (IDEA-Research) |
| Model Optimization | ONNX Runtime, Dynamic INT8 Quantization |
| Drift Detection | Evidently AI v0.7.21 |
| VLM Reasoning | Claude Vision API (Anthropic) |
| CI/CD | GitHub Actions |
| Language | Python 3.12 |

---

## Project Structure

```
mlops-edge-perception/
├── data/
│   ├── prepare_dataset.py      # KITTI download + GCS upload
│   └── autolabel.py            # Grounding DINO auto-labeling
├── training/
│   ├── train.py                # YOLOS fine-tuning script
│   └── vertex_job.py           # Vertex AI job submission
├── optimization/
│   └── export_onnx.py          # ONNX export + benchmark
├── monitoring/
│   ├── drift_detect.py         # Evidently drift detection
│   └── simulate_drift.py       # Drift simulation for testing
├── pipelines/
│   ├── pipeline.py             # Kubeflow DAG definition
│   └── kitti_pipeline.yaml     # Compiled pipeline YAML
├── vlm/
│   └── anomaly_layer.py        # Claude VLM integration
├── .github/
│   └── workflows/
│       └── drift_monitor.yml   # Auto-retrain CI/CD
└── CLAUDE.md                   # Project context for Claude Code
```

---

## GCS Bucket Structure

```
gs://mlops-edge-perception-bucket/
├── kitti/
│   ├── images/train/           # 400 training images
│   ├── images/val/             # 100 validation images
│   ├── images/val_drifted/     # Simulated drift images
│   ├── labels/train/ + val/    # YOLO format labels
│   └── autolabeled/            # Grounding DINO outputs
├── models/
│   └── runs/TIMESTAMP/         # Versioned model artifacts
│       ├── best_model/         # YOLOS weights + config
│       └── optimized/          # ONNX + benchmark results
├── monitoring/
│   └── drift_reports/          # Drift detection summaries
└── scripts/
    └── train.py                # Training script for pipeline
```

---

## Setup

### Prerequisites
- Python 3.12
- GCP project with Vertex AI and GCS enabled
- Google Cloud SDK authenticated

### Installation

```bash
git clone https://github.com/NikkiMandal/mlops-edge-perception
cd mlops-edge-perception
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # For VLM layer
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

### Running Individual Components

```bash
# Data preparation
python data/prepare_dataset.py

# Auto-labeling
python data/autolabel.py

# Training (local)
python training/train.py --epochs 20 --batch_size 4

# Training (Vertex AI)
python training/vertex_job.py

# Optimization benchmark
python optimization/export_onnx.py

# Drift detection
python monitoring/drift_detect.py

# Simulate drift
python monitoring/simulate_drift.py

# VLM analysis
python vlm/anomaly_layer.py

# Submit full pipeline
python -c "from pipelines.pipeline import submit_pipeline; submit_pipeline()"
```

---

## Key Technical Decisions

**YOLOS over RT-DETR:** RT-DETR requires `transformers>=4.44` which is incompatible with PyTorch 2.1 in Vertex AI prebuilt containers. YOLOS-tiny achieves the same transformer-based detection story with full compatibility.

**ONNX Runtime INT8 over PyTorch PTQ:** PyTorch built-in quantization (`torch.quantization`) does not support transformer attention layers. ONNX Runtime `quantize_dynamic` works correctly on any architecture.

**Evidently K-S test over simple statistics:** Kolmogorov-Smirnov test detects distribution shift without assuming normality, making it robust to real-world sensor drift patterns.

**Singleton processor in DataLoader:** Re-initializing `AutoImageProcessor.from_pretrained()` on every batch caused 2,000 unnecessary initializations per training run, leading to SIGABRT from Linux OOM killer. Fixed with global singleton pattern.

---

## Debugging Notes

Major issues encountered and resolved during development:

- **torch_xla SIGABRT:** PyTorch GPU containers include torch_xla TPU library that crashes during backward pass. Fixed by setting `PJRT_DEVICE=GPU`, `DISABLE_XLA=1`, `USE_TORCH_XLA=0` at process start.
- **DataLoader memory leak:** `collate_fn` was re-initializing `AutoImageProcessor` every batch. Fixed with singleton pattern + `num_workers=0`.
- **ONNX opset compatibility:** opset 17 requires `onnxscript` not available in all containers. Downgraded to opset 12 for compatibility.
- **Kubeflow component globals:** Pipeline components run in isolated containers - global variables from `pipeline.py` are not available. All configuration must be passed as function parameters.

---

## Author

Nikita Vinod — MS Electrical and Computer Engineering, Northeastern University (December 2025)

Specialization: ML Engineering, Edge AI Deployment, Perception Systems

[GitHub](https://github.com/NikkiMandal) | [LinkedIn](https://www.linkedin.com/in/nikitamandal03)
