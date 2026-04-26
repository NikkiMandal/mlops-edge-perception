# CLAUDE.md — Project Briefing

## What This Project Is
Production MLOps pipeline for edge computer vision.
End-to-end system: auto-label data → train transformer 
detector → optimize for edge → monitor drift → auto-retrain
→ VLM reasoning layer. Fully automated via Kubeflow DAG 
on Vertex AI, triggered by GitHub Actions CI/CD.

## The Goal
Build all stages in this priority order:
1. ✅ Stage 2 — Train RT-DETR on Vertex AI (waiting GPU quota)
2. ✅ Stage 3 — ONNX export + FP32/FP16/INT8 benchmark table
3. ✅ Stage 5 — Evidently drift detection + retrain trigger
4. ⏳ Stage 6 — Kubeflow DAG connecting all stages (pipeline.py)
5. ⏳ Stage 7 — VLM anomaly layer (Gemini API)
6. ⏳ Stage 8 — Real-time OpenCV streaming inference
7. ⏳ Stage 4 — Vertex AI Model Registry + promotion gate
8. ⏳ Stage 1 — Auto-labeling with Grounding DINO

## Current Status (as of April 25 2026)
DONE:
- Full local environment set up (Python 3.12, venv, all packages)
- GCP project created: mlops-edge-perception
- GCS bucket: gs://mlops-edge-perception-bucket
- APIs enabled: Vertex AI, Cloud Storage, Artifact Registry
- gcloud authenticated with ADC
- GitHub repo: github.com/NikkiMandal/mlops-edge-perception
- data/prepare_dataset.py — downloads KITTI from HuggingFace
  (nateraw/kitti), converts to YOLO format, uploads to GCS.
  500 samples (400 train, 100 val). TESTED AND WORKING.
- training/train.py — RT-DETR fine-tuning script for Vertex AI
- training/vertex_job.py — submits GPU training job (T4)
- optimization/export_onnx.py — ONNX export + full benchmark
- monitoring/drift_detect.py — Evidently drift detection.
  TESTED AND WORKING. Runs on GCS data locally.

WAITING:
- GPU quota approval for Vertex AI T4
  (custom_model_training_nvidia_t4_gpus, us-central1, limit=1)
  Case #70595082 submitted with urgency note.
  When approved: run python training/vertex_job.py

NOT STARTED YET:
- pipelines/pipeline.py — Kubeflow DAG
- vlm/anomaly_layer.py — Gemini VLM integration
- streaming/realtime_inference.py — OpenCV stream
- .github/workflows/pipeline_trigger.yml — CI/CD

## Tech Stack
- Model: RT-DETR (PekingU/rtdetr_r50vd_coco_o365)
  Pretrained on COCO+Objects365, fine-tuned on KITTI
  This IS transfer learning — mention explicitly
- Dataset: KITTI object detection (cars/pedestrians/cyclists)
  500 image subset from nateraw/kitti on HuggingFace
- Training: Vertex AI Custom Training (n1-standard-4 + T4 GPU)
  Container: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest
- Optimization: ONNX Runtime + PyTorch quantization (PTQ + QAT)
- Monitoring: Evidently AI v0.7.21 (new API - use evidently.Report
  and evidently.presets.DataDriftPreset, NOT legacy imports)
- Orchestration: Kubeflow Pipelines on Vertex AI (TO BUILD)
- VLM: Gemini API via google-generativeai (TO BUILD)
- CI/CD: GitHub Actions (TO BUILD)
- Storage: gs://mlops-edge-perception-bucket
- Auth: Application Default Credentials (ADC)

## GCP Configuration
- Project ID: mlops-edge-perception
- Region: us-central1
- Zone: us-central1-a
- Bucket: gs://mlops-edge-perception-bucket
- Auth: ADC via gcloud auth application-default login

## GCS Structure
gs://mlops-edge-perception-bucket/
  kitti/
    images/train/    ← 400 PNG images
    images/val/      ← 100 PNG images
    labels/train/    ← 400 YOLO .txt label files
    labels/val/      ← 100 YOLO .txt label files
    dataset_config.json
  models/
    rtdetr_kitti/    ← trained model saved here after training
      best_model/
      metrics.json
  monitoring/
    drift_reports/   ← drift JSON summaries uploaded here

## Folder Structure
- data/              → dataset download and prep scripts
- training/          → Vertex AI training scripts
- optimization/      → ONNX export, quantization benchmarks
- monitoring/        → Evidently drift detection
- pipelines/         → Kubeflow pipeline DAG (TO BUILD)
- vlm/               → Gemini VLM anomaly layer (TO BUILD)
- streaming/         → OpenCV real-time inference (TO BUILD)
- .github/workflows/ → GitHub Actions CI/CD (TO BUILD)

## Pipeline DAG Plan (pipeline.py)
Chain these components using @dsl.component decorator:
1. prepare_data_component — wraps prepare_dataset.py logic
2. train_component — submits Vertex AI training job
3. optimize_component — runs ONNX export + benchmark
4. evaluate_component — checks mAP vs threshold
5. register_component — uploads to Vertex Model Registry
6. monitor_component — runs drift detection
Trigger: GitHub Actions on push to main →
  compile pipeline → submit to Vertex AI Pipelines

## VLM Layer Plan (anomaly_layer.py)
After RT-DETR inference on a frame:
1. Serialize detections as structured JSON
   (class, confidence, bbox coordinates, count per class)
2. Send image + detection JSON to Gemini Vision API
3. Prompt: "Given these detections, describe the scene
   and flag any anomalies or safety concerns"
4. Return natural language description + anomaly flag
5. Only trigger VLM when: confidence drops below threshold
   OR unusual object count detected (cost control)
Use: google-generativeai package, gemini-1.5-flash model
     (cheapest, fast enough for this use case)

## Key Design Decisions
- RT-DETR over YOLO: transformer architecture (resume signal),
  no anchor hyperparameters, better mAP at similar speed
- CPU-only local machine: all real training on Vertex AI
- PTQ vs QAT comparison is core deliverable of Stage 3
- Evidently v0.7.21 uses NEW API:
  from evidently import Report, Dataset
  from evidently.presets import DataDriftPreset
  DO NOT use: from evidently.report import Report (old API)
- Drift threshold: 30% of features drifted = retrain trigger
- VLM only triggered conditionally (not every frame) for cost

## Resume Context
Project must produce these artifacts:
1. Benchmark table: FP32 vs FP16 vs ONNX vs INT8 PTQ vs QAT
   (columns: format, mAP@0.5, latency ms, FPS, size MB)
2. Drift detection logs showing trigger mechanism
3. Kubeflow pipeline DAG visualization screenshot
4. VLM output examples showing anomaly descriptions
5. GitHub repo with clean commit history

Resume bullets to aim for:
- "Built end-to-end MLOps pipeline on Vertex AI (Kubeflow DAG,
  GitHub Actions CI/CD) training RT-DETR on KITTI via transfer
  learning, achieving [X]% mAP with automated drift-triggered
  retraining via Evidently AI"
- "Produced FP32→FP16→TensorRT INT8 PTQ/QAT benchmark achieving
  [X]x speedup with <2% mAP drop; integrated Gemini VLM for
  natural language anomaly detection on inference outputs"

## What NOT To Do
- Do not use GPU-specific code locally (no NVIDIA GPU)
- Do not train on full KITTI dataset (too expensive)
- Do not use old Evidently API imports (breaks on v0.7.21)
- Do not commit data files (gitignored)
- Do not use pytorch-gpu container without GPU accelerator
- Do not run export_onnx.py before training completes
- GPU container: pytorch-gpu.2-1.py310:latest (WORKS)
- CPU container: pytorch-cpu.*.* (NOT SUPPORTED on Vertex AI)
  CPU fallback: use n1-standard-8 with GPU still attached

## Next Immediate Actions
1. Wait for GPU quota email (Case #70595082)
2. When quota arrives:
   a. python training/vertex_job.py (GPU, 20 epochs)
   b. python optimization/export_onnx.py
   c. Save benchmark table to optimization/outputs/
3. Then build pipelines/pipeline.py (Kubeflow DAG)
4. Then build vlm/anomaly_layer.py (Gemini integration)
5. Then build streaming/realtime_inference.py
6. Then add .github/workflows/pipeline_trigger.yml