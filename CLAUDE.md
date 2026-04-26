# CLAUDE.md — Project Briefing

## What This Project Is
Production MLOps pipeline for edge computer vision.
We train an object detection model, optimize it for 
edge hardware, and build a system that monitors itself 
and retrains automatically when it starts degrading.

## The Goal
Build 5 stages in priority order:
1. Stage 2 — Train RT-DETR on Vertex AI (Google Cloud GPU)
2. Stage 3 — Optimize model: FP32 → FP16 → INT8 (benchmark table)
3. Stage 5 — Drift detection with Evidently, auto-retrain trigger
4. Stage 1 — Auto-labeling with Grounding DINO
5. Stage 8 — Real-time OpenCV inference stream

## Tech Stack
- Model: RT-DETR (transformer-based object detector)
- Dataset: KITTI (autonomous driving, cars/pedestrians/cyclists)
- Training: Vertex AI Custom Training Jobs (Google Cloud)
- Optimization: ONNX export → TensorRT INT8 PTQ and QAT
- Monitoring: Evidently AI for data drift detection
- Storage: Google Cloud Storage (gs://mlops-edge-perception-bucket)
- Orchestration: Vertex AI Pipelines (Kubeflow)
- CI/CD: GitHub Actions
- Language: Python 3.12

## GCP Configuration
- Project ID: mlops-edge-perception
- Region: us-central1
- Bucket: gs://mlops-edge-perception-bucket
- Auth: Application Default Credentials (ADC)

## Folder Structure
- training/        → Vertex AI training scripts
- optimization/    → ONNX export, TensorRT quantization
- monitoring/      → Evidently drift detection
- pipelines/       → Kubeflow pipeline definitions
- data/            → Dataset download and prep scripts
- notebooks/       → Experimentation and analysis

## Key Design Decisions Already Made
- Using RT-DETR over YOLO for transformer architecture signal
- CPU-only local machine → all real training on Vertex AI T4 GPU
- PTQ vs QAT comparison is the core deliverable of Stage 3
- Evidently AI chosen over custom drift detection for speed
- KITTI subset only (not full dataset) to control GCP costs

## Resume Context
This project needs to produce:
1. A benchmark table: FP32 vs FP16 vs TensorRT INT8 PTQ vs QAT
   (mAP, latency ms, model size MB, FPS)
2. An automated drift → retrain loop with real trigger logs
3. A Vertex AI training job with experiment tracking

## What NOT to Do
- Do not use GPU-specific code locally (no NVIDIA GPU on this machine)
- Do not train on full KITTI dataset (too expensive)
- Do not use paid APIs unless explicitly discussed
- Do not install packages outside the venv

## Current Status
Setup complete. Starting Stage 2 — Vertex AI training.