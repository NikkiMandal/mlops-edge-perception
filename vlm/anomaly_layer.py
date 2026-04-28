"""
anomaly_layer.py — VLM-based scene analysis using Gemini
Takes RT-DETR detections + image, returns natural language
anomaly description. Only triggers on confidence drop or
unusual object counts to control API costs.
"""

import os
import json
import base64

import anthropic

from pathlib import Path
from PIL import Image, ImageDraw

# ── Configuration ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-opus-4-5"  # cheapest, fast enough
CONFIDENCE_THRESH = 0.5   # flag if avg confidence drops below this
MAX_OBJECTS_THRESH = 10   # flag if more than this many objects detected
CLASSES           = ["Car", "Pedestrian", "Cyclist"]

# ── Initialize Gemini ─────────────────────────────────────────
#genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel(MODEL_NAME)

#client = genai.Client(api_key=GEMINI_API_KEY)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Helper: Image to Base64 ───────────────────────────────────
def image_to_base64(image_path):
    """Convert image file to base64 string for Gemini API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ── Helper: Draw Detections on Image ─────────────────────────
def draw_detections(image_path, detections):
    """
    Draw bounding boxes on image for visualization.
    Returns annotated PIL image.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = {"Car": "red", "Pedestrian": "green", "Cyclist": "blue"}

    for det in detections:
        label     = det["label"]
        score     = det["score"]
        box       = det["box"]  # [x_min, y_min, x_max, y_max] normalized

        # Convert normalized to pixel coords
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        color = colors.get(label, "white")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 15), f"{label} {score:.2f}", fill=color)

    return img


# ── Anomaly Check ─────────────────────────────────────────────
def should_trigger_vlm(detections):
    """
    Decide whether to call the VLM API.
    Only trigger on potential anomalies to save API costs.

    ELI5: Don't call the expensive AI for every normal frame.
    Only call it when something looks unusual.
    """
    if not detections:
        return True, "No objects detected — unusual for this scene"

    scores = [d["score"] for d in detections]
    avg_confidence = sum(scores) / len(scores)

    if avg_confidence < CONFIDENCE_THRESH:
        return True, f"Low confidence detections (avg={avg_confidence:.2f})"

    if len(detections) > MAX_OBJECTS_THRESH:
        return True, f"Unusual object count ({len(detections)} objects)"

    # Check for pedestrian near vehicle (safety concern)
    classes_detected = [d["label"] for d in detections]
    has_pedestrian = "Pedestrian" in classes_detected
    has_vehicle    = "Car" in classes_detected
    if has_pedestrian and has_vehicle:
        return True, "Pedestrian and vehicle in same frame"

    return False, "Normal scene — VLM not triggered"


# ── Core VLM Analysis ─────────────────────────────────────────
def analyze_scene(image_path, detections):
    """
    Send image + detections to Gemini for scene analysis.
    Returns natural language description and anomaly flag.
    """
    trigger, trigger_reason = should_trigger_vlm(detections)

    if not trigger:
        return {
            "vlm_triggered":   False,
            "trigger_reason":  trigger_reason,
            "scene_description": None,
            "anomaly_detected":  False,
            "anomaly_details":   None,
        }

    print(f"\n VLM triggered: {trigger_reason}")

    # Build detection summary for prompt
    detection_summary = []
    class_counts = {}
    for det in detections:
        label = det["label"]
        score = det["score"]
        class_counts[label] = class_counts.get(label, 0) + 1
        detection_summary.append(f"{label} (confidence: {score:.2f})")

    counts_str = ", ".join([f"{v} {k}(s)" for k, v in class_counts.items()])

    # Build prompt
    prompt = f"""You are an AI safety monitor for an autonomous driving system.

Object detection model detected: {counts_str}
Individual detections: {', '.join(detection_summary)}

Analyze this traffic scene and provide:
1. A brief scene description (1-2 sentences)
2. Any safety concerns or anomalies you notice
3. A risk level: LOW, MEDIUM, or HIGH

Be specific and concise. Focus on safety-relevant observations."""

    #response = model.generate_content([prompt, image_data])
    #response_text = response.text

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_to_base64(image_path)
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )
    response_text = response.content[0].text

    # Parse risk level from response
    risk_level = "LOW"
    if "HIGH" in response_text.upper():
        risk_level = "HIGH"
    elif "MEDIUM" in response_text.upper():
        risk_level = "MEDIUM"

    return {
        "vlm_triggered":     True,
        "trigger_reason":    trigger_reason,
        "scene_description": response_text,
        "anomaly_detected":  risk_level in ["MEDIUM", "HIGH"],
        "risk_level":        risk_level,
        "anomaly_details":   response_text,
    }


# ── Test with Sample KITTI Image ──────────────────────────────
def test_with_sample():
    """
    Test the VLM layer with a real KITTI image and
    simulated detections. Run this to verify the integration.
    """
    print("=== VLM Anomaly Layer Test ===")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {'✓ Set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ NOT SET'}\n")

    # Find a sample image from local KITTI data
    sample_dirs = [
        "data/kitti_processed/images/val",
        "data/kitti_processed/images/train",
    ]

    sample_image = None
    for d in sample_dirs:
        images = list(Path(d).glob("*.png"))
        if images:
            sample_image = str(images[0])
            break

    if not sample_image:
        print("No local KITTI images found.")
        print("Run data/prepare_dataset.py first, or provide an image path.")
        return

    print(f"Using image: {sample_image}")

    # Simulate detections (in production these come from RT-DETR)
    simulated_detections = [
        {"label": "Car",        "score": 0.92, "box": [0.1, 0.4, 0.4, 0.8]},
        {"label": "Car",        "score": 0.87, "box": [0.5, 0.3, 0.8, 0.7]},
        {"label": "Pedestrian", "score": 0.45, "box": [0.6, 0.2, 0.7, 0.6]},
    ]

    print(f"Simulated detections: {len(simulated_detections)} objects")
    for d in simulated_detections:
        print(f"  {d['label']}: {d['score']:.2f}")

    # Run analysis
    result = analyze_scene(sample_image, simulated_detections)

    # Print results
    print("\n=== VLM Analysis Result ===")
    print(f"VLM triggered:    {result['vlm_triggered']}")
    print(f"Trigger reason:   {result['trigger_reason']}")

    if result['vlm_triggered']:
        print(f"Risk level:       {result.get('risk_level', 'N/A')}")
        print(f"Anomaly detected: {result['anomaly_detected']}")
        print(f"\nScene description:")
        print(f"{result['scene_description']}")

    # Save result
    output_dir = Path("vlm/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {output_path}")

    # Save annotated image
    if result['vlm_triggered']:
        annotated = draw_detections(sample_image, simulated_detections)
        annotated_path = output_dir / "annotated_sample.png"
        annotated.save(str(annotated_path))
        print(f"Annotated image saved to {annotated_path}")


if __name__ == "__main__":
    test_with_sample()