# app/services/vision.py
from __future__ import annotations

from typing import Tuple, List, Dict
import io
import random

from PIL import Image, ImageDraw

from app.model import ProductClassifier


# Single global classifier instance
_classifier = ProductClassifier()


def load_image(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes into a PIL RGB image.
    Raises ValueError if the image can't be opened.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError("Could not open image") from e


def classify_image(image: Image.Image) -> Tuple[str, float]:
    """
    Run the classifier on a PIL image and return (label, confidence).
    """
    return _classifier.predict(image)


def process_bytes_for_single_product(
    file_bytes: bytes,
    content_type: str | None,
) -> Tuple[Image.Image, str, float]:
    """
    High-level helper for /predict:

    - Validate content-type
    - Load bytes into image
    - Run classifier

    Returns:
        (image, predicted_label, confidence)
    """
    if not content_type or not content_type.startswith("image/"):
        raise ValueError("Please upload an image file.")

    image = load_image(file_bytes)
    label, confidence = classify_image(image)
    return image, label, confidence


def fake_multi_detections(image: Image.Image, base_label: str) -> List[Dict]:
    """
    Simulate multiple object detections.

    Returns a list of dicts:
    [{ "label": str, "confidence": float, "bbox": [x1, y1, x2, y2] }, ...]
    """
    width, height = image.size

    labels_pool = [base_label, "banana", "apple", "cereal", "milk"]
    # Deduplicate while keeping order
    seen = set()
    unique_labels = []
    for lbl in labels_pool:
        if lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)

    num_detections = min(3, len(unique_labels))

    detections: List[Dict] = []
    for i in range(num_detections):
        label = unique_labels[i]
        conf = round(random.uniform(0.55, 0.97), 2)

        x1 = random.randint(0, max(0, int(width * 0.5)))
        y1 = random.randint(0, max(0, int(height * 0.5)))
        x2 = random.randint(x1 + max(10, int(width * 0.15)), width)
        y2 = random.randint(y1 + max(10, int(height * 0.15)), height)

        detections.append(
            {
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return detections


def draw_detections(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Draw bounding boxes + labels on a copy of the image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=(37, 99, 235), width=3)

        text = f"{label} {int(conf * 100)}%"
        try:
            draw.rectangle([x1, y1 - 16, x1 + 110, y1], fill=(37, 99, 235))
            draw.text((x1 + 4, y1 - 14), text, fill=(255, 255, 255))
        except Exception:
            pass

    return annotated

