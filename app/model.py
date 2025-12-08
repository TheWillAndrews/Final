# app/model.py
from __future__ import annotations

from typing import Tuple
from PIL import Image


class ProductClassifier:
    """
    Placeholder product classifier.

    For now this is a stub so the app runs.
    It exposes:
        predict(image: PIL.Image.Image) -> (label: str, confidence: float)

    Later, you can replace the body of `predict` with:
    - a real CNN (e.g., MobileNet with transfer learning)
    - loading weights from disk
    - proper preprocessing, etc.
    """

    def __init__(self) -> None:
        # TODO: load your real trained model here when it's ready
        # e.g., torch.load("models/product_classifier.pt")
        self.labels = ["banana", "apple", "cereal", "milk"]

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Dummy prediction: always returns 'banana' with 0.95 confidence
        so the rest of the pipeline (DB lookup, UI) can be tested.
        """
        label = "banana"
        confidence = 0.95
        return label, confidence

