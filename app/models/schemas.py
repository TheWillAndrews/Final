from typing import Optional, Tuple, Literal, List

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from pydantic import BaseModel


ProductLabel = Literal["banana", "apple", "neither"]


class ProductClassifier:
    """
    Minimal MVP classifier:
    - Uses pretrained MobileNetV2 (ImageNet)
    - Maps fine-grained ImageNet labels to: banana / apple / neither
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use pretrained weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v2(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # These transforms match how MobileNetV2 was trained
        self.transform = weights.transforms()

        # Official ImageNet class names
        self.imagenet_classes = weights.meta["categories"]

        # Keyword groups for collapsing ImageNet → simple label
        self._banana_keywords = {"banana", "plantain"}
        self._apple_keywords = {
            "granny smith", "red delicious", "golden delicious", "apple"
        }

    def _map_label(self, imagenet_labels: list[str]) -> ProductLabel:
        low = [lbl.lower() for lbl in imagenet_labels]

        for lbl in low:
            if any(k in lbl for k in self._banana_keywords):
                return "banana"

        for lbl in low:
            if any(k in lbl for k in self._apple_keywords):
                return "apple"

        return "neither"

    def predict(self, img: Image.Image, topk: int = 5) -> Tuple[str, float, list]:
        """
        Returns:
        - coarse_label (banana / apple / neither)
        - confidence of the top ImageNet class
        - top-k imagenet predictions for debugging
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        # Get top-k predictions
        pvals, idxs = probs.topk(topk, dim=1)
        pvals = pvals[0].cpu().tolist()
        idxs = idxs[0].cpu().tolist()
        labels = [self.imagenet_classes[i] for i in idxs]

        coarse = self._map_label(labels)
        confidence = pvals[0]

        return coarse, confidence, list(zip(labels, pvals))

class DetectedProduct(BaseModel):
    name: str
    confidence: float

class ProductLocation(BaseModel):
    product_name: str
    category: str          # <-- add this
    aisle: str             # stored as TEXT in SQLite, fine as str
    section: str
    price: float
    in_stock: bool

class LocateResponse(BaseModel):
    detected_product: DetectedProduct
    location: Optional[ProductLocation] = None
    message: str
