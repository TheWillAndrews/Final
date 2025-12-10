# scripts/train_stub.py
"""
Placeholder script for transfer-learning MobileNet on grocery products.

Outline:
1. Load image dataset from data/grocery_images/ (train/val split)
2. Initialize MobileNetV2 pre-trained on ImageNet
3. Replace final layer with N-way classifier (N = #product classes)
4. Train for a few epochs
5. Save weights to models/grocery_mobilenet.pth
"""

def main() -> None:
    print("TODO: implement transfer learning training loop here.")
    print("For the demo we are using the pre-trained ImageNet weights only.")


if __name__ == "__main__":
    main()

