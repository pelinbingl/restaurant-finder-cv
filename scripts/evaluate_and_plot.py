import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
)
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models1"
PLOTS_DIR = PROJECT_ROOT / "plots1"
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_model(arch, num_classes):
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)

    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unknown arch")

    return m

def evaluate_model(arch):
    print(f"\nEvaluating {arch}")

    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    ckpt = torch.load(MODELS_DIR / f"{arch}_best.pth", map_location=DEVICE)
    num_classes = len(ckpt["class_to_idx"])

    model = load_model(arch, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=test_ds.classes
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"{arch} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{arch}_confusion_matrix.png")
    plt.close()

    print(f"Saved: {PLOTS_DIR / f'{arch}_confusion_matrix.png'}")

if __name__ == "__main__":
    evaluate_model("mobilenet_v2")
    evaluate_model("efficientnet_b0")
    evaluate_model("resnet18")
