import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs1"
MODELS_DIR = PROJECT_ROOT / "models1"

LOGS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows safe


# =========================
# TRANSFORMS
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =========================
# MODEL BUILD
# =========================
def build_model(arch: str, num_classes: int):
    if arch == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        return m

    elif arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m

    elif arch == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        m = models.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    else:
        raise ValueError(f"Unknown arch: {arch}")


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    class_to_idx = ckpt.get("class_to_idx", None)
    meta = ckpt.get("meta", {})
    return model, class_to_idx, meta


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    ys = []
    ps = []
    probs_all = []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
        probs_all.append(probs.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs_all)
    return y_true, y_pred, y_prob


def save_confusion_matrix(cm, class_names, out_path: Path, normalize=False):
    plt.figure(figsize=(12, 10))
    if normalize:
        cmn = cm.astype(np.float64) / np.maximum(1, cm.sum(axis=1, keepdims=True))
        plt.imshow(cmn, interpolation="nearest")
        plt.title("Confusion Matrix (normalized)")
        values = cmn
    else:
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        values = cm

    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Hücre üstüne sayı yaz (çok kalabalık olursa kapatabilirsin)
    fmt = ".2f" if normalize else "d"
    thresh = values.max() * 0.6 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            txt = format(v, fmt)
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if v > thresh else "black",
                     fontsize=7)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def per_class_accuracy(cm, class_names):
    # class accuracy = correct / total_true
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    acc = np.divide(correct, np.maximum(1, totals))
    df = pd.DataFrame({
        "class": class_names,
        "total": totals,
        "correct": correct,
        "accuracy": acc
    }).sort_values("accuracy", ascending=False)
    return df


def main(arch: str = "efficientnet_b0"):
    # dataset (test)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    class_names = test_ds.classes
    num_classes = len(class_names)

    # model + load best ckpt
    ckpt_path = MODELS_DIR / f"{arch}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(arch, num_classes).to(DEVICE)
    model, class_to_idx, meta = load_checkpoint(model, ckpt_path)

    # predict all
    y_true, y_pred, y_prob = predict_all(model, test_loader)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # per-class acc
    df_acc = per_class_accuracy(cm, class_names)

    # classification report (precision/recall/f1)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )

    # save outputs
    out_cm = LOGS_DIR / f"confusion_matrix_{arch}.png"
    out_cmn = LOGS_DIR / f"confusion_matrix_{arch}_normalized.png"
    out_csv = LOGS_DIR / f"per_class_accuracy_{arch}.csv"
    out_txt = LOGS_DIR / f"classification_report_{arch}.txt"

    save_confusion_matrix(cm, class_names, out_cm, normalize=False)
    save_confusion_matrix(cm, class_names, out_cmn, normalize=True)

    df_acc.to_csv(out_csv, index=False, encoding="utf-8")
    out_txt.write_text(report, encoding="utf-8")

    # quick console summary
    top1 = (y_true == y_pred).mean()
    # top3 accuracy
    top3 = 0
    top3_idx = np.argsort(-y_prob, axis=1)[:, :3]
    for i in range(len(y_true)):
        if y_true[i] in top3_idx[i]:
            top3 += 1
    top3 = top3 / len(y_true)

    print(f"[{arch}] TEST top1={top1:.4f} top3={top3:.4f}")
    print(f"Saved: {out_cm}")
    print(f"Saved: {out_cmn}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_txt}")
    print("\nWorst 8 classes by accuracy:")
    print(df_acc.sort_values("accuracy", ascending=True).head(8))


if __name__ == "__main__":
    # İstersen burayı mobilenet_v2 / resnet18 yap
    main("efficientnet_b0")