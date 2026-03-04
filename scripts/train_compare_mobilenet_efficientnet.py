import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
)


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models1"
LOGS_DIR     = PROJECT_ROOT / "logs1"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

REPORT_PATH = LOGS_DIR / "training_report1.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 32
NUM_WORKERS  = 0  # Windows'ta sorun olmasın diye 0

EPOCHS_HEAD     = 2
EPOCHS_FINETUNE = 15
LR_HEAD         = 1e-3
LR_FINETUNE     = 1e-4
WEIGHT_DECAY    = 1e-4

EARLY_STOP_PATIENCE = 4

# =========================
# TRANSFORMS
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.1),   # overfitting'e karşı ekstra regularizasyon
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =========================
# UTILS
# =========================
def log(msg: str):
    print(msg)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct1 = correct3 = total = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)

        total_loss += ce(logits, y).item()

        # Top-1
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == y).sum().item()

        # Top-3 — sınıf sayısı 3'ten azsa Top-1 ile aynı sonucu verir
        k = min(3, logits.size(1))
        top_k = logits.topk(k, dim=1).indices
        for i in range(y.size(0)):
            if y[i].item() in top_k[i].tolist():
                correct3 += 1

        total += y.size(0)

    avg_loss = total_loss / max(1, len(loader))
    acc1 = correct1 / total if total else 0.0
    acc3 = correct3 / total if total else 0.0
    return avg_loss, acc1, acc3


def train_one_epoch(model, loader, optimizer):
    model.train()
    ce = nn.CrossEntropyLoss()
    running = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item()

    return running / max(1, len(loader))


def freeze_backbone(model, arch: str):
    if arch in ("mobilenet_v2", "efficientnet_b0"):
        for p in model.features.parameters():
            p.requires_grad = False
    elif arch == "resnet18":
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
    else:
        raise ValueError(f"freeze_backbone: bilinmeyen arch '{arch}'")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)

    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    else:
        raise ValueError(f"build_model: bilinmeyen arch '{arch}'")

    return m


def save_checkpoint(path: Path, model: nn.Module, class_to_idx: dict, meta: dict):
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     class_to_idx,
        "meta":             meta,
    }, path)


def load_best(path: Path, model: nn.Module):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# =========================
# EĞİTİM
# =========================
def train_model(arch: str, train_loader, val_loader, test_loader, class_to_idx):
    num_classes = len(class_to_idx)
    model       = build_model(arch, num_classes).to(DEVICE)
    best_path   = MODELS_DIR / f"{arch}_best.pth"
    history     = []

    best_val_acc = -1.0
    best_epoch   = -1

    log("\n" + "=" * 80)
    log(f"MODEL: {arch.upper()} | Device: {DEVICE} | Sınıf sayısı: {num_classes}")
    log("=" * 80)

    # ── Phase 1: Sadece classifier ──────────────────────────────────────────────
    freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.Adam(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS_HEAD + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc1, val_acc3 = evaluate(model, val_loader)
        took       = time.time() - t0

        history.append({"phase": "head", "epoch": epoch,
                         "train_loss": train_loss, "val_loss": val_loss,
                         "val_top1": val_acc1,     "val_top3": val_acc3, "sec": took})

        log(f"[{arch}][HEAD] Epoch {epoch:>2}/{EPOCHS_HEAD} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"top1={val_acc1:.3f} | top3={val_acc3:.3f} | {took:.1f}s")

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_epoch   = epoch
            save_checkpoint(best_path, model, class_to_idx,
                            {"arch": arch, "best_phase": "head", "best_epoch": epoch})
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                log(f"[{arch}] HEAD aşamasında erken durdurma.")
                break

    # ── Phase 2: Full fine-tune ──────────────────────────────────────────────────
    load_best(best_path, model)   # en iyi checkpointten devam et
    unfreeze_all(model)

    optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    # Scheduler: her epoch LR'yi yavaşça düşürür → daha stabil fine-tune
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE, eta_min=1e-6)
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc1, val_acc3 = evaluate(model, val_loader)
        scheduler.step()
        took = time.time() - t0

        history.append({"phase": "finetune", "epoch": epoch,
                         "train_loss": train_loss, "val_loss": val_loss,
                         "val_top1": val_acc1,     "val_top3": val_acc3, "sec": took})

        log(f"[{arch}][FT]   Epoch {epoch:>2}/{EPOCHS_FINETUNE} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"top1={val_acc1:.3f} | top3={val_acc3:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {took:.1f}s")

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            best_epoch   = epoch
            save_checkpoint(best_path, model, class_to_idx,
                            {"arch": arch, "best_phase": "finetune", "best_epoch": epoch})
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                log(f"[{arch}] FINETUNE aşamasında erken durdurma.")
                break

    # ── Final test ────────────────────────────────────────────────────────────────
    best_ckpt = load_best(best_path, model)
    test_loss, test_acc1, test_acc3 = evaluate(model, test_loader)

    meta = best_ckpt.get("meta", {})
    log(f"[{arch}] BEST Val Top-1 = {best_val_acc:.3f} | "
        f"phase={meta.get('best_phase','?')} epoch={meta.get('best_epoch','?')}")
    log(f"[{arch}] TEST  Top-1 = {test_acc1:.3f} | Top-3 = {test_acc3:.3f} | loss={test_loss:.4f}")
    log(f"[{arch}] Checkpoint: {best_path}")

    return {
        "arch":          arch,
        "best_val_top1": best_val_acc,
        "test_top1":     test_acc1,
        "test_top3":     test_acc3,
        "best_path":     str(best_path),
        "history":       history,
    }


# =========================
# MAIN
# =========================
def main():
    REPORT_PATH.write_text("", encoding="utf-8")

    log("=" * 80)
    log("Yiyecek Tanıma — Eğitim Raporu")
    log(f"Proje kökü : {PROJECT_ROOT}")
    log(f"Cihaz      : {DEVICE}")
    log("=" * 80)

    # Veri setleri
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tfms)

    # Sınıf eşleşmesi tutarlılık kontrolü
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, \
        "Train / Val / Test sınıf sıralamaları uyuşmuyor!"

    class_to_idx = train_ds.class_to_idx
    log(f"\nSınıflar ({len(train_ds.classes)}): {train_ds.classes}")
    log(f"Train  : {len(train_ds)} görüntü")
    log(f"Val    : {len(val_ds)} görüntü")
    log(f"Test   : {len(test_ds)} görüntü\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    results = []
    for arch in ("mobilenet_v2", "efficientnet_b0", "resnet18"):
        results.append(train_model(arch, train_loader, val_loader, test_loader, class_to_idx))

    # Özet tablo
    log("\n" + "=" * 80)
    log("SONUÇ ÖZETİ")
    log("=" * 80)
    log(f"{'Model':<20} {'Val Top-1':>10} {'Test Top-1':>10} {'Test Top-3':>10}")
    log("-" * 55)
    for r in results:
        log(f"{r['arch']:<20} {r['best_val_top1']:>10.3f} {r['test_top1']:>10.3f} {r['test_top3']:>10.3f}")

    # JSON kaydı
    json_path = LOGS_DIR / "training_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log(f"\nRapor kaydedildi: {REPORT_PATH}")
    log(f"JSON  kaydedildi: {json_path}")


if __name__ == "__main__":
    main()