"""
train.py — Yiyecek Tanıma Modeli Eğitimi v2
=============================================
Proje yapısı:
    restaurant_finder_cv/
    ├── data/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── models1/          ← .pth dosyaları buraya kaydedilir
    ├── logs1/            ← rapor ve JSON buraya kaydedilir
    └── scripts/
        └── train.py      ← bu dosya

Çalıştırma:
    python scripts/train.py

Anti-Overfitting önlemleri (v2):
    ✓ Dropout 0.2 → 0.4
    ✓ Label Smoothing 0.1
    ✓ Mixup (alpha=0.3)
    ✓ Güçlü augmentasyon (Perspective + GaussianBlur)
    ✓ AdamW + katman bazlı LR
    ✓ OneCycleLR (warmup + cosine decay)
    ✓ Gradient Clipping (max_norm=1.0)
    ✓ Weight Decay 1e-4 → 1e-3
"""

import json
import time
from pathlib import Path

import numpy as np
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

# =============================================================================
# CONFIG — Proje yapısına göre ayarlandı
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # scripts/ → proje kökü
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models3"
LOGS_DIR     = PROJECT_ROOT / "logs3"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

REPORT_PATH  = LOGS_DIR / "training_report3.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 32
NUM_WORKERS  = 0       # Windows için 0 — sorunsuz çalışır

EPOCHS_HEAD      = 5   # Phase 1: sadece classifier
EPOCHS_FINETUNE  = 30  # Phase 2: tüm model
LR_HEAD          = 1e-3
LR_FINETUNE      = 5e-5
WEIGHT_DECAY     = 1e-3
EARLY_STOP_PAT   = 7

MIXUP_ALPHA = 0.3      # 0.0 yapınca Mixup kapanır

# =============================================================================
# TRANSFORMS
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.55, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.08),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =============================================================================
# MIXUP
# =============================================================================
def mixup_data(x, y, alpha=0.3):
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(ce, pred, y_a, y_b, lam):
    return lam * ce(pred, y_a) + (1 - lam) * ce(pred, y_b)

# =============================================================================
# UTILS
# =============================================================================
def log(msg: str):
    print(msg)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def evaluate(model, loader):
    """Val/Test değerlendirmesi — label_smoothing YOK (temiz metrik)."""
    model.eval()
    ce       = nn.CrossEntropyLoss()
    correct1 = correct3 = total = 0
    total_loss = 0.0

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += ce(logits, y).item()

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == y).sum().item()

        k = min(3, logits.size(1))
        top_k = logits.topk(k, dim=1).indices
        for i in range(y.size(0)):
            if y[i].item() in top_k[i].tolist():
                correct3 += 1
        total += y.size(0)

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, correct1 / max(1, total), correct3 / max(1, total)


def train_one_epoch(model, loader, optimizer, scheduler=None, use_mixup=False):
    model.train()
    ce      = nn.CrossEntropyLoss(label_smoothing=0.1)
    running = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        if use_mixup and MIXUP_ALPHA > 0:
            mx, ya, yb, lam = mixup_data(x, y, MIXUP_ALPHA)
            loss = mixup_criterion(ce, model(mx), ya, yb, lam)
        else:
            loss = ce(model(x), y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()   # OneCycleLR → her batch'te adım

        running += loss.item()

    return running / max(1, len(loader))

# =============================================================================
# MODEL
# =============================================================================
def build_model(arch: str, num_classes: int) -> nn.Module:
    """Dropout 0.4 ile güçlendirilmiş classifier."""
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(m.last_channel, num_classes),
        )
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, num_classes),
        )
    elif arch == "resnet18":
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(m.fc.in_features, num_classes),
        )
    else:
        raise ValueError(f"Bilinmeyen arch: '{arch}'")
    return m


def freeze_backbone(model, arch: str):
    if arch in ("mobilenet_v2", "efficientnet_b0"):
        for p in model.features.parameters():
            p.requires_grad = False
    elif arch == "resnet18":
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model, arch: str):
    """Backbone 10x daha küçük LR alır → eski bilgiyi korur."""
    if arch in ("mobilenet_v2", "efficientnet_b0"):
        backbone = list(model.features.parameters())
        head     = list(model.classifier.parameters())
    else:
        backbone = [p for n, p in model.named_parameters()
                    if not n.startswith("fc")]
        head     = list(model.fc.parameters())
    return [
        {"params": backbone, "lr": LR_FINETUNE * 0.1},
        {"params": head,     "lr": LR_FINETUNE},
    ]


def save_checkpoint(path, model, class_to_idx, meta):
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     class_to_idx,
        "meta":             meta,
    }, path)


def load_best(path, model):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt

# =============================================================================
# EĞİTİM
# =============================================================================
def train_model(arch, train_loader, val_loader, test_loader, class_to_idx):
    num_classes = len(class_to_idx)
    model       = build_model(arch, num_classes).to(DEVICE)
    best_path   = MODELS_DIR / f"{arch}_best.pth"
    history     = []

    best_val_acc  = -1.0
    patience_left = EARLY_STOP_PAT

    log("\n" + "=" * 72)
    log(f"  MODEL : {arch.upper()}")
    log(f"  Device: {DEVICE}  |  Sınıf sayısı: {num_classes}")
    log(f"  Anti-overfitting: Dropout(0.4) + LS(0.1) + Mixup + AdamW + OC-LR")
    log("=" * 72)

    # ── Phase 1: Sadece classifier (backbone dondurulmuş) ───────────────────
    log(f"\n[{arch}] Phase 1 — HEAD ({EPOCHS_HEAD} epoch)")
    freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = optim.AdamW(head_params, lr=LR_HEAD,
                               weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS_HEAD + 1):
        t0   = time.time()
        tloss = train_one_epoch(model, train_loader, optimizer, use_mixup=False)
        vloss, vacc1, vacc3 = evaluate(model, val_loader)
        sec  = time.time() - t0

        history.append({"phase": "head", "epoch": epoch,
                         "train_loss": tloss, "val_loss": vloss,
                         "val_top1": vacc1, "val_top3": vacc3})

        marker = " ◀ BEST" if vacc1 > best_val_acc else ""
        log(f"  [{arch}][HEAD] {epoch:>2}/{EPOCHS_HEAD} | "
            f"train={tloss:.4f} val={vloss:.4f} | "
            f"top1={vacc1:.3f} top3={vacc3:.3f} | {sec:.1f}s{marker}")

        if vacc1 > best_val_acc:
            best_val_acc  = vacc1
            save_checkpoint(best_path, model, class_to_idx,
                            {"arch": arch, "phase": "head", "epoch": epoch})
            patience_left = EARLY_STOP_PAT
        else:
            patience_left -= 1
            if patience_left == 0:
                log(f"  [{arch}] Erken durdurma (head).")
                break

    # ── Phase 2: Full fine-tune ─────────────────────────────────────────────
    log(f"\n[{arch}] Phase 2 — FINE-TUNE ({EPOCHS_FINETUNE} epoch, Mixup aktif)")
    load_best(best_path, model)
    unfreeze_all(model)

    optimizer = optim.AdamW(get_param_groups(model, arch),
                             weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = [LR_FINETUNE * 0.1, LR_FINETUNE],
        steps_per_epoch = len(train_loader),
        epochs         = EPOCHS_FINETUNE,
        pct_start      = 0.1,
        anneal_strategy = "cos",
        div_factor      = 10.0,
        final_div_factor = 100.0,
    )

    patience_left = EARLY_STOP_PAT

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        t0    = time.time()
        tloss = train_one_epoch(model, train_loader, optimizer,
                                 scheduler=scheduler, use_mixup=True)
        vloss, vacc1, vacc3 = evaluate(model, val_loader)
        sec   = time.time() - t0
        cur_lr = scheduler.get_last_lr()[1]   # head LR'si

        history.append({"phase": "finetune", "epoch": epoch,
                         "train_loss": tloss, "val_loss": vloss,
                         "val_top1": vacc1, "val_top3": vacc3})

        marker = " ◀ BEST" if vacc1 > best_val_acc else ""
        log(f"  [{arch}][FT] {epoch:>2}/{EPOCHS_FINETUNE} | "
            f"train={tloss:.4f} val={vloss:.4f} | "
            f"top1={vacc1:.3f} top3={vacc3:.3f} | "
            f"lr={cur_lr:.1e} | {sec:.1f}s{marker}")

        if vacc1 > best_val_acc:
            best_val_acc  = vacc1
            save_checkpoint(best_path, model, class_to_idx,
                            {"arch": arch, "phase": "finetune", "epoch": epoch})
            patience_left = EARLY_STOP_PAT
        else:
            patience_left -= 1
            if patience_left == 0:
                log(f"  [{arch}] Erken durdurma (finetune, patience={EARLY_STOP_PAT}).")
                break

    # ── Overfitting analizi ─────────────────────────────────────────────────
    if history:
        last = history[-1]
        gap  = last["val_loss"] - last["train_loss"]
        status = "⚠  Hâlâ overfitting" if gap > 0.3 else "✓  Kontrol altında"
        log(f"\n  [{arch}] Overfitting gap (val−train): {gap:+.4f}  {status}")

    # ── Final test ──────────────────────────────────────────────────────────
    best_ckpt = load_best(best_path, model)
    tloss_t, tacc1, tacc3 = evaluate(model, test_loader)
    meta = best_ckpt.get("meta", {})

    log(f"  [{arch}] BEST val top-1 = {best_val_acc:.3f} "
        f"(phase={meta.get('phase','?')} epoch={meta.get('epoch','?')})")
    log(f"  [{arch}] TEST  top-1={tacc1:.3f}  top-3={tacc3:.3f}  loss={tloss_t:.4f}")
    log(f"  [{arch}] Checkpoint → {best_path}")

    return {
        "arch":          arch,
        "best_val_top1": best_val_acc,
        "test_top1":     tacc1,
        "test_top3":     tacc3,
        "best_path":     str(best_path),
        "history":       history,
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    REPORT_PATH.write_text("", encoding="utf-8")

    log("=" * 72)
    log("  Yiyecek Tanıma — Eğitim Raporu  v2  (Anti-Overfitting)")
    log(f"  Proje : {PROJECT_ROOT}")
    log(f"  Cihaz : {DEVICE}")
    log("=" * 72)
    log("\nDeğişiklikler (v1 → v2):")
    log("  [1] Dropout          : 0.2 → 0.4")
    log("  [2] Label Smoothing  : yok → 0.1  (eğitimde)")
    log("  [3] Mixup            : yok → alpha=0.3  (fine-tune)")
    log("  [4] Augmentasyon     : +Perspective +GaussianBlur +agresif crop")
    log("  [5] Optimizer        : Adam → AdamW")
    log("  [6] LR stratejisi    : tek LR → katman bazlı (backbone 10x yavaş)")
    log("  [7] Scheduler        : CosineAnnealing → OneCycleLR (warmup)")
    log("  [8] Weight Decay     : 1e-4 → 1e-3")
    log("  [9] Gradient Clipping: yok → max_norm=1.0")
    log(" [10] Early stop pat.  : 4 → 7")

    # Veri setleri
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tfms)

    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, (
        "HATA: Train / Val / Test sınıf sıralamaları uyuşmuyor!\n"
        f"  Train : {train_ds.class_to_idx}\n"
        f"  Val   : {val_ds.class_to_idx}\n"
        f"  Test  : {test_ds.class_to_idx}"
    )

    class_to_idx = train_ds.class_to_idx
    log(f"\nSınıflar ({len(train_ds.classes)}): {train_ds.classes}")
    log(f"Train : {len(train_ds):>5} görüntü")
    log(f"Val   : {len(val_ds):>5} görüntü")
    log(f"Test  : {len(test_ds):>5} görüntü")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    results = []
    for arch in ("mobilenet_v2", "efficientnet_b0", "resnet18"):
        results.append(
            train_model(arch, train_loader, val_loader, test_loader, class_to_idx)
        )

    # ── Özet tablo ─────────────────────────────────────────────────────────
    log("\n" + "=" * 72)
    log("  SONUÇ ÖZETİ")
    log("=" * 72)
    log(f"  {'Model':<20} {'Val Top-1':>10} {'Test Top-1':>10} {'Test Top-3':>10}")
    log("  " + "-" * 55)
    best_result = max(results, key=lambda r: r["test_top1"])
    for r in results:
        star = " ★" if r["arch"] == best_result["arch"] else ""
        log(f"  {r['arch']:<20} {r['best_val_top1']:>10.3f} "
            f"{r['test_top1']:>10.3f} {r['test_top3']:>10.3f}{star}")

    log(f"\n  En iyi model: {best_result['arch']}  "
        f"(Test Top-1={best_result['test_top1']:.3f})")
    log(f"  API için: python scripts/api.py "
        f"--arch {best_result['arch']} "
        f"--model models1/{best_result['arch']}_best.pth")

    # JSON
    json_path = LOGS_DIR / "training_report.json"
    # history içindeki numpy float'larını temizle
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean(results), f, ensure_ascii=False, indent=2)

    log(f"\n  Rapor  → {REPORT_PATH}")
    log(f"  JSON   → {json_path}")
    log("=" * 72)


if __name__ == "__main__":
    main()