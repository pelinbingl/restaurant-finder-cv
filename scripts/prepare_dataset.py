import random
import shutil
from pathlib import Path

# =========================
# AYARLAR
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOOD101_DIR = PROJECT_ROOT / "food101"
OUT_DIR = PROJECT_ROOT / "data"

CLASSES = [
    "baklava",
    "chocolate_cake",
    "club_sandwich",
    "donuts",
    "dumplings",
    "eggs_benedict",
    "french_fries",
    "french_toast",
    "garlic_bread",
    "gnocchi",
    "grilled_cheese_sandwich",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "lasagna",
    "miso_soup",
    "nachos",
    "omelette",
    "pad_thai",
    "pancakes",
    "pizza",
    "ramen",
    "ravioli",
    "risotto",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "sushi",
    "tiramisu",
    "waffles",
]

N_TRAIN = 400
N_VAL   = 50
N_TEST  = 50

random.seed(42)


# =========================
# YARDIMCI FONKSİYONLAR
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_images(folder: Path):
    # .jpg ve .jpeg uzantılarını destekle
    imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    return imgs


def clean_folder(folder: Path):
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()


# =========================
# ANA İŞLEM
# =========================

def prepare():
    print("Başlıyor...\n")

    missing = []
    insufficient = []

    for cls in CLASSES:
        print(f"İşleniyor: {cls}")

        source_folder = FOOD101_DIR / cls

        if not source_folder.exists():
            print(f"  ❌ Klasör bulunamadı: {source_folder}")
            missing.append(cls)
            continue

        images = get_images(source_folder)
        total_needed = N_TRAIN + N_VAL + N_TEST

        if len(images) < total_needed:
            print(f"  ❌ Yeterli görsel yok: {len(images)} < {total_needed}")
            insufficient.append((cls, len(images)))
            continue

        random.shuffle(images)

        train_imgs = images[:N_TRAIN]
        val_imgs   = images[N_TRAIN : N_TRAIN + N_VAL]
        test_imgs  = images[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]

        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = OUT_DIR / split / cls
            ensure_dir(split_dir)
            clean_folder(split_dir)
            for img in imgs:
                shutil.copy(img, split_dir / img.name)

        print(f"  ✅ {cls}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Özet rapor
    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)

    successful = len(CLASSES) - len(missing) - len(insufficient)
    print(f"✅ Başarılı  : {successful}/{len(CLASSES)} sınıf")

    if missing:
        print(f"❌ Bulunamadı: {missing}")

    if insufficient:
        for cls, count in insufficient:
            print(f"❌ Yetersiz  : {cls} ({count} görsel, en az {N_TRAIN + N_VAL + N_TEST} gerekli)")

    print(f"\nVeri seti hazırlandı → {OUT_DIR}")


if __name__ == "__main__":
    prepare()