import json
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_JSON = PROJECT_ROOT / "logs" / "training_report.json"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def load_results():
    with open(LOG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_model_curves(model_result: dict):
    arch = model_result["arch"]
    hist = model_result["history"]

    # Phase + epoch bilgisiyle çizim için X label üretelim
    x_labels = []
    train_loss = []
    val_loss = []
    val_top1 = []
    val_top3 = []

    for row in hist:
        phase = row.get("phase", "?")
        ep = row.get("epoch", 0)
        x_labels.append(f"{phase[:2]}{ep}")  # he1, he2, fi1, fi2...
        train_loss.append(row["train_loss"])
        val_loss.append(row["val_loss"])
        val_top1.append(row["val_top1"])
        val_top3.append(row["val_top3"])

    # -----------------------
    # 1) Loss plot
    # -----------------------
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("epoch (head/finetune)")
    plt.ylabel("loss")
    plt.title(f"{arch} - Loss Curves")
    plt.legend()
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{arch}_loss_curves.png")
    plt.close()

    # -----------------------
    # 2) Accuracy plot
    # -----------------------
    plt.figure()
    plt.plot(val_top1, label="val_top1")
    plt.plot(val_top3, label="val_top3")
    plt.xlabel("epoch (head/finetune)")
    plt.ylabel("accuracy")
    plt.title(f"{arch} - Validation Accuracy")
    plt.legend()
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{arch}_val_accuracy.png")
    plt.close()

def plot_summary_bar(results):
    arches = [r["arch"] for r in results]
    best_val = [r["best_val_top1"] for r in results]
    test1 = [r["test_top1"] for r in results]
    test3 = [r["test_top3"] for r in results]

    # Best Val Top-1
    plt.figure()
    plt.bar(arches, best_val)
    plt.ylim(0, 1)
    plt.xlabel("model")
    plt.ylabel("best_val_top1")
    plt.title("Best Validation Top-1 (Comparison)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_best_val_top1.png")
    plt.close()

    # Test Top-1
    plt.figure()
    plt.bar(arches, test1)
    plt.ylim(0, 1)
    plt.xlabel("model")
    plt.ylabel("test_top1")
    plt.title("Test Top-1 (Comparison)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_test_top1.png")
    plt.close()

    # Test Top-3
    plt.figure()
    plt.bar(arches, test3)
    plt.ylim(0, 1)
    plt.xlabel("model")
    plt.ylabel("test_top3")
    plt.title("Test Top-3 (Comparison)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "compare_test_top3.png")
    plt.close()

def main():
    if not LOG_JSON.exists():
        print(f"❌ Bulunamadı: {LOG_JSON}")
        print("Önce eğitim scriptini çalıştırıp logs/training_report.json üretmelisin.")
        return

    results = load_results()

    for r in results:
        plot_model_curves(r)

    plot_summary_bar(results)

    print("✅ Grafikler kaydedildi:")
    print(f"   {PLOTS_DIR}")

if __name__ == "__main__":
    main()
