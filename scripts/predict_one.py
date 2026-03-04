import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_food.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kullanım: python scripts/predict_one.py path/to/image.jpg
img_path = r"D:\Projeler\Python\restaurant_finder_cv\data\test\cheesecake\192339.jpg"

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Checkpointten sınıf map'i al
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, len(class_to_idx))
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE)
model.eval()

img = Image.open(img_path).convert("RGB")
x = tfms(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top3 = torch.topk(probs, 3)

print("Top-3 predictions:")
for score, idx in zip(top3.values.tolist(), top3.indices.tolist()):
    print(f"- {idx_to_class[idx]} : {score:.3f}")
