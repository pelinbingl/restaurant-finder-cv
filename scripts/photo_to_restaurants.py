import os
import requests
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

# =========================
# PATHS / AYARLAR
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_food.pth"

# Ortam değişkeninden al (önerilen)
API_KEY = "AIzaSyC5RgYSBsn0p5xFNmkr0atpL6-G9Oqby7o"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Yemek sınıfı -> arama sorguları (TR/EN karışık daha iyi sonuç verir)
QUERY_MAP = {
    "cheesecake": ["cheesecake", "cheesecake pasta", "cheesecake tatlı"],
    "pizza": ["pizza", "pizza restaurant"],
    "hamburger": ["burger", "hamburger"],
    "ramen": ["ramen", "ramen restaurant"],
    "sushi": ["sushi", "sushi restaurant"],
    "tacos": ["tacos", "taco", "mexican food", "mexican restaurant"],
    "steak": ["steak", "steakhouse", "et restoranı"],
    "spaghetti_bolognese": ["bolognese", "spaghetti", "italian restaurant", "italyan restoran"],
    "caesar_salad": ["caesar salad", "salad", "healthy food", "sağlıklı yemek"],
}

# =========================
# YARDIMCI: Mesafe (km)
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# =========================
# 1) Model yükle + etiketler
# =========================
def load_model_and_labels():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, idx_to_class

def predict_topk(model, idx_to_class, image_path: Path, k: int = 3):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(str(image_path)).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k)

    preds = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        preds.append((idx_to_class[idx], float(score)))
    return preds

# =========================
# 2) Google Places: Text Search
# =========================
def places_text_search(query: str, lat: float, lng: float, radius_m: float = 5000.0, limit: int = 10):
    if not API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY ortam değişkeni yok. Önce set etmelisin.")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        # İstediğimiz alanlar
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount"
    }

    payload = {
        "textQuery": f"{query} near me",
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": radius_m
            }
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)

    # Daha anlaşılır hata için (403 vs.) JSON mesajını da gösterelim
    if r.status_code != 200:
        raise RuntimeError(f"Places API hata: {r.status_code}\n{r.text[:800]}")

    data = r.json()
    places = data.get("places", [])[:limit]

    results = []
    for p in places:
        name = p.get("displayName", {}).get("text", "")
        address = p.get("formattedAddress", "")
        rating = p.get("rating", 0.0)
        urc = p.get("userRatingCount", 0)

        loc = p.get("location", {})
        plat = loc.get("latitude")
        plng = loc.get("longitude")

        dist = 9999.0
        if plat is not None and plng is not None:
            dist = haversine_km(lat, lng, plat, plng)

        results.append({
            "name": name,
            "address": address,
            "rating": float(rating) if rating is not None else 0.0,
            "user_ratings_total": int(urc) if urc is not None else 0,
            "distance_km": dist,
            "query_used": query
        })

    return results

def rank_results(results):
    # Sıralama: yakınlık (küçük) + rating (büyük) + yorum sayısı (büyük)
    results.sort(key=lambda x: (x["distance_km"], -(x["rating"]), -(x["user_ratings_total"])))
    return results

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import sys

    # Kullanım:
    # python scripts/photo_to_restaurants.py "D:\...\foto.jpg" 41.0082 28.9784
    if len(sys.argv) < 4:
        print("Kullanım: python scripts/photo_to_restaurants.py <image_path> <lat> <lng>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    lat = float(sys.argv[2])
    lng = float(sys.argv[3])

    model, idx_to_class = load_model_and_labels()
    top3 = predict_topk(model, idx_to_class, image_path, k=3)

    print("\n📸 Fotoğraf tahmini (Top-3):")
    for dish, score in top3:
        print(f"  - {dish} : {score:.3f}")

    # Top-1 seç
    dish_top1, p_top1 = top3[0]
    print(f"\n🍽️ Seçilen yemek (Top-1): {dish_top1}  (güven: {p_top1:.3f})")

    # Bu sınıf için birden fazla arama sorgusu kullan
    queries = QUERY_MAP.get(dish_top1, [dish_top1])

    all_results = []
    for q in queries:
        all_results.extend(places_text_search(q, lat, lng, radius_m=5000.0, limit=10))

    # Aynı yerleri birleştir (isim + adres)
    seen = set()
    uniq = []
    for r in all_results:
        key = (r["name"].strip().lower(), r["address"].strip().lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    restaurants = rank_results(uniq)[:10]

    print("\n📍 Yakındaki öneriler (mesafe + yıldız + yorum):\n")
    for i, r in enumerate(restaurants, 1):
        print(f"{i}. {r['name']}  ⭐{r['rating']:.1f}  ({r['user_ratings_total']} yorum)  📍{r['distance_km']:.2f} km")
        print(f"   {r['address']}")
        print(f"   (sorgu: {r['query_used']})\n")
