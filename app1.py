# app.py
import os
import re
import requests
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from streamlit_js_eval import streamlit_js_eval

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models1" / "efficientnet_b0_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

API_KEY = "AIzaSyC5RgYSBsn0p5xFNmkr0atpL6-G9Oqby7o"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# =========================
# 1) CLASS -> CATEGORY MAP
# =========================
CLASS_TO_CATEGORY = {
    # dessert / cafe-pastane
    "baklava": "cafe",
    "cheesecake": "cafe",
    "tiramisu": "cafe",
    "donuts": "cafe",
    "ice_cream": "cafe",
    "chocolate_cake": "cafe",
    "chocolate_mousse": "cafe",
    "red_velvet_cake": "cafe",

    # brunch / kahvaltı
    "pancakes": "brunch",
    "waffles": "brunch",
    "french_toast": "brunch",
    "omelette": "brunch",
    "eggs_benedict": "brunch",

    # fastfood
    "hamburger": "fastfood",
    "hot_dog": "fastfood",
    "french_fries": "fastfood",
    "nachos": "fastfood",
    "club_sandwich": "fastfood",
    "grilled_cheese_sandwich": "fastfood",

    # restaurant
    "lasagna": "restaurant",
    "ravioli": "restaurant",
    "risotto": "restaurant",
    "gnocchi": "restaurant",
    "garlic_bread": "restaurant",
    "spaghetti_bolognese": "restaurant",
    "spaghetti_carbonara": "restaurant",
    "pad_thai": "restaurant",
    "dumplings": "restaurant",
    "spring_rolls": "restaurant",
    "miso_soup": "restaurant",
    "ramen": "restaurant",
    "sushi": "restaurant",
    "sashimi": "restaurant",
    "pho": "restaurant",
    "pizza": "restaurant",
}

DESSERT_CLASSES = {
    "baklava","cheesecake","tiramisu","donuts","ice_cream",
    "chocolate_cake","chocolate_mousse","red_velvet_cake",
}

# Yemek -> TR query ipuçları
CLASS_TO_QUERY_HINTS = {
    "baklava": ["baklava", "baklavacı", "tatlıcı"],
    "cheesecake": ["cheesecake", "tatlı", "kafe"],
    "tiramisu": ["tiramisu", "tatlı", "kafe"],
    "donuts": ["donut", "donuts", "kafe", "pastane"],
    "ice_cream": ["dondurma", "dondurmacı"],
    "chocolate_cake": ["çikolatalı pasta", "pasta", "pastane"],
    "chocolate_mousse": ["mousse", "tatlı", "kafe"],
    "red_velvet_cake": ["red velvet", "tatlı", "kafe"],

    "pancakes": ["pankek", "kahvaltı", "brunch"],
    "waffles": ["waffle", "brunch", "kahvaltı"],
    "french_toast": ["french toast", "brunch", "kahvaltı"],
    "omelette": ["omlet", "kahvaltı"],
    "eggs_benedict": ["eggs benedict", "brunch"],

    "hamburger": ["burger", "hamburger"],
    "hot_dog": ["hot dog", "sosisli"],
    "french_fries": ["patates kızartması", "fast food"],
    "nachos": ["nachos"],
    "club_sandwich": ["club sandwich", "sandviç"],
    "grilled_cheese_sandwich": ["grilled cheese", "tost"],

    "ramen": ["ramen"],
    "sushi": ["sushi"],
    "pho": ["pho"],
    "pizza": ["pizza"],
    "pad_thai": ["pad thai"],
    "dumplings": ["dumpling", "mantı", "gyoza"],
    "spring_rolls": ["spring roll", "çin böreği"],
    "miso_soup": ["miso"],
    "sashimi": ["sashimi"],
    "lasagna": ["lazanya"],
    "ravioli": ["ravioli"],
    "risotto": ["risotto"],
    "gnocchi": ["gnocchi"],
    "garlic_bread": ["garlic bread", "sarımsaklı ekmek"],
    "spaghetti_bolognese": ["bolonez", "spagetti"],
    "spaghetti_carbonara": ["carbonara", "spagetti"],
}

# =========================
# 2) CATEGORY RULES (Places)
# =========================
CATEGORY_RULES = {
    "cafe": {
        "included_types": ["cafe", "bakery"],
        "queries": ["kafe", "kahve", "pastane", "tatlıcı"],
        "block_keywords": ["sushi", "ocakbaşı", "kebap", "steak", "ramen"],
    },
    "brunch": {
        "included_types": ["cafe", "restaurant"],
        "queries": ["kahvaltı", "brunch", "serpme kahvaltı"],
        "block_keywords": ["sushi", "ocakbaşı", "kebap", "ramen"],
    },
    "fastfood": {
        "included_types": ["fast_food_restaurant", "meal_takeaway", "restaurant"],
        "queries": ["fast food", "burger", "hamburger", "pizza", "sandviç"],
        "block_keywords": ["pastane", "tatlı", "baklava", "dessert"],
    },
    "restaurant": {
        "included_types": ["restaurant"],
        "queries": ["restoran", "yemek", "lokanta", "ızgara"],
        "block_keywords": [],
    },
}

# =========================
# UTILS
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _safe_lower(s: str) -> str:
    return (s or "").strip().lower()

def choose_category_from_top3(top3):
    scores = {"cafe": 0.0, "brunch": 0.0, "fastfood": 0.0, "restaurant": 0.0}
    weights = [1.0, 0.7, 0.5]
    dessert_score = 0.0

    for i, (cls, prob) in enumerate(top3[:3]):
        cat = CLASS_TO_CATEGORY.get(cls, "restaurant")
        w = weights[i] if i < len(weights) else 0.5
        scores[cat] += float(prob) * w
        if cls in DESSERT_CLASSES:
            dessert_score += float(prob) * w

    top1_prob = float(top3[0][1]) if top3 else 0.0
    has_dessert = any(cls in DESSERT_CLASSES for cls, _ in top3[:3])

    if dessert_score >= 0.35:
        chosen = "cafe"
    elif has_dessert and top1_prob < 0.60:
        chosen = "cafe"
    else:
        chosen = max(scores, key=scores.get)

    return chosen, scores, dessert_score

def build_queries_from_top3(top3, category):
    rules = CATEGORY_RULES.get(category, CATEGORY_RULES["restaurant"])
    base_queries = list(rules["queries"])
    primary_intent = base_queries[0] if base_queries else category

    dish_terms = []
    for dish, _prob in top3[:3]:
        dish_terms.extend(CLASS_TO_QUERY_HINTS.get(dish, [dish.replace("_", " ")]))

    combined = []
    for t in dish_terms[:5]:
        combined.append(f"{t} {primary_intent}")
        combined.append(f"{t}")

    combined.extend(base_queries)

    seen, out = set(), []
    for q in combined:
        ql = q.strip().lower()
        if ql and ql not in seen:
            seen.add(ql)
            out.append(q)

    return out[:8]

def _name_address_text(place):
    return f"{place.get('name','')} {place.get('address','')}".lower()

def _extra_filter_for_category(category, place):
    txt = _name_address_text(place)

    if category == "cafe":
        bad = [r"\bkebap\b", r"\bocakbaşı\b", r"\bsteak\b", r"\bızgara\b", r"\bsushi\b", r"\bramen\b"]
        if any(re.search(p, txt) for p in bad):
            return False

    if category == "fastfood":
        bad = [r"\bpastane\b", r"\btatlı\b", r"\bbaklava\b", r"\bdessert\b"]
        if any(re.search(p, txt) for p in bad):
            return False

    return True

def places_text_search(category: str, lat: float, lng: float, top3=None, radius_m: float = 5000.0, limit: int = 10):
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        st.warning("GOOGLE_MAPS_API_KEY yok. Places araması çalışmayacak.")
        return [], []

    rules = CATEGORY_RULES.get(category, CATEGORY_RULES["restaurant"])
    queries = build_queries_from_top3(top3 or [], category)
    included_types = rules.get("included_types", []) or []
    block_keywords = set(rules.get("block_keywords", []) or [])

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount,places.types"
        ),
    }

    all_results = []

    for q in queries:
        type_candidates = included_types if included_types else [None]

        for t in type_candidates:
            payload = {
                "textQuery": q,
                "languageCode": "tr",
                "regionCode": "TR",
                "maxResultCount": min(limit, 20),
                "locationBias": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": float(radius_m),
                    }
                },
            }
            if t is not None:
                payload["includedType"] = t

            try:
                r = requests.post(url, headers=headers, json=payload, timeout=25)
            except Exception:
                continue

            if r.status_code != 200 and "includedType" in payload:
                payload.pop("includedType", None)
                try:
                    r = requests.post(url, headers=headers, json=payload, timeout=25)
                except Exception:
                    continue

            if r.status_code != 200:
                continue

            data = r.json()
            places = data.get("places", [])[:limit]

            for p in places:
                name = p.get("displayName", {}).get("text", "")
                address = p.get("formattedAddress", "")
                rating = float(p.get("rating", 0.0) or 0.0)
                urc = int(p.get("userRatingCount", 0) or 0)
                types = p.get("types", []) or []

                loc = p.get("location", {})
                plat, plng = loc.get("latitude"), loc.get("longitude")

                dist = 9999.0
                if plat is not None and plng is not None:
                    dist = haversine_km(lat, lng, plat, plng)

                all_results.append({
                    "name": name,
                    "address": address,
                    "rating": rating,
                    "user_ratings_total": urc,
                    "distance_km": dist,
                    "query_used": q,
                    "type_used": t,
                    "types": types,
                    "lat": plat,
                    "lng": plng,
                })

    # uniq
    seen, uniq = set(), []
    for r in all_results:
        key = (_safe_lower(r["name"]), _safe_lower(r["address"]))
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    # filter
    filtered = []
    for r in uniq:
        txt = _name_address_text(r)

        if any(bk in txt for bk in block_keywords):
            continue

        if not _extra_filter_for_category(category, r):
            continue

        if category == "cafe":
            tset = set(r.get("types") or [])
            if "restaurant" in tset and "cafe" not in tset and "bakery" not in tset:
                continue

        filtered.append(r)

    filtered.sort(key=lambda x: (x["distance_km"], -x["rating"], -x["user_ratings_total"]))
    return filtered[:limit], queries

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, len(class_to_idx))
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE)
    m.eval()
    return m, idx_to_class

def predict_top3(model, idx_to_class, img: Image.Image):
    x = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, 3)

    out = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        out.append((idx_to_class[idx], float(score)))
    return out

def get_browser_location():
    js = """
    new Promise((resolve) => {
      if (!navigator.geolocation) {
        resolve({ok:false, error:"Tarayıcı geolocation desteklemiyor."});
      } else {
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve({
            ok:true,
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            accuracy: pos.coords.accuracy
          }),
          (err) => resolve({ok:false, error: err.message}),
          { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
        );
      }
    })
    """
    return streamlit_js_eval(js_expressions=js, key="geo_eval")

# =========================
# UI
# =========================
st.set_page_config(page_title="Food → Place Finder", layout="centered")
st.title(" Food → Mekan Önerisi")
st.caption("Fotoğraftan yemeği tahmin eder, doğru mekan tipine yönlendirip yakındaki önerileri listeler.")

# Session defaults
if "lat" not in st.session_state:
    st.session_state.lat = 41.0082
if "lng" not in st.session_state:
    st.session_state.lng = 28.9784
if "accuracy_m" not in st.session_state:
    st.session_state.accuracy_m = None
if "loc_status" not in st.session_state:
    st.session_state.loc_status = "Henüz konum alınmadı."
if "loc_pending" not in st.session_state:
    st.session_state.loc_pending = False

st.subheader(" Konum")
c1, c2, c3 = st.columns([1.1, 1.1, 2.2])

with c1:
    if st.button("Konumumu al", use_container_width=True):
        st.session_state.loc_pending = True
        st.session_state.loc_status = "Konum isteniyor... Tarayıcı iznini kontrol et."
        st.rerun()

with c2:
    if st.button("Yenile", use_container_width=True):
        st.session_state.loc_pending = True
        st.session_state.loc_status = "Konum yeniden isteniyor..."
        st.rerun()

with c3:
    if "✅" in st.session_state.loc_status:
        st.success(st.session_state.loc_status)
    elif "alınamadı" in st.session_state.loc_status.lower():
        st.error(st.session_state.loc_status)
    else:
        st.info(st.session_state.loc_status)

if st.session_state.loc_pending:
    loc = get_browser_location()

    if loc is None:
        st.warning("Konum henüz gelmedi. İzin verdikten sonra 1 kez daha 'Yenile'ye bas.")
    elif isinstance(loc, dict) and loc.get("ok") is False:
        st.session_state.loc_pending = False
        st.session_state.loc_status = f"Konum alınamadı: {loc.get('error','Bilinmeyen hata')}"
        st.rerun()
    elif isinstance(loc, dict) and loc.get("ok") is True:
        st.session_state.loc_pending = False
        st.session_state.lat = float(loc["latitude"])
        st.session_state.lng = float(loc["longitude"])
        st.session_state.accuracy_m = float(loc.get("accuracy", 0.0)) if loc.get("accuracy") else None

        if st.session_state.accuracy_m is not None:
            st.session_state.loc_status = (
                f" Konum alındı ({st.session_state.lat:.6f}, {st.session_state.lng:.6f}) "
                f"(±{st.session_state.accuracy_m:.0f}m)"
            )
        else:
            st.session_state.loc_status = f"Konum alındı ({st.session_state.lat:.6f}, {st.session_state.lng:.6f})"

        st.rerun()
    else:
        st.session_state.loc_pending = False
        st.session_state.loc_status = "Konum okunamadı. 'Yenile' ile tekrar dene."
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    st.session_state.lat = float(st.number_input("Enlem (Latitude)", value=float(st.session_state.lat), format="%.6f"))
with col2:
    st.session_state.lng = float(st.number_input("Boylam (Longitude)", value=float(st.session_state.lng), format="%.6f"))

radius_km = st.slider("Arama yarıçapı (km)", min_value=1, max_value=20, value=5)
limit = st.slider("Kaç mekan listelensin?", min_value=5, max_value=20, value=10)

st.divider()
uploaded = st.file_uploader("Bir yemek fotoğrafı yükle (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Yüklenen foto", use_container_width=True)

    model, idx_to_class = load_model()
    top3 = predict_top3(model, idx_to_class, img)

    st.subheader(" Tahmin (Top-3)")
    for dish, p in top3:
        st.write(f"- **{dish}** : `{p:.3f}`")

    chosen_cat, _cat_scores, _dessert_score = choose_category_from_top3(top3)

    st.subheader("Sistem Kararı")
    emoji_map = {"cafe": "☕", "brunch": "🥞", "fastfood": "🍔", "restaurant": "🍽️"}
    st.markdown(
        f"""
        <div style="
            padding:18px;
            border-radius:12px;
            background: linear-gradient(135deg, #1f2937, #111827);
            border:1px solid #374151;
            text-align:center;
            font-size:20px;
            font-weight:700;">
            {emoji_map.get(chosen_cat,'🍽️')} {chosen_cat.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(" Yakındaki öneriler")
    final_cat = chosen_cat  # ✅ dropdown yok
    results, used_queries = places_text_search(
        category=final_cat,
        lat=st.session_state.lat,
        lng=st.session_state.lng,
        top3=top3,
        radius_m=float(radius_km) * 1000.0,
        limit=limit,
    )

    if not results:
        st.warning("Sonuç bulunamadı. (API key / konum / radius / sorgu kaynaklı olabilir)")
        if API_KEY == "YOUR_API_KEY_HERE":
            st.info("API key’i env ile ver: setx GOOGLE_MAPS_API_KEY \"KEY\"")
        st.caption(f"Kullanılan sorgular: {used_queries}")
    else:
        for i, r in enumerate(results, 1):
            st.markdown(f"### {i}. {r['name']}")
            st.markdown(
                f"""
- ⭐ **{r['rating']:.1f}** ({r['user_ratings_total']} yorum)
- 📍 **{r['distance_km']:.2f} km**
- 🏠 {r['address']}
                """.strip()
            )

            if r.get("lat") is not None and r.get("lng") is not None:
                maps_url = f"https://www.google.com/maps/search/?api=1&query={r['lat']},{r['lng']}"
                st.markdown(f"[📌 Google Maps'te Aç]({maps_url})")

            st.markdown("---")
else:
    st.caption("Başlamak için bir fotoğraf yükle.")