import os
import requests
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from PIL import Image, ImageFile
import base64, io

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import transforms, models
from streamlit_js_eval import streamlit_js_eval

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "food_model_b2.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_KEY      = os.environ.get("GOOGLE_MAPS_API_KEY", "")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CLASSES = [
    "asure", "baklava", "biber_dolmasi", "borek", "chocolate_cake",
    "cig_kofte", "club_sandwich", "donuts", "dumplings", "eggs_benedict",
    "enginar", "et_sote", "french_fries", "french_toast", "garlic_bread",
    "gnocchi", "gozleme", "grilled_cheese_sandwich", "hamburger", "hamsi",
    "hot_dog", "hunkar_begendi", "ice_cream", "icli_kofte", "ispanak",
    "izmir_kofte", "karniyarik", "kebap", "kisir", "kuru_fasulye",
    "lahmacun", "lasagna", "lokum", "manti", "miso_soup",
    "mucver", "nachos", "omelette", "pad_thai", "pancakes",
    "pirinc_pilavi", "pizza", "ramen", "ravioli", "risotto",
    "simit", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "sushi",
    "taze_fasulye", "tiramisu", "waffles", "yaprak_sarma"
]

CLASS_TR = {
    "asure": "Aşure", "baklava": "Baklava", "biber_dolmasi": "Biber Dolması",
    "borek": "Börek", "chocolate_cake": "Çikolatalı Kek", "cig_kofte": "Çiğ Köfte",
    "club_sandwich": "Club Sandwich", "donuts": "Donut", "dumplings": "Dumpling",
    "eggs_benedict": "Eggs Benedict", "enginar": "Enginar", "et_sote": "Et Sote",
    "french_fries": "Patates Kızartması", "french_toast": "Fransız Tostu",
    "garlic_bread": "Sarımsaklı Ekmek", "gnocchi": "Gnocchi", "gozleme": "Gözleme",
    "grilled_cheese_sandwich": "Izgara Peynirli Sandviç", "hamburger": "Hamburger",
    "hamsi": "Hamsi", "hot_dog": "Hot Dog", "hunkar_begendi": "Hünkar Beğendi",
    "ice_cream": "Dondurma", "icli_kofte": "İçli Köfte", "ispanak": "Ispanak",
    "izmir_kofte": "İzmir Köfte", "karniyarik": "Karnıyarık", "kebap": "Kebap",
    "kisir": "Kısır", "kuru_fasulye": "Kuru Fasulye", "lahmacun": "Lahmacun",
    "lasagna": "Lazanya", "lokum": "Lokum", "manti": "Mantı",
    "miso_soup": "Miso Çorbası", "mucver": "Mücver", "nachos": "Nachos",
    "omelette": "Omlet", "pad_thai": "Pad Thai", "pancakes": "Pankek",
    "pirinc_pilavi": "Pilav", "pizza": "Pizza", "ramen": "Ramen",
    "ravioli": "Ravioli", "risotto": "Risotto", "simit": "Simit",
    "spaghetti_bolognese": "Spagetti Bolonez", "spaghetti_carbonara": "Spagetti Carbonara",
    "spring_rolls": "Bahar Rulo", "sushi": "Sushi", "taze_fasulye": "Taze Fasulye",
    "tiramisu": "Tiramisu", "waffles": "Waffle", "yaprak_sarma": "Yaprak Sarma"
}

DESSERT_CLASSES = {
    "baklava", "chocolate_cake", "donuts", "ice_cream",
    "tiramisu", "lokum", "waffles", "pancakes",
}

CLASS_TO_CATEGORY = {
    "baklava": "cafe", "chocolate_cake": "cafe", "donuts": "cafe",
    "ice_cream": "cafe", "tiramisu": "cafe", "lokum": "cafe",
    "asure": "cafe", "borek": "cafe", "gozleme": "cafe", "simit": "cafe",
    "waffles": "brunch", "pancakes": "brunch", "french_toast": "brunch",
    "omelette": "brunch", "eggs_benedict": "brunch",
    "hamburger": "fastfood", "hot_dog": "fastfood", "french_fries": "fastfood",
    "nachos": "fastfood", "club_sandwich": "fastfood",
    "grilled_cheese_sandwich": "fastfood", "cig_kofte": "fastfood",
    "pizza": "restaurant", "ramen": "restaurant", "sushi": "restaurant",
    "lasagna": "restaurant", "ravioli": "restaurant", "risotto": "restaurant",
    "gnocchi": "restaurant", "spaghetti_bolognese": "restaurant",
    "spaghetti_carbonara": "restaurant", "pad_thai": "restaurant",
    "dumplings": "restaurant", "spring_rolls": "restaurant",
    "miso_soup": "restaurant", "kebap": "restaurant", "lahmacun": "restaurant",
    "manti": "restaurant", "karniyarik": "restaurant", "hunkar_begendi": "restaurant",
    "izmir_kofte": "restaurant", "icli_kofte": "restaurant",
    "kuru_fasulye": "restaurant", "hamsi": "restaurant", "et_sote": "restaurant",
    "biber_dolmasi": "restaurant", "yaprak_sarma": "restaurant",
    "kisir": "restaurant", "enginar": "restaurant", "ispanak": "restaurant",
    "mucver": "restaurant", "taze_fasulye": "restaurant",
    "pirinc_pilavi": "restaurant", "garlic_bread": "restaurant",
}

CLASS_TO_QUERY_HINTS = {
    "baklava": ["baklava", "baklavacı", "tatlıcı"],
    "chocolate_cake": ["çikolatalı pasta", "pastane"],
    "donuts": ["donut", "pastane"],
    "ice_cream": ["dondurma", "dondurmacı"],
    "tiramisu": ["tiramisu", "kafe"],
    "lokum": ["lokum", "tatlıcı"],
    "borek": ["börek", "börekçi"],
    "gozleme": ["gözleme"],
    "simit": ["simit", "simitçi"],
    "waffles": ["waffle", "kahvaltı"],
    "pancakes": ["pankek", "kahvaltı"],
    "french_toast": ["kahvaltı", "brunch"],
    "omelette": ["omlet", "kahvaltı"],
    "eggs_benedict": ["brunch", "kahvaltı"],
    "hamburger": ["burger", "hamburger"],
    "hot_dog": ["hot dog"],
    "french_fries": ["fast food"],
    "nachos": ["meksika restoran"],
    "club_sandwich": ["sandviç", "kafe"],
    "grilled_cheese_sandwich": ["tost"],
    "cig_kofte": ["çiğ köfte"],
    "pizza": ["pizza"],
    "ramen": ["ramen", "japon restoran"],
    "sushi": ["sushi", "japon restoran"],
    "lasagna": ["italyan restoran"],
    "ravioli": ["italyan restoran"],
    "risotto": ["italyan restoran"],
    "spaghetti_bolognese": ["spagetti", "italyan restoran"],
    "spaghetti_carbonara": ["carbonara", "italyan restoran"],
    "pad_thai": ["tayland restoran"],
    "dumplings": ["dumpling", "çin restoran"],
    "spring_rolls": ["çin restoran"],
    "miso_soup": ["japon restoran"],
    "kebap": ["kebap", "ocakbaşı"],
    "lahmacun": ["lahmacun", "pide"],
    "manti": ["mantı"],
    "karniyarik": ["ev yemeği", "lokanta"],
    "hunkar_begendi": ["türk mutfağı"],
    "izmir_kofte": ["köfte"],
    "icli_kofte": ["türk mutfağı"],
    "kuru_fasulye": ["ev yemeği", "lokanta"],
    "hamsi": ["balıkçı"],
    "et_sote": ["türk mutfağı"],
    "pirinc_pilavi": ["lokanta"],
}

CATEGORY_RULES = {
    "cafe": {
        "queries": ["kafe", "kahve", "pastane", "tatlıcı"],
        "block_keywords": ["sushi", "ocakbaşı", "kebap", "steak", "ramen"],
    },
    "brunch": {
        "queries": ["kahvaltı", "brunch", "serpme kahvaltı"],
        "block_keywords": ["sushi", "ocakbaşı", "kebap", "ramen"],
    },
    "fastfood": {
        "queries": ["fast food", "burger", "hamburger"],
        "block_keywords": ["pastane", "tatlı", "baklava"],
    },
    "restaurant": {
        "queries": ["restoran", "yemek", "lokanta"],
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

def choose_category(top3):
    scores = {"cafe": 0.0, "brunch": 0.0, "fastfood": 0.0, "restaurant": 0.0}
    weights = [1.0, 0.7, 0.5]
    dessert_score = 0.0
    for i, (cls, prob) in enumerate(top3[:3]):
        cat = CLASS_TO_CATEGORY.get(cls, "restaurant")
        w = weights[i] if i < len(weights) else 0.5
        scores[cat] += float(prob) * w
        if cls in DESSERT_CLASSES:
            dessert_score += float(prob) * w
    if dessert_score >= 0.35:
        return "cafe"
    return max(scores, key=scores.get)

def build_queries(top3, category):
    rules = CATEGORY_RULES.get(category, CATEGORY_RULES["restaurant"])
    base = list(rules["queries"])
    dish_terms = []
    for dish, _ in top3[:2]:
        dish_terms.extend(CLASS_TO_QUERY_HINTS.get(dish, [dish.replace("_", " ")]))
    combined = []
    for t in dish_terms[:4]:
        combined.append(t)
    combined.extend(base)
    seen, out = set(), []
    for q in combined:
        ql = q.strip().lower()
        if ql and ql not in seen:
            seen.add(ql)
            out.append(q)
    return out[:6]

def places_search(category, lat, lng, top3, radius_m=5000, limit=10):
    if not API_KEY:
        return [], []
    rules   = CATEGORY_RULES.get(category, CATEGORY_RULES["restaurant"])
    queries = build_queries(top3, category)
    block   = set(rules.get("block_keywords", []))
    url     = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,places.location,"
            "places.rating,places.userRatingCount"
        ),
    }
    all_results = []
    for q in queries:
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
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                continue
            for p in r.json().get("places", []):
                name    = p.get("displayName", {}).get("text", "")
                address = p.get("formattedAddress", "")
                rating  = float(p.get("rating", 0.0) or 0.0)
                urc     = int(p.get("userRatingCount", 0) or 0)
                loc     = p.get("location", {})
                plat, plng = loc.get("latitude"), loc.get("longitude")
                dist = haversine_km(lat, lng, plat, plng) if plat else 9999.0
                txt = f"{name} {address}".lower()
                if any(bk in txt for bk in block):
                    continue
                all_results.append({
                    "name": name, "address": address,
                    "rating": rating, "user_ratings_total": urc,
                    "distance_km": dist, "lat": plat, "lng": plng,
                })
        except:
            continue
    seen, uniq = set(), []
    for r in all_results:
        key = (r["name"].strip().lower(), r["address"].strip().lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    uniq.sort(key=lambda x: (x["distance_km"], -x["rating"], -x["user_ratings_total"]))
    return uniq[:limit], queries

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    m = models.efficientnet_b2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_to_idx))
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    return m, idx_to_class

def predict_top3(model, idx_to_class, img):
    x = tfms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        top   = torch.topk(probs, 3)
    return [(idx_to_class[int(i)], float(p)) for p, i in zip(top.values, top.indices)]

def get_browser_location():
    js = """
    new Promise((resolve) => {
      if (!navigator.geolocation) {
        resolve({ok:false, error:"Tarayıcı desteklemiyor."});
      } else {
        navigator.geolocation.getCurrentPosition(
          (pos) => resolve({ok:true, latitude:pos.coords.latitude,
                            longitude:pos.coords.longitude, accuracy:pos.coords.accuracy}),
          (err) => resolve({ok:false, error:err.message}),
          {enableHighAccuracy:true, timeout:15000, maximumAge:0}
        );
      }
    })
    """
    return streamlit_js_eval(js_expressions=js, key="geo_eval")

# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "lat": 41.0082,
        "lng": 28.9784,
        "loc_status": "Henüz konum alınmadı.",
        "loc_pending": False,
        "cam_granted": False,
        "active_source": None,   # "file" | "camera" | "paste"
        "pasted_b64": None,
        "current_img_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================
# UI
# =========================
st.set_page_config(page_title="🍽️ Restaurant Finder CV", layout="centered")
init_state()

st.title("🍽️ Restaurant Finder CV")
st.caption("Yemek fotoğrafı yükle → ne olduğunu öğren → yakınındaki restoranları bul!")
st.caption("`EfficientNet-B2 | 54 Sınıf | Top-3: %95.73`")

# ── Konum ────────────────────────────────────────────────
st.subheader("📍 Konum")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("📍 Konumumu Al", use_container_width=True):
        st.session_state.loc_pending = True
        st.session_state.loc_status  = "Konum isteniyor..."
        st.rerun()
with c2:
    if st.button("🔄 Yenile", use_container_width=True):
        st.session_state.loc_pending = True
        st.session_state.loc_status  = "Yeniden isteniyor..."
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
        st.warning("Konum bekleniyor... İzin verdikten sonra 'Yenile'ye bas.")
    elif isinstance(loc, dict) and loc.get("ok"):
        st.session_state.lat         = float(loc["latitude"])
        st.session_state.lng         = float(loc["longitude"])
        st.session_state.loc_pending = False
        acc = loc.get("accuracy")
        st.session_state.loc_status  = (
            f"✅ Konum alındı ({st.session_state.lat:.5f}, {st.session_state.lng:.5f})"
            + (f" ±{acc:.0f}m" if acc else "")
        )
        st.rerun()
    else:
        st.session_state.loc_pending = False
        st.session_state.loc_status  = f"Konum alınamadı: {loc.get('error','Hata')}"
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    st.session_state.lat = float(st.number_input("Enlem", value=float(st.session_state.lat), format="%.6f"))
with col2:
    st.session_state.lng = float(st.number_input("Boylam", value=float(st.session_state.lng), format="%.6f"))

radius_km = st.slider("Arama yarıçapı (km)", 1, 20, 5)
limit      = st.slider("Kaç mekan?", 5, 20, 10)

st.divider()

# ── Görüntü Kaynağı ──────────────────────────────────────
st.subheader("🖼️ Görüntü Yükle")
tab1, tab2, tab3 = st.tabs(["📁 Dosya Yükle", "📸 Kameradan Çek", "📋 Yapıştır (Ctrl+V)"])

img = None  # Ana görüntü değişkeni

with tab1:
    uploaded = st.file_uploader(
        "Yemek fotoğrafı seç", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="file_uploader"
    )
    if uploaded:
        # Yeni dosya geldi — diğer kaynakları temizle
        st.session_state.pasted_b64    = None
        st.session_state.active_source = "file"
        img_bytes = uploaded.read()
        st.session_state.current_img_bytes = img_bytes
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

with tab2:
    if not st.session_state.cam_granted:
        st.info("📸 Kamerayı kullanmak için önce izin ver.")
        if st.button("📷 Kamera İzni Ver", use_container_width=True):
            cam_result = streamlit_js_eval(js_expressions="""
                new Promise((resolve) => {
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then((stream) => {
                            stream.getTracks().forEach(t => t.stop());
                            resolve({ok: true});
                        })
                        .catch((err) => resolve({ok: false, error: err.message}));
                })
            """, key="cam_permission")

            if cam_result and isinstance(cam_result, dict):
                if cam_result.get("ok"):
                    st.session_state.cam_granted = True
                    st.rerun()
                else:
                    st.error(f"İzin reddedildi: {cam_result.get('error','')}")
    else:
        camera_photo = st.camera_input("Kameradan çek", label_visibility="collapsed")
        if camera_photo:
            st.session_state.pasted_b64    = None
            st.session_state.active_source = "camera"
            img_bytes = camera_photo.read()
            st.session_state.current_img_bytes = img_bytes
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

with tab3:
    st.info("Resmi kopyala (Ctrl+C), aşağıdaki alana tıkla, Ctrl+V ile yapıştır.")

    if st.session_state.pasted_b64:
        try:
            b64_data = st.session_state.pasted_b64.split(",")[1]
            img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
            st.session_state.active_source = "paste"
            st.success("✅ Görüntü yapıştırıldı!")
        except:
            st.error("Görüntü okunamadı.")
            st.session_state.pasted_b64 = None

        if st.button("🗑️ Temizle", use_container_width=True):
            st.session_state.pasted_b64    = None
            st.session_state.active_source = None
            st.session_state.current_img_bytes = None
            st.rerun()
    else:
        # key parametresi olmadan çalıştır
        components.html("""
        <style>
            body { margin: 0; padding: 0; }
            #paste-area {
                width: 100%; height: 160px; box-sizing: border-box;
                border: 2px dashed #aaa; border-radius: 10px;
                display: flex; align-items: center; justify-content: center;
                font-size: 15px; color: #888; cursor: pointer;
                background: #f9f9f9;
            }
            #paste-area:focus { outline: none; border-color: #4CAF50; color: #4CAF50; }
        </style>
        <div id="paste-area" tabindex="0">
            🖼️ Buraya tıkla → Ctrl+V ile yapıştır
        </div>
        <script>
            const area = document.getElementById('paste-area');
            area.focus();

            area.addEventListener('paste', function(e) {
                const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.startsWith('image/')) {
                        const blob = items[i].getAsFile();
                        const reader = new FileReader();
                        reader.onload = function(ev) {
                            window.parent.postMessage({
                                isStreamlitMessage: true,
                                type: 'streamlit:setComponentValue',
                                value: ev.target.result
                            }, '*');
                            area.textContent = '✅ Yapıştırıldı! Sayfa güncellenecek...';
                            area.style.borderColor = '#4CAF50';
                            area.style.color = '#4CAF50';
                        };
                        reader.readAsDataURL(blob);
                        e.preventDefault();
                        break;
                    }
                }
            });
        </script>
        """, height=180)  # key parametresi kaldırıldı

        # Yapıştırılan değeri session state'ten oku
        if "paste_component" in st.session_state:
            paste_val = st.session_state["paste_component"]
            if paste_val and isinstance(paste_val, str) and paste_val.startswith("data:image"):
                st.session_state.pasted_b64 = paste_val
                del st.session_state["paste_component"]
                st.rerun()

# ── Aktif kaynak yoksa mevcut görüntüyü kullan ──────────
if img is None and st.session_state.current_img_bytes and st.session_state.active_source:
    try:
        img = Image.open(io.BytesIO(st.session_state.current_img_bytes)).convert("RGB")
    except:
        pass

# ── Tahmin ────────────────────────────────────────────────
if img is not None:
    # Temizle butonu
    col_img, col_btn = st.columns([4, 1])
    with col_img:
        st.image(img, use_container_width=True)
    with col_btn:
        if st.button("🗑️", help="Görüntüyü temizle"):
            st.session_state.active_source     = None
            st.session_state.current_img_bytes = None
            st.session_state.pasted_b64        = None
            st.rerun()

    model, idx_to_class = load_model()
    top3 = predict_top3(model, idx_to_class, img)

    st.subheader("🎯 Tahmin (Top-3)")
    for i, (cls, prob) in enumerate(top3):
        tr    = CLASS_TR.get(cls, cls)
        emoji = ["🥇", "🥈", "🥉"][i]
        st.progress(prob, text=f"{emoji} **{tr}** — %{prob*100:.1f}")

    category = choose_category(top3)
    emoji_map = {
        "cafe": "☕ Kafe / Pastane",
        "brunch": "🥞 Kahvaltı / Brunch",
        "fastfood": "🍔 Fast Food",
        "restaurant": "🍽️ Restoran"
    }
    st.info(f"**Mekan Tipi:** {emoji_map.get(category, category)}")

    st.subheader("📍 Yakındaki Öneriler")
    with st.spinner("Restoranlar aranıyor..."):
        results, _ = places_search(
            category=category,
            lat=st.session_state.lat,
            lng=st.session_state.lng,
            top3=top3,
            radius_m=radius_km * 1000,
            limit=limit,
        )

    if not results:
        if not API_KEY:
            st.warning("⚠️ `GOOGLE_MAPS_API_KEY` env değişkeni eksik!")
        else:
            st.warning("Yakında sonuç bulunamadı. Yarıçapı artırabilirsin.")
    else:
        for i, r in enumerate(results, 1):
            with st.container():
                st.markdown(f"### {i}. {r['name']}")
                st.markdown(f"""
- ⭐ **{r['rating']:.1f}** ({r['user_ratings_total']} yorum)
- 📍 **{r['distance_km']:.2f} km**
- 🏠 {r['address']}
""")
                if r.get("lat"):
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={r['lat']},{r['lng']}"
                    st.markdown(f"[📌 Google Maps'te Aç]({maps_url})")
                st.markdown("---")
else:
    st.caption("Başlamak için bir fotoğraf yükle, kamerandan çek veya yapıştır.")