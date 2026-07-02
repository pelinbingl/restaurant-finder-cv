# 🍽️ Restaurant Finder CV

🚀 Yemek fotoğrafından **ne yemek olduğunu tanıyan** ve **yakınındaki restoranları öneren**, Google Maps navigasyonu içeren yapay zeka destekli Streamlit uygulaması.

---

## 🌐 Live Demo

👉 [restaurant-finder-cv.streamlit.app](https://restaurant-finder-cv-ned58gn7ggbutd6ehswfvg.streamlit.app/)

[![Open App](https://img.shields.io/badge/Open-App-green?style=for-the-badge)](https://restaurant-finder-cv-ned58gn7ggbutd6ehswfvg.streamlit.app/)

---

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-orange)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🎯 Özellikler

- 📸 **3 farklı görüntü kaynağı:** Dosya yükle, kameradan çek, Ctrl+V ile yapıştır
- 🤖 **54 sınıf yemek tanıma** (29 uluslararası + 25 Türk yemeği), Top-3 tahmin
- 📍 **Otomatik konum tespiti** (tarayıcı izni ile)
- 🗺️ **Google Places API** ile yakın restoran önerisi
- 📌 **Google Maps linki** ile navigasyon
- ⚡ Hızlı, etkileşimli arayüz (Streamlit)

---

## 📊 Model Performansı

| Metrik | Değer |
|--------|-------|
| Model | EfficientNet-B2 |
| Sınıf Sayısı | 54 |
| Test Top-1 Accuracy | %85.5 |
| Test Top-3 Accuracy | **%95.73** |
| Dataset | Food-101 (29 sınıf) + TurkishFoods-25 |

---

## 🍴 Desteklenen Yemekler

**Türk Yemekleri (25):**
Aşure, Baklava, Biber Dolması, Börek, Çiğ Köfte, Enginar, Et Sote, Gözleme, Hamsi, Hünkar Beğendi, İçli Köfte, Ispanak, İzmir Köfte, Karnıyarık, Kebap, Kısır, Kuru Fasulye, Lahmacun, Lokum, Mantı, Mücver, Pilav, Simit, Taze Fasulye, Yaprak Sarma

**Uluslararası (29):**
Bahar Rulo, Çikolatalı Kek, Club Sandwich, Donut, Dumpling, Eggs Benedict, Fransız Tostu, Gnocchi, Hamburger, Hot Dog, Dondurma, Lazanya, Miso Çorbası, Nachos, Omlet, Pad Thai, Pankek, Patates Kızartması, Pizza, Ramen, Ravioli, Risotto, Sarımsaklı Ekmek, Spagetti Bolonez, Spagetti Carbonara, Sushi, Tiramisu, Waffle, Izgara Peynirli Sandviç

---

## 🏗️ Teknik Detaylar

- **Model:** EfficientNet-B2 (ImageNet pretrained)
- **Eğitim:** 2 aşamalı transfer learning
  - Aşama 1: Frozen backbone (5 epoch)
  - Aşama 2: Discriminative LR fine-tune + Mixup (15 epoch)
- **Teknikler:** WeightedRandomSampler, Label Smoothing, OneCycleLR, Gradient Clipping, Mixup Augmentation
- **Frontend:** Streamlit
- **Backend:** Python
- **AI Model:** PyTorch
- **API:** Google Places API

---

## 📁 Proje Yapısı

```
restaurant-finder-cv/
├── app.py                     # Streamlit uygulaması
├── food_model_b2.pth          # Eğitilmiş model
├── Train.ipynb                # Model eğitim notebook'u
├── requirements.txt
├── .streamlit/
│   └── secrets.toml           # API anahtarları (git'e eklenmez)
└── README.md
```

---

## ⚙️ Kurulum

```bash
git clone https://github.com/pelinbingl/restaurant-finder-cv.git
cd restaurant-finder-cv
pip install -r requirements.txt
```

### Çalıştırma

```bash
streamlit run app.py
```

### 🔑 Google Maps API Key

**Yerel geliştirme için:** `.streamlit/secrets.toml` dosyası oluşturup içine ekleyin:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"
```

**Alternatif (ortam değişkeni ile):**

```bash
# Windows
setx GOOGLE_MAPS_API_KEY "YOUR_KEY_HERE"

# Linux/Mac
export GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"
```

> ⚠️ `secrets.toml` dosyasını `.gitignore`'a eklemeyi unutmayın, API anahtarınızı repoya push etmeyin.

---

## 🚀 Nasıl Kullanılır?

1. **Konum al** — "Konumumu Al" butonuna bas, tarayıcı izni ver
2. **Fotoğraf yükle** — Dosya seç, kameradan çek veya Ctrl+V ile yapıştır
3. **Sonuçları gör** — Model yemeği tanır, yakın restoranları listeler
4. **Navigasyon** — "Google Maps'te Aç" linkine tıkla

---

## 🌍 Deployment

Streamlit Cloud üzerinde deploy edildi 👉 [streamlit.io/cloud](https://streamlit.io/cloud)

---

## 📌 Gelecek Geliştirmeler

- 🗺️ İnteraktif harita üzerinde pin gösterimi
- ❤️ Favori restoranlar
- 📱 Mobil optimizasyon
- 🌍 Çoklu dil desteği

---

## 👩‍💻 Geliştirici

**Pelin Bingöl**
[LinkedIn](https://linkedin.com/in/pelin-bingöl) • [GitHub](https://github.com/pelinbingl)

---

## ⭐ Destek

Bu projeyi beğendiyseniz GitHub'da ⭐ bırakmayı unutmayın!

<br>

---
---

<br>

# 🍽️ Restaurant Finder CV (English)

🚀 An AI-powered Streamlit app that **recognizes food from a photo** and **recommends nearby restaurants**, with Google Maps navigation.

---

## 🌐 Live Demo

👉 [restaurant-finder-cv.streamlit.app](https://restaurant-finder-cv-ned58gn7ggbutd6ehswfvg.streamlit.app/)

[![Open App](https://img.shields.io/badge/Open-App-green?style=for-the-badge)](https://restaurant-finder-cv-ned58gn7ggbutd6ehswfvg.streamlit.app/)

---

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Model-orange)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🎯 Features

- 📸 **3 image input methods:** upload a file, capture from camera, or paste with Ctrl+V
- 🤖 **54-class food recognition** (29 international + 25 Turkish dishes), Top-3 prediction
- 📍 **Automatic location detection** (with browser permission)
- 🗺️ **Nearby restaurant recommendations** via Google Places API
- 📌 **Google Maps link** for turn-by-turn navigation
- ⚡ Fast, interactive UI (Streamlit)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | EfficientNet-B2 |
| Classes | 54 |
| Test Top-1 Accuracy | 85.5% |
| Test Top-3 Accuracy | **95.73%** |
| Dataset | Food-101 (29 classes) + TurkishFoods-25 |

---

## 🍴 Supported Dishes

**Turkish Dishes (25):**
Aşure, Baklava, Stuffed Peppers, Börek, Çiğ Köfte, Artichoke, Et Sote, Gözleme, Anchovy, Hünkar Beğendi, İçli Köfte, Spinach, İzmir Köfte, Karnıyarık, Kebap, Kısır, Kuru Fasulye, Lahmacun, Turkish Delight, Mantı, Mücver, Rice Pilaf, Simit, Green Beans, Yaprak Sarma

**International (29):**
Spring Rolls, Chocolate Cake, Club Sandwich, Donut, Dumpling, Eggs Benedict, French Toast, Gnocchi, Hamburger, Hot Dog, Ice Cream, Lasagna, Miso Soup, Nachos, Omelette, Pad Thai, Pancake, French Fries, Pizza, Ramen, Ravioli, Risotto, Garlic Bread, Spaghetti Bolognese, Spaghetti Carbonara, Sushi, Tiramisu, Waffle, Grilled Cheese Sandwich

---

## 🏗️ Technical Details

- **Model:** EfficientNet-B2 (ImageNet pretrained)
- **Training:** 2-stage transfer learning
  - Stage 1: Frozen backbone (5 epochs)
  - Stage 2: Discriminative LR fine-tuning + Mixup (15 epochs)
- **Techniques:** WeightedRandomSampler, Label Smoothing, OneCycleLR, Gradient Clipping, Mixup Augmentation
- **Frontend:** Streamlit
- **Backend:** Python
- **AI Model:** PyTorch
- **API:** Google Places API

---

## 📁 Project Structure

```
restaurant-finder-cv/
├── app.py                     # Streamlit application
├── food_model_b2.pth          # Trained model
├── Train.ipynb                # Model training notebook
├── requirements.txt
├── .streamlit/
│   └── secrets.toml           # API keys (not committed)
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/pelinbingl/restaurant-finder-cv.git
cd restaurant-finder-cv
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

### 🔑 Google Maps API Key

**For local development:** create `.streamlit/secrets.toml` and add:

```toml
GOOGLE_MAPS_API_KEY = "YOUR_API_KEY"
```

**Alternative (environment variable):**

```bash
# Windows
setx GOOGLE_MAPS_API_KEY "YOUR_KEY_HERE"

# Linux/Mac
export GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"
```

> ⚠️ Add `secrets.toml` to `.gitignore` — never push your API key to the repo.

---

## 🚀 How to Use

1. **Get location** — click "Get My Location" and grant browser permission
2. **Upload a photo** — choose a file, capture from camera, or paste with Ctrl+V
3. **View results** — the model identifies the dish and lists nearby restaurants
4. **Navigate** — click "Open in Google Maps"

---

## 🌍 Deployment

Deployed on Streamlit Cloud 👉 [streamlit.io/cloud](https://streamlit.io/cloud)

---

## 📌 Future Improvements

- 🗺️ Interactive map with pins
- ❤️ Favorite restaurants
- 📱 Mobile optimization
- 🌍 Multi-language support

---

## 👩‍💻 Developer

**Pelin Bingöl**
[LinkedIn](https://linkedin.com/in/pelin-bingöl) • [GitHub](https://github.com/pelinbingl)

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
