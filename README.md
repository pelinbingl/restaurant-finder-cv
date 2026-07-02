# 🍽️ Restaurant Finder CV

Yemek fotoğrafından **ne yemek olduğunu tanıyan** ve **yakınındaki restoranları öneren** yapay zeka destekli Streamlit uygulaması.

## 🎯 Özellikler

- 📸 **3 farklı görüntü kaynağı:** Dosya yükle, kameradan çek, Ctrl+V ile yapıştır
- 🤖 **54 sınıf yemek tanıma** (29 uluslararası + 25 Türk yemeği)
- 📍 **Otomatik konum tespiti** (tarayıcı izni ile)
- 🗺️ **Google Places API** ile yakın restoran önerisi
- 📌 **Google Maps linki** ile navigasyon

## 📊 Model Performansı

| Metrik | Değer |
|--------|-------|
| Model | EfficientNet-B2 |
| Sınıf Sayısı | 54 |
| Test Top-1 Accuracy | %85.5 |
| Test Top-3 Accuracy | **%95.73** |
| Dataset | Food-101 (29 sınıf) + TurkishFoods-25 |

## 🍴 Desteklenen Yemekler

**Türk Yemekleri (25):**
Aşure, Baklava, Biber Dolması, Börek, Çiğ Köfte, Enginar, Et Sote, Gözleme, Hamsi, Hünkar Beğendi, İçli Köfte, Ispanak, İzmir Köfte, Karnıyarık, Kebap, Kısır, Kuru Fasulye, Lahmacun, Lokum, Mantı, Mücver, Pilav, Simit, Taze Fasulye, Yaprak Sarma

**Uluslararası (29):**
Bahar Rulo, Çikolatalı Kek, Club Sandwich, Donut, Dumpling, Eggs Benedict, Fransız Tostu, Gnocchi, Hamburger, Hot Dog, Dondurma, Lazanya, Miso Çorbası, Nachos, Omlet, Pad Thai, Pankek, Patates Kızartması, Pizza, Ramen, Ravioli, Risotto, Sarımsaklı Ekmek, Spagetti Bolonez, Spagetti Carbonara, Sushi, Tiramisu, Waffle, Izgara Peynirli Sandviç

## 🏗️ Teknik Detaylar

- **Model:** EfficientNet-B2 (ImageNet pretrained)
- **Eğitim:** 2 aşamalı transfer learning
  - Aşama 1: Frozen backbone (5 epoch)
  - Aşama 2: Discriminative LR fine-tune + Mixup (15 epoch)
- **Teknikler:** WeightedRandomSampler, Label Smoothing, OneCycleLR, Gradient Clipping, Mixup Augmentation
- **Framework:** PyTorch + Streamlit

## ⚙️ Kurulum

```bash
git clone https://github.com/pelinbingl/restaurant-finder-cv
cd restaurant-finder-cv
pip install -r requirements.txt
```

### Çalıştırma

```bash
streamlit run app.py
```

### Google Maps API Key

```bash
# Windows
setx GOOGLE_MAPS_API_KEY "YOUR_KEY_HERE"

# Linux/Mac
export GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"
```

## 📁 Proje Yapısı

```
restaurant-finder-cv/
├── app.py                  # Streamlit uygulaması
├── food_model_b2.pth       # Eğitilmiş model
├── requirements.txt
└── README.md
```

## 🚀 Nasıl Kullanılır?

1. **Konum al** — "Konumumu Al" butonuna bas, tarayıcı izni ver
2. **Fotoğraf yükle** — Dosya seç, kameradan çek veya Ctrl+V ile yapıştır
3. **Sonuçları gör** — Model yemeği tanır, yakın restoranları listeler
4. **Navigasyon** — "Google Maps'te Aç" linkine tıkla

## 👩‍💻 Geliştirici

**Pelin Bingöl**
[LinkedIn](https://linkedin.com/in/pelin-bingöl) • [GitHub](https://github.com/pelinbingl)
