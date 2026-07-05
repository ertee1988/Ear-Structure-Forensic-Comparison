# 🔍 Forensic Ear Comparison Toolkit

> **Adli Görüntü Karşılaştırma İncelemelerinde Kulak Analizi**  
> Profesyonel bir Python toolkit'i — yüz tanıma sistemlerinin yetersiz kaldığı durumlarda kulak morfolojisini kullanarak kimlik karşılaştırması yapar.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)](https://opencv.org)

---

## 📌 Neden Kulak?

Adli görüntü inceleme alanında kulak yapısı, kimliklendirme çalışmalarında önemli ayırt edici özelliklerden biridir:

- **Uniqueness**: Kulak kepçesinin şekli, kıvrımları ve kulak memesinin yapısı kişiden kişiye farklılık gösterir
- **Stability**: Yaş, kilo değişiklikleri ve saç stili farklılıkları kulak yapısını minimum düzeyde etkiler
- **Low-Resolution Robustness**: Düşük çözünürlüklü kamera kayıtlarında yüz detayları yetersizken kulak yapısı karşılaştırma kriteri olabilir
- **Non-Invasive**: Görüntü tabanlı analiz, fiziksel temas gerektirmez

---

## 🚀 Özellikler

| Modül | Açıklama |
|-------|----------|
| `ear_extractor` | Görüntüden kulak bölgesini tespit etme ve kırpma |
| `ear_matcher` | İki kulak görüntüsü arasında benzerlik skoru hesaplama |
| `feature_analyzer` | Kulak morfolojik özelliklerini (helix, antihelix, tragus vb.) analiz etme |
| `embedding_model` | Derin öğrenme tabanlı kulak embedding vektörü üretimi |
| `generate_report` | Karşılaştırma sonuçlarını adli rapor formatında çıktı alma |

---

## 📦 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/ertee1988/forensic-ear-comparison.git
cd forensic-ear-comparison

# Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Model dosyalarını indir (isteğe bağlı)
python scripts/download_models.py
```

---

## 🎯 Hızlı Başlangıç

### 1. Tek Görüntüden Kulak Çıkarma

```python
from src.ear_extractor import EarExtractor

extractor = EarExtractor()
ear_image = extractor.extract("suspect_photo.jpg")
ear_image.save("extracted_ear.png")
```

### 2. İki Kulak Görüntüsünü Karşılaştırma

```python
from src.ear_matcher import EarMatcher

matcher = EarMatcher(method="hybrid")  # 'geometric', 'texture', 'hybrid'
score, report = matcher.compare("ear1.png", "ear2.png")

print(f"Benzerlik Skoru: {score:.3f}")
print(f"Karar: {'EŞLEŞME' if score > 0.75 else 'EŞLEŞME YOK'}")
```

### 3. Toplu İşlem ve Rapor Üretme

```bash
python scripts/batch_process.py     --reference ./data/reference/     --query ./data/query/     --output ./reports/
```

---

## 📊 Metodoloji

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Girdi Görüntü  │────▶│  Kulak Tespiti  │────▶│  Ön İşleme    │
│  (Video/Kare)   │     │  (Haar/YOLO/Mediapipe)│  │  (Normalizasyon)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                              ┌──────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ Feature Extract │
                    │  - Geometric    │
                    │  - Texture      │
                    │  - Deep Embed   │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Similarity Score│
                    │  + Threshold    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Rapor Üretimi  │
                    │  (PDF/Markdown) │
                    └─────────────────┘
```

---

## 🧬 Kulak Anatomisi Referansı

Proje, aşağıdaki anatomik yapıları analiz eder:

| Yapı | Türkçe Adı | Analiz Yöntemi |
|------|-----------|----------------|
| Helix | Kulak Kıvrımı (Dış) | Kontur analizi |
| Antihelix | Kulak Kıvrımı (İç) | Şekil eşleştirme |
| Tragus | Tragus | Boyut/oran analizi |
| Antitragus | Antitragus | Konum analizi |
| Lobule | Kulak Memesi | Dokusu ve şekli |
| Concha | Kulak Çanağı | Derinlik analizi |

Detaylı bilgi için: [docs/ear_anatomy_guide.md](docs/ear_anatomy_guide.md)

---

## 📁 Proje Yapısı

```
forensic-ear-comparison/
├── src/                    # Ana kaynak kodları
│   ├── ear_extractor.py    # Kulak bölgesi tespiti
│   ├── ear_matcher.py      # Karşılaştırma motoru
│   ├── feature_analyzer.py # Morfolojik analiz
│   └── models/             # ML modelleri
├── notebooks/              # Jupyter demo notebook'ları
├── scripts/                # CLI araçları
├── tests/                  # Birim testleri
├── docs/                   # Dokümantasyon
└── config/                 # Yapılandırma dosyaları
```

---

## ⚖️ Yasal ve Etik Uyarılar

> **ÖNEMLİ**: Bu araç yalnızca yetkili adli merciler ve lisanslı bilirkişiler tarafından kullanılmalıdır.  
> Araç tek başına kesin kimlik tespiti için yeterli değildir; uzman değerlendirmesi zorunludur.  
> Detaylar için: [docs/legal_considerations.md](docs/legal_considerations.md)

---

## 👤 Yazar

**Serkan Dinçer**  
Adli Ses ve Görüntü Karşılaştırma Uzmanı

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.
