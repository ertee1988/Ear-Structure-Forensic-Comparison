# 🔍 Forensic Ear Comparison Toolkit

> \*\*Adli Görüntü Karşılaştırma İncelemelerinde Kulak Analizi\*\*  
> Profesyonel bir Python toolkit'i — yüz tanıma sistemlerinin yetersiz kaldığı durumlarda kulak morfolojisini kullanarak kimlik karşılaştırması yapar.

[!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[!\[License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[!\[OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)](https://opencv.org)

\---

## 📌 Neden Kulak?

Adli görüntü inceleme alanında kulak yapısı, kimliklendirme çalışmalarında önemli ayırt edici özelliklerden biridir:

* **Uniqueness**: Kulak kepçesinin şekli, kıvrımları ve kulak memesinin yapısı kişiden kişiye farklılık gösterir
* **Stability**: Yaş, kilo değişiklikleri ve saç stili farklılıkları kulak yapısını minimum düzeyde etkiler
* **Low-Resolution Robustness**: Düşük çözünürlüklü kamera kayıtlarında yüz detayları yetersizken kulak yapısı karşılaştırma kriteri olabilir
* **Non-Invasive**: Görüntü tabanlı analiz, fiziksel temas gerektirmez

\---

## 🚀 Özellikler

|Modül|Açıklama|
|-|-|
|`ear\_extractor`|Görüntüden kulak bölgesini tespit etme ve kırpma|
|`ear\_matcher`|İki kulak görüntüsü arasında benzerlik skoru hesaplama|
|`feature\_analyzer`|Kulak morfolojik özelliklerini (helix, antihelix, tragus vb.) analiz etme|
|`embedding\_model`|Derin öğrenme tabanlı kulak embedding vektörü üretimi|
|`generate\_report`|Karşılaştırma sonuçlarını adli rapor formatında çıktı alma|

\---

## 📦 Kurulum

```bash
# Repoyu klonla
git clone https://github.com/ertee1988/Ear-Structure-Forensic-Comparison.git

cd Ear-Structure-Forensic-Comparison


# Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Model dosyalarını indir (isteğe bağlı)
python scripts/download\_models.py
```

\---

## 🎯 Hızlı Başlangıç

### 1\. Tek Görüntüden Kulak Çıkarma

```python
from src.ear\_extractor import EarExtractor

extractor = EarExtractor()
ear\_image = extractor.extract("suspect\_photo.jpg")
ear\_image.save("extracted\_ear.png")
```

### 2\. İki Kulak Görüntüsünü Karşılaştırma

```python
from src.ear\_matcher import EarMatcher

matcher = EarMatcher(method="hybrid")  # 'geometric', 'texture', 'hybrid'
score, report = matcher.compare("ear1.png", "ear2.png")

print(f"Benzerlik Skoru: {score:.3f}")
print(f"Karar: {'EŞLEŞME' if score > 0.75 else 'EŞLEŞME YOK'}")
```

### 3\. Toplu İşlem ve Rapor Üretme

python scripts/batch\_process.py \\

&#x20;   --reference ./data/reference/ \\

&#x20;   --query ./data/query/ \\

&#x20;   --output ./reports/```

\---

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

\---

## 🧬 Kulak Anatomisi Referansı

Proje, aşağıdaki anatomik yapıları analiz eder:

|Yapı|Türkçe Adı|Analiz Yöntemi|
|-|-|-|
|Helix|Kulak Kıvrımı (Dış)|Kontur analizi|
|Antihelix|Kulak Kıvrımı (İç)|Şekil eşleştirme|
|Tragus|Tragus|Boyut/oran analizi|
|Antitragus|Antitragus|Konum analizi|
|Lobule|Kulak Memesi|Dokusu ve şekli|
|Concha|Kulak Çanağı|Derinlik analizi|

Detaylı bilgi için: [docs/ear\_anatomy\_guide.md](docs/ear_anatomy_guide.md)

\---

## 📁 Proje Yapısı

```
forensic-ear-comparison/
├── src/                    # Ana kaynak kodları
│   ├── ear\_extractor.py    # Kulak bölgesi tespiti
│   ├── ear\_matcher.py      # Karşılaştırma motoru
│   ├── feature\_analyzer.py # Morfolojik analiz
│   └── models/             # ML modelleri
├── notebooks/              # Jupyter demo notebook'ları
├── scripts/                # CLI araçları
├── tests/                  # Birim testleri
├── docs/                   # Dokümantasyon
└── config/                 # Yapılandırma dosyaları
```

\---

## ⚖️ Yasal ve Etik Uyarılar

> \*\*ÖNEMLİ\*\*: Bu araç yalnızca yetkili adli merciler ve lisanslı bilirkişiler tarafından kullanılmalıdır.  
> Araç tek başına kesin kimlik tespiti için yeterli değildir; uzman değerlendirmesi zorunludur.  
> Detaylar için: \[docs/legal\_considerations.md](docs/legal\_considerations.md)

\---

## 👤 Yazar

**Serkan Dinçer**  
Adli Ses ve Görüntü Karşılaştırma Uzmanı

\---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

