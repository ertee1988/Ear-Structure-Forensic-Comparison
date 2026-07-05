# 🔍 Forensic Ear Comparison Toolkit

> **Adli Görüntü Karşılaştırma İncelemelerinde Kulak Analizi**  
> Kulak morfolojisini kullanan, araştırma ve adli ön değerlendirme için tasarlanmış Python tabanlı bir toolkit.

<p align="center">
  <img src="https://raw.githubusercontent.com/ertee1988/Ear-Structure-Forensic-Comparison/main/docs/assets/hero-placeholder.png" alt="Forensic Ear" width="900" style="max-width:100%; border-radius:12px; box-shadow: 0 8px 24px rgba(0,0,0,0.12);"/>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://opencv.org"><img src="https://img.shields.io/badge/OpenCV-4.8%2B-orange" alt="OpenCV"></a>
</p>

---

## İçindekiler
- [Neden Kulak?](#neden-kulak)
- [Özellikler](#özellikler)
- [Kurulum & Hızlı Başlangıç](#kurulum--hızlı-başlangıç)
- [Metodoloji Özet](#metodoloji-özet)
- [Proje Yapısı](#proje-yapısı)
- [Yasal ve Etik Uyarılar](#yasal-ve-etik-uyarılar)
- [Yazar](#yazar)
- [İletişim](#iletişim)

---

## 📌 Neden Kulak?

Kulak kepçesi (auricle) adli kimliklendirmede faydalı bir tamamlayıcıdır:

- **Ayırt Edicilik** — Helix, antihelix, lobulus gibi yapılar kişiye özgüdür.
- **Kararlılık** — Yüzdeki ifadeler veya saç değişikliklerine göre daha stabildir.
- **Düşük Çözünürlükte Çalışma** — CCTV gibi düşük kaliteli görüntülerde bile işe yarayan morfolojik izler bulunabilir.
- **Temassız ve Hızlı** — Görüntü tabanlı, non-invaziv bir ön değerlendirme sağlar.

---

## 🚀 Özellikler

- Kulak tespiti (Haar / YOLO / MediaPipe) ve kırpma
- Geometrik ve dokusal özellik çıkarımı
- Derin öğrenme tabanlı embedding üretimi
- İki görüntü arası benzerlik skoru ve eşiklendirme
- Toplu işlem, rapor üretimi (PDF/Markdown)

---

## 📦 Kurulum & Hızlı Başlangıç

Aşağıdaki adımlar GitHub üzerinde kolayca takip edilebilir; arzu ederseniz kod bloklarını kaldırıp komutları düz metin haline getirebilirim.

### Kurulum

```bash
# Repoyu klonlayın
git clone https://github.com/ertee1988/Ear-Structure-Forensic-Comparison.git
cd Ear-Structure-Forensic-Comparison

# Sanal ortam (önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Model dosyalarını indirin (opsiyonel)
python scripts/download_models.py
```

### Örnek: Kulak Çıkarma

```python
from src.ear_extractor import EarExtractor
extractor = EarExtractor()
ear = extractor.extract("suspect_photo.jpg")
ear.save("extracted_ear.png")
```

### Örnek: Karşılaştırma

```python
from src.ear_matcher import EarMatcher
matcher = EarMatcher(method="hybrid")
score, report = matcher.compare("ear1.png", "ear2.png")
print(f"Benzerlik Skoru: {score:.3f}")
```

---

## 📊 Metodoloji - Kısa Özet

Girdi → Kulak Tespiti → Ön İşleme (normalize, align) → Özellik Çıkarımı (geometric, texture, deep embeddings) → Skorlama → Rapor

Detaylı pipeline dokümanı için: `docs/` dizinindeki ilgili rehberlere bakınız.

---

## 📁 Proje Yapısı

```
Ear-Structure-Forensic-Comparison/
├── src/                    # Kaynak kodları (ear_extractor, ear_matcher, ...)
├── scripts/                # CLI araçları (download_models, batch_process)
├── docs/                   # Dokümantasyon ve rehberler
├── data/                   # Örnek veri / referans dizini
├── tests/                  # Birim testleri
└── requirements.txt
```

---

## ⚖️ Yasal ve Etik Uyarılar

**ÖNEMLİ:** Bu proje yalnızca yetkili kurumlar ve lisanslı bilirkişiler tarafından kullanılmalıdır. Bu araç tek başına kesin kimlik tespiti için yeterli değildir; adli değerlendirme uzmanı gereklidir.

---

## 👤 Yazar

**Serkan Dinçer** — Adli Ses ve Görüntü Karşılaştırma Uzmanı

---

## ✉️ İletişim

Aşağıdaki iletişim bilgilerini dosyanın en altına ekledim, projenin diğer kısımlarına veya profil sayfasına da yerleştirmek isterseniz söyleyin.

- Profiliniz: https://linkedin.com/in/engin-d-0421b748
- Web sitesi / Blog: https://serkand.carrd.co
- E-posta: ekinxc14@gmail.com

---

> Not: README'in görsel olarak daha da zenginleşmesi için `docs/assets/` altına örnek görseller, örnek rapor PDF'leri veya GIF'ler ekleyebilirim. Onay verirseniz ek görselleri repo içine yükleyip README'de bunları sergileyebilirim.
