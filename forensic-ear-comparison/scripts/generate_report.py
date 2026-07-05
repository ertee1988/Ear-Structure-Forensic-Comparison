#!/usr/bin/env python3
"""
Adli kulak karşılaştırma raporu üretici.
Karşılaştırma sonuçlarını uzman raporu formatında PDF/Markdown olarak çıktılar.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_markdown_report(data: Dict, output_path: str):
    """Markdown formatında rapor üret."""

    md = f"""# ADLİ KULAK KARŞILAŞTIRMA RAPORU

---

## Rapor Bilgileri

| Alan | Değer |
|------|-------|
| Rapor Tarihi | {datetime.now().strftime("%d.%m.%Y %H:%M")} |
| Rapor No | AKK-{datetime.now().strftime("%Y%m%d-%H%M%S")} |
| Karşılaştırma Yöntemi | {data['metadata']['method'].upper()} |
| Eşik Değeri | {data['metadata']['threshold'].upper()} |
| Toplu Karşılaştırma | {data['metadata']['total_comparisons']} |

---

## 1. Metodoloji

Bu rapor, adli görüntü karşılaştırma incelemelerinde kulak morfolojisinin ayırt edici özelliklerini kullanarak kimlik karşılaştırması yapmaktadır.

### 1.1 Kullanılan Yöntemler

"""

    method = data['metadata']['method']
    if method == 'hybrid':
        md += """
- **Geometrik Analiz**: Helix, antihelix, tragus, lobule gibi anatomik yapıların morfolojik özellikleri
- **Dokusal Analiz**: SSIM, histogram karşılaştırması, ORB özellik eşleştirme
- **Derin Öğrenme**: ResNet50 tabanlı embedding karşılaştırması
"""
    elif method == 'geometric':
        md += "- **Geometrik Analiz**: Morfolojik özellik tabanlı karşılaştırma"
    elif method == 'texture':
        md += "- **Dokusal Analiz**: Görüntü yapısı ve doku benzerliği"
    else:
        md += "- **Derin Öğrenme**: CNN tabanlı embedding karşılaştırması"

    md += f"""

### 1.2 Değerlendirme Kriterleri

| Karar | Skor Aralığı | Açıklama |
|-------|-------------|----------|
| MATCH | ≥ 0.85 | Yüksek olasılıkla aynı kişi |
| PROBABLE_MATCH | 0.75 - 0.85 | Muhtemelen aynı kişi |
| INCONCLUSIVE | 0.60 - 0.75 | Belirsiz, ek inceleme gerekli |
| NO_MATCH | < 0.60 | Farklı kişi |

---

## 2. Karşılaştırma Sonuçları

### 2.1 En Yüksek Skorlu Sonuçlar

| Sıra | Referans | Sorgu | Skor | Karar | Güven |
|------|----------|-------|------|-------|-------|
"""

    for i, res in enumerate(data['results'][:10], 1):
        ref = Path(res['reference']).name
        query = Path(res['query']).name
        score = res['score']
        decision = res['result']['decision']
        confidence = res['result']['confidence']

        md += f"| {i} | {ref} | {query} | {score:.3f} | {decision} | {confidence:.2f} |
"

    md += """

---

## 3. Detaylı Analiz

"""

    for i, res in enumerate(data['results'][:5], 1):
        md += f"""### 3.{i} Karşılaştırma: {Path(res['reference']).name} vs {Path(res['query']).name}

- **Benzerlik Skoru**: {res['score']:.4f}
- **Karar**: {res['result']['decision']}
- **Güven Seviyesi**: {res['result']['confidence']:.2%}
- **Kullanılan Eşik**: {res['result']['threshold_used']}

**Özellik Skorları:**
"""
        for feat, val in res['result']['feature_scores'].items():
            if isinstance(val, float):
                md += f"- {feat}: {val:.4f}
"
            elif isinstance(val, dict):
                md += f"- {feat}:
"
                for k, v in val.items():
                    md += f"  - {k}: {v:.4f}
"

        md += "
---
"

    md += """
## 4. Hukuki ve Etik Uyarı

> **ÖNEMLİ**: Bu rapor otomatik hesaplama araçları ile üretilmiştir ve tek başına kesin kimlik tespiti için yeterli değildir. Sonuçlar, yetkili adli görüntü inceleme uzmanı tarafından değerlendirilmelidir.
>
> - Kulak yapısı, kimlik tespitinde destekleyici bir kriterdir.
> - Düşük çözünürlüklü görüntülerde sonuçlar sınırlı güvenilirlik taşıyabilir.
> - Aydınlatma, açı ve örtülme durumları sonuçları etkileyebilir.

---

## 5. Kaynakça

1. Wang, X. et al. (2026). "Ear biometrics in forensic identification: from ear similarity quantification to kinship verification driven by deep learning approaches." *Int J Legal Med*, 140(1), 477-488.
2. Arbab-Zavar, B. et al. "On Forensic Use of Biometrics, with Face and Ear Recognition." University of Southampton.
3. Krishan, K. et al. (2019). "A study of morphological variations of the human ear for its applications in personal identification." *Egyptian Journal of Forensic Sciences*, 9(1).

---

*Bu rapor Forensic Ear Comparison Toolkit v1.0.0 ile üretilmiştir.*
*Hazırlayan: Serkan Dinçer - Adli Ses ve Görüntü Karşılaştırma Uzmanı*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Markdown raporu kaydedildi: {output_path}")


def generate_text_report(data: Dict, output_path: str):
    """Düz metin formatında rapor üret."""

    text = f"""═══════════════════════════════════════════════════════════════
         ADLİ KULAK KARŞILAŞTIRMA RAPORU
═══════════════════════════════════════════════════════════════

Rapor Tarihi    : {datetime.now().strftime("%d.%m.%Y %H:%M")}
Rapor No        : AKK-{datetime.now().strftime("%Y%m%d-%H%M%S")}
Yöntem          : {data['metadata']['method'].upper()}
Eşik Seviyesi   : {data['metadata']['threshold'].upper()}
Toplam Karşılaştırma: {data['metadata']['total_comparisons']}

───────────────────────────────────────────────────────────────
                    KARŞILAŞTIRMA SONUÇLARI
───────────────────────────────────────────────────────────────

Sıra  Skor    Karar              Referans                Sorgu
───────────────────────────────────────────────────────────────
"""

    for i, res in enumerate(data['results'][:10], 1):
        ref = Path(res['reference']).name[:22]
        query = Path(res['query']).name[:22]
        text += f"{i:<5} {res['score']:.3f}   {res['result']['decision']:<18} {ref:<22} {query}
"

    text += """
───────────────────────────────────────────────────────────────
                    YASAL UYARI
───────────────────────────────────────────────────────────────

Bu rapor otomatik hesaplama araçları ile üretilmiştir ve tek 
başına kesin kimlik tespiti için yeterli değildir. Sonuçlar,
yetkili adli görüntü inceleme uzmanı tarafından değerlendirilmelidir.

Kulak yapısı, kimlik tespitinde destekleyici bir kriterdir.

Hazırlayan: Serkan Dinçer
Adli Ses ve Görüntü Karşılaştırma Uzmanı
═══════════════════════════════════════════════════════════════
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Metin raporu kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Adli Kulak Karşılaştırma Rapor Üretici")
    parser.add_argument("--input", "-i", required=True, help="JSON karşılaştırma sonuçları")
    parser.add_argument("--output", "-o", required=True, help="Çıktı dosya yolu")
    parser.add_argument("--format", "-f", default="markdown", choices=["markdown", "text"],
                        help="Rapor formatı")

    args = parser.parse_args()

    # JSON dosyasını oku
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Rapor üret
    if args.format == "markdown":
        generate_markdown_report(data, args.output)
    else:
        generate_text_report(data, args.output)

    print(f"✅ Rapor başarıyla üretildi: {args.output}")


if __name__ == "__main__":
    main()
