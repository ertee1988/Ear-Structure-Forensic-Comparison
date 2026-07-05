#!/usr/bin/env python3
"""
Toplu kulak karşılaştırma işlemleri.
Birden fazla referans ve sorgu görüntüsünü karşılaştırır.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List
import logging

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ear_extractor import EarExtractor
from src.ear_matcher import EarMatcher
from src.utils import create_comparison_visualization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_image_files(directory: str) -> List[str]:
    """Dizindeki tüm görüntü dosyalarını bul."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted([str(f) for f in files])


def main():
    parser = argparse.ArgumentParser(
        description="Forensic Ear Comparison - Toplu İşlem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanım:
  python batch_process.py --reference ./data/ref/ --query ./data/query/ --output ./reports/
  python batch_process.py --reference ref_ear.png --query ./query_dir/ --method hybrid --threshold strict
        """
    )

    parser.add_argument("--reference", "-r", required=True,
                        help="Referans kulak görüntüsü veya dizini")
    parser.add_argument("--query", "-q", required=True,
                        help="Sorgu kulak görüntüleri dizini")
    parser.add_argument("--output", "-o", default="./reports",
                        help="Çıktı dizini (varsayılan: ./reports)")
    parser.add_argument("--method", "-m", default="hybrid",
                        choices=["geometric", "texture", "deep", "hybrid"],
                        help="Karşılaştırma yöntemi")
    parser.add_argument("--threshold", "-t", default="standard",
                        choices=["strict", "standard", "lenient"],
                        help="Eşik seviyesi")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                        help="En iyi K sonucu göster")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Karşılaştırma görselleştirmeleri üret")
    parser.add_argument("--extract-first", "-e", action="store_true",
                        help="Önce kulak bölgesini çıkar (tam görüntüler için)")

    args = parser.parse_args()

    # Çıktı dizinini oluştur
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Referans dosyalarını bul
    ref_path = Path(args.reference)
    if ref_path.is_dir():
        ref_files = find_image_files(str(ref_path))
        if not ref_files:
            logger.error(f"Referans dizininde görüntü bulunamadı: {ref_path}")
            sys.exit(1)
        logger.info(f"{len(ref_files)} referans görüntü bulundu")
    else:
        ref_files = [str(ref_path)]

    # Sorgu dosyalarını bul
    query_files = find_image_files(args.query)
    if not query_files:
        logger.error(f"Sorgu dizininde görüntü bulunamadı: {args.query}")
        sys.exit(1)
    logger.info(f"{len(query_files)} sorgu görüntü bulundu")

    # Kulak çıkarıcı (eğer tam yüz görüntüleri kullanılıyorsa)
    extractor = None
    if args.extract_first:
        extractor = EarExtractor(method="mediapipe")
        logger.info("Kulak çıkarıcı başlatıldı")

    # Matcher başlat
    matcher = EarMatcher(method=args.method, threshold_level=args.threshold)
    logger.info(f"Matcher başlatıldı: method={args.method}, threshold={args.threshold}")

    # Tüm karşılaştırmaları yap
    all_results = []

    for ref_file in ref_files:
        logger.info(f"Referans işleniyor: {ref_file}")

        # Referans kulak görüntüsünü hazırla
        if extractor:
            try:
                ref_ear = extractor.extract(ref_file)
                ref_temp = output_dir / f"ref_{Path(ref_file).stem}.png"
                import cv2
                cv2.imwrite(str(ref_temp), ref_ear)
                ref_file = str(ref_temp)
            except Exception as e:
                logger.warning(f"Referans çıkarılamadı: {e}")
                continue

        for query_file in query_files:
            try:
                # Sorgu kulak görüntüsünü hazırla
                if extractor:
                    query_ear = extractor.extract(query_file)
                    query_temp = output_dir / f"query_{Path(query_file).stem}.png"
                    import cv2
                    cv2.imwrite(str(query_temp), query_ear)
                    query_file_processed = str(query_temp)
                else:
                    query_file_processed = query_file

                # Karşılaştır
                score, result = matcher.compare(ref_file, query_file_processed)

                all_results.append({
                    "reference": ref_file,
                    "query": query_file,
                    "score": score,
                    "result": result.to_dict()
                })

                # Görselleştirme
                if args.visualize and score > 0.5:
                    vis_path = output_dir / f"compare_{Path(ref_file).stem}_{Path(query_file).stem}.png"
                    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
                    q_img = cv2.imread(query_file_processed, cv2.IMREAD_GRAYSCALE)
                    create_comparison_visualization(ref_img, q_img, score, str(vis_path))

            except Exception as e:
                logger.warning(f"Karşılaştırma hatası ({query_file}): {e}")
                continue

    # Sonuçları sırala
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # JSON raporu kaydet
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "method": args.method,
                "threshold": args.threshold,
                "total_comparisons": len(all_results),
                "reference_count": len(ref_files),
                "query_count": len(query_files)
            },
            "results": all_results[:args.top_k]
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Rapor kaydedildi: {report_path}")

    # Özet tablo yazdır
    print("\n" + "="*80)
    print(f"{'Sıra':<6}{'Skor':<10}{'Karar':<18}{'Referans':<30}{'Sorgu':<30}")
    print("="*80)

    for i, res in enumerate(all_results[:args.top_k], 1):
        ref_name = Path(res["reference"]).name[:28]
        query_name = Path(res["query"]).name[:28]
        decision = res["result"]["decision"]
        score = res["score"]
        print(f"{i:<6}{score:<10.3f}{decision:<18}{ref_name:<30}{query_name:<30}")

    print("="*80)

    # Eşleşme istatistikleri
    matches = sum(1 for r in all_results if r["result"]["decision"] == "MATCH")
    probables = sum(1 for r in all_results if r["result"]["decision"] == "PROBABLE_MATCH")
    print(f"\nÖzet: {matches} MATCH, {probables} PROBABLE_MATCH / {len(all_results)} toplam")


if __name__ == "__main__":
    main()
