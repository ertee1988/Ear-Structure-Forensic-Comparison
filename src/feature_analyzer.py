"""
Kulak morfolojik özellik analizi modülü.
Helix, antihelix, tragus, lobule gibi anatomik yapıları tespit eder ve ölçer.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EarFeatures:
    """Kulak morfolojik özellikleri veri yapısı."""
    # Geometrik özellikler
    total_height: float           # Toplam kulak yüksekliği (px)
    total_width: float            # Toplam kulak genişliği (px)
    aspect_ratio: float           # Yükseklik/genişlik oranı

    # Helix özellikleri
    helix_curvature: float        # Helix kıvrım derecesi
    helix_length: float           # Helix uzunluğu

    # Antihelix özellikleri
    antihelix_prominence: float   # Antihelix belirginliği

    # Lobule özellikleri
    lobule_length: float          # Lobule uzunluğu
    lobule_width: float           # Lobule genişliği
    lobule_attachment: str        # 'free' (serbest) veya 'attached' (bitişik)

    # Tragus özellikleri
    tragus_size: float            # Tragus boyutu

    # Concha özellikleri
    concha_depth: float           # Concha derinliği

    # Dokusal özellikler
    texture_uniformity: float     # Doku homojenliği
    ridge_density: float          # Kıvrım yoğunluğu

    def to_vector(self) -> np.ndarray:
        """Özellikleri sayısal vektöre dönüştür."""
        attachment_map = {"free": 0.0, "attached": 1.0}
        return np.array([
            self.total_height,
            self.total_width,
            self.aspect_ratio,
            self.helix_curvature,
            self.helix_length,
            self.antihelix_prominence,
            self.lobule_length,
            self.lobule_width,
            attachment_map.get(self.lobule_attachment, 0.5),
            self.tragus_size,
            self.concha_depth,
            self.texture_uniformity,
            self.ridge_density
        ], dtype=np.float32)


class FeatureAnalyzer:
    """
    Kulak görüntüsünden morfolojik özellikleri çıkaran analiz sınıfı.

    Analiz edilen yapılar:
    - Helix (dış kulak kıvrımı)
    - Antihelix (iç kulak kıvrımı)
    - Tragus
    - Antitragus
    - Lobule (kulak memesi)
    - Concha (kulak çanağı)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_component_area = self.config.get("min_component_area", 100)

    def analyze(self, ear_image: np.ndarray) -> EarFeatures:
        """
        Kulak görüntüsünden tüm morfolojik özellikleri çıkar.

        Args:
            ear_image: Gri tonlamalı, normalleştirilmiş kulak görüntüsü

        Returns:
            EarFeatures: Çıkarılan özellikler
        """
        # Ön işleme
        preprocessed = self._preprocess(ear_image)

        # Temel geometri
        height, width = preprocessed.shape
        aspect_ratio = height / width if width > 0 else 0

        # Kontur analizi
        contours = self._extract_contours(preprocessed)

        # Helix analizi
        helix_features = self._analyze_helix(preprocessed, contours)

        # Lobule analizi
        lobule_features = self._analyze_lobule(preprocessed, contours)

        # Doku analizi
        texture_features = self._analyze_texture(preprocessed)

        features = EarFeatures(
            total_height=float(height),
            total_width=float(width),
            aspect_ratio=round(aspect_ratio, 3),
            helix_curvature=helix_features.get("curvature", 0.0),
            helix_length=helix_features.get("length", 0.0),
            antihelix_prominence=helix_features.get("antihelix_prom", 0.0),
            lobule_length=lobule_features.get("length", 0.0),
            lobule_width=lobule_features.get("width", 0.0),
            lobule_attachment=lobule_features.get("attachment", "unknown"),
            tragus_size=self._analyze_tragus(preprocessed, contours),
            concha_depth=self._analyze_concha(preprocessed),
            texture_uniformity=texture_features.get("uniformity", 0.0),
            ridge_density=texture_features.get("ridge_density", 0.0)
        )

        logger.info(f"Özellik analizi tamamlandı: aspect_ratio={features.aspect_ratio}")
        return features

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü analiz için hazırla."""
        # Boyut kontrolü
        if image.shape[0] < 50 or image.shape[1] < 50:
            raise ValueError("Görüntü boyutu çok küçük (min 50x50)")

        # Gaussian blur ile yumuşatma
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Adaptif eşikleme
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morfolojik işlemler
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return cleaned

    def _extract_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """İkili görüntüden konturları çıkar."""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Alanına göre sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def _analyze_helix(self, image: np.ndarray, 
                       contours: List[np.ndarray]) -> Dict[str, float]:
        """
        Helix (dış kulak kıvrımı) analizi.

        Helix, kulak kepçesinin en dış kıvrımıdır ve en belirgin yapıdır.
        """
        if not contours:
            return {"curvature": 0.0, "length": 0.0, "antihelix_prom": 0.0}

        # En büyük konturu al (kulak dış hattı)
        main_contour = max(contours, key=cv2.contourArea)

        # Kontur uzunluğu
        perimeter = cv2.arcLength(main_contour, False)

        # Kıvrım analizi - konturun kompaktlığı
        area = cv2.contourArea(main_contour)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            curvature = 1.0 - compactness  # Daha az kompakt = daha kıvrımlı
        else:
            curvature = 0.0

        # Antihelix belirginliği - iç kıvrımların sayısı ve derinliği
        if len(contours) > 1:
            inner_contours = contours[1:]
            total_inner_area = sum(cv2.contourArea(c) for c in inner_contours)
            antihelix_prom = min(1.0, total_inner_area / (area + 1e-6))
        else:
            antihelix_prom = 0.0

        return {
            "curvature": round(curvature, 3),
            "length": round(perimeter, 1),
            "antihelix_prom": round(antihelix_prom, 3)
        }

    def _analyze_lobule(self, image: np.ndarray,
                        contours: List[np.ndarray]) -> Dict[str, any]:
        """
        Lobule (kulak memesi) analizi.

        Lobule, kulak kepçesinin en alt kısmıdır.
        Serbest (free) veya bitişik (attached) olabilir.
        """
        h, w = image.shape

        # Alt bölgeyi analiz et (lobule bölgesi)
        lobule_region = image[int(h*0.7):, :]

        # Lobule konturları
        lobule_contours, _ = cv2.findContours(
            lobule_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not lobule_contours:
            return {"length": 0.0, "width": 0.0, "attachment": "unknown"}

        # En büyük kontur
        main_lobule = max(lobule_contours, key=cv2.contourArea)
        x, y, lw, lh = cv2.boundingRect(main_lobule)

        # Bağlantı tipi tespiti
        # Bitişik lobule: alt kenar düz, serbest lobule: yuvarlak alt kenar
        hull = cv2.convexHull(main_lobule)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(main_lobule)

        if hull_area > 0:
            solidity = contour_area / hull_area
            # Solidity düşükse = yuvarlak/serbest, yüksekse = düz/bitişik
            attachment = "free" if solidity < 0.85 else "attached"
        else:
            attachment = "unknown"

        return {
            "length": round(lh, 1),
            "width": round(lw, 1),
            "attachment": attachment
        }

    def _analyze_tragus(self, image: np.ndarray, 
                        contours: List[np.ndarray]) -> float:
        """
        Tragus analizi.

        Tragus, kulak kanalının önündeki küçük çıkıntıdır.
        """
        h, w = image.shape
        # Tragus genellikle üst-orta bölgede bulunur
        tragus_region = image[:int(h*0.5), :int(w*0.4)]

        # Küçük bileşenleri tespit et
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            tragus_region, connectivity=8
        )

        # Arka planı atla, küçük bileşenleri say
        small_components = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_component_area // 4 < area < self.min_component_area * 2:
                small_components += 1

        # Tragus boyutu skoru (0-1)
        tragus_score = min(1.0, small_components / 5.0)
        return round(tragus_score, 3)

    def _analyze_concha(self, image: np.ndarray) -> float:
        """
        Concha (kulak çanağı) derinlik analizi.

        Concha, helix ve antihelix arasındaki çukur bölgedir.
        """
        h, w = image.shape
        # Merkez bölge
        center_region = image[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

        # Derinlik = iç bölgelerin karanlıklığı / dış bölgelerin aydınlığı
        center_mean = center_region.mean()
        border_mean = (
            image[:int(h*0.3), :].mean() + 
            image[int(h*0.7):, :].mean()
        ) / 2

        if border_mean > 0:
            depth_ratio = center_mean / border_mean
            return round(min(1.0, depth_ratio), 3)
        return 0.5

    def _analyze_texture(self, image: np.ndarray) -> Dict[str, float]:
        """
        Doku analizi - kulak kıvrımlarının yoğunluğu ve düzenliliği.
        """
        # Gabor filtresi ile doku analizi
        gabor_responses = []
        for theta in np.arange(0, np.pi, np.pi/4):
            for freq in [0.1, 0.2, 0.3]:
                kernel = cv2.getGaborKernel(
                    (21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                gabor_responses.append(filtered.var())

        # Kıvrım yoğunluğu
        ridge_density = np.mean(gabor_responses) / 1000.0
        ridge_density = min(1.0, ridge_density)

        # Doku homojenliği
        uniformity = 1.0 - (np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-6))
        uniformity = max(0.0, min(1.0, uniformity))

        return {
            "uniformity": round(uniformity, 3),
            "ridge_density": round(ridge_density, 3)
        }

    def compare_features(self, features1: EarFeatures, 
                         features2: EarFeatures) -> Dict[str, float]:
        """
        İki kulak özellik seti arasında karşılaştırma yap.

        Returns:
            Dict: Her özellik için benzerlik skoru (0-1)
        """
        vec1 = features1.to_vector()
        vec2 = features2.to_vector()

        # Normalize et
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)

        # Kosinüs benzerliği
        cosine_sim = np.dot(vec1_norm, vec2_norm)

        # Özellik bazlı karşılaştırma
        individual_scores = {}

        # Geometrik benzerlik
        geo_diff = abs(features1.aspect_ratio - features2.aspect_ratio)
        individual_scores["geometry"] = max(0.0, 1.0 - geo_diff)

        # Lobule benzerliği
        if features1.lobule_attachment == features2.lobule_attachment:
            individual_scores["lobule"] = 0.9
        else:
            individual_scores["lobule"] = 0.3

        # Kıvrım benzerliği
        curve_diff = abs(features1.helix_curvature - features2.helix_curvature)
        individual_scores["curvature"] = max(0.0, 1.0 - curve_diff)

        # Genel skor
        individual_scores["overall"] = round((cosine_sim + 1) / 2, 3)

        return individual_scores
