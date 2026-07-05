"""
Kulak karşılaştırma ve eşleştirme motoru.
Geometrik, dokusal ve derin öğrenme tabanlı benzerlik hesaplama.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from .feature_analyzer import FeatureAnalyzer, EarFeatures
from .utils import compute_ssim, compute_mse, compute_histogram_similarity

logger = logging.getLogger(__name__)


class MatchMethod(Enum):
    """Karşılaştırma yöntemleri."""
    GEOMETRIC = "geometric"      # Morfolojik özellik tabanlı
    TEXTURE = "texture"          # Doku ve görünüm tabanlı
    DEEP = "deep"                # Derin öğrenme embedding
    HYBRID = "hybrid"            # Tüm yöntemlerin kombinasyonu


@dataclass
class MatchResult:
    """Karşılaştırma sonuç veri yapısı."""
    similarity_score: float       # 0-1 arası genel benzerlik
    confidence: float             # Güven seviyesi (0-1)
    method: str                   # Kullanılan yöntem
    feature_scores: Dict[str, float]  # Detaylı özellik skorları
    decision: str                 # 'MATCH', 'NO_MATCH', 'INCONCLUSIVE'
    threshold_used: float         # Kullanılan eşik değeri

    def to_dict(self) -> Dict:
        return {
            "similarity_score": round(self.similarity_score, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "feature_scores": {k: round(v, 4) for k, v in self.feature_scores.items()},
            "decision": self.decision,
            "threshold_used": self.threshold_used
        }


class EarMatcher:
    """
    İki kulak görüntüsü arasında karşılaştırma yapan ana sınıf.

    Desteklenen yöntemler:
    - geometric: Morfolojik özellikler (helix, lobule, tragus vb.)
    - texture: Görüntü dokusu ve yapısal benzerlik
    - deep: Önceden eğitilmiş CNN modeli embedding karşılaştırması
    - hybrid: Tüm yöntemlerin ağırlıklı kombinasyonu
    """

    # Eşik değerleri (adli standartlara göre ayarlanabilir)
    THRESHOLDS = {
        "strict": 0.85,      # Kesin eşleşme
        "standard": 0.75,    # Standart eşleşme
        "lenient": 0.65,     # Geniş eşleşme
    }

    def __init__(self, method: str = "hybrid", 
                 threshold_level: str = "standard",
                 config: dict = None):
        """
        Args:
            method: Karşılaştırma yöntemi
            threshold_level: 'strict', 'standard', 'lenient'
            config: Ek yapılandırma
        """
        self.method = MatchMethod(method)
        self.threshold = self.THRESHOLDS.get(threshold_level, 0.75)
        self.config = config or {}
        self.feature_analyzer = FeatureAnalyzer(config)
        self.deep_model = None

        if self.method in [MatchMethod.DEEP, MatchMethod.HYBRID]:
            self._init_deep_model()

        logger.info(f"EarMatcher başlatıldı: method={method}, threshold={self.threshold}")

    def _init_deep_model(self):
        """Derin öğrenme modelini başlat."""
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision.models import resnet50, ResNet50_Weights

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ResNet50 backbone (ImageNet üzerinde önceden eğitilmiş)
            self.deep_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Son FC katmanını çıkar, embedding üret
            self.deep_model.fc = torch.nn.Identity()
            self.deep_model = self.deep_model.to(self.device)
            self.deep_model.eval()

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info("Derin öğrenme modeli yüklendi (ResNet50)")
        except ImportError:
            logger.warning("PyTorch kurulu değil, derin öğrenme devre dışı")
            self.deep_model = None

    def compare(self, ear1_path: str, ear2_path: str) -> Tuple[float, MatchResult]:
        """
        İki kulak görüntüsünü karşılaştır.

        Args:
            ear1_path: Referans kulak görüntüsü
            ear2_path: Sorgu kulak görüntüsü

        Returns:
            Tuple[float, MatchResult]: Benzerlik skoru ve detaylı sonuç
        """
        # Görüntüleri yükle
        ear1 = cv2.imread(ear1_path, cv2.IMREAD_GRAYSCALE)
        ear2 = cv2.imread(ear2_path, cv2.IMREAD_GRAYSCALE)

        if ear1 is None or ear2 is None:
            raise ValueError("Görüntülerden biri okunamadı")

        # Boyutlandırma (aynı boyuta getir)
        target_size = (256, 256)
        ear1 = cv2.resize(ear1, target_size)
        ear2 = cv2.resize(ear2, target_size)

        # Yönteme göre karşılaştır
        if self.method == MatchMethod.GEOMETRIC:
            score, details = self._compare_geometric(ear1, ear2)
        elif self.method == MatchMethod.TEXTURE:
            score, details = self._compare_texture(ear1, ear2)
        elif self.method == MatchMethod.DEEP:
            score, details = self._compare_deep(ear1, ear2)
        else:  # HYBRID
            score, details = self._compare_hybrid(ear1, ear2)

        # Karar ver
        decision = self._make_decision(score, details)

        # Güven hesapla
        confidence = self._compute_confidence(score, details)

        result = MatchResult(
            similarity_score=round(score, 4),
            confidence=round(confidence, 4),
            method=self.method.value,
            feature_scores=details,
            decision=decision,
            threshold_used=self.threshold
        )

        return score, result

    def _compare_geometric(self, ear1: np.ndarray, 
                           ear2: np.ndarray) -> Tuple[float, Dict]:
        """Morfolojik özellik tabanlı karşılaştırma."""
        features1 = self.feature_analyzer.analyze(ear1)
        features2 = self.feature_analyzer.analyze(ear2)

        scores = self.feature_analyzer.compare_features(features1, features2)

        # Ağırlıklı skor
        weights = {
            "geometry": 0.25,
            "curvature": 0.25,
            "lobule": 0.20,
            "overall": 0.30
        }

        total_score = sum(scores.get(k, 0) * w for k, w in weights.items())

        return total_score, scores

    def _compare_texture(self, ear1: np.ndarray, 
                         ear2: np.ndarray) -> Tuple[float, Dict]:
        """Doku ve görünüm tabanlı karşılaştırma."""
        details = {}

        # SSIM (Structural Similarity Index)
        ssim_score = compute_ssim(ear1, ear2)
        details["ssim"] = ssim_score

        # MSE (Mean Squared Error)
        mse_score = compute_mse(ear1, ear2)
        # MSE'yi 0-1 arasına normalize et (düşük MSE = yüksek benzerlik)
        mse_normalized = max(0, 1 - mse_score / 10000)
        details["mse_normalized"] = mse_normalized

        # Histogram benzerliği
        hist_score = compute_histogram_similarity(ear1, ear2)
        details["histogram"] = hist_score

        # ORB özellik eşleştirme
        orb_score = self._orb_match(ear1, ear2)
        details["orb_features"] = orb_score

        # Ağırlıklı kombinasyon
        score = (0.35 * ssim_score + 
                 0.20 * mse_normalized + 
                 0.25 * hist_score + 
                 0.20 * orb_score)

        return score, details

    def _compare_deep(self, ear1: np.ndarray, 
                      ear2: np.ndarray) -> Tuple[float, Dict]:
        """Derin öğrenme embedding tabanlı karşılaştırma."""
        if self.deep_model is None:
            logger.warning("Derin öğrenme modeli kullanılamıyor, texture'a geçiliyor")
            return self._compare_texture(ear1, ear2)

        import torch

        # Görüntüleri tensor'a çevir
        ear1_rgb = cv2.cvtColor(ear1, cv2.COLOR_GRAY2RGB)
        ear2_rgb = cv2.cvtColor(ear2, cv2.COLOR_GRAY2RGB)

        tensor1 = self.transform(ear1_rgb).unsqueeze(0).to(self.device)
        tensor2 = self.transform(ear2_rgb).unsqueeze(0).to(self.device)

        # Embedding üret
        with torch.no_grad():
            emb1 = self.deep_model(tensor1).cpu().numpy().flatten()
            emb2 = self.deep_model(tensor2).cpu().numpy().flatten()

        # Kosinüs benzerliği
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Öklid mesafesi
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        euclidean_score = max(0, 1 - euclidean_dist / 10)

        details = {
            "cosine_similarity": round((cosine_sim + 1) / 2, 4),
            "euclidean_score": round(euclidean_score, 4)
        }

        score = (details["cosine_similarity"] + details["euclidean_score"]) / 2

        return score, details

    def _compare_hybrid(self, ear1: np.ndarray, 
                        ear2: np.ndarray) -> Tuple[float, Dict]:
        """Tüm yöntemlerin kombinasyonu."""
        geo_score, geo_details = self._compare_geometric(ear1, ear2)
        tex_score, tex_details = self._compare_texture(ear1, ear2)

        details = {
            "geometric": round(geo_score, 4),
            "texture": round(tex_score, 4),
            "geometric_details": geo_details,
            "texture_details": tex_details
        }

        if self.deep_model is not None:
            deep_score, deep_details = self._compare_deep(ear1, ear2)
            details["deep"] = round(deep_score, 4)
            details["deep_details"] = deep_details

            # Ağırlıklı kombinasyon
            score = (0.30 * geo_score + 
                     0.35 * tex_score + 
                     0.35 * deep_score)
        else:
            score = (0.45 * geo_score + 
                     0.55 * tex_score)

        return score, details

    def _orb_match(self, ear1: np.ndarray, ear2: np.ndarray) -> float:
        """ORB özellik eşleştirme skoru."""
        orb = cv2.ORB_create(nfeatures=500)

        kp1, des1 = orb.detectAndCompute(ear1, None)
        kp2, des2 = orb.detectAndCompute(ear2, None)

        if des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return 0.0

        # İyi eşleşmeleri filtrele
        good_matches = [m for m in matches if m.distance < 50]

        # Skor = iyi eşleşme oranı
        score = len(good_matches) / max(len(kp1), len(kp2), 1)
        return min(1.0, score)

    def _make_decision(self, score: float, details: Dict) -> str:
        """Benzerlik skoruna göre karar ver."""
        if score >= self.threshold + 0.1:
            return "MATCH"
        elif score >= self.threshold:
            return "PROBABLE_MATCH"
        elif score >= self.threshold - 0.15:
            return "INCONCLUSIVE"
        else:
            return "NO_MATCH"

    def _compute_confidence(self, score: float, details: Dict) -> float:
        """Karar güven seviyesini hesapla."""
        # Temel güven: skorun eşikten ne kadar uzak olduğu
        distance_from_threshold = abs(score - self.threshold)
        base_confidence = min(1.0, distance_from_threshold * 5)

        # Detay tutarlılığı
        if "geometric" in details and "texture" in details:
            consistency = 1.0 - abs(details["geometric"] - details["texture"])
            base_confidence *= (0.7 + 0.3 * consistency)

        return min(1.0, base_confidence)

    def batch_compare(self, reference_path: str, 
                      query_paths: List[str]) -> List[Tuple[str, MatchResult]]:
        """
        Birden fazla sorgu görüntüsünü tek referansla karşılaştır.

        Returns:
            List[Tuple[str, MatchResult]]: (dosya_yolu, sonuç) çiftleri
        """
        results = []
        for query_path in query_paths:
            try:
                score, result = self.compare(reference_path, query_path)
                results.append((query_path, result))
            except Exception as e:
                logger.error(f"{query_path} karşılaştırılırken hata: {e}")

        # Skora göre sırala
        results.sort(key=lambda x: x[1].similarity_score, reverse=True)
        return results
