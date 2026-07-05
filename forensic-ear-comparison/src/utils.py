"""
Yardımcı fonksiyonlar ve görüntü işleme araçları.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from skimage.metrics import structural_similarity as ssim


def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    İki görüntü arasında Structural Similarity Index (SSIM) hesapla.

    SSIM, görüntü yapısal benzerliğini ölçer (0-1 arası, 1 mükemmel eşleşme).
    """
    # Aynı boyuta getir
    if image1.shape != image2.shape:
        h, w = max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (w, h))
        image2 = cv2.resize(image2, (w, h))

    score = ssim(image1, image2)
    # Normalize et (skimage bazen -1 ile 1 arası döner)
    return max(0.0, (score + 1) / 2)


def compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) hesapla.

    Düşük MSE = yüksek benzerlik.
    """
    if image1.shape != image2.shape:
        h, w = max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (w, h))
        image2 = cv2.resize(image2, (w, h))

    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def compute_histogram_similarity(image1: np.ndarray, image2: np.ndarray,
                                  method: int = cv2.HISTCMP_CORREL) -> float:
    """
    Histogram benzerliği hesapla.

    Args:
        method: OpenCV histogram karşılaştırma yöntemi

    Returns:
        float: 0-1 arası benzerlik skoru
    """
    # Histogramları hesapla
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Normalize et
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Karşılaştır
    similarity = cv2.compareHist(hist1, hist2, method)

    # Korelasyon yöntemi -1 ile 1 arası döner
    if method == cv2.HISTCMP_CORREL:
        return max(0.0, (similarity + 1) / 2)
    elif method in [cv2.HISTCMP_CHISQR, cv2.HISTCMP_CHISQR_ALT]:
        # Chi-square: düşük değer = yüksek benzerlik
        return max(0.0, 1.0 - similarity / 10)
    elif method == cv2.HISTCMP_INTERSECT:
        # Intersection: doğrudan benzerlik
        return min(1.0, similarity)
    elif method == cv2.HISTCMP_BHATTACHARYYA:
        # Bhattacharyya: 0 = mükemmel, 1 = farklı
        return max(0.0, 1.0 - similarity)
    else:
        return max(0.0, min(1.0, similarity))


def enhance_contrast(image: np.ndarray, 
                     method: str = "clahe") -> np.ndarray:
    """
    Görüntü kontrastını artır.

    Args:
        method: 'clahe' veya 'histogram'
    """
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        return cv2.equalizeHist(image)


def align_images(image1: np.ndarray, image2: np.ndarray,
                 max_features: int = 500,
                 good_match_percent: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Özellik tabanlı görüntü hizalama.

    Returns:
        Tuple: (hizalanmış_image1, hizalanmış_image2)
    """
    # ORB dedektör
    orb = cv2.ORB_create(max_features)

    # Anahtar noktaları ve tanımlayıcıları bul
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    if descriptors1 is None or descriptors2 is None:
        return image1, image2

    # Eşleştir
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # En iyi eşleşmeleri seç
    matches.sort(key=lambda x: x.distance)
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Noktaları çıkar
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Homografi bul
    if len(points1) >= 4:
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        if h is not None:
            height, width = image1.shape[:2]
            aligned = cv2.warpPerspective(image2, h, (width, height))
            return image1, aligned

    return image1, image2


def create_comparison_visualization(ear1: np.ndarray, 
                                     ear2: np.ndarray,
                                     score: float,
                                     save_path: Optional[str] = None) -> np.ndarray:
    """
    Karşılaştırma sonuçlarını görselleştir.

    Returns:
        np.ndarray: Yan yana karşılaştırma görüntüsü
    """
    # Renklendir
    if len(ear1.shape) == 2:
        ear1_color = cv2.cvtColor(ear1, cv2.COLOR_GRAY2BGR)
    else:
        ear1_color = ear1.copy()

    if len(ear2.shape) == 2:
        ear2_color = cv2.cvtColor(ear2, cv2.COLOR_GRAY2BGR)
    else:
        ear2_color = ear2.copy()

    # Boyutlandır
    h = max(ear1_color.shape[0], ear2_color.shape[0])
    w1 = ear1_color.shape[1]
    w2 = ear2_color.shape[1]

    ear1_color = cv2.resize(ear1_color, (w1, h))
    ear2_color = cv2.resize(ear2_color, (w2, h))

    # Skor rengi
    if score >= 0.8:
        color = (0, 255, 0)  # Yeşil
        text = "MATCH"
    elif score >= 0.6:
        color = (0, 255, 255)  # Sarı
        text = "INCONCLUSIVE"
    else:
        color = (0, 0, 255)  # Kırmızı
        text = "NO MATCH"

    # Birleştir
    gap = 20
    result = np.ones((h + 80, w1 + w2 + gap, 3), dtype=np.uint8) * 255
    result[:h, :w1] = ear1_color
    result[:h, w1+gap:w1+gap+w2] = ear2_color

    # Etiketler
    cv2.putText(result, "REFERANS", (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(result, "SORGU", (w1 + gap + 10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Skor
    score_text = f"Skor: {score:.3f} | {text}"
    cv2.putText(result, score_text, (10, h + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if save_path:
        cv2.imwrite(save_path, result)

    return result
