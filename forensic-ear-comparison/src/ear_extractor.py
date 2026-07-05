"""
Kulak bölgesi tespiti ve çıkarma modülü.
Görüntü/video karelerinden kulak bölgesini tespit eder, kırpar ve normalleştirir.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EarExtractor:
    """
    Görüntülerden kulak bölgesini tespit eden ve çıkaran ana sınıf.

    Desteklenen yöntemler:
    - haar: OpenCV Haar Cascade ile kulak tespiti
    - mediapipe: MediaPipe Face Mesh üzerinden kulak bölgesi çıkarma
    - manual: Kullanıcı tanımlı ROI
    """

    def __init__(self, method: str = "mediapipe", config: dict = None):
        """
        Args:
            method: Tespit yöntemi ('haar', 'mediapipe', 'manual')
            config: Yapılandırma parametreleri
        """
        self.method = method
        self.config = config or {}
        self._init_detector()

    def _init_detector(self):
        """Seçilen yönteme göre dedektörü başlat."""
        if self.method == "haar":
            cascade_path = self.config.get(
                "cascade_path",
                cv2.data.haarcascades + "haarcascade_mcs_leftear.xml"
            )
            if not Path(cascade_path).exists():
                logger.warning("Haar cascade bulunamadı, mediapipe'e geçiliyor")
                self.method = "mediapipe"
            else:
                self.detector = cv2.CascadeClassifier(cascade_path)

        elif self.method == "mediapipe":
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except ImportError:
                logger.warning("MediaPipe kurulu değil, manual moda geçiliyor")
                self.method = "manual"

        logger.info(f"EarExtractor başlatıldı: method={self.method}")

    def extract(self, image_path: str, side: str = "auto") -> np.ndarray:
        """
        Görüntüden kulak bölgesini çıkar.

        Args:
            image_path: Görüntü dosya yolu
            side: 'left', 'right' veya 'auto'

        Returns:
            np.ndarray: Kırpılmış ve normalleştirilmiş kulak görüntüsü

        Raises:
            ValueError: Kulak tespit edilemezse
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.method == "haar":
            ear_region = self._extract_haar(image)
        elif self.method == "mediapipe":
            ear_region = self._extract_mediapipe(image_rgb, side)
        else:
            ear_region = self._extract_manual(image)

        if ear_region is None:
            raise ValueError("Kulak bölgesi tespit edilemedi")

        return self._normalize(ear_region)

    def _extract_haar(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Haar Cascade ile kulak tespiti."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ears = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        if len(ears) == 0:
            return None
        # En büyük tespiti al
        x, y, w, h = max(ears, key=lambda r: r[2] * r[3])
        return image[y:y+h, x:x+w]

    def _extract_mediapipe(self, image_rgb: np.ndarray, side: str) -> Optional[np.ndarray]:
        """
        MediaPipe Face Mesh landmark'larından kulak bölgesini çıkar.

        Landmark indeksleri:
        - Sol kulak: 234, 93, 132, 58, 172, 136, 150, 149, 176, 148
        - Sağ kulak: 454, 323, 361, 288, 397, 365, 379, 378, 400, 377
        """
        h, w = image_rgb.shape[:2]
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Sol kulak landmark indeksleri
        left_ear_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
        # Sağ kulak landmark indeksleri  
        right_ear_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]

        if side == "auto":
            # Her iki kulak bölgesini dene, daha belirgin olanı seç
            left_region = self._crop_from_landmarks(image_rgb, landmarks, left_ear_indices, w, h)
            right_region = self._crop_from_landmarks(image_rgb, landmarks, right_ear_indices, w, h)

            if left_region is None and right_region is None:
                return None
            elif left_region is None:
                return right_region
            elif right_region is None:
                return left_region
            else:
                # Daha büyük olanı seç
                left_area = left_region.shape[0] * left_region.shape[1]
                right_area = right_region.shape[0] * right_region.shape[1]
                return left_region if left_area > right_area else right_region
        elif side == "left":
            return self._crop_from_landmarks(image_rgb, landmarks, left_ear_indices, w, h)
        else:
            return self._crop_from_landmarks(image_rgb, landmarks, right_ear_indices, w, h)

    def _crop_from_landmarks(self, image: np.ndarray, landmarks, 
                              indices: List[int], img_w: int, img_h: int) -> Optional[np.ndarray]:
        """Landmark koordinatlarından bölge kırp."""
        points = []
        for idx in indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * img_w)
                y = int(landmarks[idx].y * img_h)
                points.append((x, y))

        if len(points) < 3:
            return None

        points = np.array(points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # %20 padding ekle
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(img_w, x_max + pad_x)
        y_max = min(img_h, y_max + pad_y)

        return image[y_min:y_max, x_min:x_max]

    def _extract_manual(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Manuel ROI seçimi (interaktif)."""
        # CLI kullanımı için basit bir kırpma
        # GUI entegrasyonu için cv2.selectROI kullanılabilir
        h, w = image.shape[:2]
        # Varsayılan olarak görüntünün sağ tarafını dene (yaygın profil görünümü)
        return image[:, w//2:]

    def _normalize(self, ear_region: np.ndarray, 
                   target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Kulak görüntüsünü normalleştir.

        İşlemler:
        1. Boyutlandırma
        2. Gri tonlamaya çevirme
        3. Histogram eşitleme (kontrast artırma)
        4. Gürültü azaltma
        """
        # Boyutlandırma
        normalized = cv2.resize(ear_region, target_size, interpolation=cv2.INTER_LANCZOS4)

        # Gri tonlama
        if len(normalized.shape) == 3:
            gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        else:
            gray = normalized

        # Histogram eşitleme
        equalized = cv2.equalizeHist(gray)

        # Gürültü azaltma
        denoised = cv2.fastNlMeansDenoising(equalized, None, 10, 7, 21)

        return denoised

    def extract_from_video(self, video_path: str, 
                           frame_interval: int = 30) -> List[np.ndarray]:
        """
        Video dosyasından periyodik olarak kulak görüntüleri çıkar.

        Args:
            video_path: Video dosya yolu
            frame_interval: Kaç karede bir örnek alınacağı

        Returns:
            List[np.ndarray]: Çıkarılan kulak görüntüleri listesi
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ear = self._extract_mediapipe(frame_rgb, "auto")
                    if ear is not None:
                        frames.append(self._normalize(ear))
                except Exception as e:
                    logger.warning(f"Frame {frame_count} işlenirken hata: {e}")

            frame_count += 1

        cap.release()
        logger.info(f"Video'dan {len(frames)} kulak görüntüsü çıkarıldı")
        return frames

    def get_quality_score(self, ear_image: np.ndarray) -> float:
        """
        Çıkarılan kulak görüntüsünün kalite skorunu hesapla.

        Skor 0-1 arası:
        - Çözünürlük
        - Keskinlik (Laplacian varyansı)
        - Kontrast
        """
        # Çözünürlük skoru
        h, w = ear_image.shape[:2]
        resolution_score = min(1.0, (h * w) / (256 * 256))

        # Keskinlik skoru
        laplacian_var = cv2.Laplacian(ear_image, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)

        # Kontrast skoru
        contrast = ear_image.std() / 128.0
        contrast_score = min(1.0, contrast)

        # Ağırlıklı ortalama
        quality = (0.3 * resolution_score + 
                   0.4 * sharpness_score + 
                   0.3 * contrast_score)

        return round(quality, 3)
