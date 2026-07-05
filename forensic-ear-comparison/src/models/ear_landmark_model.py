"""
Kulak anatomik landmark tespiti için derin öğrenme modeli.
Helix, antihelix, tragus, lobule gibi yapıların koordinatlarını tespit eder.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EarLandmarkModel:
    """
    Kulak anatomik landmark tespiti.

    Landmark yapıları:
    - Helix üst noktası
    - Helix alt noktası
    - Antihelix tepe noktası
    - Tragus ucu
    - Lobule alt noktası
    - Concha merkezi
    """

    LANDMARK_NAMES = [
        "helix_top", "helix_bottom", "antihelix_peak",
        "tragus_tip", "lobule_bottom", "concha_center"
    ]

    def __init__(self, model_path: str = None, device: str = None):
        """
        Args:
            model_path: Önceden eğitilmiş model yolu (opsiyonel)
            device: 'cuda' veya 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_landmarks = len(self.LANDMARK_NAMES)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("Önceden eğitilmiş model bulunamadı, varsayılan yapı kullanılıyor")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_model(self, model_path: str):
        """Modeli yükle."""
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            self.model = None

    def predict(self, ear_image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Kulak görüntüsünden landmark koordinatlarını tespit et.

        Args:
            ear_image: Gri tonlamalı kulak görüntüsü

        Returns:
            Dict: Landmark ismi -> (x, y) koordinatları
        """
        if self.model is not None:
            return self._predict_with_model(ear_image)
        else:
            return self._predict_heuristic(ear_image)

    def _predict_with_model(self, ear_image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Eğitilmiş model ile landmark tespiti."""
        h, w = ear_image.shape[:2]

        # Tensor'a çevir
        if len(ear_image.shape) == 2:
            ear_rgb = cv2.cvtColor(ear_image, cv2.COLOR_GRAY2RGB)
        else:
            ear_rgb = ear_image

        tensor = self.transform(ear_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            landmarks = self.model(tensor).cpu().numpy().flatten()

        # Normalize edilmiş koordinatları piksele çevir
        landmarks = landmarks.reshape(-1, 2)
        landmarks[:, 0] *= w
        landmarks[:, 1] *= h

        result = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            if i < len(landmarks):
                result[name] = (int(landmarks[i][0]), int(landmarks[i][1]))

        return result

    def _predict_heuristic(self, ear_image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Sezgisel (heuristic) landmark tespiti.

        Kulak morfolojisine dayalı basit geometrik kurallar kullanır.
        """
        h, w = ear_image.shape[:2]

        # Kontur analizi
        _, binary = cv2.threshold(ear_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {name: (w//2, h//2) for name in self.LANDMARK_NAMES}

        main_contour = max(contours, key=cv2.contourArea)

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(main_contour)

        # Sezgisel landmark pozisyonları
        landmarks = {
            "helix_top": (x + bw // 2, y + int(bh * 0.1)),
            "helix_bottom": (x + bw // 2, y + int(bh * 0.5)),
            "antihelix_peak": (x + int(bw * 0.55), y + int(bh * 0.25)),
            "tragus_tip": (x + int(bw * 0.15), y + int(bh * 0.45)),
            "lobule_bottom": (x + bw // 2, y + int(bh * 0.9)),
            "concha_center": (x + int(bw * 0.45), y + int(bh * 0.5))
        }

        return landmarks

    def visualize_landmarks(self, ear_image: np.ndarray,
                            landmarks: Dict[str, Tuple[int, int]],
                            save_path: str = None) -> np.ndarray:
        """
        Landmark'ları görüntü üzerinde çiz.

        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        if len(ear_image.shape) == 2:
            vis = cv2.cvtColor(ear_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = ear_image.copy()

        colors = [
            (0, 0, 255),    # Kırmızı
            (0, 255, 0),    # Yeşil
            (255, 0, 0),    # Mavi
            (0, 255, 255),  # Sarı
            (255, 0, 255),  # Magenta
            (255, 255, 0)   # Cyan
        ]

        for i, (name, (x, y)) in enumerate(landmarks.items()):
            color = colors[i % len(colors)]
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.putText(vis, name, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if save_path:
            cv2.imwrite(save_path, vis)

        return vis

    def compute_landmark_distances(self, 
                                    landmarks1: Dict[str, Tuple[int, int]],
                                    landmarks2: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """
        İki landmark seti arasındaki mesafeleri hesapla.

        Returns:
            Dict: Landmark ismi -> Öklid mesafesi
        """
        distances = {}
        for name in self.LANDMARK_NAMES:
            if name in landmarks1 and name in landmarks2:
                p1 = np.array(landmarks1[name])
                p2 = np.array(landmarks2[name])
                dist = np.linalg.norm(p1 - p2)
                distances[name] = round(dist, 2)

        return distances
