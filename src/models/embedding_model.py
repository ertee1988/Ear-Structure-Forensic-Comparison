"""
Kulak embedding modeli - Siamese Network mimarisi ile kulak benzerliği öğrenme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class EarEmbeddingModel(nn.Module):
    """
    Kulak görüntülerinden düşük boyutlu embedding vektörü üreten model.

    Mimarisi:
    - ResNet50 backbone (önceden eğitilmiş)
    - Özelleştirilmiş FC katmanları
    - L2 normalize edilmiş embedding çıktısı
    """

    def __init__(self, embedding_dim: int = 128, 
                 backbone: str = "resnet50",
                 pretrained: bool = True):
        """
        Args:
            embedding_dim: Çıktı embedding boyutu
            backbone: 'resnet50', 'efficientnet_b0', 'mobilenet_v3'
            pretrained: ImageNet önceden eğitilmiş ağırlıkları kullan
        """
        super(EarEmbeddingModel, self).__init__()

        self.embedding_dim = embedding_dim

        # Backbone seçimi
        if backbone == "resnet50":
            base_model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone}")

        # Embedding projeksiyon katmanları
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

        logger.info(f"EarEmbeddingModel oluşturuldu: {backbone}, dim={embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri yayılım.

        Args:
            x: Girdi görüntü tensor'u (B, C, H, W)

        Returns:
            torch.Tensor: L2 normalize edilmiş embedding (B, embedding_dim)
        """
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        embedding = self.projection(features)
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def get_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Tek görüntüden embedding çıkar.

        Args:
            image_tensor: (1, C, H, W) veya (C, H, W) şeklinde tensor

        Returns:
            np.ndarray: Embedding vektörü
        """
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            embedding = self.forward(image_tensor)

        return embedding.cpu().numpy().flatten()


class TripletLoss(nn.Module):
    """
    Triplet Loss - Siamese eğitimi için.

    anchor, positive (aynı kişi), negative (farklı kişi)
    """

    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Referans embedding
            positive: Aynı sınıf embedding
            negative: Farklı sınıf embedding

        Returns:
            torch.Tensor: Triplet loss değeri
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Kontrastif Loss - İkili karşılaştırma için.
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor,
                embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1, embedding2: Karşılaştırılacak embedding'ler
            label: 1 (eşleşme) veya 0 (eşleşme yok)

        Returns:
            torch.Tensor: Contrastive loss
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2)

        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss


def compute_similarity(embedding1: np.ndarray, 
                       embedding2: np.ndarray,
                       metric: str = "cosine") -> float:
    """
    İki embedding arasında benzerlik hesapla.

    Args:
        embedding1, embedding2: Embedding vektörleri
        metric: 'cosine', 'euclidean', 'manhattan'

    Returns:
        float: 0-1 arası benzerlik skoru
    """
    if metric == "cosine":
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )
        # -1, 1 aralığını 0, 1 aralığına çevir
        return (similarity + 1) / 2

    elif metric == "euclidean":
        dist = np.linalg.norm(embedding1 - embedding2)
        # Uzaklığı benzerliğe çevir
        return max(0, 1 - dist / 2)

    elif metric == "manhattan":
        dist = np.sum(np.abs(embedding1 - embedding2))
        return max(0, 1 - dist / 10)

    else:
        raise ValueError(f"Desteklenmeyen metrik: {metric}")
