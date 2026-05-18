"""
Federe Öğrenme - Yardımcı Araçlar (Dataset, Transform, Model, Param I/O)
=========================================================================

Bu modül; Flower istemci/sunucu betiklerinin ortak kullanacağı şu
bileşenleri tek bir yerden sağlar:
    - FedClientDataset : client_{i}.csv dosyasından (image_path, label)
      okuyan PyTorch Dataset.
    - build_transforms : train_baseline.py ile BİREBİR aynı ön işleme
      pipeline'ı (SquarePad -> Resize(224,224) -> ToTensor -> ImageNet).
    - build_model      : ResNet50 (weights=None) + ikili sınıflandırma
      başlığı.
    - get_/set_parameters : PyTorch state_dict <-> NumPy parametre listesi
      (Flower'ın beklediği format) dönüşümleri.

Not:
----
Sentetik görüntüler (data/synthetic/generated_malign/*.jpg) 128x128
üretiliyor; HAM10000 gerçek görüntüler 600x450 civarı. SquarePad +
Resize(224,224) ikisini de güvenle aynı formata getirir.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models

# src/data_loader.py içindeki SquarePad'i tek kaynaktan paylaşalım
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from data_loader import SquarePad  # noqa: E402


# =============================================================================
# 1. DATASET
# =============================================================================
class FedClientDataset(Dataset):
    """
    Federe istemci için CSV tabanlı görüntü veri seti.

    CSV beklenen sütunlar:
        - image_path : Görselin tam (absolute) dosya yolu.
        - label      : 0 (Benign) veya 1 (Malign).
    """

    def __init__(self, csv_path: str | Path, transform=None):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")

        self.df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV zorunlu sütunları eksik: {sorted(missing)} ({csv_path})"
            )

        # Etiketleri int64'e sabitle (CrossEntropyLoss için)
        self.df["label"] = self.df["label"].astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # Kolay tanı çıktısı
    def class_counts(self) -> Tuple[int, int]:
        n0 = int((self.df["label"] == 0).sum())
        n1 = int((self.df["label"] == 1).sum())
        return n0, n1


# =============================================================================
# 2. TRANSFORMS  (train_baseline.py ile BİREBİR aynı)
# =============================================================================
def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=18),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =============================================================================
# 3. MODEL  (ResNet50 + binary head)
# =============================================================================
def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    ResNet50 iskeletini kurar, son FC katmanını num_classes çıktılı yapar.
    Federe kurulumda ağırlıkları sunucu dağıtacağı için varsayılan
    pretrained=False'tur (sunucu tarafı ilk roundda kendi yapısıyla
    ilk ağırlıkları paylaşır).
    """
    weights = "DEFAULT" if pretrained else None
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# =============================================================================
# 4. PARAMETRE I/O  (PyTorch state_dict <-> NumPy list)
# =============================================================================
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Model state_dict'i Flower'ın istediği NumPy listesine çevirir."""
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Flower'dan gelen NumPy listesini model state_dict'e yükler."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)
