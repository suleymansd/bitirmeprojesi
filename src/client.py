"""
Flower Federe İstemci (Hastane Simülasyonu)
=============================================

Bu betik, bir hastane istemcisini temsil eder. `client_{i}.csv` dosyasını
yükler, global model ağırlıklarını sunucudan alır, 1 epoch yerel eğitim
yapar ve güncellenmiş ağırlıkları sunucuya geri döndürür.

Kullanım:
---------
    python src/client.py --client_id 1
    python src/client.py --client_id 2

Tüm istemciler ayrı terminallerde çalıştırıldığında, `src/server.py`
FedAvg stratejisiyle ağırlıkları birleştirir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import flwr as fl

# src/ klasörünü import path'ine ekle
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from federated_utils import (   # noqa: E402
    FedClientDataset,
    build_train_transforms,
    build_model,
    get_parameters,
    set_parameters,
)
import config  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON (varsayılanlar)
# =============================================================================
DEFAULTS = {
    "federated_dir": config.FEDERATED_SPLITS_DIR,
    "server_address": "127.0.0.1:8080",
    "batch_size": config.BATCH_SIZE,
    "local_epochs": 2,       # Tez final tuning: her round daha güçlü yerel öğrenme
    "learning_rate": 1e-4,
    "num_workers": 0,
    "num_classes": config.NUM_CLASSES,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "focal_gamma": 1.5,
}


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# =============================================================================
# 2. EĞİTİM / DEĞERLENDİRME YARDIMCILARI
# =============================================================================
def _make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    num_workers = DEFAULTS["num_workers"]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                     max_grad_norm: float):
    model.train()
    running_loss, seen = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()

        # AMP ile ölçeklenen gradyanları gerçek ölçeğe indirip clip uygula.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        bs = labels.size(0)
        running_loss += loss.item() * bs
        seen += bs
    return running_loss / max(seen, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, seen, correct = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        seen += bs
    avg_loss = running_loss / max(seen, 1)
    acc = correct / max(seen, 1)
    return avg_loss, acc


# =============================================================================
# 3. FLOWER İSTEMCİSİ
# =============================================================================
class CiltKanseriClient(fl.client.NumPyClient):
    """HAM10000 + sentetik veriyle yerel eğitim yapan istemci."""

    def __init__(self, client_id: int, csv_path: Path, device: torch.device,
                 batch_size: int, local_epochs: int, lr: float):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

        # Veri
        transform = build_train_transforms()
        self.dataset = FedClientDataset(csv_path, transform=transform)
        self.loader = _make_loader(self.dataset, batch_size, shuffle=True)
        n_b, n_m = self.dataset.class_counts()
        print(f"[Client {client_id}] CSV: {csv_path.name} | "
              f"Örnek: {len(self.dataset)} | Benign: {n_b} | Malign: {n_m}")

        # Model (yapı) - ağırlıklar server'dan gelecek
        self.model = build_model(num_classes=DEFAULTS["num_classes"],
                                 pretrained=False).to(device)

        # Class-weighted loss (sklearn "balanced" formülü):
        # w_c = total / (num_classes * n_c)
        if n_b <= 0 or n_m <= 0:
            raise ValueError(
                f"[Client {client_id}] Sınıf dağılımı hatalı: "
                f"Benign={n_b}, Malign={n_m}."
            )
        total = n_b + n_m
        w = torch.tensor(
            [total / (2 * n_b), total / (2 * n_m)],
            dtype=torch.float32,
            device=device,
        )
        loss_mode = os.getenv("LOSS_MODE", "focal").strip().lower()
        if loss_mode == "ce":
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = WeightedFocalLoss(
                weight=w,
                gamma=DEFAULTS["focal_gamma"],
            )
        print(f"[Client {client_id}] Class weights: "
              f"w0={w[0].item():.4f}, w1={w[1].item():.4f} | "
              f"loss={loss_mode}")

    # --- Flower API ---
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict
            ) -> Tuple[List[np.ndarray], int, Dict]:
        # 1) Global parametreleri yerel modele yükle
        set_parameters(self.model, parameters)

        # 2) Yerel eğitim (optimizer her fit çağrısında sıfırdan kurulur -
        #    Flower'da klasik pratik budur).
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=DEFAULTS["weight_decay"],
        )
        scaler = GradScaler(enabled=(self.device.type == "cuda"))

        last_loss = 0.0
        for _ in range(self.local_epochs):
            last_loss = _train_one_epoch(
                self.model, self.loader, self.criterion,
                optimizer, scaler, self.device,
                max_grad_norm=DEFAULTS["max_grad_norm"],
            )

        print(f"[Client {self.client_id}] fit() bitti. Loss={last_loss:.4f} "
              f"| N={len(self.dataset)}")

        return (
            get_parameters(self.model),
            len(self.dataset),
            {"train_loss": float(last_loss)},
        )

    def evaluate(self, parameters: List[np.ndarray], config: Dict
                 ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        loss, acc = _evaluate(self.model, self.loader, self.criterion,
                              self.device)
        print(f"[Client {self.client_id}] evaluate() Loss={loss:.4f} "
              f"| Acc={acc:.4f}")
        return float(loss), len(self.dataset), {"accuracy": float(acc)}


# =============================================================================
# 4. GİRİŞ NOKTASI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower federe istemcisi")
    parser.add_argument("--client_id", type=int, required=True,
                        help="İstemci kimliği (1..2)")
    parser.add_argument("--server_address", type=str,
                        default=DEFAULTS["server_address"])
    parser.add_argument("--batch_size", type=int,
                        default=DEFAULTS["batch_size"])
    parser.add_argument("--local_epochs", type=int,
                        default=DEFAULTS["local_epochs"])
    parser.add_argument("--lr", type=float,
                        default=DEFAULTS["learning_rate"])
    parser.add_argument("--federated_dir", type=str,
                        default=str(DEFAULTS["federated_dir"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.federated_dir) / f"client_{args.client_id}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client {args.client_id}] Device: {device}")
    if device.type == "cuda":
        print(f"[Client {args.client_id}] GPU: "
              f"{torch.cuda.get_device_name(0)}")

    client = CiltKanseriClient(
        client_id=args.client_id,
        csv_path=csv_path,
        device=device,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )

    # NumPyClient'ı Flower'ın yeni API'sine çevir
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
