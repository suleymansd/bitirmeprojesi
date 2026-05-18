"""
Proje Geneli Sabitler ve Yollar
================================

Tüm script'lerin (baseline, GAN, federe) paylaştığı tek kaynak.
Seed, split oranları, istemci sayısı, path'ler gibi değerlerin
dosyadan dosyaya sapmasını (drift) önler.

Kural: Aşağıdaki değerleri yalnızca burada değiştir; diğer dosyalar
bu modülden import etmeli.
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# 1. YOLLAR
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR               = ROOT_DIR / "data"
METADATA_CSV           = DATA_DIR / "raw" / "HAM10000_metadata.csv"
IMAGES_DIR             = DATA_DIR / "raw"
SYNTHETIC_MALIGN_DIR   = DATA_DIR / "synthetic" / "generated_malign"
SYNTHETIC_SAMPLES_DIR  = DATA_DIR / "synthetic" / "samples"
FEDERATED_SPLITS_DIR   = DATA_DIR / "federated_splits"

CHECKPOINT_DIR         = ROOT_DIR / "checkpoints"
BASELINE_CKPT_DIR      = CHECKPOINT_DIR / "baseline"
FEDERATED_CKPT_DIR     = CHECKPOINT_DIR / "federated"
GAN_CKPT_DIR           = CHECKPOINT_DIR / "gan"


# =============================================================================
# 2. SINIFLANDIRMA
# =============================================================================
NUM_CLASSES = 2
CLASS_NAMES = ["Benign (0)", "Malign (1)"]


# =============================================================================
# 3. VERİ BÖLÜMLEME  (train_baseline / evaluate / train_gan / federe - HEPSİ AYNI)
# =============================================================================
SPLIT_SEED   = 42
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1
TEST_RATIO   = 0.1


# =============================================================================
# 4. EĞİTİM
# =============================================================================
BATCH_SIZE = 32


# =============================================================================
# 5. FEDERE ÖĞRENME
# =============================================================================
# Hastane istemci sayısı. 5 -> 2 geçişi için buradan değiştirilir;
# prepare_federated_splits.py ve server.py bu değeri okur.
NUM_CLIENTS = 2
NUM_ROUNDS = 20  # Tez final tuning: daha uzun federated yakınsama


# =============================================================================
# 6. DATALOADER
# =============================================================================
def default_num_workers() -> int:
    """
    Windows'ta 0 (DataLoader multiprocessing çakışmaları), Unix/macOS'ta 4.
    client.py / baseline / evaluate bu yardımcıyı kullanmalı.
    """
    return 0 if os.name == "nt" else 4
