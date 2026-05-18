"""
Baseline (Referans) Model Eğitim Betiği
=======================================

Amaç:
-----
Cilt Kanseri (HAM10000) veri setini, DENGESİZ haliyle ikili (Binary)
sınıflandırmaya tabi tutmak ve baseline metrikleri elde etmektir.

ÖNEMLİ: Bu aşamada sınıf ağırlıklandırma (class weighting), oversampling
veya undersampling KULLANILMAZ. Baseline modelin Malign (kanser) sınıfını
kaçırma eğilimini (düşük Recall) kanıtlamak istiyoruz. Bu çıktı, DCGAN +
Federe Öğrenme aşamasındaki iyileşmeyi ölçebileceğimiz kıyas noktasıdır.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

# Proje kökünü bulup src/ klasörünü import path'ine ekle
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config                                          # noqa: E402
from data_loader import HAM10000Dataset                # noqa: E402
from federated_utils import build_transforms, build_model  # noqa: E402
from split_utils import lesion_level_split_indices  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    # Yollar (config modülünden)
    "metadata_csv":    config.METADATA_CSV,
    "images_dir":      config.IMAGES_DIR,
    "checkpoint_dir":  config.BASELINE_CKPT_DIR,
    "best_model_name": "baseline_best_model.pth",

    # Eğitim hiperparametreleri
    "num_classes":   config.NUM_CLASSES,
    "batch_size":    config.BATCH_SIZE,
    "epochs":        10,
    "learning_rate": 1e-4,

    # Veri bölümleme (config modülünden - drift'i önlemek için)
    "train_ratio": config.TRAIN_RATIO,
    "val_ratio":   config.VAL_RATIO,
    "test_ratio":  config.TEST_RATIO,
    "seed":        config.SPLIT_SEED,
}


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================
def set_global_seed(seed: int) -> None:
    """Tekrarlanabilirlik için tüm kütüphanelerde seed'i sabitler."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(dataset, cfg: dict):
    """Datasetti %80/%10/%10 böler ve DataLoader'ları üretir."""
    train_idx, val_idx, test_idx = lesion_level_split_indices(
        dataset.df, cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    train_size, val_size, test_size = len(train_ds), len(val_ds), len(test_ds)

    num_workers = config.default_num_workers()
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"[Split] Train: {train_size} | Val: {val_size} | Test: {test_size}")
    return train_loader, val_loader, test_loader


# =============================================================================
# 3. EĞİTİM / DOĞRULAMA DÖNGÜLERİ
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, epoch_idx, total_epochs):
    """Tek bir epoch için eğitim adımlarını yürütür (AMP destekli)."""
    model.train()
    running_loss = 0.0
    seen = 0

    pbar = tqdm(loader, desc=f"[Train {epoch_idx}/{total_epochs}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision ileri geçiş
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Ölçeklenmiş geri yayılım
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_n = labels.size(0)
        running_loss += loss.item() * batch_n
        seen += batch_n
        pbar.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}")

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "Val"):
    """Loss + Macro F1 + Malign Recall (pos_label=1) döndürür."""
    model.eval()
    running_loss = 0.0
    seen = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_n = labels.size(0)
        running_loss += loss.item() * batch_n
        seen += batch_n

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    avg_loss = running_loss / max(seen, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    # Proje amacının özü: Malign (1) sınıfı için Recall
    malign_recall = recall_score(
        all_labels, all_preds, pos_label=1, zero_division=0
    )
    return avg_loss, macro_f1, malign_recall


# =============================================================================
# 4. ANA AKIŞ
# =============================================================================
def main() -> None:
    cfg = CONFIG
    set_global_seed(cfg["seed"])

    # Checkpoint klasörü
    cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    best_model_path = cfg["checkpoint_dir"] / cfg["best_model_name"]

    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # Dataset (Binary mapping: data_loader.py içinde yapılıyor)
    transform_pipeline = build_transforms()
    dataset = HAM10000Dataset(
        csv_path=cfg["metadata_csv"],
        image_dir=cfg["images_dir"],
        transform=transform_pipeline,
    )

    # Global sınıf dağılımı (dengesizliği kanıt olarak terminale düş)
    n_benign = int((dataset.df["label"] == 0).sum())
    n_malign = int((dataset.df["label"] == 1).sum())
    print(f"[Dataset] Toplam: {len(dataset)} | Benign(0): {n_benign} "
          f"| Malign(1): {n_malign}  (DENGESİZ - kasıtlı)")

    # Split + DataLoader
    train_loader, val_loader, _test_loader = build_dataloaders(dataset, cfg)

    # Model, loss, optimizer, AMP scaler
    # Baseline ImageNet ön eğitimli ağırlıklarla başlar (federe taraf False).
    model = build_model(num_classes=cfg["num_classes"], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()                       # class_weights YOK
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Eğitim döngüsü
    best_val_f1 = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['epochs']} =====")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg["epochs"],
        )
        val_loss, val_macro_f1, val_malign_recall = evaluate(
            model, val_loader, criterion, device, desc=f"Val {epoch}"
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"TrainLoss: {train_loss:.4f} | "
            f"ValLoss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Recall(Malign=1): {val_malign_recall:.4f}"
        )

        # Best-model checkpoint (kriter: Macro F1)
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_malign_recall": val_malign_recall,
                    "val_loss": val_loss,
                    "config": {k: str(v) for k, v in cfg.items()},
                },
                best_model_path,
            )
            print(f"  -> Yeni en iyi model kaydedildi: {best_model_path} "
                  f"(Macro-F1={best_val_f1:.4f})")

    print("\n[Eğitim tamamlandı]")
    print(f"En iyi Val Macro-F1: {best_val_f1:.4f}")
    print(f"Model yolu: {best_model_path}")


if __name__ == "__main__":
    main()
