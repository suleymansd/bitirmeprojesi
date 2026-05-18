"""
Baseline Model - Test Seti Değerlendirme Betiği
================================================

Amaç:
-----
`train_baseline.py` ile eğitilen ve
`checkpoints/baseline/baseline_best_model.pth` yoluna kaydedilen en iyi
baseline modelin, daha önce HİÇ görmediği test seti üzerindeki gerçek
performansını ölçmek.

Çıktılar:
---------
1) Terminale `classification_report` (Malign=1 Recall/F1 vurgulu).
2) Terminale Confusion Matrix (ham sayısal).
3) `checkpoints/baseline/baseline_confusion_matrix.png` dosyası (seaborn).

ÇOK KRİTİK:
-----------
Data leakage olmaması için veri bölümleme, `train_baseline.py` içindekiyle
BİREBİR aynıdır:
    - Oranlar: %80 train, %10 val, %10 test
    - Seed:    42 (torch.Generator().manual_seed(42))
    - Sıra:    random_split(dataset, [train_size, val_size, test_size], gen)
Bu dosyada yalnızca test_loader kullanılır.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")          # Başsız (headless) çizim - ekrana açma
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Proje kökünü ekle ve data_loader'ı içe aktar
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config                                          # noqa: E402
from data_loader import HAM10000Dataset                # noqa: E402
from federated_utils import build_transforms, build_model  # noqa: E402
from split_utils import lesion_level_split_indices  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON  (train_baseline.py ile TUTARLI)
# =============================================================================
CONFIG = {
    # Yollar (config modülünden)
    "metadata_csv":    config.METADATA_CSV,
    "images_dir":      config.IMAGES_DIR,
    "checkpoint_dir":  config.BASELINE_CKPT_DIR,
    "best_model_name": "baseline_best_model.pth",
    "cm_plot_name":    "baseline_confusion_matrix.png",

    # Sınıflandırma
    "num_classes": config.NUM_CLASSES,
    "class_names": config.CLASS_NAMES,

    # Data loading
    "batch_size": config.BATCH_SIZE,

    # Bölümleme (config modülünden - train ile drift olmaz)
    "train_ratio": config.TRAIN_RATIO,
    "val_ratio":   config.VAL_RATIO,
    "test_ratio":  config.TEST_RATIO,
    "seed":        config.SPLIT_SEED,
}


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
# =============================================================================
def build_test_loader(dataset, cfg: dict) -> DataLoader:
    """
    Dataset'i TRAIN betiğindeki ile TAMAMEN AYNI şekilde böler
    (aynı seed, aynı oranlar, aynı sıra) ve yalnızca test kısmını döndürür.
    """
    train_idx, val_idx, test_idx = lesion_level_split_indices(
        dataset.df, cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
    )
    _train_ds = Subset(dataset, train_idx)
    _val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    train_size, val_size, test_size = len(_train_ds), len(_val_ds), len(test_ds)

    env_workers = os.getenv("EVAL_NUM_WORKERS")
    num_workers = int(env_workers) if env_workers is not None else config.default_num_workers()
    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Split] Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print(f"[Info ] Sadece test_loader kullanılacak (n={len(test_ds)}).")
    return test_loader


def load_checkpoint(model: nn.Module, ckpt_path: Path,
                    device: torch.device) -> dict:
    """Checkpoint dosyasından yalnızca model_state_dict'i modele yükler."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {ckpt_path}\n"
            f"Önce `python src/train_baseline.py` ile modeli eğitmelisiniz."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Betiğimiz hem dict tipi (bizim kaydettiğimiz) hem de düz state_dict ile
    # çalışabilsin.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        checkpoint = {}

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


# =============================================================================
# 3. ÇIKARIM (INFERENCE)
# =============================================================================
@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device):
    """Tüm test seti üzerinde tahminleri toplar."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="[Test]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    return y_true, y_pred


# =============================================================================
# 4. RAPOR VE GÖRSEL ÇIKTI
# =============================================================================
def print_classification_report(y_true, y_pred, class_names):
    """Classification report'u net başlıklarla terminale basar."""
    print("\n" + "=" * 72)
    print("CLASSIFICATION REPORT  (Test Seti)")
    print("=" * 72)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)


def print_confusion_matrix(cm: np.ndarray, class_names):
    """Ham confusion matrix'i okunaklı biçimde terminale basar."""
    print("=" * 72)
    print("CONFUSION MATRIX  (satır=Gerçek, sütun=Tahmin)")
    print("=" * 72)

    col_width = max(12, max(len(n) for n in class_names) + 2)
    header = " " * col_width + "".join(
        f"{f'Pred {n}':>{col_width}}" for n in class_names
    )
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"{'True ' + row_name:<{col_width}}"
        row += "".join(f"{int(cm[i, j]):>{col_width}}" for j in range(len(class_names)))
        print(row)
    print()


def save_confusion_matrix_plot(cm: np.ndarray, class_names, output_path: Path):
    """Akademik görünüşlü bir seaborn heatmap üretir ve diske kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )

    ax.set_title(
        "Baseline Model - Confusion Matrix (Test Seti)",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Tahmin Edilen Değerler", fontsize=11, labelpad=8)
    ax.set_ylabel("Gerçek Değerler", fontsize=11, labelpad=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot] Confusion matrix kaydedildi: {output_path}")


# =============================================================================
# 5. ANA AKIŞ
# =============================================================================
def main() -> None:
    cfg = CONFIG

    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # Dataset (binary label mapping data_loader.py içinde yapılıyor)
    transform_pipeline = build_transforms()
    dataset = HAM10000Dataset(
        csv_path=cfg["metadata_csv"],
        image_dir=cfg["images_dir"],
        transform=transform_pipeline,
    )
    print(f"[Dataset] Toplam örnek: {len(dataset)}")

    # Test loader (train ve val kullanılmaz)
    test_loader = build_test_loader(dataset, cfg)

    # Modeli kur ve checkpoint'i yükle (ağırlıklar checkpoint'ten gelecek,
    # ImageNet pretraining gerekmez)
    model = build_model(num_classes=cfg["num_classes"], pretrained=False).to(device)
    ckpt_path = cfg["checkpoint_dir"] / cfg["best_model_name"]
    ckpt_meta = load_checkpoint(model, ckpt_path, device)

    if ckpt_meta:
        print(
            f"[Ckpt]  Yüklendi: {ckpt_path} "
            f"(epoch={ckpt_meta.get('epoch', 'N/A')}, "
            f"val_macro_f1={ckpt_meta.get('val_macro_f1', 'N/A')})"
        )
    else:
        print(f"[Ckpt]  Yüklendi: {ckpt_path} (meta yok - düz state_dict)")

    # Çıkarım
    y_true, y_pred = run_inference(model, test_loader, device)

    # Raporlar
    print_classification_report(y_true, y_pred, cfg["class_names"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print_confusion_matrix(cm, cfg["class_names"])

    # Görsel
    plot_path = cfg["checkpoint_dir"] / cfg["cm_plot_name"]
    save_confusion_matrix_plot(cm, cfg["class_names"], plot_path)

    # Proje odağının hızlı özeti
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        malign_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print("-" * 72)
        print("ÖZET (Proje Odak Metrikleri)")
        print(f"  Malign (1) Recall : {malign_recall:.4f}  "
              f"[TP={tp}, FN={fn}]  <-- KAÇIRILAN KANSER VAKA SAYISI: {fn}")
        print(f"  Benign (0) True Neg: {tn}  |  False Pos: {fp}")
        print("-" * 72)


if __name__ == "__main__":
    main()
