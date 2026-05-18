"""
Federe Öğrenme - Global Modelin Test Seti Değerlendirmesi
==========================================================

Son raundta birleştirilen global modeli
(`checkpoints/federated/federated_model_round_{N}.pth`) bizim en baştaki
"Orijinal %10 Test" setimiz üzerinde değerlendirir ve `evaluate.py` ile
tamamen aynı raporları üretir:

    - classification_report (Macro-F1 + Malign Recall)
    - Confusion Matrix (terminal + seaborn PNG)

ÇOK KRİTİK - LEAKAGE YOK:
-------------------------
Bölümleme burada da `train_baseline.py` / `evaluate.py` ile BİREBİR
aynı (seed=42, 80/10/10). Test setine sentetik veya federe eğitim
verisi kesinlikle karışmaz; yalnızca 3. parça olan `test_ds` kullanılır.

Kullanım:
---------
    python src/evaluate_federated.py
    python src/evaluate_federated.py --round 5   # belirli bir raund
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# src/ klasörünü import path'ine ekle
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from data_loader import HAM10000Dataset, SquarePad  # noqa: E402
from federated_utils import build_model                # noqa: E402
from split_utils import lesion_level_split_indices     # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON  (train_baseline.py / evaluate.py ile tutarlı)
# =============================================================================
CONFIG = {
    # Yollar
    "metadata_csv":   ROOT_DIR / "data" / "raw" / "HAM10000_metadata.csv",
    "images_dir":     ROOT_DIR / "data" / "raw",
    "checkpoint_dir": ROOT_DIR / "checkpoints" / "federated",
    "ckpt_prefix":    "federated_model_round",
    "cm_plot_name":   "federated_confusion_matrix.png",

    # Sınıflandırma
    "num_classes": 2,
    "class_names": ["Benign (0)", "Malign (1)"],

    # Data loading
    "batch_size": 32,

    # Bölümleme (BİREBİR train_baseline.py ile aynı)
    "train_ratio": 0.8,
    "val_ratio":   0.1,
    "test_ratio":  0.1,
    "seed": 42,

    # Varsayılan raund: en büyük round numarasını otomatik bul
    "default_round": None,
}


# =============================================================================
# 2. YARDIMCI FONKSİYONLAR
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


def build_test_loader(dataset, cfg: dict) -> DataLoader:
    train_idx, val_idx, test_idx = lesion_level_split_indices(
        dataset.df, cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"]
    )
    _train_ds = Subset(dataset, train_idx)
    _val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    # Allow overriding worker count for restricted environments.
    env_workers = os.getenv("EVAL_NUM_WORKERS")
    if env_workers is not None:
        num_workers = int(env_workers)
    else:
        num_workers = 0 if os.name == "nt" else 4
    return DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def find_checkpoint(cfg: dict, round_arg: int | None) -> Path:
    """Kullanıcı --round verdiyse onu; vermediyse en büyük round dosyasını seçer."""
    ckpt_dir = cfg["checkpoint_dir"]
    prefix = cfg["ckpt_prefix"]
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Federe checkpoint klasörü yok: {ckpt_dir}\n"
            f"Önce `python src/server.py` + 5 istemci ile eğitim koşturun."
        )

    if round_arg is not None:
        candidate = ckpt_dir / f"{prefix}_{round_arg}.pth"
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint bulunamadı: {candidate}")
        return candidate

    # Aksi hâlde en büyük round numarasını bul
    matches = []
    for p in ckpt_dir.glob(f"{prefix}_*.pth"):
        try:
            r = int(p.stem.rsplit("_", 1)[1])
            matches.append((r, p))
        except ValueError:
            continue
    if not matches:
        raise FileNotFoundError(
            f"{ckpt_dir} içinde {prefix}_*.pth bulunamadı."
        )
    matches.sort()
    return matches[-1][1]


def load_model(ckpt_path: Path, device: torch.device, num_classes: int):
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict, meta = checkpoint, {}
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


# =============================================================================
# 3. ÇIKARIM VE RAPOR
# =============================================================================
@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="[Federated Test]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    return y_true, y_pred


def print_classification_report(y_true, y_pred, class_names):
    print("\n" + "=" * 72)
    print("FEDERATED MODEL - CLASSIFICATION REPORT  (Test Seti)")
    print("=" * 72)
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=class_names,
        digits=4, zero_division=0,
    ))


def print_confusion_matrix(cm, class_names):
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
        row += "".join(
            f"{int(cm[i, j]):>{col_width}}" for j in range(len(class_names))
        )
        print(row)
    print()


def save_confusion_matrix_plot(cm, class_names, output_path: Path,
                               title_suffix: str = ""):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        cbar=True, square=True, linewidths=0.5, linecolor="white",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(
        f"Federated Global Model - Confusion Matrix{title_suffix}",
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
# 4. ANA AKIŞ
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federe global modeli orijinal test setinde değerlendir."
    )
    parser.add_argument("--round", type=int, default=None,
                        help="Değerlendirilecek raund numarası "
                             "(belirtilmezse en büyük raund).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Checkpoint seç
    ckpt_path = find_checkpoint(cfg, args.round)
    print(f"[Ckpt]   Kullanılacak: {ckpt_path}")

    # Test seti (leakage-proof)
    dataset = HAM10000Dataset(
        csv_path=cfg["metadata_csv"],
        image_dir=cfg["images_dir"],
        transform=build_transforms(),
    )
    test_loader = build_test_loader(dataset, cfg)
    print(f"[Data]   Test örnek sayısı: {len(test_loader.dataset)}")

    # Model
    model, meta = load_model(ckpt_path, device, cfg["num_classes"])
    if meta:
        print(f"[Meta]   {meta}")

    # Çıkarım + raporlar
    y_true, y_pred = run_inference(model, test_loader, device)
    print_classification_report(y_true, y_pred, cfg["class_names"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print_confusion_matrix(cm, cfg["class_names"])

    round_suffix = f"  (Round {meta.get('round', '?')})" if meta else ""
    plot_path = cfg["checkpoint_dir"] / cfg["cm_plot_name"]
    save_confusion_matrix_plot(cm, cfg["class_names"], plot_path,
                               title_suffix=round_suffix)

    # Proje odağı: Malign Recall
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        malign_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print("-" * 72)
        print("ÖZET (Proje Odak Metrikleri)")
        print(f"  Malign (1) Recall : {malign_recall:.4f}  "
              f"[TP={tp}, FN={fn}]  <-- KAÇIRILAN KANSER: {fn}")
        print(f"  Benign (0) TN={tn}  |  FP={fp}")
        print("-" * 72)
        print("Not: Bu sonucu `baseline_best_model.pth` ile karşılaştır "
              "(evaluate.py çıktısı). Proje hipotezi: DCGAN + FL ile "
              "Malign Recall yükselir, kaçırılan kanser azalır.")


if __name__ == "__main__":
    main()
