"""
Federe Öğrenme için İstemci Veri Setlerinin Hazırlanması
=========================================================

Amaç:
-----
1) `train_baseline.py` / `evaluate.py` / `train_gan.py` ile BİREBİR aynı
   bölümlemeyi uygulayarak orijinal HAM10000 TRAIN parçasını çıkarmak
   (val ve test'e asla dokunmadan).
2) DCGAN ile üretilen sentetik Malign görüntüleri (label=1) bu TRAIN
   listesine ekleyerek sınıf dengesini sağlamak.
3) Dengeli birleşik listeyi config'teki istemci sayısına göre (hastane
   simülasyonu) eşit ve IID
   olacak şekilde dağıtıp `data/federated_splits/client_{i}.csv` olarak
   kaydetmek.
4) Her istemci için Benign/Malign dağılımını terminale tablo halinde
   yazdırarak bölümlemenin doğruluğunu kanıtlamak.

Kullanım:
---------
    python src/prepare_federated_splits.py
"""

import os
import random
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import torch

# Proje kökünü ekleyip data_loader'ı içe aktar
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config  # noqa: E402
from data_loader import HAM10000Dataset  # noqa: E402
from split_utils import lesion_level_split_indices  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    # Yollar
    "metadata_csv":    config.METADATA_CSV,
    "images_dir":      config.IMAGES_DIR,
    "synthetic_dir":   config.SYNTHETIC_MALIGN_DIR,
    "federated_dir":   config.FEDERATED_SPLITS_DIR,

    # Bölümleme (train_baseline.py ile BİREBİR aynı)
    "train_ratio": config.TRAIN_RATIO,
    "val_ratio":   config.VAL_RATIO,
    "test_ratio":  config.TEST_RATIO,
    "split_seed":  config.SPLIT_SEED,

    # Karıştırma (shuffle) ve istemci sayısı
    "shuffle_seed": config.SPLIT_SEED,
    "num_clients":  config.NUM_CLIENTS,

    # Sentetik dengeleme
    "max_synth_cap": 1000,
    "target_malign_ratio_max": 0.45,

    # Sentetik görüntü uzantıları
    "synthetic_patterns": ("*.jpg", "*.jpeg", "*.png"),
}


# =============================================================================
# 2. ORİJİNAL TRAIN (Leakage korumalı) LİSTESİNİ ÇIKARMA
# =============================================================================
def extract_original_train_records(cfg: dict):
    """
    train_baseline.py ile aynı seed+oran+sıra kullanarak yalnızca TRAIN
    parçasındaki fotoğrafların (absolute_path, label) listesini döndürür.
    Val ve Test örneklerine erişilmez.
    """
    # HAM10000Dataset, image_id -> tam path eşlemesini ve binary label'ı
    # zaten kendi içinde yapıyor. Transform gerekmiyor.
    dataset = HAM10000Dataset(
        csv_path=cfg["metadata_csv"],
        image_dir=cfg["images_dir"],
        transform=None,
    )

    train_idx, val_idx, test_idx = lesion_level_split_indices(
        dataset.df,
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg["test_ratio"],
        cfg["split_seed"],
    )

    df = dataset.df
    records = []
    for idx in train_idx:
        abs_path = str(Path(df.loc[idx, "path"]).resolve())
        label = int(df.loc[idx, "label"])
        records.append((abs_path, label))

    print(f"[Split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"[Info ] Val/Test dokunulmadı. Yalnızca TRAIN listesi alındı.")
    return records


# =============================================================================
# 3. SENTETİK MALİGN LİSTESİNİ ÇIKARMA
# =============================================================================
def extract_synthetic_records(cfg: dict):
    """
    data/synthetic/generated_malign/ altındaki tüm görüntülerin
    absolute path'lerini toplar ve label=1 atar.
    """
    synth_dir = cfg["synthetic_dir"]
    if not synth_dir.is_dir():
        raise FileNotFoundError(
            f"Sentetik veri klasörü bulunamadı: {synth_dir}\n"
            f"Önce `python src/generate_synthetic.py` ile üretimi yapın."
        )

    paths = []
    for pattern in cfg["synthetic_patterns"]:
        paths.extend(glob(str(synth_dir / pattern)))
    paths = sorted(set(paths))   # deterministik sıra

    if not paths:
        raise RuntimeError(f"{synth_dir} içinde görsel bulunamadı.")

    records = [(str(Path(p).resolve()), 1) for p in paths]
    return records


def cap_synthetic_records(
    synthetic_records,
    n_orig_benign: int,
    n_orig_malign: int,
    max_synth_cap: int,
):
    """
    Sentetik Malign miktarını aşırı enjeksiyonu önleyecek şekilde kısıtlar.

    Kural:
      allowed = min(n_benign - n_malign, max_synth_cap)

    Not: Liste zaten sorted() ile deterministik sırada üretildiği için,
    ilk N kaydı almak deterministik davranış sağlar.
    """
    gap = max(n_orig_benign - n_orig_malign, 0)
    allowed = min(gap, max_synth_cap)
    capped = synthetic_records[:allowed]
    return capped, allowed, gap


# =============================================================================
# 4. İSTEMCİLERE DAĞITIM (IID) VE CSV YAZIMI
# =============================================================================
def split_into_clients(records, num_clients: int, shuffle_seed: int):
    """
    Listeyi sabit seed ile karıştırıp num_clients eşit parçaya böler.
    Küçük kalan (mod) örnekler, ilk istemcilere birer birer dağıtılır.
    """
    rng = random.Random(shuffle_seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    base = total // num_clients
    extras = total - base * num_clients   # 0..num_clients-1

    client_splits = []
    cursor = 0
    for c in range(num_clients):
        size = base + (1 if c < extras else 0)
        client_splits.append(shuffled[cursor: cursor + size])
        cursor += size
    return client_splits


def save_client_csvs(client_splits, output_dir: Path):
    """Her bir istemci için iki sütunlu CSV (image_path, label) yazar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, records in enumerate(client_splits, start=1):
        df = pd.DataFrame(records, columns=["image_path", "label"])
        out_path = output_dir / f"client_{i}.csv"
        df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


# =============================================================================
# 5. RAPOR
# =============================================================================
def print_distribution_table(client_splits, total_before, total_after,
                             n_orig, n_synth):
    """Her istemciye kaç Benign(0) / Malign(1) düştüğünü tablo olarak bas."""
    print()
    print("=" * 72)
    print("Federe Dağıtım Raporu")
    print("=" * 72)
    print(f"Orijinal Train    : {n_orig}")
    print(f"Sentetik Malign   : {n_synth}")
    print(f"Birleşik Toplam   : {total_before}  (dengeleme öncesi karşılaştırma)")
    print(f"Dağıtılan Toplam  : {total_after}")
    print("-" * 72)
    header = f"{'Client':<10}{'Benign(0)':>12}{'Malign(1)':>12}{'Total':>10}{'%Malign':>10}"
    print(header)
    print("-" * 72)

    sum_b = sum_m = sum_t = 0
    for i, records in enumerate(client_splits, start=1):
        n_b = sum(1 for _, y in records if y == 0)
        n_m = sum(1 for _, y in records if y == 1)
        n_t = n_b + n_m
        pct = (n_m / n_t * 100.0) if n_t > 0 else 0.0
        print(f"client_{i:<3}{n_b:>12}{n_m:>12}{n_t:>10}{pct:>9.2f}%")
        sum_b += n_b
        sum_m += n_m
        sum_t += n_t

    print("-" * 72)
    overall_pct = (sum_m / sum_t * 100.0) if sum_t > 0 else 0.0
    print(f"{'TOTAL':<10}{sum_b:>12}{sum_m:>12}{sum_t:>10}{overall_pct:>9.2f}%")
    print("=" * 72)


# =============================================================================
# 6. ANA AKIŞ
# =============================================================================
def main() -> None:
    cfg = CONFIG

    # 1) Orijinal TRAIN (val/test'e DOKUNULMAZ)
    original_records = extract_original_train_records(cfg)
    n_orig_benign = sum(1 for _, y in original_records if y == 0)
    n_orig_malign = sum(1 for _, y in original_records if y == 1)
    print(f"[Orig ] Benign(0)={n_orig_benign} | Malign(1)={n_orig_malign} "
          f"| Total={len(original_records)}")

    # 2) Sentetik Malign (tamamı label=1)
    synthetic_all_records = extract_synthetic_records(cfg)
    synthetic_records, allowed, gap = cap_synthetic_records(
        synthetic_all_records,
        n_orig_benign=n_orig_benign,
        n_orig_malign=n_orig_malign,
        max_synth_cap=cfg["max_synth_cap"],
    )
    print(f"[Synth] Diskte bulunan sentetik: {len(synthetic_all_records)}")
    print(f"[Synth] Benign-Malign farkı: {gap} | Üst sınır: {cfg['max_synth_cap']} "
          f"| Kullanılan: {allowed}")

    # 3) Birleşik balanced liste
    merged = original_records + synthetic_records
    print(f"[Merge] Birleşik toplam: {len(merged)}  "
          f"(Benign={n_orig_benign}, "
          f"Malign={n_orig_malign + len(synthetic_records)})")

    merged_malign_ratio = (
        (n_orig_malign + len(synthetic_records)) / len(merged)
        if len(merged) > 0 else 0.0
    )
    print(f"[Merge] Toplam Malign oranı: {merged_malign_ratio * 100:.2f}%")
    if merged_malign_ratio > cfg["target_malign_ratio_max"]:
        print(
            "[Warn ] Malign oranı hedef eşiği aştı "
            f"(>{cfg['target_malign_ratio_max'] * 100:.0f}%). "
            "Bir sonraki denemede max_synth_cap düşürülebilir."
        )

    # 4) IID şekilde istemcilere dağıt
    client_splits = split_into_clients(
        merged,
        num_clients=cfg["num_clients"],
        shuffle_seed=cfg["shuffle_seed"],
    )

    # 5) CSV olarak yaz
    written = save_client_csvs(client_splits, cfg["federated_dir"])
    print(f"[Write] {len(written)} adet CSV yazıldı -> {cfg['federated_dir']}")
    for p in written:
        # Dosya boyutunu da basalım
        size_kb = os.path.getsize(p) / 1024
        print(f"   - {p.name}  ({size_kb:.1f} KB)")

    # 6) Dağılım raporu
    print_distribution_table(
        client_splits,
        total_before=len(merged),
        total_after=sum(len(c) for c in client_splits),
        n_orig=len(original_records),
        n_synth=len(synthetic_records),
    )


if __name__ == "__main__":
    main()
