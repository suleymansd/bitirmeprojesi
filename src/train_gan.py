"""
DCGAN Eğitim Betiği - Malign (Kanser) Sentetik Görüntü Üretimi
===============================================================

Amaç:
-----
HAM10000 veri setindeki azınlık sınıfı olan "Malign" (mel, bcc, akiec)
lezyon görüntülerini çoğaltmak için bir DCGAN (Deep Convolutional GAN)
eğitmek. Üretilen sentetik görüntüler, federe öğrenme aşamasında azınlık
sınıfını dengelemek için istemcilere dağıtılacaktır.

ÇOK KRİTİK - VERİ İZOLASYONU:
-----------------------------
- Bölümleme `train_baseline.py` ve `evaluate.py` ile BİREBİR aynıdır.
  (%80 Train / %10 Val / %10 Test, seed=42)
- GAN YALNIZCA train_dataset içindeki label==1 (Malign) örneklerini görür.
- Val ve Test setleri, GAN'a ASLA sızdırılmaz. Böylece sentetik verinin
  bilgi sızıntısına yol açması engellenir.

Referans: Radford et al., "Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks" (DCGAN).
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm

# Proje kökünü ekle ve data_loader'ı içe aktar
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from data_loader import HAM10000Dataset, SquarePad  # noqa: E402


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    # Yollar
    "metadata_csv":   ROOT_DIR / "data" / "raw" / "HAM10000_metadata.csv",
    "images_dir":     ROOT_DIR / "data" / "raw",
    "samples_dir":    ROOT_DIR / "data" / "synthetic" / "samples",
    "checkpoint_dir": ROOT_DIR / "checkpoints" / "gan",
    "generator_name": "generator_final.pth",

    # Bölümleme (train_baseline.py ile BİREBİR aynı olmak zorunda)
    "train_ratio": 0.8,
    "val_ratio":   0.1,
    "test_ratio":  0.1,
    "seed": 42,

    # Görüntü / Dataloader
    "image_size": 128,     # DCGAN için 128x128 (klasik 64x64'ten daha iyi doku)
    "batch_size": 64,
    "num_workers": 8,      # GPU'yu aç kalmasın diye bolca worker

    # Ağ hiperparametreleri
    "nc":  3,              # Kanal sayısı (RGB)
    "nz":  100,            # Gizli vektör (latent) boyutu
    "ngf": 64,             # Generator base feature map sayısı
    "ndf": 64,             # Discriminator base feature map sayısı

    # Eğitim
    "epochs": 100,
    # TTUR (Two Time-Scale Update Rule): D'yi yavaşlatıp G'ye nefes aldırıyoruz.
    # Bu, "Discriminator Overpowering" ve LossD->0 çökmesini engeller.
    "lr_D": 1e-4,
    "lr_G": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,

    # İzleme
    "sample_every": 10,    # Her N epoch'ta bir örnek kaydet
    "num_fixed_samples": 16,
    # One-sided label smoothing: Discriminator aşırı özgüvenden kaçınır,
    # gradyanlar sıfıra sıkışmaz. Sadece GERÇEK etiketler yumuşatılır.
    "real_label": 0.9,
    "fake_label": 0.0,
}


# =============================================================================
# 2. AĞIRLIK İLKLEME (DCGAN orijinal makalesindeki şekilde)
# =============================================================================
def weights_init(m: nn.Module) -> None:
    """Conv katmanları N(0, 0.02), BatchNorm katmanları N(1, 0.02) ile ilklenir."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


# =============================================================================
# 3. MODELLER (128x128 DCGAN)
# =============================================================================
class Generator(nn.Module):
    """
    Latent vector z (nz x 1 x 1) -> 128x128 RGB görüntü.

    Upsampling zinciri (stride=2 ConvTranspose2d):
        1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    Son aktivasyon: Tanh  => çıktılar [-1, 1] aralığında.
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # Giriş: (nz) x 1 x 1  ->  (ngf*16) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 16, kernel_size=4, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True),

            # (ngf*16) x 4 x 4 -> (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # (ngf*8) x 8 x 8 -> (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # (ngf*4) x 16 x 16 -> (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # (ngf*2) x 32 x 32 -> (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # (ngf) x 64 x 64 -> (nc) x 128 x 128
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """
    128x128 RGB görüntü -> gerçek/sahte skalar olasılık.

    Downsampling zinciri (stride=2 Conv2d):
        128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
    Son aktivasyon: Sigmoid  => [0, 1] olasılık.

    STABİLİZASYON:
    - Orta bloklara LeakyReLU sonrası Dropout(0.3) eklenmiştir. Bu,
      Discriminator'ın aşırı güçlenip LossD'yi 0'a çekmesini engeller
      (Generator'a sağlıklı gradyan akışı sağlar). Dropout sadece
      Discriminator'a eklenir, Generator'a DEĞİL.
    """

    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 128 x 128 -> (ndf) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 64 x 64 -> (ndf*2) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (ndf*2) x 32 x 32 -> (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (ndf*4) x 16 x 16 -> (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # (ndf*8) x 8 x 8 -> (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*16) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 1, 1) -> (B,)
        return self.main(x).view(-1)


# =============================================================================
# 4. VERİ HAZIRLIĞI (Split + Malign filtreleme)
# =============================================================================
def build_gan_transforms(image_size: int) -> transforms.Compose:
    """
    GAN için ön işleme:
    - SquarePad: görüntü bozulmasın / kırpılmasın
    - Resize: 128x128
    - ToTensor + Normalize [-1, 1]  (Tanh çıkışıyla uyumlu)
    """
    return transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def build_malign_train_subset(dataset: HAM10000Dataset, cfg: dict) -> Subset:
    """
    1) train_baseline.py ile aynı bölümlemeyi uygular (aynı seed, aynı oranlar).
    2) Yalnızca TRAIN parçasını alır; val/test'e DOKUNMAZ.
    3) TRAIN içindeki label==1 (Malign) örneklerini filtreleyip Subset döner.
    """
    total = len(dataset)
    train_size = int(cfg["train_ratio"] * total)
    val_size   = int(cfg["val_ratio"]   * total)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(cfg["seed"])
    train_ds, _val_ds, _test_ds = random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator,
    )

    # Subset.indices -> orijinal dataset satır indeksleri
    original_indices = list(train_ds.indices)
    labels = dataset.df["label"].values
    malign_indices = [i for i in original_indices if int(labels[i]) == 1]

    print(f"[Split]  Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print(f"[Filter] Train içindeki Malign (label=1) sayısı: "
          f"{len(malign_indices)} / {train_size}")

    if len(malign_indices) == 0:
        raise RuntimeError(
            "Train split içinde hiç Malign örnek bulunamadı; "
            "etiketleme/bölümleme mantığını kontrol edin."
        )

    return Subset(dataset, malign_indices)


# =============================================================================
# 5. EĞİTİM DÖNGÜSÜ
# =============================================================================
def train_dcgan(cfg: dict) -> None:
    # --- Cihaz ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # --- Klasörleri hazırla ---
    cfg["samples_dir"].mkdir(parents=True, exist_ok=True)
    cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    # --- Veri ---
    transform_pipeline = build_gan_transforms(cfg["image_size"])
    full_dataset = HAM10000Dataset(
        csv_path=cfg["metadata_csv"],
        image_dir=cfg["images_dir"],
        transform=transform_pipeline,
    )
    malign_train = build_malign_train_subset(full_dataset, cfg)

    # drop_last=True -> BatchNorm'un stabil çalışması için sonuncu eksik batch
    # atılır.
    loader = DataLoader(
        malign_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(cfg["num_workers"] > 0),
    )
    print(f"[Loader] Batch={cfg['batch_size']} | Adım/epoch={len(loader)}")

    # --- Modeller ---
    netG = Generator(nz=cfg["nz"], ngf=cfg["ngf"], nc=cfg["nc"]).to(device)
    netD = Discriminator(nc=cfg["nc"], ndf=cfg["ndf"]).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # --- Loss & Optim (TTUR: D yavaş, G hızlı) ---
    criterion = nn.BCELoss()
    optG = optim.Adam(netG.parameters(),
                      lr=cfg["lr_G"], betas=(cfg["beta1"], cfg["beta2"]))
    optD = optim.Adam(netD.parameters(),
                      lr=cfg["lr_D"], betas=(cfg["beta1"], cfg["beta2"]))
    print(f"[TTUR]   lr_D={cfg['lr_D']} | lr_G={cfg['lr_G']}")
    print(f"[Smooth] real_label={cfg['real_label']} | fake_label={cfg['fake_label']}")

    # --- Görsel takip için sabit gürültü ---
    fixed_noise = torch.randn(
        cfg["num_fixed_samples"], cfg["nz"], 1, 1, device=device
    )

    real_label = float(cfg["real_label"])
    fake_label = float(cfg["fake_label"])

    print(f"[Start]  DCGAN eğitimi - {cfg['epochs']} epoch")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        running_d = 0.0
        running_g = 0.0
        seen_batches = 0

        pbar = tqdm(loader, desc=f"[Epoch {epoch}/{cfg['epochs']}]", leave=False)
        for real_images, _labels in pbar:
            bs = real_images.size(0)
            real_images = real_images.to(device, non_blocking=True)

            # ============================================================
            # (1) Discriminator eğitimi: max log D(x) + log(1 - D(G(z)))
            # ============================================================
            netD.zero_grad(set_to_none=True)

            # 1a) Gerçek batch
            labels_real = torch.full((bs,), real_label, dtype=torch.float,
                                     device=device)
            out_real = netD(real_images)
            lossD_real = criterion(out_real, labels_real)
            lossD_real.backward()
            D_x = out_real.mean().item()

            # 1b) Sahte batch
            noise = torch.randn(bs, cfg["nz"], 1, 1, device=device)
            fake_images = netG(noise)
            labels_fake = torch.full((bs,), fake_label, dtype=torch.float,
                                     device=device)
            out_fake = netD(fake_images.detach())
            lossD_fake = criterion(out_fake, labels_fake)
            lossD_fake.backward()
            D_G_z1 = out_fake.mean().item()

            lossD = lossD_real + lossD_fake
            optD.step()

            # ============================================================
            # (2) Generator eğitimi: max log D(G(z))  (hedef: label=1)
            # ============================================================
            netG.zero_grad(set_to_none=True)
            labels_gen = torch.full((bs,), real_label, dtype=torch.float,
                                    device=device)
            out_fake_for_g = netD(fake_images)   # detach YOK
            lossG = criterion(out_fake_for_g, labels_gen)
            lossG.backward()
            D_G_z2 = out_fake_for_g.mean().item()
            optG.step()

            running_d += lossD.item()
            running_g += lossG.item()
            seen_batches += 1
            global_step += 1

            pbar.set_postfix(
                D=f"{running_d / seen_batches:.3f}",
                G=f"{running_g / seen_batches:.3f}",
                Dx=f"{D_x:.2f}",
                DGz=f"{D_G_z1:.2f}/{D_G_z2:.2f}",
            )

        # Epoch özeti
        avg_d = running_d / max(seen_batches, 1)
        avg_g = running_g / max(seen_batches, 1)
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{cfg['epochs']}] "
            f"LossD: {avg_d:.4f} | LossG: {avg_g:.4f} | "
            f"Toplam süre: {elapsed/60:.1f} dk"
        )

        # --- Her N epoch'ta bir sabit gürültüden örnek kaydet ---
        if (epoch % cfg["sample_every"] == 0) or (epoch == cfg["epochs"]):
            netG.eval()
            with torch.no_grad():
                fake_fixed = netG(fixed_noise).detach().cpu()
            # [-1, 1] -> [0, 1]
            grid_path = cfg["samples_dir"] / f"epoch_{epoch:03d}.png"
            save_image(
                fake_fixed,
                grid_path,
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            print(f"  -> Örnek grid kaydedildi: {grid_path}")
            netG.train()

    # --- Son Generator ağırlıklarını kaydet ---
    gen_path = cfg["checkpoint_dir"] / cfg["generator_name"]
    torch.save(
        {
            "model_state_dict": netG.state_dict(),
            "config": {
                "nz":  cfg["nz"],
                "ngf": cfg["ngf"],
                "nc":  cfg["nc"],
                "image_size": cfg["image_size"],
                "epochs": cfg["epochs"],
            },
        },
        gen_path,
    )
    print(f"\n[Finish] Generator ağırlıkları kaydedildi: {gen_path}")
    print(f"[Finish] Toplam süre: {(time.time() - t0)/60:.1f} dk")


# =============================================================================
# 6. ANA AKIŞ
# =============================================================================
def main() -> None:
    # cuDNN benchmark -> sabit giriş boyutunda hızlanma
    torch.backends.cudnn.benchmark = True

    # Üretkenlikte çeşitlilik için seed'i SABİTLEMİYORUZ; yalnızca bölümleme
    # seed'i (cfg['seed']=42) sabit ve tutarlı.
    train_dcgan(CONFIG)


if __name__ == "__main__":
    main()
