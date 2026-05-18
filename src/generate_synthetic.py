"""
Sentetik Malign (Kanser) Görüntü Üretim Betiği
===============================================

Amaç:
-----
`train_gan.py` ile eğitilmiş DCGAN Generator modelini
(`checkpoints/gan/generator_final.pth`) yükleyip veri dengesizliğini
gidermek amacıyla 6450 adet sentetik "Malign" cilt lezyonu görüntüsü
üretir ve `data/synthetic/generated_malign/` altına JPEG olarak kaydeder.

Kullanım:
---------
    python src/generate_synthetic.py

Not:
----
Üretilen görseller yalnızca EĞİTİM veri setini dengelemek için kullanılır;
hiçbir şekilde test setine karıştırılmaz (bkz. proje yol haritası, Adım 6).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

# Proje kökünü bulup src/ klasörünü import path'ine ekle
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

import config                            # noqa: E402
from train_gan import Generator          # noqa: E402  (tek kaynak: train_gan)


# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    # Yollar (config modülünden)
    "generator_ckpt": config.GAN_CKPT_DIR / "generator_final.pth",
    "output_dir":     config.SYNTHETIC_MALIGN_DIR,

    # Generator mimari parametreleri (train_gan.py ile birebir aynı)
    "nz":  100,     # Latent vektör boyutu
    "ngf": 64,      # Generator base feature map
    "nc":  3,       # Kanal sayısı (RGB)

    # Üretim ayarları
    "num_images": 6450,    # Toplam üretilecek sentetik kanser görüntüsü
    "batch_size": 64,      # VRAM dostu parça boyutu
    "filename_prefix": "synth_malign",
    "filename_digits": 5,  # synth_malign_00001.jpg ... 06450.jpg
    "jpeg_quality": 95,
}


# =============================================================================
# 3. YARDIMCI FONKSİYONLAR
# =============================================================================
def load_generator(cfg: dict, device: torch.device) -> nn.Module:
    """Checkpoint'ten Generator ağırlıklarını yükler ve eval moduna alır."""
    ckpt_path = cfg["generator_ckpt"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Generator checkpoint bulunamadı: {ckpt_path}\n"
            f"Önce `python src/train_gan.py` ile DCGAN'i eğitmelisiniz."
        )

    netG = Generator(nz=cfg["nz"], ngf=cfg["ngf"], nc=cfg["nc"]).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    netG.load_state_dict(state_dict)
    netG.eval()
    return netG


def compute_batch_sizes(total: int, batch_size: int):
    """Toplam sayıyı eşit/sonundan eksik batch'lere böler."""
    n_full = total // batch_size
    remainder = total - n_full * batch_size
    sizes = [batch_size] * n_full
    if remainder > 0:
        sizes.append(remainder)
    return sizes


# =============================================================================
# 4. ANA ÜRETİM AKIŞI
# =============================================================================
@torch.no_grad()
def generate(cfg: dict) -> None:
    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # Çıkış klasörü
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    print(f"[Output] {cfg['output_dir']}")

    # Generator
    netG = load_generator(cfg, device)
    print(f"[Model]  Generator yüklendi -> eval() modunda")

    total = int(cfg["num_images"])
    batch_sizes = compute_batch_sizes(total, int(cfg["batch_size"]))
    print(f"[Plan]   Toplam: {total} | Batch={cfg['batch_size']} "
          f"| Adım sayısı: {len(batch_sizes)}")

    prefix = cfg["filename_prefix"]
    digits = int(cfg["filename_digits"])

    generated = 0
    pbar = tqdm(
        total=total,
        desc=f"Üretiliyor: {total} adet sentetik kanser verisi",
        unit="img",
    )

    for bs in batch_sizes:
        # Rastgele gürültü -> sentetik görüntü
        noise = torch.randn(bs, cfg["nz"], 1, 1, device=device)
        fake = netG(noise)   # [-1, 1] aralığında (Tanh)

        # Her bir görüntüyü ayrı JPEG olarak kaydet.
        # save_image(normalize=True, value_range=(-1, 1)) ile otomatik olarak
        # [0, 1] aralığına taşınıp doğru şekilde JPEG'e yazılır.
        for i in range(bs):
            generated += 1
            fname = f"{prefix}_{generated:0{digits}d}.jpg"
            out_path = cfg["output_dir"] / fname
            save_image(
                fake[i].detach().cpu(),
                out_path,
                normalize=True,
                value_range=(-1, 1),
            )
            pbar.update(1)

    pbar.close()
    print(f"\n[Done]   {generated} sentetik görüntü kaydedildi.")
    print(f"[Dir ]   {cfg['output_dir']}")


# =============================================================================
# 5. GİRİŞ NOKTASI
# =============================================================================
def main() -> None:
    torch.backends.cudnn.benchmark = True
    generate(CONFIG)


if __name__ == "__main__":
    main()
