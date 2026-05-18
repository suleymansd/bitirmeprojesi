import os
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from glob import glob
from pathlib import Path


# HAM10000 -> Binary sınıf eşlemesi (Proje Yol Haritası Adım-1)
# Malign (1): mel, bcc, akiec
# Benign (0): nv, bkl, df, vasc
MALIGN_CLASSES = {'mel', 'bcc', 'akiec'}
BENIGN_CLASSES = {'nv', 'bkl', 'df', 'vasc'}
VALID_DX_CLASSES = MALIGN_CLASSES | BENIGN_CLASSES
LABEL_MAP = {dx: 1 for dx in MALIGN_CLASSES}
LABEL_MAP.update({dx: 0 for dx in BENIGN_CLASSES})


class SquarePad:
    """
    Görüntüleri kırpmadan veya pikselleri ezmeden 1:1 en-boy oranına getirir.
    Eksik kalan kısımları siyah (0,0,0) ile doldurur (Letterboxing).
    """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left = (max_wh - image.size[0]) // 2
        p_top = (max_wh - image.size[1]) // 2
        p_right = max_wh - image.size[0] - p_left
        p_bottom = max_wh - image.size[1] - p_top
        padding = (p_left, p_top, p_right, p_bottom)
        return ImageOps.expand(image, padding, fill=(0, 0, 0))


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): HAM10000_metadata.csv dosyasının yolu.
            image_dir (str): Fotoğrafların bulunduğu ana klasörün yolu.
            transform (callable, optional): Görüntü ön işleme adımları.
        """
        csv_path = Path(csv_path)
        image_dir = Path(image_dir)

        if not csv_path.is_file():
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {csv_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Görsel klasörü bulunamadı: {image_dir}")

        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # 1. Aşama: Sınıfları Binary (0 ve 1) Olarak Haritalama
        # Malign (Kötü Huylu) -> 1, Benign (İyi Huylu) -> 0
        # Malign: mel, bcc, akiec
        # Benign: nv, bkl, df, vasc
        unknown_dx = set(self.df['dx'].unique()) - VALID_DX_CLASSES
        if unknown_dx:
            raise ValueError(
                "HAM10000 metadata içinde beklenmeyen dx sınıfları bulundu: "
                f"{sorted(unknown_dx)}"
            )
        self.df['label'] = self.df['dx'].map(LABEL_MAP).astype('int64')
        
        # 2. Aşama: Görsellerin tam dosya yollarını eşleştirme
        # Görüntüler part_1 ve part_2 klasörlerinde olabileceği için tüm yolları tarıyoruz
        image_paths = glob(str(image_dir / '**' / '*.jpg'), recursive=True)
        self.image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_paths}
        
        # Veri setinde fotoğrafı bulunamayan satırları temizleme (Güvenlik adımı)
        self.df['path'] = self.df['image_id'].map(self.image_path_dict)
        self.df = self.df.dropna(subset=['path']).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Fotoğrafı yükle
        img_path = self.df.loc[idx, 'path']
        image = Image.open(img_path).convert('RGB')
        
        # Etiketi al (0 veya 1)
        label = torch.tensor(self.df.loc[idx, 'label'], dtype=torch.long)
        
        # Transform (Ön işleme) uygula
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ==========================================
# ÇALIŞTIRMA VE TEST BLOKU
# ==========================================
if __name__ == "__main__":
    # Proje kökünü otomatik bul (çalıştırma dizininden bağımsız)
    ROOT_DIR = Path(__file__).resolve().parents[1]
    METADATA_CSV_PATH = ROOT_DIR / 'data' / 'raw' / 'HAM10000_metadata.csv'
    IMAGES_FOLDER_PATH = ROOT_DIR / 'data' / 'raw'
    
    # Derin Öğrenme Modelleri için Standart Ön İşleme
    transform_pipeline = transforms.Compose([
        SquarePad(),                              # Görüntüyü bozmadan 1:1 kare yap
        transforms.Resize((224, 224)),            # ResNet/MobileNet standart boyutu
        transforms.ToTensor(),                    # PyTorch Tensor'üne çevir (0-1 arası)
        transforms.Normalize(                     # ImageNet standartlarına göre normalize et
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Veri setini oluştur
    dataset = HAM10000Dataset(
        csv_path=METADATA_CSV_PATH, 
        image_dir=IMAGES_FOLDER_PATH, 
        transform=transform_pipeline
    )
    
    # DataLoader oluştur (GPU optimizasyonlu)
    # Donanımındaki gücü kullanmak için num_workers=4 veya 8, pin_memory=True yapıldı.
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Kontrol Çıktısı
    print(f"Toplam geçerli veri sayısı: {len(dataset)}")
    print(f"Benign (0) Sayısı: {len(dataset.df[dataset.df['label'] == 0])}")
    print(f"Malign (1) Sayısı: {len(dataset.df[dataset.df['label'] == 1])}")
    
    # İlk batch'i çekip boyutları kontrol etme
    images, labels = next(iter(dataloader))
    print(f"\nBatch Görüntü Boyutu: {images.shape}") # Beklenen: [32, 3, 224, 224]
    print(f"Batch Etiketleri: {labels.tolist()}")