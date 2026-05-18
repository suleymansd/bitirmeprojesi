# 📘 Proje Rehberi ve Dosya Sözlüğü

> **Proje Adı:** Cilt Kanseri Teşhisi — DCGAN ile Sentetik Veri + Federe Öğrenme Hibrit Sistemi
>
> **Amaç:** Bu döküman, projede **neyi**, **neden**, **nasıl** yaptığımızı; klasörlerin ve dosyaların **tek tek** ne işe yaradığını, kodu hiç görmemiş bir takım arkadaşımızın bile rahatça anlayabileceği düzeyde anlatır. Başka hiçbir şey okumana gerek kalmayacak şekilde yazılmıştır.

---

## 📑 İçindekiler

1. [Bir Dakikada Proje](#1-bir-dakikada-proje)
2. [Problem, Hipotez ve Çözüm Mantığı](#2-problem-hipotez-ve-çözüm-mantığı)
3. [Bilinmesi Gereken Terimler (Aptala Anlatır Gibi)](#3-bilinmesi-gereken-terimler-aptala-anlatır-gibi)
4. [Projenin Omurgası: Tekrar Eden 5 Altın Kural](#4-projenin-omurgası-tekrar-eden-5-altın-kural)
5. [Klasör Ağacı (Görsel Harita)](#5-klasör-ağacı-görsel-harita)
6. [Büyük Resim: Uçtan Uca İş Akışı](#6-büyük-resim-uçtan-uca-iş-akışı)
7. [Klasör Sözlüğü](#7-klasör-sözlüğü)
8. [Dosya Sözlüğü (Her Dosya Tek Tek)](#8-dosya-sözlüğü-her-dosya-tek-tek)
9. [Çalıştırma Sırası (Copy-Paste Komutlar)](#9-çalıştırma-sırası-copy-paste-komutlar)
10. [Çıktılar Nasıl Okunur?](#10-çıktılar-nasıl-okunur)
11. [Sık Karşılaşılan Hatalar ve Çözümleri](#11-sık-karşılaşılan-hatalar-ve-çözümleri)
12. [Jüri/Sunum Hazırlık Notları](#12-jüriden-gelebilecek-sorular-ve-cevapları)

---

## 1. Bir Dakikada Proje

HAM10000 adında 10.015 adet cilt lezyonu fotoğrafı içeren açık bir veri seti var. Biz bu fotoğrafları kullanarak **bir fotoğrafın kanser (Malign) mi, zararsız (Benign) mi olduğunu** söyleyen bir yapay zekâ eğitiyoruz.

İki büyük problem vardı:

- **(1) Veri dengesizdi:** ~8000 Benign, ~1950 Malign. Model Benign sınıfına çalışınca bile %80 doğruluk alıyor ama **kanser vakalarını kaçırıyordu**. Bu projede en önemli şey bu vakaları kaçırmamak.
- **(2) Hastaneler verilerini paylaşamıyor (KVKK/GDPR):** Tek bir yerde veri toplayıp eğitmek gerçek dünyada mümkün değil.

Çözümümüz **iki büyük yenilikle** bu sorunları aşıyor:

- **DCGAN:** Bir üretici yapay zekâ eğittik. Bu ağ, azınlık sınıf olan Malign (kanser) fotoğraflarına bakıp onlara benzer **6450 adet sentetik kanser fotoğrafı** üretti. Böylece veri setini dengeledik.
- **Federated Learning (Flower):** Veriyi tek bir yerde toplamak yerine 5 hastaneyi (client) simüle ettik. Her hastane kendi verisiyle yerelde eğitildi, sadece **ağırlıkları** (öğrendikleri) merkeze gönderdi. Merkez **FedAvg** ile bu ağırlıkları birleştirip global bir süper model üretti.

Sonuç olarak hem veri gizliliğini koruyoruz hem de kanser yakalama oranını (Recall) arttırıyoruz.

---

## 2. Problem, Hipotez ve Çözüm Mantığı

### 2.1. Problem (Dengesiz Veri)

| Sınıf | Açıklama | Adet |
|---|---|---|
| Benign (0) | İyi huylu lezyonlar: `nv` (ben), `bkl`, `df`, `vasc` | 8061 |
| Malign (1) | Kötü huylu / kanser öncüsü: `mel`, `bcc`, `akiec` | 1954 |
| **Toplam** | | **10015** |

Malign sınıfı yaklaşık **5’te 1** oranında. Model bu yüzden kanseri “öğrenmek yerine görmezden gelmeyi” öğrenir.

### 2.2. Hipotez

> "Azınlık sınıf olan Malign’i DCGAN ile çoğaltıp dengelersek ve hastane verilerini yerel tutan bir Federated Learning sistemine koyarsak; **Malign Recall (kanseri kaçırmama oranı)** baseline’a göre anlamlı biçimde artar."

### 2.3. Çözüm Sırası

1. **Baseline** (Adım-1): Dengesiz veriyle klasik bir ResNet50 eğit. Ne kadar kanser kaçırdığını kaydet.
2. **DCGAN** (Adım-2): Malign fotoğraflara bakarak 6450 yeni sentetik Malign üret.
3. **Federated Splits** (Adım-3): Gerçek Train + Sentetik Malign’i 5 hastaneye IID (eşit karışık) dağıt.
4. **Federated Learning** (Adım-4): 5 istemci + 1 sunucu, FedAvg ile 5 raund eğitim.
5. **Karşılaştırma** (Adım-5): Global modeli aynı gerçek test setinde değerlendir ve baseline ile karşılaştır.

> 🎯 **Nihai karşılaştırmada test seti yalnızca gerçek HAM10000 fotoğraflarından oluşur.** Sentetik veri test’te asla kullanılmaz.

---

## 3. Bilinmesi Gereken Terimler (Aptala Anlatır Gibi)

- **Dataset (veri seti):** Modelin öğrenmek için kullanacağı örneklerin (foto + etiket) topluluğu.
- **Train/Val/Test Split:** Veriyi 3’e bölmek. **Train** ile öğretiriz, **Val** ile eğitim sırasında başarıyı takip ederiz, **Test** ise en sonda, ilk kez göreceği sınav sorularıdır. Test’e kesinlikle dokunulmaz.
- **Label/Etiket:** Bir fotoğrafın cevabı. Bizde `0 = Benign`, `1 = Malign`.
- **Binary Classification:** İki sınıflı sınıflandırma. "Kanser mi değil mi?"
- **Imbalance (Dengesizlik):** Bir sınıfın diğerinden çok daha fazla olması. Bizde `~4:1` oran.
- **Transfer Learning:** Önceden başka büyük bir veri setinde (ImageNet) eğitilmiş model ağırlıklarını alıp kendi problemimize uyarlamak. Sıfırdan eğitmekten çok daha hızlı ve başarılıdır.
- **ResNet50:** 50 katmanlı ünlü bir CNN mimarisi. Bizim sınıflandırıcı iskeletimiz.
- **DCGAN (Deep Convolutional GAN):** İki ağın "kedi–fare" oyunu. **Generator** sahte fotoğraflar üretir, **Discriminator** gerçek/sahte diye ayırt etmeye çalışır. Bu oyun zamanla Generator’ı çok iyi sahtecilik yapan bir ressama çevirir.
- **Latent vector (z):** Generator’a verilen rastgele gürültü. Her farklı z farklı bir fotoğraf üretir.
- **Mode Collapse:** Generator’ın sadece birkaç tip fotoğraf üretmesi (çeşitlilik çökmesi). DCGAN’ın klasik derdi.
- **TTUR (Two Time-Scale Update Rule):** Discriminator’ı yavaşlatmak, Generator’ı nispeten hızlandırmak için farklı learning rate’ler kullanmak.
- **Label Smoothing:** Gerçek örnekler için `1.0` yerine `0.9` etiket vermek. D’nin aşırı özgüvene kapılmasını engeller.
- **Dropout:** Eğitim sırasında rastgele nöronları kapatmak. Aşırı öğrenmeyi (overfit) engeller.
- **AMP (Automatic Mixed Precision):** Hesaplamaların bir kısmını `float16` ile yapmak. GPU hafızasından (VRAM) tasarruf ve hız.
- **Federated Learning (Federe Öğrenme):** Veriyi merkeze toplamadan, modelin kendisini gezdirerek eğitmek.
- **FedAvg:** Federated learning’in en temel algoritması. Her istemciden gelen ağırlıkları (veri sayısına göre) ortalamak.
- **Round (Raund):** Federe eğitimin bir iterasyonu. Bizde 5 raund var.
- **IID (Independent & Identically Distributed):** Her istemcinin aynı dağılımdan veri görmesi. Biz veriyi karıştırıp eşit dağıttığımız için bizde IID.
- **Recall (Duyarlılık) — Malign (1):** Gerçek kanser hastalarından kaçının kanser olarak tespit edildiği oranı. `TP / (TP + FN)`. **Projenin odak metriği.** FN = kaçırılan kanser = hayati risk.
- **Precision:** Kanser dediklerimizin kaçı gerçekten kanser.
- **F1 / Macro-F1:** Precision ve Recall’un dengeli özeti. Macro, sınıf başına F1’in düz ortalaması (dengesizliğe daha adil).
- **Confusion Matrix:** `[TN, FP; FN, TP]` şeklinde 2×2 tablo. Modelin nerede yanıldığını net gösterir.
- **Checkpoint (.pth):** Modelin ağırlıklarının bir anlık fotoğrafı. Daha sonra yüklemek için diskte tutulan dosya.
- **Seed (tohum):** Rastgeleliği tekrarlanabilir yapmak için verilen sabit sayı. Bizde daima `42`.

---

## 4. Projenin Omurgası: Tekrar Eden 5 Altın Kural

Bu kurallar bütün betiklerde **birebir aynı** uygulanır. Aksi davranmak veri sızıntısı (data leakage) yaratır ve sonuçları geçersiz kılar.

1. **Sınıf eşlemesi (binary):**
   - Malign (1) = `mel`, `bcc`, `akiec`
   - Benign (0) = `nv`, `bkl`, `df`, `vasc`
2. **Split oranı:** `%80 Train`, `%10 Val`, `%10 Test`.
3. **Split seed:** `torch.Generator().manual_seed(42)` — hepsi aynı split'i yeniden üretir.
4. **Ön işleme (baseline/federated sınıflandırma):** `SquarePad → Resize(224,224) → ToTensor → Normalize(ImageNet ortalama/std)`.
5. **Test seti kutsaldır:** Sentetik veri ve federe veri test'e **asla** karışmaz. Test yalnızca gerçek HAM10000'in 10%'luk parçasıdır.

---

## 5. Klasör Ağacı (Görsel Harita)

```
Cilt_Kanseri_Bitirme/
├── data/                               ← Ham + türetilmiş veriler (Git'e yüklenmez)
│   ├── raw/                            ← HAM10000 orijinali (10015 fotoğraf)
│   │   ├── HAM10000_images_part_1/     ← 1. parça JPG'ler
│   │   ├── HAM10000_images_part_2/     ← 2. parça JPG'ler
│   │   └── HAM10000_metadata.csv       ← image_id, dx, hasta bilgisi
│   ├── synthetic/                      ← DCGAN çıktıları
│   │   ├── generated_malign/           ← 6450 sentetik Malign JPG
│   │   └── samples/                    ← GAN eğitim ilerleme grid'leri
│   └── federated_splits/               ← 5 istemciye dağıtılmış CSV'ler
│       ├── client_1.csv ... client_5.csv
│
├── src/                                ← Tüm Python kaynak kodları
│   ├── data_loader.py                  ← HAM10000 Dataset sınıfı
│   ├── train_baseline.py               ← Baseline ResNet50 eğitimi
│   ├── evaluate.py                     ← Baseline test değerlendirmesi
│   ├── train_gan.py                    ← DCGAN eğitimi
│   ├── generate_synthetic.py           ← Sentetik Malign fotoğraf üretimi
│   ├── prepare_federated_splits.py     ← 5 istemci CSV dağıtımı
│   ├── federated_utils.py              ← Ortak yardımcılar (Dataset, model, param I/O)
│   ├── client.py                       ← Flower istemcisi
│   ├── server.py                       ← Flower sunucusu + FedAvg + kayıt
│   └── evaluate_federated.py           ← Global modelin test değerlendirmesi
│
├── checkpoints/                        ← Eğitilmiş modellerin ağırlıkları
│   ├── baseline/                       ← baseline_best_model.pth, CM PNG, log
│   ├── gan/                            ← generator_final.pth
│   └── federated/                      ← federated_model_round_{1..5}.pth, CM PNG
│
├── run_federated.ps1                   ← 1 komutla server+5 client başlatır
├── project_dict.py                     ← (opsiyonel) PDF rehberi üreteci (WeasyPrint)
├── klasor_mimarisi.txt                 ← Klasör yapısının kısa özeti
├── yol_haritasi.pdf                    ← Yüksek seviyeli proje planı
└── PROJE_REHBERI.md                    ← (bu dosya)
```

---

## 6. Büyük Resim: Uçtan Uca İş Akışı

Her kutu bir çalıştırma adımıdır. Okları izleyerek bütün proje sırayla çalışır.

```
 ┌──────────────────────────────┐
 │ 1) HAM10000 ham veri         │  data/raw/*.jpg + metadata.csv
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 2) data_loader.py            │  dx -> 0/1 binary label mapping
 │   + binary mapping           │  (tüm betikler bu sınıfı kullanır)
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 3) train_baseline.py         │  seed=42 split, ResNet50, AMP
 │   ↳ baseline_best_model.pth  │  ► "Baseline" = Kıyas noktası
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 4) evaluate.py               │  Baseline'ın test setindeki skoru
 │   ↳ Malign Recall (DÜŞÜK!)   │  ► Problemi kanıtladık
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 5) train_gan.py (DCGAN)      │  Train içindeki SADECE Malign'ler
 │   ↳ generator_final.pth      │  ► Sahte kanser üreticisi hazır
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 6) generate_synthetic.py     │  6450 sentetik Malign JPG üret
 │   ↳ data/synthetic/...       │
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 7) prepare_federated_splits  │  Train + Sentetik → shuffle → 5 CSV
 │   ↳ client_1..5.csv          │  Benign/Malign ~55% Malign (dengeli)
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 8) server.py + 5x client.py  │  5 raund FedAvg, AMP, her raund kayıt
 │   ↳ federated_model_round_N  │
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 9) evaluate_federated.py     │  Aynı test setinde Malign Recall
 │   ↳ BASELINE ile karşılaştır │  ► Hipotez doğrulandı mı?
 └──────────────────────────────┘
```

---

## 7. Klasör Sözlüğü

### 📁 `data/`
Tüm veri dosyalarının yaşadığı yer. Git'e **yüklenmez** (büyük ve gizlilik içerir).

- **`data/raw/`** — Kaggle’dan indirilen orijinal HAM10000.
  - `HAM10000_images_part_1/` ve `HAM10000_images_part_2/`: Aynı klasör mantığı, Kaggle iki parça halinde veriyor.
  - `HAM10000_metadata.csv`: 10015 satır. Bizim kullandığımız sütunlar `image_id` (dosya adı) ve `dx` (hastalık kodu).
- **`data/synthetic/generated_malign/`** — `generate_synthetic.py` tarafından üretilen 6450 JPG. Ad: `synth_malign_00001.jpg` → `synth_malign_06450.jpg`.
- **`data/synthetic/samples/`** — GAN eğitimi sırasında her 10 epoch’ta bir kaydedilen `epoch_010.png`, `epoch_020.png` … GAN’ın gelişimini gözle takip etmek için.
- **`data/federated_splits/`** — `prepare_federated_splits.py` çıktısı 5 CSV: `client_1.csv` … `client_5.csv`. Her biri `image_path,label` sütunlarını içerir.

### 📁 `src/`
Bütün Python kaynak kodları buradadır. Her dosyanın detayı ileride.

### 📁 `checkpoints/`
Eğitilmiş model ağırlıkları. **Not:** `.pth` dosyaları büyük olduğundan Git’e yüklenmez.

- **`checkpoints/baseline/`**
  - `baseline_best_model.pth` → En iyi Val Macro-F1’i yakalayan ResNet50 ağırlıkları.
  - `baseline_confusion_matrix.png` → `evaluate.py` çıktısı.
  - `terminal_loglari.png` → Eğitim ekran logu (manuel ekran görüntüsü).
- **`checkpoints/gan/`**
  - `generator_final.pth` → DCGAN Generator son ağırlıkları + mimari config (nz, ngf, nc, image_size).
- **`checkpoints/federated/`**
  - `federated_model_round_{1..5}.pth` → Her raundun sonunda FedAvg ile birleşen global model.
  - `federated_confusion_matrix.png` → `evaluate_federated.py` çıktısı.

### 📁 Kök dizindeki yardımcı dosyalar
- **`run_federated.ps1`** — Windows PowerShell betiği. Sunucuyu ve 5 istemciyi yeni pencerelerde otomatik başlatır.
- **`project_dict.py`** — WeasyPrint kullanarak HTML’den PDF rehber üreten küçük bir yardımcı (opsiyonel, dokümantasyon amaçlı).
- **`klasor_mimarisi.txt`** — Klasör yapısının kısa özeti.
- **`yol_haritasi.pdf`** — Projenin 6 aşamasını tanımlayan planlama belgesi.

---

## 8. Dosya Sözlüğü (Her Dosya Tek Tek)

> Her başlıkta: **Ne işe yarar**, **nasıl çalışır**, **girdi**, **çıktı**, **kritik noktalar**.

---

### 8.1. `src/data_loader.py` — Veri Kapısı

**Ne işe yarar:** HAM10000’i PyTorch’un anlayacağı `Dataset` sınıfına dönüştürür; sınıfları `mel/bcc/akiec → 1`, `nv/bkl/df/vasc → 0` olarak ikili etikete çevirir; eksik/hatalı girişleri temizler.

**Önemli parçalar:**
- `MALIGN_CLASSES = {'mel','bcc','akiec'}`, `BENIGN_CLASSES = {'nv','bkl','df','vasc'}`
- `VALID_DX_CLASSES` — Tanımsız bir `dx` geldiğinde **sessizce benign’e düşmek yerine `ValueError` fırlatır**. Bu, insan hatasına karşı bir emniyet kilidi.
- `SquarePad` — Görüntüyü kırpmadan **1:1 kare yapan letterbox**. Dermatoloji görüntülerinde sınırlardaki lezyon bilgisini korumak için şart.
- `HAM10000Dataset` — `__len__`, `__getitem__`, `transform` destekli klasik PyTorch Dataset.

**Girdi:** `csv_path`, `image_dir`, `transform`.
**Çıktı:** `(image_tensor, label_tensor)` çiftleri.

**Kritik:** `transform=None` olabilir. `prepare_federated_splits.py` yalnızca path/label lazım olduğu için transform’u `None` gönderir.

---

### 8.2. `src/train_baseline.py` — Baseline (Kıyas Noktası) Eğitimi

**Ne işe yarar:** Orijinal dengesiz veriyle ResNet50 eğitir. Bu sonuç bize “DCGAN + Federe bunu ne kadar iyileştiriyor?” sorusunun cevabını veren **kıyas noktası**dır.

**Nasıl çalışır:**
1. `HAM10000Dataset`’i ImageNet normalize transform’uyla oluşturur.
2. `torch.Generator().manual_seed(42)` ile `%80/%10/%10` split yapar.
3. `resnet50(weights='DEFAULT')` yükler, son katmanı `Linear(num_ftrs, 2)` yapar.
4. 10 epoch eğitim: **AMP (autocast + GradScaler)**, Adam(lr=1e-4), CrossEntropyLoss.
5. Her epoch sonunda validasyonda `Macro-F1` + `Recall(pos_label=1)` hesaplar.
6. Val Macro-F1 her iyileştiğinde modeli `checkpoints/baseline/baseline_best_model.pth` olarak kaydeder.

**Önemli:** Class weighting/oversampling/undersampling **YOKTUR**. Çünkü amacımız baseline’ın kanser kaçırdığını kanıtlamak.

**Çıktı:** `baseline_best_model.pth` + terminalde epoch başına metrikler.

---

### 8.3. `src/evaluate.py` — Baseline Testi

**Ne işe yarar:** Baseline modeli, **aynı seed ile üretilen %10’luk test setinde** sınar.

**Nasıl çalışır:**
- `train_baseline.py` ile birebir aynı split’i üretir (seed=42, aynı oranlar, aynı `random_split` çağrı sırası). Sadece 3. parçayı (`test_ds`) alır.
- ResNet50 iskeletini yükler, `model_state_dict`’i checkpoint’ten load eder, `eval()` + `torch.no_grad()`.
- **Terminale:** `classification_report` (Benign/Malign için precision, recall, F1) + ham confusion matrix.
- **Diske:** `checkpoints/baseline/baseline_confusion_matrix.png` (seaborn heatmap).
- Ekstra özet: `Malign Recall = TP/(TP+FN)` ve `FN = kaçırılan kanser sayısı`.

**Kritik:** `matplotlib.use("Agg")` ile headless mod — sunucularda/terminal’da takılmaz.

---

### 8.4. `src/train_gan.py` — DCGAN Eğitimi (Sentetik Kanser Üretici)

**Ne işe yarar:** Train split’indeki **yalnızca Malign (label=1)** fotoğraflara bakarak sahte kanser fotoğrafı üretmeyi öğrenen bir Generator eğitir.

**Veri izolasyonu (kritik):**
- Split, `train_baseline.py` ile birebir aynı (seed=42).
- Train split’inin orijinal indisleri alınır; bunların içinden `label==1` olanlar filtrelenir.
- **Val ve test GAN'a asla gösterilmez.** Aksi halde sentetik veri test bilgisini "ezberleyebilir".

**Mimari (128×128):**
- **Generator:** 6 × `ConvTranspose2d` + `BatchNorm2d` + `ReLU`. `1×1 → 4 → 8 → 16 → 32 → 64 → 128`. Son aktivasyon `Tanh` ⇒ çıktı `[-1, 1]`.
- **Discriminator:** 6 × `Conv2d` + `BatchNorm2d` + `LeakyReLU(0.2)`. `128 → 64 → 32 → 16 → 8 → 4 → 1`. Ortalara **3 × `Dropout(0.3)`** eklendi. Son aktivasyon `Sigmoid`.
- **Ağırlık ilkleme (DCGAN makalesi):** Conv → `N(0, 0.02)`, BatchNorm weight → `N(1, 0.02)`, bias → `0`.

**Stabilizasyon önlemleri:**
- **TTUR:** `lr_D = 1e-4`, `lr_G = 2e-4` — Discriminator’ı yavaşlatıp Generator’ın nefes almasını sağlar.
- **One-sided label smoothing:** Gerçek etiket `0.9`, sahte `0.0`. D’nin aşırı özgüvenle LossD→0’a çakılmasını engeller.
- **Discriminator Dropout:** Ortalara 3 × Dropout(0.3). Hafif körleştirme.
- **BCELoss + Adam (beta1=0.5, beta2=0.999)** — klasik DCGAN seçimleri.

**Ön işleme:**
- `SquarePad → Resize(128,128) → ToTensor → Normalize(mean=0.5, std=0.5)`.
- **Önemli:** Burada ImageNet normalizasyonu **kullanılmaz** çünkü Generator’ın Tanh çıkışı `[-1, 1]` aralığı ister.

**İzleme ve kayıt:**
- Her 10 epoch’ta `fixed_noise`’dan 16 görüntü grid’i → `data/synthetic/samples/epoch_NNN.png`.
- Son epoch sonunda Generator ağırlıkları → `checkpoints/gan/generator_final.pth`.

---

### 8.5. `src/generate_synthetic.py` — Sentetik Fabrika

**Ne işe yarar:** Eğitilmiş Generator’ı yükleyip **6450 adet** sentetik Malign JPG üretir.

**Nasıl çalışır:**
- `Generator` sınıfı bu dosyaya **birebir kopyalanmıştır** (bağımsızlık için).
- Checkpoint `checkpoints/gan/generator_final.pth` yüklenir, `eval()` moduna alınır.
- `torch.no_grad()` içinde batch’ler halinde (batch_size=64) 101 adımda üretim yapılır.
- Her çıktı `torchvision.utils.save_image(..., normalize=True, value_range=(-1, 1))` ile `[-1,1] → [0,1]`’e taşınır ve JPG olarak yazılır.
- Dosya adları 5 haneli: `synth_malign_00001.jpg` … `synth_malign_06450.jpg`.

**Çıktı:** `data/synthetic/generated_malign/` altında 6450 JPG.

---

### 8.6. `src/prepare_federated_splits.py` — 5 Hastaneye Dağıtım

**Ne işe yarar:** Gerçek Train + Sentetik Malign verilerini tek bir büyük listede birleştirir, karıştırır, 5 eşit parçaya böler ve CSV olarak yazar.

**Adımlar:**
1. `HAM10000Dataset` + `seed=42` split → yalnızca `train_ds.indices` alınır (val/test’e dokunulmaz).
2. Her Train örneği için `(absolute_path, label)` kaydı çıkarılır.
3. `data/synthetic/generated_malign/*.jpg` tam yolları toplanır, tamamına `label=1` atanır.
4. İki liste birleştirilir (`merged = original + synthetic`).
5. `random.Random(42).shuffle(merged)` ile karıştırılır.
6. 5 eşit parçaya bölünür (kalanlar ilk client’lara birer birer dağıtılır).
7. Her parça `client_i.csv` olarak yazılır.
8. Terminale Benign/Malign dağılımı tablosu basılır.

**Tipik sayılar:**

| Kaynak | Benign | Malign | Toplam |
|---|---|---|---|
| Orijinal Train | 6466 | 1546 | 8012 |
| Sentetik | 0 | 6450 | 6450 |
| **Birleşik** | **6466** | **7996** | **14462** |
| Her client (~) | ~1287 | ~1600 | ~2892 |

~%55 Malign oranıyla her client dengeli bir eğitim seti alır.

---

### 8.7. `src/federated_utils.py` — Ortak Yardımcılar

**Ne işe yarar:** Client ve server’ın paylaştığı parçaları tek yerde tutar.

**İçinde ne var:**
- `FedClientDataset`: CSV’den `image_path, label` okuyan PyTorch Dataset. `class_counts()` hızlı tanı için.
- `build_transforms()`: `train_baseline.py` ile birebir aynı pipeline (SquarePad + Resize 224 + ImageNet Normalize).
- `build_model(num_classes=2, pretrained=False)`: ResNet50 + `Linear(num_ftrs, 2)`. Federe senaryoda pretrained=False; ağırlıklar sunucudan gelir.
- `get_parameters(model)`: `state_dict → List[np.ndarray]` — Flower’ın istediği format.
- `set_parameters(model, params)`: Tersi yönde yükleme.

---

### 8.8. `src/client.py` — Flower İstemcisi (Hastane)

**Ne işe yarar:** Bir hastane gibi davranır. Kendi CSV’sini (`client_{i}.csv`) okur, global ağırlıkları sunucudan alır, **1 epoch** yerel eğitim yapar, güncel ağırlıkları sunucuya döner.

**Önemli metodlar (`CiltKanseriClient(NumPyClient)`):**
- `get_parameters(config)` → `List[np.ndarray]`.
- `fit(parameters, config)` → `(new_parameters, num_samples, metrics)`. AMP ile 1 epoch Adam eğitimi.
- `evaluate(parameters, config)` → `(loss, num_samples, metrics)`. Lokal veriyle accuracy/loss.

**CLI:**
```powershell
python src/client.py --client_id 1
python src/client.py --client_id 2 --server_address 127.0.0.1:8080 --batch_size 32
```

**Optimizasyonlar:** CUDA + AMP (`autocast + GradScaler`), `pin_memory=True`, Windows’ta `num_workers=0` (diğer OS’de 8), `zero_grad(set_to_none=True)`, `non_blocking=True`.

---

### 8.9. `src/server.py` — Flower Sunucusu + FedAvg + Kayıt

**Ne işe yarar:** Tüm istemcileri orkestra eder. Her raundun sonunda FedAvg ile ağırlıkları birleştirip bunu `.pth` olarak diske kaydeder.

**Ayarlar:**
- `num_rounds=5`.
- `min_fit_clients=5`, `min_evaluate_clients=5`, `min_available_clients=5` → 5 istemcinin tamamı bağlanmadan raund başlamaz.
- `fraction_fit=1.0`, `fraction_evaluate=1.0`.
- `initial_parameters` → deterministik (seed=42) başlangıç: Bütün istemciler aynı sıfır noktasından başlasın.
- `fit_metrics_aggregation_fn = weighted_average`: Örnek sayısına göre ağırlıklı metrik ortalaması.

**Özel strateji `SaveFedAvg(FedAvg)`:**
- `aggregate_fit` override edilmiştir: Ana FedAvg birleştirmesi çağrıldıktan sonra
  `parameters_to_ndarrays → set_parameters(template) → torch.save(...)` ile her raund için
  `checkpoints/federated/federated_model_round_{N}.pth` yazılır.

---

### 8.10. `src/evaluate_federated.py` — Global Model Testi

**Ne işe yarar:** Federe eğitim bittikten sonra, son raundun global modelini **orijinal %10 test setinde** sınar.

**Nasıl çalışır:**
- `evaluate.py` ile birebir aynı split (seed=42, 80/10/10) — yalnızca 3. parça.
- `--round N` verilebilir; verilmezse en büyük round numarasını otomatik bulur.
- `classification_report` + terminal Confusion Matrix + `checkpoints/federated/federated_confusion_matrix.png`.
- Ekstra özet: Malign Recall, FN (kaçırılan kanser). **Baseline ile bu sayının karşılaştırması ana sunum materyalimiz.**

---

### 8.11. `run_federated.ps1` — Windows Tek-Komut Başlatıcı

**Ne işe yarar:** `python src/server.py` ve `python src/client.py --client_id {1..5}` komutlarını ayrı PowerShell pencerelerinde otomatik açar.

**Akış:**
1. Server’ı yeni pencerede başlatır.
2. 5 saniye bekler (sunucunun ayağa kalkması için).
3. 1’den 5’e kadar istemcileri başlatır, her biri arasında 2 saniye bekler (VRAM sıçramasını önlemek için).

**Kullanım:**
```powershell
# İlk seferde izin almak gerekebilir:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./run_federated.ps1
```

---

### 8.12. `project_dict.py` — (Opsiyonel) PDF Rehber Üreteci

**Ne işe yarar:** Basit bir HTML şablonu oluşturup `weasyprint` ile PDF’e çevirir. Bu markdown zaten detaylı bir rehber olduğundan opsiyoneldir; sunumlarda kısa özet basılı döküman ihtiyacı için bırakılmıştır.

---

### 8.13. `klasor_mimarisi.txt` ve `yol_haritasi.pdf`

- **`klasor_mimarisi.txt`** — Klasör ağacının kısa bir metin özeti.
- **`yol_haritasi.pdf`** — Projenin 6 adımını tarif eden planlama belgesi (bu dokümanın kaynağı).

---

## 9. Çalıştırma Sırası (Copy-Paste Komutlar)

### 9.1. Ortam Kurulumu

```powershell
# Sanal ortam aktif ise:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy pillow scikit-learn matplotlib seaborn tqdm
pip install "flwr>=1.7,<2.0"
```

### 9.2. Baseline (Kıyas Noktası)

```powershell
python src/train_baseline.py
python src/evaluate.py
```

### 9.3. DCGAN + Sentetik Veri

```powershell
python src/train_gan.py
python src/generate_synthetic.py
```

### 9.4. Federated Splits Hazırlığı

```powershell
python src/prepare_federated_splits.py
```

### 9.5. Federated Learning (Tek Komut — Windows)

```powershell
./run_federated.ps1
```

### 9.5. Federated Learning (Manuel — 6 pencere)

Terminal 1:
```powershell
python src/server.py
```
Terminal 2..6:
```powershell
python src/client.py --client_id 1
python src/client.py --client_id 2
python src/client.py --client_id 3
python src/client.py --client_id 4
python src/client.py --client_id 5
```

### 9.6. Federated Test Değerlendirmesi

```powershell
python src/evaluate_federated.py
# ya da belirli bir raund için
python src/evaluate_federated.py --round 5
```

---

## 10. Çıktılar Nasıl Okunur?

### 10.1. `classification_report` örneği

```
              precision    recall  f1-score   support
  Benign (0)     0.91      0.96      0.93       806
  Malign (1)     0.72      0.52      0.60       196
    accuracy                         0.88      1002
   macro avg     0.82      0.74      0.77      1002
weighted avg     0.87      0.88      0.87      1002
```

- **Malign (1) Recall = 0.52** → Modelin gerçek kanserlerin sadece %52’sini yakaladığı anlamına gelir. **Kötü**.
- **Macro avg F1** → Sınıf büyüklüğünden bağımsız, adil özet. Dengesiz veride `accuracy`’dan çok daha anlamlıdır.

### 10.2. Confusion Matrix

```
             Pred Benign(0)  Pred Malign(1)
True Benign        780            26
True Malign         94            102
```

- **FN = 94** → 94 kanser hastası sağlıklı dediğimiz için kaçırıldı. **Hayati risk.**
- Federe + DCGAN sonrası FN’in düşmesi bizim asıl başarı kriterimiz.

### 10.3. Proje başarısı özetlemesi

| Metrik | Baseline (Adım 4) | Federated (Adım 9) | İyileşme |
|---|---|---|---|
| Malign Recall | `X%` | `Y%` | `+ΔY` |
| Kaçırılan Kanser (FN) | `N1` | `N2` | `−(N1−N2)` |
| Macro-F1 | `F1_a` | `F1_b` | `+ΔF1` |

Bu tablo sunumun özü olacak.

---

## 11. Sık Karşılaşılan Hatalar ve Çözümleri

- **`FileNotFoundError: HAM10000_metadata.csv`** → Yol yanlış. Tüm betikler proje kökünden çalıştırılmalı: `python src/xxx.py` şeklinde.
- **`ValueError: beklenmeyen dx sınıfları`** → CSV manipüle edildi. Orijinal `HAM10000_metadata.csv` kullan.
- **`CUDA out of memory`** → `batch_size` düşür, `num_workers` azalt, arka plandaki başka süreçleri kapat.
- **GAN LossD → 0, LossG → 50+** → Mode collapse. `train_gan.py` zaten TTUR + label smoothing + Dropout ile düzeltildi; ama sorun sürerse öğrenme oranlarını daha da küçült.
- **Flower bağlantısı kurulamıyor** → `server.py` önce başlamalı, 5 saniye bekle, sonra istemciler.
- **Sadece 3–4 istemci bağlandı, raund başlamıyor** → `min_available_clients=5` bekliyor. Kalan istemciyi aç veya ayarı düşür (gerçek projede düşürülmez).
- **Windows’ta `num_workers>0` deadlock** → Kod zaten Windows’ta otomatik 0’a düşürüyor. Elle değiştirildiyse geri al.
- **PowerShell betiği çalışmıyor** → `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ile bir kez aç.

---

## 12. Jüriden Gelebilecek Sorular ve Cevapları

- **"Neden DCGAN? Başka augmentasyon yapamaz mıydın?"**
  - Geleneksel flip/rotate gibi augmentasyonlar **aynı** örneğin varyantını üretir; yeni bilgi taşımaz. DCGAN, eğitim setindeki Malign dağılımından **öğrenerek** yeni örnekler üretir; özellik çeşitliliğini artırır.
- **"Sentetik veri teste sızmış olabilir mi?"**
  - Hayır. Test seti, `HAM10000Dataset` üzerinde `random_split(seed=42)` ile oluşturulan **3. parçadır** ve hiçbir betik içinde değiştirilmez. Sentetik veriler yalnızca federated train CSV’lerine girer.
- **"Federated neden? Tek makinede de eğitebilirdin."**
  - Evet ama o durumda hastanelerin veri paylaşması gerekirdi (KVKK ihlali). FL, veriyi yerinde tutup sadece ağırlıkları iletir; gerçek dünya problemine uygundur.
- **"FedAvg dışında başka strateji denemedin mi?"**
  - Hayır; FedAvg sade ve karşılaştırma için referans algoritmadır. `FedProx` veya `FedAdam` gelecek iş olarak not edildi.
- **"Baseline’da neden class weight yok?"**
  - Çünkü problemi olduğu gibi göstermek istedik. Class weight eklersek baseline yapay olarak yükselir, proje katkısı zayıflar. Baseline = "iyileştirmeden önceki ham hal".
- **"Neden ResNet50? MobileNet/EfficientNet daha iyi değil mi?"**
  - ResNet50 dermatoloji benchmark’larında kanıtlı ve pretrained ağırlıkları güçlü. Federated senaryoda parametre sayısı makul ve ağ üzerinden taşımada pratik.
- **"Recall’a bu kadar odaklanmak precision’ı düşürmez mi?"**
  - Bir miktar düşebilir. Ama klinik olarak kanseri kaçırmak (FN), sağlıklıya yanlış pozitif demekten (FP) çok daha tehlikelidir. Bu yüzden Recall bilinçli bir önceliktir.

---

## 📌 Son Söz

Bu rehber, projedeki her dosyanın "neden var, neyi yapıyor, nasıl yapıyor" sorularını cevaplamak için yazıldı. Yeni bir takım arkadaşı geldiğinde ona bu dosyayı okutmak, kodu baştan anlatmaktan çok daha hızlı olacaktır. Sunum sırasında kullanılacak tüm sayısal karşılaştırmalar için **yalnızca `evaluate.py` (baseline) ve `evaluate_federated.py` (federated) çıktıları** referans alınmalıdır. Diğer tüm betikler bu iki karşılaştırmanın altyapısını hazırlar.
