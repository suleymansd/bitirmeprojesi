# Cilt Kanseri Teşhisi — DCGAN Destekli Federe Öğrenme

Lisans bitirme projesi. HAM10000 veri setinde **ikili (Malign / Benign) cilt kanseri sınıflandırması** üzerinden, merkezî bir baseline ile **DCGAN ile üretilmiş sentetik Malign görüntülerle zenginleştirilmiş 5 istemcili federe öğrenme (FedAvg)** yaklaşımının karşılaştırılması.

## Motivasyon

Gerçek dünyada hastaneler hasta görüntülerini gizlilik nedeniyle merkezî bir sunucuya gönderemez. Federe öğrenme bunu çözer ama veri dengesizliği (Benign >> Malign) tanıyı zorlaştırır. Bu proje, **dengesizliği DCGAN sentetik örnekleriyle kapatmanın** federe kurulumdaki etkisini ölçer.

## Pipeline Özeti

```
HAM10000 (7,470 gerçek) ──► Baseline ResNet50 (merkezî, dengesiz)
        │
        └──► DCGAN (sadece Malign) ──► 6,450 sentetik Malign
                                          │
                         Gerçek + Sentetik ──► 5 istemciye IID dağıtım
                                          │
                                          ▼
                               Flower FedAvg (5 raund)
                                          │
                                          ▼
                        Nihai global model — aynı %10 test setinde ölçülür
```

Detaylı anlatım: [`PROJE_REHBERI.md`](PROJE_REHBERI.md). Klasör yapısı: [`klasor_mimarisi.txt`](klasor_mimarisi.txt).

## Kurulum

Python 3.11 gerekli. Detaylı adımlar ve hangi `requirements` dosyasını kullanacağın [`CLAUDE.md`](CLAUDE.md)'de:

- **GPU'da eğitim yapacaksan:** `requirements.txt` + uygun CUDA wheel
- **Sadece kod geliştireceksen:** `requirements-dev.txt` + CPU PyTorch

Ham veri git'e dahil değil — HAM10000'i Kaggle'dan indirip `data/raw/` altına koyman gerekiyor.

## Çalıştırma

```bash
# 1. Baseline
python src/train_baseline.py
python src/evaluate.py

# 2. DCGAN + sentetik üretim
python src/train_gan.py
python src/generate_synthetic.py

# 3. Federe split
python src/prepare_federated_splits.py

# 4. Federe eğitim (ayrı terminallerde)
python src/server.py
python src/client.py --client_id 1    # ... 2, 3, 4, 5

# 5. Nihai değerlendirme
python src/evaluate_federated.py
```

Windows için `run_federated.ps1` tek komutla sunucu + 5 istemciyi ayrı pencerelerde başlatır.

## Sonuç Görselleri

- `federe_sunucu_egitimi.png` — Sunucu tarafı eğitim logu
- `federe_dagitim_raporu.png` — 5 istemciye dağıtım dengesi
- `federe_sonuc_raporu.png` — Nihai global modelin test performansı

## Teknoloji Yığını

- **PyTorch 2.7.1** (ResNet50 + DCGAN)
- **Flower 1.29** (federe öğrenme — FedAvg)
- **scikit-learn** (metrikler)
- AMP (mixed precision) ile GPU belleği optimizasyonu

## Lisans & Veri

HAM10000 Harvard Dataverse lisansı altındadır (akademik kullanım). Bu kod, lisans bitirme tezi kapsamında açılmıştır.

## Live Demo Arayüzü (FastAPI + Web UI)

Bu proje için profesyonel demo arayüzü eklendi.

### Çalıştırma

```bash
source .venv/bin/activate
pip install -r requirements.txt
./run_demo.sh
```

Arayüz: `http://localhost:8090`

### Model Seçimi

- Varsayılan: en son federated checkpoint
- Baseline kullanmak için:

```bash
MODEL_MODE=baseline ./run_demo.sh
```

- Belirli federated round seçmek için:

```bash
FEDERATED_ROUND=10 ./run_demo.sh
```

### API

- `GET /api/health`
- `POST /api/predict?threshold=0.5` (multipart `file`)

Not: Çıktılar araştırma/demo amaçlıdır, klinik karar yerine geçmez.
