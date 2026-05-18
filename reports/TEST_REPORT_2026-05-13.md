# Test Veri Raporu (13 Mayıs 2026)

## Amaç
Bu rapor, test seti üzerindeki güncel model performansını karşılaştırmalı olarak sunar.

## Kapsam
- Round 2 (Current run): Yeni başlatılan federated koşunun erken aşaması
- Round 12 (Reference): Sunum için referans alınan checkpoint
- Thesis Optimized (Round 3): deployment_config.json içindeki optimize profil

## Test Metrikleri

| Konfigürasyon | Accuracy | Macro F1 | Malign Recall | Malign Precision | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 2 (Current run) | 0.2606 | 0.2478 | 0.9950 | 0.2093 | 66 | 748 | 1 | 198 |
| Round 12 (Reference) | 0.8381 | 0.7493 | 0.6181 | 0.5829 | 726 | 88 | 76 | 123 |
| Thesis Optimized (Round 3) | 0.6756 | 0.6391 | 0.8775 | 0.3737 | 498 | 300 | 25 | 179 |

## Yorum
1. Round 12, genel doğruluk ve dengeli performans açısından en iyi referans noktadır.
2. Current Run Round 2, eğitim erken aşamada olduğu için yüksek false positive üretmektedir.
3. Thesis Optimized (Round 3), malign recall tarafında güçlü fakat precision tarafında geliştirmeye açıktır.

## Görseller
- [test_metrics_comparison.png](figures/test_metrics_comparison.png)
- [confusion_components_comparison.png](figures/confusion_components_comparison.png)
- [round12_confusion_matrix_reference.png](figures/round12_confusion_matrix_reference.png)

## Sonuç
Sunum ve yayın senaryosu için Round 12 (Reference) metriklerinin baz alınması önerilir.
