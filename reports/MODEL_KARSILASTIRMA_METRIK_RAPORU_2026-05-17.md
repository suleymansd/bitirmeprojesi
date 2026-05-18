# Model Karsilastirma, Metrik ve Confusion Matrix Raporu

## 1. Raporun Amaci

Bu raporda cilt kanseri teshisi projesinde egitilen modellerin test sonuclari karsilastirilmistir. Degerlendirme HAM10000 test verisi uzerinden yapilmistir. Modeller benign ve malign olmak uzere iki sinifli siniflandirma yapmaktadir.

Bu calismada tek basina accuracy degeri yeterli gorulmemistir. Cilt kanseri teshisi gibi klinik risk tasiyan problemlerde malign vakanin benign olarak tahmin edilmesi, yani false negative hatasi, en kritik hata turudur. Bu nedenle degerlendirmede ozellikle malign recall ve false negative sayisi dikkate alinmistir.

## 2. Karsilastirilan Modeller

Bu raporda uc model profili karsilastirilmistir:

| Model | Aciklama |
|---|---|
| Merkezi Baseline | Tum egitim verisine merkezi olarak erisen ResNet50 modeli |
| Recall Odakli Federe Model | Malign vakalari kacirmamaya oncelik veren federe model |
| Dengeli Federe Round 16 | Accuracy ve macro F1 degerlerini artiran alternatif federe model |

Merkezi baseline modeli, federe ogrenme modelleri icin referans karsilastirma noktasi olarak kullanilmistir. Federe modeller ise verinin merkezi olarak toplanmadigi, istemcilerde lokal egitim yapildigi mahremiyet odakli senaryoyu temsil etmektedir.

## 3. Test Metrikleri

| Metrik | Merkezi Baseline | Recall Odakli Federe Model | Dengeli Federe Round 16 |
|---|---:|---:|---:|
| Accuracy | 0.9585 | 0.6756 | 0.7147 |
| Macro F1 | 0.9353 | 0.6391 | 0.6614 |
| Balanced Accuracy | 0.9419 | 0.7527 | 0.7503 |
| Malign Precision | 0.8792 | 0.3737 | 0.3908 |
| Malign Recall | 0.9146 | 0.8775 | 0.8090 |
| False Negative | 17 | 25 | 38 |
| True Positive | 182 | 179 | 161 |

Not: Baseline icin balanced accuracy, benign recall ve malign recall ortalamasi olarak hesaplanmistir:

```text
Balanced Accuracy = (0.9693 + 0.9146) / 2 = 0.9419
```

## 4. Merkezi Baseline Model Confusion Matrix

Merkezi baseline modelin test setindeki confusion matrix sonucu:

```text
                 Tahmin Benign   Tahmin Malign
Gercek Benign        789              25
Gercek Malign         17             182
```

| Deger | Anlam | Sayi |
|---|---|---:|
| TN | Benign olup benign tahmin edilen | 789 |
| FP | Benign olup malign tahmin edilen | 25 |
| FN | Malign olup benign tahmin edilen | 17 |
| TP | Malign olup malign tahmin edilen | 182 |

Baseline modelin malign recall degeri:

```text
Malign Recall = TP / (TP + FN)
Malign Recall = 182 / (182 + 17)
Malign Recall = 0.9146
```

Bu sonuca gore baseline model test setindeki 199 malign vakanin 182 tanesini dogru yakalamis, 17 tanesini kacirmistir.

## 5. Recall Odakli Federe Model Confusion Matrix

Tez kapsaminda gecerliligi en yuksek federe model olarak recall odakli federe model secilmistir. Bu modelin test setindeki confusion matrix sonucu:

```text
                 Tahmin Benign   Tahmin Malign
Gercek Benign        498             300
Gercek Malign         25             179
```

| Deger | Anlam | Sayi |
|---|---|---:|
| TN | Benign olup benign tahmin edilen | 498 |
| FP | Benign olup malign tahmin edilen | 300 |
| FN | Malign olup benign tahmin edilen | 25 |
| TP | Malign olup malign tahmin edilen | 179 |

Recall odakli federe modelin malign recall degeri:

```text
Malign Recall = TP / (TP + FN)
Malign Recall = 179 / (179 + 25)
Malign Recall = 0.8775
```

Bu model 204 malign vakanin 179 tanesini dogru yakalamis, 25 tanesini kacirmistir. False positive sayisi yuksektir; yani model bazi benign vakalari malign olarak isaretlemektedir. Ancak projenin klinik onceligi malign vakalari kacirmamak oldugu icin bu model federe sistem icin daha savunulabilir profil olarak degerlendirilmistir.

## 6. Dengeli Federe Round 16 Confusion Matrix

Iyilestirme denemeleri sonucunda accuracy ve macro F1 degerleri daha yuksek olan alternatif model round 16 olarak belirlenmistir. Bu modelin test setindeki confusion matrix sonucu:

```text
                 Tahmin Benign   Tahmin Malign
Gercek Benign        563             251
Gercek Malign         38             161
```

| Deger | Anlam | Sayi |
|---|---|---:|
| TN | Benign olup benign tahmin edilen | 563 |
| FP | Benign olup malign tahmin edilen | 251 |
| FN | Malign olup benign tahmin edilen | 38 |
| TP | Malign olup malign tahmin edilen | 161 |

Dengeli federe modelin malign recall degeri:

```text
Malign Recall = TP / (TP + FN)
Malign Recall = 161 / (161 + 38)
Malign Recall = 0.8090
```

Bu model accuracy ve macro F1 degerlerini artirmistir. Ancak false negative sayisi 25'ten 38'e cikmistir. Bu nedenle kanseri kacirmamak hedefi acisindan recall odakli federe model kadar uygun degildir.

## 7. Confusion Matrix Karsilastirmasi

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Merkezi Baseline | 789 | 25 | 17 | 182 |
| Recall Odakli Federe Model | 498 | 300 | 25 | 179 |
| Dengeli Federe Round 16 | 563 | 251 | 38 | 161 |

Bu tabloya gore merkezi baseline model hem false negative hem de false positive acisindan en iyi sonucu vermistir. Bunun temel nedeni baseline modelin tum egitim verisine merkezi olarak erisebilmesidir.

Federe modellerde performans dususu gozlenmistir. Bu durum federe ogrenmenin dogal sonucudur. Federe ogrenmede veriler merkezi sunucuya aktarilmaz; istemciler lokal egitim yapar ve global model agirlik ortalamasi ile olusturulur. Bu nedenle merkezi egitime gore bilgi kaybi olusabilir.

## 8. Metriklerin Yorumu

### Accuracy

Accuracy, tum dogru tahminlerin toplam tahminlere oranidir. Baseline modelde accuracy 0.9585 ile oldukca yuksektir. Federe modellerde accuracy daha dusuktur. Dengeli federe round 16 modeli, recall odakli federe modele gore daha yuksek accuracy vermistir.

Ancak accuracy tek basina yeterli degildir. Veri dengesizligi olan saglik problemlerinde model benign sinifi iyi tahmin ederek yuksek accuracy elde edebilir, fakat malign vakalari kacirabilir.

### Macro F1

Macro F1, siniflarin F1 skorlarini esit agirlikla ortalar. Bu nedenle dengesiz veri setlerinde accuracy'den daha adil bir metriktir. Dengeli federe round 16 modeli macro F1 acisindan recall odakli federe modelden daha iyi sonuc vermistir.

### Malign Recall

Malign recall, gercek malign vakalarin ne kadarinin dogru malign olarak tahmin edildigini gosterir. Bu proje icin en kritik metriklerden biridir.

| Model | Malign Recall |
|---|---:|
| Merkezi Baseline | 0.9146 |
| Recall Odakli Federe Model | 0.8775 |
| Dengeli Federe Round 16 | 0.8090 |

Bu tabloya gore recall odakli federe model, dengeli federe modele gore malign vakalari daha iyi yakalamaktadir.

### False Negative

False negative, malign bir vakanin benign olarak tahmin edilmesidir. Klinik acidan en riskli hata turudur.

| Model | False Negative |
|---|---:|
| Merkezi Baseline | 17 |
| Recall Odakli Federe Model | 25 |
| Dengeli Federe Round 16 | 38 |

Bu nedenle nihai federe model seciminde false negative sayisi daha dusuk olan recall odakli federe model tercih edilmistir.

## 9. Genel Degerlendirme

Merkezi baseline model en yuksek performansi vermistir. Bunun nedeni modelin tum veriye merkezi olarak erisebilmesidir. Bu sonuc teknik olarak beklenen bir durumdur. Ancak saglik verilerinde tum verinin merkezi bir noktada toplanmasi gizlilik ve mahremiyet acisindan her zaman mumkun degildir.

Federe ogrenme modelleri merkezi modele gore daha dusuk performans vermistir. Buna karsin federe ogrenme, hasta verilerinin kurumlar disina cikmadan model egitimine katilmasini saglamaktadir. Bu nedenle federe ogrenme, performans ve mahremiyet arasinda bir denge sunmaktadir.

Federe modeller arasinda iki farkli profil gozlenmistir:

- Recall odakli model, malign vakalari daha az kacirmaktadir.
- Dengeli model, accuracy ve macro F1 degerlerini artirmakta ancak daha fazla malign vakayi kacirmaktadir.

Bu tezde cilt kanseri teshisinde malign vakayi kacirmamak onceliklendirildigi icin recall odakli federe model gecerli federe model olarak secilmistir.

## 10. Tezde Kullanilabilecek Sonuc Paragrafi

Deneysel sonuclara gore merkezi baseline model, accuracy, macro F1 ve malign recall metriklerinde en yuksek performansi elde etmistir. Baseline model test setinde 0.9585 accuracy, 0.9353 macro F1 ve 0.9146 malign recall degerlerine ulasmis, 17 malign vakayi kacirmistir. Bu sonuc, tum veriye merkezi olarak erisebilen modellerin performans acisindan avantajli oldugunu gostermektedir.

Federe ogrenme modellerinde merkezi modele gore performans dususu gozlenmistir. Ancak federe ogrenme yaklasimi, hasta verilerinin merkezi bir ortama aktarilmasina gerek kalmadan egitim yapilabilmesini saglamaktadir. Bu nedenle federe ogrenme, saglik verilerinde mahremiyetin korunmasi acisindan onemli bir alternatif sunmaktadir.

Federe modeller arasinda recall odakli model, 0.8775 malign recall ve 25 false negative degeriyle malign vakalari daha az kacirdigi icin nihai federe model olarak tercih edilmistir. Dengeli round 16 modeli daha yuksek accuracy ve macro F1 degerleri uretmesine ragmen false negative sayisini 38'e cikardigi icin klinik oncelik acisindan ana model olarak secilmemistir. Bu nedenle model seciminde yalnizca genel dogruluk degil, problemin klinik riski olan kanser vakasini kacirmama kriteri temel alinmistir.
