# Cilt Kanseri Teshisi Projesi Egitim Sureci ve Sonuc Raporu

## 1. Projenin Amaci

Bu calismada HAM10000 veri seti kullanilarak cilt lezyonlarinin ikili siniflandirilmasi hedeflenmistir. Siniflar benign ve malign olarak ele alinmistir. Projenin temel amaci, cilt kanseri teshisinde karar destek sistemi olarak kullanilabilecek bir derin ogrenme modeli gelistirmek ve bunu hasta verisinin merkezi bir yerde toplanmasina gerek kalmadan federe ogrenme yaklasimi ile egitmektir.

Saglik verilerinde hasta mahremiyeti kritik oldugu icin klasik merkezi egitim yaklasimi her zaman uygulanabilir degildir. Bu nedenle projede federe ogrenme tercih edilmistir. Federe ogrenme sayesinde veriler istemcilerde kalmakta, merkezi sunucuya yalnizca model agirliklari veya model guncellemeleri aktarilmaktadir. Bu yontem, veri gizliligini korurken ortak bir global model egitilmesini saglamaktadir.

Projede ikinci onemli problem sinif dengesizligidir. HAM10000 veri setinde benign ornek sayisi malign orneklere gore daha fazladir. Bu durum modelin benign sinifa egilim gostermesine ve malign vakalari kacirma riskinin artmasina neden olabilir. Bu problemi azaltmak icin DCGAN ile sentetik malign goruntuler uretilmis ve federe egitim surecinde malign sinif desteklenmistir.

## 2. Kullanilan Yontemler

Projede kullanilan temel yontemler sunlardir:

- ResNet50 tabanli ikili siniflandirma modeli
- HAM10000 veri seti
- DCGAN ile sentetik malign goruntu uretimi
- Federe ogrenme yaklasimi
- FedAvg agirlik birlestirme algoritmasi
- Class-weighted loss ve focal loss denemeleri
- Threshold optimizasyonu
- Accuracy, precision, recall, macro F1, balanced accuracy ve confusion matrix ile degerlendirme

Model mimarisi olarak ResNet50 tercih edilmistir. ResNet50, goruntu siniflandirma problemlerinde yaygin kullanilan, derin katmanli ve kararliligi yuksek bir konvolusyonel sinir agidir. Projede son katman ikili siniflandirma yapacak sekilde duzenlenmistir.

## 3. Veri Hazirlama Sureci

HAM10000 veri setindeki lezyonlar benign ve malign olmak uzere iki ana sinifa indirgenmistir. Veri seti egitim, dogrulama ve test olarak ayrilmistir. Test seti, model egitiminde veya threshold seciminde kullanilmamistir. Bu sayede nihai test sonuclarinin egitim surecinden bagimsiz olmasi saglanmistir.

Veri on isleme asamasinda goruntuler once kare forma getirilmis, ardindan ResNet50 modeline uygun boyuta yeniden olceklendirilmis ve ImageNet normalizasyon degerleriyle normalize edilmistir. Bu islem hem gercek HAM10000 goruntuleri hem de sentetik goruntuler icin ayni sekilde uygulanmistir.

Federe ogrenme senaryosunda veri iki istemciye dagitilmistir. Her istemci kendi lokal verisi uzerinde egitim yapmis, ardindan model agirliklari global model olusturmak icin birlestirilmistir.

## 4. Baseline Egitim

Ilk asamada merkezi bir baseline model egitilmistir. Baseline modelin amaci, federe ogrenme yaklasimi icin karsilastirma noktasi olusturmaktir. Bu asamada model merkezi veri uzerinden egitilmis ve test setinde degerlendirilerek temel performans seviyesi belirlenmistir.

Baseline egitim, projenin ilerleyen asamalarinda federe ogrenme ve sentetik veri katkisini yorumlamak icin referans olarak kullanilmistir.

## 5. DCGAN ile Sentetik Veri Uretimi

Veri setindeki malign sinif az oldugu icin modelin malign vakalari kacirma riski bulunmaktadir. Bu nedenle DCGAN ile sentetik malign goruntuler uretilmistir. Sentetik verinin amaci gercek verinin yerini almak degil, egitim sirasinda malign sinifin temsilini guclendirmektir.

Sentetik goruntuler yalnizca egitim surecinde kullanilmistir. Test degerlendirmesi gercek HAM10000 test verileri uzerinden yapilmistir. Bu nokta metodolojik olarak onemlidir, cunku modelin basarisi sentetik test verisi uzerinden degil, gercek veri uzerinden olculmustur.

## 6. Federe Ogrenme Egitim Sureci

Federe egitimde FedAvg algoritmasi kullanilmistir. Her round icinde istemciler global model agirliklarini almis, kendi lokal verileri uzerinde egitim yapmis ve egitim sonunda guncellenmis agirliklari sunucuya gondermistir. Sunucu, istemcilerden gelen agirliklari ornek sayilarina gore ortalayarak yeni global modeli olusturmustur.

Egitim surecinde round bazli checkpointler kaydedilmistir. Bu sayede her round sonunda olusan global model ayri ayri degerlendirilebilir hale getirilmistir. Egitim sonunda yalnizca son round degil, tum roundlar arasinda en uygun model secilmistir. Bu yaklasim onemlidir, cunku federe ogrenmede son round her zaman en iyi test performansini vermeyebilir.

## 7. Ilk Deployment Modeli ve Sonuclari

Ilk deployment icin recall odakli model secilmistir. Bu modelde klinik risk dikkate alinarak malign vakalari kacirmamak onceliklendirilmiştir. Bu nedenle model secimi sadece accuracy degerine gore yapilmamistir.

Secilen ilk recall odakli modelin test sonuclari asagidaki gibidir:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.6756 |
| Macro F1 | 0.6391 |
| Malign Recall | 0.8775 |
| Malign Precision | 0.3737 |
| True Negative | 498 |
| False Positive | 300 |
| False Negative | 25 |
| True Positive | 179 |

Bu modelin en onemli avantaji malign recall degerinin yuksek olmasidir. Malign recall 0.8775 oldugu icin model malign vakalarin buyuk kismini yakalamaktadir. False negative sayisi 25 olarak olculmustur. Cilt kanseri teshisi gibi bir problemde false negative, yani malign bir vakanin benign olarak tahmin edilmesi en kritik hata turudur. Bu nedenle bu model klinik hassasiyet acisindan guclu bir aday olarak degerlendirilmistir.

## 8. Iyilestirme Denemeleri

Daha yuksek genel performans elde etmek icin ek iyilestirme denemeleri yapilmistir. Bu asamada egitim surecine daha guclu augmentation, weighted focal loss ve threshold optimizasyonu eklenmistir.

Uygulanan iyilestirmeler sunlardir:

- Random resized crop
- Horizontal ve vertical flip
- Rotation
- Color jitter
- Weighted focal loss
- Validation set uzerinde threshold taramasi
- Macro F1 ve balanced accuracy odakli model secimi

Bu denemelerde amac accuracy, macro F1 ve balanced accuracy degerlerini yukari cekmekti. Ancak ayni zamanda malign recall degerinin cok fazla dusmemesi hedeflenmistir.

## 9. 5 Round Iyilestirme Denemesi

Ilk hizli iyilestirme denemesi 5 round olarak gerceklestirilmistir. Bu deneme sonucunda en iyi aday round 4 olarak belirlenmistir.

5 round iyilestirme denemesinin test sonuclari:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.6624 |
| Macro F1 | 0.6215 |
| Balanced Accuracy | 0.7330 |
| Malign Recall | 0.8492 |
| Malign Precision | 0.3514 |
| True Negative | 502 |
| False Positive | 312 |
| False Negative | 30 |
| True Positive | 169 |

Bu deneme, egitim stratejisinin calistigini gostermis ancak ilk recall odakli modeli gecememistir. Ozellikle macro F1 ve accuracy degerleri beklenen seviyede iyilesmemis, false negative sayisi 25'ten 30'a cikmistir. Bu nedenle 5 round iyilestirme denemesi ana deployment modeli olarak secilmemistir.

## 10. 20 Round Tam Iyilestirme Egitimi

Daha saglikli bir karsilastirma yapabilmek icin 20 round tam egitim sureci baslatilmistir. Bu egitimde lokal FedAvg simülasyonu kullanilmistir. Yontem olarak yine federe ogrenme mantigi korunmustur: her istemci kendi verisi uzerinde egitilmis, ardindan agirliklar FedAvg ile birlestirilmistir.

20 round egitim surecinde loss degerleri genel olarak dusmustur. Bu, modelin egitim verisi uzerinde daha iyi ogrenmeye basladigini gostermektedir.

Ornek loss degisimleri:

| Round | Client 1 Loss | Client 2 Loss |
|---:|---:|---:|
| 1 | 0.2423 | 0.2406 |
| 5 | 0.1892 | 0.1946 |
| 10 | 0.1732 | 0.1778 |
| 14 | 0.1645 | 0.1680 |
| 16 | 0.1619 | 0.1616 |

Egitim 17. roundda kesintiye ugramistir. Ancak 16. rounda kadar checkpointler basarili sekilde kaydedilmistir. Bu nedenle round 1-16 arasindaki tum modeller validation ve test metrikleriyle degerlendirilmistir.

## 11. Round 1-16 Metrik Taramasi

Round 1-16 arasindaki checkpointler icin validation seti uzerinde threshold optimizasyonu yapilmistir. Model seciminde macro F1, balanced accuracy, malign recall ve malign precision birlikte dikkate alinmistir.

Bu tarama sonucunda en iyi dengeli model round 16 olarak belirlenmistir.

Round 16 validation sonuclari:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.7257 |
| Balanced Accuracy | 0.7540 |
| Malign Recall | 0.8000 |
| Malign Precision | 0.3948 |
| Macro F1 | 0.6676 |
| True Negative | 565 |
| False Positive | 233 |
| False Negative | 38 |
| True Positive | 152 |

Round 16 test sonuclari:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.7147 |
| Balanced Accuracy | 0.7503 |
| Malign Recall | 0.8090 |
| Malign Precision | 0.3908 |
| Macro F1 | 0.6614 |
| True Negative | 563 |
| False Positive | 251 |
| False Negative | 38 |
| True Positive | 161 |

Round 16 modeli, accuracy ve macro F1 acisindan ilk deployment modelinden daha iyi sonuc vermistir. Ancak malign recall degeri dusmus ve false negative sayisi artmistir.

## 12. Recall Odakli Model ile Dengeli Modelin Karsilastirmasi

Iki ana model karsilastirildiginda su tablo elde edilmistir:

| Metrik | Recall Odakli Model | Dengeli Round 16 Model |
|---|---:|---:|
| Accuracy | 0.6756 | 0.7147 |
| Macro F1 | 0.6391 | 0.6614 |
| Malign Recall | 0.8775 | 0.8090 |
| Malign Precision | 0.3737 | 0.3908 |
| False Negative | 25 | 38 |
| True Positive | 179 | 161 |

Dengeli Round 16 modeli accuracy ve macro F1 degerlerini artirmistir. Buna karsilik malign recall degeri 0.8775'ten 0.8090'a dusmus, false negative sayisi 25'ten 38'e cikmistir.

Bu proje kapsaminda cilt kanseri teshisinde en kritik hata malign bir vakanin benign olarak tahmin edilmesidir. Bu nedenle sadece accuracy degerine gore model secimi yapmak uygun degildir. Klinik risk acisindan false negative sayisini dusuk tutmak daha onemlidir.

## 13. Nihai Model Secimi

Projenin amaci kanser vakalarini mumkun oldugunca kacirmamak oldugu icin nihai model olarak recall odakli modelin kullanilmasi daha uygun gorulmustur.

Nihai model secimi:

| Ozellik | Deger |
|---|---|
| Model profili | Recall odakli model |
| Checkpoint | `checkpoints/federated/federated_model_round_12_fp16.pth` |
| Threshold | 0.32 |
| Test Accuracy | 0.6756 |
| Test Macro F1 | 0.6391 |
| Test Malign Recall | 0.8775 |
| Test Malign Precision | 0.3737 |
| False Negative | 25 |

Bu secim, klinik acidan daha savunulabilir bir yaklasimdir. Accuracy degeri daha yuksek olan model bulunmasina ragmen, bu model daha fazla malign vakayi kacirdigi icin ana sistem modeli olarak tercih edilmemistir.

## 14. Sonuc ve Degerlendirme

Bu proje kapsaminda HAM10000 veri seti uzerinde cilt lezyonlarinin benign ve malign olarak siniflandirilmesi icin federe ogrenme tabanli bir sistem gelistirilmistir. Veri gizliligi problemi federe ogrenme ile ele alinmis, sinif dengesizligi problemi ise DCGAN ile sentetik malign veri uretimi ve agirliklandirilmis egitim yontemleriyle desteklenmistir.

Egitim surecinde farkli roundlar, farkli threshold degerleri ve farkli optimizasyon stratejileri denenmistir. Denemeler sonucunda iki farkli model profili ortaya cikmistir:

- Recall odakli model: malign vakalari daha az kacirmaktadir.
- Dengeli model: accuracy ve macro F1 degerlerini artirmakta ancak daha fazla malign vakayi kacirmaktadir.

Cilt kanseri teshisinde false negative hatasi klinik acidan en kritik hata oldugu icin nihai sistemde recall odakli modelin kullanilmasi daha uygun bulunmustur. Bu karar, model seciminin yalnizca genel accuracy degerine gore degil, problemin klinik riskleri dikkate alinarak yapildigini gostermektedir.

## 15. Tezde Kullanilabilecek Kisa Ozet

Bu calismada cilt kanseri teshisi icin HAM10000 veri seti uzerinde ResNet50 tabanli bir siniflandirma modeli gelistirilmistir. Hasta verilerinin merkezi ortamda toplanmasinin gizlilik acisindan riskli olmasi nedeniyle federe ogrenme yaklasimi kullanilmistir. Veri setindeki benign-malign dengesizligini azaltmak amaciyla DCGAN ile sentetik malign goruntuler uretilmis ve egitim surecinde kullanilmistir.

Egitim surecinde farkli round checkpointleri kaydedilmis, her checkpoint validation ve test metrikleriyle degerlendirilmistir. Accuracy, macro F1, balanced accuracy, malign precision ve malign recall metrikleri birlikte incelenmistir. Genel accuracy ve macro F1 degerlerini artiran alternatif bir model elde edilmesine ragmen, bu modelin malign recall degerini dusurdugu ve false negative sayisini artirdigi gorulmustur. Cilt kanseri teshisinde malign vakayi kacirmamak daha kritik oldugu icin nihai model seciminde malign recall onceliklendirilmis ve false negative sayisi daha dusuk olan recall odakli model tercih edilmistir.

Sonuc olarak, gelistirilen sistem hasta verisini merkezilestirmeden egitim yapabilen, sentetik veri ile sinif dengesizligini azaltmayi hedefleyen ve malign vakalari yakalamaya oncelik veren bir karar destek prototipi olarak ortaya konmustur. Sistem klinik karar yerine gecmemekte, arastirma ve demo amacli bir karar destek yaklasimi sunmaktadir.
