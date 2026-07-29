# ClinVar Varyant Patojenite Sınıflandırıcısı

Genetik varyantların (DNA'daki tekli nükleotid değişikliklerinin) hastalığa yol açma olasılığını (patojenite) tahmin eden bir makine öğrenmesi projesi. ClinVar veritabanından alınan yüksek kaliteli etiketli varyantlarla eğitilmiş bir XGBoost modeli, biyolojik olarak anlamlı üç feature (CADD, phyloP, gnomAD popülasyon frekansı) kullanarak Benign/Pathogenic ayrımı yapıyor — ve bu modeli, klinik pratikte hâlâ "belirsiz" (Uncertain Significance, VUS) olarak sınıflandırılmış \~294.000 varyanta uyguluyor.

İnteraktif bir Streamlit demosu ile hem manuel feature girişi hem de gerçek zamanlı varyant sorgusu (myvariant.info API üzerinden) yapılabiliyor, her tahmin SHAP ile açıklanıyor.

## Neden bu proje

Klinik genetikte en büyük pratik sorunlardan biri "Variant of Uncertain Significance" (VUS) yığılması — laboratuvarların, hastanın genomunda bulduğu ama hastalıkla ilişkisini kesin olarak belirleyemediği binlerce varyant. Bu proje, VUS'ların bir kısmı için makine öğrenmesi ile yön gösterici bir olasılık skoru üretmeyi, ve bunu şeffaf/açıklanabilir bir şekilde (SHAP) sunmayı amaçlıyor.

## Veri

* **Kaynak**: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) `variant\_summary.txt` (GRCh38)
* **Etiket kalitesi filtresi**: Sadece 2+ yıldız review status'a sahip varyantlar (`criteria provided, multiple submitters, no conflicts`, `reviewed by expert panel`, `practice guideline`) — düşük güvenilirlikli tekil-submitter etiketleri dışlandı
* **Eğitim seti**: 239.734 Benign/Pathogenic etiketli varyant (157.528 benign / 82.206 pathogenic)
* **VUS seti**: Uncertain Significance etiketli, ayrı tutulan ve modelin sonunda skorladığı varyantlar

### Kapsam kararı: sadece coding-region varyantları

Feature'ların kaynağı olan dbNSFP, yalnızca protein değiştiren (missense/nonsense) ve splice-site varyantlarını kapsıyor — synonymous ve intronic varyantlar için veri sağlamıyor. Bu nedenle proje, bilinçli olarak **coding-region varyantları** ile sınırlandırıldı. Bu, CADD/REVEL/PolyPhen gibi yerleşik pathogenicity araçlarının da kullandığı standart bir kapsam. Final eğitim seti: **79.586 SNV** (48k pathogenic / 31k benign — coding-only filtrede oran tersine döndü, çünkü sessiz/intronik varyantlar ağırlıklı olarak benign etiketlenmiş).

## Feature'lar

myvariant.info API'sinden (assembly=hg38) çekilen üç feature:

|Feature|Açıklama|
|-|-|
|`cadd\_phred`|CADD skoru — varyantın genel zararlılık tahmini|
|`phylop\_100way`|phyloP konservasyon skoru — pozisyonun evrimsel olarak ne kadar korunduğu|
|`gnomad\_af\_log`|gnomAD popülasyon frekansının log10 dönüşümü — popülasyonda görülmeme durumu 0 ile dolduruldu|

GERP skoru, phyloP ile yüksek redundancy nedeniyle (ikisi de konservasyon ölçüyor) modele dahil edilmedi.

## EDA bulguları

* Her üç feature de Benign/Pathogenic sınıflarını görsel olarak net ayırıyor (bkz. notebook)
* `gnomad\_af\_log`: Pathogenic varyantlar neredeyse tamamen "popülasyonda görülmemiş" ucunda yığılıyor; Benign varyantlar geniş bir frekans aralığına yayılıyor
* `cadd\_phred` ile `phylop\_100way` arasında orta düzey korelasyon (r=0.54) — CADD'in kendi hesaplamasında phyloP'yi zaten girdi olarak kullanmasından kaynaklanıyor

## Model seçimi

12 algoritma, 5-fold cross-validation ile ROC-AUC üzerinden karşılaştırıldı:

|Model|ROC-AUC (mean ± std)|
|-|-|
|**XGBoost**|**0.9968 ± 0.0002**|
|**LightGBM**|**0.9968 ± 0.0002**|
|Gradient Boosting|0.9965 ± 0.0002|
|AdaBoost|0.9965 ± 0.0002|
|Random Forest|0.9952 ± 0.0004|
|Logistic Regression (L1/L2)|0.9947|
|Linear SVM|0.9947 ± 0.0004|
|Ridge Classifier|0.9936 ± 0.0004|
|KNN|0.9933 ± 0.0004|
|Decision Tree|0.9923 ± 0.0008|
|Naive Bayes|0.9921 ± 0.0004|

XGBoost seçildi (LightGBM ile istatistiksel olarak eşit performans, SHAP entegrasyonu göz önünde bulunduruldu). `RandomizedSearchCV` (30 iterasyon, 5-fold) ile hiperparametre tuning yapıldı, CV skoru 0.9968 → 0.9969'a marjinal olarak iyileşti — bu, baseline modelin zaten performans tavanına yakın olduğunu gösteriyor.

**Final model — hold-out test seti sonucu: 0.9962 ROC-AUC**

## Model yorumlama (SHAP)

SHAP analizi, feature önem sıralamasını doğruluyor:

1. **`gnomad\_af\_log`** — en etkili feature; düşük/sıfır popülasyon frekansı güçlü şekilde pathogenic yönüne itiyor
2. **`cadd\_phred`** — ikinci en etkili; yüksek skor pathogenic yönünde
3. **`phylop\_100way`** — CADD ile collinearity nedeniyle marjinal katkı

## Önemli metodolojik not: circularity riski

CADD gibi tahmin araçları, kendi eğitim verilerinde dolaylı olarak ClinVar etiketleriyle örtüşen kaynaklar kullanmış olabilir (literatürde "type 1 circularity" olarak bilinen bir sorun). Elde edilen yüksek ROC-AUC, kısmen bu örtüşmeden besleniyor olabilir — bu, sonucun geçersiz olduğu anlamına gelmiyor, ama modelin sınırlarını değerlendirirken göz önünde bulundurulmalı.

## VUS skorlama

Final model, 293.971 VUS varyantına uygulandı:

* **\~%35'i** >0.9 pathogenic olasılığıyla, **\~%15'i** <0.1 pathogenic olasılığıyla (yani yüksek güvenle benign) skorlandı — VUS'ların kabaca yarısı için model güçlü bir yön işareti veriyor
* En yüksek pathogenic olasılıklı VUS'lar arasında bilinen klinik öneme sahip genler var: **MYH7** (hipertrofik kardiyomiyopati), **EGFR** (kanser genetiği), **CDH1** (kalıtsal mide kanseri riski) — modelin biyolojik olarak anlamlı bir sinyal yakaladığının bir işareti

## Streamlit Demo (`clinvar\_app/`)

Dört sekmeli interaktif uygulama:

1. **Manuel Feature Girişi** — CADD/phyloP/gnomAD değerlerini elle girip anlık tahmin + SHAP açıklaması alma
2. **Varyant Ara (HGVS ID)** — myvariant.info API'sine canlı sorgu atarak gerçek bir varyantı ID ile arama (hazır örnek butonlarıyla)
3. **VUS Gezgini** — skorlanmış VUS listesini gen adına göre filtreleyip inceleme
4. **Model Performansı** — 12 algoritma karşılaştırması, confusion matrix, ROC eğrisi, genel SHAP özeti

### Çalıştırma

```bash
cd clinvar\_app
pip install streamlit xgboost pandas numpy scikit-learn shap requests plotly
streamlit run app.py
```

## Kullanılan araçlar

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, myvariant.info API

