# ClinVar SNV araştırma ve önceliklendirme

GRCh38 ClinVar SNV'lerinin etiketlerini CADD, phyloP ve gnomAD frekansı ile inceleyen, tekrarlanabilir bir araştırma paketi ve Streamlit demosu. Çıktı **model önceliklendirme skorudur**; hasta hastalık riski, penetrans, ACMG/AMP sınıfı veya VUS yeniden sınıflandırma kararı değildir.

V2; açık etiket politikası, kayıp AF'nin korunması, gen bağlantılarını dikkate alan ayrımlar, yalnızca geliştirme verisinde model seçimi, ayrı kalibrasyon ve tarihli dış değerlendirme kullanır. Eski `0.9962` AUC, model seçimine giren eski test verisine aittir; yeni sonucun yerine gösterilmez.

## Kurulum ve çalıştırma

Test edilen ortam **Python 3.12**. Repo root'unda:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m streamlit run clinvar_app/app.py
```

Yollar uygulama dosyasına göre çözülür; çalışma dizinine bağımlı değildir. Başka bir doğrulanmış model bundle'ı seçmek için `CLINVAR_BUNDLE_DIR` ortam değişkenini kullan. VUS/performans dosyalarının eksik olması manuel tahmini durdurmaz; ilgili sayfa açıklama gösterir.

## Veri ve kapsam

- Kaynak: [NCBI tarihli ClinVar arşivleri](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/). Snapshot dosyası, URL'si ve SHA-256 kaydedilir.
- Assembly: **GRCh38**. Desteklenen giriş genomik tek nükleotid değişimidir: `chr11:g.126275389C>T`. c./p. transkript ifadeleri ve indeller bu sözleşmeye girmez.
- Review filtresi: `criteria provided, multiple submitters, no conflicts`, `reviewed by expert panel`, `practice guideline`. Yıldızlar inceleme/uzlaşma düzeyidir; mutlak doğru etiket garantisi değildir. [ClinVar tanımı](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/).
- İkili etiketler: P, LP, P/LP → 1; B, LB, B/LB → 0. Düşük penetrans, risk faktörü/risk alleli, çelişkili ve diğer özel etiketler dışlanır. Ham etiketler ve kategori sayımları saklanır.
- `Uncertain significance`, yalnızca skorlanacak ayrı kohorttur. Eğitim etiketi olarak kullanılmaz.
- Gerçek kapsam: **gerekli dbNSFP CADD ve phyloP anotasyonu bulunan SNV'ler**. Anotasyon bulunmaması tek başına non-coding consequence kanıtı değildir. dbNSFP non-synonymous ve splice-site SNV'leri kapsar; CADD'in kendisi genom genelinde skor verir. [dbNSFP](https://www.dbnsfp.org/), [CADD](https://cadd.bihealth.org/info).

HFE p.Cys282Tyr (`chr6:g.26092913G>A`) patojenik/düşük penetranslı bir istisnadır; bilinen benign kontrol değildir. Snapshot'ta bilinen özel etiketli kimlikler bundle'da tutulur ve canlı sorguda kapsam dışı olarak reddedilir. Bu liste bütün klinik istisnaların eksiksiz bir kataloğu değildir. [ClinVar Variation 9](https://www.ncbi.nlm.nih.gov/clinvar/variation/9/).

## Özellik sözleşmesi

| Özellik | Politika |
|---|---|
| `cadd_phred` | Gerekli, sonlu CADD Phred değeri |
| `phylop_100way` | Gerekli, sonlu phyloP 100way vertebrate değeri |
| `gnomad_af_log` | AF mevcutsa `log10(AF + 1e-9)`; yoksa NaN korunur |
| `gnomad_af_missing` | Frekans bilinmiyorsa 1, sayısal değer varsa 0 |

Gerçek sayısal AF=0 için log=-9 ve missing=0. Alan yoksa log=NaN ve missing=1. **Kaynakta alanın olmaması popülasyonda kesin gözlenmeme anlamına gelmez.** Mevcut AC/AN değerleri korunur; olmayan coverage/AN bilgisi üretilmez. [gnomAD kapsam ve allele-number açıklaması](https://gnomad.broadinstitute.org/news/2024-04-gnomad-v4-1/).

Aynı fonksiyon eğitim, manuel giriş ve canlı sorguda kullanılır. Anotasyonun kimliği istenen kimlikten farklıysa veya çoklu değerler belirsizse skor üretilmez. Canlı kaynak metaverisi eğitim bundle'ından farklıysa yeniden eğitim/eşleşen kaynak gerekir.

## Tam deneyin yeniden üretilmesi

Aşağıdaki komutlar bir kez tamamlanan koşuyu üzerine yazmaz. Büyük ham veri ve API cache'i `data/` altında yereldir; Git'e dahil edilmez. Anotasyon yarıda kalırsa aynı komutu/cache'i kullanmak tamamlanan kimlikleri tekrar indirmez. Başarısız batch'ler `pending.json` içinde tutulur; eksik bir dataset başarıyla bitmiş sayılmaz. Aynı cache'e eşzamanlı ikinci yazıcı kilitle reddedilir.

```bash
python scripts/download_snapshot.py 2026-07
python scripts/download_snapshot.py 2026-10
python -m clinvar_classifier.cli prepare data/raw/variant_summary_2026-07.txt.gz data/processed/2026-07 --source-url https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-07.txt.gz
python -m clinvar_classifier.cli prepare data/raw/variant_summary_2026-10.txt.gz data/processed/2026-10 --source-url https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/variant_summary_2026-10.txt.gz
python -m clinvar_classifier.cli annotate data/processed/2026-07/cohort.parquet data/annotation_cache data/processed/2026-07 --workers 4
python -m clinvar_classifier.cli annotate data/processed/2026-10/cohort.parquet data/annotation_cache data/processed/2026-10 --workers 4
python -m clinvar_classifier.cli train data/processed/2026-07/dataset.parquet data/processed/2026-07/dataset_manifest.json artifacts/experiments/reproduction --iterations 12 --folds 5
python -m clinvar_classifier.cli evaluate-external data/processed/2026-07/dataset.parquet data/processed/2026-10/dataset.parquet artifacts/experiments/reproduction
python -m clinvar_classifier.cli score-vus data/processed/2026-10/dataset.parquet artifacts/experiments/reproduction
```

Arşiv hash'ini `download_snapshot.py --expected-sha256 HASH` ile doğrulayabilirsin. Tam yeniden üretimde anotasyon kaynağı da aynı kalmalı: kaynak metaverisi değişmişse yeni bir cache/run gerekir. Eski anotasyon cache'ini korumak tam dataset tekrarını sağlar. Manifestte snapshot/annotation/dataset kimlikleri ve eğitim kodu hash'leri bulunur.

## Değerlendirme tasarımı

1. Aynı normalize varyant kimlikleri tekrar/etiket çelişkisi için kontrol edilir. Birleştirmeler one-to-one doğrulanır.
2. Ortak genleri olan çok genli anotasyonlar bağlı bileşenlere birleştirilir. Grupların yaklaşık %20'si gen testi için ayrılır; bu gruplar geliştirme ve kalibrasyonda bulunmaz.
3. Kalan satırların %20'si varyant testi; geri kalanların %20'si kalibrasyon olarak ayrılır. Varyant kimlikleri tüm partition'lar arasında ayrıdır. Satır oranları gen grubu büyüklüklerine göre değişir.
4. Algoritma karşılaştırması, ablation ve randomized search yalnızca **fit** satırlarında, gen ayrık CV fold'larıyla yapılır. Logistic regression preprocessing'i fold-local Pipeline içindedir. XGBoost model ailesi önceden seçilmiştir; test skoruna göre algoritma tercih edilmez.
5. Ayrı kalibrasyon satırlarının raw marginleriyle ağırlıksız sigmoid fit edilir. Varyant ve gen testleri bundan sonra açılır. Fit ve kalibrasyon arasında gen ayrıklığı iddiası yoktur; satır kimlikleri ayrıdır.
6. Temmuz'daki tüm etiketli kohort kimlikleri Ekim dış testinden çıkarılır. Yeni etiketli varyantlar ve Temmuz VUS olup Ekim P/LP–B/LB olanlar ayrıca değerlendirilir. Kaynak anotasyon sürümleri sabit tutulur.

Ölçümler: ROC-AUC, average precision, confusion matrix, sensitivity/specificity, precision/NPV, Brier, log loss, örneklem sayılarıyla kalibrasyon kutuları, AF alt grupları ve tek özellikli skor baseline'ları. 500 tekrarlı sınıfa göre tabakalı satır bootstrap aralığı gen bağımlılığını kapsamaz. Stop/splice Name desenleri yalnızca proxy'dir; kesin VEP veya missense-only sonucu diye sunulmaz.

Kalibrasyon eğitim dağılımındaki skoru ayarlar; hedef klinik/VUS popülasyonundaki mutlak olasılığı garanti etmez. Zaman bazlı ClinVar kontrolü fonksiyonel deney yerine geçmez; etiket değişimleri bilgisayarlı kanıtlardan etkilenmiş olabilir. `0.1/0.9` gösterim bantları ACMG/AMP kriterleri değildir. [ClinGen PP3/BP4 kalibrasyon çalışması](https://pmc.ncbi.nlm.nih.gov/articles/PMC9748256/).

CADD'in temel eğitim etiketleri insan soyunda türemiş ve simüle varyantlara dayanır. Bu repoda doğrudan ClinVar eğitim örtüşmesi kanıtlanmış değildir. Kaynak predictor tuning'i, protein/gen bağımlılığı ve ClinVar kürasyonundaki ortak kanıtlar ayrı değerlendirme konularıdır. [CADD eğitim verileri](https://cadd.bihealth.org/training), [circularity çalışması](https://pubmed.ncbi.nlm.nih.gov/25684150/).

## Mimari ve artefaktlar

```text
src/clinvar_classifier/  # schemas, data, annotation, features, split, train, evaluate, inference, cli
clinvar_app/app.py       # ince arayüz; yalnızca seçilen sayfa çalışır
clinvar.ipynb           # ortak modülleri kullanan çevrimdışı inceleme
tests/                  # veri/split/cache/model ve Streamlit testleri
scripts/                # snapshot indirme, ortam kilidi, notebook, tarihsel model dönüşümü
artifacts/research-v2/   # manifest + native model + kimlikli ölçüm ve Parquet çıktıları
artifacts/legacy/        # eski modelin yalnızca tarihsel karşılaştırma kopyası
data/                   # yerel ham snapshot, tam kohortlar ve devam ettirilebilir cache
```

Uygulama pickle yüklemez. Model native UBJSON, tablo verileri Parquet, metaveri JSON'dur. Bundle'ın zorunlu model hash'i başlangıçta; isteğe bağlı dosya hash'leri ilgili sayfa açıldığında doğrulanır. Model feature order manifest ile uyuşmalıdır. Genel SHAP, XGBoost native Tree SHAP ile üretilir ve raw-margin additivity kontrolünden geçer. Katkılar log-odds ölçeğindedir; kalibre skorun yüzde puan etkileri değildir.

VUS gezgini yalnızca gereken kolonları okur; literal gen araması yapar, en çok 50 satır seçer ve gösterilen/toplam sayısını ayrı yazar. Güncel sonuçların tamamı [koşu raporunda](docs/RUN_RESULTS.md), denetimdeki 26 bulguya verilen yanıtlar [uygulama raporunda](docs/IMPLEMENTATION_REPORT.md) yer alır.

## Kontroller

```bash
python -m pytest -q
python -m ruff check src clinvar_app tests scripts
python -m ruff format --check src clinvar_app tests scripts
python scripts/check_notebook.py
```

CI bu küçük kontrolleri çalıştırır; her PR'da büyük veri indirme ve tam eğitimi tekrar etmez. Notebook baştan sona çevrimdışı çalışır; bundle yoksa sahte performans yerine eksik koşu bilgisini gösterir.
