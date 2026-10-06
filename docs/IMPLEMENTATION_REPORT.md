# Düzeltme ve iyileştirme raporu

Tarih: 7 Ekim 2026. Başlangıç sürümü: `05511a08d2c30d496d011bcb80ff37234c4a3f7b`. Çalışma dalı: `codex/scientific-architecture-fixes`.

Kod, veri hazırlığı, yeni model eğitimi, iç/zaman bazlı değerlendirme ve uygulama kontrolleri tamamlandı. Eski model bilimsel sınırlamalarıyla tarihsel karşılaştırma için ayrıldı. Yeni sonuçlar [ölçülen koşu raporunda](RUN_RESULTS.md).

## 26 bulguya verilen yanıt

| ID | Yapılan işlem | Ayrıntı / sınır |
|---|---|---|
| B01 | Model seçimi testten ayrıldı | fit-only grouped CV; ayrı calibration/variant_test/gene_test; kalıcı kimlik manifesti |
| B02 | CV preprocessing düzeltildi | Logistic regression için fold-local imputation/scaling Pipeline; ağaçlar ham ortak özellikleri kullanır |
| B03 | Eksik AF ve sıfır ayrıldı | NaN + missing flag; af_available, AC/AN/status korunur; coverage yoksa üretilmez |
| B04 | Etiket politikası düzeltildi | P/LP/B/LB izin listesi; özel etiketler dışlanır; kategori sayımları saklanır |
| B05 | Kimlik/gen/zaman kontrolleri eklendi | çok genli anotasyon bağlı bileşenleri; gen ayrık test/CV; yeni tarihli dış kontrol |
| B06 | Kohort kapsamı açıklanıp ölçüldü | annotation status ve sınıf/VUS retention; stop/splice/missense proxy alt grupları; kesin consequence için VEP hâlâ gerekir |
| B07 | Skor sunumu düzeltildi ve kalibrasyon eklendi | ayrı satırlarda sigmoid; Brier/log loss/reliability; temporal VUS kontrolü; klinik olasılık iddiası kaldırıldı |
| B08 | Yanlış benign örneği kaldırıldı | HFE doğrulanmış özel kapsam dışı kimlik; canlı sorguda skor reddi; klinik sınıf değiştirilmedi |
| B09 | Baseline ve ablation eklendi | fit-only CADD-only, AF-only, CADD+AF ve tam XGBoost; Logistic/RF karşılaştırmaları; test tek özellik sıralamaları |
| B10 | Circularity açıklaması düzeltildi | CADD temel eğitim mekanizması doğru anlatıldı; doğrudan örtüşme kanıtı ile olasılık ayrıldı; kaynak metaverisi korundu |
| B11 | Desteksiz istatistiksel iddialar kaldırıldı | fold skorları/search sonuçları korunur; yakın CV ortalamaları eşdeğerlik veya performans tavanı sayılmaz |
| B12 | SHAP ölçeği ve kanıt sınırı açıklandı | native Tree SHAP raw margin; additivity doğrulaması; fit-only genel özet; klinik/nedensel kanıt iddiası yok |
| B13 | Yeni koşu provenance'ı oluşturuldu | tarihli snapshot/hash; API namespace; dataset/code/environment hash; kimlikli test/split çıktısı |
| M01 | Tanımsız estimator/notebook akışı kaldırıldı | eğitim import edilebilir modüle taşındı; temiz offline notebook ve izole kernel doğrulaması |
| M02 | Manuel 502 devam akışı kaldırıldı | tek devam ettirilebilir batch akışı; atomik tamamlanma checkpoint'i |
| M03 | HTTP güvenilirliği iyileştirildi | timeout, sınırlı retry/backoff, Retry-After, pending ID listesi, değişen batch sınırlarında kimlik reuse; tek cache yazıcı kilidi |
| M04 | Join sözleşmesi güçlendirildi | tek-pass snapshot kolonları; canonical kimlik, kopya/çelişki ve one-to-one kontrolü |
| M05 | Tekrarlı feature/inference kodu kaldırıldı | ortak tipli Annotation, parser, make_features, Predictor; belirsiz çoklu/yanlış tip/numerik değer kontrolü |
| M06 | Girdi ve assembly açık hale getirildi | GRCh38 genomic SNV doğrulaması; source metadata uyumsuzluğunda skor yok; annotation yokluğu consequence sayılmaz |
| M07 | Regex kullanıcı hatası düzeltildi | literal gene text filter; '[' regresyon testi |
| M08 | CWD bağımlılığı kaldırıldı | Path(__file__) tabanlı yollar; isteğe bağlı sayfa artefaktları ayrı yüklenir |
| M09 | Ortam ve model formatı düzeltildi | pyproject + tested transitive lock; native UBJSON/JSON/Parquet; runtime'da pickle yok |
| M10 | Rerun maliyeti azaltıldı | yalnızca seçili sayfa; form submission; hash-key cache; nlargest/nsmallest; native SHAP; figür kapatma |
| M11 | Büyük/kopya artefaktlar konsolide edildi | eski 97.6 MB CSV ve çift pickle/CSV kopyaları kaldırıldı; tek v2 bundle; eski native model ayrı historical dizinde |
| M12 | Test ve CI kapısı eklendi | veri/split/cache/model ve Streamlit regresyonları; lint/format/notebook CI; tam training her PR'da indirilmez |
| M13 | Dokümantasyon/sunum yeniden yazıldı | gerçek fit/calibration/VUS sayıları manifestten; 12-iteration protokol; model skoru ve gösterilen/toplam ayrımı |

## Yeni bilimsel çalışma

Temmuz 2026 tarihli snapshot'tan açık etiket politikasıyla yeni veri hazırlandı; anotasyonlar kaynak metaverisi sabitlenerek indirildi. Native model, yalnızca fit partition'ında seçilen parametrelerle eğitildi. Ayrı calibration satırlarında sigmoid fit edildi. Varyant ve bağlı gen testleri seçim tamamlandıktan sonra değerlendirildi. Ekim 2026 snapshot'ında yeni etiketli varyantlar ve önceki VUS'tan P/LP–B/LB olanlar kimlik bazında ayrı dış kontrol olarak ölçüldü. Ekim'deki in-scope VUS'lar yeni modelle skorlandı.

Fit=53,731; calibration=13,433; VUS skorlanan=298,885.

Varyant testi ROC-AUC=0.993771; gen testi=0.995308; Ekim'deki yeni uygun etiketli varyantlar=0.993656. Reclassified VUS n=15, ROC-AUC=0.839286. Son grup çok küçük ve daha zayıftır; klinik VUS doğrulaması olarak sunulmaz.

## Yazılım ve uygulama

Notebook, ortak paket fonksiyonlarını çağıran inceleme katmanına dönüştürüldü. CLI veri hazırlama, devam ettirilebilir anotasyon, eğitim, dış değerlendirme ve VUS skorlama adımlarını açık komutlara ayırır. UI aynı feature sözleşmesini kullanır; manuel unknown AF korunur; canlı source metadata değişmişse skor reddedilir. VUS ve performans sayfaları yalnızca seçildiğinde çalışır. Model startup native hash/feature-order doğrulaması yapar; yardımcı veri eksikliği manuel inference'ı bozmaz.

Eski çift CSV/pickle kopyaları ve 97.6 MB VUS CSV'si kaldırıldı. Eski model, sınırlı one-time dönüştürme script'iyle native UBJSON'a çıkarılıp historical manifest/metrics ile saklandı. Default runtime bu eski modeli kullanmaz. Tam kaynak veriler/checkpoint'ler yerel ignored data/ dizininde korundu; bundle yalnızca uygulama ve denetlenebilir sonuç dosyalarını içerir.

## Yapılan doğrulamalar

```json
{
  "pytest_tests": 60,
  "pytest_failures": 0,
  "ruff_lint": "passed",
  "ruff_format": "passed",
  "pip_check": "passed",
  "offline_notebook_6_code_cells": "passed",
  "isolated_jupyter_restart_run_all": "passed",
  "training_source_hashes_match_git_base_commit": "passed",
  "live_query": {
    "annotation_status": "ok",
    "live_source_namespace_matches": true,
    "model_score": 0.9242546795290338,
    "direction": "Pathogenic direction"
  },
  "real_bundle": {
    "default_app_pages": "passed",
    "manual_unknown_af": "passed",
    "literal_gene_filter": "passed",
    "hfe_scope_rejection": "passed",
    "mandatory_and_optional_checksums": "passed",
    "variant_gene_and_cv_fold_disjointness": "passed",
    "test_prediction_rows": 34251,
    "max_saved_prediction_difference": 0.0,
    "native_shap_additivity": "passed on 100 test rows",
    "features": [
      "cadd_phred",
      "phylop_100way",
      "gnomad_af_log",
      "gnomad_af_missing"
    ],
    "shap_rows": 100,
    "shap_base_rows": 100
  }
}
```

Testler yanlış/çoklu annotation, missing-vs-zero AF, özel penetrans scope reddi, kimlik/gen/fold ayrıklığı, batch eksikliği, process-restart cache reuse, cache writer kilidi, model checksum/feature order, SHAP raw-margin additivity ve Streamlit kullanıcı akışlarını kapsar. Notebook temiz namespace'te ve ayrı Jupyter kernel'inde yürütüldü. Büyük veri indirme/eğitim fixture smoke ile karıştırılmadı; gerçek yeni koşu ölçümleri ayrıca kaydedildi.

## Kalan bilimsel sınırlar

- Bu çalışma klinik patojenite, penetrans veya hasta hastalık riski validasyonu değildir. Skorlar ACMG/AMP yeniden sınıflandırma eşikleri olarak kullanılmaz.
- Kaynakta AF'nin eksik olması artık sıfır değildir; ancak bilinmeyen coverage/AN bilgisi bu yazılımla yeniden üretilemez. Klinik yokluk kanıtı ayrıca incelenmelidir.
- Kesin consequence/transcript stratifikasyonu için sürümlü VEP/MANE anotasyonu gerekir. Mevcut stop/splice/missense Name desenleri proxy olarak etiketlidir.
- ClinVar zaman testi bağımsız fonksiyonel veya hasta/segregasyon verisi değildir. Kaynak predictorların eğitim/tuning veri örtüşmesi tamamen dışlanmış sayılmaz.
- Kalibrasyon, seçilmiş araştırma kohortundaki skor davranışını ayarlar; tüm VUS/gen/ancestry gruplarında mutlak olasılık garantisi değildir. Satır bootstrap gen bağımlılığını kapsamaz.
- Fit kohortunda AF bilinmeme oranı B/LB için yaklaşık %3,85, P/LP için %75,43'tür. Eksik-sıfır ayrımı düzeltilmiş olsa da kaynak bulunabilirliği güçlü bir sınıf sinyali taşır; teknik ve biyolojik sinyal bütünüyle ayrıştırılmış sayılmaz.
- Eski eğitimin eksik ham snapshot/kimlikleri geçmişe dönük uydurulmadı. Yeni koşuda bunlar saklanıyor; eski ve yeni AUC farkı aynı veri üzerinde kontrollü iyileşme olarak sunulmaz.

## Tekrar çalıştırma ve teslim

Kurulum, CLI ve kontroller README'de. Her run ayrı output directory kullanır. Büyük kaynak dosyaları Git'e eklenmez. Tam snapshot URL/hash ve anotasyon namespace'ini manifestten doğrula; yeniden indirme sırasında kaynak sürümü değişmişse yeni bir deney oluştur. GitHub değişiklikleri ayrı dal/taslak PR üzerinden incelenebilir; main ile birleştirme bu çalışmanın otomatik adımı değildir.
