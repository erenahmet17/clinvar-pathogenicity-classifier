"""Generate measured run and implementation reports after completed model evaluation."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "research-v2"


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((BUNDLE / "metrics.json").read_text(encoding="utf-8"))
    external = json.loads((BUNDLE / "external_metrics.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (ROOT / "validation_logs" / "verification.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Ölçülen yeni koşu sonuçları",
        "",
        "Tarih: 7 Ekim 2026. Koşu: `research-v2`. Aşağıdaki değerler tamamlanmış yeni koşudan üretilir; eski `0.9962` sonuç yeni başarı olarak kullanılmaz.",
        "",
        "## İç değerlendirme",
        "",
        "| Partition | N | B/LB | P/LP | ROC-AUC | Average precision | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics["test"].items():
        v = result["calibrated_score"]
        lines.append(
            f"| {name} | {v['n']:,} | {v['n_benign']:,} | {v['n_pathogenic']:,} | {fmt(v['roc_auc'])} | {fmt(v['average_precision'])} | {fmt(v['brier'])} | {fmt(v['log_loss'])} |"
        )
    lines += [
        "",
        "Gen testi geliştirme kohortundaki bağlı gen gruplarından ayrıdır. Varyant testi ve kalibrasyon satır kimlikleri fit satırlarından ayrıdır. Model/özellik/tuning kararları fit-only gen ayrık CV ile alınmıştır. Hasta/VUS klinik olasılığı iddiası yoktur.",
        "",
        "## Zaman bazlı dış kontrol",
        "",
        "Temmuz 2026'daki tüm etiketli kohort kimlikleri Ekim 2026 dış değerlendirmesinden çıkarılmıştır. Reclassified VUS grubu, Temmuz'da VUS olup Ekim'de P/LP veya B/LB olarak kayıtlı varyantları içerir.",
        "",
        "| Grup | N | B/LB | P/LP | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in external["groups"].items():
        lines.append(
            f"| {name} | {value.get('n', 0):,} | {value.get('n_benign', 0):,} | {value.get('n_pathogenic', 0):,} | {fmt(value.get('roc_auc'))} | {fmt(value.get('brier'))} |"
        )
    lines += [
        "",
        "Bu, zaman bazlı ClinVar etiket kontrolüdür. Genetik hastalık riskini, penetransı veya bağımsız fonksiyonel doğruluğu kanıtlamaz. ClinVar etiketleri hesaplamalı kanıtlardan etkilenmiş olabilir.",
        "",
        f"**Reclassified VUS örneklemi yalnızca {external['groups']['reclassified_vus']['n']} varyanttır. AUC={fmt(external['groups']['reclassified_vus'].get('roc_auc'))}, Brier={fmt(external['groups']['reclassified_vus'].get('brier'))}. Bu küçük gruptaki daha zayıf sonuç geniş VUS popülasyonu için güçlü doğrulama sayılmaz. Yeni etiketli grup, önceki 2+ yıldızlı eğitim kohortuna yeni uygun hale gelen kimliklerdir; her kimliğin ilk kez ClinVar'da etiketlendiği iddiası değildir.**",
        "",
        "## Model geliştirme",
        "",
        f"Fit satırları: **{manifest['n_fit']:,}**. Kalibrasyon satırları: **{manifest['n_calibration']:,}**. CV fold: **5**. Arama iterasyonu: **12**.",
        "",
        "En iyi geliştirme CV ROC-AUC: `"
        + fmt(metrics["development"]["best_cv_roc_auc"])
        + "`. Seçilen en iyi CV değerinde arama iyimserliği olabilir; bu değer dış test değildir.",
        "",
        "```json",
        json.dumps(metrics["development"]["best_params"], indent=2),
        "```",
        "",
        "## VUS çıktı kümesi",
        "",
        "```json",
        json.dumps(manifest["vus_counts"], ensure_ascii=False, indent=2),
        "```",
        "",
        "Bu skorlar ClinVar VUS sınıflarını değiştirmez. Eşik bantları yalnızca gösterim/önceliklendirme içindir.",
        "",
        "## Kaynak ve tekrar üretim",
        "",
        f"Eğitim snapshot: `{manifest['provenance']['cohort']['snapshot_filename']}`",
        "",
        f"SHA-256: `{manifest['provenance']['cohort']['snapshot_sha256']}`",
        "",
        f"VUS snapshot: `{manifest['vus_source']['cohort']['snapshot_filename']}`",
        "",
        f"SHA-256: `{manifest['vus_source']['cohort']['snapshot_sha256']}`",
        "",
        "Anotasyon kaynak metaverisi, namespace, tüm model/özellik sırası, bağımlılık sürümleri ve kod hash'leri `manifest.json` içinde saklanır. Tam kaynak dataset/cache yerel `data/` altında korunur. NCBI snapshot'ları tarihli arşiv URL'lerinden yeniden indirilebilir; tam anotasyon tekrarında aynı cache veya aynı kaynak sürümü gerekir.",
        "",
        "## Grafikler",
        "",
        "![Fit-only EDA](../artifacts/research-v2/development_eda.png)",
        "",
        "![CV, test ve kalibrasyon](../artifacts/research-v2/evaluation.png)",
        "",
        "![Tree SHAP özeti](../artifacts/research-v2/shap_summary.png)",
        "",
        "## Teknik doğrulama",
        "",
        "```json",
        json.dumps(verification, ensure_ascii=False, indent=2),
        "```",
        "",
        "ROC-AUC aralıkları sınıfa göre tabakalı satır bootstrap'ına dayanır; gen kümelenmesi ve domain shift bu aralıklarda modellenmez. Name desenleri consequence proxy'sidir; kesin VEP/missense-only iddiası yapılmaz. Ayrıntılı alt grup, kutu sayıları ve tahmin kimlikleri bundle içinde saklanır.",
    ]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    lines += [
        "",
        "## Kohort seçimi ve nadir AF alt grupları",
        "",
        "| Kaynak | Benign önce/sonra | Patojenik önce/sonra | VUS önce/sonra |",
        "|---|---:|---:|---:|",
    ]
    for name, source in [("Temmuz", manifest["provenance"]), ("Ekim", manifest["vus_source"])]:
        keep = source["retention"]
        lines.append(
            f"| {name} | {keep['class_retention']['0']['before']:,} / {keep['class_retention']['0']['after']:,} | {keep['class_retention']['1']['before']:,} / {keep['class_retention']['1']['after']:,} | {keep['vus_before']:,} / {keep['vus_after']:,} |"
        )
    lines += ["", "| Test / alt grup | N | B/LB | P/LP | ROC-AUC |", "|---|---:|---:|---:|---:|"]
    for name, values in metrics["test"].items():
        for group, value in values["subgroups"].items():
            if isinstance(value, dict):
                lines.append(
                    f"| {name} / {group} | {value['n']:,} | {value['n_benign']:,} | {value['n_pathogenic']:,} | {fmt(value['roc_auc'])} |"
                )
    lines += [
        "",
        "Bu alt gruplar önceden tanımlanan betimsel ölçümlerdir. az sayıda benign/patojenik kayıt bulunan gruplar dikkatle okunmalıdır. Name desenleri yalnızca consequence proxy'sidir.",
    ]
    (docs / "RUN_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fixes = [
        (
            "B01",
            "Model seçimi testten ayrıldı",
            "fit-only grouped CV; ayrı calibration/variant_test/gene_test; kalıcı kimlik manifesti",
        ),
        (
            "B02",
            "CV preprocessing düzeltildi",
            "Logistic regression için fold-local imputation/scaling Pipeline; ağaçlar ham ortak özellikleri kullanır",
        ),
        (
            "B03",
            "Eksik AF ve sıfır ayrıldı",
            "NaN + missing flag; af_available, AC/AN/status korunur; coverage yoksa üretilmez",
        ),
        (
            "B04",
            "Etiket politikası düzeltildi",
            "P/LP/B/LB izin listesi; özel etiketler dışlanır; kategori sayımları saklanır",
        ),
        (
            "B05",
            "Kimlik/gen/zaman kontrolleri eklendi",
            "çok genli anotasyon bağlı bileşenleri; gen ayrık test/CV; yeni tarihli dış kontrol",
        ),
        (
            "B06",
            "Kohort kapsamı açıklanıp ölçüldü",
            "annotation status ve sınıf/VUS retention; stop/splice/missense proxy alt grupları; kesin consequence için VEP hâlâ gerekir",
        ),
        (
            "B07",
            "Skor sunumu düzeltildi ve kalibrasyon eklendi",
            "ayrı satırlarda sigmoid; Brier/log loss/reliability; temporal VUS kontrolü; klinik olasılık iddiası kaldırıldı",
        ),
        (
            "B08",
            "Yanlış benign örneği kaldırıldı",
            "HFE doğrulanmış özel kapsam dışı kimlik; canlı sorguda skor reddi; klinik sınıf değiştirilmedi",
        ),
        (
            "B09",
            "Baseline ve ablation eklendi",
            "fit-only CADD-only, AF-only, CADD+AF ve tam XGBoost; Logistic/RF karşılaştırmaları; test tek özellik sıralamaları",
        ),
        (
            "B10",
            "Circularity açıklaması düzeltildi",
            "CADD temel eğitim mekanizması doğru anlatıldı; doğrudan örtüşme kanıtı ile olasılık ayrıldı; kaynak metaverisi korundu",
        ),
        (
            "B11",
            "Desteksiz istatistiksel iddialar kaldırıldı",
            "fold skorları/search sonuçları korunur; yakın CV ortalamaları eşdeğerlik veya performans tavanı sayılmaz",
        ),
        (
            "B12",
            "SHAP ölçeği ve kanıt sınırı açıklandı",
            "native Tree SHAP raw margin; additivity doğrulaması; fit-only genel özet; klinik/nedensel kanıt iddiası yok",
        ),
        (
            "B13",
            "Yeni koşu provenance'ı oluşturuldu",
            "tarihli snapshot/hash; API namespace; dataset/code/environment hash; kimlikli test/split çıktısı",
        ),
        (
            "M01",
            "Tanımsız estimator/notebook akışı kaldırıldı",
            "eğitim import edilebilir modüle taşındı; temiz offline notebook ve izole kernel doğrulaması",
        ),
        (
            "M02",
            "Manuel 502 devam akışı kaldırıldı",
            "tek devam ettirilebilir batch akışı; atomik tamamlanma checkpoint'i",
        ),
        (
            "M03",
            "HTTP güvenilirliği iyileştirildi",
            "timeout, sınırlı retry/backoff, Retry-After, pending ID listesi, değişen batch sınırlarında kimlik reuse; tek cache yazıcı kilidi",
        ),
        (
            "M04",
            "Join sözleşmesi güçlendirildi",
            "tek-pass snapshot kolonları; canonical kimlik, kopya/çelişki ve one-to-one kontrolü",
        ),
        (
            "M05",
            "Tekrarlı feature/inference kodu kaldırıldı",
            "ortak tipli Annotation, parser, make_features, Predictor; belirsiz çoklu/yanlış tip/numerik değer kontrolü",
        ),
        (
            "M06",
            "Girdi ve assembly açık hale getirildi",
            "GRCh38 genomic SNV doğrulaması; source metadata uyumsuzluğunda skor yok; annotation yokluğu consequence sayılmaz",
        ),
        (
            "M07",
            "Regex kullanıcı hatası düzeltildi",
            "literal gene text filter; '[' regresyon testi",
        ),
        (
            "M08",
            "CWD bağımlılığı kaldırıldı",
            "Path(__file__) tabanlı yollar; isteğe bağlı sayfa artefaktları ayrı yüklenir",
        ),
        (
            "M09",
            "Ortam ve model formatı düzeltildi",
            "pyproject + tested transitive lock; native UBJSON/JSON/Parquet; runtime'da pickle yok",
        ),
        (
            "M10",
            "Rerun maliyeti azaltıldı",
            "yalnızca seçili sayfa; form submission; hash-key cache; nlargest/nsmallest; native SHAP; figür kapatma",
        ),
        (
            "M11",
            "Büyük/kopya artefaktlar konsolide edildi",
            "eski 97.6 MB CSV ve çift pickle/CSV kopyaları kaldırıldı; tek v2 bundle; eski native model ayrı historical dizinde",
        ),
        (
            "M12",
            "Test ve CI kapısı eklendi",
            "veri/split/cache/model ve Streamlit regresyonları; lint/format/notebook CI; tam training her PR'da indirilmez",
        ),
        (
            "M13",
            "Dokümantasyon/sunum yeniden yazıldı",
            "gerçek fit/calibration/VUS sayıları manifestten; 12-iteration protokol; model skoru ve gösterilen/toplam ayrımı",
        ),
    ]
    body = [
        "# Düzeltme ve iyileştirme raporu",
        "",
        "Tarih: 7 Ekim 2026. Başlangıç sürümü: `05511a08d2c30d496d011bcb80ff37234c4a3f7b`. Çalışma dalı: `codex/scientific-architecture-fixes`.",
        "",
        "Kod, veri hazırlığı, yeni model eğitimi, iç/zaman bazlı değerlendirme ve uygulama kontrolleri tamamlandı. Eski model bilimsel sınırlamalarıyla tarihsel karşılaştırma için ayrıldı. Yeni sonuçlar [ölçülen koşu raporunda](RUN_RESULTS.md).",
        "",
        "## 26 bulguya verilen yanıt",
        "",
        "| ID | Yapılan işlem | Ayrıntı / sınır |",
        "|---|---|---|",
    ]
    body.extend(f"| {key} | {title} | {detail} |" for key, title, detail in fixes)
    body += [
        "",
        "## Yeni bilimsel çalışma",
        "",
        "Temmuz 2026 tarihli snapshot'tan açık etiket politikasıyla yeni veri hazırlandı; anotasyonlar kaynak metaverisi sabitlenerek indirildi. Native model, yalnızca fit partition'ında seçilen parametrelerle eğitildi. Ayrı calibration satırlarında sigmoid fit edildi. Varyant ve bağlı gen testleri seçim tamamlandıktan sonra değerlendirildi. Ekim 2026 snapshot'ında yeni etiketli varyantlar ve önceki VUS'tan P/LP–B/LB olanlar kimlik bazında ayrı dış kontrol olarak ölçüldü. Ekim'deki in-scope VUS'lar yeni modelle skorlandı.",
        "",
        f"Fit={manifest['n_fit']:,}; calibration={manifest['n_calibration']:,}; VUS skorlanan={manifest['vus_counts']['n_scored']:,}.",
        "",
        f"Varyant testi ROC-AUC={fmt(metrics['test']['variant_test']['calibrated_score']['roc_auc'])}; gen testi={fmt(metrics['test']['gene_test']['calibrated_score']['roc_auc'])}; Ekim'deki yeni uygun etiketli varyantlar={fmt(external['groups']['new_labeled_variants'].get('roc_auc'))}. Reclassified VUS n={external['groups']['reclassified_vus']['n']}, ROC-AUC={fmt(external['groups']['reclassified_vus'].get('roc_auc'))}. Son grup çok küçük ve daha zayıftır; klinik VUS doğrulaması olarak sunulmaz.",
        "",
        "## Yazılım ve uygulama",
        "",
        "Notebook, ortak paket fonksiyonlarını çağıran inceleme katmanına dönüştürüldü. CLI veri hazırlama, devam ettirilebilir anotasyon, eğitim, dış değerlendirme ve VUS skorlama adımlarını açık komutlara ayırır. UI aynı feature sözleşmesini kullanır; manuel unknown AF korunur; canlı source metadata değişmişse skor reddedilir. VUS ve performans sayfaları yalnızca seçildiğinde çalışır. Model startup native hash/feature-order doğrulaması yapar; yardımcı veri eksikliği manuel inference'ı bozmaz.",
        "",
        "Eski çift CSV/pickle kopyaları ve 97.6 MB VUS CSV'si kaldırıldı. Eski model, sınırlı one-time dönüştürme script'iyle native UBJSON'a çıkarılıp historical manifest/metrics ile saklandı. Default runtime bu eski modeli kullanmaz. Tam kaynak veriler/checkpoint'ler yerel ignored data/ dizininde korundu; bundle yalnızca uygulama ve denetlenebilir sonuç dosyalarını içerir.",
        "",
        "## Yapılan doğrulamalar",
        "",
        "```json",
        json.dumps(verification, ensure_ascii=False, indent=2),
        "```",
        "",
        "Testler yanlış/çoklu annotation, missing-vs-zero AF, özel penetrans scope reddi, kimlik/gen/fold ayrıklığı, batch eksikliği, process-restart cache reuse, cache writer kilidi, model checksum/feature order, SHAP raw-margin additivity ve Streamlit kullanıcı akışlarını kapsar. Notebook temiz namespace'te ve ayrı Jupyter kernel'inde yürütüldü. Büyük veri indirme/eğitim fixture smoke ile karıştırılmadı; gerçek yeni koşu ölçümleri ayrıca kaydedildi.",
        "",
        "## Kalan bilimsel sınırlar",
        "",
        "- Bu çalışma klinik patojenite, penetrans veya hasta hastalık riski validasyonu değildir. Skorlar ACMG/AMP yeniden sınıflandırma eşikleri olarak kullanılmaz.",
        "- Kaynakta AF'nin eksik olması artık sıfır değildir; ancak bilinmeyen coverage/AN bilgisi bu yazılımla yeniden üretilemez. Klinik yokluk kanıtı ayrıca incelenmelidir.",
        "- Kesin consequence/transcript stratifikasyonu için sürümlü VEP/MANE anotasyonu gerekir. Mevcut stop/splice/missense Name desenleri proxy olarak etiketlidir.",
        "- ClinVar zaman testi bağımsız fonksiyonel veya hasta/segregasyon verisi değildir. Kaynak predictorların eğitim/tuning veri örtüşmesi tamamen dışlanmış sayılmaz.",
        "- Kalibrasyon, seçilmiş araştırma kohortundaki skor davranışını ayarlar; tüm VUS/gen/ancestry gruplarında mutlak olasılık garantisi değildir. Satır bootstrap gen bağımlılığını kapsamaz.",
        "- Fit kohortunda AF bilinmeme oranı B/LB için yaklaşık %3,85, P/LP için %75,43'tür. Eksik-sıfır ayrımı düzeltilmiş olsa da kaynak bulunabilirliği güçlü bir sınıf sinyali taşır; teknik ve biyolojik sinyal bütünüyle ayrıştırılmış sayılmaz.",
        "- Eski eğitimin eksik ham snapshot/kimlikleri geçmişe dönük uydurulmadı. Yeni koşuda bunlar saklanıyor; eski ve yeni AUC farkı aynı veri üzerinde kontrollü iyileşme olarak sunulmaz.",
        "",
        "## Tekrar çalıştırma ve teslim",
        "",
        "Kurulum, CLI ve kontroller README'de. Her run ayrı output directory kullanır. Büyük kaynak dosyaları Git'e eklenmez. Tam snapshot URL/hash ve anotasyon namespace'ini manifestten doğrula; yeniden indirme sırasında kaynak sürümü değişmişse yeni bir deney oluştur. GitHub değişiklikleri ayrı dal/taslak PR üzerinden incelenebilir; main ile birleştirme bu çalışmanın otomatik adımı değildir.",
    ]
    (docs / "IMPLEMENTATION_REPORT.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("Wrote measured run results and the 26-finding implementation report.")


if __name__ == "__main__":
    main()
