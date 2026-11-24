# Döngüsellik (circularity) ablasyonu — gnomAD AF

Tarih: 23 Kasım 2026. Script: `experiments/circularity_ablation.py`. Çıktılar: `artifacts/experiments/circularity_ablation/` (`ablation_metrics.json`, `ablation_predictions.parquet`).
Protokol research-v2 ile birebir aynı: seed 42, aynı fit/kalibrasyon/variant_test/gene_test bölmeleri, aynı gen gruplu 5 katlı CV, 12 iterasyonluk randomized search, ayrı sigmoid kalibrasyon. Tek fark kullanılan özellik sütunları. Tam modelin skorları yayınlanan bundle'la aynı (0.993771 / 0.995308 / 0.993656 / 0.839286).

## Şüphe: gnomAD AF (`gnomad_af_log` + `gnomad_af_missing`)

ClinVar gönderenleri P/LP ve B/LB etiketlerini ACMG/AMP kriterleriyle verir. Bu kriterlerin bir kısmı doğrudan popülasyon frekansına bakar: BA1 (AF > %5, tek başına benign), BS1 (hastalık için fazla yaygın) ve PM2 (popülasyon veritabanlarında yok ya da çok nadir). Model aynı frekansı özellik olarak alıyor, yani etiketin bir kısmı bu özellikten türetilmiş.

Temmuz 2026 etiketli kohortu (N = 101,415):

| gnomAD AF | B/LB | P/LP | P/LP oranı |
|---|---:|---:|---:|
| yok (missing) | 1,447 | 44,579 | 0.969 |
| 0 | 48 | 203 | 0.809 |
| (0, 1e-4] | 6,670 | 13,998 | 0.677 |
| (1e-4, 1e-3] | 7,813 | 841 | 0.097 |
| (1e-3, %1] | 14,491 | 86 | 0.006 |
| (%1, %5] | 6,063 | 6 | 0.001 |
| > %5 (BA1) | 5,168 | 2 | 0.0004 |

- Benign etiketlerin %26.9'unun (11,231 / 41,700) AF değeri ≥ %1; 5,168 tanesi BA1 eşiğinin üstünde.
- Patojenik etiketlerin %74.7'sinde AF yok. Bu, PM2 kriteriyle örtüşüyor.
- Tek özellik AUC'leri: AF (missing = -9 olarak kodlanmış) 0.965, yalnızca AF_missing 0.856, CADD 0.957, phyloP 0.790.

## ROC-AUC (kalibre skor, 500 tekrarlı satır bootstrap %95 GA)

| Model | CV (fit) | variant_test (16,791) | gene_test (17,460) | Ekim yeni etiketli (6,999) | Reclassified VUS (15) |
|---|---:|---:|---:|---:|---:|
| Tam (CADD+phyloP+AF+AF_missing) | 0.9941 | 0.9938 [0.9930–0.9946] | 0.9953 [0.9946–0.9959] | 0.9937 [0.9923–0.9949] | 0.839 [0.60–1.00] |
| **AF'siz (CADD+phyloP)** | **0.9599** | **0.9572 [0.9545–0.9599]** | **0.9627 [0.9602–0.9652]** | **0.9607 [0.9566–0.9647]** | **0.714 [0.38–1.00]** |
| CADD'siz (phyloP+AF) | 0.9775 | 0.9771 | 0.9830 | 0.9691 | 0.679 |
| Yalnızca AF | 0.9637 | 0.9650 | 0.9754 | 0.9465 | 0.759 |
| Yalnızca phyloP | 0.7934 | 0.7813 | 0.7874 | 0.8023 | 0.446 |

Eşleştirilmiş fark, Tam − AF'siz (aynı satırlar, paired bootstrap %95 GA):

- variant_test: +0.0366 [0.0339–0.0391]
- gene_test: +0.0326 [0.0305–0.0348]
- Ekim yeni etiketli: +0.0329 [0.0295–0.0364]
- Reclassified VUS: +0.125 [−0.107–0.375]. N = 15 olduğu için anlamlı değil.

Eşik 0.5'te variant_test: Tam modelde duyarlılık 0.970, özgüllük 0.956, Brier 0.027. AF'siz modelde duyarlılık 0.920, özgüllük 0.853, Brier 0.078.

## AF katmanları içinde AUC (Tam vs AF'siz)

| Test / AF katmanı | N | B/LB | P/LP | Tam | AF'siz |
|---|---:|---:|---:|---:|---:|
| variant_test / AF yok | 7,807 | 248 | 7,559 | 0.9707 | 0.9710 |
| variant_test / AF ≤ 1e-4 | 3,418 | 1,148 | 2,270 | 0.9560 | 0.9431 |
| variant_test / 1e-4 < AF ≤ 1e-3 | 1,497 | 1,364 | 133 | 0.9044 | 0.8914 |
| variant_test / AF > 1e-3 | 4,069 | 4,058 | 11 | 0.8731 | 0.8593 |
| gene_test / AF yok | 7,040 | 153 | 6,887 | 0.9710 | 0.9719 |
| gene_test / AF ≤ 1e-4 | 3,732 | 957 | 2,775 | 0.9577 | 0.9426 |
| gene_test / 1e-4 < AF ≤ 1e-3 | 1,583 | 1,414 | 169 | 0.9490 | 0.9339 |
| gene_test / AF > 1e-3 | 5,105 | 5,087 | 18 | 0.9326 | 0.9055 |

Yorum: Toplamda AF yaklaşık 0.033–0.037 AUC kazandırıyor. Aynı AF katmanının içinde ise kazanç yalnızca 0.00–0.03. Kazancın büyük kısmı varyantları, etiketi zaten frekansla belirlenmiş kutulara ayırmaktan geliyor. Döngüselliğin beklenen imzası bu.

## Sınırlar

- CADD de tamamen bağımsız değil. Küratörler PP3/BP4 kapsamında in-silico skorları kanıt olarak kullanıyor. CADD'in kendi eğitimi ClinVar'a dayanmasa da etiketlerle kısmi bir döngüsellik olabilir. AF'siz sonuç (~0.96) bu yüzden üst sınır olarak okunmalı.
- Bootstrap aralıkları gen kümelenmesini modellemiyor.
- Reclassified VUS grubu yalnızca 15 varyant; bu gruptan sonuç çıkarılmamalı.

---

# Adım 1: AF'siz model, nadir missense değerlendirmesi

Script: `experiments/rare_missense_eval.py`. Çıktı: `artifacts/experiments/rare_missense/rare_missense_metrics.json`.
Bölmeler research-v2 ile aynı. Missense, `subgroup_report`'taki Name proxy'si ile belirlendi (p.XxxNNNYyy, Ter değil). Nadir: gnomAD AF ≤ 1e-4 veya AF yok. "AF gözlenen" alt kümesi, AF'nin olmamasının kendisinin sinyal olarak kullanılamadığı satırlardır.

## Değerlendirme kümeleri

| Küme | N | B/LB | P/LP | Yalnızca CADD | Yalnızca phyloP |
|---|---:|---:|---:|---:|---:|
| variant_test, nadir missense | 5,215 | 1,188 | 4,027 | 0.8903 | 0.8193 |
| variant_test, nadir missense, AF gözlenen | 2,028 | 1,037 | 991 | 0.8770 | 0.7825 |
| gene_test, nadir missense | 4,442 | 953 | 3,489 | 0.8893 | 0.8058 |
| gene_test, nadir missense, AF gözlenen | 2,025 | 857 | 1,168 | 0.8799 | 0.7705 |
| Ekim yeni etiketli, nadir missense | 2,462 | 947 | 1,515 | 0.8919 | 0.8368 |
| Ekim yeni etiketli, nadir missense, AF gözlenen | 1,049 | 840 | 209 | 0.8736 | 0.7753 |

## ROC-AUC [%95 GA], tüm varyantlarla eğitilmiş model (research-v2 eğitim kapsamı)

| Küme | Tam (AF dahil) | **AF'siz** | Fark (eşleştirilmiş) |
|---|---:|---:|---:|
| variant_test, nadir missense | 0.9559 [0.950–0.962] | **0.8981 [0.888–0.908]** | +0.058 [0.050–0.066] |
| variant_test, AF gözlenen | 0.9062 [0.892–0.917] | **0.8816 [0.867–0.895]** | +0.025 [0.017–0.032] |
| gene_test, nadir missense | 0.9529 [0.945–0.960] | **0.8962 [0.885–0.908]** | +0.057 [0.049–0.065] |
| gene_test, AF gözlenen | 0.9138 [0.900–0.925] | **0.8815 [0.865–0.896]** | +0.032 [0.025–0.040] |
| Ekim, nadir missense | 0.9717 [0.966–0.978] | **0.8999 [0.887–0.914]** | +0.072 [0.061–0.082] |
| Ekim, AF gözlenen | 0.9220 [0.902–0.938] | **0.8775 [0.854–0.897]** | +0.045 [0.033–0.057] |

Eğitimi yalnızca missense (33,640 satır) veya yalnızca nadir missense (16,988 satır) satırlarıyla yapmak sonucu değiştirmedi. Bütün kümelerde AF'siz model için fark en fazla 0.003 AUC (ör. variant_test nadir missense: 0.8981 / 0.8987 / 0.8980).

## Yorum

1. Modelin gerçekten kullanılacağı popülasyon nadir missense varyantlar. AF'siz, dürüst performans burada **~0.90 AUC** (0.896–0.901). Yayınlanan 0.994 bu popülasyonu temsil etmiyor.
2. AF'nin katkısı hedef popülasyonda daha da büyük: +0.057–0.072, genel testte +0.033–0.037 idi. Bunun büyük kısmı "AF yok" kısayolu. Nadir missense içinde AF'si olmayanların %95'i patojenik; bu PM2 kriterinin yansıması.
3. AF'nin gözlendiği ≤1e-4 aralığında bile AF hâlâ 0.025–0.045 ekliyor. Bu, "aşırı düşük frekans" (PM2) bilgisinin modele sızması.
4. AF'siz model, tek başına CADD'yi yalnızca ~0.005–0.008 AUC geçiyor (0.890 → 0.898). Model şu an pratikte "kalibre edilmiş CADD + biraz phyloP". Özellik setini güçlendirmek (Adım 2: ESM1b, AlphaMissense vb.) bu yüzden gerekli.

---

# Adım 2: Düşük döngüsellikli dbNSFP skorlarıyla AF'siz model

Scriptler: `experiments/annotate_v3.py` (anotasyon), `experiments/v3_features_eval.py` (eğitim/değerlendirme).
Kaynak: myvariant.info, dbNSFP 4.8a (build 2025-06-24). 108,414 etiketli kimliğin tamamı bulundu. Çekilen CADD değerleri dondurulmuş dataset'tekiyle %100 aynı (|fark| ≤ 0.05).
Çıktılar: `data/processed/v3_scores.parquet`, `artifacts/experiments/v3_features/v3_metrics.json`.
Bölmeler research-v2 ile aynı. Eğitim missense fit satırlarıyla yapıldı (33,640 satır); kalibrasyon ayrı satırlarda.

## Özellik seçimi ve döngüsellik riski

| Skor | Eğitimde ClinVar/HGMD etiketi? | Kullanım |
|---|---|---|
| ESM1b | Hayır (denetimsiz protein dil modeli) | Temiz |
| SIFT4G, PROVEAN | Hayır (homoloji) | Temiz |
| GERP 91 memeli, phyloP 100/470, phastCons | Hayır (korunmuşluk) | Temiz |
| AlphaMissense | ClinVar yok; popülasyon frekansını zayıf etiket olarak kullanır | Kısmi risk |
| CADD | ClinVar yok; küratörler PP3/BP4 olarak kullanabilir | Kısmi risk |
| REVEL | HGMD/ESP ile eğitilmiş, ClinVar örtüşmesi var | Yalnızca karşılaştırma |

## Tek skor AUC (nadir missense)

| Skor | variant_test | gene_test | Ekim yeni etiketli |
|---|---:|---:|---:|
| CADD | 0.890 | 0.889 | 0.892 |
| ESM1b | 0.927 | 0.904 | 0.926 |
| AlphaMissense | 0.946 | 0.946 | 0.956 |
| PROVEAN | 0.904 | 0.906 | 0.898 |
| SIFT4G | 0.886 | 0.888 | 0.888 |
| phyloP 100way | 0.819 | 0.806 | 0.837 |
| REVEL (karşılaştırma) | 0.968 | 0.966 | 0.961 |

## Model ROC-AUC [%95 GA], nadir missense

| Model | variant_test (5,215) | gene_test (4,442) | Ekim (2,462) | variant AF gözlenen | gene AF gözlenen | Ekim AF gözlenen |
|---|---:|---:|---:|---:|---:|---:|
| S1 CADD+phyloP (Adım 1) | 0.899 [0.889–0.909] | 0.894 [0.882–0.906] | 0.901 [0.888–0.914] | 0.880 | 0.879 | 0.876 |
| **S2 temiz** | **0.950 [0.943–0.956]** | **0.943 [0.934–0.951]** | **0.947 [0.938–0.956]** | 0.927 | 0.927 | 0.924 |
| S3 temiz + AlphaMissense | 0.960 [0.955–0.966] | 0.958 [0.951–0.965] | 0.963 [0.955–0.970] | 0.931 | 0.939 | 0.934 |
| **S4 temiz + AM + CADD** | **0.962 [0.957–0.968]** | **0.961 [0.954–0.967]** | **0.964 [0.956–0.971]** | 0.935 | 0.945 | 0.938 |
| S5 referans: S4 + AF (döngüsel) | 0.979 | 0.978 | 0.985 | 0.952 | 0.959 | 0.961 |

Eşleştirilmiş fark, S1'e göre (paired bootstrap %95 GA):
- S2: variant +0.051 [0.043–0.059], gene +0.049 [0.039–0.058], Ekim +0.046 [0.036–0.056].
- S4: variant +0.063 [0.056–0.071], gene +0.066 [0.057–0.076], Ekim +0.063 [0.053–0.072].

## Yorum

1. Temiz özelliklerle (S2) nadir missense AUC'si 0.894–0.901'den **0.943–0.950**'ye çıktı. AF'siz CADD modelinin kaybettiğinin büyük kısmı geri geldi ve bunun için AF ya da etiketli eğitilmiş bir araç gerekmedi. Kazanç AF'nin gözlendiği alt kümede de aynı büyüklükte (+0.047–0.049), yani "AF yok" kısayolundan gelmiyor.
2. En iyi AF'siz model (S4) **~0.961–0.964**. Bu, REVEL'in tek başına aldığı skorla (0.961–0.968) aynı düzeyde. Üstelik REVEL'in ClinVar örtüşmesi var, S4'te yok.
3. AF eklemek (S5) hâlâ +0.017–0.021 ekliyor. Bu, Adım 0–1'de gösterilen döngüsel katkı; ana model olarak kullanılmamalı.
4. Gen ayrık test (gene_test) ile varyant testi arasında belirgin fark yok. Gen düzeyinde bir ezberleme görünmüyor.

## Sınırlar

- Tüm in-silico skorlar ClinVar küratörlerince PP3/BP4 kanıtı olarak kullanılmış olabilir. ClinVar etiketiyle test edildikçe bu tip döngüsellik tamamen dışlanamaz. Adım 3 (ClinGen erepo kanıt kodları ve MAVE fonksiyonel verisi) bu yüzden gerekli.
- Sonuçlar dbNSFP 4.8a'ya bağlı; ESM1b kapsamı missense fit satırlarında %94.
- Missense, `Name` proxy'si ile belirlendi; VEP sonucu değil.

---

# Adım 3: Bağımsız dış doğrulama (ClinGen erepo + BAP1 SGE)

Scriptler: `experiments/prepare_erepo.py`, `experiments/prepare_mave_bap1.py`, `experiments/annotate_external.py`, `experiments/external_validation.py`.
Çıktı: `artifacts/experiments/external_validation/external_metrics.json`. Ham veri: `data/external/`.

## Veri

- **ClinGen Evidence Repository** (erepo.clinicalgenome.org, tam export, 13,934 satır). Retracted ve VUS kayıtlar çıkarıldı, GRCh38 SNV kimliği tekil olanlar alındı: 7,886 varyant. Uygulanan ("Met") ACMG kodlarına göre her varyant işaretlendi. Hesaplamalı kodlar: PP3, BP4, BP7. Frekans kodları: BA1, BS1, PM2. Analizde dbNSFP anotasyonu bulunan missense varyantlar kullanıldı: 3,874.
- **BAP1 saturation genome editing** (Waters ve ark. 2024, MaveDB `urn:mavedb:00000662-0-1`, CC BY 4.0). ENST00000460680.6 c. pozisyonları Ensembl ekson/translation koordinatlarıyla GRCh38'e çevrildi. Doğrulama: dbNSFP gen adı %100 BAP1, aminoasit pozisyonu %99.98 uyumlu. Etiketler MaveDB "BBI-GMM" eşiklerinden: anormal ≤ −0.026891 → 1, normal ≥ −0.018367 → 0, aradaki indeterminate değerler çıkarıldı. Missense: 633 anormal, 3,957 normal. Etiket tamamen deneysel; ClinVar'dan, AF'den ve in-silico skorlardan bağımsız.

## Tasarım

- erepo varyantlarının 5,480'i (%70) Temmuz eğitim kohortunda da vardı. Modeller dış setler çıkarılarak yeniden eğitildi:
  - **varyant ayrık:** erepo ve BAP1 kimlikleri fit/kalibrasyondan çıkarıldı; fit 31,571, kalibrasyon 7,841 satır.
  - **gen ayrık:** buna ek olarak 140 erepo geni ve BAP1'i içeren tüm gen grupları çıkarıldı; fit 27,405, kalibrasyon 6,807 satır.
- Diğer her şey Adım 2 ile aynı (S1–S5 özellik setleri, missense eğitimi, gen gruplu CV, ayrı kalibrasyon).

## ROC-AUC [%95 GA], gen ayrık tasarım

| Değerlendirme kümesi | N (B / P) | S1 CADD+phyloP | S2 temiz | **S4 temiz+AM+CADD** | S5 S4+AF (döngüsel) |
|---|---:|---:|---:|---:|---:|
| erepo, tümü | 3,874 (1,195 / 2,679) | 0.890 [0.878–0.901] | 0.926 [0.917–0.935] | **0.950 [0.942–0.958]** | 0.973 [0.968–0.978] |
| erepo, hesaplamalı kod yok | 1,224 (692 / 532) | 0.786 [0.758–0.810] | 0.841 [0.818–0.862] | **0.898 [0.882–0.914]** | 0.954 [0.942–0.964] |
| erepo, hesaplamalı ve frekans kodu yok | 186 (129 / 57) | 0.759 [0.679–0.832] | 0.720 [0.645–0.796] | **0.802 [0.734–0.869]** | 0.714 [0.623–0.804] |
| erepo, tümü, nadir | 3,239 (626 / 2,613) | 0.884 | 0.923 | **0.944** | 0.962 |
| erepo, hesaplamalı kod yok, nadir | 859 (341 / 518) | 0.781 | 0.835 | **0.893** | 0.934 |
| erepo, hesaplamalı ve frekans kodu yok, nadir | 158 (114 / 44) | 0.769 | 0.752 | **0.841 [0.778–0.892]** | 0.825 |
| **BAP1 SGE missense (fonksiyonel)** | 4,590 (3,957 / 633) | 0.827 [0.808–0.843] | 0.894 [0.879–0.907] | **0.899 [0.883–0.911]** | 0.898 [0.883–0.911] |

BAP1 SGE için eşiksiz Spearman ρ (model skoru ile −SGE skoru, 4,740 missense): S1 0.306, S2 0.374, S4 0.391, S5 0.396.
Varyant ayrık tasarımın sonuçları gen ayrık tasarımla 0.004 AUC içinde aynı. Gen düzeyinde bir ezberleme görünmüyor.

Tek skor AUC'leri:

| Küme | CADD | ESM1b | AlphaMissense | PROVEAN | SIFT4G | REVEL (karşılaştırma) |
|---|---:|---:|---:|---:|---:|---:|
| BAP1 SGE missense | 0.832 | **0.890** | 0.867 | 0.870 | 0.827 | 0.884 |
| erepo, tümü | 0.886 | 0.888 | 0.933 | 0.881 | 0.868 | 0.948 |
| erepo, hesaplamalı kod yok | 0.780 | 0.780 | 0.865 | 0.751 | 0.784 | 0.847 |

## Yorum

1. **AF döngüselliği bağımsız etiketlerde doğrulandı.** ClinVar benzeri etiketlerde AF hâlâ +0.023 (erepo tümü) ile +0.056 (hesaplamalı kod yok) AUC ekliyor. Etiketin frekans kodu kullanmadığı alt kümede ise AF eklemek performansı **düşürüyor**: S4 0.802 → S5 0.714. Etiketin tamamen deneysel olduğu BAP1 SGE'de AF hiçbir şey eklemiyor (0.899 → 0.898). Not: 186 varyantlık alt kümede güven aralıkları örtüşüyor; bu eşleştirilmiş bir test değil, nokta tahmini.
2. **Modelin bağımsız fonksiyonel veride gerçek performansı ~0.90.** BAP1 SGE'de S4 0.899, S2 0.894; tek başına CADD 0.832 ve S1 0.827. Adım 2'deki özellik genişletmesi bağımsız veride de +0.07 AUC kazandırıyor.
3. **Hesaplamalı kodlar çıkarıldıkça performans düşüyor:** S4'te 0.950 → 0.898 → 0.802. Bu, PP3/BP4 kaynaklı döngüselliğin varlığıyla uyumlu. Ancak bir seçim etkisi de var: VCEP'ler PP3/BP4'ü genellikle predictor skoru ara bölgedeyken "not met" olarak işaretliyor. Bu alt küme bu yüzden zor varyantlarla zenginleşmiş durumda ve iki etki burada ayrılamıyor. Bu ayrımı yapan temiz kontrol BAP1 SGE.
4. **En temiz kombinasyon:** BAP1'de ESM1b tek başına (0.890) REVEL'i (0.884) ve AlphaMissense'i (0.867) geçiyor. S2 (yalnızca etiket görmemiş skorlar) 0.894 ile S4'e çok yakın.

## Sınırlar

- MAVE doğrulaması tek gen (BAP1). Gen bazında performans değişebilir; BRCA1, RAD51C ve VHL SGE setleri eklenerek genişletilebilir.
- BBI-GMM eşikleri ClinVar ile kalibre edilmiş; ölçümün kendisi bağımsız. Eşiksiz Spearman sonucu da aynı sıralamayı veriyor.
- erepo "hesaplamalı ve frekans kodu yok" alt kümesi küçük (186); güven aralıkları geniş.
- Missense tanımı erepo'da Name proxy'si, BAP1'de dbNSFP aminoasit alanı.
