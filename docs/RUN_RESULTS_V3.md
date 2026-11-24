# Koşu sonuçları — `research-v3` (S4, `missense-noaf-v3`)

Oluşturma: 2026-11-24. Bütün değerler bu bundle'daki dosyalardan okunur.

Kapsam: 2+ review-status GRCh38 dbNSFP missense SNVs with CADD/phyloP; population frequency not used as a feature; special penetrance/risk labels excluded; no inferred clinical classification.

Özellikler: `esm1b`, `sift4g`, `provean`, `gerp_91way`, `phylop_100way`, `phylop_470way`, `phastcons_100way`, `alphamissense`, `cadd_phred`. **gnomAD AF modele girmez**; gerekçe ve kanıt: [experiments/CIRCULARITY_RESULTS.md](../experiments/CIRCULARITY_RESULTS.md).

Kapsam sayımları: {"labeled_rows": 101415, "in_scope_missense_rows": 65883}. Fit: **33,665**, kalibrasyon: **8,417**.

## İç test (kalibre skor)

| Partition | N | B/LB | P/LP | ROC-AUC | %95 GA (satır bootstrap) | AP | Brier |
|---|---:|---:|---:|---:|---|---:|---:|
| variant_test | 10,521 | 6,568 | 3,953 | 0.9709 | 0.9682–0.9735 | 0.9535 | 0.0627 |
| gene_test | 13,280 | 8,377 | 4,903 | 0.9694 | 0.9667–0.9720 | 0.9496 | 0.0637 |

## Nadir ve AF gözlenen alt gruplar

| Partition | Alt grup | N | B/LB | P/LP | ROC-AUC |
|---|---|---:|---:|---:|---:|
| variant_test | rare_af_le_1e_4_or_unknown | 5,155 | 1,304 | 3,851 | 0.9673 |
| variant_test | af_observed_le_1e_4 | 2,096 | 1,082 | 1,014 | 0.9459 |
| variant_test | af_unknown | 3,059 | 222 | 2,837 | 0.9846 |
| variant_test | af_observed_gt_1e_3 | 4,031 | 4,018 | 13 | 0.8619 |
| gene_test | rare_af_le_1e_4_or_unknown | 6,452 | 1,681 | 4,771 | 0.9684 |
| gene_test | af_observed_le_1e_4 | 2,666 | 1,334 | 1,332 | 0.9484 |
| gene_test | af_unknown | 3,786 | 347 | 3,439 | 0.9795 |
| gene_test | af_observed_gt_1e_3 | 5,202 | 5,187 | 15 | 0.8376 |

## Tek skor karşılaştırması (test satırları)

| Partition | cadd | phylop | inverted_observed_af | esm1b | alphamissense | sift4g | provean |
|---|---:|---:|---:|---:|---:|---:|---:|
| variant_test | 0.9188 | 0.8604 | 0.9371 | 0.9265 | 0.9584 | 0.8771 | 0.9034 |
| gene_test | 0.9225 | 0.8550 | 0.9365 | 0.9158 | 0.9547 | 0.8749 | 0.8877 |

## Geliştirme CV'si (yalnızca fit satırları, gen gruplu 5 kat)

| Model / ablasyon | ROC-AUC ortalama | SD |
|---|---:|---:|
| XGBoost CADD only | 0.9184 | 0.0102 |
| XGBoost CADD + phyloP (no AF) | 0.9230 | 0.0095 |
| XGBoost label-free scores only (S2) | 0.9493 | 0.0065 |
| XGBoost S2 + AlphaMissense | 0.9649 | 0.0057 |
| Logistic regression (fold-local imputation/scaling) | 0.9669 | 0.0058 |
| Random forest | 0.9680 | 0.0054 |
| XGBoost baseline | 0.9685 | 0.0052 |
| XGBoost selected parameters | 0.9693 | 0.0054 |
| Reference only, not selectable: + gnomAD AF (circular) | 0.9936 | 0.0013 |

"Reference only" satırı AF eklenmiş halidir; seçilemez, döngüsel katkının büyüklüğünü gösterir. Seçilen parametreler: `{"subsample": 0.8, "n_estimators": 400, "min_child_weight": 5, "max_depth": 3, "learning_rate": 0.03, "colsample_bytree": 1.0}`.

## Zaman bazlı dış kontrol (Temmuz → Ekim 2026)

| Grup | N | B/LB | P/LP | ROC-AUC | %95 GA | Brier | Kapsam dışı (non-missense) |
|---|---:|---:|---:|---:|---|---:|---:|
| new_labeled_variants | 4,541 | 2,997 | 1,544 | 0.9727 | 0.9685–0.9763 | 0.0603 | 2,455 |
| reclassified_vus | 15 | 8 | 7 | 0.5714 | 0.2500–0.8750 | 0.3633 | 0 |

## VUS skorları

```json
{
  "n_scored": 296423,
  "by_direction": {
    "Benign direction": 141799,
    "Indeterminate": 123258,
    "Pathogenic direction": 31366
  },
  "unknown_af": 133295,
  "known_out_of_scope_not_scored": 0,
  "outside_feature_contract_not_scored": 2462
}
```

## Kaynak

Eğitim snapshot SHA-256: `b3da921c707ff50cb4822f04d961834a9e89ef864b4c28bb5308245b93ac4077`

Dataset SHA-256: `65480c4a528864ddb787c027d644170c30b884546ac7fba13277fed1c40fa8b3`

Anotasyon namespace: `3923d1685accee1e28fbbed30102b593aa6e7ae2a1f46c5d8fb394cc49dfa338`

Bootstrap aralıkları gen kümelenmesini modellemez. Skorlar ClinVar sınıflarını değiştirmez; 0.1/0.9 bantları ACMG/AMP kriteri değildir.
