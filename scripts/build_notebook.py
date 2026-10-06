"""Generate an offline, restart-and-run-all research notebook around shared modules."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    add(
        "markdown",
        "# ClinVar SNV araştırma akışı — v2\n\nBu notebook çevrimdışı inceleme katmanıdır. Veri indirme, anotasyon ve eğitim işlemleri ortak `clinvar_classifier` paketinin CLI komutlarıyla yapılır. Eski çalışmadan kalan hücre çıktıları güncel sonuç gibi gösterilmez. Model skoru klinik patojenite olasılığı değildir.\n",
    )
    add(
        "code",
        "from pathlib import Path\nimport json\nimport sys\nimport pandas as pd\n\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\nsys.path.insert(0, str(ROOT / 'src'))\nfrom clinvar_classifier.data import map_label\nfrom clinvar_classifier.features import make_features\nfrom clinvar_classifier.inference import Predictor\nBUNDLE = ROOT / 'artifacts' / 'research-v2'\n",
    )
    add(
        "markdown",
        "## Açık etiket ve eksiklik politikası\n\nP/LP ve B/LB etiketleri açık izin listesiyle eşlenir. Düşük penetrans, risk alleli ve karışık/çelişkili etiketler ikili hedefe dahil edilmez. Eksik AF ve gözlenen sıfır ayrı tutulur.\n",
    )
    add(
        "code",
        "labels = ['Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic', 'Benign', 'Likely benign', 'Benign/Likely benign', 'Uncertain significance', 'Pathogenic, low penetrance']\nprint(pd.DataFrame({'ClinicalSignificance': labels, 'label': [map_label(x) for x in labels]}).to_string(index=False))\nraw = pd.DataFrame({'cadd_phred': [20.,20.], 'phylop_100way': [2.,2.], 'gnomad_af': [None,0.], 'af_available': [False,True]})\nprint(make_features(raw))\n",
    )
    add(
        "markdown",
        "## Tekrarlanabilir tam koşu\n\nRepo root'unda `scripts/download_snapshot.py` ile tarihli snapshot indirilir. Ardından `prepare`, `annotate`, `train`, `evaluate-external`, `score-vus` komutları kullanılır; ayrıntılar README'dedir.\n\nModel seçimi yalnızca fit partition ve gen ayrık CV fold'larında yapılır. Kalibrasyon için farklı satırlar kullanılır. Varyant ve bağlı gen grubu testleri tüm seçimler bittikten sonra açılır. Ekim snapshot'ındaki yeni etiketli ve Temmuz VUS'tan değişen kayıtlar, Temmuz'daki tüm etiketli kohorttan kimlik bazında ayrılır.\n",
    )
    add(
        "code",
        "manifest = json.loads((BUNDLE / 'manifest.json').read_text(encoding='utf-8')) if (BUNDLE / 'manifest.json').exists() else None\nif manifest:\n    print({k: manifest.get(k) for k in ['run_id','n_fit','n_calibration','feature_contract','validation_status']})\n    print('Training snapshot SHA256:', manifest.get('provenance', {}).get('cohort', {}).get('snapshot_sha256'))\nelse:\n    print('No research-v2 bundle yet. Run the documented training commands first; no placeholder performance is displayed.')\n",
    )
    add(
        "markdown",
        "## Kayıtlı değerlendirme ve ablation\n\nBu hücre sonuç dosyalarını okur; test verisiyle yeniden model veya özellik seçimi yapmaz. Benzer CV ortalamaları istatistiksel eşdeğerlik kanıtı sayılmaz.\n",
    )
    add(
        "code",
        "if manifest and (BUNDLE / 'metrics.json').exists():\n    evaluation = json.loads((BUNDLE / 'metrics.json').read_text(encoding='utf-8'))\n    summaries = []\n    for name, result in evaluation['test'].items():\n        v = result['calibrated_score']\n        summaries.append({'partition':name,'n':v['n'],'ROC-AUC':v['roc_auc'],'Brier':v['brier'],'log_loss':v['log_loss']})\n    print(pd.DataFrame(summaries).to_string(index=False))\n    scores = pd.DataFrame(evaluation['development']['fold_scores'])\n    print(scores.groupby('model').roc_auc.agg(['mean','std']))\n",
    )
    add(
        "markdown",
        "## Bağımsız satırlarda kalibrasyon ve zaman kontrolü\n\nKalibrasyon, bu veri dağılımında skorları ayarlamaktır; hasta riski veya penetrans doğrulaması değildir. Zaman bazlı ClinVar etiket kontrolü bağımsız fonksiyonel deney yerine geçmez. Çıktıdaki alt gruplar ve kutu örneklem sayıları birlikte okunmalıdır.\n",
    )
    add(
        "code",
        "if manifest and (BUNDLE / 'external_metrics.json').exists():\n    external = json.loads((BUNDLE / 'external_metrics.json').read_text(encoding='utf-8'))\n    print(external['interpretation'])\n    print(pd.DataFrame({name:{k:value.get(k) for k in ['n','n_benign','n_pathogenic','roc_auc','brier']} for name,value in external['groups'].items()}).T)\n",
    )
    add(
        "markdown",
        "## Model davranışı ve kapsam\n\nTree SHAP raw log-odds marginini açıklar; klinik kanıt veya bağımsız nedensel açıklama değildir. HFE p.Cys282Tyr, düşük penetrans/sık patojenik allel sınırlamasının belgelenmiş örneğidir ve bilinen benign kontrol olarak sunulmaz.\n",
    )
    add(
        "code",
        "if manifest:\n    predictor = Predictor(BUNDLE)\n    example = pd.DataFrame([{'cadd_phred':25.,'phylop_100way':2.,'gnomad_af':None,'af_available':False}])\n    print(predictor.predict(example))\n    features, values, base = predictor.explain(example)\n    print(pd.Series(values[0], index=features.columns, name='Tree SHAP raw-margin contribution'))\n    print('Base margin:', base[0])\n",
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"research-v2-{i + 1}"
    (ROOT / "clinvar.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
