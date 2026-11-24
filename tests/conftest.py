import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(84)
    rows = []
    for gene in range(30):
        for item in range(20):
            label = item % 2
            index = gene * 20 + item + 1
            rows.append(
                {
                    "hgvs_id": f"chr1:g.{index}A>G",
                    "AlleleID": str(index),
                    "VariationID": str(index),
                    "GeneSymbol": f"GENE{gene:02}",
                    "Name": f"GENE{gene:02}:c.{index}A>G (p.Arg1Gly)",
                    "label": label,
                    "is_vus": False,
                    "ClinicalSignificance": "Pathogenic" if label else "Benign",
                    "LastEvaluated": "Jan 01, 2026",
                    "cadd_phred": float(np.clip(10 + label * 18 + rng.normal(0, 4), 0, 99)),
                    "phylop_100way": float(rng.normal(3, 1)),
                    "gnomad_af": None if item % 5 == 0 else (1e-5 if label else 0.01),
                    "af_available": item % 5 != 0,
                    "esm1b": None
                    if item % 7 == 0
                    else float(np.clip(0.3 + 0.5 * label + rng.normal(0, 0.15), 0, 1)),
                    "alphamissense": float(np.clip(0.2 + 0.6 * label + rng.normal(0, 0.15), 0, 1)),
                    "sift4g": float(rng.uniform(0, 1)),
                    "provean": float(rng.uniform(0, 1)),
                    "gerp_91way": float(rng.normal(2, 1)),
                    "phylop_470way": float(rng.normal(2, 1)),
                    "phastcons_100way": float(rng.uniform(0, 1)),
                    "consequence": "stop_gained" if item == 19 else "missense",
                }
            )
    return pd.DataFrame(rows)
