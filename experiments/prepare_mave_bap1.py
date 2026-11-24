"""Step 3b: BAP1 saturation genome editing (Waters et al. 2024, MaveDB urn:mavedb:00000662-0-1)
-> GRCh38 coding SNV identities with functional labels.

c. positions on ENST00000460680.6 are mapped to GRCh38 with Ensembl exon/translation
coordinates. Labels use the MaveDB "BBI - GMM" score thresholds: abnormal <= -0.026891
(label 1), normal >= -0.018367 (label 0), indeterminate excluded. The continuous score is
kept for threshold-free (Spearman) evaluation.

    python experiments/prepare_mave_bap1.py data/external/mave/bap1_scores.csv \
        data/external/mave/bap1_snv.parquet
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

TRANSCRIPT = "ENST00000460680"
ABNORMAL_MAX, NORMAL_MIN = -0.026891, -0.018367
COMP = str.maketrans("ACGT", "TGCA")


def cds_map():
    r = requests.get(
        f"https://rest.ensembl.org/lookup/id/{TRANSCRIPT}",
        params={"expand": 1, "content-type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    t = r.json()
    strand, chrom = t["strand"], t["seq_region_name"]
    lo, hi = t["Translation"]["start"], t["Translation"]["end"]
    exons = sorted(t["Exon"], key=lambda e: e["start"], reverse=strand == -1)
    positions = []
    for e in exons:
        rng = range(e["start"], e["end"] + 1)
        for g in rng if strand == 1 else reversed(rng):
            if lo <= g <= hi:
                positions.append(g)
    meta = {"transcript": t["id"] + "." + str(t["version"]), "assembly": t["assembly_name"],
            "chrom": chrom, "strand": strand, "cds_length": len(positions)}
    return chrom, strand, positions, meta


def main(src, out):
    chrom, strand, cds, meta = cds_map()
    print(meta)
    assert meta["transcript"] == "ENST00000460680.6" and meta["cds_length"] % 3 == 0
    d = pd.read_csv(src)
    m = d.hgvs_nt.str.extract(r":c\.(\d+)([ACGT])>([ACGT])$")
    d = d[m[0].notna()].copy()
    m = m.loc[d.index]
    rows = []
    for (cpos, ref, alt), score in zip(m.itertuples(index=False), d.score):
        g = cds[int(cpos) - 1]
        if strand == -1:
            ref, alt = ref.translate(COMP), alt.translate(COMP)
        rows.append({"hgvs_id": f"chr{chrom}:g.{g}{ref}>{alt}", "c_pos": int(cpos),
                     "mave_score": float(score)})
    f = pd.DataFrame(rows)
    f["GeneSymbol"] = "BAP1"
    f["label"] = pd.NA
    f.loc[f.mave_score <= ABNORMAL_MAX, "label"] = 1
    f.loc[f.mave_score >= NORMAL_MIN, "label"] = 0
    f = f.drop_duplicates("hgvs_id")
    f.to_parquet(out, index=False)
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(len(f), f.label.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
