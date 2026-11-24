"""Step 3a: ClinGen Evidence Repository -> GRCh38 SNV identities with evidence-code flags.

python experiments/prepare_erepo.py data/external/erepo/erepo.tsv data/external/erepo/erepo_snv.parquet
"""

import re
import sys

import pandas as pd

GRCH38 = {
    1: 11,
    2: 12,
    3: 12,
    4: 12,
    5: 10,
    6: 12,
    7: 14,
    8: 11,
    9: 12,
    10: 11,
    11: 10,
    12: 12,
    13: 11,
    14: 9,
    15: 10,
    16: 10,
    17: 11,
    18: 10,
    19: 10,
    20: 11,
    21: 9,
    22: 11,
    23: 11,
    24: 10,
}
SNV = re.compile(r"NC_0000(\d{2})\.(\d+):g\.(\d+)([ACGT])>([ACGT])\b")
LABELS = {"pathogenic": 1, "likely pathogenic": 1, "benign": 0, "likely benign": 0}
COMPUTATIONAL = {"PP3", "BP4", "BP7"}
FREQUENCY = {"BA1", "BS1", "PM2"}


def grch38_id(hgvs):
    ids = set()
    for chrom, ver, pos, ref, alt in SNV.findall(str(hgvs)):
        c = int(chrom)
        if GRCH38.get(c) == int(ver):
            name = {23: "X", 24: "Y"}.get(c, str(c))
            ids.add(f"chr{name}:g.{pos}{ref}>{alt}")
    return ids.pop() if len(ids) == 1 else None


def codes(text):
    return {
        c.strip().split("_")[0] for c in str(text).split(",") if c.strip() and c.strip() != "nan"
    }


def main(src, out):
    e = pd.read_csv(src, sep="\t", dtype=str)
    e = e[e["Retracted"].fillna("false").str.lower() != "true"]
    e["label"] = e["Assertion"].str.strip().str.lower().map(LABELS)
    e = e[e.label.notna()].copy()
    e["hgvs_id"] = e["HGVS Expressions"].map(grch38_id)
    e = e[e.hgvs_id.notna()].copy()
    met = e["Applied Evidence Codes (Met)"].map(codes)
    e["codes_met"] = met.map(lambda s: ",".join(sorted(s)))
    e["uses_computational"] = met.map(lambda s: bool(s & COMPUTATIONAL))
    e["uses_frequency"] = met.map(lambda s: bool(s & FREQUENCY))
    e["uses_functional"] = met.map(lambda s: bool(s & {"PS3", "BS3"}))
    e["Name"] = e["#Variation"]
    # Same identity classified for several diseases: keep only if labels agree.
    agree = e.groupby("hgvs_id").label.nunique().eq(1)
    e = e[e.hgvs_id.isin(agree[agree].index)]
    agg = (
        e.groupby("hgvs_id")
        .agg(
            label=("label", "first"),
            Name=("Name", "first"),
            GeneSymbol=("HGNC Gene Symbol", "first"),
            VariationID=("ClinVar Variation Id", "first"),
            expert_panel=("Expert Panel", "first"),
            uses_computational=("uses_computational", "any"),
            uses_frequency=("uses_frequency", "any"),
            uses_functional=("uses_functional", "any"),
            codes_met=("codes_met", lambda s: " | ".join(sorted(set(s)))),
        )
        .reset_index()
    )
    agg.to_parquet(out, index=False)
    print(len(agg), agg.label.value_counts().to_dict())
    print(agg.groupby(["uses_computational", "uses_frequency", "label"]).size())


if __name__ == "__main__":
    main(*sys.argv[1:3])
