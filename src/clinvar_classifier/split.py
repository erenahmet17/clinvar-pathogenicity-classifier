"""Persisted variant and connected gene-group partitions before model selection."""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .schemas import ValidationError


def gene_groups(frame: pd.DataFrame) -> pd.Series:
    parents = {}

    def find(gene):
        parents.setdefault(gene, gene)
        if parents[gene] != gene:
            parents[gene] = find(parents[gene])
        return parents[gene]

    tokens = []
    for text in frame.GeneSymbol.fillna("").astype(str):
        genes = sorted(
            {
                g.strip().upper()
                for g in re.split(r"[;,|]", text)
                if g.strip() not in {"", "-", "na"}
            }
        ) or ["__UNKNOWN_GENE__"]
        tokens.append(genes)
        for gene in genes:
            parents[find(gene)] = find(genes[0])
    return pd.Series([find(genes[0]) for genes in tokens], index=frame.index, name="gene_group")


def assert_disjoint(*frames, gene_disjoint=False):
    sets = [set(frame.hgvs_id) for frame in frames]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            if a & b:
                raise ValidationError("Variant identity leakage across partitions.")
    if gene_disjoint:
        groups = [set(frame.gene_group) for frame in frames]
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                if a & b:
                    raise ValidationError("Connected gene-group leakage across partitions.")


def make_splits(frame: pd.DataFrame, seed=42) -> pd.DataFrame:
    frame = frame.sort_values("hgvs_id").reset_index(drop=True).copy()
    if frame.hgvs_id.duplicated().any() or frame.label.isna().any():
        raise ValidationError("Splitting requires unique labeled variants.")
    frame["gene_group"] = gene_groups(frame)
    if frame.gene_group.nunique() < 10:
        raise ValidationError(
            "At least 10 connected gene groups required for meaningful held-out evaluation."
        )
    develop, gene_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            frame, frame.label, frame.gene_group
        )
    )
    fit_calibration, variant_test = train_test_split(
        develop, test_size=0.2, stratify=frame.iloc[develop].label, random_state=seed
    )
    fit, calibration = train_test_split(
        fit_calibration,
        test_size=0.2,
        stratify=frame.iloc[fit_calibration].label,
        random_state=seed + 1,
    )
    frame["split"] = ""
    for name, ids in [
        ("fit", fit),
        ("calibration", calibration),
        ("variant_test", variant_test),
        ("gene_test", gene_test),
    ]:
        frame.loc[ids, "split"] = name
        if frame.iloc[ids].label.nunique() < 2:
            raise ValidationError(
                f"Partition {name} has only one class; revise the predeclared split design."
            )
    assert_disjoint(
        *(
            frame[frame.split == name]
            for name in ["fit", "calibration", "variant_test", "gene_test"]
        )
    )
    assert_disjoint(
        frame[frame.split != "gene_test"], frame[frame.split == "gene_test"], gene_disjoint=True
    )
    if (frame.split == "").any() or not np.isfinite(frame.label.astype(float)).all():
        raise ValidationError("Invalid split labels.")
    return frame
