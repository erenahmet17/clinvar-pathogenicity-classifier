"""Explicit preparation, annotation, training and scoring commands."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .annotation import AnnotationClient
from .data import merge_annotations, read_cohort
from .evaluate import external_cohort, metrics, subgroup_report
from .inference import Predictor
from .io import sha256, write_json, write_parquet
from .train import train_run


def score_vus(dataset, bundle):
    predictor = Predictor(bundle)
    frame = dataset[dataset.is_vus].copy().reset_index(drop=True)
    excluded = frame.hgvs_id.isin(predictor.manifest.get("known_scope_exclusions", {}))
    excluded_count = int(excluded.sum())
    frame = frame.loc[~excluded].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No in-scope VUS rows available for scoring.")
    output = predictor.predict(frame)
    frame["model_score"], frame["direction"] = output.model_score, output.direction
    path = Path(bundle) / "vus_scores.parquet"
    write_parquet(
        path,
        frame[
            [
                "hgvs_id",
                "AlleleID",
                "VariationID",
                "GeneSymbol",
                "Name",
                "model_score",
                "direction",
                "gnomad_af",
                "af_available",
            ]
        ],
    )
    manifest = predictor.manifest
    manifest["optional_files"][path.name] = sha256(path)
    manifest["vus_counts"] = {
        "n_scored": len(frame),
        "by_direction": frame.direction.value_counts().to_dict(),
        "unknown_af": int(frame.gnomad_af.isna().sum()),
        "known_out_of_scope_not_scored": excluded_count,
    }
    write_json(Path(bundle) / "manifest.json", manifest)
    return manifest["vus_counts"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("snapshot", type=Path)
    prep.add_argument("out", type=Path)
    prep.add_argument("--source-url")
    annotate = sub.add_parser("annotate")
    annotate.add_argument("cohort", type=Path)
    annotate.add_argument("cache", type=Path)
    annotate.add_argument("out", type=Path)
    annotate.add_argument("--workers", type=int, default=4)
    train = sub.add_parser("train")
    train.add_argument("dataset", type=Path)
    train.add_argument("provenance", type=Path)
    train.add_argument("out", type=Path)
    train.add_argument("--iterations", type=int, default=12)
    train.add_argument("--folds", type=int, default=5)
    score = sub.add_parser("score-vus")
    score.add_argument("dataset", type=Path)
    score.add_argument("bundle", type=Path)
    ext = sub.add_parser("evaluate-external")
    ext.add_argument("earlier_dataset", type=Path)
    ext.add_argument("later_dataset", type=Path)
    ext.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        _, manifest = read_cohort(args.snapshot, args.out, args.source_url)
        print(json.dumps(manifest["counts"]))
    elif args.command == "annotate":
        cohort = pd.read_parquet(args.cohort)
        client = AnnotationClient(args.cache)
        metadata = client.pin_metadata()
        annotations = client.annotate(cohort.hgvs_id, workers=args.workers)
        # Compare source versions again before completing a dataset.
        client.pin_metadata()
        write_parquet(args.out / "annotations.parquet", annotations)
        dataset, retention = merge_annotations(cohort, annotations)
        write_parquet(args.out / "dataset.parquet", dataset)
        cohort_manifest = json.loads(
            args.cohort.with_name("cohort_manifest.json").read_text(encoding="utf-8")
        )
        provenance = {
            "cohort": cohort_manifest,
            "annotation": metadata,
            "retention": retention,
            "dataset_sha256": sha256(args.out / "dataset.parquet"),
        }
        exclusions = pd.read_parquet(args.cohort.with_name("scope_exclusions.parquet"))
        provenance["known_scope_exclusions"] = dict(
            zip(exclusions.hgvs_id, exclusions.ClinicalSignificance)
        )
        write_json(args.out / "dataset_manifest.json", provenance)
        print(json.dumps(retention))
    elif args.command == "train":
        dataset = pd.read_parquet(args.dataset)
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        if sha256(args.dataset) != provenance["dataset_sha256"]:
            raise ValueError("Dataset hash differs from provenance.")
        report = train_run(
            dataset, args.out, provenance, iterations=args.iterations, folds=args.folds
        )
        print(
            json.dumps(
                {
                    name: value["calibrated_score"]["roc_auc"]
                    for name, value in report["test"].items()
                }
            )
        )
    elif args.command == "score-vus":
        result = score_vus(pd.read_parquet(args.dataset), args.bundle)
        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["vus_source"] = json.loads(
            args.dataset.with_name("dataset_manifest.json").read_text(encoding="utf-8")
        )
        write_json(args.bundle / "manifest.json", manifest)
        print(json.dumps(result))
    else:
        earlier, later = pd.read_parquet(args.earlier_dataset), pd.read_parquet(args.later_dataset)
        predictor = Predictor(args.bundle)
        new, reclassified = external_cohort(
            earlier[earlier.label.notna()], later, earlier[earlier.is_vus]
        )
        report = {
            "earlier_dataset_sha256": sha256(args.earlier_dataset),
            "later_dataset_sha256": sha256(args.later_dataset),
            "interpretation": "Temporal label evaluation with frozen annotation sources; not independent functional validation. Label reclassification may depend on computational evidence.",
            "groups": {},
        }
        frames = []
        for name, frame in [("new_labeled_variants", new), ("reclassified_vus", reclassified)]:
            if frame.empty:
                report["groups"][name] = {"n": 0, "roc_auc": None}
                continue
            frame = frame.copy().reset_index(drop=True)
            predictions = predictor.predict(frame)
            frame["model_score"] = predictions.model_score
            frame["evaluation_group"] = name
            report["groups"][name] = {
                **metrics(frame.label, frame.model_score, bootstrap=500),
                "subgroups": subgroup_report(frame, frame.model_score),
            }
            frames.append(
                frame[
                    [
                        "hgvs_id",
                        "VariationID",
                        "GeneSymbol",
                        "Name",
                        "ClinicalSignificance",
                        "LastEvaluated",
                        "label",
                        "model_score",
                        "evaluation_group",
                    ]
                ]
            )
        write_json(args.bundle / "external_metrics.json", report)
        if frames:
            write_parquet(
                args.bundle / "external_predictions.parquet", pd.concat(frames, ignore_index=True)
            )
        manifest = predictor.manifest
        manifest["optional_files"]["external_metrics.json"] = sha256(
            args.bundle / "external_metrics.json"
        )
        if frames:
            manifest["optional_files"]["external_predictions.parquet"] = sha256(
                args.bundle / "external_predictions.parquet"
            )
        write_json(args.bundle / "manifest.json", manifest)
        print(json.dumps({key: value.get("n") for key, value in report["groups"].items()}))


if __name__ == "__main__":
    main()
