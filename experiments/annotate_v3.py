"""Step 2a: add low-circularity dbNSFP predictors for labeled variants.

Pulls dbNSFP fields from myvariant.info (hg38) for the July-2026 labeled cohort and the
October-2026 labeled cohort, with a resumable per-batch JSON cache. Values are taken
as dbNSFP rankscores where a transcript-level list exists (dbNSFP computes them on the
transcript-aggregated score), otherwise the single score.

    python experiments/annotate_v3.py data/processed/2026-07/dataset.parquet \
        data/processed/2026-10/dataset.parquet data/annotation_cache_v3 \
        data/processed/v3_scores.parquet
"""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

URL = "https://myvariant.info/v1"
FIELDS = {
    # name: (dbNSFP path, circularity note)
    "esm1b": ("esm1b.rankscore", "unsupervised protein LM; no labels"),
    "alphamissense": ("alphamissense.rankscore", "no ClinVar; population frequency weak labels"),
    "sift4g": ("sift4g.converted_rankscore", "homology; no labels"),
    "provean": ("provean.converted_rankscore", "homology; no labels"),
    "gerp_91way": ("gerp.91_mammals.score", "conservation; no labels"),
    "phylop_470way": ("phylop.470way_mammalian.score", "conservation; no labels"),
    "phastcons_100way": ("phastcons.100way_vertebrate.score", "conservation; no labels"),
    "cadd_check": ("cadd.phred", "identity check against frozen dataset"),
    "revel": ("revel.rankscore", "COMPARATOR ONLY: trained on HGMD/ESP, ClinVar overlap"),
}
BATCH = 1000


def get(doc, path):
    for key in path.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    if isinstance(doc, list):
        values = {v for v in doc if v is not None}
        return values.pop() if len(values) == 1 else None
    return doc


def fetch(ids, session):
    for attempt in range(5):
        try:
            r = session.post(
                URL + "/variant",
                data={
                    "ids": ",".join(ids),
                    "fields": ",".join(
                        sorted({"dbnsfp." + p.split(".")[0] for p, _ in FIELDS.values()})
                    ),
                    "assembly": "hg38",
                },
                timeout=(10, 120),
            )
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("batch failed after retries")


def main(july, october, cache_dir, out):
    cache_dir = Path(cache_dir)
    (cache_dir / "batches").mkdir(parents=True, exist_ok=True)
    meta = requests.get(URL + "/metadata", params={"assembly": "hg38"}, timeout=30).json()
    (cache_dir / "source_metadata.json").write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(UTC).isoformat(),
                "build_date": meta.get("build_date"),
                "dbnsfp_version": meta["src"]["dbnsfp"].get("version"),
                "fields": FIELDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ids = sorted(
        set(pd.read_parquet(july, columns=["hgvs_id", "label"]).dropna().hgvs_id)
        | set(pd.read_parquet(october, columns=["hgvs_id", "label"]).dropna().hgvs_id)
    )
    print(f"{len(ids)} labeled identities; dbNSFP {meta['src']['dbnsfp'].get('version')}")
    session = requests.Session()
    rows = []
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        key = hashlib.sha256(json.dumps(chunk).encode()).hexdigest()[:24]
        path = cache_dir / "batches" / f"{key}.json"
        if path.exists():
            hits = json.loads(path.read_text(encoding="utf-8"))
        else:
            hits = fetch(chunk, session)
            path.write_text(json.dumps(hits), encoding="utf-8")
        by_query = {}
        for hit in hits:
            by_query.setdefault(hit.get("query"), []).append(hit)
        for hgvs in chunk:
            found = [h for h in by_query.get(hgvs, []) if not h.get("notfound")]
            row = {"hgvs_id": hgvs, "v3_status": "ok"}
            if len(found) != 1 or not isinstance(found[0].get("dbnsfp"), dict):
                row["v3_status"] = "not_found" if not found else "ambiguous_or_no_dbnsfp"
            else:
                for name, (p, _) in FIELDS.items():
                    v = get(found[0]["dbnsfp"], p)
                    row[name] = float(v) if isinstance(v, (int, float)) else None
            rows.append(row)
        if (start // BATCH) % 10 == 0:
            print(f"  {start + len(chunk)}/{len(ids)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out, index=False)
    print(frame.v3_status.value_counts().to_dict())
    print(frame.drop(columns=["hgvs_id", "v3_status"]).notna().mean().round(3).to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:5])
