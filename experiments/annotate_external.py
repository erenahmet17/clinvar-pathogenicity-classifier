"""Step 3c: annotate external validation variants (ClinGen erepo, BAP1 SGE).

Same dbNSFP 4.8a fields as annotate_v3.py, plus the research-v2 features (CADD phred,
phyloP 100way, gnomAD genome AF) and dbNSFP gene / amino-acid fields for mapping checks.

    python experiments/annotate_external.py IN.parquet data/annotation_cache_v3 OUT.parquet
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from annotate_v3 import FIELDS, get

URL = "https://myvariant.info/v1/variant"
REQUEST_FIELDS = ",".join(
    sorted(
        {"dbnsfp." + p.split(".")[0] for p, _ in FIELDS.values()}
        | {"dbnsfp.genename", "dbnsfp.aa", "dbnsfp.phylop", "dbnsfp.cadd", "gnomad_genome.af"}
    )
)


def fetch(ids, session):
    for attempt in range(5):
        try:
            r = session.post(
                URL,
                data={"ids": ",".join(ids), "fields": REQUEST_FIELDS, "assembly": "hg38"},
                timeout=(10, 120),
            )
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("batch failed")


def main(src, cache_dir, out):
    frame = pd.read_parquet(src)
    cache = Path(cache_dir) / "external_batches"
    cache.mkdir(parents=True, exist_ok=True)
    session, rows = requests.Session(), []
    ids = frame.hgvs_id.tolist()
    for start in range(0, len(ids), 1000):
        chunk = ids[start : start + 1000]
        path = cache / (hashlib.sha256(json.dumps(chunk).encode()).hexdigest()[:24] + ".json")
        if path.exists():
            hits = json.loads(path.read_text(encoding="utf-8"))
        else:
            hits = fetch(chunk, session)
            path.write_text(json.dumps(hits), encoding="utf-8")
        by_q = {}
        for h in hits:
            by_q.setdefault(h.get("query"), []).append(h)
        for hgvs in chunk:
            found = [h for h in by_q.get(hgvs, []) if not h.get("notfound")]
            row = {"hgvs_id": hgvs, "ext_status": "ok"}
            if len(found) != 1:
                row["ext_status"] = "not_found" if not found else "ambiguous"
                rows.append(row)
                continue
            hit, db = found[0], found[0].get("dbnsfp")
            af = get(hit, "gnomad_genome.af.af")
            row["gnomad_af"] = float(af) if isinstance(af, (int, float)) else None
            row["af_available"] = row["gnomad_af"] is not None
            if not isinstance(db, dict):
                row["ext_status"] = "no_dbnsfp"
                rows.append(row)
                continue
            for name, (p, _) in FIELDS.items():
                v = get(db, p)
                row[name] = float(v) if isinstance(v, (int, float)) else None
            row["cadd_phred"] = row.pop("cadd_check")
            v = get(db, "phylop.100way_vertebrate.score")
            row["phylop_100way"] = float(v) if isinstance(v, (int, float)) else None
            genes = db.get("genename")
            row["dbnsfp_gene"] = ";".join(
                sorted(set(genes if isinstance(genes, list) else [genes]))
            )
            aa = db.get("aa") or {}
            aa = aa if isinstance(aa, list) else [aa]
            for key in ["ref", "alt", "pos"]:
                values = set()
                for item in aa:
                    v = item.get(key) if isinstance(item, dict) else None
                    values |= set(map(str, v)) if isinstance(v, list) else {str(v)}
                row[f"aa_{key}"] = ";".join(sorted(values - {"None"}))
            rows.append(row)
    ann = pd.DataFrame(rows)
    merged = frame.merge(ann, on="hgvs_id", how="left", validate="one_to_one")
    merged.to_parquet(out, index=False)
    print(merged.ext_status.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:4])
