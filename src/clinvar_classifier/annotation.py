"""Bounded HTTP calls and durable identity-based annotation cache."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from filelock import FileLock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .io import utc_now, write_json
from .schemas import Annotation, ValidationError, finite_number, normalize_hgvs

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://myvariant.info/v1"
FIELDS = "dbnsfp.cadd.phred,dbnsfp.phylop.100way_vertebrate,gnomad_genome.af.af,gnomad_genome.ac.ac,gnomad_genome.an.an"


class AnnotationError(RuntimeError):
    """A batch could not be completed or the endpoint returned invalid JSON."""


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    # POST requests here are idempotent annotation reads, never writes.
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def nested(value, *keys):
    for key in keys:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValidationError(
                f"Expected object at {key}; ambiguous transcript objects are unsupported."
            )
        value = value.get(key)
    return value


def parse_annotation(payload: dict, hgvs_id: str | None = None) -> Annotation:
    if not isinstance(payload, dict):
        raise ValidationError("Expected one annotation object.")
    identity = normalize_hgvs(hgvs_id or payload.get("query") or payload.get("_id", ""))
    returned_id = payload.get("_id")
    if returned_id is not None and normalize_hgvs(returned_id) != identity:
        raise ValidationError("Returned variant identity differs from requested GRCh38 identity.")
    if payload.get("notfound"):
        return Annotation(identity, annotation_status="not_found")
    cadd = finite_number(nested(payload, "dbnsfp", "cadd", "phred"), "cadd", minimum=0, maximum=99)
    phylop_field = nested(payload, "dbnsfp", "phylop", "100way_vertebrate")
    phylop = finite_number(
        phylop_field.get("score") if isinstance(phylop_field, dict) else phylop_field,
        "phyloP",
        minimum=-20,
        maximum=15,
    )
    af = finite_number(nested(payload, "gnomad_genome", "af", "af"), "AF", minimum=0, maximum=1)
    ac = finite_number(nested(payload, "gnomad_genome", "ac", "ac"), "AC", minimum=0)
    an = finite_number(nested(payload, "gnomad_genome", "an", "an"), "AN", minimum=0)
    if ac is not None and an is not None and ac > an:
        raise ValidationError("Allele count exceeds allele number.")
    status = "ok" if cadd is not None and phylop is not None else "missing_required_features"
    return Annotation(identity, cadd, phylop, af, af is not None, ac, an, status)


class AnnotationClient:
    def __init__(self, cache_dir: Path | None = None, session=None, timeout=(5, 45)):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or make_session()
        self.timeout = timeout
        self.namespace = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (
            FileLock(str(self.cache_dir / ".writer.lock"), timeout=0) if self.cache_dir else None
        )

    def _json(self, method: str, url: str, **kwargs):
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise AnnotationError(f"Annotation request failed: {exc}") from exc

    def pin_metadata(self) -> dict:
        if self._lock:
            with self._lock:
                return self._pin_metadata()
        return self._pin_metadata()

    def _pin_metadata(self) -> dict:
        if self.cache_dir is not None:
            metadata_path = self.cache_dir / "source_metadata.json"
            if metadata_path.exists():
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                current = self._json("GET", BASE_URL + "/metadata", params={"assembly": "hg38"})
                if previous["metadata"] != current:
                    raise AnnotationError(
                        "Source metadata changed; use a new cache directory and dataset run."
                    )
                self.namespace = previous["namespace"]
                return previous
        metadata = self._json("GET", BASE_URL + "/metadata", params={"assembly": "hg38"})
        identity = {"assembly": "hg38", "fields": FIELDS, "metadata": metadata}
        self.namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        record = {**identity, "namespace": self.namespace, "retrieved_at": utc_now()}
        if self.cache_dir is not None:
            write_json(self.cache_dir / "source_metadata.json", record)
        return record

    def get(self, hgvs_id: str) -> Annotation:
        identity = normalize_hgvs(hgvs_id)
        try:
            payload = self._json(
                "GET",
                BASE_URL + "/variant/" + quote(identity, safe=""),
                params={"assembly": "hg38", "fields": FIELDS},
            )
        except AnnotationError as exc:
            if (
                isinstance(exc.__cause__, requests.HTTPError)
                and exc.__cause__.response.status_code == 404
            ):
                return Annotation(identity, annotation_status="not_found")
            raise
        return parse_annotation(payload, identity)

    def _batch(self, identities: list[str]) -> list[dict]:
        key = hashlib.sha256(
            json.dumps({"namespace": self.namespace, "ids": identities}, sort_keys=True).encode()
        ).hexdigest()
        path = self.cache_dir / "batches" / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["requested_ids"] != identities:
                raise AnnotationError("Annotation cache identity mismatch.")
            return record["annotations"]
        # Each concurrent batch owns its own Session; connection state is never shared across threads.
        client = (
            AnnotationClient(session=make_session(), timeout=self.timeout)
            if self.session.__class__ is requests.Session
            else self
        )
        try:
            payload = client._json(
                "POST",
                BASE_URL + "/variant",
                data={"ids": ",".join(identities), "fields": FIELDS, "assembly": "hg38"},
            )
            if not isinstance(payload, list):
                raise AnnotationError("Batch endpoint must return a list.")
            by_id = {}
            for item in payload:
                if not isinstance(item, dict) or item.get("query") not in identities:
                    raise AnnotationError("Unexpected or missing query identity in batch response.")
                query = item["query"]
                if query in by_id:
                    raise AnnotationError(
                        f"Multiple results for {query}; no implicit transcript selection."
                    )
                by_id[query] = item
            if set(by_id) != set(identities):
                raise AnnotationError("Incomplete batch response; batch remains pending for retry.")
            annotations = []
            for query in identities:
                try:
                    result = parse_annotation(by_id[query], query)
                except ValidationError as exc:
                    result = Annotation(
                        query, annotation_status="invalid_annotation", annotation_error=str(exc)
                    )
                annotations.append(result.to_dict())
            if path:
                write_json(
                    path,
                    {
                        "requested_ids": identities,
                        "annotations": annotations,
                        "completed_at": utc_now(),
                    },
                )
            return annotations
        finally:
            if client is not self:
                client.session.close()

    def annotate(self, identities, batch_size=1000, workers=4) -> pd.DataFrame:
        if self._lock:
            with self._lock:
                return self._annotate(identities, batch_size, workers)
        return self._annotate(identities, batch_size, workers)

    def _annotate(self, identities, batch_size=1000, workers=4) -> pd.DataFrame:
        identities = sorted({normalize_hgvs(value) for value in identities})
        if not 1 <= batch_size <= 1000 or not 1 <= workers <= 8:
            raise ValueError("Use 1–1000 variants per batch and 1–8 workers.")
        if self.namespace is None:
            self.pin_metadata()
        # Reuse completed identities even when batch boundaries or the cohort change.
        requested = set(identities)
        cached = {}
        if self.cache_dir:
            for path in (self.cache_dir / "batches").glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                for item in record["annotations"]:
                    if item["hgvs_id"] in requested:
                        cached[item["hgvs_id"]] = item
        missing = [value for value in identities if value not in cached]
        batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
        results, failures = list(cached.values()), []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(self._batch, batch): batch for batch in batches}
            for future in as_completed(pending):
                try:
                    results.extend(future.result())
                except (AnnotationError, requests.RequestException) as exc:
                    failures.append({"ids": pending[future], "error": str(exc)})
                LOGGER.info(
                    "Annotation progress: %s / %s; failed batches=%s",
                    len(results),
                    len(identities),
                    len(failures),
                )
        if self.cache_dir:
            write_json(self.cache_dir / "pending.json", failures)
        if failures:
            raise AnnotationError(
                f"{len(failures)} batches pending; rerun with the same cache. No dataset was silently completed."
            )
        frame = pd.DataFrame(results, columns=Annotation.__dataclass_fields__)
        if len(frame) != len(identities) or frame.hgvs_id.duplicated().any():
            raise AnnotationError("Annotation cardinality differs from unique inputs.")
        return frame.sort_values("hgvs_id").reset_index(drop=True)
