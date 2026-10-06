import json

import pytest

from clinvar_classifier.annotation import AnnotationClient, AnnotationError, make_session


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, incomplete=False):
        self.calls = []
        self.incomplete = incomplete

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("metadata"):
            return Response({"build_version": "fixed", "src": {}})
        ids = kwargs["data"]["ids"].split(",")
        if self.incomplete:
            ids = ids[:-1]
        return Response([{"query": value, "notfound": True} for value in ids])


def test_idempotent_read_retry_policy_and_timeout():
    retry = make_session().get_adapter("https://").max_retries
    assert {429, 502, 503, 504}.issubset(retry.status_forcelist)
    assert retry.total == 4 and "POST" in retry.allowed_methods
    session = Session()
    client = AnnotationClient(session=session)
    client.pin_metadata()
    assert session.calls[0][2]["timeout"] == (5, 45)


def test_incomplete_batch_never_gets_checkpointed(tmp_path):
    client = AnnotationClient(tmp_path, session=Session(incomplete=True))
    with pytest.raises(AnnotationError):
        client.annotate(["chr1:g.1A>G", "chr1:g.2A>G"], workers=1)
    assert len(json.loads((tmp_path / "pending.json").read_text())) == 1
    assert not list((tmp_path / "batches").glob("*.json"))


def test_checkpoint_survives_client_restart(tmp_path):
    first = Session()
    client = AnnotationClient(tmp_path, session=first)
    original = client.annotate(["chr1:g.1A>G"], workers=1)
    restarted = Session()
    resumed = AnnotationClient(tmp_path, session=restarted).annotate(["chr1:g.1A>G"], workers=1)
    assert original.equals(resumed)
    assert not any(method == "POST" for method, _, _ in restarted.calls)


def test_source_drift_requires_new_cache(tmp_path):
    client = AnnotationClient(tmp_path, session=Session())
    client.pin_metadata()
    record = json.loads((tmp_path / "source_metadata.json").read_text())
    record["metadata"]["build_version"] = "old"
    (tmp_path / "source_metadata.json").write_text(json.dumps(record))
    with pytest.raises(AnnotationError):
        client.pin_metadata()


def test_changed_batch_boundaries_reuse_completed_ids(tmp_path):
    original = Session()
    AnnotationClient(tmp_path, session=original).annotate(
        ["chr1:g.1A>G", "chr1:g.2A>G"], batch_size=2, workers=1
    )
    resumed = Session()
    AnnotationClient(tmp_path, session=resumed).annotate(
        ["chr1:g.1A>G", "chr1:g.2A>G", "chr1:g.3A>G"], batch_size=1, workers=1
    )
    posts = [call for call in resumed.calls if call[0] == "POST"]
    assert len(posts) == 1 and posts[0][2]["data"]["ids"] == "chr1:g.3A>G"


def test_concurrent_cache_writer_rejected(tmp_path):
    from filelock import FileLock, Timeout

    client = AnnotationClient(tmp_path, session=Session())
    with FileLock(str(tmp_path / ".writer.lock")):
        with pytest.raises(Timeout):
            client.annotate(["chr1:g.1A>G"], workers=1)
