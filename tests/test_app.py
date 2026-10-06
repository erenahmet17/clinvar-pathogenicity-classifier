from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_training import tiny_bundle  # noqa: F401

APP = Path(__file__).resolve().parents[1] / "clinvar_app" / "app.py"


@pytest.mark.integration
def test_app_manual_unknown_af_and_literal_gene_search(monkeypatch, tiny_bundle):  # noqa: F811
    monkeypatch.setenv("CLINVAR_BUNDLE_DIR", str(tiny_bundle))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].label == "Model fitting rows"
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Model prioritization score" for metric in app.metric)
    app.radio[0].set_value("VUS explorer").run()
    app.text_input[0].set_value("[").run()
    assert not app.exception
    assert any("0 of 0 matching" in caption.value for caption in app.caption)
    app.radio[0].set_value("Performance").run()
    assert not app.exception
    assert app.warning
    app.radio[0].set_value("Variant search").run()
    app.text_input[0].set_value("chr6:g.26092913G>A")
    app.button[0].click().run()
    assert not app.exception
    assert "outside the binary model scope" in app.error[0].value


@pytest.mark.integration
def test_missing_bundle_has_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setenv("CLINVAR_BUNDLE_DIR", str(tmp_path / "missing"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "Model bundle unavailable" in app.error[0].value
