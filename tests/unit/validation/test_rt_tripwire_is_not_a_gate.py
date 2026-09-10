""" Phase-A meta-test: RT can NEVER be the phase gate.

The whole point of DD8 is that OPSIN round-trip (RT) is demoted to an advisory tripwire. This guard-rail
asserts that ``scripts/v22_gate.py``'s exit code (the phase PASS/FAIL) is INDEPENDENT of the RT number:
even a catastrophic RT collapse must not fail the default phase gate. Only the explicit, opt-in standalone
``--rt-tripwire PREV CUR`` mode is allowed to exit non-zero on a drop — and it is never the phase gate.

This stops RT from silently creeping back as the optimization target.

See internal notes §C.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
gate = importlib.import_module("v22_gate")


def _passing_conformance(*_a, **_k):
    return {
        "aggregate": {"total": 10, "matched": 8, "pass_rate": 0.8, "protect_regressions": 0,
                      "target_total": 4, "target_passes": 4, "by_verdict": {}, "by_def": {}},
        "packs": {}, "protect_fail_rows": [], "records": [],
    }


def _passing_determinism(*_a, **_k):
    return {"total_probes": 5, "deterministic": 5, "nondeterministic": 0,
            "new_nondeterministic": [], "still_quarantined": [], "dequarantine_ready": [], "records": []}


@pytest.fixture
def _stub_gate(monkeypatch):
    """Stub the expensive naming runs so the gate logic is exercised without OPSIN/naming."""
    monkeypatch.setattr(gate, "load_packs", lambda *a, **k: [("stub", [])])
    monkeypatch.setattr(gate, "run_conformance", _passing_conformance)
    monkeypatch.setattr(gate, "_collect_probes", lambda *a, **k: [])
    monkeypatch.setattr(gate, "_load_quarantine", lambda *a, **k: [])
    monkeypatch.setattr(gate, "run_determinism", _passing_determinism)
    return gate


def _corpus_report(tmp_path, name, rt_count):
    p = tmp_path / name
    p.write_text(json.dumps({"rt_count": rt_count}), encoding="utf-8")
    return str(p)


def test_default_gate_passes_without_rt(_stub_gate):
    assert _stub_gate.main(["--baseline", "/nonexistent"]) == 0


def test_catastrophic_rt_drop_does_NOT_fail_the_phase_gate(_stub_gate, tmp_path):
    good = _corpus_report(tmp_path, "prev.json", 1000)
    collapsed = _corpus_report(tmp_path, "cur.json", 1)  # 99.9% RT collapse
    rc = _stub_gate.main(["--baseline", "/nonexistent", "--rt-prev", good, "--rt-cur", collapsed,
                          "--rt-floor", "1.0"])
    assert rc == 0, "RT collapse must be advisory-only — it must NOT fail the v22 phase gate"


def test_gate_exit_code_is_identical_with_and_without_rt(_stub_gate, tmp_path):
    good = _corpus_report(tmp_path, "p.json", 1000)
    collapsed = _corpus_report(tmp_path, "c.json", 1)
    without = _stub_gate.main(["--baseline", "/nonexistent"])
    with_rt = _stub_gate.main(["--baseline", "/nonexistent", "--rt-prev", good, "--rt-cur", collapsed])
    assert without == with_rt == 0


def test_standalone_rt_tripwire_DOES_gate_on_catastrophic_drop(tmp_path):
    """The one explicit, opt-in mode where RT may fail — never invoked as the phase gate."""
    good = _corpus_report(tmp_path, "prev.json", 1000)
    collapsed = _corpus_report(tmp_path, "cur.json", 1)
    assert gate.main(["--rt-tripwire", good, collapsed, "--rt-floor", "1.0"]) == 1
    assert gate.main(["--rt-tripwire", good, good, "--rt-floor", "1.0"]) == 0
