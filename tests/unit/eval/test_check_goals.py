"""goals.json enforcer (eval/check_goals.py) — the executable half of the PE
loop contract. Proves it ENFORCES: green on a clean run, exit 1 on any hard-bound
violation, advisory on a relaxable breach.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECK = ROOT / "eval" / "check_goals.py"

_RUN = {
    "naming_seconds": 42.0, "elapsed_seconds": 44.0,
    "metrics": {"primary": {
        "n": 500, "rt_exact": 228, "rt_exact_rate": 0.456,
        "rt_constitutional": 231, "emitted": 231,
        "emit_rate": 0.462, "precision_on_emitted": 0.987,
    }},
}
_GATE = {"verdict": "PASS", "target_passes": 1652, "target_total": 1652,
         "protect_new": 0, "determinism_new": 0}


def _write(tmp, name, obj):
    p = tmp / name
    p.write_text(json.dumps(obj))
    return p


def _run(tmp, run, gate):
    r = subprocess.run(
        [sys.executable, str(CHECK), "--run", str(_write(tmp, "r.json", run)),
         "--gate", str(_write(tmp, "g.json", gate))],
        capture_output=True, text=True)
    return r.returncode, r.stdout


def test_clean_run_passes(tmp_path):
    code, out = _run(tmp_path, _RUN, _GATE)
    assert code == 0, out
    assert "all evaluated bounds hold" in out
    for b in ("H1", "H2", "H3", "B1", "B2", "B3"):
        assert f"{b} PASS" in out, (b, out)


def test_structure_wrong_fails_hard(tmp_path):
    run = json.loads(json.dumps(_RUN))
    run["metrics"]["primary"]["rt_constitutional"] = 229  # structure_wrong = 2
    code, out = _run(tmp_path, run, _GATE)
    assert code == 1 and "H1 FAIL" in out and "HARD-BOUND VIOLATION" in out


def test_pin_gold_regression_fails_hard(tmp_path):
    gate = dict(_GATE, target_passes=1651)
    code, out = _run(tmp_path, _RUN, gate)
    assert code == 1 and "H2 FAIL" in out


def test_pin_gold_incomplete_fails_hard(tmp_path):
    # target_passes == floor but != target_total must still fail (H2 is AND).
    gate = dict(_GATE, target_passes=1652, target_total=1653)
    code, out = _run(tmp_path, _RUN, gate)
    assert code == 1 and "H2 FAIL" in out


def test_byte_identical_regression_fails_hard(tmp_path):
    gate = dict(_GATE, protect_new=1)
    code, out = _run(tmp_path, _RUN, gate)
    assert code == 1 and "H3 FAIL" in out


def test_relaxable_breach_is_advisory_not_fatal(tmp_path):
    run = json.loads(json.dumps(_RUN))
    run["metrics"]["primary"]["emit_rate"] = 0.30  # below B2 floor 0.40
    code, out = _run(tmp_path, run, _GATE)
    assert code == 0 and "B2 FAIL" in out and "relaxable bound" in out
