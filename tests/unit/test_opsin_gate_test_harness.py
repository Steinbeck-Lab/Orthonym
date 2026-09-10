"""The OPSIN-validity-gate test harness, tested.

`tests/conftest.py` disables the OPSIN validity gate for every test by default.
That is deliberate — most tests assert RAW generator output and must not pay a
per-name OPSIN call — but it is a trap that has been sprung: a test written
*about* gate behaviour is green-but-blind unless it re-enables the gate, and
nothing tells you.

This file is the tripwire for that. It asserts the gate harness is real in both
directions, so the harness cannot rot silently: if `test_the_canary_*` pair ever
stops disagreeing, every gate-enabled test in the suite has gone blind and this
file goes red first.

Nothing here tests nomenclature. It tests the thing the nomenclature tests
stand on.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from orthonym import Orthonym
from orthonym.errors import _DESCRIPTIVE_FALLBACK_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The canary. Its first-matching naming class produces `(5-carbamoylpentyl)-
# oxirane`, which OPSIN parses to NC(=O)CCCCCC1CO1 — a DIFFERENT molecule than
# the input. Only catches that; no other gate in the stack does. So this
# molecule's output is a direct readout of whether the gate is live.
CANARY = "CC(=O)N(CC1CO1)C(C)C"
CANARY_UNGATED_NAME = "(5-carbamoylpentyl)oxirane"


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------

def test_the_gate_flag_exists_under_the_name_the_harness_uses():
    """The conftest fixture sets this attribute with raising=True, so a rename
    is a loud error rather than a suite-wide silent no-op. This test names the
    contract so the rename is caught even if no gate test runs."""
    import orthonym.namer as _namer
    assert hasattr(_namer, "_DISABLE_VALIDITY_GATE")


# ---------------------------------------------------------------------------
# Default OFF / marker ON / fixture ON
# ---------------------------------------------------------------------------

def test_gate_is_off_by_default():
    import orthonym.namer as _namer
    assert _namer._DISABLE_VALIDITY_GATE is True


@pytest.mark.opsin_gate
def test_marker_turns_the_gate_on():
    import orthonym.namer as _namer
    assert _namer._DISABLE_VALIDITY_GATE is False


def test_fixture_form_turns_the_gate_on(opsin_gate):
    import orthonym.namer as _namer
    assert _namer._DISABLE_VALIDITY_GATE is False


def test_the_marker_does_not_leak_into_the_next_test():
    """monkeypatch undoes the setattr per test; if it ever did not, gate-ON
    would bleed across the suite and quietly slow / change unrelated tests."""
    import orthonym.namer as _namer
    assert _namer._DISABLE_VALIDITY_GATE is True


# ---------------------------------------------------------------------------
# The canary — the pair that proves the harness is load-bearing
# ---------------------------------------------------------------------------

def test_the_canary_ships_a_wrong_molecule_when_the_gate_is_off():
    """With the gate off, Orthonym emits a name for a DIFFERENT molecule.

    This is not a defect to fix here — it is the measurement that gives the
    gate-on assertion below its meaning. An assertion that the canary abstains
    is worthless unless the ungated path demonstrably does something else.
    """
    assert Orthonym(style="pin").name(CANARY) == CANARY_UNGATED_NAME


@pytest.mark.opsin_gate
def test_the_canary_is_suppressed_when_the_gate_is_on():
    assert Orthonym(style="pin").name(CANARY) in _DESCRIPTIVE_FALLBACK_NAMES


# ---------------------------------------------------------------------------
# The jar guard
# ---------------------------------------------------------------------------

def test_a_gate_test_without_a_jar_is_skipped_not_passed():
    """`_final_opsin_validity_gate` fails OPEN on a missing jar , so
    "gate on, no jar" is green-but-blind by a second route — and it is the
    realistic one: a worktree checkout has no jar, the `opsin` gitlink has no
    .gitmodules to fetch from, and CI may not build it.

    `pytest_runtest_call` in conftest must turn that into a SKIP. Verified in a
    subprocess with the jar finder stubbed out, because the condition cannot be
    created in-process without breaking the running test session.
    """
    probe = PROJECT_ROOT / "tests" / "unit" / "_gate_jar_absent_probe.py"
    probe.write_text(
        "import pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def _no_jar(monkeypatch):\n"
        "    import orthonym.validation.opsin_roundtrip as _rt\n"
        "    monkeypatch.setattr(_rt, '_find_opsin_jar', lambda *a, **k: None)\n"
        "    yield\n"
        "@pytest.mark.opsin_gate\n"
        "def test_must_not_run_blind():\n"
        "    assert False, 'this body must never execute without a jar'\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
        )
        combined = r.stdout + r.stderr
        assert "1 skipped" in combined, (
            "a gate-enabled test ran with no OPSIN jar instead of being "
            f"skipped — the blindness guard is not working.\n{combined[-2000:]}"
        )
    finally:
        probe.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Ratchet: no NEW hand-rolled re-enables
# ---------------------------------------------------------------------------

# Files that predate the `opsin_gate` marker and still hand-roll
# `monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", False)`. They are not
# broken — the conftest hook covers them for the jar case — but the marker is
# the one supported way to write a NEW one. This list may shrink, never grow.
_LEGACY_HANDROLLED = frozenset({
    "tests/conftest.py",
    "tests/integration/test_polyfunctional_split.py",
    "tests/integration/test_v29_p3reg_gate_target_oracle.py",
    "tests/ledger/test_reaudit_w8.py",
    "tests/unit/assembly/test_ring_free_valence_class.py",
    "tests/unit/assembly/test_v26_p1_monocycle_engine.py",
    "tests/unit/assembly/test_v26_p4_multifragment.py",
    "tests/unit/assembly/test_v26_p5_charged_general.py",
    "tests/unit/assembly/test_v26_p6_retained_preference.py",
    "tests/unit/assembly/test_v26_p7_stereo_failclosed.py",
    "tests/unit/assembly/test_w2f_p1_amino_branch.py",
    "tests/unit/assembly/test_w2f_p3_branched_substituent.py",
    "tests/unit/namer/test_gate_stereo_carveout_self01.py",
    "tests/unit/namer/test_self_consistency_gate.py",
    "tests/unit/rules/test_adducts.py",
    "tests/unit/rules/test_ester_alkyl_stereo.py",
    "tests/unit/rules/test_pnictogen_oxoacids.py",
    "tests/unit/rules/test_ring_hydrazide_nn.py",
    "tests/unit/rules/test_tier3b_sulfoxide_substitutive.py",
    "tests/unit/rules/test_tier4_fused_catalog.py",
    "tests/unit/rules/test_tier5c_acyl_prefix_on_senior.py",
    "tests/unit/rules/test_tier6_exotics.py",
    "tests/unit/rules/test_v26_p2_aromatic_vonbaeyer.py",
    "tests/unit/rules/test_v26_p3_failclosed_routing.py",
    "tests/unit/rules/test_w2e_p1_amide.py",
    "tests/unit/rules/test_w2f_p3_polyfunctional_branch.py",
    "tests/unit/rules/test_wave2_completion_c2.py",
    "tests/unit/rules/test_wave2_completion_d.py",
    "tests/unit/rules/test_wave2_p5_bridged.py",
    "tests/unit/rules/test_wave2_p5_fused.py",
    "tests/unit/test_bbr_gate_stereo.py",
    "tests/unit/test_g1_bridged_fused.py",
    "tests/unit/test_mba172_moving_base_atom.py",
    "tests/unit/test_opsin_validity_gate.py",
    "tests/unit/test_polyfunctional_basic.py",
    "tests/unit/validation/test_pin_conformance_packs.py",
})


def test_no_new_file_hand_rolls_the_gate_re_enable():
    r = subprocess.run(
        ["grep", "-rl", "--include=*.py",
         r'_DISABLE_VALIDITY_GATE"\?,\? *False\|_DISABLE_VALIDITY_GATE = False',
         "tests"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    found = {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}
    # This file names the pattern in order to forbid it.
    found.discard("tests/unit/test_opsin_gate_test_harness.py")
    new = found - _LEGACY_HANDROLLED
    assert not new, (
        "these files re-enable the OPSIN validity gate by hand:\n  "
        + "\n  ".join(sorted(new))
        + "\n\nUse the marker instead — it also skips when the OPSIN jar is "
          "absent, which a hand-rolled setattr does not:\n"
          "    pytestmark = pytest.mark.opsin_gate\n"
    )
