"""a phase: Handler-level stereo injection - OPSIN round-trip integration tests.

Per /, every test compound is round-tripped through OPSIN
(name -> SMILES -> canonicalize) and compared with the original canonical
SMILES INCLUDING the stereo layer. The four handler classes
(TestHeterocycle, TestBenzene, TestCycloalkane, TestCycloalkene) are
populated from JSON fixtures under tests/data/stereo_handlers/. Every
fixture entry is sourced per (no hand-curated SMILES).

Skipped if Java or OPSIN JAR is unavailable.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound

JAVA_AVAILABLE = shutil.which("java") is not None
OPSIN_JAR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "opsin-cli-2.9.0-jar-with-dependencies.jar",
    )
)
OPSIN_AVAILABLE = JAVA_AVAILABLE and os.path.isfile(OPSIN_JAR)

pytestmark = [
    pytest.mark.skipif(
        not OPSIN_AVAILABLE,
        reason="Java or OPSIN JAR (2.9.0) not available for round-trip tests",
    ),
    pytest.mark.roundtrip,
]


def opsin_parse(name: str) -> str:
    """Parse IUPAC name to SMILES using OPSIN 2.9.0."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


_FIXTURES_DIR = Path(__file__).parent.parent / "data" / "stereo_handlers"


def _load_cases(handler: str):
    """Load fixture cases for *handler* from the JSON file under
    tests/data/stereo_handlers/<handler>/cases.json. Each entry MUST have
    smiles, expected_stereo_prefix_pattern, source_corpus, source_id .
    """
    path = _FIXTURES_DIR / handler / "cases.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data.get("compounds", [])


def _assert_handler_roundtrip(case: dict, handler: str):
    """Assert handler stereo correctness in two stages.

     FIX (2026-05-03 — see internal notes + internal notes):
    Previously this function called pytest.xfail before ANY assertion,
    making xfail-marked fixtures execute zero asserts. Now the assertion
    pipeline is split:
      (1) ALWAYS-RUN GATES: name_compound returns non-empty AND
           prefix matches `expected_stereo_prefix_pattern`. These run
          for every fixture, regardless of xfail_reason. They detect
          regressions in the a phase wiring contract independently of
          OPSIN's parsing capability.
      (2) CONDITIONAL OPSIN ROUND-TRIP: name -> OPSIN -> canonical SMILES
          -> compare. Only THIS step is xfail-able when the fixture's
          `xfail_reason` indicates a known OPSIN-side or
          handler-output-format issue tracked outside a phase.
    """
    import re as _re
    smiles = case["smiles"]
    name = name_compound(smiles)

    # ---- ALWAYS-RUN GATES (fix) ----
    assert name and name != "unknown", (
        f"name_compound returned empty/unknown for {smiles} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )
    pat = case["expected_stereo_prefix_pattern"]
    assert _re.match(pat, name), (
        f"P-91 prefix mismatch: pattern={pat!r} name={name!r} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )

    # ---- CONDITIONAL OPSIN ROUND-TRIP (still xfail-able) ----
    if case.get("xfail_reason"):
        pytest.xfail(case["xfail_reason"])
    opsin_smiles = opsin_parse(name)
    assert opsin_smiles, (
        f"OPSIN could not parse name {name!r} (handler={handler}, "
        f"source={case.get('source_corpus')}/{case.get('source_id')})"
    )
    canonical_input = Chem.CanonSmiles(smiles)
    canonical_output = Chem.CanonSmiles(opsin_smiles)
    assert canonical_input == canonical_output, (
        f"SMILES round-trip mismatch: input={canonical_input} "
        f"output={canonical_output} name={name!r} (handler={handler})"
    )


# ----------------------------------------------------------------------
# Heterocycle (commit 3)
# ----------------------------------------------------------------------

_HETEROCYCLE_CASES = _load_cases("heterocycle")


@pytest.mark.parametrize(
    "case",
    _HETEROCYCLE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestHeterocycle:
    def test_heterocycle_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "heterocycle")


# ----------------------------------------------------------------------
# Benzene (commit 4)
# ----------------------------------------------------------------------

_BENZENE_CASES = _load_cases("benzene")


@pytest.mark.parametrize(
    "case",
    _BENZENE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestBenzene:
    def test_benzene_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "benzene")


# ----------------------------------------------------------------------
# Cycloalkane (commit 5)
# ----------------------------------------------------------------------

_CYCLOALKANE_CASES = _load_cases("cycloalkane")


@pytest.mark.parametrize(
    "case",
    _CYCLOALKANE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestCycloalkane:
    def test_cycloalkane_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "cycloalkane")


# ----------------------------------------------------------------------
# Cycloalkene (commit 5)
# ----------------------------------------------------------------------

_CYCLOALKENE_CASES = _load_cases("cycloalkene")


@pytest.mark.parametrize(
    "case",
    _CYCLOALKENE_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestCycloalkene:
    def test_cycloalkene_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "cycloalkene")


# ----------------------------------------------------------------------
# a phase: ComplexRing (commit 3 -- / / / wiring + fixtures)
# ----------------------------------------------------------------------

_COMPLEX_RING_CASES = _load_cases("complex_ring")


@pytest.mark.parametrize(
    "case",
    _COMPLEX_RING_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}",
)
class TestComplexRing:
    def test_complex_ring_stereo_roundtrip(self, case):
        _assert_handler_roundtrip(case, "complex_ring")


# ----------------------------------------------------------------------
# a phase: Cycloalkene >= 8 mandatory E/Z (commit 4 -- errata)
# ----------------------------------------------------------------------

_CYCLOALKENE_8PLUS_CASES = _load_cases("cycloalkene_8plus")


@pytest.mark.parametrize(
    "case",
    _CYCLOALKENE_8PLUS_CASES,
    ids=lambda c: f"{c.get('source_corpus','?')}/{c.get('source_id','?')}-r{c.get('ring_size','?')}",
)
class TestCycloalkene8PlusMandatory:
    def test_cycloalkene_8plus_emits_ez_block(self, case):
        # First the standard 2-stage gates (name + prefix + OPSIN
        # round-trip per split-stage assertion).
        _assert_handler_roundtrip(case, "cycloalkene_8plus")
        # a phase /: in addition, the prefix MUST contain
        # an [EZ] descriptor, located anywhere in the leading block
        # (the regex tolerates names like '(4S,7Z,...)' where R/S
        # descriptors precede the E/Z one).
        import re as _re
        smiles = case["smiles"]
        name = name_compound(smiles)
        assert _re.search(r"^\([^)]*[0-9][EZ]", name), (
            f"P-31.1.3 mandatory E/Z block missing for ring_size="
            f"{case.get('ring_size')} name={name!r} (handler=cycloalkene_8plus, "
            f"source={case.get('source_corpus')}/{case.get('source_id')})"
        )
