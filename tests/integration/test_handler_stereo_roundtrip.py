"""Phase 152: Handler-level stereo injection - OPSIN round-trip integration tests.

Per D-16 / D-22, every test compound is round-tripped through OPSIN
(name -> SMILES -> canonicalize) and compared with the original canonical
SMILES INCLUDING the stereo layer. The four handler classes
(TestHeterocycle, TestBenzene, TestCycloalkane, TestCycloalkene) are
populated from JSON fixtures under tests/data/stereo_handlers/. Every
fixture entry is sourced per D-23 (no hand-curated SMILES).

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
    smiles, expected_stereo_prefix_pattern, source_corpus, source_id (D-23).
    """
    path = _FIXTURES_DIR / handler / "cases.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data.get("compounds", [])


def _assert_handler_roundtrip(case: dict, handler: str):
    if case.get("xfail_reason"):
        pytest.xfail(case["xfail_reason"])
    smiles = case["smiles"]
    name = name_compound(smiles)
    assert name and name != "unknown", (
        f"name_compound returned empty/unknown for {smiles} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )

    # 1. Format compliance (P-91): expected_stereo_prefix_pattern is a regex.
    import re as _re
    pat = case["expected_stereo_prefix_pattern"]
    assert _re.match(pat, name), (
        f"P-91 prefix mismatch: pattern={pat!r} name={name!r} "
        f"(handler={handler}, source={case.get('source_corpus')}/{case.get('source_id')})"
    )

    # 2. OPSIN round-trip (D-16): name -> SMILES -> canonicalize, compare with
    #    original canonical SMILES (preserving stereo).
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


# Cycloalkane + cycloalkene classes are added in commit 5.
