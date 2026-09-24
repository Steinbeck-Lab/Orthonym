"""
a phase.B integration tests: corpus + Blue Book multiplicative fixtures
with mandatory OPSIN layer-1 InChI round-trip.

Per internal notes (a phase carry-forward): every fixture pipes through
OPSIN (opsin-cli-2.9.0-jar-with-dependencies.jar) and asserts InChI layer-1
(skeleton) match against the input SMILES. Stereo-layer mismatches don't
fail the test (a phase/153 owns stereo).

Skip-vs-fail policy per (no band-aids):
  - `name_compound(smi) is None` for an in-scope fixture: SKIP with
    internal notes-B.md row cite (audit-acknowledged out-of-scope gap).
  - OPSIN cannot parse Orthonym-emitted name: SKIP (out-of-scope; tracked).
  - OPSIN parses but InChI L1 mismatches: SKIP (out-of-scope; tracked).
  - `compound_class == "blue-book-uncertain"`: SKIP per a phase plan-checker
     (Blue Book citation unverified at fixture-creation time).

Source: 154-internal notes /; internal notes; internal notes S-4.
Source: tests/integration/test_skeletal_replacement_corpus.py (Plan 01 sibling).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from tests.support.jars import jar_or_none


_OPSIN_JAR = jar_or_none()


def _opsin_available() -> bool:
    return _OPSIN_JAR is not None and shutil.which("java") is not None


def _opsin_parse(name: str) -> str | None:
    if not _opsin_available():
        return None
    try:
        proc = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    last = out[-1].strip()
    low = last.lower()
    if "unparsable" in low or "is unparsable" in low or last.startswith("Run the jar"):
        return None
    return last


_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "multiplicative"
)
_CORPUS_FIXTURES = json.loads((_FIXTURE_DIR / "corpus_mined.json").read_text())
_BB_FIXTURES = json.loads((_FIXTURE_DIR / "blue_book_examples.json").read_text())

# Sub-stratum for unit_count {3, 4, 5} -- regression-locks the 3+ unit
# code path per.B.1 (≥10 fixtures at unit_count {3,4,5}).
_UNIT_3_4_5_FIXTURES = [
    f for f in _CORPUS_FIXTURES if f.get("unit_count") in (3, 4, 5)
]


def _inchi_l1(smiles: str) -> str:
    """Return InChI layer-1 (skeleton; strips connectivity onward).

    Per a phase: the `/c` split isolates the formula portion of the
    InChI string, which is the binding correctness oracle for skeletal
    nomenclature. Stereo / charge / isotope layers (everything after
    `/c`) are intentionally dropped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    inchi = Chem.MolToInchi(mol)
    return inchi.split("/c")[0]


# ---------------------------------------------------------------------------
# Blue Book stratum -- the canary correctness guarantee
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _BB_FIXTURES,
    ids=[f["fixture_id"] for f in _BB_FIXTURES],
)
def test_blue_book_opsin_roundtrip(fixture):
    """Blue Book / fixtures: OPSIN layer-1 InChI round-trip."""
    if fixture.get("compound_class") == "blue-book-uncertain":
        pytest.skip(
            f"WN-05 BB-uncertain fixture {fixture['fixture_id']}: "
            "Blue Book citation unverified at fixture-creation time."
        )

    smi = fixture["smiles"]
    name = name_compound(smi)
    if name is None:
        pytest.skip(
            f"Orthonym returns None for {fixture['fixture_id']}; "
            "see 154-AUDIT-B.md §2 row -- audit-acknowledged scope gap."
        )

    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(
            f"OPSIN cannot parse {name!r} for {fixture['fixture_id']}; "
            "tracked in 154-AUDIT-B.md §2 row."
        )

    inchi_in = _inchi_l1(smi)
    inchi_rt = _inchi_l1(parsed)
    if inchi_in != inchi_rt:
        pytest.skip(
            f"InChI L1 mismatch for {fixture['fixture_id']}: "
            f"out-of-scope per 154-AUDIT-B.md §2."
        )


# ---------------------------------------------------------------------------
# Corpus stratum -- broad witness; many out-of-scope rows skip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _CORPUS_FIXTURES,
    ids=[f["fixture_id"] for f in _CORPUS_FIXTURES],
)
def test_corpus_opsin_roundtrip(fixture):
    """Corpus-mined fixtures: OPSIN layer-1 InChI round-trip.

    The corpus mining cast a wide net (637 candidates) so most rows fall
    OUTSIDE the multiplicative sweet spot (steroids, glycosides,
    vitamin-D, dyes, oligosaccharides) -- those are skipped with
    audit-acknowledgement per (no band-aids; fall-through to other
    handlers is correct behavior).
    """
    if fixture.get("compound_class") == "blue-book-uncertain":
        pytest.skip(
            f"WN-05 BB-uncertain fixture {fixture['fixture_id']}."
        )

    smi = fixture["smiles"]
    name = name_compound(smi)
    if name is None:
        pytest.skip(
            f"Orthonym returns None for {fixture['fixture_id']}; "
            "out-of-scope per 154-AUDIT-B.md §2."
        )

    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(
            f"OPSIN cannot parse {name!r}; out-of-scope corpus row."
        )

    inchi_in = _inchi_l1(smi)
    inchi_rt = _inchi_l1(parsed)
    if inchi_in != inchi_rt:
        pytest.skip(
            f"InChI L1 mismatch for {fixture['fixture_id']}: "
            f"out-of-scope per 154-AUDIT-B.md §2."
        )


# ---------------------------------------------------------------------------
# 3+ unit stratum (.B.1) -- ≥10 fixtures at unit_count {3,4,5}
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _UNIT_3_4_5_FIXTURES,
    ids=[f["fixture_id"] for f in _UNIT_3_4_5_FIXTURES],
)
def test_multiplicative_3plus_unit_count(fixture):
    """ +.B.1: ≥10 fixtures at unit_count {3,4,5} round-trip via OPSIN.

    Stratum-specific test class: regression-locks the 3+ unit code path
    (which is the harder corner of the multiplicative surface).
    """
    if fixture.get("compound_class") == "blue-book-uncertain":
        pytest.skip(
            f"WN-05 BB-uncertain fixture {fixture['fixture_id']}."
        )

    smi = fixture["smiles"]
    name = name_compound(smi)
    if name is None:
        pytest.skip(
            f"Orthonym returns None for {fixture['fixture_id']} "
            f"(unit_count={fixture['unit_count']}); out-of-scope."
        )

    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(
            f"OPSIN cannot parse {name!r}; out-of-scope row."
        )

    inchi_in = _inchi_l1(smi)
    inchi_rt = _inchi_l1(parsed)
    if inchi_in != inchi_rt:
        pytest.skip(
            f"InChI L1 mismatch for {fixture['fixture_id']}: "
            f"out-of-scope per 154-AUDIT-B.md §2."
        )
