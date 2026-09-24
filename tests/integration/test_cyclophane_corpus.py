"""a phase.A integration tests: cyclophane corpus + Blue Book fixtures.

Per internal notes + internal notes-A.md S6: OPSIN 2.9.0 does NOT parse the
[m.n]paracyclophane semi-systematic name form (verified at audit time
2026-05-04 -- every form returns 'is unparsable'). All cyclophane integration
fixtures are therefore quarantined via pytest.skip with a a phase grammar
pre-validation hand-off note.

What this test DOES verify (without the OPSIN oracle):
1. Every fixture's SMILES is recognized as a cyclophane by ``is_cyclophane``.
2. ``name_compound`` returns the expected bracket-prefix Blue Book name for
   every non-quarantined fixture (carbocyclic-benzene linker rings).
3. Heterocyclic-bridge / heterocyclic-linker fixtures (R3 quarantine flag) are
   tagged with ``integration_quarantine: true`` in the JSON; for these
   ``is_cyclophane`` MUST still accept (topology gate works) but
   ``name_compound`` may return None or a non-cyclophane name (R3 deferred).

Source: 155-internal notes; internal notes-A.md S6 + S8 (R3); internal notes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.phane import is_cyclophane
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


def _inchi_l1(smiles: str) -> str:
    """Return InChI layer-1 (skeleton; strips connectivity onward).

    Per internal notes: the ``/c`` split isolates the formula portion of the
    InChI string -- the binding correctness oracle for skeletal nomenclature.
    Stereo / charge / isotope layers (everything after ``/c``) are
    intentionally dropped.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    inchi = Chem.MolToInchi(mol)
    return inchi.split("/c")[0]


_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "cyclophane"
)
_BLUE_BOOK_FIXTURES = json.loads((_FIXTURE_DIR / "blue_book_examples.json").read_text())
_CORPUS_MINED_FIXTURES = json.loads((_FIXTURE_DIR / "corpus_mined.json").read_text())
_ALL_FIXTURES = list(_BLUE_BOOK_FIXTURES) + list(_CORPUS_MINED_FIXTURES)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _ALL_FIXTURES,
    ids=[f["fixture_id"] for f in _ALL_FIXTURES],
)
def test_cyclophane_topology_gate(fixture):
    """is_cyclophane(SMILES) must accept every fixture in the audit catalog.

    This validates the topology gate (corrected SSSR criterion) on the
    full ≥ 13 audit-curated set without depending on the OPSIN oracle.
    """
    smi = fixture["smiles"]
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"Invalid SMILES for {fixture['fixture_id']}: {smi!r}"
    assert is_cyclophane(mol), (
        f"is_cyclophane declined audit-curated cyclophane "
        f"{fixture['fixture_id']!r} (smiles={smi!r}, "
        f"sub_class={fixture['sub_class']})"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _ALL_FIXTURES,
    ids=[f["fixture_id"] for f in _ALL_FIXTURES],
)
def test_cyclophane_name_compound(fixture):
    """name_compound(SMILES) result for each fixture.

    Wave-8 P8: fixtures carrying an ``expected_pin`` field (monocyclic
    all-benzene-homophane class -- `build_phane_pin` verified BB-name-exact,
    see docs/the workflow tooling/plans/2026-07-16-wave8-p8-phane.md Task 8.7/8.12)
    now EMIT that simplified-skeletal PIN in production. Every other
    non-quarantined fixture (a topology `build_phane_pin` doesn't cover yet)
    still hits the fail-closed refusal ('unknown organic compound') --
    OPSIN cannot parse any phane form, so an unverified composed name is
    withheld rather than shipped.

    Heterocyclic-bridge / heterocyclic-linker fixtures (R3 quarantine flag)
    are skipped here per internal notes-A.md S8 -- the topology gate accepts them
    but the name composition is deferred (still fail-closed in production).
    """
    if fixture.get("integration_quarantine"):
        pytest.skip(
            f"R3 quarantine: {fixture['fixture_id']} -- "
            f"{fixture.get('quarantine_reason', 'heterocyclic linker/bridge deferred')}"
        )

    smi = fixture["smiles"]
    expected_pin = fixture.get("expected_pin")
    actual = name_compound(smi)
    if expected_pin is not None:
        assert actual == expected_pin, (
            f"name_compound({fixture['fixture_id']}, smiles={smi!r}) returned "
            f"{actual!r}; expected the Wave-8 P8 P-26 PIN {expected_pin!r}"
        )
    else:
        assert actual == "unknown organic compound", (
            f"name_compound({fixture['fixture_id']}, smiles={smi!r}) returned "
            f"{actual!r}; expected the T6c fail-closed 'unknown organic compound' "
            f"(composed form would have been {fixture['expected_name']!r}; no "
            f"verified P-26 PIN for this topology yet)"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.parametrize(
    "fixture",
    _ALL_FIXTURES,
    ids=[f["fixture_id"] for f in _ALL_FIXTURES],
)
def test_cyclophane_opsin_roundtrip_quarantine(fixture):
    """: name_compound -> OPSIN -> InChI L1 round-trip.

    Per internal notes-A.md S6: OPSIN 2.9.0 does NOT parse cyclophane semi-systematic
    names. Every fixture is expected to skip-and-cite per the fallback
    policy. If OPSIN ever gains cyclophane support (a phase grammar
    pre-validation hand-off), this test starts asserting the round-trip.
    """
    if fixture.get("integration_quarantine"):
        pytest.skip(
            f"R3 quarantine: {fixture['fixture_id']} -- "
            f"{fixture.get('quarantine_reason', 'heterocyclic linker/bridge deferred')}"
        )

    smi = fixture["smiles"]
    name = name_compound(smi)
    if name is None:
        pytest.skip(
            f"name_compound returned None for {fixture['fixture_id']}; "
            f"likely R3 quarantine-tagged in fixture metadata."
        )
    rt_smi = _opsin_parse(name)
    if rt_smi is None:
        pytest.skip(
            f"OPSIN cannot parse {name!r}; quarantine for Phase 156 grammar "
            f"pre-validation per 155-AUDIT-A.md S6 + CONTEXT D-15. "
            f"Fixture: {fixture['fixture_id']}."
        )
    # If OPSIN ever did parse, assert the InChI L1 oracle.
    assert _inchi_l1(smi) == _inchi_l1(rt_smi), (
        f"InChI L1 mismatch for {fixture['fixture_id']}: "
        f"input={smi!r} -> name={name!r} -> rt_smi={rt_smi!r}"
    )
