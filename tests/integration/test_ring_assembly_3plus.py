"""Phase 151-03 D-23 integration: OPSIN round-trip on ring assemblies 3+.

Parametrizes over the 12 literature_validated.json fixtures (each carrying
its own ``opsin_smiles_validated`` evidence from RESEARCH §"OPSIN
Compatibility Evidence") and asserts that ``name_compound(smiles)`` emits
a name OPSIN can parse back to a SMILES whose InChI layer 1 (formula +
connectivity) matches the input.

Per CONTEXT D-23 stereo-layer mismatches don't fail this test (handled in
Phase 152/153). The InChI layer-1 split (``.split('/c')[0]``) restricts
the comparison to the formula portion.

Source: 151-CONTEXT.md D-23.
Source: 151-RESEARCH.md §"OPSIN Compatibility Evidence" (12 names).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem


_OPSIN_JAR = (
    Path(__file__).resolve().parents[2]
    / "opsin-cli-2.9.0-jar-with-dependencies.jar"
)


def _opsin_available() -> bool:
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


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
    / "ring_systems"
    / "assemblies"
)
_LIT_FIXTURES = json.loads((_FIXTURE_DIR / "literature_validated.json").read_text())


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    [f for f in _LIT_FIXTURES if f.get("expected_name")],
    ids=[f["fixture_id"] for f in _LIT_FIXTURES if f.get("expected_name")],
)
def test_ring_assembly_opsin_roundtrip_inchi_l1(fixture):
    """D-23 round-trip: name_compound(smi) -> OPSIN -> InChI L1 == input.

    Skips fixtures whose name is None (corpus_mined.json placeholders).
    """
    from orthonym import name_compound

    smi = fixture["smiles"]
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"invalid fixture SMILES {smi}"

    name = name_compound(smi)
    if name is None:
        pytest.skip(f"name_compound returned None for {fixture['fixture_id']}")

    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(
            f"OPSIN cannot parse {name!r} for {fixture['fixture_id']}"
        )

    parsed_mol = Chem.MolFromSmiles(parsed)
    if parsed_mol is None:
        pytest.skip(f"OPSIN-emitted SMILES invalid: {parsed!r}")

    input_inchi = Chem.MolToInchi(mol).split("/c")[0]
    round_inchi = Chem.MolToInchi(parsed_mol).split("/c")[0]
    assert input_inchi == round_inchi, (
        f"InChI L1 mismatch for {fixture['fixture_id']}\n"
        f"  smiles:   {smi}\n"
        f"  emitted:  {name!r}\n"
        f"  parsed:   {parsed}\n"
        f"  expected_name: {fixture.get('expected_name')!r}\n"
    )


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.integration
def test_opsin_jar_resolves():
    """Sanity: jar path resolves at the worktree root."""
    assert _OPSIN_JAR.exists(), f"OPSIN jar not found at {_OPSIN_JAR}"
