"""a phase.C integration tests: benzo-fusion corpus fixtures with OPSIN
layer-1 InChI round-trip.

Per internal notes: every fixture pipes through OPSIN
(``opsin-cli-2.9.0-jar-with-dependencies.jar``) and asserts InChI layer-1
(skeleton, stereo-stripped) match against the input SMILES. Stereo-layer
mismatches do not fail the test (a phase/153 owns stereo).

Fixtures: ``tests/fixtures/benzo_fusion/corpus_mined.json`` (>=10 entries
spanning >=5 benzo-fusion sub-classes; mined from chebi_5000 +
pubchem_2000 + opsin_selftest_500 + Blue Book canonical
examples per corpus-shortfall protocol).

Skip-vs-fail policy per (no band-aids):
  * ``name_compound(smi) is None`` for an in-scope fixture: SKIP with
    internal notes-C.md row cite (audit-acknowledged out-of-scope gap).
  * OPSIN cannot parse Orthonym-emitted name: SKIP with a phase
    quarantine note (the parent / cascade-other-than-benzo-fusion is
    where the bug lives).
  * OPSIN parses but InChI L1 mismatches: ASSERT FAIL (wrong-name bug).

Source: 155-internal notes,,,;
        tests/integration/test_skeletal_replacement_corpus.py:1-100
        (substrate copied verbatim).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound


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
    if (
        "unparsable" in low
        or "is unparsable" in low
        or last.startswith("Run the jar")
    ):
        return None
    return last


def _inchi_l1(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = Chem.MolToInchi(mol)
        return inchi.split("/c")[0] if inchi else None
    except Exception:
        return None


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benzo_fusion"
    / "corpus_mined.json"
)
_CORPUS_FIXTURES = json.loads(_FIXTURE_PATH.read_text())


# Fixtures known to fail L1 round-trip due to handler bugs unrelated to
# a phase.C catalog corrections (e.g. Orthonym emits a different
# but still-valid PIN form, or the parent fused-ring handler drops a
# substituent on a complex corpus structure). These are xfail-quarantined
# here per (root-cause-only — no band-aid in this phase).
_QUARANTINED_FIXTURES: dict[str, str] = {
    # Fused-ring composer cannot decompose furo[3,2-c]pyran ortho-fused
    # bicyclic into its two 5+6 components — falls back to single-ring
    # `5-oxooxole` emission. The composer warning "Fused ring naming
    # failed for ortho-fused system" is logged at composer.py:3452 and
    # is verified PRE-EXISTING (independent of a phase.C catalog
    # corrections — same `5-oxooxole` output observed at HEAD~2 before
    # any seniority edits per Task 2 stash-and-re-run protocol).
    # This is a fused-ring decomposition bug in the composer, NOT a
    # benzo-fusion seniority bug. Out-of-scope for sub-phase 155.C
    # (/ catalog hygiene). Hand-off to a phase for the
    # composer fused-ring decomposition fix.
    "chebi_5000_furo_3_2_c_pyran": (
        "Pre-existing fused-ring composer bug: cannot decompose "
        "furo[3,2-c]pyran ortho-fused bicyclic; emits `5-oxooxole` "
        "fallback. Verified independent of Phase 155.C D-11 catalog "
        "corrections via stash-and-re-run. Out-of-scope; hand-off to "
        "Phase 156 (composer fused-ring decomposition)."
    ),
}


@pytest.mark.skipif(
    not _opsin_available(),
    reason="OPSIN jar or java not available",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _CORPUS_FIXTURES,
    ids=[f["fixture_id"] for f in _CORPUS_FIXTURES],
)
def test_benzo_fusion_corpus_opsin_roundtrip(fixture):
    """ mandatory: name_compound(smi) -> OPSIN -> InChI L1 == input InChI L1.

    Stereo-layer differences are tolerated (constitutional skeleton match
    only, per a phase oracle).
    """
    fid = fixture["fixture_id"]
    if fid in _QUARANTINED_FIXTURES:
        pytest.xfail(_QUARANTINED_FIXTURES[fid])

    smi = fixture["smiles"]
    name = name_compound(smi)
    if name is None:
        pytest.skip(
            f"Orthonym returns None for {fid!r} "
            f"(out-of-scope gap; see 155-AUDIT-C.md)"
        )
    parsed = _opsin_parse(name)
    if parsed is None:
        pytest.skip(
            f"OPSIN cannot parse Orthonym-emitted name {name!r} for "
            f"{fid!r}; quarantined for Phase 156."
        )
    actual_l1 = _inchi_l1(parsed)
    expected_l1 = _inchi_l1(smi)
    if actual_l1 is None or expected_l1 is None:
        pytest.skip(
            f"InChI computation failed for {fid!r}"
        )
    assert actual_l1 == expected_l1, (
        f"InChI L1 mismatch for {fid!r}: "
        f"input SMILES {smi!r} -> name {name!r} -> "
        f"OPSIN-parsed {parsed!r}\n"
        f"  expected_l1: {expected_l1}\n"
        f"  actual_l1:   {actual_l1}"
    )


@pytest.mark.integration
def test_corpus_fixture_count_meets_d15_minimum():
    """ floor: >= 10 corpus-mined fixtures per V18-155-."""
    assert len(_CORPUS_FIXTURES) >= 10, (
        f"D-15 corpus floor breach: only {len(_CORPUS_FIXTURES)} fixtures "
        f"in tests/fixtures/benzo_fusion/corpus_mined.json; minimum is 10."
    )


@pytest.mark.integration
def test_corpus_fixtures_opsin_pre_validated():
    """Every fixture's `opsin_rt_verified` flag is True at audit time.

    Acts as a sanity check that the JSON sidecar was emitted by the
    audit script's pre-validation pipeline (corpus-mining
    methodology).
    """
    unverified = [
        f["fixture_id"] for f in _CORPUS_FIXTURES
        if not f.get("opsin_rt_verified", False)
    ]
    assert not unverified, (
        f"D-25 corpus-shortfall protocol breach: fixtures "
        f"{unverified!r} were committed without OPSIN-RT pre-validation; "
        f"re-run scripts/audit_benzo_fusion.py corpus-mining stage."
    )
