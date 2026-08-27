"""Aminoglycoside / charged-cyclitol glycosides on the universal (T4) floor.

WHY THIS FILE EXISTS -- a REFUTATION, pinned so it cannot silently rot.

A v38 tail brief asserted that ``assembly.universal_substituent.
name_universal_substitutive`` MIS-RENDERS aminoglycoside / cyclitol-glycoside
inputs "via skeletal ('a')-replacement, producing OPSIN-UNPARSEABLE names"
(``2-oxacyclohexane``, ``1-oxamethan-1-yl``, ``2-oxaethan-1-yl``), and that the
renderer therefore needed to be rewritten to emit Hantzsch-Widman / ``oxy``
forms to rescue lost breadth.

Measured at HEAD ``c7d8643b`` with OPSIN 2.9.0 (the gate parser), that premise
is FALSE.  The oxa-replacement forms the module emits are *valid* OPSIN
replacement nomenclature; OPSIN parses every one of them and they round-trip:

  * the three brief witnesses (below) each parse AND full-InChIKey round-trip
    (constitution + stereo + charge);
  * a hand-built diverse class corpus (oxane/oxolane/thiane/piperidine-
    iminosugar/oxepane rings, mono- & di-saccharides, charged & neutral) is
    12/12 full-InChIKey round-trip;
  * the whole 413-member ``sugar_glycan`` backlog witness set
    (``): the raw producer emits
    279 and abstains 134, and of the 279 emitted names **0 are OPSIN-
    unparseable** -- 277 skeleton-match, 265 full-InChIKey-match through
    ``opsin_parse``.

So there is no unparseable-rendering defect and no breadth to rescue on this
class: every emitted name already parses.  ``2-oxacyclohexane`` and
``oxane`` are BOTH accepted by OPSIN and denote the same ring; converting the
former to the latter would be a pure spelling change on NON-PIN best-effort
(T4) output -- it moves neither the round-trip metric nor the PIN oracle, and
touches ``_name_ring_spine`` which serves the entire monocyclic-hetero class,
so it carries regression risk with zero breadth upside.  The renderer was
therefore left unchanged; full evidence in the task report.

These tests pin the load-bearing invariant that actually matters -- the class
ROUND-TRIPS through the universal floor -- and are deliberately SPELLING-
AGNOSTIC so that a future, separately-motivated PIN-spelling improvement
(``oxane`` / ``(oxan-2-yl)oxy``) does not have to fight them: it only has to
keep the round-trip.

Targeted-file run only (project convention -- avoids the OPSIN-pipe deadlock
of a full ``pytest tests/`` run):
    .venv/bin/python -m pytest \
        tests/unit/assembly/test_universal_substituent_hetero_ring.py -q
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.validation.opsin_roundtrip import opsin_parse
from orthonym.validation.reconstruct import verify_or_none


def _heavy(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _skel_inchikey(smiles: str):
    """InChIKey with stereo and charge stripped -- the constitution skeleton."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    for a in mol.GetAtoms():
        a.SetFormalCharge(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _emit(smiles: str):
    """Name via the universal floor; assert it emits with complete coverage.

    Returns the ``UniversalResult``.  A None here is a real breadth loss for
    this class (not merely an ugly spelling), so it is a hard failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    result = name_universal_substitutive(mol)
    assert result is not None, f"universal floor abstained on {smiles!r}"
    assert result.covers == _heavy(smiles), (
        f"atom-coverage gap for {smiles!r}: covers={result.covers} "
        f"heavy={_heavy(smiles)}"
    )
    return result


# The three brief witnesses -- an aminoglycoside (W1), a charged cyclitol
# glycoside (W2), and a 2-deoxystreptamine-style diaminocyclitol glycoside
# (W3).  Each emits, PARSES in OPSIN, and full-InChIKey round-trips.
_WITNESSES = {
    "W1_aminoglycoside": (
        "[NH3+]C[C@H]1O[C@H](OC2[C@@H](O)[C@H](O)C([NH3+])C[C@H]2O)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    ),
    "W2_charged_cyclitol_glycoside": (
        "OC1[C@@H](O)[C@H](O)C(OC2OC(CO)C(O)C(O)C2O)C([NH3+])[C@H]1O"
    ),
    "W3_diaminocyclitol_glycoside": (
        "NC1CC(N)C(OC2OC(CO)C(O)C(O)C2O)C(O)C1O"
    ),
}


@pytest.mark.roundtrip
@pytest.mark.parametrize("tag,smiles", list(_WITNESSES.items()), ids=list(_WITNESSES))
def test_brief_witness_parses_and_full_round_trips(tag, smiles):
    """The renderer's oxa-replacement output is OPSIN-parseable and
    full-InChIKey correct -- refutes the 'OPSIN-UNPARSEABLE' premise."""
    result = _emit(smiles)
    # Explicit no-unparseable guard (the exact claim under test).
    assert opsin_parse(result.name) is not None, (
        f"{tag}: OPSIN could not parse {result.name!r}"
    )
    # And it denotes the right molecule down to stereo + charge.
    assert verify_or_none(result.name, smiles) == result.name, (
        f"{tag}: {result.name!r} did not full-InChIKey round-trip"
    )


# A minimal saturated-hetero-monocycle sanity spread across ring size and
# heteroatom.  Spelling-agnostic: the invariant is that each sugar/cyclitol
# ring is rendered as an OPSIN-parseable hetero ring that round-trips (whether
# spelled ``oxane`` or ``oxacyclohexane`` is not asserted).
_RING_CLASS = {
    "methyl_glucoside_6O": "COC1OC(CO)C(O)C(O)C1O",         # oxane (6-ring, 1 O)
    "ribofuranoside_5O": "COC1OC(CO)C(O)C1O",               # oxolane (5-ring, 1 O)
    "thiosugar_6S": "OCC1SC(O)C(O)C(O)C1O",                 # thiane (6-ring, 1 S)
    "iminosugar_6N": "OCC1NC(CO)C(O)C(O)C1O",               # piperidine (6-ring, 1 N)
    "cellobiose": "OCC1OC(OC2C(O)C(O)C(CO)OC2O)C(O)C(O)C1O",  # O-glycoside bridge
}


@pytest.mark.roundtrip
@pytest.mark.parametrize("tag,smiles", list(_RING_CLASS.items()), ids=list(_RING_CLASS))
def test_hetero_ring_and_glycoside_bridge_round_trip(tag, smiles):
    """Saturated hetero monocycles (O/S/N; 5- and 6-membered) and an
    O-glycosidic ether bridge all render to OPSIN-parseable, round-tripping
    names via the universal floor."""
    result = _emit(smiles)
    parsed = opsin_parse(result.name)
    assert parsed is not None, f"{tag}: unparseable name {result.name!r}"
    # Skeleton match: these class reps carry stereo the raw floor may omit on
    # off-spine centres, so assert the constitution (spelling-agnostic).
    assert _skel_inchikey(smiles) == _skel_inchikey(parsed), (
        f"{tag}: skeleton mismatch for {result.name!r}"
    )
