""" a phase (B2 / remainder) — pin the 5 two-ring fused-parent PINs.

Each of these ring systems is a verbatim Blue-Book ``(PIN)`` example whose name the
systematic fusion / von-Baeyer path degrades (it lacks these small-ring components /
fusion-path indicated hydrogen and emits a von-Baeyer name, e.g.
``2,7-dioxabicyclo[4.3.0]nona-1(9),3,5-triene`` for furo[3,2-b]pyran). They are named
by an exact canonical-SMILES entry in ``FUSED_HETEROCYCLE_DATA`` (consulted on the main
fused-ring path at ``rules/fused_rings.py:1624``, ahead of the algorithmic fusion /
von-Baeyer producer). The ``iupac_locants`` maps are OPSIN-authoritative (built via
``opsin_atom_locant_map``). Mirrors 6f1fe6333 (a phase B3b).

Row 4 (``1H-cyclopenta[8]annulene``) is a pure carbocycle and is confirmed to route
through this same lookup (``1H-indene`` / ``pyrene`` are named by this table too).

Blue Book citations (``the Blue Book Blue Book``; the ``expected`` column of
``benchmarks/bb_conformance/bb_measure_rows.baseline.jsonl`` IS the verbatim PIN):
  * 2H-furo[3,2-b]pyran, the Blue Book (PIN)
  * 2H-1,3-benzoxathiole, the Blue Book (PIN)
  * pyrrolo[3,2-b]pyrrole, the Blue Book (PIN) [fully mancude, no iH]
  * 1H-cyclopenta[8]annulene, the Blue Book (PIN)
  * 1H,3H-thieno[3,4-c]thiophene, the Blue Book (PIN)
"""
import shutil
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    get_fused_heterocycle_name,
)

_OPSIN_JAR = Path(__file__).resolve().parents[3] / "opsin-cli-2.9.0-jar-with-dependencies.jar"
_OPSIN_OK = shutil.which("java") is not None and _OPSIN_JAR.is_file()

# (canonical SMILES key, expected PIN, tautomer_locant, parent_atoms)
_ROWS = [
    ("C1=COC2=CCOC2=C1",       "2H-furo[3,2-b]pyran",           2,    9),
    ("c1ccc2c(c1)OCS2",        "2H-1,3-benzoxathiole",          2,    9),
    ("C1=CC2=NC=CC2=N1",       "pyrrolo[3,2-b]pyrrole",         None, 8),
    ("C1=CC=CC2=C(C=C1)C=CC2", "1H-cyclopenta[8]annulene",      1,    11),
    ("c1scc2c1CSC2",           "1H,3H-thieno[3,4-c]thiophene",  1,    8),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_key_is_canonical(smiles, expected, taut, natoms):
    """The dict key must be the RDKit-canonical SMILES (else the lookup misses it)."""
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == smiles


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_catalog_entry_present_and_shaped(smiles, expected, taut, natoms):
    """Each system has a catalog entry with the baked PIN and a full locant map."""
    assert smiles in FUSED_HETEROCYCLE_DATA
    data = FUSED_HETEROCYCLE_DATA[smiles]
    assert data["name"] == expected
    assert data["tautomer_locant"] == taut
    assert data["parent_atoms"] == natoms
    # non-empty locant map that covers every heavy atom
    locs = data["iupac_locants"]
    assert set(locs.keys()) == set(range(natoms))


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_get_fused_heterocycle_name(smiles, expected, taut, natoms):
    """The exact-match lookup returns the PIN string verbatim."""
    got = get_fused_heterocycle_name(Chem.MolFromSmiles(smiles))
    assert got is not None
    assert got[0] == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_name_compound_emits_pin(smiles, expected, taut, natoms):
    """End-to-end: name_compound emits the Blue-Book PIN (not the von-Baeyer
    degradation / abstain these rows shipped before this task)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_neighbor_substituted_form_not_over_matched():
    """A methyl-substituted form has a DIFFERENT canonical SMILES, so the exact key
    must NOT emit the bare-parent name (exact-match keys never over-match)."""
    sub = "Cc1ccc2c(c1)OCS2"  # 6-methyl on the benzoxathiole benzo ring
    assert Chem.MolToSmiles(Chem.MolFromSmiles(sub)) not in FUSED_HETEROCYCLE_DATA
    assert name_compound(sub) != "2H-1,3-benzoxathiole"


@pytest.mark.unit
def test_existing_neighbor_entries_unchanged():
    """Purely-additive change: pre-existing sibling entries still name as before,
    including the carbocyclic siblings that share row 4's route (the Blue Book)."""
    assert name_compound("c1cnc2sccc2c1") == "thieno[2,3-b]pyridine"
    assert name_compound("c1ccn2ccnc2c1") == "imidazo[1,2-a]pyridine"
    assert name_compound("C1=Cc2ccccc2C1") == "1H-indene"
    assert name_compound("c1cc2ccc3cccc4ccc(c1)c2c34") == "pyrene"


@pytest.mark.roundtrip
@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JAR / Java runtime unavailable")
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_pin_roundtrips_to_input(smiles, expected, taut, natoms):
    """0-wrong: the emitted PIN parses back (OPSIN) to the input structure."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    rt = opsin_roundtrip_check(smiles, expected)
    assert rt["passed"], f"{expected!r} did not round-trip: {rt}"
