""" a phase (B3b) — pin the 7 two-ring hetero/hetero fused-heterocycle PINs.

Each of these ring systems is a verbatim Blue-Book ``(PIN)`` example whose name the
systematic fusion builder cannot yet spell (it lacks the 1,3-dioxole / 1,2-oxazine /
1,3-oxathiole / 1,2,4-triazine / 1,2,3-oxathiazole components, and never computes
fusion-path indicated hydrogen). They are named by an exact canonical-SMILES entry in
``FUSED_HETEROCYCLE_DATA`` (consulted on the main fused-ring path at
``rules/fused_rings.py:1603``, ahead of the algorithmic fusion producer). The
``iupac_locants`` maps are OPSIN-authoritative (built via ``opsin_atom_locant_map``).

Blue Book citations (``the Blue Book Blue Book``; the ``expected`` column of
``benchmarks/bb_conformance/bb_measure_rows.baseline.jsonl`` IS the verbatim PIN):
  * 2H-furo[2,3-d][1,3]dioxole P-25.3.2.4, the Blue Book (PIN)
  * 5H-pyrido[2,3-d][1,2]oxazine P-25.3.3.1.2, the Blue Book (PIN)
  * 2H,4H-[1,3]oxathiolo[5,4-b]pyrrole the Blue Book (PIN)
  * imidazo[1,2-b][1,2,4]triazine P-25.3.3.1.2(c), the Blue Book (PIN)
  * 3H,5H-[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole the Blue Book (PIN)
  * [1,3]selenazolo[5,4-d][1,3]thiazole P-25.3.2.4(f), the Blue Book (PIN)
  * 1H-thieno[2,3-d]imidazole P-25.3.3.1.2, the Blue Book (PIN)
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
    ("c1cc2c(o1)OCO2",      "2H-furo[2,3-d][1,3]dioxole",                       2,    8),
    ("C1=NOCc2cccnc21",     "5H-pyrido[2,3-d][1,2]oxazine",                     5,    10),
    ("c1cc2c([nH]1)OCS2",   "2H,4H-[1,3]oxathiolo[5,4-b]pyrrole",               2,    8),
    ("c1cnn2ccnc2n1",       "imidazo[1,2-b][1,2,4]triazine",                    None, 9),
    ("[nH]1oc2os[nH]c=2s1", "3H,5H-[1,3,2]oxathiazolo[4,5-d][1,2,3]oxathiazole", 3,    8),
    ("c1nc2[se]cnc2s1",     "[1,3]selenazolo[5,4-d][1,3]thiazole",              None, 8),
    ("c1nc2sccc2[nH]1",     "1H-thieno[2,3-d]imidazole",                        1,    8),
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
    """End-to-end: name_compound emits the Blue-Book PIN (not von Baeyer / abstain /
    the wrong systematic spelling that rows 6-7 shipped before this task)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
def test_neighbor_substituted_form_not_over_matched():
    """A methyl-substituted form has a DIFFERENT canonical SMILES, so the exact key
    must NOT emit the bare-parent name (exact-match keys never over-match)."""
    sub = "Cc1nc2[se]cnc2s1"  # 2-methyl on the selenazolo-thiazole CH
    assert Chem.MolToSmiles(Chem.MolFromSmiles(sub)) not in FUSED_HETEROCYCLE_DATA
    assert name_compound(sub) != "[1,3]selenazolo[5,4-d][1,3]thiazole"


@pytest.mark.unit
def test_existing_neighbor_entries_unchanged():
    """Purely-additive change: pre-existing sibling entries still name as before."""
    assert name_compound("c1cnc2sccc2c1") == "thieno[2,3-b]pyridine"
    assert name_compound("c1ccn2ccnc2c1") == "imidazo[1,2-a]pyridine"
    assert name_compound("c1cnc2[nH]cnc2c1") == "3H-imidazo[4,5-b]pyridine"


@pytest.mark.roundtrip
@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JAR / Java runtime unavailable")
@pytest.mark.parametrize("smiles,expected,taut,natoms", _ROWS)
def test_pin_roundtrips_to_input(smiles, expected, taut, natoms):
    """0-wrong: the emitted PIN parses back (OPSIN) to the input structure."""
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

    rt = opsin_roundtrip_check(smiles, expected)
    assert rt["passed"], f"{expected!r} did not round-trip: {rt}"


# --------------------------------------------------------------------------- #
# RISK-5 — the catalog substructure matcher must not name the WRONG #
# indicated-hydrogen tautomer. #
# #
# `match_fused_heterocycle_core` accepts a hit by heavy-atom skeleton and #
# returns the reference tautomer's baked indicated-H locant. The 3H tautomer #
# of thieno[2,3-d]imidazole (NH on the fusion N adjacent to S) has a DIFFERENT #
# canonical SMILES, so the exact-match entry misses it, but the substructure #
# matcher used to accept it and emit `1H-thieno[2,3-d]imidazole` — a name that #
# states an indicated H the molecule does not have (RT-MISMATCH on canonical #
# SMILES; a wrong PIN). P-25.7.1.3 (the Blue Book): "In preferred IUPAC names, all #
# indicated hydrogen atoms must be cited when the names are constructed in #
# accordance with the principles of fusion nomenclature." The guard verifies #
# the input's real indicated-H locant and re-anchors the descriptor #
# (`1H-` -> `3H-`), which round-trips 0-wrong to the 3H input. #
# --------------------------------------------------------------------------- #
class TestRisk5IndicatedHydrogenTautomerGuard:
    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", [
        "c1[nH]c2sccc2n1",   # 3H tautomer, one writing
        "[nH]1c2sccc2nc1",   # 3H tautomer, another writing (same molecule)
    ])
    def test_wrong_tautomer_gets_corrected_locant(self, smiles):
        # was `1H-thieno[2,3-d]imidazole` (wrong PIN); now the true 3H.
        assert name_compound(smiles) == "3H-thieno[2,3-d]imidazole"

    @pytest.mark.unit
    def test_correct_tautomer_and_substituted_unchanged(self):
        # P-25.3.3.1.2 / the Blue Book: the 1H tautomer and its 2-methyl derivative
        # keep `1H-` (the guard only fires on a genuine locant mismatch).
        assert name_compound("c1nc2sccc2[nH]1") == "1H-thieno[2,3-d]imidazole"
        assert (name_compound("Cc1nc2sccc2[nH]1")
                == "2-methyl-1H-thieno[2,3-d]imidazole")

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # The guard must NOT over-fire on any correct-tautomer catalog hit:
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("Cn1ccc2ccccc21", "1-methyl-1H-indole"),        # N-substituted-at-iH
        ("c1ccc2[nH]cnc2c1", "1H-benzimidazole"),
        ("c1ccc2c(c1)Nc1ccccc1O2", "10H-phenoxazine"),
        ("c1ccc2c(c1)Cc1ccccc1O2", "9H-xanthene"),       # sp3 CH2 indicated H
        ("O=C1c2ccccc2-c2ccccc21", "9H-fluoren-9-one"),  # oxo-consumed iH: no misfire
    ])
    def test_guard_does_not_overfire(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.roundtrip
    @pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JAR / Java runtime unavailable")
    def test_corrected_tautomer_roundtrips(self):
        # 0-wrong: the corrected `3H-` name parses back to the 3H input.
        from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

        rt = opsin_roundtrip_check("c1[nH]c2sccc2n1", "3H-thieno[2,3-d]imidazole")
        assert rt["passed"], f"3H- did not round-trip: {rt}"
