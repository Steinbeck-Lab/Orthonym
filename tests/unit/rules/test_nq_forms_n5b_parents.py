"""Roadmap N5b (ring parents): a fusable ring system takes its fusion name, not a von Baeyer
name, as the parent of the general engine and of the floor.

* (the Blue Book): "Fusion nomenclature gives preferred IUPAC names only
  to compounds having at least two rings of at least five or more members.... When fusion
  names are not allowed, unsaturated von Baeyer ring system names are preferred IUPAC
  names".
* (:24221): hydro prefixes for the partly and fully saturated systems.
* (:24864): "After the introduction of indicated and 'added indicated hydrogen'
  atoms, all substituent groups not expressed as suffixes are cited as prefixes" -- the
  ring C=O of venadaparib's phthalazinone cited as 'oxo' on '1,2-dihydrophthalazine'.

Writers: ``general_engine._emit_fused_ring`` (from ``name_general_ring`` and the assembly
tier ``_name_terminal_ring_assembly``) and the floor's ``_book_fused_spine``.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.assembly.universal_substituent import (
    name_universal_substituent_prefix, name_universal_substitutive)
from orthonym.cli import _emit_tier_flags

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


@pytest.mark.parametrize("smiles,expected", [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("CC1CCc2ccccc2C1", "2-methyl-1,2,3,4-tetrahydronaphthalene"),
])
def test_floor_fused_parent(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert name_universal_substitutive(mol).name == expected
    with mechanical_forms():
        assert "bicyclo[4.4.0]" in name_universal_substitutive(mol).name


def test_floor_fused_branch():
    mol = Chem.MolFromSmiles("CCc1ccc2ccccc2c1")
    assert name_universal_substituent_prefix(
        mol, list(range(2, mol.GetNumAtoms())), 2) == "naphthalen-2-yl"


def test_floor_keeps_von_baeyer_for_a_four_membered_ring():
    # rings of 6 and 4 members: the von Baeyer name is the book's (:23725)
    mol = Chem.MolFromSmiles("CC1Cc2ccccc21")
    assert "bicyclo[4.2.0]" in name_universal_substitutive(mol).name


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


VENADAPARIB = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["valid", "complete", "best-effort"])
def test_venadaparib_carries_no_reject_on_sight_form(tier):
    """Spec Q4: venadaparib free of every form at the valid, complete and best-effort
    tiers. The fused parent is '1-oxo-1,2-dihydrophthalazine' (prefix form; the suffix
    form 'phthalazin-1(2H)-one' is the PIN path's, lanes L1b/L3)."""
    row = _row(VENADAPARIB, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, VENADAPARIB), row
    forms = {t for t in F.detect(name) if t in F.TARGET}
    assert forms == set(), (name, forms)
    assert "1,2-dihydrophthalazine" in name or "phthalazin-1(2H)-one" in name, name
