"""A fusable ring system the catalogue has no entry for takes the parent the book prints (the best-effort writers).

 "Five-membered ring requirement" (the Blue Book): "Fusion nomenclature gives preferred IUPAC names
only to compounds having at least two rings of at least five or more members." 1,3-Benzodioxole is such a system
 "Heteromonocyclic components fused to a benzene ring" (:11813): "an isolated benzo component... ortho-fused
to a heteromonocyclic component of five or more members... treated together as a one-component unit (a benzoheterocycle)"), but ``data.fused_heterocycles`` holds no entry for it, so
``fused_system_form`` declined and the best-effort writers spelled its von Baeyer name
('4-chloro-7,9-dioxabicyclo[4.3.0]nona-1(6),2,4-trien-3-yl'). The parent tables of the PIN fusion lane
(``bridged_fused_pin.parent_table.BB_PARENTS``) hold the name and the numbering; the indicated hydrogen and the
hydro prefixes come from the structure, as for a catalogue parent.

``terminal_ring._book_fused`` uses the printed parents at every tier, so a fusable system the catalogue lacks (the
steroid skeleton, 'cyclopenta[a]phenanthrene') can reach the PIN tier through that writer: intended, as steroids take
the systematic fusion name at the PIN tier (user decision 2026-10-31). The writers that call ``fused_system_form``
without ``printed_parents`` (the default) keep the catalogue alone.
"""
import pytest
from rdkit import Chem

from orthonym.rules import fused_forms
from orthonym.rules.terminal_fragment import terminal_fragment_name
from orthonym.rules.terminal_ring import terminal_ring_name
from tests.support.large_polycycles import be, pin

pytestmark = pytest.mark.unit

# the benzodioxole of PubChem CID 98563356 with its chlorine and its CH2 link, and a 1,4-benzodioxine
PIPERONYL = "Clc1cc2OCOc2cc1CNCC"
BENZODIOXINE = "c1ccc2OCCOc2c1CN"


def _ring_and_attach(mol):
    ring = sorted(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    attach = next(i for i in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()
                  if n.GetIdx() not in ring and n.GetSymbol() == "C")
    return ring, attach


@pytest.mark.parametrize("smiles, prefix", [
    ("c1cc2OCOc2cc1CN", "2H-1,3-benzodioxol-5-yl"),
    (BENZODIOXINE, "2,3-dihydro-1,4-benzodioxin-5-yl"),
])
def test_the_printed_parent_names_a_benzo_heterocycle_the_catalogue_lacks(smiles, prefix):
    mol = Chem.MolFromSmiles(smiles)
    ring, attach = _ring_and_attach(mol)
    #: the fusion name, where the von Baeyer one stood
    assert terminal_ring_name(mol, ring, attach).name == prefix
    # the default of fused_system_form is the catalogue alone (the PIN-tier writers)
    assert fused_forms.fused_system_form(mol, ring, attach) is None
    assert fused_forms.fused_system_form(mol, ring, attach, printed_parents=True).prefix(attach, 1) == prefix


def test_the_fragment_cites_its_prefix_on_the_numbering_of_the_printed_parent():
    # (c): the free valence takes locant 5, then (f) the chlorine takes 6 ('6-chloro-2H-1,3-benzodioxol-5-yl')
    mol = Chem.MolFromSmiles(PIPERONYL)
    ring, attach = _ring_and_attach(mol)
    group = set(ring) | {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"}
    assert terminal_fragment_name(mol, group, attach).name == "6-chloro-2H-1,3-benzodioxol-5-yl"


def test_a_ring_ketone_of_the_printed_parent_takes_the_suffix_and_the_indicated_hydrogen():
    # (:28410): a ketone of a mancude parent with indicated hydrogen is named by direct substitution of
    # its >CH2: '2H-1,3-benzodioxol-2-one'
    mol = Chem.MolFromSmiles("O=C1Oc2ccccc2O1")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    carbonyl = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()
                and any(n.GetSymbol() == "O" and not n.IsInRing() for n in a.GetNeighbors())]
    assert fused_forms.fused_system_ketone_form(mol, ring, carbonyl) is None
    assert fused_forms.fused_system_ketone_form(mol, ring, carbonyl, printed_parents=True).join() == "2H-1,3-benzodioxol-2-one"


def test_a_name_assembled_from_components_is_not_a_printed_parent():
    # a skeleton no table holds ('3H-imidazo[4,5-c]quinoline') has a fusion name only when the components are assembled and
    # OPSIN numbers it; ``tables_only`` refuses it, so it keeps the von Baeyer spelling: the part of the class not built
    mol = Chem.MolFromSmiles("c1ccc2c(c1)ncc1[nH]cnc12")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert fused_forms._printed_parents(mol, ring) == []


@pytest.mark.opsin_gate
def test_the_benzodioxole_of_the_cage_molecule_has_its_fusion_name_in_the_best_effort_name():
    # PubChem CID 98563356: the name was '4-chloro-7,9-dioxabicyclo[4.3.0]nona-1(6),2,4-trien-3-yl' for the PIN tier's
    # '6-chloro-2H-1,3-benzodioxol-5-yl'
    name = be("C1CC(=O)N([C@H]1CCNCC2=CC3=C(C=C2Cl)OCO3)C[C@@H]4C[C@H]5C=C[C@H]4C56CC6")["name"]
    assert "6-chloro-2H-1,3-benzodioxol-5-yl" in name
    assert "dioxabicyclo" not in name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    # deoxycorticosterone acetate (the 17S,10R natural configuration)
    ("CC(=O)OCC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "2-[(8S,9S,10R,13S,14S,17S)-10,13-dimethyl-3-oxo-2,3,6,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-1H-"
     "cyclopenta[a]phenanthren-17-yl]-2-oxoethyl acetate"),
    # the same skeleton without stereo: 3-hydroxy, 5-ene
    ("CC(=O)OCC(=O)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
     "2-(3-hydroxy-10,13-dimethyl-2,3,4,7,8,9,10,11,12,13,14,15,16,17-tetradecahydro-1H-"
     "cyclopenta[a]phenanthren-17-yl)-2-oxoethyl acetate"),
])
def test_a_steroid_ester_takes_the_fusion_name_of_its_skeleton_at_the_pin_tier(smiles, name):
    """The default tier names the steroid skeleton by fusion nomenclature, not by a von Baeyer or stereoparent spelling.

     "Five-membered ring requirement" (the Blue Book): "Fusion nomenclature gives preferred IUPAC
    names only to compounds having at least two rings of at least five or more members" (6 + 6 + 6 + 5 here).
     "Numbering fused ring systems" (:12490,:12493): "Anthracene, phenanthrene, acridine, carbazole, xanthene
    and its chalcogen analogues, purine, and cyclopenta[a]phenanthrene are exceptions; traditional numberings are
    retained", so the 17-yl is the steroid's C-17. Steroids take the systematic fusion name at the PIN tier (user
    decision 2026-10-31); the hydro prefixes follow (:24221). OPSIN 2.9.0 reads each name to the input's
    full InChIKey (scripts/diagnose.py: OK; an InChIKey and an InChIKey). Before,
    the default tier declined these rows and the best-effort tier gave the pregnane or the von Baeyer name.
    """
    row = pin(smiles)
    assert (row["tier"], row["name"]) == ("pin_verified", name)
