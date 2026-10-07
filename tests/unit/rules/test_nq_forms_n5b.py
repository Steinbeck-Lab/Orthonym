"""Roadmap N5b (ring substituents): a fusable ring system takes its fusion name, not a von
Baeyer name, at the writers of ring substituents.

* 'Five-membered ring requirement' (the Blue Book;:23710): "Fusion
  nomenclature gives preferred IUPAC names only to compounds having at least two rings of
  at least five or more members.... When fusion names are not allowed, unsaturated von
  Baeyer ring system names are preferred IUPAC names".
* (:24221): hydro prefixes for the partly and fully saturated systems,
  'decahydronaphthalene (PIN) bicyclo[4.4.0]decane' (:24233).
* (:3219) numbering: (b) indicated hydrogen (:3246), (c) free valences (:3256),
  (e) hydro prefixes (:3288), (f) prefixes (:3301).

``rules.fused_forms.fused_system_form`` builds the name from the catalogue's mancude parent
(``partial_saturation._resolve_oxo_parent``) and the hydrogen rule of
``rules.ring_hydrogen``; the writers -- ``ring_substituents.polycyclic_core_numbering``
(the recursive composer's ring cores) and ``terminal_ring.terminal_ring_name`` (the
terminal-fragment namer's ring cores) -- use it before the von Baeyer analyzers.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.cli import _emit_tier_flags
from orthonym.rules.fused_forms import fused_system_form, is_fusable_system, locant_key
from orthonym.rules.ring_substituents import polycyclic_core_numbering
from orthonym.rules.terminal_ring import terminal_ring_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


def _ring_atoms(mol):
    return sorted({a for r in mol.GetRingInfo().AtomRings() for a in r})


@pytest.mark.parametrize("smiles,fusable", [
    ("c1ccc2ccccc2c1", True),          # naphthalene
    ("C1Cc2ccccc2C1", True),           # indane: rings of 5 and 6 members
    ("C1CCC2CCCCC2C1", True),          # decalin (:24233)
    ("c1ccc2c(c1)CC2", False),         # 6 + 4: von Baeyer is the book's (:23725)
    ("C1CC2CCC1C2", False),            # norbornane: bridged
    ("C1CCC2(CC1)CCCC2", False),       # a spiro system
])
def test_is_fusable_system(smiles, fusable):
    mol = Chem.MolFromSmiles(smiles)
    assert is_fusable_system(mol, _ring_atoms(mol)) is fusable


def test_locant_key_orders_lettered_locants():
    assert sorted([5, "4a", 4, "8a", 1], key=locant_key) == [1, 4, "4a", 5, "8a"]


@pytest.mark.parametrize("smiles,parent", [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("C1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalene"),
    ("C1Cc2ccccc2C1", "2,3-dihydro-1H-indene"),
    ("C1CCC2CCCCC2C1", "decahydronaphthalene"),       # (:3007),:24233
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("C1Cc2ccccc2N1", "2,3-dihydro-1H-indole"),
    ("c1ccc2ncccc2c1", "quinoline"),
])
def test_fused_parent_names(smiles, parent):
    mol = Chem.MolFromSmiles(smiles)
    form = fused_system_form(mol, _ring_atoms(mol))
    assert form is not None and form.parent == parent


@pytest.mark.parametrize("smiles,parent,oxo_locant", [
    # a ring C=O cited as 'oxo' is a saturated position of the parent:24864)
    ("O=c1[nH]ncc2ccccc12", "1,2-dihydrophthalazine", 1),
    ("O=c1ccoc2ccccc12", "4H-1-benzopyran", 4),
    ("O=c1ccc2ccccc2o1", "2H-1-benzopyran", 2),
])
def test_fused_parent_with_an_oxo_prefix(smiles, parent, oxo_locant):
    mol = Chem.MolFromSmiles(smiles)
    ring = _ring_atoms(mol)
    carbonyl = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetIdx() in ring and any(nb.GetSymbol() == "O" and nb.GetIdx() not in ring
                                                  for nb in a.GetNeighbors()))
    form = fused_system_form(mol, ring, None, [carbonyl])
    assert form is not None
    assert (form.parent, form.numbering[carbonyl]) == (parent, oxo_locant)


# (the Blue Book): with one indicated hydrogen and one free valence,
# "the indicated hydrogen atoms are placed at peripheral atoms that will accommodate
# these... free valences": a free valence on an atom with no hydrogen in the mancude
# parent takes the indicated hydrogen, the hydro prefixes come after Example 5,
#:25462-:25477, 'Parent hydride with free valence 2H-isoindol-2-yl' -> '1,3-dioxo-1,3-
# dihydro-2H-isoindol-2-yl (preferred prefix)'; '2-(1,3,4,5-tetrahydro-2H-2-benzazepin-2-
# yl)ethan-1-ol (PIN)',:24774). A ring carbon keeps a hydrogen in the mancude parent, so
# the lowest locant takes it ('2,3-dihydro-1H-inden-2-yl (preferred prefix)',:17374).
# (Table 3.2,:17380, prints 'isoindolin-2-yl... 2,3-dihydro-1H-isoindol-2-yl'; the rule
# text and the two PIN examples above are followed here.)
@pytest.mark.parametrize("smiles,prefix", [
    ("CN1C(=O)c2ccccc2C1=O", "1,3-dihydro-2H-isoindol-2-yl"),
    ("CN1CCCc2ccccc2C1", "1,3,4,5-tetrahydro-2H-2-benzazepin-2-yl"),
    ("Cn1c(Br)nc2c(=O)[nH]c(N)nc12", "1,6-dihydro-9H-purin-9-yl"),
    ("CC1Cc2ccccc2C1", "2,3-dihydro-1H-inden-2-yl"),
    ("CN1CCc2ccccc21", "2,3-dihydro-1H-indol-1-yl"),
    ("Cn1cc2ccccc2n1", "2H-indazol-2-yl"),
])
def test_the_free_valence_atom_takes_the_indicated_hydrogen(smiles, prefix):
    mol = Chem.MolFromSmiles(smiles)
    ring = _ring_atoms(mol)
    fv = next(a for a in ring if mol.GetBondBetweenAtoms(0, a))
    carriers = [a for a in ring if a != fv and any(
        not nb.IsInRing() and nb.GetIdx() != 0 for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
    form = fused_system_form(mol, ring, fv, carriers)
    assert form is not None and form.prefix(fv) == prefix, form


@pytest.mark.parametrize("smiles", [
    "c1ccc2c(c1)CC2",                  # 6 + 4 (:23725)
    "c1ccc2c(c1)OCO2",                 # 1,3-benzodioxole: no catalogue parent here
    "c1ccc2[n+](C)cccc2c1",            # a charged ring atom
    # a lambda-4 ring sulfur, the Blue Book;:12452): the
    # catalogue's '[1,3,2]benzodioxathiole' is the divalent-S ring, another molecule
    "c1ccc2c(c1)O[SH2]O2",
])
def test_fused_system_form_declines(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert fused_system_form(mol, _ring_atoms(mol)) is None


def test_fused_form_does_not_depend_on_atom_order():
    """The numbering tie-break is the canonical rank, so the name and the locant of
    every atom are the same from any atom order of the input."""
    import random
    random.seed(20261004)
    mol = Chem.MolFromSmiles("CC1CCc2ccccc2C1")
    seen = set()
    for _ in range(8):
        perm = list(range(mol.GetNumAtoms()))
        random.shuffle(perm)
        m2 = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm),
                                                 canonical=False))
        ring = _ring_atoms(m2)
        methyl_carrier = next(a for a in ring if any(
            nb.GetIdx() not in ring for nb in m2.GetAtomWithIdx(a).GetNeighbors()))
        form = fused_system_form(m2, ring, None, [methyl_carrier])
        seen.add((form.parent, form.numbering[methyl_carrier]))
    assert seen == {("1,2,3,4-tetrahydronaphthalene", 2)}


def test_fused_system_form_is_off_in_mechanical_mode():
    mol = Chem.MolFromSmiles("C1CCc2ccccc2C1")
    with mechanical_forms():
        assert fused_system_form(mol, _ring_atoms(mol)) is None


def test_terminal_ring_names_a_fused_substituent():
    # (1,2,3,4-tetrahydronaphthalen-2-yl)methanol: the free valence takes 2, not 3
    mol = Chem.MolFromSmiles("OCC1CCc2ccccc2C1")
    ring = _ring_atoms(mol)
    tr = terminal_ring_name(mol, ring, 2)
    assert (tr.name, tr.basis) == ("1,2,3,4-tetrahydronaphthalen-2-yl", "fused_book")
    assert tr.numbering[2] == 2
    with mechanical_forms():
        assert terminal_ring_name(mol, ring, 2).basis == "von_baeyer"


def test_polycyclic_core_numbering_uses_the_fusion_name():
    # [(2S)-1,2,3,4-tetrahydronaphthalen-2-yl]methanol: the ring fragment's SMILES
    # carries its stereocentre, so the two catalogue lookups keyed by that SMILES miss
    # it (the study's dev2000 row of this writer) and the fusion form names it;
    # the free valence takes 2, not 3, the Blue Book)
    mol = Chem.MolFromSmiles("OC[C@H]1CCc2ccccc2C1")
    ring = tuple(_ring_atoms(mol))
    pos, tail = polycyclic_core_numbering(mol, ring, 2, [], allow_mancude=True)
    assert tail == "1,2,3,4-tetrahydronaphthalen-2-yl" and pos[2] == 2
    with mechanical_forms():
        _pos, old_tail = polycyclic_core_numbering(mol, ring, 2, [], allow_mancude=True)
    assert old_tail == "bicyclo[4.4.0]deca-1,3,5-trien-8-yl"


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


#: rows whose ring substituent was a von Baeyer name at the base (writers
#: ``ring_substituents._cage_core_numbering`` and ``terminal_ring._spell_ring_analysis``)
E2E = [
    ("NCc1c[nH]c(=S)n1[C@H]1CCc2c(F)cc(F)cc2C1", "best-effort",
     "tetrahydronaphthalen-2-yl"),
    ("CC(=O)CCc1cc2ccc(=O)oc2cc1O", "best-effort", "2H-1-benzopyran"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,text", E2E)
def test_fused_ring_substituents_end_to_end(smiles, tier, text):
    row = _row(smiles, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    assert "VBF" not in F.detect(name), name
    assert text in name, name
