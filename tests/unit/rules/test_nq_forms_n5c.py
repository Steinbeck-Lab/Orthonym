"""Roadmap N5c: short or hetero-ended 'a' chains become substitutive prefixes, at the writers.

* (the Blue Book, 'Skeletal replacement ('a') nomenclature for acyclic
  parent hydrides'): "The chain must be terminated by a C atom or one of the following
  heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, or Tl."
* (:23348): an 'a' name is used "when four or more heterounits are present in a
  unbranched chain containing at least one carbon atom".
* (:27633) 'R-oxy' and (:27667) the retained 'methoxy', 'ethoxy',
  'phenoxy'; (:27649) '(R)sulfanyl'; '(R)amino';
  (:32998) the amido prefix 'acetamido'.

The writers -- the terminal-fragment namer (``rules.terminal_fragment``) and the floor
(``assembly.universal_substituent``) -- walk the carbon chain when the 'a' chain is not
licensed and every heteroatom it would cut can root a composed prefix
(``book_prefixes.hetero_roots_composable``); a charged or hypervalent heteroatom (a
sulfoxide S+, a nitrone N+, a sulfate S) keeps the chain spelling, so the writer never
trades one 'a' chain for another inside a branch.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import (
    a_chain_licensed, hetero_roots_composable, mechanical_forms)
from orthonym.assembly.universal_substituent import name_universal_substituent_prefix
from orthonym.cli import _emit_tier_flags
from orthonym.rules.terminal_fragment import terminal_fragment_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402


@pytest.mark.parametrize("smiles,licensed", [
    ("CCCC", True),                    # a carbon chain
    ("CCOCCOCCOCCOCC", True),          # four O, C ends:23348)
    ("COCCOCCOC", False),              # three heteroatoms
    ("OCCOCCOCCOC", False),            # an O end:6465)
    ("SSSSS", False),                  # no carbon atom
    # heterounits, not heteroatoms:23350: "-SS-, disulfanediyl" is one)
    ("CCOCCSSCCOCC", False),           # O, SS, O: three units
    ("CCOCCSSCCOCCOCC", True),         # O, SS, O, O: four units
    ("C[SiH2]O[SiH2]COCCOC", False),   # SiOSi (one unit,:23350), O, O
    ("CSSSCOCCOCCOC", False),          # a trisulfane run is no heterounit (:23385)
])
def test_a_chain_licence(smiles, licensed):
    mol = Chem.MolFromSmiles(smiles)
    assert a_chain_licensed(mol, list(range(mol.GetNumAtoms()))) is licensed


@pytest.mark.parametrize("smiles,composable", [
    ("COC", True),
    ("C[N+](C)(C)C", True),            # ammonium: 'azaniumyl'
    ("C[Si](C)(C)C", True),
    ("CN=CC", True),
    ("C[S+](C)[O-]", False),           # a sulfoxide written S+-O-
    ("CC=[N+](C)[O-]", False),         # a nitrone N+
    ("COS(=O)(=O)OC", False),          # a hypervalent sulfur
    ("COOC", False),                   # an -OO- link: '(R)peroxy' is not built here
    ("CSSC", True),                    # '[(R)sulfanyl]sulfanyl'
])
def test_hetero_roots_composable(smiles, composable):
    mol = Chem.MolFromSmiles(smiles)
    assert hetero_roots_composable(mol, [a.GetIdx() for a in mol.GetAtoms()]) is composable


#: (SMILES with the group after a benzene ring written first, book spelling, terminal
#: fragment's mechanical spelling, floor's mechanical spelling); the group is every atom
#: from index 6 on, attached at atom 6
GROUPS = [
    ("c1ccccc1CNC1CC1", "(cyclopropylamino)methyl", "2-(cyclopropan-1-yl)-2-azaethyl",
     "2-(cyclopropan-1-yl)-2-azaethan-1-yl"),
    ("c1ccccc1COC", "methoxymethyl", "2-oxapropyl", "2-oxapropan-1-yl"),
    ("c1ccccc1OC", "methoxy", "1-oxaethyl", "1-oxaethan-1-yl"),
    ("c1ccccc1OCCOC", "2-methoxyethoxy", "1,4-dioxapentyl", "1,4-dioxapentan-1-yl"),
    ("c1ccccc1SC", "methylsulfanyl", "1-thiaethyl", "1-thiaethan-1-yl"),
    ("c1ccccc1N(C)C", "dimethylamino", "1-methyl-1-azaethyl", "1-methyl-1-azaethan-1-yl"),
    ("c1ccccc1NC(C)=O", "acetamido", "2-methyl-3-oxa-1-azaprop-2-en-1-yl",
     "2-oxo-1-azapropan-1-yl"),
]


def _group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, set(range(6, mol.GetNumAtoms()))


@pytest.mark.parametrize("smiles,book,tf_old,floor_old", GROUPS)
def test_terminal_fragment_composes_the_prefix(smiles, book, tf_old, floor_old):
    mol, frag = _group(smiles)
    assert terminal_fragment_name(mol, frag, 6).name == book
    with mechanical_forms():
        assert terminal_fragment_name(mol, frag, 6).name == tf_old


@pytest.mark.parametrize("smiles,book,tf_old,floor_old", GROUPS)
def test_floor_composes_the_prefix(smiles, book, tf_old, floor_old):
    mol, frag = _group(smiles)
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == book
    with mechanical_forms():
        assert name_universal_substituent_prefix(mol, sorted(frag), 6) == floor_old


def test_floor_composes_an_ammonium_root():
    mol, frag = _group("c1ccccc1CC[N+](C)(C)C")
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == \
        "2-(trimethylazaniumyl)ethyl"


def test_a_licensed_a_chain_stays():
    # four O, C ends: the 'a' chain is the book's:23348)
    mol, frag = _group("c1ccccc1COCCOCCOCCOC")
    assert terminal_fragment_name(mol, frag, 6).name == "2,5,8,11-tetraoxadodecyl"
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == \
        "2,5,8,11-tetraoxadodecan-1-yl"


def test_a_charged_heteroatom_keeps_the_chain_spelling():
    # a sulfoxide S+ has no composed prefix here: the writer keeps its chain spelling
    # rather than cut the chain into a branch spelled as another 'a' chain
    mol, frag = _group("c1ccccc1C[S+](C)[O-]")
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == \
        "2-oxido-2-thiapropan-2-ium-1-yl"


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym().name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


VENADAPARIB = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"

#: rows whose name carried an 'a' chain the book does not allow at the base
E2E = [
    (VENADAPARIB, "best-effort", "{3-[(cyclopropylamino)methyl]azetidine-1-carbonyl}"),
    (VENADAPARIB, "valid", "{3-[(cyclopropylamino)methyl]azetidine-1-carbonyl}"),
    ("C[Si](C)(C)O[Si](CCCNC(=O)OC=C)(O[Si](C)(C)C)O[Si](C)(C)C", "best-effort",
     "{tris[(trimethylsilyl)oxy]silyl}"),
    ("CC(=O)CC(=O)OCCN(C)C=O", "best-effort", "[formyl(methyl)amino]ethyl"),
    ("CNC(=O)[C@@H](NC(=O)[C@H](CCCc1ccccc1)[C@H](C)N(O)C=O)C(C)(C)C", "valid",
     "[formyl(hydroxy)amino]"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,text", E2E)
def test_no_short_or_hetero_ended_chain_end_to_end(smiles, tier, text):
    row = _row(smiles, tier)
    name = row.get("name")
    assert name and name_is_rt_exact(name, smiles), row
    forms = F.detect(name)
    assert "ACH" not in forms and "ACH_TERM" not in forms, (name, forms)
    assert text in name, name


@pytest.mark.opsin_gate
def test_an_acyl_part_of_an_amino_prefix_is_enclosed():
    """An acyl group on the nitrogen that roots '(R)amino' is cited with its enclosing
    marks when it is substituted: (the Blue Book) "Parentheses are used
    around compound... prefixes"; '4-(acetylamino)' (:33011). Never 'aminoacetylamino',
    which OPSIN reads as another molecule."""
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    smiles = "NCC(=O)NC(C1CCCCC1)C(=O)NC"
    res = name_universal_substitutive(Chem.MolFromSmiles(smiles))
    assert res is not None and "[(aminoacetyl)amino]" in res.name, res
    assert "aminoacetylamino" not in res.name
    assert name_is_rt_exact(res.name, smiles), res.name

# (the Blue Book,:23350): four or more heterounits, -SS- counting as one.
# A chain of three units is cut (the heteroatoms root prefixes, OPSIN reads the name
# back); an -OO- link keeps the chain spelling (residual: the '(R)peroxy' writer).
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,book,mechanical", [
    ("c1ccccc1COCCSSCCOC", "(2-{[(2-methoxyethyl)sulfanyl]sulfanyl}ethoxy)methyl",
     "2,9-dioxa-5,6-dithiadecyl"),
    ("c1ccccc1COCCOOCCOC", "2,5,6,9-tetraoxadecyl", "2,5,6,9-tetraoxadecyl"),
])
def test_heterounits_decide_the_cut(smiles, book, mechanical):
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    mol = Chem.MolFromSmiles(smiles)
    frag = set(range(6, mol.GetNumAtoms()))
    assert terminal_fragment_name(mol, frag, 6).name == book
    with mechanical_forms():
        assert terminal_fragment_name(mol, frag, 6).name == mechanical
    res = name_universal_substitutive(mol)
    assert res is not None and name_is_rt_exact(res.name, smiles), res
