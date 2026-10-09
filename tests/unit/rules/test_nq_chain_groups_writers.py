"""Roadmap item 12a (task t2): the writers name an N/O group by its book prefix, not as an 'a' chain.

Rule: (the Blue Book, section General rules): "The chain must be
terminated by a C atom or one of the following heteroatoms: P, As, Sb, Bi, Si, Ge, Sn, Pb, B,
Al, Ga, In, or Tl." The prefixes are those of ``test_nq_chain_groups_shapes.py``.

The two writers of the best-effort tier are tested on the same groups, written after a benzene
ring (the group is every atom from index 6 on, attached at atom 6), in the book's spelling and
in the mechanical spelling (``mechanical_forms``), which must stay what it was:

* the terminal-fragment writer (``rules.terminal_fragment``), which has no charged groups and
  no alkylidene parts;
* the floor (``assembly.universal_substituent``), whose branch entry point
  (``name_universal_substituent_prefix``) is the one ``name_substituent`` and the radical rung
  of ``t4_coverage`` call.
"""
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.book_prefixes import hetero_roots_composable, mechanical_forms
from orthonym.assembly.universal_substituent import (
    name_universal_substitutive, name_universal_substituent_prefix)
from orthonym.cli import _emit_tier_flags
from orthonym.rules.terminal_fragment import terminal_fragment_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from name_quality import forms as F  # noqa: E402
from tests.support.rt_assert import name_is_rt_exact  # noqa: E402

#: (SMILES, terminal-fragment book prefix, floor book prefix, terminal mechanical, floor
#: mechanical); ``None``: the writer does not build the group
GROUPS = [
    ("c1ccccc1N=Nc1ccccc1", "phenyldiazenyl", "phenyldiazenyl",
     "2-(cyclohexa-1,3,5-trien-1-yl)-1,2-diazaeth-1-en-1-yl",
     "2-(cyclohexa-1,3,5-trien-1-yl)-1,2-diazaeth-1-en-1-yl"),
    ("c1ccccc1/N=N/C", "(1E)-methyldiazenyl", "methyldiazenyl",
     "(1E)-1,2-diazaprop-1-en-1-yl", "1,2-diazaprop-1-en-1-yl"),
    ("c1ccccc1OO", "hydroperoxy", "hydroperoxy", "1,2-dioxaethyl", "1,2-dioxaethan-1-yl"),
    ("c1ccccc1OOC(C)(C)C", "tert-butylperoxy", "tert-butylperoxy",
     "3,3-dimethyl-1,2-dioxabutyl", "3,3-dimethyl-1,2-dioxabutan-1-yl"),
    ("c1ccccc1N=O", "nitroso", "nitroso", "2-oxa-1-azaeth-1-en-1-yl",
     "2-oxa-1-azaeth-1-en-1-yl"),
    ("c1ccccc1NN", "hydrazinyl", "hydrazinyl", "1,2-diazaethyl", "1,2-diazaethan-1-yl"),
    ("c1ccccc1NNc1ccccc1", "2-phenylhydrazinyl", "2-phenylhydrazinyl",
     "2-(cyclohexa-1,3,5-trien-1-yl)-1,2-diazaethyl",
     "2-(cyclohexa-1,3,5-trien-1-yl)-1,2-diazaethan-1-yl"),
    ("c1ccccc1C=NO", "(hydroxyimino)methyl", "(hydroxyimino)methyl",
     "3-oxa-2-azaprop-1-en-1-yl", "3-oxa-2-azaprop-1-en-1-yl"),
    ("c1ccccc1C=NOC", "(methoxyimino)methyl", "(methoxyimino)methyl",
     "3-oxa-2-azabut-1-en-1-yl", "3-oxa-2-azabut-1-en-1-yl"),
    ("c1ccccc1C(C)=NNC(N)=O", "1-(carbamoylhydrazinylidene)ethyl",
     "1-(carbamoylhydrazinylidene)ethyl",
     "4-amino-1-methyl-5-oxa-2,3-diazapenta-1,4-dien-1-yl",
     "1-methyl-4-oxo-2,3,5-triazapent-1-en-1-yl"),
    ("c1ccccc1C=NN(C)C", "(dimethylhydrazinylidene)methyl", "(dimethylhydrazinylidene)methyl",
     "3-methyl-2,3-diazabut-1-en-1-yl", "3-methyl-2,3-diazabut-1-en-1-yl"),
    ("c1ccccc1N=C(N)N", "(diaminomethylidene)amino", "(diaminomethylidene)amino",
     "2-amino-1,3-diazaprop-1-en-1-yl", "2-amino-1,3-diazaprop-1-en-1-yl"),
    ("c1ccccc1/N=C/c1ccccc1", "[(1E)-phenylmethylidene]amino", "benzylideneamino",
     "(1E)-2-(cyclohexa-1,3,5-trien-1-yl)-1-azaeth-1-en-1-yl",
     "2-(cyclohexa-1,3,5-trien-1-yl)-1-azaeth-1-en-1-yl"),
    ("c1ccccc1ON=Cc1ccccc1", "[(phenylmethylidene)amino]oxy", "(benzylideneamino)oxy",
     "3-(cyclohexa-1,3,5-trien-1-yl)-1-oxa-2-azaprop-2-en-1-yl",
     "3-(cyclohexa-1,3,5-trien-1-yl)-1-oxa-2-azaprop-2-en-1-yl"),
    ("c1ccccc1ONC(C)=O", "acetamidooxy", "acetamidooxy",
     "3-methyl-1,4-dioxa-2-azabut-3-en-1-yl", "3-oxo-1-oxa-2-azabutan-1-yl"),
    ("c1ccccc1C(=O)NO", "hydroxycarbamoyl", "hydroxycarbamoyl",
     "1-oxo-3-oxa-2-azapropyl", "1-oxo-3-oxa-2-azapropan-1-yl"),
    ("c1ccccc1[N+](C)(C)[O-]", None, "dimethyl(oxido)azaniumyl", None,
     "1-methyl-1-oxido-1-azaethan-1-ium-1-yl"),
]


def _group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, set(range(6, mol.GetNumAtoms()))


def _terminal(mol, frag):
    got = terminal_fragment_name(mol, frag, 6)
    return got.name if got is not None else None


@pytest.mark.parametrize("smiles,tf_book,floor_book,tf_mech,floor_mech", GROUPS)
def test_terminal_fragment_composes_the_prefix(smiles, tf_book, floor_book, tf_mech,
                                               floor_mech):
    mol, frag = _group(smiles)
    assert _terminal(mol, frag) == tf_book
    with mechanical_forms():
        assert _terminal(mol, frag) == tf_mech


@pytest.mark.parametrize("smiles,tf_book,floor_book,tf_mech,floor_mech", GROUPS)
def test_floor_composes_the_prefix(smiles, tf_book, floor_book, tf_mech, floor_mech):
    mol, frag = _group(smiles)
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == floor_book
    with mechanical_forms():
        assert name_universal_substituent_prefix(mol, sorted(frag), 6) == floor_mech


def test_an_alkylidene_part_is_left_to_the_chain_spelling():
    """The terminal writer has one-carbon ylidene parts only ('phenylmethylidene'); an
    alkylidene would come out as 'methylmethylidene' for 'ethylidene', so the group is not
    built there (the floor builds '(ethylideneamino)oxy')."""
    mol, frag = _group("c1ccccc1ON=CC")
    assert "methylmethylidene" not in (_terminal(mol, frag) or "")
    assert name_universal_substituent_prefix(mol, sorted(frag), 6) == "(ethylideneamino)oxy"


def test_an_oxime_hydroxy_is_the_hydroxy_prefix_not_an_oxa_chain():
    """'(1-oxamethyl)imino' read -OH wrongly through the part namer of the acyl writer."""
    mol, frag = _group("c1ccccc1C(=O)NO")
    assert "oxamethyl" not in (_terminal(mol, frag) or "")


@pytest.mark.parametrize("smiles,composable", [
    ("COOC", True),                    # -OO- is cut at its first oxygen: '(R)peroxy'
    ("CC=[N+](C)C", True),             # an iminium roots 'azaniumylidene':42298)
    ("CC=[OH+]", True),                # an oxonium roots 'oxidaniumylidene' (:42350)
    ("C[S+](C)[O-]", False),           # a sulfoxide written S+-O- still keeps its chain
    ("CC=[N+](C)[O-]", False),         # a nitrone N+ (two bonds to the path) is not cut here
])
def test_hetero_roots_composable_now_cuts_peroxides_and_iminium(smiles, composable):
    mol = Chem.MolFromSmiles(smiles)
    assert hetero_roots_composable(mol, [a.GetIdx() for a in mol.GetAtoms()]) is composable


# ---- end to end -----------------------------------------------------------------------------

def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


E2E = [
    ("ON=CCc1ccc(O)cc1", "1-hydroxy-4-[2-(hydroxyimino)ethyl]benzene"),
    ("CN(C)c1ccc(/N=N/c2ccccc2)cc1", "{(1E)-[4-(dimethylamino)phenyl]diazenyl}benzene"),
    ("Clc1ccc(CO/N=C(\\Cn2ccnc2)c2ccc(Cl)cc2Cl)c(Cl)c1",
     "2,4-dichloro-1-[({[(1Z)-1-(2,4-dichlorophenyl)-2-(1H-imidazol-1-yl)ethylidene]amino}oxy)methyl]benzene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", E2E)
def test_end_to_end_names_round_trip_and_carry_no_a_chain(smiles, name):
    row = _row(smiles, "best-effort")
    assert row["name"] == name
    assert not F.ach_hits(row["name"])
    assert name_is_rt_exact(row["name"], smiles)


def test_a_chain_the_book_allows_stays_an_a_chain():
    """ (the Blue Book): four or more heterounits in a chain with C ends are an
    'a' chain; the group prefixes do not replace it."""
    mol, frag = _group("c1ccccc1C=NNC(=O)N(C)N=CC")
    assert _terminal(mol, frag) == "5-methyl-4-oxo-2,3,5,6-tetraazaocta-1,6-dien-1-yl"
