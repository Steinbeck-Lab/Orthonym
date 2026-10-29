"""Radical centres away from the parent: the best-effort floor gives them a name that reads back.

A molecule with a radical centre is named by the best-effort floor only when the name carries every centre and
OPSIN (``-r``) reads it back to the same graph, radical placement included
(``validation.radical_identity``). Three rules decide how a centre is carried.

* A divalent or trivalent centre is cited by the lambda convention, "The λ-convention"
  (the Blue Book, sentence:40562): "Divalent and trivalent radical centers in a parent hydride formally
  derived by the removal of two or three hydrogen atoms from the same skeletal atom in its standard valence state
  may be described by the λ-convention (see. Locants for the radical centers are followed by the symbol
  λ^n, where 'n' is the bonding number of the skeletal atom. This method is only for general nomenclature."
  So the names below are labelled ``systematic_verified`` and never ``pin_verified``.
* A monovalent centre has no lambda (the same sentence names two and three removed hydrogens only): it is the free
  valence of the parent. "CHOICE OF PARENT RADICAL" (the Blue Book), criterion (a): the "Parent with
  the maximum number of radical centers of any kind in a single parent structure"; "Multiple radical
  centers (polyradicals)" (:40524): the suffix 'yl' for each monovalent centre with the numerical prefixes
  ('propane-1,2,3-triyl').
* A monovalent centre that no one parent carries has no name here (the book's 'ylo',, the Blue Book, is
  not read by OPSIN), and the molecule keeps its abstention: never 'λ3-silyl' for a monovalent centre. Centres in
  separate identical parts of a molecule are the exception, ``assembly/radical_assembly.py`` "ASSEMBLIES OF
  PARENT RADICALS", the Blue Book): two dimethylsilyl radicals on one benzene ring are '(benzene-1,2-diyl)bis(dimethylsilyl)',
  which this test used to list as declined (the ring atoms between the two centres are not a chain,
  and puts the centres on ONE parent; covers radical centres "located in different parts of the structure").
"""
import pytest
from rdkit import Chem

from orthonym.rules.lambda_convention import radical_lambda_bonding_number
from orthonym.validation.radical_identity import radical_identity_verdict
from tests.support.rt_assert import _independent_parse, name_best_effort, name_is_rt_exact

# the radical offer (``assembly/radical_offer.py``) makes no name that OPSIN does not read back, so the OPSIN validity
# gate is on in every test of this module
pytestmark = pytest.mark.opsin_gate


def _assert_named(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
    parsed = _independent_parse(res["name"], smiles)
    assert parsed is not None and radical_identity_verdict(smiles, parsed) == "ok", (res, parsed)


# --- divalent and trivalent centres: the lambda convention -------------------------------------------------------

LAMBDA_ROWS = [
    # one-atom roots, the bonding number after the atom's own locant-free prefix
    ("[B]c1ccccc1[B]", "1,2-di(λ1-boranyl)benzene"),
    ("[B]CC1=C(C=CC(=C1)OC)[B]", "1-(λ1-boranyl)-2-[(λ1-boranyl)methyl]-4-methoxybenzene"),
    # a silicon centre with a ligand roots 'methyl-λ2-silyl'
    ("C[Si]CC[B]", "1-(λ1-boranyl)-2-(methyl-λ2-silyl)ethane"),
    # a carbon end that is a radical centre is the group 'λ2-methyl' (dichloro-λ2-methane, the Blue Book), not an
    # atom of the chain, which cites no lambda on a carbon atom
    ("[CH]P([CH])CP([CH])CP([CH])CP([CH])[CH]", "1,1,3,5,7,7-hexa(λ2-methyl)-1,3,5,7-tetraphosphaheptane"),
    ("C=C([CH])COCC(=O)OC([CH])([CH])[CH]", "2-(λ2-methyl)-3-{2-oxo-2-[tri(λ2-methyl)methoxy]ethoxy}prop-1-ene"),
    # a skeletal centre: the symbol follows the locant of the atom, the Blue Book)
    ("C1CC[Si]CC1C[B]", "3-[(λ1-boranyl)methyl]-1λ2-silacyclohexane"),
    # one monovalent centre (the free valence: the parent hydride radical 'silyl',, the Blue Book)
    # and one trivalent one (lambda) in one name
    ("CC[Si](C)CC[Si]", "ethyl(methyl)[2-(λ1-silyl)ethyl]silyl"),
    # the monovalent centre is an amino nitrogen: its group closes the free valence, the boron is cited inside it
    ("[B]CC[NH]", "[2-(λ1-boranyl)ethyl]amino"),
]


@pytest.mark.parametrize("smiles,expected", LAMBDA_ROWS)
def test_divalent_and_trivalent_centres_are_cited_by_the_lambda_convention(smiles, expected):
    _assert_named(smiles, expected)


# --- several monovalent centres: the radical of the parent that carries them all ---------------------------------

POLYRADICAL_ROWS = [
    ("CCCC[C]1[CH][CH][CH][CH]1", "1-butylcyclopentane-1,2,3,4,5-pentayl"),
    ("C[Si](C)C[Si](C)C", "1,1,3,3-tetramethyl-1,3-disilapropane-1,3-diyl"),
    # the stereo of the input is composed on the name (the read-back of a name with free valences takes -r)
    ("C[C@@H]1[CH][CH][C@H]([CH]1)O", "(3R,5R)-3-hydroxy-5-methylcyclopentane-1,2,4-triyl"),
]


@pytest.mark.parametrize("smiles,expected", POLYRADICAL_ROWS)
def test_monovalent_centres_are_the_free_valences_of_one_parent(smiles, expected):
    _assert_named(smiles, expected)


# --- a candidate that does not read back does not end the search ---------------------------------------------------

def test_an_earlier_candidate_that_loses_the_radical_does_not_end_the_search():
    """The general engine's ring parent names the indene of this acylselanyl radical with an 'acetylselanylidene'
    group that holds no radical, and the caller's read-back refused it; that candidate, the first of the rungs,
    ended the search before the radical rung ran, and the molecule had no name (``t4_coverage._radical_alternative``)."""
    smiles = "C1=CC=C2C(=C1)C=CC2=CC(=O)[Se]"
    res = name_best_effort(smiles)
    assert res["name"] and res["name"].endswith("selanyl") and "inden-1-ylidene" in res["name"], res
    assert res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res


# --- what stays declined ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    # two monovalent centres in two ring systems (two cyclopentane-pentayl units): an assembly of multi-centre units is not
    # offered, because OPSIN reads '(<linker>)bis(cyclopentane-1,2,3,4,5-pentayl)' with the radical of the atom bonded to
    # the linker lost
    "CN(C)C[C]1[CH][CH][CH][C]1CN(CCCCCCO)C[C]2[CH][CH][CH][C]2CN(C)C",
])
def test_a_monovalent_centre_no_parent_carries_is_not_named_with_a_lambda(smiles):
    res = name_best_effort(smiles)
    assert res["name"] is None or "λ3" not in res["name"], res
    assert res["name"] is None, res


# --- the table of the lambda bonding numbers ---------------------------------------------------------------------

@pytest.mark.parametrize("smiles,atom,expected", [
    ("[B]C", 0, 1),            # boron, one bond, two electrons: λ1 (the Blue Book 'fluoro-λ1-methane')
    ("C[Si]C", 1, 2),          # silicon, two bonds, two electrons
    ("[Si]C", 0, 1),           # silicon, one bond, three electrons
    ("Cl[C]Cl", 1, 2),         # 'dichloro-λ2-methane'
    ("c1ccccc1[N]", 6, 1),     # 'phenyl-λ1-azane'
    ("C[Si](C)C", 1, None),    # monovalent: no lambda licenses two and three removed hydrogens)
    ("[CH3]", 0, None),
    ("[B-](C)(C)C", 0, None),  # no radical
])
def test_radical_lambda_bonding_number(smiles, atom, expected):
    assert radical_lambda_bonding_number(Chem.MolFromSmiles(smiles), atom) == expected


def test_a_charged_radical_centre_has_no_lambda():
    mol = Chem.MolFromSmiles("[B-][P+](C)(C)C")
    assert mol.GetAtomWithIdx(0).GetNumRadicalElectrons() == 3
    assert radical_lambda_bonding_number(mol, 0) is None
