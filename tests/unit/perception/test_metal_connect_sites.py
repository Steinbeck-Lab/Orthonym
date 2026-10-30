"""``perception.metals.connect_anion_to_organometal_cation_sites``: the bonds that join a free anion
to the organometallic cation it is drawn beside, and the drawings it refuses.

Blue Book: INTRODUCTION, "Organometallic compounds are compounds having at least one bond
between one metal atom and one carbon atom": the cation has one, so the drawing names an
organometallic compound; the anion is joined to the metal by a single bond to give the bonded
drawing that the organometallic producers read (``[Cl-].[Zn+]c1ccccc1`` gives ``Cl[Zn]c1ccccc1``).
The sites are single bonds from a metal admitted to ``CONNECT_ANION`` (``data/metal_scope.py``) to
the donor atom of an anion that is a fragment of its own; the metal's charge plus its number of
bonds must be one of its valences in the table, and the pairing must not be a choice.
"""
import pytest
from rdkit import Chem

from orthonym.perception.metals import connect_anion_to_organometal_cation_sites

pytestmark = pytest.mark.unit


def _sites(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return connect_anion_to_organometal_cation_sites(mol)


@pytest.mark.parametrize("smiles, expected", [
    ("[Cl-].[Zn+]c1ccccc1", [(1, 0)]),                    # an aryl cation and chloride
    ("CC[Zn+].[OH-]", [(2, 3)]),                          # an alkyl cation and hydroxide
    ("[Br-].[Zn+]Cc1ccccc1", [(1, 0)]),                   # a benzyl cation and bromide
    ("CC[Zn+].[Cl-]", [(2, 3)]),
    ("C[Zn+].[Cl-]", [(1, 2)]),
    ("CC(=O)[O-].C[Zn+]", [(5, 3)]),                      # the donor of an ambident anion is the charged atom
    ("CC[Zn+].[S-]CC", [(2, 3)]),
    ("C[Zn+].c1ccccc1[N-]c1ccccc1", [(1, 8)]),
    ("C[Zn+].[O-]C(C)=O.O", [(1, 2)]),                    # a neutral fragment beside them does not matter
])
def test_sites_of_a_drawing_that_the_connection_covers(smiles, expected):
    assert _sites(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "[CH2-]C.[Zn+]Br",                                    # no bond to carbon: a carbanion beside a halometal cation
    "[Zn+]Br",
    "C[Zn+]",                                             # no anion at all
    "C[Zn+].[Cl-].[Br-]",                                 # two anions for one charge: the pairing is a choice
    "C[Zn+].[Cl-].[Cl-]",
    "C[Zn+].[Cl-].[CH3-]",                                # an anion that is not the one: the pairing is a choice
    "C[Zn+].[CH3-]",                                      # a carbanion is no donor (the key of its bonded drawing is another)
    "CC[Zn+].[O-][N+](=O)[O-]",                           # two negatively charged atoms: which one takes the bond
    "C[Zn+2].[Cl-].[Cl-]",                                # +2 and one bond: +3, no valence of zinc
    "C[Zn+]Cl",                                           # +1 and two bonds: +3
    "[Zn+]1CCCC1.[Cl-]",
    "C[Cd+].[Cl-]",                                       # a metal that the table does not admit
    "C[Ni+].[Cl-]",                                       # (mercury is admitted since 2026-10-30)
    "C[Cu+].[Cl-]",
    "C[Mg+].[Br-]",                                       # Group 2: admitted to ION_SALTS only
    "C[Zn+].[Na+].[Cl-]",                                 # two cations
    "C[Zn+].[O-]C(=O)C[Zn+]",
    "CC[Zn]CC.[Cl-]",                                     # no positively charged metal
    "C=[Zn+].[Cl-]",                                      # a double bond from the metal
    "[O-]CC[Zn+]",                                        # the anion is not a fragment of its own
    "[Cl-].[Cl-].[Zn+2]",                                 # an ionic salt: M1's drawing, no bond to carbon
    "CCO",
])
def test_drawings_that_are_left_as_written(smiles):
    assert _sites(smiles) is None


def test_none_is_not_a_drawing():
    assert connect_anion_to_organometal_cation_sites(None) is None


def test_the_sites_name_atoms_of_the_molecule_whatever_the_order_of_the_fragments():
    """The pairs are indices of the molecule that was read: the same drawing in another fragment
    order gives the pair that names the same two atoms."""
    for smiles in ("[Cl-].[Zn+]c1ccccc1", "[Zn+]c1ccccc1.[Cl-]", "c1ccc([Zn+])cc1.[Cl-]"):
        mol = Chem.MolFromSmiles(smiles)
        (metal_idx, donor_idx), = connect_anion_to_organometal_cation_sites(mol)
        assert mol.GetAtomWithIdx(metal_idx).GetSymbol() == "Zn"
        assert mol.GetAtomWithIdx(donor_idx).GetSymbol() == "Cl"
        assert mol.GetAtomWithIdx(donor_idx).GetFormalCharge() == -1
