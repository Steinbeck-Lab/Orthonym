"""``perception.metals.single_bond_metal_heteroatom_sites``: the bonds that a charge-conserving
heterolytic disconnection of a metal drawing may break, and the drawings it refuses.

Blue Book: INTRODUCTION, "Organometallic compounds are compounds having at least one bond
between one metal atom and one carbon atom": a drawing with a metal-carbon bond is not rewritten
(its standard InChIKey is not the key of its ionic drawing). The sites are single bonds from a
metal admitted to ``DISCONNECT`` (``data/metal_scope.py``) to an atom that is neither carbon nor a
metal; the metal's charge after the disconnection must be one of its valences in the table.
"""
import pytest
from rdkit import Chem

from orthonym.perception.metals import single_bond_metal_heteroatom_sites

pytestmark = pytest.mark.unit


def _sites(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return single_bond_metal_heteroatom_sites(mol)


@pytest.mark.parametrize("smiles, expected", [
    ("Cl[Zn]Cl", [(1, 0), (1, 2)]),
    ("O[Zn]O", [(1, 0), (1, 2)]),
    ("Cl[Zn+]", [(1, 0)]),                                              # +1 and one bond: +2
    ("Cl[Zn-2](Cl)(Cl)Cl", [(1, 0), (1, 2), (1, 3), (1, 4)]),           # -2 and four bonds: +2
    ("[Na+].[Na+].Cl[Zn-2](Cl)(Cl)Cl", [(3, 2), (3, 4), (3, 5), (3, 6)]),   # a counter-ion with no bond does not matter
    ("[NH3+][Zn-2]([NH3+])([NH3+])[NH3+]", [(1, 0), (1, 2), (1, 3), (1, 4)]),
    ("CC(=O)O[Zn]OC(C)=O", [(4, 3), (4, 5)]),
    ("CC(=O)O[Cu]OC(C)=O", [(4, 3), (4, 5)]),                           # copper: admitted since 2026-10-30
    ("Cl[Zn]O[Cu]Cl", [(1, 0), (1, 2), (3, 2), (3, 4)]),                # two bonded metals, both admitted
])
def test_sites_of_a_drawing_that_the_disconnection_covers(smiles, expected):
    assert _sites(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "CC[Zn]CC",                                   # a metal-carbon bond
    "CC(=O)O[Zn]CC",                              # one of them is enough
    "N#C[Zn]C#N",                                 # the cyanide carbon
    "O=[Zn]",                                     # a double bond from the metal
    "S=[Zn]",
    "N->[Zn](Cl)Cl<-N",                           # a dative bond
    "Cl[Zn](Cl)N",                                # three bonds: [Zn+3] is no valence of zinc
    "Cl[Zn]",                                     # one bond: [Zn+1] is no valence of zinc
    "CC(=O)O[Tc]OC(C)=O",                         # technetium is not admitted to DISCONNECT
    "Cl[Zn]O[Th]Cl",                              # one bonded metal that is not admitted
    "CC(=O)O[Sn]OC(C)=O",                         # tin: PARENT_HYDRIDE_IONS and CONNECT_ANION only, no DISCONNECT
    "[Zn]1[Zn]Cl1",                               # a bond between two metals
    "C[Si](C)(C)[Zn]Cl",                          # a bond to a metalloid
    "[Cl-].[Cl-].[Zn+2]",                         # no bond at all
    "[Zn]",
    "CCO",
    "C/C=N/[Zn]Cl",                               # a double bond at the donor carries a configuration
])
def test_drawings_that_are_left_as_written(smiles):
    assert _sites(smiles) is None


def test_none_is_not_a_drawing():
    assert single_bond_metal_heteroatom_sites(None) is None
