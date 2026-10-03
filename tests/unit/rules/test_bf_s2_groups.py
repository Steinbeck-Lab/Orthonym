"""The parent choice against other ring systems:19408-:19418), the suffixes
slice S2 adds:34720 'carbonitrile',:34941 'carbaldehyde',
 :32669 'carboxamide') and the 'nitro' prefix:25935). Every
name was read back by OPSIN 2.9.0 to the input's full InChIKey (S2 planning notes)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build
from orthonym.rules.bridged_fused_pin.selection import p44_2_1_key, ring_systems


def _name(smiles):
    res = build(Chem.MolFromSmiles(smiles))
    return res[0] if res else None


@pytest.mark.parametrize("smiles,name", [
    # (a) the bridged heterocycle is senior to a benzene ring
    ("c1ccc(cc1)-n1ncc2c1C1CCC2C1", "1-phenyl-4,5,6,7-tetrahydro-1H-4,7-methanoindazole"),
    # (d) three rings are senior to two
    ("c1ccc2cc(ccc2c1)C1CC2CC1c1ccccc12",
     "2-(naphthalen-2-yl)-1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
    ("C1CCC(CC1)C1C=CC2C3C=CC(C3)C12", "1-cyclohexyl-3a,4,7,7a-tetrahydro-1H-4,7-methanoindene"),
    # (d) the bridged heterocycle (3 rings) is senior to a pyridine ring (the S1 check
    # declined any other heterocyclic ring system)
    ("c1ccc(nc1)-n1ncc2c1C1CCC2C1",
     "1-(pyridin-2-yl)-4,5,6,7-tetrahydro-1H-4,7-methanoindazole"),
])
def test_the_senior_bridged_system_is_the_parent(smiles, name):
    assert _name(smiles) == name


def test_a_senior_heterocycle_elsewhere_declines():
    # (a): pyridine is senior to the carbocyclic bridged system; the composer chooses
    assert _name("c1cnccc1C1CC2CC1c1ccccc12") is None


def test_p44_2_1_key_order():
    mol = Chem.MolFromSmiles("c1cnccc1C1CC2CC1c1ccccc12")
    pyridine, bridged = sorted(ring_systems(mol), key=len)
    assert p44_2_1_key(mol, pyridine) < p44_2_1_key(mol, bridged)


def test_a_ring_system_tied_on_p44_2_1_is_not_junior():
    # (a)-(g) cannot choose between two identical bridged systems, so neither is
    # the parent by that rule: the check declines a tie, as it declines a senior system
    from orthonym.rules.bridged_fused_pin import _other_ring_systems_are_junior
    twin = Chem.MolFromSmiles("c1cc2c(cc1Cc1ccc3c(c1)C1CCC3C1)C1CCC2C1")
    first, second = ring_systems(twin)
    assert p44_2_1_key(twin, first) == p44_2_1_key(twin, second)
    assert not _other_ring_systems_are_junior(twin, first)
    lone = Chem.MolFromSmiles("c1ccc(Cc2ccc3c(c2)C2CCC3C2)cc1")      # beside a benzene ring
    assert _other_ring_systems_are_junior(lone, ring_systems(lone)[0])


@pytest.mark.parametrize("smiles,name", [
    ("N#Cc1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carbonitrile"),
    ("N#Cc1cc(C#N)c2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-5,7-dicarbonitrile"),
    ("O=Cc1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carbaldehyde"),
    ("O=CC1CC2CC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-2-carbaldehyde"),
    ("O=CC1C=CC2C3C=CC(C3)C12", "3a,4,7,7a-tetrahydro-1H-4,7-methanoindene-1-carbaldehyde"),
    ("NC(=O)c1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carboxamide"),
    ("NC(=O)C1CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethanoanthracene-11-carboxamide"),
    ("[O-][N+](=O)c1ccc2c(c1)C1CCC2C1", "6-nitro-1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
])
def test_suffixes_and_prefixes(smiles, name):
    assert _name(smiles) == name


@pytest.mark.parametrize("smiles", [
    "CNC(=O)c1ccc2c(c1)C1CCC2C1",     # N-substituted amide: N-locant prefixes not spelled
    "CC(=O)c1ccc2c(c1)C1CCC2C1",      # ketone on the chain: the chain is the parent
    "NS(=O)(=O)c1ccc2c(c1)C1CCC2C1",  # sulfonamide: not spelled
])
def test_other_principal_groups_decline(smiles):
    assert _name(smiles) is None


def _nitro(smiles):
    from orthonym.rules.polycyclics import _nitro_group_atoms
    mol = Chem.MolFromSmiles(smiles)
    n = next(a for a in mol.GetAtoms() if a.GetSymbol() == "N")
    ring = next(nb for nb in n.GetNeighbors() if nb.IsInRing())
    return _nitro_group_atoms(mol, n.GetIdx(), ring.GetIdx())


@pytest.mark.parametrize("smiles", [
    "[O-][N+](=O)c1ccc2c(c1)C1CCC2C1",
    "O=[N+]([O-])c1ccc2c(c1)C1CCC2C1",
])
def test_a_nitro_group_is_read_in_either_atom_order(smiles):
    assert _nitro(smiles) is not None and len(_nitro(smiles)) == 3


@pytest.mark.parametrize("smiles", [
    "ON(O)c1ccc2c(c1)C1CCC2C1",             # -N(OH)2: the same N, O, O atoms, another group
    "[O]N([O])c1ccc2c(c1)C1CCC2C1",         # -N(O*)2
    "[O-][N+](=[18O])c1ccc2c(c1)C1CCC2C1",  # 'nitro' carries no isotope descriptor
    "[O-][15N+](=O)c1ccc2c(c1)C1CCC2C1",
])
def test_other_n_o_o_groups_are_not_read_as_nitro(smiles):
    assert _nitro(smiles) is None


def test_a_nitro_prefix_beside_a_suffix():
    assert _name("OC(=O)c1cc2C3CCC(C3)c2cc1[N+](=O)[O-]") == (
        "7-nitro-1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carboxylic acid")


@pytest.mark.parametrize("smiles", [
    "OC(=O)c1cc2C3CCC(C3)c2cc1N(O)O",
    "OC(=O)c1cc2C3CCC(C3)c2cc1[N+](=[18O])[O-]",
])
def test_a_group_nitro_would_misname_gets_no_nitro_name(smiles):
    # read as nitro, these gave '7-nitro-...-6-carboxylic acid', another molecule
    from tests.support.rt_assert import name_is_rt_exact
    name = _name(smiles)
    assert name is None or name_is_rt_exact(name, smiles), name
