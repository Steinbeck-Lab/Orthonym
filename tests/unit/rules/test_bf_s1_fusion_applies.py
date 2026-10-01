"""When a fusion (or bridged fusion) name exists for a ring system, so that its von Baeyer
name is not the PIN (``rules/polycyclic.fusion_nomenclature_applies``).

 (the Blue Book): a bridged fused ring system is one "in which some of the
rings constitute a fused ring system... and the remaining rings are created by one or more
bridges"; (:14025): "An atom or group of atoms is named as a bridge". A ring
system whose ortho-fused rings of five or more members already hold every ring atom, while
it has more rings than they do, closes its other rings by bonds only: no bridge atom, so
no bridged fused name, and it is not a fused ring system either. (:23710):
"When fusion names are not allowed, unsaturated von Baeyer ring system names are
preferred IUPAC names". An exhaustive check over every atom subset of these four dev rows
finds no fused residual with two rings of five or more members (the bridged fused S1
TRIAGE section)."""
import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import fusion_nomenclature_applies


def _applies(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    return fusion_nomenclature_applies(mol, ring_atoms)


@pytest.mark.parametrize("smiles", [
    # tricyclo[4.4.0.0^5,10]decane / [4.4.0.0^2,7]: a decalin whose fourth ring (a
    # cyclobutane) is closed by a bond (1 milestone1500 row, 3 dev2000 rows)
    "CC1=CC[C@H]2[C@H]3C1[C@@]2(C)CC[C@H]3C(C)C",
    "CC(C)[C@@H]1C[C@H](O)[C@@]2(CO)[C@@H]3CC[C@](C)(O)[C@H]2[C@@H]31",
    "C/C(CO)=C1/CCC2(C)C3CCC(C)(O)C2C13",
    "C=C1CCC2C3C(C(C)(C)O)CCC2(C)C13",
])
def test_a_ring_closed_by_a_bond_only_is_not_a_bridge(smiles):
    assert _applies(smiles) is False


@pytest.mark.parametrize("smiles", [
    "C1CCC2CCCCC2C1",                 # decahydronaphthalene (PIN), BB:24233
    "C1CC2CCC3CCCC4CCC(C1)C2C34",     # perhydropyrene: ortho- and peri-fused
    "C1CCC2CC3CCCCC3CC2C1",           # perhydroanthracene
    "C1CC23CCCCC2(CC1)CC3",           # octahydro-4a,8a-ethanonaphthalene (bridge on a fusion bond)
    "C1=CC23C=CC=CC2(C=C1)CC3",       # 4a,8a-ethanonaphthalene (PIN), BB:14249
    "C1CC2CC1c1ccccc12",              # 1,2,3,4-tetrahydro-1,4-methanonaphthalene
    "C1CC2C3CCC(C3)C2C1",             # octahydro-1H-4,7-methanoindene skeleton, BB:49311
])
def test_fused_and_bridged_fused_systems_still_apply(smiles):
    assert _applies(smiles) is True


@pytest.mark.parametrize("name,flagged", [
    # the same rule where the label reads a shipped name's von Baeyer descriptor
    ("tricyclo[4.4.0.0^2,7]decane", False),
    ("2-[6-methyl-8-methylidenetricyclo[4.4.0.0^2,7]decan-3-yl]propan-2-ol", False),
    # (:9685): "The superscript locants for the secondary bridges must be as
    # low as possible"; tricyclo[4.4.0.0^5,10] is the same skeleton as [4.4.0.0^2,7], so a
    # name that cites 5,10 is not in its PIN form (the engine cites it for three of the
    # four dev rows; they stay below the PIN label)
    ("tricyclo[4.4.0.0^5,10]decane", True),
    ("2,6-dimethyl-9-(propan-2-yl)tricyclo[4.4.0.0^5,10]dec-2-ene", True),
    ("tricyclo[5.2.1.0^2,6]decane", True),            # 4,7-methanoindene, BB:49311
    ("tricyclo[12.3.1.0^5,10]octadecane", True),      # 8,12-methanobenzo[13]annulene, BB:23875
    ("tricyclo[6.2.1.0^2,7]undecane", True),          # decahydro-1,4-methanonaphthalene
    ("bicyclo[4.4.0]decane", True),                   # decahydronaphthalene, BB:24233
    ("bicyclo[2.2.1]heptane", False),
])
def test_the_name_vocabulary_check_reads_the_descriptor_the_same_way(name, flagged):
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert (non_pin_vocabulary(name) is not None) is flagged
