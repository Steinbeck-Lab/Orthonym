"""A ring assembly is one ring system in the seniority of the parent ring.

 (``the Blue Book Blue Book``): "Seniority criteria for ring assemblies are based on the
appropriate criteria for rings and ring systems given in ", so '[1,1'-biphenyl]' (2 rings, 12 skeletal atoms)
is compared with a fused ring system by " The senior ring or ring system has the greater number of skeletal
atoms" (:19469): it is senior to naphthalene (10) and to indane (9), junior to fluorene (3 rings,,:19461)
and to a heterocycle. With equal criteria the fused ring system is senior to the ring assembly, the
printed pair 'benzo[8]annulene (PIN) > 1,1'-biphenyl (PIN) [fused ring system (c) > ring assembly (g)]' (:19563,
 "Polycyclic systems",:19536). The parent of '2-([1,1'-biphenyl]-4-yl)naphthalene' is therefore the biphenyl
and its name is a valid systematic name and not the PIN; the PIN '4-(naphthalen-2-yl)-1,1'-biphenyl' is not built.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.rules.ring_selection import ring_assembly_outranks, select_principal_ring_system
from orthonym.perception.rings import get_ring_systems


def _outranks(smiles):
    mol = Chem.MolFromSmiles(smiles)
    chosen = select_principal_ring_system(mol, get_ring_systems(mol))
    return ring_assembly_outranks(mol, set(chosen))


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc2ccccc2c1",        # naphthalene: 2 rings, 10 atoms < biphenyl 2 rings, 12 atoms
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc2CCCc2c1",         # indane: 9 atoms
    "c1ccc(cc1)-c1ccccc1-c1ccc2ccccc2c1",
])
def test_the_assembly_that_is_senior_to_the_chosen_ring_system_is_seen(smiles):
    assert _outranks(smiles)


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc2c(c1)Cc1ccccc1-2",        # fluorene: 3 rings > 2
    "c1ccc(cc1)-c1ccc(cc1)-c1ccc2ccc3cccc4ccc1c2c34",      # pyrene
    "c1ccc(cc1)-c1ccc(cc1)-c1ccccn1",                      # pyridine: a heterocycle is senior
    "c1ccc2ccccc2c1-c1ccccc1",                             # phenyl is no assembly
    "c1ccc2ccccc2c1-c1ccc2ccccc2c1",                       # one assembly, nothing beside it
])
def test_a_ring_system_that_is_senior_to_the_assembly_is_not_outranked(smiles):
    assert not _outranks(smiles)


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc2ccccc2c1", "2-([1,1'-biphenyl]-4-yl)naphthalene"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1cccc2ccccc12", "1-([1,1'-biphenyl]-4-yl)naphthalene"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc2CCCc2c1", "5-([1,1'-biphenyl]-4-yl)-2,3-dihydro-1H-indene"),
])
def test_the_name_on_the_junior_parent_is_not_labelled_a_pin(smiles, name):
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "systematic_verified" and row["is_pin"] is False
    assert "P-44.2.2.2.7" in [f["rule"] for f in row["spelling_failures"]]


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc2c(c1)Cc1ccccc1-2", "2-([1,1'-biphenyl]-4-yl)-9H-fluorene"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc2ccc3cccc4ccc1c2c34", "1-([1,1'-biphenyl]-4-yl)pyrene"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccccn1", "2-([1,1'-biphenyl]-4-yl)pyridine"),
])
def test_the_name_on_the_senior_parent_stays_a_pin(smiles, name):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified"
