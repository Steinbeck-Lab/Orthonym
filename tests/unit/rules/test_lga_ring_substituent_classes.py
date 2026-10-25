"""LGA: ring substituents of a ring parent that the PIN path declined.

Four causes, each a class (every row below is a molecule of the class, none a special case):

1. A ring substituent of more than ten atoms on a benzene parent was never offered to the substituent
   namer (``rules/benzene.py`` ``_MAX_FALLBACK_ATOMS``): '4-(9H-fluoren-2-yl)benzoic acid', pyrenyl,
   carbazol-3-yl, dibenzofuranyl.
2. The free valence of an anthracene or phenanthrene substituent was numbered by a heuristic that
   numbers naphthalene only (``polycyclics._map_pah_atoms_to_iupac``): every locant was wrong and the
   round trip refused the name. "General methodology for naming substituent groups"
   (the Blue Book): the free valence takes the lowest locant the fixed numbering allows
    gives anthracene and phenanthrene their numbering).
3. A ring prefix with decorations took the multiplier 'di' from its spelling when the stem was one the
   spelling reader does not know ('9,9-dioctyl-9H-fluoren-2-yl', '3,6-di-tert-butyl-9H-carbazol-9-yl'):
    "General methodology" (c) (the Blue Book): "any component which is substituted
   automatically requires use of the multiplicative forms 'bis', 'tris', etc."; the writers now record
   it (``assembly.prefix_derivation``).
4. A spiro system as a ring-yl prefix: 'spirobi' (:10146) with the suffix 'yl'
   ; ``9,9'-spirobi[fluoren]-3-yl``.

A fifth: a molecule with no principal characteristic group whose fragment skeleton repeats (two
carbazoles) was refused by ``strict_ring_prefix.skeleton_twin`` although a ring system outside the
fragment is strictly senior: 1,10-phenanthroline, 14 skeletal atoms, against carbazole, 13).
"""
import pytest
from rdkit import RDLogger

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.tiers import emit_tier_flags
from tests.support.rt_assert import assert_full_rt

RDLogger.DisableLog("rdApp.*")

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    with jvm_slots(1, purpose="lga-test"):
        yield Orthonym(style="pin", **emit_tier_flags("pin"))   # emit_tier="pin": the PIN-only tier


BIG_RING_ON_BENZENE = [
    ("fluoren-2-yl", "C1c2ccccc2-c2ccc(cc12)-c1ccc(cc1)C(O)=O", "4-(9H-fluoren-2-yl)benzoic acid"),
    ("9,9-dioctylfluoren-2-yl", "CCCCCCCCC1(CCCCCCCC)c2ccccc2-c2ccc(cc12)-c1ccc(cc1)C(O)=O",
     "4-(9,9-dioctyl-9H-fluoren-2-yl)benzoic acid"),
    ("carbazol-3-yl", "OC(=O)c1ccc(cc1)-c1ccc2[nH]c3ccccc3c2c1", "4-(9H-carbazol-3-yl)benzoic acid"),
    ("dibenzofuran-2-yl", "OC(=O)c1ccc(cc1)-c1ccc2oc3ccccc3c2c1", "4-(dibenzo[b,d]furan-2-yl)benzoic acid"),
    ("pyren-1-yl", "OC(=O)c1ccc(cc1)-c1ccc2ccc3cccc4ccc1c2c34", "4-(pyren-1-yl)benzoic acid"),
]


@pytest.mark.parametrize("_id,smiles,pin", BIG_RING_ON_BENZENE, ids=[r[0] for r in BIG_RING_ON_BENZENE])
def test_a_ring_substituent_of_more_than_ten_atoms_is_named_on_a_benzene_parent(namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


PAH_YL = [
    ("anthracen-2-yl", "OC(=O)c1ccc(cc1)-c1ccc2cc3ccccc3cc2c1", "4-(anthracen-2-yl)benzoic acid"),
    ("anthracen-1-yl", "OC(=O)c1ccc(cc1)-c1cccc2cc3ccccc3cc12", "4-(anthracen-1-yl)benzoic acid"),
    ("anthracen-9-yl", "OC(=O)c1ccc(cc1)-c1c2ccccc2cc2ccccc12", "4-(anthracen-9-yl)benzoic acid"),
    ("phenanthren-2-yl", "OC(=O)c1ccc(cc1)-c1ccc2c(c1)ccc1ccccc12", "4-(phenanthren-2-yl)benzoic acid"),
    ("phenanthren-3-yl", "OC(=O)c1ccc(cc1)-c1ccc2ccc3ccccc3c2c1", "4-(phenanthren-3-yl)benzoic acid"),
    ("phenanthren-9-yl", "OC(=O)c1ccc(cc1)-c1cc2ccccc2c2ccccc12", "4-(phenanthren-9-yl)benzoic acid"),
    ("naphthalen-2-yl", "OC(=O)c1ccc(cc1)-c1ccc2ccccc2c1", "4-(naphthalen-2-yl)benzoic acid"),
    ("anthracen-9-yl on quinoline", "c1ccc2nc(ccc2c1)-c1c2ccccc2cc2ccccc12", "2-(anthracen-9-yl)quinoline"),
]


@pytest.mark.parametrize("_id,smiles,pin", PAH_YL, ids=[r[0] for r in PAH_YL])
def test_the_free_valence_of_a_retained_hydrocarbon_takes_its_own_locant(namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


MULTIPLIER = [
    ("bis(dioctylfluorenyl)", "CC1(C)c2ccccc2-c2ccc(cc12)C(O)(c1ccccc1)c1ccc2c(c1)C(C)(C)c1ccccc12",
     "bis(9,9-dimethyl-9H-fluoren-2-yl)(phenyl)methanol"),
    ("bis(di-tert-butylcarbazolyl)", "CC(C)(C)c1ccc2c(c1)c1cc(C(C)(C)C)ccc1n2-c1ccnc2c1ccc1c(-n3c4ccc(C(C)(C)C)cc4"
     "c4cc(C(C)(C)C)ccc43)ccnc12", "4,7-bis(3,6-di-tert-butyl-9H-carbazol-9-yl)-1,10-phenanthroline"),
]


@pytest.mark.parametrize("_id,smiles,pin", MULTIPLIER, ids=[r[0] for r in MULTIPLIER])
def test_a_decorated_ring_prefix_is_multiplied_by_bis(namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_an_undecorated_ring_prefix_keeps_the_simple_multiplier(namer):
    # (a) (the Blue Book): a simple prefix with a locant is enclosed and multiplied by 'di'
    smiles = "c1cnc2c(c1-n1c3ccccc3c3ccccc13)ccc1c(ccnc12)-n1c2ccccc2c2ccccc12"
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == ("4,7-di(9H-carbazol-9-yl)-1,10-phenanthroline", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


SPIROBI_YL = [
    ("on carbazole", "c1ccc2c(c1)c1ccccc1n2-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "9-(9,9'-spirobi[fluoren]-3-yl)-9H-carbazole"),
    ("on pyridine", "c1ccc(nc1)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "2-(9,9'-spirobi[fluoren]-3-yl)pyridine"),
    ("on benzoic acid", "OC(=O)c1ccc(cc1)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "4-(9,9'-spirobi[fluoren]-3-yl)benzoic acid"),
    ("on a triazine", "c1ccc(cc1)-c1nc(nc(n1)-c1ccccc1)-c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12",
     "2,4-diphenyl-6-(9,9'-spirobi[fluoren]-3-yl)-1,3,5-triazine"),
]


@pytest.mark.parametrize("_id,smiles,pin", SPIROBI_YL, ids=[r[0] for r in SPIROBI_YL])
def test_a_spirobi_system_is_a_ring_yl_prefix_of_the_senior_ring(namer, _id, smiles, pin):
    row = namer.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_the_spirobi_prefix_of_one_structure_is_written_once():
    # an oligomer of spirobi[fluorene] units asks for the same few ring systems dozens of times and each
    # was assembled and numbered from scratch (row 9 of the user's molecules: pin tier 0.75 s to 5.8 s)
    from rdkit import Chem
    from orthonym.rules import spirobi_substituent as SB
    sub = Chem.MolFromSmiles("c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccccc12")
    atoms = [a.GetIdx() for a in sub.GetAtoms() if a.GetIsAromatic() and a.GetDegree() == 2]
    calls = []
    real = SB._write
    SB._WRITTEN.clear()
    SB._write = lambda s, a: calls.append(a) or real(s, a)
    try:
        names = [SB.spirobi_substituent_name(sub, a) for a in atoms for _ in range(3)]
    finally:
        SB._write = real
    assert len(calls) <= len(atoms) and len(set(names)) <= 4 and None not in names, (calls, set(names))
    assert "9,9'-spirobi[fluoren]-2-yl" in names


@pytest.fixture(scope="module")
def best_effort_namer():
    with jvm_slots(1, purpose="lga-test"):
        yield Orthonym(style="pin", **emit_tier_flags("best-effort"))


def test_identical_polycyclic_components_of_a_spiro_system_are_spirobi_in_every_prefix(best_effort_namer):
    # (the Blue Book) "Monospiro ring systems consisting of two identical polycyclic ring
    # components are named by placing the nondetachable prefix 'spirobi' before the name of the component
    # ring system enclosed in square brackets": the general engine spelled the parent
    # 'spiro[fluorene-9,9'-fluorene]' and the prefix 'spiro[fluorene-9,9'-fluorene]-2'-yl' (a valid name,
    # not the one the book builds); the prefix of the pin tier is '9,9'-spirobi[fluoren]-2-yl'
    smiles = "c1ccc2c(c1)-c1ccccc1C21c2ccccc2-c2ccc(cc12)-c1ccc2c(c1)C1(c3ccccc3-2)c2ccccc2-c2ccccc12"
    row = best_effort_namer.name_tiered(smiles)
    assert "spiro[fluorene-9,9'-fluorene]" not in row["name"], row
    assert row["name"].count("9,9'-spirobi[") == 2 and "9,9'-spirobi[fluoren]-2-yl" in row["name"], row
    assert not row["is_pin"] and row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)
