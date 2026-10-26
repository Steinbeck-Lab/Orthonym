"""PIN class program, Task 22: adducts (ADDUCT_ORDER, mixed organic - inorganic adducts).

 "Organic adducts" (the Blue Book), "For adducts composed solely of organic compounds, the
individual components are cited in the order of seniority of classes (see in formulas, no longer
according to the number of species in the adduct, nor in accordance with the alphanumerical order"
(:3769). The class of a component is that of its principal characteristic group; two components of one
class are ranked as the parent structures that carry the group are: more principal groups first
,:18850), then the senior ring system,:19408: a heterocycle, one with nitrogen, more
rings, more skeletal atoms...): 'benzene-1,2,3-triol—quinolin-8-ol (1/2)' (three -ol groups against one)
and '1H-indole-2-carboxylic acid—3,5-dinitrobenzoic acid (1/1)' (the indole is a nitrogen heterocycle).
The Blue Book's own pairs hold: 'coronene—1,3,5-trinitrobenzene (1/1) (PIN)' and '9H-fluorene—1-methyl-
naphthalene (1/1) (PIN)' (more rings), 'ethanol—pyridine (1/1) (PIN)' (an alcohol before a ring with no
group).

 "Mixed organic - inorganic adducts" (:4663): "Although organic components are preferred IUPAC
names, preferred IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names have not yet
been determined for inorganic components." The name of such an adduct is labelled below the PIN, so the
PIN tier declines it and the default tier keeps it (a name that reads back to the input). "Retained names"
(:7952) settles water: the name 'water' "is used in these recommendations but its use as a preferred IUPAC name is
deferred...; thus, no PIN label will be assigned in names including them (see ": a hydrate has no PIN label either.

Not built here (internal notes): the order of two components that have no principal
group beyond the size rule ('1H-pyrrole' before '2H-1-benzopyran',,:19449, which is no adduct
of the book; 'benzene—pyridine' is a project decision that other tests pin).
"""
import pytest

from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # (:3769),,
    ("Oc1cccc(O)c1O.Oc1cccc2cccnc12.Oc1cccc2cccnc12", "benzene-1,2,3-triol—quinolin-8-ol (1/2)"),
    ("O=C(O)c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1.O=C(O)c1cc2ccccc2[nH]1",
     "1H-indole-2-carboxylic acid—3,5-dinitrobenzoic acid (1/1)"),
]

# The Blue Book pairs and the project's rows that the new key has to keep.
CONTROL_ROWS = [
    ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61.O=[N+]([O-])c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1",
     "coronene—1,3,5-trinitrobenzene (1/1)"),
    ("C1c2ccccc2-c2ccccc12.Cc1cccc2ccccc12", "9H-fluorene—1-methylnaphthalene (1/1)"),
    ("CCO.c1ccncc1", "ethanol—pyridine (1/1)"),
    ("c1ccccc1.c1ccncc1", "benzene—pyridine (1/1)"),
]

#: an organic component with an inorganic one (not water) has no PIN.
MIXED_ROWS = [
    "C(CCC(=N)N)C[C@@H](C(=O)O)N.Cl",      # (2S)-2,7-diamino-7-iminoheptanoic acid—hydrogen chloride (1/1)
    "CC(CCCC(C)(C)O)N1CCOCC1.Cl",          # 2-methyl-6-(morpholin-4-yl)heptan-2-ol—hydrogen chloride (1/1)
    "CN1CCCC1c1cccnc1.Cl",                 # nicotine—hydrogen chloride, the pattern of:4675
    # "Retained names" (:7952): "no PIN label will be assigned in names including" 'water' (see
    "O.OC(=O)C(=O)O",                      # oxalic acid—water (1/1)
    "OC(=O)C(=O)O.NCCN.O.O.OC(=O)C(=O)O.NCCN.O",   # oxalic acid—ethane-1,2-diamine—water (2/2/3)
    "CCO.O",
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + CONTROL_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", MIXED_ROWS)
def test_mixed_adduct_is_declined_at_the_pin_tier(smiles):
    assert_declined_at_default(smiles)


def test_mixed_adduct_keeps_its_name_at_the_default_tier():
    from tests.support.rt_assert import name_best_effort
    row = name_best_effort(MIXED_ROWS[1])
    assert row["name"] == "2-methyl-6-(morpholin-4-yl)heptan-2-ol—hydrogen chloride (1/1)", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert any(f.get("rule") == "P-14.8.2" for f in row.get("spelling_failures") or ()), row
