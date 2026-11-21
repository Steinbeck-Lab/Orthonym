"""An unsaturated lactone ring kept as the parent beside a senior fused ring ketone is not the PIN.

 (the Blue Book): "There is no seniority order difference between ketones and pseudoketones...
between rings and ring systems, are considered". (d) (:19415): the system with more rings is senior, so a
1-benzopyran-4-one outranks a 2H-pyran-2-one ring. The heterocycle producer names such a molecule on the lactone
ring; rules.lactones.lactone_parent_checked labels that name below the PIN (best-effort keeps it), and the label
stays on the name when the input's stereodescriptors are placed on it (metrics.provenance.record_derived_non_pin_label).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C1OC(CC=C1)C1CC(=O)c2ccccc2O1", "6-(4-oxo-3,4-dihydro-2H-1-benzopyran-2-yl)-5,6-dihydro-2H-pyran-2-one"),
    ("CC1=C(C)C(=O)OC(C1)C1CC(=O)c2ccccc2O1",
     "3,4-dimethyl-6-(4-oxo-3,4-dihydro-2H-1-benzopyran-2-yl)-5,6-dihydro-2H-pyran-2-one"),
    ("O=C1C=CC(O1)C1CC(=O)c2ccccc2O1", "5-(4-oxo-3,4-dihydro-2H-1-benzopyran-2-yl)furan-2(5H)-one"),
    # the stereo form keeps the label of the stereo-free name
    ("C[C@H](C1CC(C)=C(C)C(=O)O1)C1CC(=O)c2ccccc2O1",
     "3,4-dimethyl-6-[(1S)-1-(4-oxo-3,4-dihydro-2H-1-benzopyran-2-yl)ethyl]-5,6-dihydro-2H-pyran-2-one"),
])
def test_lactone_parent_beside_senior_fused_ketone_is_not_pin(smiles, name):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # a heterocycle outranks a carbocycle (a),:19412): the lactone ring stays the parent
    ("O=C1OC(CC=C1)C1CCC(=O)c2ccccc21", "6-(4-oxo-1,2,3,4-tetrahydronaphthalen-1-yl)-5,6-dihydro-2H-pyran-2-one"),
    ("O=C1OC(CC=C1)C1CC(=O)c2ccccc21", "6-(3-oxo-2,3-dihydro-1H-inden-1-yl)-5,6-dihydro-2H-pyran-2-one"),
    ("O=C1OC(CC=C1)C1CCC(=O)CC1", "6-(4-oxocyclohexyl)-5,6-dihydro-2H-pyran-2-one"),
    # the dione ring has more ring carbonyls and is the parent already: unchanged
    ("O=C1Oc2ccccc2C=C1C1CC(=O)NC1=O", "3-(2-oxo-2H-1-benzopyran-3-yl)pyrrolidine-2,5-dione"),
])
def test_lactone_parent_beside_carbocyclic_ketone_keeps_pin(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
