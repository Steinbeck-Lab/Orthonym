"""A ring assembly that is a substituent keeps the groups on its rings.

 / name the bare assembly as a prefix ('[1,1'-biphenyl]-4-yl', the Blue Book); a group on
one of its rings is cited with it ('4'-chloro[1,1'-biphenyl]-4-yl',,:15560 and:7455
'(4'-cyano[1,1'-biphenyl]-4-yl)oxy'). The composer's multi-ring pass built the bare prefix for a decorated
assembly too, so the chloro, ethyl or ethenyl group was dropped from the name; the OPSIN validity gate
withdraws such a name, but a charged molecule is named through its neutral form with the gate off, and the
dropped group then made the whole molecule abstain. The suite runs with the gate off, so these rows see
the producer itself; every name is read back to the input by an independent OPSIN call.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

DECORATED_ASSEMBLY_ON_A_CHAIN = [
    "Clc1ccc(c2ccc(C(N)OC)cc2)cc1",
    "CCc1ccc(c2ccc(C(N)OC)cc2)cc1",
    "C=Cc1ccc(c2ccc(CC(N)OC)cc2)cc1",
    "C=CC1CCC(C2CCC(C(N)OC)CC2)CC1",
    "C=CC1CCC(C2CCC(C(N)OC(N)c3ccc(C(C)(C)C)cc3)CC2)CC1",
]


@pytest.mark.parametrize("smiles", DECORATED_ASSEMBLY_ON_A_CHAIN)
def test_decorated_ring_assembly_prefix_keeps_its_groups(smiles):
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res


def test_bare_assembly_keeps_the_book_prefix():
    smiles = "c1ccc(c2ccc(C(N)OC)cc2)cc1"
    assert name_best_effort(smiles)["name"] == "1-([1,1'-biphenyl]-4-yl)-1-methoxymethanamine"


# ---------------------------------------------------------------------------------------------------------------
# What the name is labelled: the PIN cites an assembly as one unit DEFINITIONS, the Blue Book;
# prefixes,:50157), and a compound prefix takes 'bis',:4857).
# ---------------------------------------------------------------------------------------------------------------
from tests.support.pin_tiers import name_breadth, name_default  # noqa: E402

NESTED = [  # three or more rings with a group: the chokepoint names one ring and cites the others
    "OC(=O)CCc1ccc(-c2ccc(-c3ccc(Cl)cc3)cc2)cc1",
    "OC(=O)CCc1ccc(-c2ccc(-c3ccc(C)cc3)cc2)cc1",
    "OC(=O)CCc1ccc(-c2ccc(-c3ccccc3)cc2Cl)cc1",
    "OC(=O)CCc1ccc(-c2ccc(-c3ccccc3)cc2)c(Cl)c1",
    "O=C(O)CCc1ccc(-c2ccccc2-c2ccccc2Cl)cc1",
    "NCc1ccc(-c2ccc(-c3ccc(-c4ccc(F)cc4)cc3)cc2)cc1",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", NESTED)
def test_nested_assembly_name_is_not_a_pin(smiles):
    # the default tier declines it, as before the group was kept, and the wider tier labels it below the PIN
    assert name_default(smiles).get("tier") != "pin_verified"
    res = name_breadth(smiles)
    assert res["name"] and res["tier"] != "pin_verified" and name_is_rt_exact(res["name"], smiles), res


@pytest.mark.opsin_gate
def test_two_ring_assembly_with_a_group_keeps_the_assembly_prefix_pin():
    smiles = "OC(=O)CCc1ccc(-c2ccc(Cl)cc2)cc1"
    assert name_default(smiles)["name"] == "3-(4'-chloro[1,1'-biphenyl]-4-yl)propanoic acid"


@pytest.mark.opsin_gate
def test_repeated_decorated_assembly_prefix_takes_bis():
    smiles = "OC(=O)C(c1ccc(-c2ccc(Cl)cc2)cc1)c1ccc(-c2ccc(Cl)cc2)cc1"
    assert name_breadth(smiles)["name"] == "bis(4'-chloro[1,1'-biphenyl]-4-yl)acetic acid"
