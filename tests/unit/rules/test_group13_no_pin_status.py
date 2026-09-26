"""Aluminium, gallium, indium and thallium compounds carry no PIN status (fix round
2, wp7; whole-branch verification panel NIT on 'dimethylaluminum').

the Blue Book: "Names of organic compounds based on aluminium, gallium, indium,
and thallium are not followed by the parenthetical abbreviation (PIN), because the
decision to choose between a name based on organic or inorganic principles has not yet
been reached.":2062: names based on alumane, gallane, indigane and thallane "currently
do not have PIN status". C[Al]C shipped 'dimethylaluminum' at pin_verified (the name is
right: OPSIN parses it to the same radical graph).
"""
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "C[Al]C", "C[Al](C)C", "CC[Ga](CC)CC", "C[In](C)C", "[Al+3].[Cl-].[Cl-].[Cl-]"])
def test_group13_names_are_never_pin_verified(smiles):
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["tier"] not in ("pin_verified", "pin_unverified") and not r["is_pin"], r
    assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["CCO", "C[Si](C)(C)C", "CB(C)C"])
def test_other_elements_keep_their_labels(smiles):
    from orthonym import Orthonym
    assert Orthonym(style="pin").name_tiered(smiles)["tier"] == "pin_verified"
