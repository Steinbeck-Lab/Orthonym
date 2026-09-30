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
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC[Ga](CC)CC",
    "C[Al](C)C",
    "C[Al]C",
    "C[In](C)C",
    "[Al+3].[Cl-].[Cl-].[Cl-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "C[Al]C", "C[Al](C)C", "CC[Ga](CC)CC", "C[In](C)C", "[Al+3].[Cl-].[Cl-].[Cl-]"])
def test_group13_names_are_never_pin_verified(smiles):
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    r = _dt_row(smiles)
    assert r["tier"] not in ("pin_verified", "pin_unverified") and not r["is_pin"], r
    assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["CCO", "C[Si](C)(C)C", "CB(C)C"])
def test_other_elements_keep_their_labels(smiles):
    from orthonym import Orthonym
    assert _dt_row(smiles)["tier"] == "pin_verified"
