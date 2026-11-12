import pytest
from orthonym.rules.charged_router import _parent_has_chain_locants
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
# The unsaturated betaine and L-carnitine left the list when the cation prefix stopped being glued in front of the
# parent's descriptor and prefixes: their best-effort names are pinned below and in test_azaniumyl_zwitterions.py.
DEFAULT_TIER_DECLINES = frozenset()
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles, on_failure="label")


# Direct unit tests for _parent_has_chain_locants (JVM-free)
def test_parent_has_chain_locants_saturated():
    """Saturated chain anions with chain locants."""
    assert _parent_has_chain_locants("propanoate") is True


def test_parent_has_chain_locants_unsaturated_enoate():
    """Unsaturated chain anions ending in -enoate."""
    assert _parent_has_chain_locants("(2E)-but-2-enoate") is True


def test_parent_has_chain_locants_dioate():
    """Dicarboxylic chain anions ending in -dioate."""
    assert _parent_has_chain_locants("butanedioate") is True


def test_parent_has_chain_locants_ynone():
    """Triple-bond chain anions ending in -ynoate."""
    assert _parent_has_chain_locants("prop-2-ynoate") is True


def test_parent_has_chain_locants_retained_acetate():
    """Retained anion names without chain locants."""
    assert _parent_has_chain_locants("acetate") is False


def test_parent_has_chain_locants_retained_benzoate():
    """Retained aromatic anion names."""
    assert _parent_has_chain_locants("benzoate") is False


def test_parent_has_chain_locants_retained_2_naphthoate():
    """Retained naphthoate (positional, not chain-numbered)."""
    assert _parent_has_chain_locants("2-naphthoate") is False


# Integration tests with Orthonym.name — require OPSIN gate ON
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # saturated betaine (regression check)
    ("C[N+](C)(C)CCC(=O)[O-]", "3-(N,N-dimethylmethanaminiumyl)propanoate"),
])
def test_unsaturated_betaine_integration(namer, smi, expected):
    """Integration tests for unsaturated and saturated betaine zwitterions."""
    assert _dt_obj_name(namer, smi) == expected


@pytest.mark.opsin_gate
def test_the_unsaturated_betaine_cites_its_descriptor_in_front():
    """ "NAMING OF STEREOISOMERS" (the Blue Book),:44643: stereodescriptors "are placed at the
    front of the complete name when related to the parent structure". The route names the parent without the
    cation and used to glue the prefix in front of the parent's descriptor ('4-(trimethylazaniumyl)(2E)-but-2-
    enoate'); the parent's E/Z block is now cited in front of the whole name."""
    from tests.support.pin_tiers import tier_row
    from tests.support.rt_assert import assert_full_rt
    smi = "C[N+](C)(C)C/C=C/C(=O)[O-]"
    row = tier_row(smi, "best-effort")
    assert (row["name"], row["tier"]) == ("(2E)-4-(trimethylazaniumyl)but-2-enoate", "systematic_verified"), row
    assert_full_rt(row["name"], smi)
    # the PIN spells the cation as '(N,N-dimethylmethanaminiumyl)acetate (PIN)' does (:42473)
    row = tier_row(smi, "pin")
    assert (row["name"], row["tier"]) == ("(2E)-4-(N,N-dimethylmethanaminiumyl)but-2-enoate", "pin_verified"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # longer saturated betaine (regression)
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(N,N-dimethylmethanaminiumyl)butanoate"),
    # L-carnitine: tests/unit/rules/test_azaniumyl_zwitterions.py (_COMPOSED)
])
def test_betaine_regressions(namer, smi, expected):
    """Ensure saturated cases still work correctly."""
    assert _dt_obj_name(namer, smi) == expected
