"""Choline-family acid-ester-anion zwitterions (a phase B1).

Wires the cation-bearing-owner-substituent capability  into the
GUARD-4 zwitterion path (`charged_router._route_zwitterion`) so a cation +
acid-ester anion zwitterion (a single P/S oxoacid mono-ester whose owner arm
carries the quaternary cation, e.g. choline sulfate/phosphate) gets a real
name instead of abstaining. RT-gated (`@pytest.mark.opsin_gate`); the
top-level /OPSIN gate is the 0-wrong backstop.
"""
import pytest
from orthonym import Orthonym
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
    "C[N+](C)(C)CCC(=O)[O-]",
    "C[N+](C)(C)CCOP(=O)([O-])O",
    "C[N+](C)(C)CCOP(=O)([O-])[O-]",
    "C[N+](C)(C)CCOS(=O)(=O)[O-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


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



@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# --- the 3 goal molecules -----------------------------------------------
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("C[N+](C)(C)CCOS(=O)(=O)[O-]", "2-(trimethylazaniumyl)ethyl sulfate"),
    ("C[N+](C)(C)CCOP(=O)([O-])[O-]", "2-(trimethylazaniumyl)ethyl phosphate"),
    ("C[N+](C)(C)CCOP(=O)([O-])O", "2-(trimethylazaniumyl)ethyl hydrogen phosphate"),
])
def test_choline_ester_anion_zwitterion(namer, smi, expected):
    assert _dt_obj_name(namer, smi) == expected


# --- regressions: untouched paths ----------------------------------------
@pytest.mark.opsin_gate
def test_carboxylate_betaine_unchanged(namer):
    # GUARD 4's existing (…azaniumyl)-prefix betaine path must be
    # completely unaffected by the new ester-anion helper (its central-atom
    # check declines instantly for a carboxylate O, whose neighbour is C).
    assert _dt_obj_name(namer, "C[N+](C)(C)CCC(=O)[O-]") == "3-(trimethylazaniumyl)propanoate"


@pytest.mark.opsin_gate
def test_pure_anion_dodecyl_sulfate_unchanged(namer):
    # Slice A's pure-anion producer (no cation at all) is a completely
    # separate dispatch path (route_charged never calls _route_zwitterion
    # when there is no cation site) -- must stay byte-identical.
    assert _dt_obj_name(namer, "CCCCCCCCCCCCOS(=O)(=O)[O-]") == "dodecyl sulfate"


# --- fail-closed: a bad shape must never emit a sulfate/phosphate name ---
@pytest.mark.opsin_gate
def test_thiophosphate_ester_zwitterion_failclosed(namer):
    # P=S (thio) ester zwitterion -- not the clean mono-ester shape; the new
    # helper must decline (not fabricate a phosphate/sulfate name). Abstain
    # (or some non-phosphate/sulfate fallback name) is acceptable; a name
    # containing 'phosphate' or 'sulfate' is NOT.
    out = _dt_obj_name(namer, "C[N+](C)(C)CCOP(=S)([O-])[O-]")
    assert out is not None
    assert "phosphate" not in out
    assert "sulfate" not in out
