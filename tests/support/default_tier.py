"""The default tier's emission rule, for tests that asserted a default-tier name.

The submitted paper, Methods, "Tiers": "The default configuration emits a name only
when the pipeline can build the preferred IUPAC name (PIN); otherwise, it declines."
User decision 2026-09-30 ("Ship it in 1.0.2"): the default tier emits a name only when
the strict PIN path built it and verified it (pin_verified), or when it is one of the
paper's default-tier exceptions; every other name is declined with the reason code
NO_VERIFIED_PIN, and the wider tiers keep every name.

A test that asserted a default-tier name which the code records as not the PIN now
asserts, for that name:

1. the default tier declines it: tier ``abstain``, ``limit_code`` NO_VERIFIED_PIN, and
   the plain call returns a label (``assert_default_tier_declines``);
2. the strict path still builds exactly that name (the emission rule switched off,
   ``strict_path_name``), so the name stays under test;
3. the best-effort tier gives exactly that name, read back by a fresh OPSIN call to the
   input's full InChIKey (``assert_best_effort_gives``). For the few molecules whose
   best-effort name is another one, ``declined_at_default(..., best_effort_same=False)``
   asserts instead that the best-effort name reads back exactly.
"""
from orthonym import Orthonym
from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt, name_best_effort


def default_tier_row(smiles: str) -> dict:
    """The default tier's provenance row (a fresh engine, the rule applied)."""
    return Orthonym(style="pin").name_tiered(smiles)


def assert_default_tier_declines(smiles: str) -> dict:
    """The default tier declines ``smiles`` with NO_VERIFIED_PIN; returns the row."""
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row
    assert is_failure_name(row["name"]), row
    assert row["is_pin"] is False, row
    return row


def strict_path_name(smiles: str) -> str:
    """The name the strict path builds for ``smiles`` with the default tier's emission
    rule switched off (the name the default tier used to return)."""
    import orthonym.namer as namer
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        return Orthonym(style="pin").name(smiles)
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)


def strict_path_row(smiles: str) -> dict:
    """``name_tiered`` of the strict path with the emission rule switched off."""
    import orthonym.namer as namer
    token = namer._DEFAULT_TIER_POLICY_OFF.set(True)
    try:
        return Orthonym(style="pin").name_tiered(smiles)
    finally:
        namer._DEFAULT_TIER_POLICY_OFF.reset(token)


def assert_best_effort_gives(smiles: str, name: str) -> dict:
    """The best-effort tier gives exactly ``name``, read back to the full InChIKey."""
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert_full_rt(name, smiles)
    return row


def declined_at_default(smiles: str, name: str, *, best_effort_same: bool = True) -> dict:
    """Steps 1-3 of the module docstring for a name the default tier used to return;
    returns the best-effort row."""
    assert_default_tier_declines(smiles)
    assert strict_path_name(smiles) == name, (smiles, name)
    assert_full_rt(name, smiles)
    if best_effort_same:
        return assert_best_effort_gives(smiles, name)
    row = name_best_effort(smiles)
    assert not is_failure_name(row["name"]), row
    assert_full_rt(row["name"], smiles)
    return row


def declined_pin_row(smiles: str, *, best_effort_same: bool = True) -> dict:
    """For a molecule whose default-tier name the code records as not the PIN: the
    default tier declines it (NO_VERIFIED_PIN); the best-effort tier gives the strict
    path's name exactly, read back to the full InChIKey (``best_effort_same=False``:
    a best-effort name of its own that reads back exactly). Returns the strict path's
    row -- the name and the label the default tier derives before it declines -- so a
    test keeps asserting that name and label."""
    assert_default_tier_declines(smiles)
    row = strict_path_row(smiles)
    assert not is_failure_name(row["name"]), row
    assert_full_rt(row["name"], smiles)
    be = name_best_effort(smiles)
    if best_effort_same:
        assert be["name"] == row["name"], (row, be)
    else:
        assert not is_failure_name(be["name"]), be
        assert_full_rt(be["name"], smiles)
    return row


def default_tier_rule_applies() -> bool:
    """True when a default-tier call is under the emission rule: the OPSIN validity
    gate on (the suite switches it off unless a test asks for it). The rule applies
    with or without the OPSIN jar (namer.py ``_default_tier_policy_applies``): in the
    reduced mode without a jar the default tier still declines a name that is not a
    PIN."""
    import orthonym.namer as namer
    return not namer._DISABLE_VALIDITY_GATE
