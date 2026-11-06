"""Both-tier assertions of the PIN class program (docs/the workflow tooling/plans/2026-09-30-pin-class-program.md).

The default tier is ``Orthonym`` (the PIN tier: it emits a name only when the pipeline can
build the PIN). The breadth tier is the best-effort instance of ``tests/support/rt_assert``.
Every read-back is an independent OPSIN call (``rt_assert.name_is_rt_exact``)."""
from orthonym import Orthonym

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

_DEFAULT = Orthonym()


def name_default(smiles: str) -> dict:
    return _DEFAULT.name_tiered(smiles)


def name_breadth(smiles: str) -> dict:
    return name_best_effort(smiles)


def assert_pin_at_both_tiers(smiles: str, pin: str) -> None:
    assert name_is_rt_exact(pin, smiles), f"the expected PIN does not read back to {smiles}: {pin!r}"
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (pin, "pin_verified", True), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (pin, "pin_verified"), b


def assert_not_pin_labelled(smiles: str, non_pin: str) -> None:
    for res in (name_default(smiles), name_breadth(smiles)):
        assert not (res.get("name") == non_pin and res.get("tier") == "pin_verified"), res
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


def assert_declined_at_default(smiles: str) -> None:
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


#: The rule a carbon-free name follows (user decision 2026-11-02): PREFERRED IUPAC NAMES
#: (the Blue Book) "the label 'PIN' is added to the names of compounds whose parent hydride contains
#: at least one of the following elements: B, Si,... At, and that also contain at least one carbon atom in
#: their structure";:2058 "... the label 'preselected name' is added to appropriate names";
#: PRESELECTED NAMES (:2062).
CARBON_FREE_RULE = "P-12.1 (BlueBookV2.md:2056, :2058): a carbon-free name is a preselected name, never a PIN"


def assert_preselected_at_both_tiers(smiles: str, name: str) -> None:
    """A carbon-free structure (``CARBON_FREE_RULE``): the default tier declines it (NO_VERIFIED_PIN) and the
    breadth tier gives ``name``, unchanged, labelled systematic_verified, read back by an independent OPSIN call."""
    assert name_is_rt_exact(name, smiles), f"the expected name does not read back to {smiles}: {name!r}"
    d = name_default(smiles)
    assert (d.get("tier"), d.get("limit_code")) == ("abstain", "NO_VERIFIED_PIN"), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier"), b.get("is_pin")) == (name, "systematic_verified", False), b


def is_carbon_free(smiles: str) -> bool:
    """True iff the structure has no carbon atom (computed from the SMILES alone, not by the code under test)."""
    from rdkit import Chem
    return not any(a.GetAtomicNum() == 6 for a in Chem.MolFromSmiles(smiles).GetAtoms())


def tier_row(smiles: str, tier: str) -> dict:
    """``name_tiered`` of a fresh engine at ``tier`` ('pin', 'valid', 'complete', 'best-effort'): the
    engine of the command line's ``--emit-tier`` (``cli._tiered_namer``); 'pin' is ``default_tier.default_tier_row``."""
    from orthonym.cli import _tiered_namer
    return _tiered_namer("pin", tier).name_tiered(smiles)


def name_and_tier(smiles: str) -> tuple:
    """``(name, tier)`` of a fresh default engine (``default_tier.default_tier_row``)."""
    from tests.support.default_tier import default_tier_row
    row = default_tier_row(smiles)
    return row["name"], row["tier"]
