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
