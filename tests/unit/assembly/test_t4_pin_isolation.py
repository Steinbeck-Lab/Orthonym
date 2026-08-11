"""PIN-isolation regression harness for the T4 best-effort namer (coverage-by-
construction breadth build, 2026-08-11, Task 1).

The T4 namer fires ONLY after the PIN/default path abstains, so PIN output
must be byte-identical whether or not T4 is opted in. This is the regression
witness that proves it: a fixed set of molecules that emit a PIN name today,
asserted both under the plain default path and under an ``Orthonym``
instance with the T4/general-fallback recovery path fully enabled.

Correction over the naive brief: constructing ``Orthonym`` with only
``general_fallback_unverified=True`` is a silent no-op — the real master
switch for the general-fallback/T4 recovery path is ``general_fallback``
(an early ``return None`` at ``namer.py:3123`` when it is false). So this
test turns BOTH flags on to make the isolation assertion meaningful: if PIN
witnesses are still emitted identically with the recovery path fully live,
isolation is genuinely proven, not proven by the path never running.
"""

import pytest

from orthonym import name_compound
from orthonym.namer import Orthonym

# Molecules that emit a PIN name today; their default output must never change.
PIN_WITNESSES = {
    "CCO": "ethanol",
    "CC(=O)O": "acetic acid",
    "c1ccccc1": "benzene",
    "CC(=O)OC": "methyl acetate",
    "CC(=O)OC1CCCCC1": "cyclohexyl acetate",
    "OC(=O)c1ccccc1": "benzoic acid",
    "C1CCCCC1": "cyclohexane",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "2-[4-(2-methylpropyl)phenyl]propanoic acid",
}


@pytest.mark.parametrize("smi,expected", list(PIN_WITNESSES.items()))
def test_pin_default_unchanged(smi, expected):
    assert name_compound(smi) == expected


@pytest.mark.parametrize("smi,expected", list(PIN_WITNESSES.items()))
def test_t4_namer_does_not_change_pin_emissions(smi, expected):
    # A molecule the PIN path CAN name must be named identically regardless of
    # the T4 opt-in, because T4 fires only after the PIN path abstains.
    # Both flags must be on: ``general_fallback`` is the master switch for the
    # general-fallback/T4 recovery path (namer.py:3123); enabling only
    # ``general_fallback_unverified`` never runs that path at all.
    t4 = Orthonym(general_fallback=True, general_fallback_unverified=True)
    assert t4.name(smi) == expected
