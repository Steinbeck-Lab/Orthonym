"""CQ5 Task A — open the RT-failure fall-through at best-effort tier.

Root cause (spy-confirmed at HEAD on all 3 witnesses, invariant 8):
for each witness the primary producer builds a NON-failure but RT-INVALID name;
``_final_opsin_validity_gate`` -> ``_self_consistency_decision`` voids it to the
descriptive fallback; the late ``_try_general_engine_recovery`` (namer.py:3278) is
then reached, but INSIDE ``name()`` the general engine's substituent recursion runs
with the best-effort contextvars set + an elevated session-depth floor, so it emits
the SAME RT-failing name (not the systematic replacement-nomenclature name) and
declines at ``namer.py:4061``. The RT-verifying systematic candidate that
``name_general`` produces in a CLEAN context (best-effort contextvars reset +
depth-0 isolated session) is never consulted, and the molecule abstains.

Fix (offer-not-return, invariant 18): on the ship-a-failure path, best-effort only,
re-invoke the RT-gated recovery in that clean context and adopt its result iff it is
a real (RT-verified) name. 0-wrong via the recovery's existing OPSIN round-trip gate
(invariant 9 — a candidate that does not round-trip stays abstained). PIN/complete
tiers are untouched (gated on ``_general_fallback_unverified``).
"""
from rdkit import Chem
import random

import pytest

from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

# opsin_gate: the whole point of Task A is the RT/validity gate voiding the
# primary and RT-gating the fall-through, so the gate MUST be enabled (it is
# disabled suite-wide by default — see tests/conftest.py).
pytestmark = [pytest.mark.integration, pytest.mark.roundtrip,
              pytest.mark.opsin_gate]


# (smiles, expected_besteffort_name_or_None). None => assert RT-match only.
WITNESSES = [
    ("C=C(O)N(C)[C@H](CCC)C(C)O/C=C/CF",
     "1-[(2R,5E)-7-fluoro-1,3-dimethyl-2-propyl-4-oxa-1-azahept-5-en-1-yl]"
     "eth-1-en-1-ol"),
    ("CC(C)N(C)[SiH](I)I",
     "2-(2,2-diiodo-1-methyl-1-aza-2-silaethyl)propane"),
    # #3 may vary in exact spelling -> assert RT-match, not the literal string.
    ("Oc1c(N=Nc2cccc(C(F)(F)F)c2)c2cc(F)cc(F)c2n1C1CSC1", None),
]

# A QM9 dispiro whose name_general output carries a WRONG dispiro descriptor that
# does NOT round-trip. It must STAY abstained (0-wrong; invariant 9).
NEGATIVE_DISPIRO = "C1C2(CCC2)C11CCO1"

# PIN gold controls (byte-identical current PIN output; must not shift).
PIN_GOLD = [
    ("CCCCCCCc1ccccc1", "heptylbenzene"),
    ("CCCCCCCCC1CCCCC1", "octylcyclohexane"),
    ("ClCCCCC", "1-chloropentane"),
    ("ClCC(F)C", "1-chloro-2-fluoropropane"),
    ("FCCCl", "1-chloro-2-fluoroethane"),
    ("O=C1CCNCC1", "piperidin-4-one"),
    ("Cc1ccc(O)nc1", "5-methylpyridin-2-ol"),
    ("O=C(O)CCS(=O)(=O)[O-]", "2-carboxyethane-1-sulfonate"),
]


def _besteffort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _rt_inchikey_match(name, smiles):
    """OPSIN-parse ``name`` and compare its InChIKey to the input's."""
    assert not is_failure_name(name), f"expected a real name, got {name!r}"
    osmi = opsin_parse(name)
    assert osmi is not None, f"OPSIN could not parse {name!r}"
    return _inchikey(osmi) == _inchikey(smiles)


@pytest.mark.parametrize("smiles,expected", WITNESSES)
def test_witness_converts_and_roundtrips(smiles, expected):
    be = _besteffort()
    out = be.name(smiles)
    assert not is_failure_name(out), (
        f"best-effort abstained on {smiles!r} (got {out!r}); the RT-verifying "
        f"general-engine candidate was never consulted")
    if expected is not None:
        assert out == expected
    assert _rt_inchikey_match(out, smiles), (
        f"shipped name {out!r} does not OPSIN-round-trip to {smiles!r}")


def test_pin_tier_unchanged_on_witnesses():
    """PIN tier (gfu=False) must be byte-identical to HEAD: it abstains on all
    three. If the fall-through leaked into PIN, PIN would emit a name here."""
    pin = Orthonym()
    for smiles, _ in WITNESSES:
        out = pin.name(smiles)
        assert is_failure_name(out), (
            f"PIN tier changed on {smiles!r}: emitted {out!r} (must stay abstain)")


@pytest.mark.parametrize("smiles,expected", PIN_GOLD)
def test_pin_gold_byte_identical(smiles, expected):
    assert Orthonym().name(smiles) == expected


def test_negative_dispiro_stays_abstained():
    """The general engine builds a WRONG dispiro descriptor for this QM9 cage
    that does not round-trip; the RT gate must keep it abstained (invariant 9 —
    verify what SHIPS, never ship a non-round-tripping name)."""
    out = _besteffort().name(NEGATIVE_DISPIRO)
    assert is_failure_name(out), (
        f"shipped a non-round-tripping name {out!r} for the dispiro negative")


def test_determinism_two_smiles_orders():
    """Two different atom orderings of witness #1 must yield the identical name."""
    smi = WITNESSES[0][0]
    mol = Chem.MolFromSmiles(smi)
    n = mol.GetNumAtoms()
    order_a = list(range(n))
    order_b = list(range(n))
    random.Random(1).shuffle(order_a)
    random.Random(2).shuffle(order_b)
    smi_a = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_a), canonical=False)
    smi_b = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_b), canonical=False)
    assert smi_a != smi_b  # genuinely different spellings
    be = _besteffort()
    name_a = be.name(smi_a)
    name_b = be.name(smi_b)
    assert not is_failure_name(name_a)
    assert name_a == name_b, f"order-dependent name: {name_a!r} != {name_b!r}"
