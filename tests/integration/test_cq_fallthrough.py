"""CQ5 Task A — open the RT-failure fall-through at best-effort tier.

Root cause (trace-confirmed at HEAD on all 3 witnesses, a project rule):
for each witness the primary producer builds a NON-failure but RT-INVALID name;
``_final_opsin_validity_gate`` -> ``_self_consistency_decision`` voids it to the
descriptive fallback; the late ``_try_general_engine_recovery`` (namer.py:3278) is
then reached, but INSIDE ``name`` the general engine's substituent recursion runs
with the best-effort contextvars set + an elevated session-depth floor, so it emits
the SAME RT-failing name (not the systematic replacement-nomenclature name) and
declines at ``namer.py:4061``. The RT-verifying systematic candidate that
``name_general`` produces in a CLEAN context (best-effort contextvars reset +
depth-0 isolated session) is never consulted, and the molecule abstains.

Fix (offer-not-return, a project rule): on the ship-a-failure path, best-effort only,
re-invoke the RT-gated recovery in that clean context and adopt its result iff it is
a real (RT-verified) name. 0-wrong via the recovery's existing OPSIN round-trip gate
(a project rule — a candidate that does not round-trip stays abstained). PIN/complete
tiers are untouched (gated on ``_general_fallback_unverified``).
"""
from rdkit import Chem
import random
from unittest.mock import patch

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

# A QM9 dispiro. At the time this Task-A test file was written, name_general's
# output for it carried a WRONG dispiro descriptor (adjacent spiro atoms --
# `rules/spiro.py::_walk_ring_between_spiros` picked the LONGER middle-ring arc
# first, disagreeing with the descriptor string's own shorter-arc-first
# convention) that did NOT round-trip, so it correctly stayed abstained
# (0-wrong; a project rule). Task F (CQ5/QM9 finding,
# internal notes) fixed that root cause in
# ``rules/spiro.py`` directly, so this molecule now NAMES correctly
# at both tiers -- see ``test_dispiro_now_converts_after_taskF_descriptor_fix``
# below, which supersedes the old "stays abstained" assertion.
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


def test_dispiro_now_converts_after_taskF_descriptor_fix():
    """Task F fixed the ``rules/spiro.py`` root cause shorter-arc-
    first numbering), so this molecule -- formerly this file's "stays
    abstained" negative witness -- now emits a real, OPSIN-RT-verified name at
    BOTH best-effort and PIN tiers (a genuine PIN improvement: it previously
    had no PIN output at all, so this is not a byte-identity regression)."""
    out = _besteffort().name(NEGATIVE_DISPIRO)
    assert not is_failure_name(out), (
        f"best-effort still abstains on the Task-F-fixed dispiro {NEGATIVE_DISPIRO!r}")
    assert _rt_inchikey_match(out, NEGATIVE_DISPIRO), (
        f"shipped name {out!r} does not OPSIN-round-trip to {NEGATIVE_DISPIRO!r}")

    pin_out = Orthonym().name(NEGATIVE_DISPIRO)
    assert not is_failure_name(pin_out), (
        f"PIN tier still abstains on the Task-F-fixed dispiro {NEGATIVE_DISPIRO!r}")
    assert _rt_inchikey_match(pin_out, NEGATIVE_DISPIRO)


def test_negative_rt_mismatch_stays_abstained():
    """ a performance pass, smaller finding 1: this file's original negative
    witness (a QM9 dispiro whose name_general output carried a WRONG
    descriptor) was CONSUMED when Task F fixed that root cause -- the
    molecule now correctly converts (see
    ``test_dispiro_now_converts_after_taskF_descriptor_fix`` above), leaving
    this file with NO regression coverage for a project rule ("a wrong name is
    never shipped"). A search for a fresh, naturally-occurring RT-wrong
    general-engine candidate (dispiro/trispiro/cage variants, several dozen
    constructed + randomized-atom-order probes) found none currently live --
    every candidate tried round-trips (0-wrong is, at present, measured
    robust here). Rather than leave this regression UNTESTED (or invent a
    fake "fix" for a bug that does not reproduce), this tests the gate
    MECHANISM directly, the same way ``test_oligosaccharides_be_rt_gate.py``
    already does in this tree: force ``Orthonym._rt_match`` (trace-confirmed
    the actual gate this witness's fall-through consults -- 3 calls,
    ``certify_general_result`` 5 calls, ``opsin_roundtrip_check`` 0 calls) to
    report a mismatch for a real witness that would otherwise convert, and
    assert the pipeline abstains rather than shipping it. This is immune to
    going stale when some future patch fixes a specific bug (a real-molecule
    -based test would silently stop testing anything), and proves the exact
    mechanism this file's fall-through was built around never ships an
    RT-failing candidate."""
    be = _besteffort()
    smi = WITNESSES[1][0]
    assert not is_failure_name(be.name(smi)), (
        "sanity: this witness must normally convert (so the mock below is "
        "the only thing causing the abstain)")
    with patch.object(Orthonym, "_rt_match", staticmethod(lambda *a, **kw: False)):
        out = be.name(smi)
    assert is_failure_name(out), (
        f"shipped {out!r} for {smi!r} even though _rt_match reported no "
        f"match -- invariant 9 violated")


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
