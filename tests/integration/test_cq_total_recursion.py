"""CQ1 Task B — make name_general produce correct whole-graph names, not abstain.

Bucket B1 (S-rooted sulfonyl/sulfinyl substituent). MEASURED root cause at HEAD
(spy-confirmed, invariant 8): the plan's named leaf
``substituent_naming.py::name_substituent_fragment`` records ZERO None-returns for
these; the on-path leaf is ``substituent_enumerator.py::name_substituent``. Its
Tier-1.85 sulfinyl/sulfonyl intercept already existed but its guard required the
PARENT-side attachment atom to be carbon, so a sulfonyl attached to the parent via
NITROGEN (a ring-N sulfonamide -- the diaza-spiro/-cycloalkane class) failed the
guard, fell to the skeletal-replacement generator, and emitted a
constitution-WRONG ``1-oxo-2-oxa-1λ6-thiaeth-1-en-1-yl`` token (an extra in-chain
oxa + one dropped =O) that the whole-graph OPSIN RT gate then voided -> abstain.

Fix (offer-not-return, best-effort only): allow the parent-side atom to be C or N,
so the substituent names as ``{R}sulfonyl`` / ``{R}sulfinyl`` (P-63.6), which OPSIN
round-trips. 0-wrong via the existing RT gate (invariant 9); PIN byte-identical
(gated on ``allow_mancude``).
"""
import random

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = [pytest.mark.integration, pytest.mark.roundtrip,
              pytest.mark.opsin_gate]


# Molecules that ABSTAINED at HEAD (name_general built a constitution-wrong
# skeletal-replacement token that the RT gate voided) and must now emit an
# RT-verifying whole-graph name. None => assert RT-match only (spelling may vary).
B1_CONVERTS = [
    # CQ1 diff witness: ring-N ethanesulfonyl on a diazaspiro (was abstain).
    ("CCS(=O)(=O)N1CC2(CCNC2)C1", None),
    # ring-N methanesulfinyl.
    ("CS(=O)N1CCCC1", None),
]

# Molecules that already round-tripped (ugly λ token or clean C-parent form); they
# must STILL round-trip after the fix (no regression). Assert RT-match only.
B1_NO_REGRESSION = [
    "CS(=O)(=O)N1CCCC1",       # ring-N methanesulfonyl (was ugly λ token)
    "CCS(=O)(=O)N1CCCCC1",     # ring-N ethanesulfonyl on piperidine
    "CCS(=O)(=O)c1ccccc1",     # C-parent: (ethanesulfonyl)benzene
    "CCS(=O)(=O)C1CCCCC1",     # C-parent: (ethanesulfonyl)cyclohexane
    "CS(=O)(=O)CCC",           # C-parent: 1-(methanesulfonyl)propane
]

# PIN gold controls: byte-identical current PIN output (the fix is best-effort
# gated on allow_mancude, so PIN must not shift).
PIN_GOLD = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("CCCCCCCc1ccccc1", "heptylbenzene"),
    ("ClCCCCC", "1-chloropentane"),
    ("O=C1CCNCC1", "piperidin-4-one"),
    ("Cc1ccc(O)nc1", "5-methylpyridin-2-ol"),
    ("CCS(=O)(=O)c1ccccc1", "(ethanesulfonyl)benzene"),
    ("CS(=O)(=O)CCC", "1-(methanesulfonyl)propane"),
]


def _besteffort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _rt_inchikey_match(name, smiles):
    assert not is_failure_name(name), f"expected a real name, got {name!r}"
    osmi = opsin_parse(name)
    assert osmi is not None, f"OPSIN could not parse {name!r}"
    return _inchikey(osmi) == _inchikey(smiles)


@pytest.mark.parametrize("smiles,expected", B1_CONVERTS)
def test_b1_converts_and_roundtrips(smiles, expected):
    be = _besteffort()
    out = be.name(smiles)
    assert not is_failure_name(out), (
        f"best-effort abstained on {smiles!r} (got {out!r}); the ring-N "
        f"sulfonyl/sulfinyl substituent must name as {{R}}sulfonyl, not the "
        f"constitution-wrong skeletal-replacement token")
    if expected is not None:
        assert out == expected
    assert _rt_inchikey_match(out, smiles), (
        f"shipped name {out!r} does not OPSIN-round-trip to {smiles!r}")


@pytest.mark.parametrize("smiles", B1_NO_REGRESSION)
def test_b1_no_regression_roundtrips(smiles):
    out = _besteffort().name(smiles)
    assert not is_failure_name(out), f"regressed to abstain on {smiles!r}"
    assert _rt_inchikey_match(out, smiles), (
        f"name {out!r} no longer round-trips to {smiles!r}")


@pytest.mark.parametrize("smiles,expected", PIN_GOLD)
def test_pin_gold_byte_identical(smiles, expected):
    assert Orthonym().name(smiles) == expected


def test_determinism_two_smiles_orders():
    """Two atom orderings of the diazaspiro sulfonyl witness -> identical name."""
    smi = B1_CONVERTS[0][0]
    mol = Chem.MolFromSmiles(smi)
    n = mol.GetNumAtoms()
    order_a = list(range(n))
    order_b = list(range(n))
    random.Random(1).shuffle(order_a)
    random.Random(2).shuffle(order_b)
    smi_a = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_a), canonical=False)
    smi_b = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order_b), canonical=False)
    assert smi_a != smi_b
    be = _besteffort()
    name_a = be.name(smi_a)
    name_b = be.name(smi_b)
    assert not is_failure_name(name_a)
    assert name_a == name_b, f"order-dependent name: {name_a!r} != {name_b!r}"
