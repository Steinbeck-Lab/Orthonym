"""P2 -- labels and public claims: the exact-match coordination list.

The D1 table (``data/coordination_retained.py``: heme, chlorophyll, cobalamin,
siroheme, coenzyme F430) maps the exact standard InChIKey of a ChEBI structure to its
ChEBI name. OPSIN 2.9.0 reads none of these names, so the check is the identity of the
InChIKey (``verified`` 'identity'). The names are ChEBI names of coordination entities,
not Preferred IUPAC Names:

* INTRODUCTION (the Blue Book): "Although coordination nomenclature is not
  discussed in these recommendations,..." and PINs are recommended "when there is a
  choice within the limits of the nomenclature of organic compounds,... but not between
  a coordination or binary name and an organic name".
* INTRODUCTION (the Blue Book): "However, neither preferred IUPAC names or
  preselected names (see for organometallic compounds involving the transition
  elements (including the Group 3 elements) and Groups 1 and 2 elements, except for
  'ocene' compounds, are noted."

* INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not
  identified for the compounds in this Chapter" (the porphyrin, corrin and phorbine parents
  of these names are parents).

User decision 2026-11-02 ("decide what the Blue Book says", R1-NP), which replaces the
paper-run labels of 2026-09-30 (35 ``pin_verified`` and 3 ``best_effort`` on ChEBI):
changed asserted labels, every list name is ``systematic_verified`` with ``is_pin`` False
at the wider tiers, and the PIN tier declines it (NO_VERIFIED_PIN). The name, the source,
'identity' and the gate outcome are unchanged.

Every row is also checked independently of the engine: the input's InChIKey is the
table key of that exact name, and a FRESH OPSIN call (``tests.support.rt_assert.
_independent_parse``) reads nothing for the name, so no label may claim an OPSIN
read-back for it.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.data.coordination_retained import COORDINATION_RETAINED
from orthonym.validation.twin_spellings import twin_of
from tests.support.rt_assert import _independent_parse
from tests.unit.rules.test_d1_coordination_v36 import (
    CHLOROPHYLL_A,
    CYANOCOBALAMIN,
    FIXTURES,
    HEME_B,
)

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

TIERS = ["pin", "valid", "complete", "best-effort"]


def _row(smiles, tier):
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


def _assert_identity_label(fx, row):
    assert row["name"] == fx["name"], row                  # name unchanged
    assert COORDINATION_RETAINED.get(_key(fx["smiles"])) == fx["name"]  # identity holds
    assert row["verified"] == "identity", row
    assert row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False, row
    assert _independent_parse(row["name"]) is None, row["name"]  # OPSIN reads nothing


def _assert_declined(row):
    assert (row["tier"], row["limit_code"], row["is_pin"]) == ("abstain", "NO_VERIFIED_PIN", False), row


# The PIN and best-effort tiers run in the PR run; the 'valid' and 'complete' tiers (30 s
# of the 3 x 4 rows) run in the nightly run only.
_NIGHTLY_TIERS = {"valid", "complete"}


@pytest.mark.parametrize(
    "tier",
    [pytest.param(t, marks=pytest.mark.slow) if t in _NIGHTLY_TIERS else t for t in TIERS])
@pytest.mark.parametrize("fx", [HEME_B, CHLOROPHYLL_A, CYANOCOBALAMIN],
                         ids=["heme_b", "chlorophyll_a", "cyanocobalamin"])
def test_coordination_list_name_is_below_the_pin_at_every_tier(fx, tier):
    # heme b, chlorophyll a and cyanocobalamin: declined at the PIN tier, systematic_verified at the others
    if tier == "pin":
        _assert_declined(_row(fx["smiles"], tier))
        return
    _assert_identity_label(fx, _row(fx["smiles"], tier))


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["chebi"] for f in FIXTURES])
def test_every_listed_complex_is_declined_at_the_default_tier_and_named_at_best_effort(fx):
    _assert_declined(_row(fx["smiles"], "pin"))
    _assert_identity_label(fx, _row(fx["smiles"], "best-effort"))


def test_a_certified_pin_keeps_pin_verified():
    # control: the rule touches the identity list only
    caffeine = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    row = _row(caffeine, "pin")
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin", row
    assert _key(_independent_parse(row["name"])) == _key(caffeine)


def test_a_pin_verified_label_means_an_opsin_read_back_or_the_list_identity():
    # The claims-code-a rule: a pin_verified row carries verified 'opsin' (checked
    # here by a fresh OPSIN call), 'opsin_twin' (a fresh OPSIN call reads the twin
    # spelling of the name to the input, and reads nothing for the name itself) or
    # 'identity' (the input's key is the table key of exactly that name).
    for smiles in ("CCO", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", HEME_B["smiles"],
                   CHLOROPHYLL_A["smiles"], "C12C=CC(C3=CC=CC=C13)S2"):
        row = _row(smiles, "pin")
        if row["tier"] == "pin_verified":
            assert row["verified"] in ("opsin", "opsin_twin", "identity"), row
            if row["verified"] == "opsin":
                assert _key(_independent_parse(row["name"])) == _key(smiles), row
            elif row["verified"] == "opsin_twin":
                twin, _rules = twin_of(row["name"])
                assert _independent_parse(row["name"], smiles) is None, row
                assert _key(_independent_parse(twin, smiles)) == _key(smiles), row
            else:
                assert COORDINATION_RETAINED.get(_key(smiles)) == row["name"], row
