"""P2 -- labels and public claims (2026-09-28): the exact-match coordination list.

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

So at every tier such a name is a checked name that is not the preferred name, from a
table of retained names: ``systematic_verified``, ``is_pin`` False (the paper's tiers:
"pin_verified, built by the strict PIN path and verified"; "systematic_verified, a
verified systematic name that is not the PIN"). Before this change heme b and
chlorophyll a read ``pin_verified`` / ``is_pin`` True, and the cobalamins with a
cobalt-carbon bond read ``best_effort`` (the demotion looked at the OPSIN label
only). The name, the source, 'identity' and the gate outcome are unchanged.

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


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("fx", [HEME_B, CHLOROPHYLL_A, CYANOCOBALAMIN],
                         ids=["heme_b", "chlorophyll_a", "cyanocobalamin"])
def test_coordination_list_name_is_systematic_verified_at_every_tier(fx, tier):
    # heme b / chlorophyll a were pin_verified (is_pin True); cyanocobalamin, with a
    # Co-C bond, was best_effort through the demotion.
    _assert_identity_label(fx, _row(fx["smiles"], tier))


@pytest.mark.parametrize("fx", FIXTURES, ids=[f["chebi"] for f in FIXTURES])
def test_every_listed_complex_is_labelled_systematic_verified(fx):
    _assert_identity_label(fx, _row(fx["smiles"], "pin"))


def test_a_certified_pin_keeps_pin_verified():
    # control: the rule touches the identity list only
    caffeine = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    row = _row(caffeine, "pin")
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin", row
    assert _key(_independent_parse(row["name"])) == _key(caffeine)


def test_a_pin_verified_label_always_means_an_opsin_read_back():
    # With the identity names moved to systematic_verified, every pin_verified row
    # carries verified 'opsin' (the claims-code-a rule allowed 'identity' too).
    for smiles in ("CCO", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", HEME_B["smiles"],
                   CHLOROPHYLL_A["smiles"]):
        row = _row(smiles, "pin")
        if row["tier"] == "pin_verified":
            assert row["verified"] == "opsin", row
            assert _key(_independent_parse(row["name"])) == _key(smiles), row
