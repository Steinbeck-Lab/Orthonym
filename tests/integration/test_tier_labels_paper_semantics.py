"""Tier labels follow the paper's semantics (label-fix, 2026-09-28).

The paper's tiers (Methods, "Tiers"; the brief of this job):

- pin_verified: built by the strict PIN path, certified as the PIN, and verified.
- pin_unverified: a verified name from the PIN path that is not certified as the PIN.
- systematic_verified: a verified systematic name from the general engine or a
  trivial/retained table that is not the PIN ("for example, a von Baeyer name for a
  fused ring system").
- best_effort: the best-effort rescue producers (the last-resort floor offer), and a
  name no round trip of the shipped string verified.

Since the code the paper measured, two label paths put VERIFIED non-PIN
names at best_effort:

1. ``name_tiered`` copied the tier of the winning primary ``Offer``, which ``_finish``
   derives from the general engine's own round-trip value (``prov['opsin']``) alone.
   The inline engine lane records no such value, so a name the validity gate verified
   read best_effort. The case below moved onto that lane with bdd69a673 (the legacy
   pipeline no longer names one ring of a polycyclic system as a monocycle, so the
   gate no longer suppresses a wrong name and the late-recovery lane no longer runs).
   The same copy put every verified PIN-path name carrying a recorded non-PIN part
   (``record_non_pin_fragment``, bbab2b06c and the classes of 09bb567c5, decision A,
   ...) at best_effort: prov['opsin'] is never set on the PIN path.
2. The shipped-name check of ba4c1a58c verifies a name after its label was derived;
   the label kept the best_effort it read while unverified.

Labels only: no name changes (the end checks compare every row of a dev split,
milestone1500 and dev2000 at both tiers against a fresh run of b757091f2).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as pv
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import _independent_parse

pytestmark = [pytest.mark.integration]

# The user's case: a von Baeyer name of an ortho-fused system (not the PIN; the PIN
# would be a hydro-fusion name), built by the general engine, verified by the gate
#  and read back by OPSIN to the input's full InChIKey.
CASE = "C1CC2CCC1(CC2)C3CCC4(CCC5(CCCC5C4C3)C)C"
CASE_NAME = "12-(bicyclo[2.2.2]octan-1-yl)-6,9-dimethyltricyclo[7.4.0.0^2,6]tridecane"


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


@pytest.mark.opsin_gate
def test_verified_general_engine_von_baeyer_name_is_systematic_verified():
    jar_or_skip()
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(CASE)
    assert row["name"] == CASE_NAME, row
    assert row["source"] == "general_engine", row
    assert row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False, row
    assert row["opsin"] == "verified" and row["verified"] == "opsin", row
    # independent OPSIN read-back (not through the engine), full InChIKey
    assert _key(_independent_parse(row["name"])) == _key(CASE)


# ---------------------------------------------------------------------------
# The label rule itself, on a hand-built offer pool (the way `_finish` leaves it)
# ---------------------------------------------------------------------------

def _fake_name_with_primary_offer(source, name, offer_tier, *, gate_outcome=None,
                                  non_pin_fragment=None):
    """A stand-in for ``Orthonym.name`` that leaves behind what ``_finish`` leaves for
    a primary-offer win: the provenance of the producer and ONE primary Offer whose
    tier ``_finish`` computed from ``prov['opsin']`` (None here, as on the PIN path
    and the inline engine lane)."""
    from orthonym.assembly.offer_pool import Offer

    def _fake(self, smiles):
        if source != "pin_path":
            pv.record_source(source)
        if non_pin_fragment:
            pv.record_non_pin_fragment(non_pin_fragment)
        if gate_outcome:
            pv.record_gate_outcome(gate_outcome, name)
        offer = Offer(name=name, result_obj=None, is_pin=False, tier=offer_tier,
                      source=source, complete=True)
        self._offers = [offer]
        self._last_selected_offer = offer
        return name
    return _fake


def test_inline_lane_primary_offer_verified_by_the_gate_is_systematic_verified(
        monkeypatch):
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "general_engine", "ethanol", "best_effort",
        gate_outcome=pv.GATE_OUTCOME_SELF01))
    row = Orthonym().name_tiered("CCO")
    assert row["tier"] == "systematic_verified", row
    assert row["source"] == "general_engine" and row["is_pin"] is False, row
    assert row["verified"] == "opsin", row


def test_pin_path_name_with_a_non_pin_part_verified_is_pin_unverified(monkeypatch):
    name = "(trimethylazaniumyl)acetate"
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "pin_path", name, "best_effort", gate_outcome=pv.GATE_OUTCOME_SELF01,
        non_pin_fragment="trimethylazaniumyl"))
    row = Orthonym().name_tiered("C[N+](C)(C)CC(=O)[O-]")
    assert row["tier"] == "pin_unverified", row
    assert row["is_pin"] is False and row["source"] == "pin_path", row
    assert row["verified"] == "opsin", row


def test_pin_path_name_with_a_non_pin_part_unverified_stays_best_effort(monkeypatch):
    # no verified outcome recorded for the string; the default tier runs no
    # shipped-name check, so nothing verified it
    name = "(trimethylazaniumyl)acetate"
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "pin_path", name, "best_effort", non_pin_fragment="trimethylazaniumyl"))
    row = Orthonym().name_tiered("C[N+](C)(C)CC(=O)[O-]")
    assert row["tier"] == "best_effort", row
    assert row["verified"] == "unverified", row


def test_general_engine_primary_offer_without_a_round_trip_stays_best_effort(
        monkeypatch):
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "general_engine", "ethanol", "best_effort"))
    row = Orthonym().name_tiered("CCO")
    assert row["tier"] == "best_effort", row
    assert row["verified"] == "unverified", row


@pytest.mark.opsin_gate
def test_name_verified_by_the_shipped_name_check_takes_its_verified_tier(monkeypatch):
    """No gate outcome was recorded for the shipped string (an offer the gate never
    judged); the best-effort tier's shipped-name check reads it back to the full key,
    and the label follows: systematic_verified for the general engine."""
    jar_or_skip()
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "general_engine", "ethanol", "best_effort"))
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered("CCO")
    assert row["name"] == "ethanol", row
    assert row["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED, row
    assert row["tier"] == "systematic_verified", row
    assert row["verified"] == "opsin", row


@pytest.mark.opsin_gate
def test_pin_path_non_pin_name_verified_by_the_shipped_name_check_is_pin_unverified(
        monkeypatch):
    jar_or_skip()
    name = "(trimethylazaniumyl)acetate"
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "pin_path", name, "best_effort", non_pin_fragment="trimethylazaniumyl"))
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(
        "C[N+](C)(C)CC(=O)[O-]")
    assert row["name"] == name, row
    assert row["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED, row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row


@pytest.mark.opsin_gate
def test_no_pin_status_element_name_verified_by_the_shipped_name_check_is_systematic(
        monkeypatch):
    """the Blue Book (alumane names "currently do not have PIN status"): such a
    name is a verified systematic name at most. The demotion reads the final
    verification, so a name the shipped-name check verified is systematic_verified,
    not the best_effort it read before the check ran."""
    jar_or_skip()
    monkeypatch.setattr(Orthonym, "name", _fake_name_with_primary_offer(
        "pin_path", "trimethylalumane", "pin_verified"))
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(
        "C[Al](C)C")
    assert row["name"] == "trimethylalumane", row
    assert row["gate_outcome"] == pv.GATE_OUTCOME_FULL_KEY_VERIFIED, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert _key(_independent_parse(row["name"])) == _key("C[Al](C)C")


def test_last_resort_floor_offer_stays_best_effort(monkeypatch):
    from orthonym.assembly.offer_pool import Offer

    def _fake(self, smiles):
        pv.record_gate_outcome(pv.GATE_OUTCOME_SELF01, "ethanol")
        primary = Offer(name="wrong-name", result_obj=None, is_pin=True,
                        tier="pin_verified", source="pin_path", complete=True)
        floor = Offer(name="ethanol", result_obj=None, is_pin=False,
                      tier="best_effort", source="t4_floor", complete=True)
        self._offers = [primary, floor]
        self._last_selected_offer = floor
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", _fake)
    row = Orthonym().name_tiered("CCO")
    assert row["tier"] == "best_effort", row
    assert row["source"] == "t4_floor" and row["is_pin"] is False, row
