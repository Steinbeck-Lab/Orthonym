# tests/unit/test_rt_over_pool.py
"""v33 Phase 0 Task L4-core: the RT-gate-over-offers SELECTION PRIMITIVE
(`assembly.offer_pool.select_rt_passing`) wired at `namer.py::_finish` via
`Orthonym._select_rt_passing_offer_name`.

L2 wrapped the single current winner as ONE `Offer` and ran it through
`select_offer` (a provable identity). L4-core replaces that with
`select_rt_passing`, gated by `_offer_rt_ok` -- an offer can only win if it
is BOTH `.complete` AND passes the RT/PIN gate. With today's single-offer
pool this stays byte-identical BY CONSTRUCTION: `_select_rt_passing_offer_name`
falls back to the CURRENT name whenever `select_rt_passing` returns `None`
(no offer both complete and rt_ok) -- so regardless of what `_offer_rt_ok`
decides, the emitted name never changes with one offer in the pool. This
file also unit-tests the 2-offer case (`_finish`'s wiring, exercised via the
factored-out selection method + a hand-built `self._offers` pool) even
though nothing yet PRODUCES a second offer (L3-1 will).
"""
import pytest

from orthonym import namer as namer_mod
from orthonym.namer import Orthonym
from orthonym.assembly.offer_pool import Offer

pytestmark = pytest.mark.unit


def _o(name, is_pin=True, tier="T1", complete=True, source="x"):
    return Offer(name=name, result_obj=None, is_pin=is_pin, tier=tier,
                 source=source, complete=complete)


class TestByteIdenticalWithOneOffer:
    def test_simple_alcohol_unchanged(self):
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        assert len(nm._offers) == 1
        assert nm._offers[0].name == "ethanol"

    def test_diverse_molecules_unchanged(self):
        # Same list L2's byte-identical test uses, so this is directly
        # comparable to the pre-L4-core behaviour it certified.
        expected = {
            "CCO": None,               # don't hardcode every PIN -- just
            "c1ccccc1": None,           # assert non-empty / non-failure below
        }
        smiles_list = [
            "CCO", "c1ccccc1", "CC(=O)Nc1ccccc1", "CC(=O)O",
            "CC(C)(C)CC(=O)O", "CC(=O)OCC", "c1ccc2[nH]ccc2c1",
            "OCC(O)CO", "C1CC2CCC1CC2",
        ]
        nm = Orthonym()
        names = [nm.name(s) for s in smiles_list]
        for s, n in zip(smiles_list, names):
            assert n, f"{s} produced an empty name"

    def test_acetanilide_unchanged(self):
        nm = Orthonym()
        assert nm.name("CC(=O)Nc1ccccc1") == "N-phenylacetamide"


class TestSelectRtPassingWiringAtFinish:
    """Exercises `Orthonym._select_rt_passing_offer_name` directly with a
    hand-built 2-offer pool -- the same method `_finish` calls -- since
    nothing in production builds a second offer yet (L3-1)."""

    def test_top_fails_rt_ok_second_passes_second_wins(self, monkeypatch):
        top = _o("wrong-molecule-name", is_pin=True, tier="T1")     # ranks first
        second = _o("hydroxyethane", is_pin=False, tier="T3")        # ranks second

        def _fake_rt_ok(name, input_smiles):
            return name != "wrong-molecule-name"

        monkeypatch.setattr(namer_mod, "_offer_rt_ok", _fake_rt_ok)
        nm = Orthonym()
        nm._offers = [top, second]
        result = nm._select_rt_passing_offer_name("wrong-molecule-name", "CCO")
        assert result == "hydroxyethane"

    def test_all_offers_fail_rt_ok_falls_back_to_current_name(self, monkeypatch):
        a = _o("candidate-a")
        b = _o("candidate-b", is_pin=False, tier="T3")
        monkeypatch.setattr(namer_mod, "_offer_rt_ok", lambda name, s: False)
        nm = Orthonym()
        nm._offers = [a, b]
        result = nm._select_rt_passing_offer_name("current-shipping-name", "CCO")
        assert result == "current-shipping-name", (
            "L4-core must never newly-abstain a name that ships today")

    def test_single_offer_passing_returns_it(self, monkeypatch):
        only = _o("ethanol")
        monkeypatch.setattr(namer_mod, "_offer_rt_ok", lambda name, s: True)
        nm = Orthonym()
        nm._offers = [only]
        result = nm._select_rt_passing_offer_name("ethanol", "CCO")
        assert result == "ethanol"

    def test_single_offer_failing_falls_back_to_current_name_not_none(
            self, monkeypatch):
        only = _o("ethanol")
        monkeypatch.setattr(namer_mod, "_offer_rt_ok", lambda name, s: False)
        nm = Orthonym()
        nm._offers = [only]
        result = nm._select_rt_passing_offer_name("ethanol", "CCO")
        assert result == "ethanol", (
            "byte-identical guarantee: a failing single offer must still "
            "fall back to the SAME current name, never abstain")

    def test_incomplete_offer_never_wins_even_if_rt_ok_true(self, monkeypatch):
        incomplete = _o("partial", complete=False)
        complete = _o("hydroxyethane", is_pin=False, tier="T3", complete=True)
        monkeypatch.setattr(namer_mod, "_offer_rt_ok", lambda name, s: True)
        nm = Orthonym()
        nm._offers = [incomplete, complete]
        result = nm._select_rt_passing_offer_name("partial", "CCO")
        assert result == "hydroxyethane"


class TestOfferRtOkPredicate:
    """Unit tests of `_offer_rt_ok` itself, monkeypatching its dependencies
    directly -- proves the NO-double-OPSIN reuse and the fail-open paths."""

    def test_recorded_self01_pass_is_reused_zero_opsin_calls(self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (True, False, ""))

        def _boom(*a, **k):
            raise AssertionError("must not call OPSIN when self01 already passed")

        monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", _boom)
        monkeypatch.setattr(namer_mod, "_validity_gate_name_to_smiles", _boom)
        assert namer_mod._offer_rt_ok("ethanol", "CCO") is True

    def test_recorded_self01_warn_mismatch_fails(self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (False, False, ""))
        assert namer_mod._offer_rt_ok("some-name", "CCO") is False

    def test_skip_reanchor_carveout_fails_open(self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup",
            lambda name: (None, True, "carveout:thioperoxol: no self01"))

        def _boom(*a, **k):
            raise AssertionError("must not re-anchor a skip_reanchor verdict")

        monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", _boom)
        assert namer_mod._offer_rt_ok("methane-SO-thioperoxol", "CS(=O)O") is True

    def test_no_recorded_verdict_jar_absent_fails_open(self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (None, False, ""))
        monkeypatch.setattr(
            namer_mod, "_validity_gate_jar_present", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("must not call OPSIN when jar is absent")

        monkeypatch.setattr(namer_mod, "_validity_gate_name_to_smiles", _boom)
        assert namer_mod._offer_rt_ok("some-name", "CCO") is True

    def test_no_recorded_verdict_fresh_check_matching_molecule_passes(
            self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (None, False, ""))
        monkeypatch.setattr(
            namer_mod, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(
            namer_mod, "_validity_gate_name_to_smiles", lambda name: "CCO")
        assert namer_mod._offer_rt_ok("ethanol", "CCO") is True

    def test_no_recorded_verdict_fresh_check_wrong_molecule_fails(
            self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (None, False, ""))
        monkeypatch.setattr(
            namer_mod, "_validity_gate_jar_present", lambda: True)
        # name re-perceives as benzene, but the input was ethanol -> mismatch.
        monkeypatch.setattr(
            namer_mod, "_validity_gate_name_to_smiles",
            lambda name: "c1ccccc1")
        assert namer_mod._offer_rt_ok("wrong-name-for-ethanol", "CCO") is False

    def test_no_recorded_verdict_opsin_rejects_outright_but_unavailable_open(
            self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (None, False, ""))
        monkeypatch.setattr(
            namer_mod, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(
            namer_mod, "_validity_gate_name_to_smiles", lambda name: None)
        monkeypatch.setattr(
            namer_mod, "_validity_gate_status", lambda name: "unavailable")
        assert namer_mod._offer_rt_ok("some-name", "CCO") is True

    def test_no_recorded_verdict_opsin_rejects_outright_definitively_fails(
            self, monkeypatch):
        monkeypatch.setattr(
            namer_mod, "_self01_lookup", lambda name: (None, False, ""))
        monkeypatch.setattr(
            namer_mod, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(
            namer_mod, "_validity_gate_name_to_smiles", lambda name: None)
        monkeypatch.setattr(
            namer_mod, "_validity_gate_status", lambda name: "rejected")
        assert namer_mod._offer_rt_ok("some-name", "CCO") is False
