# tests/unit/test_offer_at_finish.py
""" a phase Task L2.2: `select_offer` passthrough wired at `namer.py::_finish`.

L2 builds the whole-molecule Offer + rank_offers/select_offer selector
(`assembly/offer_pool.py`, task L2.1) and wires a ONE-offer pool at `_finish`:
the current winner is wrapped as a single `Offer` and run through
`select_offer`, which is a provable IDENTITY with one offer in the pool --
`name` MUST come out byte-identical. It converts 0 breadth by design; it is
the enabling structure for L3 (a second, systematic-floor offer) and L4
(RT-gated retry offers over the ranked pool).
"""
import pytest

from orthonym.namer import Orthonym
from orthonym.assembly.offer_pool import Offer

pytestmark = pytest.mark.unit


class TestOnePassthroughIsIdentity:
    def test_simple_alcohol_unchanged_and_offer_recorded(self):
        nm = Orthonym()
        assert nm.name("CCO") == "ethanol"
        offers = getattr(nm, "_offers", None)
        assert offers is not None
        assert len(offers) == 1
        assert isinstance(offers[0], Offer)
        assert offers[0].name == "ethanol"

    def test_second_molecule_byte_identical_to_pre_l2(self):
        # A ring/amide-bearing molecule -- acetanilide -- picked because it
        # exercises a different (composer/PIN) code path than a plain chain
        # alcohol, so the identity guarantee is not just proven on one shape.
        nm = Orthonym()
        assert nm.name("CC(=O)Nc1ccccc1") == "N-phenylacetamide"
        offers = nm._offers
        assert len(offers) == 1
        assert offers[0].name == "N-phenylacetamide"

    def test_offers_reset_per_top_level_molecule(self):
        nm = Orthonym()
        nm.name("CCO")
        assert nm._offers[0].name == "ethanol"
        nm.name("c1ccccc1")
        assert len(nm._offers) == 1
        assert nm._offers[0].name != "ethanol"

    def test_diverse_molecules_unchanged_with_and_without_offer_wiring(
            self, monkeypatch):
        # Prove the wiring is byte-identical by comparing against the SAME
        # molecules named with the coverage-audit machinery (and therefore the
        # offer-collection code inside it) switched off entirely.
        smiles_list = [
            "CCO", "c1ccccc1", "CC(=O)Nc1ccccc1", "CC(=O)O",
            "CC(C)(C)CC(=O)O", "CC(=O)OCC", "c1ccc2[nH]ccc2c1",
            "OCC(O)CO", "C1CC2CCC1CC2",
        ]
        nm_on = Orthonym()
        names_on = [nm_on.name(s) for s in smiles_list]
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "off")
        nm_off = Orthonym()
        names_off = [nm_off.name(s) for s in smiles_list]
        assert names_on == names_off


class TestNameTieredHonoursWinningOffer:
    """ a phase cleanup T1: `name_tiered` must label a row from the
    OFFER THAT ACTUALLY WON, not the process-wide provenance contextvar
    (which still describes whichever producer ran LAST -- the losing
    primary, on a floor win).

    The floor-win case is exercised with a monkeypatched ``Orthonym.name``
    that hand-builds ``self._offers``/``self._last_selected_offer`` exactly
    the way `_finish`/`_select_rt_passing_offer_name` do on a real floor
    win, matching the "independently testable with a hand-built pool"
    design `_select_rt_passing_offer_name`'s own docstring calls out --
    this decouples the test from any particular molecule's producer
    behaviour, which is what the pre-existing `test_t4_floor_offer.py`
    floor-win fixtures were NOT decoupled from (they went stale under an
    unrelated stereo-fabrication fix landed just before this task; see the
    cleanup report).
    """

    def test_normal_pin_case_unchanged(self):
        # Guard: the default single-offer/primary case must report the
        # SAME tier/is_pin/source as before this fix -- real machinery,
        # no monkeypatching.
        nm = Orthonym()
        row = nm.name_tiered("CCO")
        assert row["name"] == "ethanol"
        assert row["tier"] == "pin_verified"
        assert row["is_pin"] is True
        assert row["source"] == "pin_path"

    def test_floor_won_row_reports_t4_not_t1(self, monkeypatch):
        from orthonym.assembly.offer_pool import Offer

        floor_name = "2,3-dihydroxypropyl formate"  # arbitrary stand-in name

        def _fake_name(self, smiles):
            # Mirrors what `_finish` leaves behind after a REAL floor win:
            # a losing primary offer (pin_path/T1) and a winning floor
            # offer (t4_floor/T4) in `self._offers`, with
            # `_last_selected_offer` pointing at the winner -- but the
            # provenance CONTEXTVAR (never touched here) still reads as
            # whatever the PRIMARY last set (default "pin_path"), which is
            # the exact staleness this fix corrects for.
            primary = Offer(name="wrong-stereo-name", result_obj=None,
                             is_pin=True, tier="pin_verified", source="pin_path",
                             complete=True)
            floor = Offer(name=floor_name, result_obj=None, is_pin=False,
                          tier="best_effort", source="t4_floor", complete=True)
            self._offers = [primary, floor]
            self._last_selected_offer = floor
            return floor_name

        monkeypatch.setattr(Orthonym, "name", _fake_name)
        nm = Orthonym()
        row = nm.name_tiered("irrelevant-smiles")
        assert row["name"] == floor_name
        assert row["tier"] == "best_effort", row
        assert row["is_pin"] is False, row
        assert row["source"] == "t4_floor", row

    def test_no_winner_falls_back_to_contextvar_derivation(self, monkeypatch):
        """`_last_selected_offer` is `None` whenever `select_rt_passing`
        found no complete+RT-passing offer (or the offer machinery never
        ran at all, e.g. `_disable_opsin_validity_gate`) -- `name_tiered`
        must derive tier/is_pin/source the OLD way in that case, exactly
        as it always has."""
        def _fake_name(self, smiles):
            self._offers = []
            self._last_selected_offer = None
            return "ethanol"

        monkeypatch.setattr(Orthonym, "name", _fake_name)
        nm = Orthonym()
        row = nm.name_tiered("CCO")
        assert row["tier"] == "pin_verified"
        assert row["is_pin"] is True
        assert row["source"] == "pin_path"
