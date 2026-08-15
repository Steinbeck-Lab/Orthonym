# tests/unit/test_offer_at_finish.py
"""v33 Phase 0 Task L2.2: `select_offer` passthrough wired at `namer.py::_finish`.

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
