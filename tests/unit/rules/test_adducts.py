"""Unit tests for the P-14.8 adduct/hydrate assembler (Wave-2 P0A).

Blue Book P-14.8.1: components joined by em-dash (U+2014) with parenthesized
solidus proportions; P-14.8.2: organic first, inorganic next, water last.
All expected names OPSIN-2.9.0 verified 2026-07-09 (plan doc, Oracle table).
"""
import pytest
from rdkit import Chem

pytestmark = pytest.mark.unit


class TestSplitComponents:
    def test_dedupes_identical_fragments_with_counts(self):
        from orthonym.rules.adducts import split_components
        mol = Chem.MolFromSmiles("O.O.OC(=O)C(=O)O")
        comps = split_components(mol)
        assert comps is not None
        d = dict(comps)
        assert d[Chem.CanonSmiles("O")] == 2
        assert d[Chem.CanonSmiles("OC(=O)C(=O)O")] == 1
        assert len(comps) == 2  # two DISTINCT components

    def test_single_fragment_returns_none(self):
        from orthonym.rules.adducts import split_components
        assert split_components(Chem.MolFromSmiles("CCO")) is None

    def test_223_proportions(self):
        from orthonym.rules.adducts import split_components
        mol = Chem.MolFromSmiles(
            "OC(=O)C(=O)O.OC(=O)C(=O)O.NCCN.NCCN.O.O.O")
        comps = dict(split_components(mol))
        assert comps[Chem.CanonSmiles("OC(=O)C(=O)O")] == 2
        assert comps[Chem.CanonSmiles("NCCN")] == 2
        assert comps[Chem.CanonSmiles("O")] == 3


class TestNameComponent:
    def test_multiatom_fragment_names_via_pipeline(self):
        from orthonym.rules.adducts import _name_component
        assert _name_component("CC(N)=O", "pin") == "acetamide"
        assert _name_component("OC(=O)C(=O)O", "pin") == "oxalic acid"

    def test_single_atom_table(self):
        from orthonym.rules.adducts import _name_component
        assert _name_component("O", "pin") == "water"
        assert _name_component("Cl", "pin") == "hydrogen chloride"

    def test_unrecognized_single_atom_fails_closed(self):
        # bare metal atoms are NOT adduct components (organometallics
        # own them at dispatch priority 50)
        from orthonym.rules.adducts import _name_component
        assert _name_component("[Fe]", "pin") is None

    def test_unnameable_fragment_fails_closed(self):
        # A fragment the PIN namer cannot build -> component naming must refuse,
        # never emit/propagate an unknown placeholder. (Fixture updated 2026-09-03:
        # the old NCC(=O)Nc1ccc(OCC)cc1 now names as the engine improved; this
        # silabicyclic ring still abstains at PIN. Do NOT re-pin to the old name --
        # its emitted spelling 'ethanamide' is itself a spelling-layer defect,
        # should be 'acetamide' per BB P-66.1.1.1.)
        from orthonym.rules.adducts import _name_component
        assert _name_component("c1ccc2c(c1)[SiH2]cc2", "pin") is None


class TestComponentOrdering:
    """P-14.8.1 -a: P-41 class seniority; P-14.8.2 -a buckets (internal
    oracle — OPSIN parses any component order, so ORDER is asserted here)."""

    @staticmethod
    def _ordered(smiles_list):
        from orthonym.rules.adducts import component_sort_key
        canon = [Chem.CanonSmiles(s) for s in smiles_list]
        return sorted(canon, key=component_sort_key)

    def test_alcohol_before_no_pcg_ring(self):
        # the Blue Book: 'ethanol—pyridine (1/1) (PIN)'
        assert self._ordered(["c1ccncc1", "CCO"]) == [
            Chem.CanonSmiles("CCO"), Chem.CanonSmiles("c1ccncc1")]

    def test_acid_before_amine(self):
        # the Blue Book: 'oxalic acid—ethane-1,2-diamine—water (2/2/3)'
        assert self._ordered(["NCCN", "OC(=O)C(=O)O"]) == [
            Chem.CanonSmiles("OC(=O)C(=O)O"), Chem.CanonSmiles("NCCN")]

    def test_no_pcg_tie_break_benzene_before_pyridine(self):
        # WAVE2-SCOPE-DECISIONS-RESOLVED decision 2: 'benzene—pyridine (1/1)'
        assert self._ordered(["c1ccncc1", "c1ccccc1"]) == [
            Chem.CanonSmiles("c1ccccc1"), Chem.CanonSmiles("c1ccncc1")]

    def test_organic_before_inorganic_acid(self):
        # the Blue Book: '...pentane-1,4-diamine—phosphoric acid (1/2)':
        # phosphoric acid carries a PCG rank (idx 23 < amine 84) but has NO
        # carbon -> inorganic bucket, cited AFTER every organic component.
        assert self._ordered(["OP(=O)(O)O", "NCCN"]) == [
            Chem.CanonSmiles("NCCN"), Chem.CanonSmiles("OP(=O)(O)O")]

    def test_organic_before_hydrogen_chloride(self):
        # the Blue Book nicotine—hydrogen chloride pattern
        assert self._ordered(["Cl", "CN1CCCC1c1cccnc1"]) == [
            Chem.CanonSmiles("CN1CCCC1c1cccnc1"), Chem.CanonSmiles("Cl")]

    def test_water_cited_last(self):
        # P-14.8.2 line 4665: 'water (if present), is cited last' — even
        # after other inorganics.
        assert self._ordered(["O", "Cl", "CCO"]) == [
            Chem.CanonSmiles("CCO"), Chem.CanonSmiles("Cl"),
            Chem.CanonSmiles("O")]


class TestNameAdduct:
    """P-14.8.1 / -b / -c + P-14.8.2 -b: em-dash + (n/m/...) assembly.
    Every expected name OPSIN-2.9.0 verified (plan Oracle table)."""

    @staticmethod
    def _name(smi, style="pin"):
        from orthonym.rules.adducts import name_adduct
        return name_adduct(Chem.MolFromSmiles(smi), style=style)

    def test_hydrate_1_1(self):
        assert self._name("O.OC(=O)C(=O)O") == "oxalic acid—water (1/1)"

    def test_hydrate_1_2(self):
        assert self._name("O.O.OC(=O)C(=O)O") == "oxalic acid—water (1/2)"

    def test_organic_pair_em_dash(self):
        # P-14.8.1 -b: em-dash U+2014, never hyphen
        name = self._name("c1ccccc1.c1ccncc1")
        assert name == "benzene—pyridine (1/1)"
        assert "—" in name and " (1/1)" in name

    def test_bb_ethanol_pyridine(self):
        # the Blue Book worked example
        assert self._name("CCO.c1ccncc1") == "ethanol—pyridine (1/1)"

    def test_bb_2_2_3_proportions(self):
        # the Blue Book: 'oxalic acid—ethane-1,2-diamine—water (2/2/3)'
        smi = "OC(=O)C(=O)O.OC(=O)C(=O)O.NCCN.NCCN.O.O.O"
        assert self._name(smi) == (
            "oxalic acid—ethane-1,2-diamine—water (2/2/3)")

    def test_any_unnameable_fragment_fails_closed(self):
        # a fragment the PIN namer cannot build -> whole adduct fails closed
        # (fixture updated 2026-09-03; see test_unnameable_fragment_fails_closed)
        assert self._name("O.c1ccc2c(c1)[SiH2]cc2") is None

    def test_all_identical_fragments_decline(self):
        # >=2 DISTINCT components required (adducts are combinations of
        # SEPARATE molecular entities); identical-only sets keep the
        # frozen legacy space-join in the dispatch handler (Task 4).
        assert self._name("CCO.OCC") is None

    def test_unrecognized_single_atom_declines(self):
        # bare-metal fragment -> not an adduct (organometallics@50 own it)
        assert self._name("[Ni].C=CC.C=CC") is None

    def test_single_component_declines(self):
        assert self._name("CCO") is None


class TestAdductDispatch:
    """End-to-end via name_compound with the OPSIN validity gate FORCE-ENABLED
    (production semantics).

    These are production-faithful acceptance tests. The unit suite disables
    the module-wide OPSIN validity gate by default (conftest
    _disable_opsin_validity_gate_for_tests), but for the adduct dispatch the
    gate is load-bearing: (a) an OPSIN-unparseable single-fragment name (e.g.
    the HEAD name '2-amino-1-anilinoethanamide' for the aryl-glycinamide) must
    resolve to 'unknown organic compound', not leak; (b) the organometallic
    '[Ni].C=CC.C=CC' names to the OPSIN-unparseable 'bis(η3-...)nickel' which
    the gate turns into 'nickel compound (not supported)'. So we re-enable the
    gate here exactly as test_opsin_validity_gate.py does — this is the true
    production path that scripts/diagnose.py exercises."""

    @pytest.fixture(autouse=True)
    def _force_enable_gate(self, monkeypatch):
        import orthonym.namer as _namer
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False)

    @staticmethod
    def _nc(smi, style="pin"):
        from orthonym.namer import name_compound
        return name_compound(smi, style=style)

    def test_oxalic_monohydrate_pin(self):
        # was 'unknown organic compound' at HEAD (predicate required >=2
        # multi-atom fragments; water has 1 heavy atom)
        assert self._nc("O.OC(=O)C(=O)O") == "oxalic acid—water (1/1)"

    def test_benzene_pyridine_flips_to_adduct_pin(self):
        # was the space-join 'benzene pyridine' at HEAD
        assert self._nc("c1ccccc1.c1ccncc1") == "benzene—pyridine (1/1)"

    def test_mixed_hydrochloride(self):
        # P-14.8.2 pattern (the Blue Book). DIVERGENCE from the plan's stale
        # expected: at this HEAD the organic fragment names to the RETAINED
        # name 'nicotine' (not '3-(1-methylpyrrolidin-2-yl)pyridine'); the
        # resulting 'nicotine—hydrogen chloride (1/1)' OPSIN-RTs cleanly to
        # Cl.N1=CC(C2N(C)CCC2)=CC=C1 (== input). Verified 2026-07-09.
        assert self._nc("Cl.CN1CCCC1c1cccnc1") == (
            "nicotine—hydrogen chloride (1/1)")

    def test_mixed_phosphoric_1_2(self):
        # P-14.8.2 -a/-b (the Blue Book pattern; OPSIN verified)
        assert self._nc("OP(=O)(O)O.OP(=O)(O)O.NCCN") == (
            "ethane-1,2-diamine—phosphoric acid (1/2)")

    def test_mixed_three_buckets_water_last(self):
        # organic -> inorganic -> water (P-14.8.2 -a; OPSIN verified)
        assert self._nc("O.Cl.CCO") == (
            "ethanol—hydrogen chloride—water (1/1/1)")

    def test_salt_routing_protected(self):
        # charged multi-fragment stays with the ions/salt router
        assert self._nc("[Na+].[Cl-]") == "sodium chloride"

    def test_organometallic_not_swallowed(self):
        # (item 4, FABLE -P1 review I3): the ORGANOMETALLIC handler used to
        # name [Ni].C=CC.C=CC as bis(η³-prop-2-en-1-yl)nickel (C6H10Ni) for a
        # C6H12Ni input -- data/organometallics.py maps neutral propene 'C=CC' to
        # the η³-allyl ligand (C3H5, drops 1 H per ligand) and the carve-out
        # (_ORGANOMETALLIC_ADDITIVE_PIN_RE) shipped it bypassing OPSIN. The
        # atom-conservation veto (rules.organometallics._organometallic_conserves)
        # now declines it -> the cascade abstains (organometallics are out of
        # scope, P-69). ORGANOMETALLIC@50 owns bare-metal dot-SMILES; the adduct
        # predicate also declines ([Ni] not in the single-atom table).
        assert self._nc("[Ni].C=CC.C=CC") == "nickel compound (not supported)"

    def test_identical_fragments_keep_frozen_space_join(self):
        # Plan-01 byte-identical representative
        # (tests/unit/routing/test_dispatcher.py:61)
        assert self._nc("CCO.OCC") == "ethanol ethanol"

    def test_unnameable_distinct_set_fails_closed(self):
        # was a structure-dropping hazard: the legacy handler skipped
        # unnameable fragments and joined the rest (fixture updated 2026-09-03;
        # see test_unnameable_fragment_fails_closed)
        assert self._nc("O.c1ccc2c(c1)[SiH2]cc2") == "unknown organic compound"


class TestHydrateWordForms:
    """P-14.8.1 -e / P-14.8.2 -c: general-nomenclature hydrate word forms.
    OPSIN parses every emitted form (plan Oracle table)."""

    @staticmethod
    def _nc(smi, style="general"):
        from orthonym.namer import name_compound
        return name_compound(smi, style=style)

    def test_monohydrate(self):
        assert self._nc("O.OC(=O)C(=O)O") == "oxalic acid monohydrate"

    def test_dihydrate(self):
        assert self._nc("O.O.OC(=O)C(=O)O") == "oxalic acid dihydrate"

    def test_trihydrate(self):
        assert self._nc("O.O.O.OC(=O)C(=O)O") == "oxalic acid trihydrate"

    def test_hemihydrate(self):
        # water:parent = 1/2 -> hemi (line 4657)
        assert self._nc("O.OC(=O)C(=O)O.OC(=O)C(=O)O") == (
            "oxalic acid hemihydrate")

    def test_sesquihydrate_multi_parent(self):
        # the Blue Book: 'oxalic acid—ethane-1,2-diamine sesquihydrate'
        # (2/2/3): non-water components em-dash joined WITHOUT proportions
        smi = "OC(=O)C(=O)O.OC(=O)C(=O)O.NCCN.NCCN.O.O.O"
        assert self._nc(smi) == (
            "oxalic acid—ethane-1,2-diamine sesquihydrate")

    def test_mixed_inorganic_monohydrate(self):
        # P-14.8.2 -c (OPSIN verified)
        assert self._nc("O.Cl.CCO") == (
            "ethanol—hydrogen chloride monohydrate")

    def test_pin_style_unchanged(self):
        # P-14.8: PINs MUST use the proportion notation
        assert self._nc("O.OC(=O)C(=O)O", style="pin") == (
            "oxalic acid—water (1/1)")

    def test_general_without_water_uses_adduct_form(self):
        assert self._nc("c1ccccc1.c1ccncc1") == "benzene—pyridine (1/1)"

    def test_unequal_parent_counts_fall_back_to_proportions(self):
        # 2 oxalic + 1 pyridine + 1 water: no single parent count -> the
        # word form is undefined; fall back to the always-valid PIN form
        smi = "OC(=O)C(=O)O.OC(=O)C(=O)O.c1ccncc1.O"
        name = self._nc(smi)
        assert name.endswith("(2/1/1)") and "hydrate" not in name
