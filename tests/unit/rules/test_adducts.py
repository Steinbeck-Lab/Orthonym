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
        # NCC(=O)Nc1ccc(OCC)cc1 is unnameable at HEAD (probe 2026-07-09:
        # 'unknown organic compound') -> component naming must refuse,
        # never emit/propagate an unknown placeholder.
        from orthonym.rules.adducts import _name_component
        assert _name_component("NCC(=O)Nc1ccc(OCC)cc1", "pin") is None


class TestComponentOrdering:
    """P-14.8.1 -a: P-41 class seniority; P-14.8.2 -a buckets (internal
    oracle — OPSIN parses any component order, so ORDER is asserted here)."""

    @staticmethod
    def _ordered(smiles_list):
        from orthonym.rules.adducts import component_sort_key
        canon = [Chem.CanonSmiles(s) for s in smiles_list]
        return sorted(canon, key=component_sort_key)

    def test_alcohol_before_no_pcg_ring(self):
        # BB line 4661: 'ethanol—pyridine (1/1) (PIN)'
        assert self._ordered(["c1ccncc1", "CCO"]) == [
            Chem.CanonSmiles("CCO"), Chem.CanonSmiles("c1ccncc1")]

    def test_acid_before_amine(self):
        # BB line 4686: 'oxalic acid—ethane-1,2-diamine—water (2/2/3)'
        assert self._ordered(["NCCN", "OC(=O)C(=O)O"]) == [
            Chem.CanonSmiles("OC(=O)C(=O)O"), Chem.CanonSmiles("NCCN")]

    def test_no_pcg_tie_break_benzene_before_pyridine(self):
        # WAVE2-SCOPE-DECISIONS-RESOLVED decision 2: 'benzene—pyridine (1/1)'
        assert self._ordered(["c1ccncc1", "c1ccccc1"]) == [
            Chem.CanonSmiles("c1ccccc1"), Chem.CanonSmiles("c1ccncc1")]

    def test_organic_before_inorganic_acid(self):
        # BB line 4675: '...pentane-1,4-diamine—phosphoric acid (1/2)':
        # phosphoric acid carries a PCG rank (idx 23 < amine 84) but has NO
        # carbon -> inorganic bucket, cited AFTER every organic component.
        assert self._ordered(["OP(=O)(O)O", "NCCN"]) == [
            Chem.CanonSmiles("NCCN"), Chem.CanonSmiles("OP(=O)(O)O")]

    def test_organic_before_hydrogen_chloride(self):
        # BB line 4677 nicotine—hydrogen chloride pattern
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
        # BB line 4661 worked example
        assert self._name("CCO.c1ccncc1") == "ethanol—pyridine (1/1)"

    def test_bb_2_2_3_proportions(self):
        # BB line 4686: 'oxalic acid—ethane-1,2-diamine—water (2/2/3)'
        smi = "OC(=O)C(=O)O.OC(=O)C(=O)O.NCCN.NCCN.O.O.O"
        assert self._name(smi) == (
            "oxalic acid—ethane-1,2-diamine—water (2/2/3)")

    def test_any_unnameable_fragment_fails_closed(self):
        # glycinamide-aryl fragment unnameable at HEAD (probe 2026-07-09)
        assert self._name("O.NCC(=O)Nc1ccc(OCC)cc1") is None

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
