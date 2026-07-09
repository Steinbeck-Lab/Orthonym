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
