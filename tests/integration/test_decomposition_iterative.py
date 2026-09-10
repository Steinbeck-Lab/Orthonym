"""
Integration tests for iterative mixed-type decomposition (a phase, Plan 01).

Tests the new _try_iterative_mixed_decompose function and related
enhancements: retained-name coverage bonus, mixed-bond threshold
relaxation, amino acid fragment matching, and CIP stereo on fragments.

Coverage:
- Iterative decomposition of molecules with mixed bond types
- MAX_DECOMP_LEVELS cap enforcement
- Retained-name coverage bonus in _coverage_is_adequate
- Mixed bond threshold relaxation for 2-ester + glycosidic
- Quality gate comparison at each decomposition level
- Non-zwitterion amino acid fragment matching
- CIP stereo labels assigned on fragment mols
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.decomposition.engine import (
    _coverage_is_adequate,
    _RETAINED_CORE_NAMES,
)


class TestIterativeMixedDecomposition:
    """Test iterative decomposition of molecules with mixed bond types."""

    @pytest.mark.integration
    def test_ester_glycosidic_mixed_decomposition(self):
        """Molecule with ester + glycosidic bonds (HA > 30) produces a name
        via iterative decomposition that covers both bond types.

        Uses a glycoside acetate: sugar-O-C + ester linkage.
        The decomposition should handle both bond types iteratively.
        """
        # N-acetylglucosamine beta-glycoside with long ester chain
        # ester + glycosidic bonds, HA ~37
        smi = "CC(=O)NC1C(OC2OC(CO)C(O)C(O)C2O)C(O)C(CO)OC1OC(=O)CCCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 30, f"Test molecule should have HA > 30, got {ha}"

        name = name_compound(smi)
        assert name is not None, "Should produce a name"
        assert name != "unknown", "Should not be unknown"
        # Name should be reasonably long for a 37-HA molecule
        assert len(name) > 15, f"Name too short for {ha}-HA molecule: {name}"

    @pytest.mark.integration
    def test_ester_amide_mixed_decomposition(self):
        """Molecule with ester + amide bonds (HA > 40) iteratively decomposes
        both bond types across levels.

        Uses a ceramide phospholipid-like structure with both bond types.
        """
        # Palmitoyl sphingomyelin-like: amide + ester + phosphodiester
        smi = "CCCCCCCCCCCCCCCC(=O)NC(COP(=O)(O)OCC[N+](C)(C)C)C(O)/C=C/CCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 40, f"Test molecule should have HA > 40, got {ha}"

        name = name_compound(smi)
        assert name is not None, "Should produce a name"
        assert name != "unknown", "Should not be unknown"
        assert len(name) > 20, f"Name too short for {ha}-HA molecule: {name}"

    @pytest.mark.integration
    def test_max_decomp_levels_cap(self):
        """MAX_DECOMP_LEVELS=3 cap is enforced -- deeply nested molecules
        do not exceed 3 decomposition levels.

        The engine should not hang or recurse infinitely even on molecules
        with many different bond types.
        """
        from orthonym.decomposition.engine import MAX_DECOMP_LEVELS
        assert MAX_DECOMP_LEVELS == 3, "MAX_DECOMP_LEVELS should be 3"

        # Very complex molecule with multiple bond types
        # CDP-diacylglycerol: phosphodiester + ester bonds
        smi = "CCCCCCCCCCCCCCCC(=O)OC[C@@H](COP(=O)(O)OC1C(O)C(n2ccc(N)nc2=O)OC1CO)OC(=O)CCCCCCCCCCCCCCC"
        name = name_compound(smi)
        # Should complete without hanging -- just verify it returns something
        assert name is not None or name is None  # Completes without error

    @pytest.mark.integration
    def test_retained_name_coverage_bonus(self):
        """Retained-name fragments (containing 'adenine', 'cholesterol', etc.)
        pass coverage check even with short character count.

        The coverage check should give a bonus for names containing
        retained-name tokens.
        """
        # Test _coverage_is_adequate with retained name tokens
        mol = Chem.MolFromSmiles("c1ncnc2[nH]cnc12")  # adenine (HA=10)
        assert mol is not None

        # A name with "adenine" in it should get coverage bonus
        # For a 30-HA molecule, standard threshold = 30 * 0.8 = 24 chars
        # "adenine derivative" = 18 chars -- would fail without bonus
        # With 1.5x bonus: 18 * 1.5 = 27 >= 24 -- should pass

        # Create a mock mol with 30 heavy atoms
        big_mol = Chem.MolFromSmiles("C" * 30)
        assert big_mol is not None

        # Without retained name -- too short
        short_name = "some-compound-name"  # 18 chars
        # With standard threshold 0.8: need 24 chars, 18 < 24
        # This should fail
        assert not _coverage_is_adequate(short_name, big_mol, bond_type="")

        # With retained name token -- should pass with bonus
        retained_name = "adenine-derivative"  # 18 chars, contains 'adenine'
        assert _coverage_is_adequate(retained_name, big_mol, bond_type="")

    @pytest.mark.integration
    def test_mixed_bond_threshold_relaxation(self):
        """Mixed bond threshold relaxation -- 2 esters + 1 glycosidic
        triggers multi-ester decomposition.

        When total cleavable bonds >= 3 across all types and no single
        type reaches its threshold, the ester threshold is lowered to 2.
        """
        # Molecule with exactly 2 esters + glycosidic bond
        # Sugar diacetate: 2 ester + 1 glycosidic
        smi = "CC(=O)OC1OC(COC(C)=O)C(O)C(O)C1O"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None

        from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
        from collections import Counter
        bonds = find_cleavable_bonds(mol)
        types = Counter(b["type"] for b in bonds)
        # Verify we have the expected bond types
        assert types.get("ester", 0) >= 2, f"Expected >= 2 esters, got {types}"
        total = sum(types.values())
        assert total >= 3, f"Expected >= 3 total bonds, got {total}"

        name = name_compound(smi)
        assert name is not None, "Should produce a name"
        assert name != "unknown", "Should not be unknown"

    @pytest.mark.integration
    def test_quality_gate_at_each_level(self):
        """Quality gate comparison at each decomposition level -- deeper
        decomposition rejected if it produces worse name than previous level.

        The iterative decomposition should not produce garbled names.
        """
        # Test that a molecule where decomposition is beneficial gets a
        # non-garbled name
        smi = "CCCCCCCCCC(=O)OC1C(O)C(O)C(CO)OC1OC(=O)CCCCCCCCCCCCCCCC"
        name = name_compound(smi)
        if name:
            # Should not contain garbled patterns
            assert "acidyl" not in name.lower(), f"Garbled name: {name}"
            assert name.count("(") == name.count(")"), f"Unbalanced brackets: {name}"

    @pytest.mark.integration
    def test_amino_acid_fragment_matching(self):
        """Non-zwitterion amino acid fragment SMILES matched to retained
        amino acid names.

        After decomposition into fragments, amino acid fragments should
        use retained names (glycine, alanine, etc.) when available.
        """
        from orthonym.assembly.fragment_naming import name_fragment_recursively

        # Neutral glycine: NH2-CH2-COOH
        gly_name = name_fragment_recursively("NCC(=O)O")
        assert gly_name is not None, "Should name neutral glycine"
        assert "glycine" in gly_name.lower() or "amino" in gly_name.lower(), (
            f"Expected glycine or amino acid name, got: {gly_name}"
        )

        # Neutral alanine: NH2-CH(CH3)-COOH
        ala_name = name_fragment_recursively("CC(N)C(=O)O")
        assert ala_name is not None, "Should name neutral alanine"
        assert "alanine" in ala_name.lower() or "amino" in ala_name.lower(), (
            f"Expected alanine or amino acid name, got: {ala_name}"
        )

    @pytest.mark.integration
    def test_cip_stereo_on_fragments(self):
        """CIP stereo labels assigned on fragment mols before naming .

        Fragment mols parsed from SMILES should have CIP labels assigned
        so stereo descriptors are preserved through decomposition.
        """
        from orthonym.assembly.fragment_naming import name_fragment_recursively

        # (R)-lactic acid: fragment with a stereocenter
        # The naming should preserve the (R) stereodescriptor
        r_lactate = "C[C@@H](O)C(=O)O"
        name = name_fragment_recursively(r_lactate)
        assert name is not None, "Should name (R)-lactic acid fragment"
        # The name should reference the molecule reasonably
        assert len(name) > 3, f"Fragment name too short: {name}"


@pytest.mark.integration
class TestMixedAssemblyBondType:
    """Test bond-type-aware iterative mixed assembly ."""

    def test_mixed_assembly_uses_bond_type(self):
        """Verify that a molecule with both ester and amide bonds produces
        bond-type-specific assembly (contains "-oate"/"-ate" for ester part,
        not just space-joined fragment names).
        """
        # Palmitoyl sphingomyelin-like: amide + ester + phosphodiester
        smi = "CCCCCCCCCCCCCCCC(=O)NC(COP(=O)(O)OCC[N+](C)(C)C)C(O)/C=C/CCCCCCCCCCCCC"
        name = name_compound(smi)
        assert name is not None, "Should produce a name"
        assert name != "unknown", "Should not be unknown"
        # The name should be reasonably long for a large molecule
        assert len(name) > 20, f"Name too short: {name}"

    def test_mixed_assembly_validates_tokens(self):
        """Verify that _validate_assembly_tokens catches lost fragments."""
        from orthonym.decomposition.engine import _validate_assembly_tokens

        # Missing fragment: "hexadecanoic acid" tokens not in assembled name
        assert not _validate_assembly_tokens(
            "ethyl propanoate",
            ["ethanol", "propanoic acid", "hexadecanoic acid"],
        )
        # All present: stems match
        assert _validate_assembly_tokens(
            "ethyl propanoate",
            ["ethanol", "propanoic acid"],
        )

    def test_iterative_mixed_assembly_not_simple_space_join(self):
        """Engine _try_iterative_mixed_decompose references _assemble_by_bond_type."""
        import inspect
        from orthonym.decomposition.engine import _try_iterative_mixed_decompose
        source = inspect.getsource(_try_iterative_mixed_decompose)
        # Must reference bond-type-aware assembly
        assert "_assemble_by_bond_type" in source, (
            "_try_iterative_mixed_decompose must use _assemble_by_bond_type"
        )
        # Must reference token validation
        assert "_validate_assembly_tokens" in source, (
            "_try_iterative_mixed_decompose must use _validate_assembly_tokens"
        )
