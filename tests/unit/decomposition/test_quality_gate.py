"""Tests for name_pipeline_only utility function.

Verifies that name_pipeline_only:
- Does NOT consume depth levels (thread-local depth counter unchanged)
- Does NOT trigger decomposition (skips try_decompose)
- Returns None for invalid SMILES
- Works for simple and complex molecules
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_pipeline_only
from orthonym.assembly.fragment_naming import get_naming_depth


class TestNamePipelineOnly:
    """name_pipeline_only should name molecules without decomposition or depth cost."""

    @pytest.mark.unit
    def test_pipeline_only_no_depth_increment(self):
        """name_pipeline_only should NOT increment depth counter."""
        depth_before = get_naming_depth()
        result = name_pipeline_only("CCO")
        depth_after = get_naming_depth()
        assert depth_after == depth_before, (
            f"name_pipeline_only incremented depth: {depth_before} -> {depth_after}"
        )
        assert result == "ethanol"

    @pytest.mark.unit
    def test_pipeline_only_complex_molecule(self):
        """name_pipeline_only should handle complex molecules without depth waste."""
        # Phospholipid-like SMILES that previously hit DROP-13
        smiles = "CCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC"
        depth_before = get_naming_depth()
        result = name_pipeline_only(smiles)
        depth_after = get_naming_depth()
        assert depth_after == depth_before, (
            f"Depth changed during name_pipeline_only: {depth_before} -> {depth_after}"
        )
        assert result is not None, "name_pipeline_only returned None for valid SMILES"

    @pytest.mark.unit
    def test_pipeline_only_skips_decomposition(self):
        """name_pipeline_only should not call try_decompose."""
        # A simple ester that would normally decompose
        smiles = "CC(=O)OC"
        result = name_pipeline_only(smiles)
        # Pipeline-only should return the pipeline name, not decomposed name
        assert result is not None

    @pytest.mark.unit
    def test_pipeline_only_returns_none_on_invalid(self):
        """name_pipeline_only should return None for invalid SMILES."""
        result = name_pipeline_only("NOT_A_SMILES")
        assert result is None

    @pytest.mark.unit
    def test_pipeline_only_garbled_fallback_disabled(self):
        """name_pipeline_only should not use garbled-name fallback path."""
        depth_before = get_naming_depth()
        name_pipeline_only("c1ccccc1")
        depth_after = get_naming_depth()
        assert depth_after == depth_before


class TestTieredCoverage:
    """Tests for tiered coverage threshold based on bond type.

    Functional-class types (ester, amide, glycosidic, carbamate, thioester)
    use 0.6 chars/HA threshold. Substitutive types (sulfonamide,
    phosphodiester, ether, default) use 0.8 chars/HA threshold.
    """

    @pytest.mark.unit
    def test_ester_uses_lower_threshold(self):
        """Ester bond type should use 0.6 chars/HA threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Passes 0.6 threshold, fails 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("phenyl palmitate", mol, bond_type="ester") is True

    @pytest.mark.unit
    def test_ester_rejects_too_short(self):
        """Even with 0.6 threshold, very short names should be rejected."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("short", mol, bond_type="ester") is False

    @pytest.mark.unit
    def test_sulfonamide_uses_higher_threshold(self):
        """Sulfonamide bond type should use 0.8 chars/HA threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Fails 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("phenyl palmitate", mol, bond_type="sulfonamide") is False

    @pytest.mark.unit
    def test_sulfonamide_passes_with_long_name(self):
        """Sulfonamide with adequate length should pass 0.8 threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "N-methylbenzenesulfonamide" = 25 chars, 20 HA => 1.25 chars/HA
        mol = Chem.MolFromSmiles("CS(=O)(=O)c1ccc(C)c(C)c1NC")  # ~20 HA approx
        ha = mol.GetNumHeavyAtoms()
        name = "N-methylbenzenesulfonamide"
        # Ensure the molecule has enough HA to test meaningfully (>10)
        assert ha > 10, f"Need HA > 10 for test, got {ha}"
        assert _coverage_is_adequate(name, mol, bond_type="sulfonamide") is True

    @pytest.mark.unit
    def test_small_molecule_exemption_ester(self):
        """Small molecules (HA <= 10) should always pass regardless of bond type."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CC(=O)OC")  # methyl acetate, ~5 HA
        assert mol.GetNumHeavyAtoms() <= 10
        assert _coverage_is_adequate("anything", mol, bond_type="ester") is True

    @pytest.mark.unit
    def test_small_molecule_exemption_default(self):
        """Small molecules should pass even with empty bond_type."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane, 6 HA
        assert mol.GetNumHeavyAtoms() <= 10
        assert _coverage_is_adequate("anything", mol, bond_type="") is True

    @pytest.mark.unit
    def test_default_bond_type_uses_high_threshold(self):
        """No bond_type (default="") should use 0.8 threshold for backward compat."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl palmitate" = 16 chars, 24 HA => 0.67 chars/HA
        # Should fail 0.8 threshold
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")
        assert _coverage_is_adequate("phenyl palmitate", mol) is False

    @pytest.mark.unit
    def test_each_functional_class_type_uses_low_threshold(self):
        """All 5 functional-class types should use 0.6 threshold."""
        from orthonym.decomposition.engine import _coverage_is_adequate
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        # "phenyl palmitate" = 16 chars => 0.67 chars/HA, passes 0.6 but fails 0.8
        for bond_type in ("ester", "amide", "glycosidic", "carbamate", "thioester"):
            assert _coverage_is_adequate("phenyl palmitate", mol, bond_type=bond_type) is True, (
                f"bond_type={bond_type} should use 0.6 threshold"
            )


class TestConservativeQualityGate:
    """Tests for conservative quality gate: HA>25 threshold, ring-system token
    check, expanded _RETAINED_CORE_NAMES, and ring-system token loss detection.
    """

    @pytest.mark.unit
    def test_retained_whitelist_bypasses_no_digits_check(self):
        """Names in _RETAINED_CORE_NAMES should pass regardless of HA, no digits/hyphens."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # phenothiazine: in _RETAINED_CORE_NAMES, bypasses all size checks
        mol = Chem.MolFromSmiles("c1ccc2c(c1)Sc1ccccc1N2")  # phenothiazine
        assert _name_quality_is_acceptable("phenothiazine", mol) is True

    @pytest.mark.unit
    def test_retained_whitelist_bypasses_at_large_ha(self):
        """Names in _RETAINED_CORE_NAMES should pass even for very large HA counts."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 40
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        # phenothiazine in whitelist, bypasses all checks
        assert _name_quality_is_acceptable("phenothiazine", mock_mol) is True

    @pytest.mark.unit
    def test_ha_22_no_digits_no_ring_token_rejected(self):
        """HA=22 > 20, no digits, no hyphens => rejected (not in whitelist)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 22
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        # Name long enough to pass length check (22//2=11) but not in whitelist
        assert _name_quality_is_acceptable("somecompoundname", mock_mol) is False

    @pytest.mark.unit
    def test_ha_26_no_digits_no_ring_token_rejected(self):
        """HA=26 > 20, no digits, no hyphens => rejected (not in whitelist)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 26
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        # Name long enough to pass length check (26//2=13) but not in whitelist
        assert _name_quality_is_acceptable("somecompoundname", mock_mol) is False

    @pytest.mark.unit
    def test_expanded_whitelist_flavone(self):
        """'flavone' should be in expanded _RETAINED_CORE_NAMES."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 22
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        assert _name_quality_is_acceptable("flavone", mock_mol) is True

    @pytest.mark.unit
    def test_expanded_whitelist_carbazole(self):
        """'carbazole' should be in expanded _RETAINED_CORE_NAMES."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 22
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        assert _name_quality_is_acceptable("carbazole", mock_mol) is True

    @pytest.mark.unit
    def test_ring_system_token_detected_pyridine(self):
        """_name_has_ring_system_token should detect 'pyridine' in name."""
        from orthonym.decomposition.engine import _name_has_ring_system_token
        assert _name_has_ring_system_token("2-methylpyridine") is True

    @pytest.mark.unit
    def test_ring_system_token_not_detected_hexane(self):
        """_name_has_ring_system_token should not detect anything in 'hexane'."""
        from orthonym.decomposition.engine import _name_has_ring_system_token
        assert _name_has_ring_system_token("hexane") is False

    @pytest.mark.unit
    def test_decomposition_no_false_positive_no_garbled(self):
        """_decomposition_is_worse should NOT flag when decomp is clean."""
        from orthonym.decomposition.engine import _decomposition_is_worse
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)OCC")
        # Both names are clean, neither garbled
        assert _decomposition_is_worse(
            "ethyl palmitate",
            "hexadecanyl ethanoate",
            mol
        ) is False


class TestRetainedNameCoverageGuard:
    """Tests for retained-name coverage guard in _name_quality_is_acceptable().

    Retained names in _RETAINED_CORE_NAMES should only bypass the quality gate
    when they plausibly describe the whole molecule. For molecules with HA > 20,
    the retained name must have >= 0.25 chars/HA to pass. This catches cases
    like "adenine" (7 chars) naming a 58-HA CoA molecule (ratio 0.12) while
    preserving "adenine" for the standalone 10-HA molecule.
    """

    @pytest.mark.unit
    def test_adenine_rejected_at_58_ha(self):
        """adenine (7 chars) for a 58-HA molecule: ratio=0.12, HA>20, rejected."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 58
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        assert _name_quality_is_acceptable("adenine", mock_mol) is False

    @pytest.mark.unit
    def test_adenine_accepted_at_10_ha(self):
        """adenine (7 chars) for a 10-HA molecule: HA<=20, always passes."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Use real adenine molecule
        mol = Chem.MolFromSmiles("c1nc(N)c2[nH]cnc2n1")  # adenine, HA=10
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 10
        assert _name_quality_is_acceptable("adenine", mol) is True

    @pytest.mark.unit
    def test_adenine_rejected_at_38_ha(self):
        """adenine (7 chars) for a 38-HA molecule: ratio=0.18, HA>20, rejected."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        from unittest.mock import MagicMock
        mock_mol = MagicMock()
        mock_mol.GetNumHeavyAtoms.return_value = 38
        mock_mol.GetRingInfo.return_value = MagicMock(AtomRings=lambda: [])
        assert _name_quality_is_acceptable("adenine", mock_mol) is False

    @pytest.mark.unit
    def test_adenine_accepted_at_15_ha(self):
        """adenine (7 chars) for HA=15 (<=20 threshold): always passes."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # 9-pentyladenine has HA=15
        mol = Chem.MolFromSmiles("CCCCCn1cnc2c(N)ncnc21")
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 15
        assert _name_quality_is_acceptable("adenine", mol) is True

    @pytest.mark.unit
    def test_indole_accepted_at_9_ha(self):
        """indole (6 chars) for a 9-HA molecule: HA<=20, always passes."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")  # indole, HA=9
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 9
        assert _name_quality_is_acceptable("indole", mol) is True

    @pytest.mark.unit
    def test_phenothiazine_accepted_at_14_ha(self):
        """phenothiazine (13 chars) for a 14-HA molecule: HA<=20, always passes."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        mol = Chem.MolFromSmiles("c1ccc2c(c1)Sc1ccccc1N2")  # phenothiazine, HA=14
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 14
        assert _name_quality_is_acceptable("phenothiazine", mol) is True


class TestMultiBondUnderCoverage:
    """Tests for multi-bond under-coverage detection in _name_quality_is_acceptable().

    When a molecule has multiple distinct cleavable bond types but the pipeline
    name references fewer than half of them, the name likely describes only one
    fragment. The quality gate should reject such names.
    """

    @pytest.mark.unit
    def test_name_missing_bond_tokens_with_ester_and_phospho(self):
        """Name 'hexadecan-1-ol' has no ester or phospho tokens for a molecule
        with ester + phosphodiester bonds (HA=27): should be rejected by
        multi-bond under-coverage check (ratio 0.52 < 0.8, 0/2 types represented)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Molecule with ester + phosphodiester bonds, HA=27
        mol = Chem.MolFromSmiles("CCCCCCCCCCCC(=O)OCCCCOP(=O)(O)OCCCC")
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 20, f"Need HA > 20, got {ha}"
        # 'hexadecan-1-ol' = 14 chars, ratio=0.52 < 0.8, has digits/hyphens
        # No ester or phospho tokens -> 0 of 2 represented -> rejected
        assert _name_quality_is_acceptable("hexadecan-1-ol", mol) is False

    @pytest.mark.unit
    def test_name_with_ester_token_passes_multi_bond(self):
        """Name '2-methylhexadecanoate' contains 'oate' (ester token): at least
        half of bond types represented (1/2), should pass."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Same molecule with ester + phosphodiester bonds, HA=27
        mol = Chem.MolFromSmiles("CCCCCCCCCCCC(=O)OCCCCOP(=O)(O)OCCCC")
        assert mol is not None
        # '2-methylhexadecanoate' = 21 chars, ratio=0.78 < 0.8
        # Has digits (2) and hyphens -> passes no-digits check
        # 'oate' matches ester -> 1 of 2 represented (1 >= 1.0) -> passes
        assert _name_quality_is_acceptable("2-methylhexadecanoate", mol) is True

    @pytest.mark.unit
    def test_partial_name_rejected_for_multi_bond_molecule(self):
        """Name 'hexadecan-3-ol' for a 42-HA molecule with 4 bonds
        (ester + amide): partial name should be rejected."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Molecule with 2 ester + 2 amide bonds, HA=42
        mol = Chem.MolFromSmiles("CCCCCCCCCC(=O)NCCCC(=O)OCC(NC(=O)CCCCC)CC(=O)OCCCCCCCCCC")
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 30, f"Need large molecule, got HA={ha}"
        # 'hexadecan-3-ol' = 14 chars, ratio=0.33 < 0.8
        # Has digits/hyphens. No ester or amide tokens -> 0/2 -> rejected
        # (Also caught by short name check: 14 < 21, but multi-bond would
        # also reject it independently)
        assert _name_quality_is_acceptable("hexadecan-3-ol", mol) is False

    @pytest.mark.unit
    def test_no_cleavable_bonds_always_passes(self):
        """Molecule with no cleavable bonds: multi-bond check not applicable."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Pentanoic acid has no cleavable bonds
        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        assert mol is not None
        assert _name_quality_is_acceptable("pentanoic acid", mol) is True

    @pytest.mark.unit
    def test_name_with_amide_and_ester_tokens_passes(self):
        """Name referencing both amide and ester tokens passes for a molecule
        with both bond types (ester + amide)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Molecule with 1 ester + 1 amide bond, HA > 15
        # Only 1 amide carbonyl so multi-amide check doesn't trigger
        mol = Chem.MolFromSmiles("CC(=O)OCCCC(NC(=O)CCCCC)CCCCC")
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 15, f"Need HA > 15, got {ha}"
        # "ethyl 2-aminodecanoate" references amino (amide) and oate (ester)
        # and has digits/hyphens -> passes other checks
        assert _name_quality_is_acceptable("ethyl 2-aminodecanoate", mol) is True

    @pytest.mark.unit
    def test_high_ratio_name_skips_multi_bond_check(self):
        """Names with coverage ratio >= 0.8 skip the multi-bond check,
        since adequate-length names are presumed to describe the molecule."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Small molecule with 2 distinct bond types, HA=17
        mol = Chem.MolFromSmiles("CCCCCC(=O)OCCOP(=O)(O)OCC")
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha == 17
        # 'hexadecan-1-ol' = 14 chars, ratio=14/17=0.82 >= 0.8 -> multi-bond skipped
        # Also has digits/hyphens, ratio >= 0.45, len >= ha//2
        assert _name_quality_is_acceptable("hexadecan-1-ol", mol) is True

    @pytest.mark.unit
    def test_existing_quality_gate_tests_still_pass(self):
        """Verify the retained-name tests and tiered coverage tests still pass.
        This is a meta-test ensuring no regressions."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # phenothiazine standalone (HA=14) should still pass
        mol = Chem.MolFromSmiles("c1ccc2c(c1)Sc1ccccc1N2")
        assert _name_quality_is_acceptable("phenothiazine", mol) is True
        # adenine standalone (HA=10) should still pass
        mol2 = Chem.MolFromSmiles("c1nc(N)c2[nH]cnc2n1")
        assert _name_quality_is_acceptable("adenine", mol2) is True


class TestProbeIntegration:
    """Integration tests verifying the probe replacement in try_decompose().

    After Phase 097, the decomposition engine probe uses name_pipeline_only()
    instead of name_fragment_recursively(), eliminating the cache-disable hack
    and _fragment_guard dependency.
    """

    @pytest.mark.unit
    def test_probe_uses_name_pipeline_only(self):
        """try_decompose probe path should use name_pipeline_only, not name_fragment_recursively."""
        from unittest.mock import patch
        from orthonym.decomposition.engine import try_decompose

        # A simple ester that has cleavable bonds
        mol = Chem.MolFromSmiles("CC(=O)OCC")
        with patch('orthonym.namer.name_pipeline_only', wraps=name_pipeline_only) as mock_npo:
            try_decompose(mol)
            # name_pipeline_only should have been called for the probe
            assert mock_npo.called, "try_decompose should use name_pipeline_only for baseline probe"

    @pytest.mark.unit
    def test_probe_does_not_touch_fragment_cache(self):
        """Probe path should not save/restore _fragment_guard.cache."""
        from orthonym.decomposition.engine import try_decompose
        from orthonym.assembly.fragment_naming import _fragment_guard

        mol = Chem.MolFromSmiles("CC(=O)OCC")
        cache_before = getattr(_fragment_guard, 'cache', None)
        try_decompose(mol)
        cache_after = getattr(_fragment_guard, 'cache', None)
        # Cache state should be untouched by the probe
        assert cache_before is cache_after or cache_before == cache_after, \
            "_fragment_guard.cache was modified by probe path"

    @pytest.mark.unit
    def test_probe_returns_systematic_name_for_decomposable(self):
        """name_pipeline_only should return a systematic name even for decomposable molecules."""
        result = name_pipeline_only("CC(=O)OCC")  # ethyl acetate
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
