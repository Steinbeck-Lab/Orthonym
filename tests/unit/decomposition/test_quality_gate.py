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
        """Sulfonamide bond type should use 0.8 chars/HA threshold.

        Task Z3 changed the probe name, not the threshold. This test used
        "phenyl palmitate", which DENOTES this molecule exactly -- so once a
        failing character count started being checked against the real oracle,
        it was (correctly) no longer discarded, and the test could no longer
        demonstrate the 0.6/0.8 tier. "phenyl acetate" has the same length
        behaviour at both thresholds (14 chars: >= int(24*0.6)=14, <
        int(24*0.8)=19) but describes only 10 of the 24 heavy atoms, so the
        oracle agrees it is partial and the tier is exercised as intended.
        """
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl acetate" = 14 chars, 24 HA => 0.58 chars/HA
        # Passes 0.6 threshold, fails 0.8 threshold; genuinely partial.
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")  # ~24 HA
        assert _coverage_is_adequate("phenyl acetate", mol, bond_type="ester") is True
        assert _coverage_is_adequate("phenyl acetate", mol, bond_type="sulfonamide") is False

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
        """No bond_type (default="") should use 0.8 threshold for backward compat.

        Probe name changed by Task Z3 for the reason given in
        test_sulfonamide_uses_higher_threshold: the old probe covered the
        molecule, so it is no longer discardable on length alone.
        """
        from orthonym.decomposition.engine import _coverage_is_adequate
        # "phenyl acetate" = 14 chars, 24 HA => 0.58 chars/HA; partial name.
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)Oc1ccccc1")
        assert _coverage_is_adequate("phenyl acetate", mol) is False
        # And the covering name is NOT discarded, at either threshold. This is
        # the Task Z3 behaviour change stated positively.
        assert _coverage_is_adequate("phenyl palmitate", mol) is True

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


class TestNameSizeCoverage:
    """Tests for _name_covers_molecule() heuristic in quality gate.

    The name-size coverage heuristic rejects pipeline names that describe
    less than ~45% of the molecule's heavy atoms when cleavable bonds
    exist and decomposition hasn't been attempted yet. This catches cases
    like "5-chloroquinoline" (17 chars) naming a 26-HA ester molecule
    where the name only describes the quinoline fragment.
    """

    @pytest.mark.unit
    def test_small_molecule_always_passes(self):
        """_name_covers_molecule returns True for small/medium molecules
        (HA <= 20) regardless of name length."""
        from orthonym.decomposition.engine import _name_covers_molecule
        # naphthalene: 10 HA, "naphthalene" = 11 chars
        mol = Chem.MolFromSmiles("c1cccc2ccccc12")  # naphthalene, HA=10
        assert mol is not None
        assert mol.GetNumHeavyAtoms() <= 20
        assert _name_covers_molecule("naphthalene", mol) is True

        # heptanamide: 18 HA, amide bond, "heptanamide" = 11 chars
        # HA=18 <= 20 => bypass => passes
        mol2 = Chem.MolFromSmiles("CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12")
        assert mol2 is not None
        assert mol2.GetNumHeavyAtoms() <= 20
        assert _name_covers_molecule("heptanamide", mol2) is True

    @pytest.mark.unit
    def test_adequate_coverage_passes(self):
        """_name_covers_molecule returns True when estimated coverage >= 0.45
        (e.g., '2-methylnaphthalene' for 11-HA molecule)."""
        from orthonym.decomposition.engine import _name_covers_molecule
        # 2-methylnaphthalene: 11 HA, "2-methylnaphthalene" = 19 chars
        # estimated_ha = 19/1.5 = 12.7; coverage = 12.7/11 = 1.15 > 0.55
        mol = Chem.MolFromSmiles("Cc1ccc2ccccc2c1")  # 2-methylnaphthalene, HA=11
        assert mol is not None
        assert _name_covers_molecule("2-methylnaphthalene", mol) is True

    @pytest.mark.unit
    def test_partial_coverage_rejected(self):
        """_name_covers_molecule returns False when estimated coverage < 0.45
        (e.g., '5-chloroquinoline' for 26-HA ester)."""
        from orthonym.decomposition.engine import _name_covers_molecule
        # Long-chain ester with 5-chloroquinoline as the alcohol portion.
        # "5-chloroquinoline" = 17 chars, estimated_ha = 17/1.5 = 11.3
        # HA = 26, coverage = 11.3/26 = 0.44 < 0.55 => rejected.
        # Molecule has 1 ester bond (cleavable).
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12")  # 26 HA ester
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 15, f"Need HA > 15, got {ha}"
        assert _name_covers_molecule("5-chloroquinoline", mol) is False

    @pytest.mark.unit
    def test_no_cleavable_bonds_always_passes(self):
        """_name_covers_molecule returns True when no cleavable bonds exist,
        even if coverage ratio is low, because pipeline name is the only option."""
        from orthonym.decomposition.engine import _name_covers_molecule
        # Anthracene: 14 HA, no cleavable bonds, short name
        # "anthracene" = 10 chars, estimated_ha = 10/1.5 = 6.7
        # coverage = 6.7/14 = 0.48 < 0.55, BUT no cleavable bonds => pass
        mol = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")  # anthracene, 14 HA
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha == 14
        assert _name_covers_molecule("anthracene", mol) is True

    @pytest.mark.unit
    def test_decomposition_context_always_passes(self):
        """_name_covers_molecule returns True when visited set is non-empty
        (decomposition context) to avoid rejecting decomposition results."""
        from orthonym.decomposition.engine import _name_covers_molecule
        from orthonym.assembly.fragment_naming import _fragment_guard
        # Simulate decomposition context: set visited to non-empty
        old_visited = getattr(_fragment_guard, 'visited', None)
        _fragment_guard.visited = {"some_smiles_in_progress"}
        try:
            # Large molecule with short name that would normally fail
            mol = Chem.MolFromSmiles("CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12")  # 26 HA
            assert _name_covers_molecule("5-chloroquinoline", mol) is True
        finally:
            _fragment_guard.visited = old_visited

    @pytest.mark.unit
    def test_quality_gate_rejects_partial_names(self):
        """_name_quality_is_acceptable rejects partial names that pass all
        existing checks but fail name-size coverage."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable
        # Molecule: long-chain ester with 26 HA, pipeline names as
        # "5-chloroquinoline". This name has digits AND hyphens (passes
        # no-digits check), length 17 >= 26//2=13 (passes short check),
        # ratio 17/26=0.65 (passes 0.45 threshold). But it only describes
        # ~11 HA of a 26-HA molecule (coverage 0.44 < 0.45).
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCC(=O)Oc1ccc(Cl)c2ncccc12")
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 15
        # Should be rejected by name-size coverage
        assert _name_quality_is_acceptable("5-chloroquinoline", mol) is False

    @pytest.mark.unit
    def test_integration_large_molecule_partial_name_triggers_decomp(self):
        """Integration: molecules with HA>15 and partial pipeline names
        should trigger decomposition (try_decompose returns non-None)."""
        from orthonym.decomposition.engine import _name_covers_molecule
        # (22E)-stigmasta-7,22-diene naming a 52-HA glycoside
        # Name = 27 chars, estimated_ha = 27/1.5 = 18; coverage = 18/52 = 0.35 < 0.55
        # Large steroid glycoside
        smiles = "CC(C)C(C)CC=CC(C)C1CCC2C3=CCC4CC(OC5OC(CO)C(O)C(O)C5O)CCC4(C)C3CCC12C"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        ha = mol.GetNumHeavyAtoms()
        assert ha > 30, f"Need large molecule, got HA={ha}"
        # Name covers only the steroid, not the sugar
        assert _name_covers_molecule("(22E)-stigmasta-7,22-diene", mol) is False


class TestBoundaryAwareTokenMatching:
    """Tests for IUPAC morpheme-aware boundary token matching in quality gate.

    Per D-13/D-14: tokens should match at IUPAC nomenclature boundaries
    (after hyphen, after paren, at start/end of name), not as arbitrary
    substrings. Per D-16: known polymer names like "polyester" must NOT
    match bond-type tokens.
    """

    @pytest.mark.unit
    def test_polyester_does_not_match_ester(self):
        """'polyester' must NOT trigger ester bond type token (D-16)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("polyester", patterns, "ester") is False

    @pytest.mark.unit
    def test_polyamide_does_not_match_amide(self):
        """'polyamide' must NOT trigger amide bond type token (D-16)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("polyamide", patterns, "amide") is False

    @pytest.mark.unit
    def test_acetate_matches_ester(self):
        """'methyl acetate' MUST trigger ester bond type token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("methyl acetate", patterns, "ester") is True

    @pytest.mark.unit
    def test_propanamide_matches_amide(self):
        """'propanamide' MUST trigger amide bond type token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("propanamide", patterns, "amide") is True

    @pytest.mark.unit
    def test_propanoate_matches_ester(self):
        """'ethyl propanoate' MUST trigger ester bond type token ('ate' at word end)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("ethyl propanoate", patterns, "ester") is True

    @pytest.mark.unit
    def test_benzoyloxy_matches_ester(self):
        """'benzoyloxycyclohexane' MUST trigger ester bond type token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("benzoyloxycyclohexane", patterns, "ester") is True

    @pytest.mark.unit
    def test_amino_after_hyphen_matches_amide(self):
        """'2-aminoethanol' MUST trigger amide bond type token (after hyphen)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("2-aminoethanol", patterns, "amide") is True

    @pytest.mark.unit
    def test_phosph_at_start_matches(self):
        """'phosphoric acid' MUST trigger phosphodiester token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("phosphoric acid", patterns, "phosphodiester") is True

    @pytest.mark.unit
    def test_oyl_as_suffix_matches(self):
        """'propanoyl chloride' MUST trigger amide bond type token ('oyl' at boundary)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("propanoyl chloride", patterns, "amide") is True

    @pytest.mark.unit
    def test_empty_token_set_passes(self):
        """Ether bond type (empty token set) always returns True."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("anything at all", patterns, "ether") is True

    @pytest.mark.unit
    def test_polycarbonate_does_not_match_carbamate(self):
        """'polycarbonate' must NOT trigger carbamate token (D-16)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("polycarbonate", patterns, "carbamate") is False

    @pytest.mark.unit
    def test_polyurethane_does_not_match(self):
        """'polyurethane' must NOT trigger carbamate token (D-16)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("polyurethane", patterns, "carbamate") is False

    @pytest.mark.unit
    def test_unusual_input_no_crash(self):
        """Token matching never raises an exception even with unusual input (D-17)."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        # Empty string
        try:
            _token_matches_name("", patterns, "ester")
        except Exception as e:
            pytest.fail(f"Raised exception on empty string: {e}")
        # Unicode characters
        try:
            _token_matches_name("\u00e9th\u00e8r-\u00e4mide", patterns, "amide")
        except Exception as e:
            pytest.fail(f"Raised exception on unicode: {e}")
        # Very long string
        try:
            _token_matches_name("a" * 10000, patterns, "ester")
        except Exception as e:
            pytest.fail(f"Raised exception on long string: {e}")
        # Unknown bond type
        try:
            result = _token_matches_name("test", patterns, "unknown_type")
            assert result is True  # empty patterns = benefit of doubt
        except Exception as e:
            pytest.fail(f"Raised exception on unknown bond type: {e}")

    @pytest.mark.unit
    def test_carbamate_at_start_matches(self):
        """'carbamate ester' MUST trigger carbamate bond type token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("carbamate ester", patterns, "carbamate") is True

    @pytest.mark.unit
    def test_sulfonamide_matches(self):
        """'N-methylbenzenesulfonamide' MUST trigger sulfonamide token."""
        from orthonym.decomposition.engine import (
            _compile_token_patterns, _token_matches_name, _BOND_TYPE_TOKENS,
        )
        patterns = _compile_token_patterns(_BOND_TYPE_TOKENS)
        assert _token_matches_name("N-methylbenzenesulfonamide", patterns, "sulfonamide") is True
