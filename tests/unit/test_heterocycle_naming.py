"""
Tests for heterocycle naming (Plan 03-03).

Tests retained names and Hantzsch-Widman systematic naming for:
- HETERO-01: 3-membered saturated heterocycles
- HETERO-02: 5-membered saturated heterocycles
- HETERO-03: 5-membered aromatic heterocycles
- HETERO-04: 6-membered saturated heterocycles
- HETERO-05: 6-membered aromatic heterocycles
- HETERO-06: HW systematic naming (fallback)
- HETERO-07: 4-membered heterocycles

Reference: IUPAC 2013 Blue Book, Section P-22 (Heterocycles)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.heterocycles import (
    name_heterocycle,
    build_hw_name,
    get_ring_canonical_smiles,
    classify_heterocycle,
    orient_heterocycle,
    get_heteroatom_locants,
)


# =============================================================================
# HETERO-01: 3-membered saturated heterocycles
# =============================================================================

class TestThreeMemberedHeterocycles:
    """Tests for 3-membered saturated heterocycles (HETERO-01)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Oxygen
        ("C1CO1", "oxirane"),
        # Nitrogen
        ("C1CN1", "aziridine"),
        # Sulfur
        ("C1CS1", "thiirane"),
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 3-membered heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_oxirane_ring_smiles(self):
        """Test canonical SMILES extraction for oxirane."""
        mol = Chem.MolFromSmiles("C1CO1")
        ring = mol.GetRingInfo().AtomRings()[0]
        ring_smiles = get_ring_canonical_smiles(mol, ring)
        # Should match retained name lookup
        assert name_compound(ring_smiles) == "oxirane"

    @pytest.mark.unit
    def test_aziridine_classification(self):
        """Test classification of aziridine."""
        mol = Chem.MolFromSmiles("C1CN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 3
        assert info['is_saturated'] is True
        assert info['is_aromatic'] is False
        assert info['dominant_heteroatom'] == 'N'
        assert info['num_heteroatoms'] == 1

    @pytest.mark.unit
    def test_thiirane_classification(self):
        """Test classification of thiirane."""
        mol = Chem.MolFromSmiles("C1CS1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 3
        assert info['is_saturated'] is True
        assert info['dominant_heteroatom'] == 'S'


# =============================================================================
# HETERO-02: 5-membered saturated heterocycles
# =============================================================================

class TestFiveMemberedSaturatedHeterocycles:
    """Tests for 5-membered saturated heterocycles (HETERO-02)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Oxygen - oxolane (THF)
        ("C1CCOC1", "oxolane"),
        # Nitrogen - pyrrolidine
        ("C1CCNC1", "pyrrolidine"),
        # Sulfur - tetrahydrothiophene (thiolane)
        ("C1CCSC1", "tetrahydrothiophene"),
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 5-membered saturated heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_thf_classification(self):
        """Test classification of oxolane."""
        mol = Chem.MolFromSmiles("C1CCOC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 5
        assert info['is_saturated'] is True
        assert info['is_aromatic'] is False
        assert info['dominant_heteroatom'] == 'O'

    @pytest.mark.unit
    def test_pyrrolidine_classification(self):
        """Test classification of pyrrolidine."""
        mol = Chem.MolFromSmiles("C1CCNC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 5
        assert info['is_saturated'] is True
        assert info['dominant_heteroatom'] == 'N'


# =============================================================================
# HETERO-03: 5-membered aromatic heterocycles
# =============================================================================

class TestFiveMemberedAromaticHeterocycles:
    """Tests for 5-membered aromatic heterocycles (HETERO-03)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Single heteroatom
        ("c1ccoc1", "furan"),
        # v23 IH-01: N-H azoles carry the leading indicated hydrogen in the PIN.
        ("c1cc[nH]c1", "1H-pyrrole"),
        ("c1ccsc1", "thiophene"),
        # Two heteroatoms - 1,3 arrangement
        ("c1c[nH]cn1", "1H-imidazole"),  # N at 1,3
        ("c1cnco1", "oxazole"),       # O at 1, N at 3
        ("c1cncs1", "thiazole"),      # S at 1, N at 3
        # Two heteroatoms - 1,2 arrangement
        ("c1cc[nH]n1", "1H-pyrazole"),   # N at 1,2
        ("c1ccno1", "isoxazole"),     # O at 1, N at 2
        ("c1ccsn1", "isothiazole"),   # S at 1, N at 2
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 5-membered aromatic heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_furan_classification(self):
        """Test classification of furan."""
        mol = Chem.MolFromSmiles("c1ccoc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 5
        assert info['is_aromatic'] is True
        assert info['is_saturated'] is False
        assert info['dominant_heteroatom'] == 'O'

    @pytest.mark.unit
    def test_imidazole_classification(self):
        """Test classification of imidazole."""
        mol = Chem.MolFromSmiles("c1c[nH]cn1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 5
        assert info['is_aromatic'] is True
        assert info['num_heteroatoms'] == 2
        # Both heteroatoms are N
        elements = [elem for _, elem in info['heteroatoms']]
        assert elements.count('N') == 2

    @pytest.mark.unit
    def test_thiophene_orientation(self):
        """Test that thiophene S is at position 1."""
        mol = Chem.MolFromSmiles("c1ccsc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, atom_to_locant = orient_heterocycle(mol, ring)
        # S should be at position 1 (first in oriented list)
        s_idx = oriented[0]
        assert mol.GetAtomWithIdx(s_idx).GetSymbol() == 'S'


# =============================================================================
# HETERO-04: 6-membered saturated heterocycles
# =============================================================================

class TestSixMemberedSaturatedHeterocycles:
    """Tests for 6-membered saturated heterocycles (HETERO-04)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Oxygen - oxane
        ("C1CCOCC1", "oxane"),
        # Nitrogen - piperidine
        ("C1CCNCC1", "piperidine"),
        # Sulfur - thiane
        ("C1CCSCC1", "thiane"),
        # Two heteroatoms
        ("C1COCCN1", "morpholine"),   # O and N
        ("C1CNCCN1", "piperazine"),   # Two N
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 6-membered saturated heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_piperidine_classification(self):
        """Test classification of piperidine."""
        mol = Chem.MolFromSmiles("C1CCNCC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 6
        assert info['is_saturated'] is True
        assert info['dominant_heteroatom'] == 'N'

    @pytest.mark.unit
    def test_morpholine_orientation(self):
        """Test that morpholine O is at position 1 (higher priority than N)."""
        mol = Chem.MolFromSmiles("C1COCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        # O has higher priority than N, so O should be at position 1
        o_idx = oriented[0]
        assert mol.GetAtomWithIdx(o_idx).GetSymbol() == 'O'

    @pytest.mark.unit
    def test_morpholine_heteroatom_locants(self):
        """Test that morpholine has O at 1 and N at 4."""
        mol = Chem.MolFromSmiles("C1COCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        # Should be [(1, 'O'), (4, 'N')]
        assert len(locants) == 2
        assert locants[0] == (1, 'O')
        assert locants[1] == (4, 'N')

    @pytest.mark.unit
    def test_piperazine_heteroatom_locants(self):
        """Test that piperazine has N at 1 and N at 4."""
        mol = Chem.MolFromSmiles("C1CNCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        # Should have two N atoms at positions 1 and 4
        assert len(locants) == 2
        assert locants[0][1] == 'N'
        assert locants[1][1] == 'N'
        assert locants[0][0] == 1
        assert locants[1][0] == 4


# =============================================================================
# HETERO-05: 6-membered aromatic heterocycles
# =============================================================================

class TestSixMemberedAromaticHeterocycles:
    """Tests for 6-membered aromatic heterocycles (HETERO-05)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Single nitrogen - pyridine
        ("c1ccncc1", "pyridine"),
        # Two nitrogens - diazines
        ("c1ccnnc1", "pyridazine"),   # 1,2-diazine
        ("c1cncnc1", "pyrimidine"),   # 1,3-diazine
        ("c1cnccn1", "pyrazine"),     # 1,4-diazine
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 6-membered aromatic heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_pyridine_classification(self):
        """Test classification of pyridine."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 6
        assert info['is_aromatic'] is True
        assert info['is_saturated'] is False
        assert info['dominant_heteroatom'] == 'N'
        assert info['num_heteroatoms'] == 1

    @pytest.mark.unit
    def test_pyridine_n_at_position_1(self):
        """Test that pyridine N is at position 1."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        n_idx = oriented[0]
        assert mol.GetAtomWithIdx(n_idx).GetSymbol() == 'N'

    @pytest.mark.unit
    def test_pyrimidine_classification(self):
        """Test classification of pyrimidine."""
        mol = Chem.MolFromSmiles("c1cncnc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 6
        assert info['is_aromatic'] is True
        assert info['num_heteroatoms'] == 2

    @pytest.mark.unit
    def test_pyrimidine_n_locants(self):
        """Test that pyrimidine has N at 1,3 (not 1,5)."""
        mol = Chem.MolFromSmiles("c1cncnc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        oriented, _ = orient_heterocycle(mol, ring)
        locants = get_heteroatom_locants(oriented, mol)
        # Should be [(1, 'N'), (3, 'N')] not [(1, 'N'), (5, 'N')]
        positions = sorted([loc for loc, _ in locants])
        assert positions == [1, 3]


# =============================================================================
# HETERO-06: Hantzsch-Widman systematic naming
# =============================================================================

class TestHWSystematicNaming:
    """Tests for HW systematic naming when no retained name exists (HETERO-06)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("heteroatoms,ring_size,saturated,aromatic,expected", [
        # Single heteroatom, saturated
        ([(1, 'O')], 5, True, False, "oxolane"),
        ([(1, 'N')], 5, True, False, "azolidine"),
        ([(1, 'S')], 5, True, False, "thiolane"),
        # Single heteroatom, aromatic
        ([(1, 'O')], 5, False, True, "oxole"),
        ([(1, 'N')], 6, False, True, "azine"),
        # Multiple same heteroatoms
        ([(1, 'O'), (3, 'O')], 5, True, False, "1,3-dioxolane"),
        ([(1, 'N'), (4, 'N')], 6, True, False, "1,4-diazinane"),
        # 3-membered
        ([(1, 'O')], 3, True, False, "oxirane"),
        ([(1, 'N')], 3, True, False, "aziridine"),
        # 6-membered saturated with different heteroatoms
        ([(1, 'O')], 6, True, False, "oxane"),
        ([(1, 'N')], 6, True, False, "azinane"),
    ])
    def test_build_hw_name(self, heteroatoms, ring_size, saturated, aromatic, expected):
        """Test HW name building from components."""
        result = build_hw_name(heteroatoms, ring_size, saturated, aromatic)
        assert result == expected

    @pytest.mark.unit
    def test_hw_name_a_elision(self):
        """Test that terminal 'a' is elided before vowel stem."""
        # oxa + irane = oxirane (not oxairane)
        result = build_hw_name([(1, 'O')], 3, True, False)
        assert result == "oxirane"
        assert "oxairane" not in result

    @pytest.mark.unit
    def test_hw_name_multiple_different_heteroatoms(self):
        """Test HW naming with multiple different heteroatoms."""
        # O has higher priority than N, so oxa comes first
        # Morpholine-like: O at 1, N at 4
        result = build_hw_name([(1, 'O'), (4, 'N')], 6, True, False)
        # Should be something like oxazinane (if not a retained name)
        assert result.startswith("oxa")
        assert "az" in result  # Contains aza prefix

    @pytest.mark.unit
    def test_hw_prefix_priority_order(self):
        """Test that heteroatom prefixes are ordered by IUPAC priority."""
        # O > S > N in priority
        result = build_hw_name([(1, 'O'), (3, 'S'), (5, 'N')], 7, True, False)
        # oxa should come before thia, which should come before aza
        oxa_pos = result.find("oxa")
        thia_pos = result.find("thia")
        aza_pos = result.find("aza") if "aza" in result else result.find("az")
        assert oxa_pos < thia_pos < aza_pos


# =============================================================================
# HETERO-07: 4-membered heterocycles
# =============================================================================

class TestFourMemberedHeterocycles:
    """Tests for 4-membered heterocycles (HETERO-07)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Oxygen - oxetane
        ("C1COC1", "oxetane"),
        # Nitrogen - azetidine
        ("C1CNC1", "azetidine"),
        # Sulfur - thietane
        ("C1CSC1", "thietane"),
    ])
    def test_retained_names(self, smiles, expected):
        """Test retained names for common 4-membered heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_oxetane_classification(self):
        """Test classification of oxetane."""
        mol = Chem.MolFromSmiles("C1COC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 4
        assert info['is_saturated'] is True
        assert info['dominant_heteroatom'] == 'O'

    @pytest.mark.unit
    def test_azetidine_classification(self):
        """Test classification of azetidine."""
        mol = Chem.MolFromSmiles("C1CNC1")
        ring = mol.GetRingInfo().AtomRings()[0]
        info = classify_heterocycle(mol, ring)
        assert info['ring_size'] == 4
        assert info['is_saturated'] is True
        assert info['dominant_heteroatom'] == 'N'


# =============================================================================
# Additional heterocycles - multiple same heteroatoms
# =============================================================================

class TestMultipleSameHeteroatoms:
    """Tests for heterocycles with multiple same heteroatoms."""

    @pytest.mark.unit
    def test_dioxolane_hw_name(self):
        """Test 1,3-dioxolane HW systematic name."""
        result = build_hw_name([(1, 'O'), (3, 'O')], 5, True, False)
        assert result == "1,3-dioxolane"

    @pytest.mark.unit
    def test_dithiane_pattern(self):
        """Test dithiane-like structure naming."""
        # 1,3-dithiane (6-membered with 2 S)
        result = build_hw_name([(1, 'S'), (3, 'S')], 6, True, False)
        assert "dithia" in result  # Contains di- multiplier + thia

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # Tetrazole - 4 nitrogens in 5-membered aromatic ring.
        # v23 IH-01: this tautomer is the 1H- form (OPSIN-RT verified).
        ("c1nnn[nH]1", "1H-tetrazole"),
    ])
    def test_multiple_nitrogen_aromatics(self, smiles, expected):
        """Test aromatic heterocycles with multiple nitrogens."""
        assert name_compound(smiles) == expected


# =============================================================================
# Triazines
# =============================================================================

class TestTriazines:
    """Tests for triazine isomers."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("c1nncnn1", "1,2,4-triazine"),
        ("c1ncncn1", "1,3,5-triazine"),
    ])
    def test_triazine_retained_names(self, smiles, expected):
        """Test retained names for triazine isomers."""
        assert name_compound(smiles) == expected


# =============================================================================
# Edge cases and regression tests
# =============================================================================

class TestEdgeCases:
    """Edge case and regression tests."""

    @pytest.mark.unit
    def test_empty_heteroatoms_hw_name(self):
        """Test build_hw_name with empty heteroatom list."""
        result = build_hw_name([], 5, True, False)
        assert result == ""

    @pytest.mark.unit
    def test_single_heteroatom_no_locant_in_name(self):
        """Test that single heteroatom doesn't get explicit locant."""
        # Single O in 5-membered saturated = oxolane (not 1-oxolane)
        result = build_hw_name([(1, 'O')], 5, True, False)
        assert result == "oxolane"
        assert "1-" not in result

    @pytest.mark.unit
    def test_imidazole_alternate_smiles(self):
        """Test that alternate SMILES for imidazole gives same name."""
        # Different tautomeric SMILES representations
        smiles_variants = [
            "c1c[nH]cn1",  # Standard
            "c1cnc[nH]1",  # Different starting position
        ]
        for smiles in smiles_variants:
            assert name_compound(smiles) == "1H-imidazole"  # v23 IH-01: leading indicated-H


# =============================================================================
# Integration with full naming pipeline
# =============================================================================

class TestPipelineIntegration:
    """Tests for full naming pipeline integration."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # 3-membered
        ("C1CO1", "oxirane"),
        ("C1CN1", "aziridine"),
        ("C1CS1", "thiirane"),
        # 4-membered
        ("C1COC1", "oxetane"),
        ("C1CNC1", "azetidine"),
        ("C1CSC1", "thietane"),
        # 5-membered saturated
        ("C1CCOC1", "oxolane"),
        ("C1CCNC1", "pyrrolidine"),
        ("C1CCSC1", "tetrahydrothiophene"),
        # 5-membered aromatic (v23 IH-01: N-H azoles carry leading 1H-)
        ("c1ccoc1", "furan"),
        ("c1cc[nH]c1", "1H-pyrrole"),
        ("c1ccsc1", "thiophene"),
        ("c1c[nH]cn1", "1H-imidazole"),
        ("c1cnco1", "oxazole"),
        ("c1cncs1", "thiazole"),
        ("c1cc[nH]n1", "1H-pyrazole"),
        ("c1ccno1", "isoxazole"),
        ("c1ccsn1", "isothiazole"),
        # 6-membered saturated
        ("C1CCOCC1", "oxane"),
        ("C1CCNCC1", "piperidine"),
        ("C1CCSCC1", "thiane"),
        ("C1COCCN1", "morpholine"),
        ("C1CNCCN1", "piperazine"),
        # 6-membered aromatic
        ("c1ccncc1", "pyridine"),
        ("c1ccnnc1", "pyridazine"),
        ("c1cncnc1", "pyrimidine"),
        ("c1cnccn1", "pyrazine"),
    ])
    def test_full_pipeline_naming(self, smiles, expected):
        """Test complete naming pipeline for all common heterocycles."""
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_heterocycles_dont_break_alkanes(self):
        """Regression: heterocycle support doesn't break simple alkanes."""
        assert name_compound("CC") == "ethane"
        assert name_compound("CCC") == "propane"
        assert name_compound("CCCC") == "butane"

    @pytest.mark.unit
    def test_heterocycles_dont_break_cycloalkanes(self):
        """Regression: heterocycle support doesn't break cycloalkanes."""
        assert name_compound("C1CC1") == "cyclopropane"
        assert name_compound("C1CCC1") == "cyclobutane"
        assert name_compound("C1CCCC1") == "cyclopentane"
        assert name_compound("C1CCCCC1") == "cyclohexane"

    @pytest.mark.unit
    def test_heterocycles_dont_break_benzene(self):
        """Regression: heterocycle support doesn't break benzene."""
        assert name_compound("c1ccccc1") == "benzene"


# =============================================================================
# HW stem selection tests
# =============================================================================

class TestHWStemSelection:
    """Tests for correct HW stem selection based on ring size and saturation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("ring_size,saturated,heteroatom,expected_ending", [
        # 3-membered
        (3, True, 'O', "irane"),
        (3, True, 'N', "iridine"),
        (3, False, 'O', "irene"),
        # 4-membered
        (4, True, 'O', "etane"),
        (4, True, 'N', "etidine"),
        (4, False, 'O', "ete"),
        # 5-membered
        (5, True, 'O', "olane"),
        (5, True, 'N', "olidine"),
        (5, False, 'O', "ole"),
        # 6-membered (different endings for O/S vs N)
        (6, True, 'O', "ane"),
        (6, True, 'N', "inane"),
        (6, False, 'N', "ine"),
        # 7-membered
        (7, True, 'O', "epane"),
        (7, False, 'O', "epine"),
    ])
    def test_hw_stem_endings(self, ring_size, saturated, heteroatom, expected_ending):
        """Test that HW names have correct stem endings."""
        result = build_hw_name([(1, heteroatom)], ring_size, saturated, not saturated)
        assert result.endswith(expected_ending), f"Expected {result} to end with {expected_ending}"


# =============================================================================
# Direct name_heterocycle tests
# =============================================================================

class TestNameHeterocycleFunction:
    """Direct tests for name_heterocycle function."""

    @pytest.mark.unit
    def test_pyridine_direct(self):
        """Test name_heterocycle directly for pyridine."""
        mol = Chem.MolFromSmiles("c1ccncc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert name_heterocycle(mol, ring) == "pyridine"

    @pytest.mark.unit
    def test_furan_direct(self):
        """Test name_heterocycle directly for furan."""
        mol = Chem.MolFromSmiles("c1ccoc1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert name_heterocycle(mol, ring) == "furan"

    @pytest.mark.unit
    def test_oxirane_direct(self):
        """Test name_heterocycle directly for oxirane."""
        mol = Chem.MolFromSmiles("C1CO1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert name_heterocycle(mol, ring) == "oxirane"

    @pytest.mark.unit
    def test_morpholine_direct(self):
        """Test name_heterocycle directly for morpholine."""
        mol = Chem.MolFromSmiles("C1COCCN1")
        ring = mol.GetRingInfo().AtomRings()[0]
        assert name_heterocycle(mol, ring) == "morpholine"
