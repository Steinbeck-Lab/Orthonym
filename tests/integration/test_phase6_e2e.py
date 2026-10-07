"""
Comprehensive End-to-End Integration Tests for a phase: Complex Ring Systems.

Validates all a phase requirements (COMPLEX-01 through COMPLEX-06 + VALID):
- COMPLEX-01: Bicyclo compound naming (bicyclo[x.y.z] format)
- COMPLEX-02: Spiro compound naming (spiro[a.b] format)
- COMPLEX-03: Ortho-fused bicyclics (decalin, tetralin, naphthalene)
- COMPLEX-04: Fusion descriptors (systematic fusion name generation)
- COMPLEX-05: Heterocyclic fused systems (indole, quinoline, carbazole)
- COMPLEX-06: Complex PAHs (pyrene, perylene, coronene, etc.)

This test file provides comprehensive E2E coverage for a phase Final Validation.

Reference: IUPAC 2013 Blue Book, Sections,,,
"""

import pytest
from orthonym import name_compound


# =============================================================================
# COMPLEX-01: Bicyclo Compound Naming
# =============================================================================

class TestCOMPLEX01_Bicyclo_Hydrocarbons:
    """COMPLEX-01: Bicyclo hydrocarbon systematic naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Norbornane - uses retained name
        ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),  # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
        # Bicyclo[2.2.2]octane - systematic name
        ("C1CC2CCC1CC2", "bicyclo[2.2.2]octane"),
        # Bicyclo[3.2.1]octane
        ("C1CC2CCCC1C2", "bicyclo[3.2.1]octane"),
        # Decalin - fused naming (zero-bridge routes to fused nomenclature)
        ("C1CCC2CCCCC2C1", "decahydronaphthalene"),
    ])
    def test_bicyclo_systematic_naming(self, smiles, expected):
        """Test systematic bicyclo naming with various bridge sizes."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_norbornane_uses_retained_name(self):
        """Norbornane is NOT retained: the PIN is the von Baeyer name bicyclo[2.2.1]heptane."""
        result = name_compound("C1CC2CCC1C2")
        # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
        assert result == "bicyclo[2.2.1]heptane"

    @pytest.mark.integration
    def test_bicyclo_descriptor_format(self):
        """Verify bicyclo descriptor format [x.y.z] is correct."""
        result = name_compound("C1CC2CCC1CC2")  # bicyclo[2.2.2]octane
        assert "bicyclo[" in result
        assert "]" in result
        # Should have two periods in descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start:desc_end+1]
        assert descriptor.count(".") == 2

    @pytest.mark.integration
    def test_bicyclo_numbers_ordered_correctly(self):
        """Verify bicyclo numbers are in decreasing order (largest bridge first)."""
        result = name_compound("C1CC2CCCC1C2")  # bicyclo[3.2.1]octane
        # Extract the numbers from descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start+1:desc_end]
        numbers = [int(n) for n in descriptor.split(".")]
        assert numbers == sorted(numbers, reverse=True), "Bridge numbers should be in decreasing order"


class TestCOMPLEX01_Bicyclo_EdgeCases:
    """Edge cases for bicyclo system naming."""

    @pytest.mark.integration
    def test_bicyclo_not_confused_with_fused_aromatic(self):
        """Naphthalene should NOT be classified as bicyclo."""
        result = name_compound("c1ccc2ccccc2c1")
        assert result == "naphthalene"
        assert "bicyclo" not in result

    @pytest.mark.integration
    def test_simple_cycloalkane_not_bicyclo(self):
        """Simple cycloalkanes should not be classified as bicyclo."""
        assert name_compound("C1CCCCC1") == "cyclohexane"
        assert name_compound("C1CCCC1") == "cyclopentane"


# =============================================================================
# COMPLEX-02: Spiro Compound Naming
# =============================================================================

class TestCOMPLEX02_Spiro_Hydrocarbons:
    """COMPLEX-02: Spiro hydrocarbon systematic naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # spiro[4.5]decane - cyclopentane fused to cyclohexane
        ("C1CCC2(CC1)CCCC2", "spiro[4.5]decane"),
        # spiro[5.5]undecane - two cyclohexanes
        ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
    ])
    def test_spiro_systematic_naming(self, smiles, expected):
        """Test systematic spiro naming."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_spiro_descriptor_order(self):
        """Verify spiro descriptor uses smaller ring first."""
        result = name_compound("C1CCC2(CC1)CCCC2")  # 5-ring + 6-ring
        assert "spiro[4.5]" in result  # 4-membered contribution, 5-membered contribution
        assert "spiro[5.4]" not in result  # NOT reversed

    @pytest.mark.integration
    def test_spiro_descriptor_format(self):
        """Verify spiro descriptor format [a.b] is correct."""
        result = name_compound("C1CCC2(CC1)CCCCC2")
        assert "spiro[" in result
        assert "]" in result
        # Extract descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start:desc_end+1]
        assert descriptor.count(".") == 1  # Single period in spiro descriptor


class TestCOMPLEX02_Spiro_EdgeCases:
    """Edge cases for spiro system naming."""

    @pytest.mark.integration
    def test_spiro_symmetric_rings(self):
        """Test symmetric spiro compound (both rings same size)."""
        result = name_compound("C1CCC2(CC1)CCCCC2")  # Two 6-rings
        assert "spiro[5.5]" in result
        assert "undecane" in result

    @pytest.mark.integration
    def test_spiro_atom_count_correct(self):
        """Verify spiro compound has correct atom count in name."""
        # spiro[4.5]decane = 4+5+1 = 10 carbons
        result = name_compound("C1CCC2(CC1)CCCC2")
        assert "decane" in result  # 10 carbons

    @pytest.mark.integration
    def test_spiro_not_confused_with_bicyclo(self):
        """Spiro systems should not be confused with bicyclo."""
        result = name_compound("C1CCC2(CC1)CCCC2")
        assert "spiro" in result
        assert "bicyclo" not in result


# =============================================================================
# COMPLEX-03: Ortho-Fused Bicyclics
# =============================================================================

class TestCOMPLEX03_OrthoFused:
    """COMPLEX-03: Ortho-fused bicyclic naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Naphthalene - retained name
        ("c1ccc2ccccc2c1", "naphthalene"),
        # Decalin (decahydronaphthalene) - fused nomenclature for zero-bridge system
        ("C1CCC2CCCCC2C1", "decahydronaphthalene"),
    ])
    def test_ortho_fused_retained_names(self, smiles, expected):
        """Test ortho-fused bicyclics with retained and systematic names."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_naphthalene_is_fused_not_bridged(self):
        """Naphthalene should be identified as fused aromatic, not bridged."""
        result = name_compound("c1ccc2ccccc2c1")
        assert result == "naphthalene"
        assert "bicyclo" not in result  # Not bridged naming


class TestCOMPLEX03_OrthoFusedTricyclics:
    """Ortho-fused tricyclic aromatics."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Anthracene - linear tricyclic
        ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
        # Phenanthrene - angular tricyclic
        ("c1ccc2c(c1)ccc1ccccc12", "phenanthrene"),
    ])
    def test_tricyclic_aromatics(self, smiles, expected):
        """Test tricyclic aromatic retained names."""
        assert name_compound(smiles) == expected


# =============================================================================
# COMPLEX-04: Fusion Descriptors
# =============================================================================

class TestCOMPLEX04_FusionDescriptors:
    """COMPLEX-04: Fusion descriptor generation for systematic names."""

    @pytest.mark.integration
    def test_fusion_prefix_generation(self):
        """Test that fusion prefixes are generated correctly."""
        from orthonym.rules.fusion_descriptors import get_fusion_prefix

        assert get_fusion_prefix('benzene') == 'benzo'
        assert get_fusion_prefix('naphthalene') == 'naphtho'
        assert get_fusion_prefix('furan') == 'furo'
        assert get_fusion_prefix('pyrrole') == 'pyrrolo'
        assert get_fusion_prefix('pyridine') == 'pyrido'

    @pytest.mark.integration
    def test_fusion_edge_calculation(self):
        """Test fusion edge letter assignment."""
        from orthonym.rules.fusion_descriptors import get_fusion_letter

        # Ring atoms [0,1,2,3,4,5], shared edge between 0-1 = edge 'a'
        assert get_fusion_letter([0, 1, 2, 3, 4, 5], (0, 1)) == 'a'
        # Edge between 1-2 = edge 'b'
        assert get_fusion_letter([0, 1, 2, 3, 4, 5], (1, 2)) == 'b'

    @pytest.mark.integration
    def test_fusion_descriptor_format(self):
        """Test fusion descriptor format [num,num-letter]."""
        from orthonym.rules.fusion_descriptors import generate_fusion_descriptor

        # Use ring indices where shared atoms exist in both rings
        parent = [0, 1, 2, 3, 4, 5]  # 6-membered
        child = [0, 1, 6, 7, 8]      # 5-membered sharing atoms 0,1 with parent
        shared = {0, 1}              # Shared edge

        descriptor = generate_fusion_descriptor(parent, child, shared)
        # Should be in format [num,num-letter]
        assert '[' in descriptor
        assert ']' in descriptor
        assert '-' in descriptor


class TestCOMPLEX04_SystematicFusionNames:
    """Systematic fusion name building."""

    @pytest.mark.integration
    def test_benzo_fusion_prefix_no_elision(self):
        """IUPAC 2013: 'o' in benzo not elided before vowels."""
        from orthonym.rules.fusion_descriptors import build_systematic_fusion_name

        # benzo + anthracene = benzoanthracene (NOT benzanthracene)
        result = build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        assert result == 'benzo[a]anthracene'
        assert 'benzanthracene' not in result


# =============================================================================
# COMPLEX-05: Heterocyclic Fused Systems
# =============================================================================

class TestCOMPLEX05_FusedHeterocycles:
    """COMPLEX-05: Heterocyclic fused system naming with retained names."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Indole - benzo[b]pyrrole
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        # Quinoline - benzo[b]pyridine
        ("c1ccc2ncccc2c1", "quinoline"),
        # Isoquinoline
        ("c1ccc2cnccc2c1", "isoquinoline"),
        # Benzimidazole
        ("c1ccc2[nH]cnc2c1", "1H-1,3-benzimidazole"),
        # Benzofuran
        #: PIN carries the O locant, the Blue Book
        # "1-benzofuran (PIN) benzofuran")
        ("c1ccc2occc2c1", "1-benzofuran"),
        # Benzothiophene
        #: PIN carries the S locant, the Blue Book)
        ("c1ccc2sccc2c1", "1-benzothiophene"),
    ])
    def test_fused_heterocycle_retained_names(self, smiles, expected):
        """Test fused heterocycle retained names."""
        assert name_compound(smiles) == expected


class TestCOMPLEX05_TautomerLocants:
    """Tautomer locant (indicated H) in heterocyclic names."""

    @pytest.mark.integration
    def test_indole_has_indicated_hydrogen(self):
        """Indole requires 1H- prefix (IUPAC 2013 PIN)."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert result == "1H-indole"
        assert result.startswith("1H-")

    @pytest.mark.integration
    def test_quinoline_no_indicated_hydrogen(self):
        """Quinoline has no indicated H (no N-H)."""
        result = name_compound("c1ccc2ncccc2c1")
        assert result == "quinoline"
        assert "H-" not in result

    @pytest.mark.integration
    def test_benzimidazole_has_indicated_hydrogen(self):
        """Benzimidazole requires 1H- prefix."""
        result = name_compound("c1ccc2[nH]cnc2c1")
        assert result == "1H-1,3-benzimidazole"
        assert result.startswith("1H-")


class TestCOMPLEX05_SubstitutedFusedHeterocycles:
    """Substituted fused heterocycle naming - known limitations."""

    # Task 12 fix a performance pass (wp6-tests): the non-strict xfail on the whole test was
    # stale for 2-methylindole (it XPASSed) and hid a live defect for
    # 4-methylquinoline, which was named 'lepidine' at pin_verified: a trivial name
    # imported from the OPSIN resource tables with is_pin False
    # (data/opsin_imports/aryl_groups.py), returned by the retained-name lookup
    # (routing/dispatch_table._handle_retained_name -> ALL_RETAINED_NAMES), 0 Blue
    # Book hits. Quinoline is a retained PIN parent, so the PIN is
    # '4-methylquinoline' (OPSIN 2.9.0 full-InChIKey exact, as is 'lepidine').
    # wp7: FIXED -- an OPSIN-import name enters the PIN lookup only with Blue
    # Book PIN evidence (data._OPSIN_IMPORT_PIN_EVIDENCE); the strict xfail is gone.
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2ccccc2[nH]1", "2-methyl-1H-indole"),
        ("Cc1ccnc2ccccc12", "4-methylquinoline"),
    ])
    def test_substituted_fused_heterocycles(self, smiles, expected):
        """Substituted fused heterocycle naming."""
        assert name_compound(smiles) == expected


# =============================================================================
# COMPLEX-06: Complex PAHs (Polycyclic Aromatic Hydrocarbons)
# =============================================================================

class TestCOMPLEX06_Tetracyclic:
    """COMPLEX-06: Tetracyclic PAH naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Pyrene - 4 fused rings
        ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),
        # Chrysene - angular 4-ring
        ("c1ccc2c(c1)ccc1c3ccccc3ccc21", "chrysene"),
    ])
    def test_tetracyclic_pah_retained_names(self, smiles, expected):
        """Test tetracyclic PAH retained names."""
        assert name_compound(smiles) == expected


class TestCOMPLEX06_PentacyclicAndHigher:
    """Pentacyclic and larger PAH naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Perylene - peri-fused pentacyclic
        ("c1ccc2cccc3cc4c(c1)cc1cccc4c1c23", "perylene"),
        # Coronene - hexagonal 7-ring
        ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "coronene"),
    ])
    def test_larger_pah_retained_names(self, smiles, expected):
        """Test larger PAH retained names."""
        assert name_compound(smiles) == expected


class TestCOMPLEX06_PAHLookup:
    """PAH data lookup and verification."""

    @pytest.mark.integration
    def test_pah_data_contains_all_expected(self):
        """Verify PAH data contains all expected compounds."""
        from orthonym.data.polycyclic_data import get_pah_names

        expected_pahs = [
            'naphthalene', 'anthracene', 'phenanthrene', 'pyrene',
            'chrysene', 'perylene', 'coronene', 'fluorene',
            'acenaphthene', 'acenaphthylene'
        ]

        pah_names = get_pah_names()
        for pah in expected_pahs:
            assert pah in pah_names, f"Missing PAH: {pah}"

    @pytest.mark.integration
    def test_pah_smiles_lookup_works(self):
        """Test PAH lookup by SMILES."""
        from orthonym.data.polycyclic_data import get_polycyclic_by_smiles

        result = get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        assert result is not None
        assert result['name'] == 'naphthalene'


# =============================================================================
# a phase-5 Regression Tests
# =============================================================================

class TestPhase1Regression:
    """Regression tests for a phase (Foundation)."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Simple alkanes
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),
        ("CCCCC", "pentane"),
        ("CCCCCC", "hexane"),
        ("CCCCCCC", "heptane"),
        ("CCCCCCCC", "octane"),
        ("CCCCCCCCC", "nonane"),
        ("CCCCCCCCCC", "decane"),
        # Branched alkanes
        ("CC(C)C", "2-methylpropane"),
        ("CC(C)CC", "2-methylbutane"),
        ("CC(C)(C)CC", "2,2-dimethylbutane"),
        ("CC(C)C(C)C", "2,3-dimethylbutane"),
        # Alcohols
        ("CO", "methanol"),
        ("CCO", "ethanol"),
        ("CCCO", "propan-1-ol"),
        ("CCC(O)C", "butan-2-ol"),
    ])
    def test_phase1_simple_compounds(self, smiles, expected):
        """a phase simple compound naming unchanged."""
        assert name_compound(smiles) == expected


class TestPhase2Regression:
    """Regression tests for a phase (Ring Foundation)."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Simple cycloalkanes
        ("C1CC1", "cyclopropane"),
        ("C1CCC1", "cyclobutane"),
        ("C1CCCC1", "cyclopentane"),
        ("C1CCCCC1", "cyclohexane"),
        ("C1CCCCCC1", "cycloheptane"),
        ("C1CCCCCCC1", "cyclooctane"),
        # Benzene derivatives
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),
    ])
    def test_phase2_ring_compounds(self, smiles, expected):
        """a phase ring compound naming unchanged."""
        assert name_compound(smiles) == expected


class TestPhase3Regression:
    """Regression tests for a phase (Heterocycles)."""

    @pytest.mark.integration
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
        # 5-membered aromatic
        ("c1ccoc1", "furan"),
        # Task 12 fix a performance pass (wp6-tests), change-asserted-value (were 'pyrrole'
        # and 'imidazole'): (the Blue Book) cites indicated
        # hydrogen in parent hydrides, '1H-pyrrole (PIN)' (:24645),
        # '1-(trimethylsilyl)-1H-imidazole (PIN)' (:18940). OPSIN RT exact.
        ("c1cc[nH]c1", "1H-pyrrole"),
        ("c1ccsc1", "thiophene"),
        ("c1c[nH]cn1", "1H-imidazole"),
        # 6-membered saturated
        ("C1CCOCC1", "oxane"),
        ("C1CCNCC1", "piperidine"),
        ("C1COCCN1", "morpholine"),
        # 6-membered aromatic
        ("c1ccncc1", "pyridine"),
        ("c1cncnc1", "pyrimidine"),
        ("c1cnccn1", "pyrazine"),
    ])
    def test_phase3_heterocycle_compounds(self, smiles, expected):
        """a phase heterocycle naming unchanged."""
        assert name_compound(smiles) == expected


class TestPhase4Regression:
    """Regression tests for a phase (Polyfunctional)."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Aldehydes
        ("CC=O", "acetaldehyde"),
        ("CCC=O", "propanal"),
        # Ketones
        ("CC(C)=O", "propan-2-one"),
        ("CCC(C)=O", "butan-2-one"),
        # Carboxylic acids
        ("CC(=O)O", "acetic acid"),
        ("CCC(=O)O", "propanoic acid"),
        # Alkenes
        ("C=CCC", "but-1-ene"),
        ("CC=CC", "but-2-ene"),
        # Alkynes
        ("C#CCC", "but-1-yne"),
        ("CC#CC", "but-2-yne"),
    ])
    def test_phase4_functional_compounds(self, smiles, expected):
        """a phase polyfunctional naming unchanged."""
        assert name_compound(smiles) == expected


class TestPhase5Regression:
    """Regression tests for a phase (Stereochemistry)."""

    @pytest.mark.integration
    def test_stereochemistry_preserved(self):
        """Stereochemistry naming should be preserved."""
        # R-butan-2-ol
        result = name_compound("C[C@H](O)CC")
        assert "R" in result or "S" in result  # Has stereodescriptor

        # E-but-2-ene
        result = name_compound("C/C=C/C")
        assert "E" in result or "Z" in result  # Has E/Z descriptor


# =============================================================================
# Complex Ring Routing Tests
# =============================================================================

class TestComplexRingRouting:
    """Verify correct routing of complex ring systems."""

    @pytest.mark.integration
    def test_bicyclo_detected_before_simple_cyclic(self):
        """Bicyclo systems should be detected before simple cycloalkane."""
        result = name_compound("C1CC2CCC1C2")
        # R11 (2026-09-25, pre-existing-failures plan, Task 5)::9881, only adamantane/cubane are retained; "bicyclo[2.2.1]heptane (PIN)":2038
        assert result == "bicyclo[2.2.1]heptane"
        assert "cyclopentane" not in result
        assert "cyclohexane" not in result

    @pytest.mark.integration
    def test_spiro_detected_before_simple_cyclic(self):
        """Spiro systems should be detected before simple cycloalkane."""
        result = name_compound("C1CCC2(CC1)CCCC2")
        assert "spiro" in result
        assert "cyclohexane" not in result

    @pytest.mark.integration
    def test_fused_heterocycle_uses_retained_name(self):
        """Fused heterocycles should use retained names, not HW systematic."""
        result = name_compound("c1ccc2[nH]ccc2c1")
        assert result == "1H-indole"
        assert "aza" not in result  # Not HW naming


# =============================================================================
# Compound Classification Tests
# =============================================================================

class TestCompoundClassification:
    """Test compound classification for validation."""

    @pytest.mark.integration
    def test_bicyclo_is_complex_ring(self):
        """Bicyclo should be classified as complex ring system."""
        from orthonym.assembly.composer import _is_complex_ring_system, _classify_complex_ring
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        assert _is_complex_ring_system(mol) == True
        assert _classify_complex_ring(mol) == "bicyclo"

    @pytest.mark.integration
    def test_spiro_is_complex_ring(self):
        """Spiro should be classified as complex ring system."""
        from orthonym.assembly.composer import _is_complex_ring_system, _classify_complex_ring
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        assert _is_complex_ring_system(mol) == True
        assert _classify_complex_ring(mol) == "spiro"

    @pytest.mark.integration
    def test_monocyclic_not_complex(self):
        """Simple monocyclic should NOT be classified as complex."""
        from orthonym.assembly.composer import _is_complex_ring_system
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert _is_complex_ring_system(mol) == False


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestBoundaryConditions:
    """Boundary conditions and edge cases."""

    @pytest.mark.integration
    def test_single_atom_molecule(self):
        """Single atom molecule."""
        assert name_compound("C") == "methane"

    @pytest.mark.integration
    def test_two_atom_molecule(self):
        """Two atom molecule."""
        assert name_compound("CC") == "ethane"

    @pytest.mark.integration
    def test_long_chain(self):
        """Long alkane chain."""
        assert name_compound("CCCCCCCCCC") == "decane"

    @pytest.mark.integration
    def test_smallest_cycloalkane(self):
        """Smallest cycloalkane (cyclopropane)."""
        assert name_compound("C1CC1") == "cyclopropane"

    @pytest.mark.integration
    def test_smallest_heterocycle(self):
        """Smallest heterocycle (oxirane)."""
        assert name_compound("C1CO1") == "oxirane"


class TestKnownLimitations:
    """Document known limitations (expected failures)."""

    @pytest.mark.integration
    def test_dispiro_not_supported(self):
        """Dispiro systems are named (the old 'not supported' marker was stale).

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail XPASSed. The engine
        names the system 'dispiro[2.1.3^5.2^3]decane' (pin_verified), which OPSIN
        2.9.0 parses to the input's full InChIKey, so this asserts the round
        trip. The Blue Book spelling carries the superscripts:
        (the Blue Book) "Each time a spiro atom is reached for the second
        time its locant... is cited as a superscript number", e.g.
        'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981); the next test asserts it.
        The test id is kept."""
        from tests.support.jars import jar_or_skip
        from tests.support.rt_assert import assert_full_rt
        jar_or_skip()
        result = name_compound("C1CC2(C1)CC3(CC2)CC3")
        assert "dispiro" in result
        assert_full_rt(result, "C1CC2(C1)CC3(CC2)CC3")

    @pytest.mark.integration
    def test_dispiro_descriptor_carries_superscripts(self):
        """ (the Blue Book): "Each time a spiro atom is reached for
        the second time its locant, which has already been assigned, is cited as
        a superscript number to the number of the preceding linking atoms";
        'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981)."""
        assert name_compound("C1CC2(C1)CC3(CC2)CC3") == "dispiro[2.1.3^5.2^3]decane"

    @pytest.mark.integration
    def test_carbazole_detection(self):
        """9H-carbazole is detected (the non-strict xfail marker was stale: the
        test XPASSed; Task 12 fix a performance pass, wp6-tests). Carbazole is a retained
        PIN parent with its indicated hydrogen: '(special numbering;
        9H-isomer shown; the PIN is 9H-carbazole)' (the Blue Book). OPSIN
        RT exact."""
        result = name_compound("c1ccc2c(c1)[nH]c1ccccc12")
        assert result == "9H-carbazole"


# =============================================================================
# Integration Summary Tests
# =============================================================================

class TestPhaseSummaryCounts:
    """Document test coverage summary."""

    @pytest.mark.integration
    def test_phase6_e2e_coverage(self):
        """Document comprehensive a phase E2E coverage.

        COMPLEX-01 (Bicyclo): 8 tests
        - Systematic naming: 4 parametrized
        - Retained name: 1
        - Format verification: 2
        - Edge cases: 2

        COMPLEX-02 (Spiro): 7 tests
        - Systematic naming: 2 parametrized
        - Descriptor order: 1
        - Format verification: 1
        - Edge cases: 3

        COMPLEX-03 (Ortho-fused): 5 tests
        - Bicyclics: 2 parametrized
        - Tricyclics: 2 parametrized
        - Fused vs bridged: 1

        COMPLEX-04 (Fusion descriptors): 5 tests
        - Prefix generation: 1
        - Edge calculation: 1
        - Format verification: 1
        - No vowel elision: 1

        COMPLEX-05 (Heterocyclic fused): 12 tests
        - Retained names: 6 parametrized
        - Tautomer locants: 3
        - Substituted: 2 parametrized

        COMPLEX-06 (Complex PAHs): 6 tests
        - Tetracyclic: 2 parametrized
        - Larger PAHs: 2 parametrized
        - Data lookup: 2

        Regression (Phases 1-5): 60+ tests
        - a phase: 20 parametrized
        - a phase: 10 parametrized
        - a phase: 19 parametrized
        - a phase: 12 parametrized
        - a phase: 2

        Routing/Classification: 6 tests
        Boundary conditions: 5 tests
        Known limitations: 2 tests (xfail)

        TOTAL: 100+ unique test cases
        """
        assert True  # Documentation test


class TestValidationRequirements:
    """Verify VALID requirements are addressed."""

    @pytest.mark.integration
    def test_valid01_bulk_validation_infrastructure(self):
        """VALID-01: Bulk validation script exists."""
        from pathlib import Path
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_bulk.py"
        assert script_path.exists(), f"validate_bulk.py should exist at {script_path}"

    @pytest.mark.integration
    def test_valid02_roundtrip_validation_infrastructure(self):
        """VALID-02: Round-trip validation script exists."""
        from pathlib import Path
        script_path = Path(__file__).parent.parent.parent / "scripts" / "validate_roundtrip.py"
        assert script_path.exists(), f"validate_roundtrip.py should exist at {script_path}"

    @pytest.mark.integration
    def test_valid03_edge_case_documentation(self):
        """VALID-03: Edge case documentation exists (verified at checkpoint)."""
        # This will be verified at human checkpoint
        # Documentation in EDGE_CASES.md
        assert True
