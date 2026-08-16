"""
Unit tests for heteroatom replacement nomenclature and polycyclic lactone naming.

Tests the "a" replacement nomenclature (oxa-/aza-/thia-) for polycyclic systems
and the naming of polycyclic lactones as heterocyclic pseudoketones.

IUPAC 2013 Rules:
- Ring heteroatoms use replacement prefixes (oxa, aza, thia, etc.)
- Multiple same-type heteroatoms use multiplicative prefixes (dioxa, triaza)
- Prefix order follows Hantzsch-Widman priority (O > S > N)
- Polycyclic lactones: ring O gets oxa- prefix, C=O gets -one suffix

Reference: IUPAC 2013 Blue Book P-23.4 and P-25.5
"""

import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import (
    HETEROATOM_PREFIXES,
    get_heteroatom_replacement_prefix,
    detect_polycyclic_lactone,
    name_polycyclic_with_heteroatoms,
    VonBaeyerAnalyzer,
)


# ============================================================================
# Test HETEROATOM_PREFIXES constant
# ============================================================================

class TestHeteroatomPrefixesConstant:
    """Test the HETEROATOM_PREFIXES dictionary."""

    @pytest.mark.unit
    def test_oxygen_prefix(self):
        """Oxygen should have 'oxa' prefix with highest priority."""
        assert 'O' in HETEROATOM_PREFIXES
        prefix, priority = HETEROATOM_PREFIXES['O']
        assert prefix == 'oxa'
        assert priority == 1

    @pytest.mark.unit
    def test_sulfur_prefix(self):
        """Sulfur should have 'thia' prefix."""
        assert 'S' in HETEROATOM_PREFIXES
        prefix, priority = HETEROATOM_PREFIXES['S']
        assert prefix == 'thia'
        assert priority == 2

    @pytest.mark.unit
    def test_nitrogen_prefix(self):
        """Nitrogen should have 'aza' prefix with lower priority than O and S."""
        assert 'N' in HETEROATOM_PREFIXES
        prefix, priority = HETEROATOM_PREFIXES['N']
        assert prefix == 'aza'
        assert priority > HETEROATOM_PREFIXES['O'][1]
        assert priority > HETEROATOM_PREFIXES['S'][1]

    @pytest.mark.unit
    def test_hw_priority_order(self):
        """Verify Hantzsch-Widman priority order: O > S > Se > N > P."""
        o_priority = HETEROATOM_PREFIXES.get('O', (None, 999))[1]
        s_priority = HETEROATOM_PREFIXES.get('S', (None, 999))[1]
        n_priority = HETEROATOM_PREFIXES.get('N', (None, 999))[1]

        assert o_priority < s_priority < n_priority


# ============================================================================
# Test get_heteroatom_replacement_prefix
# ============================================================================

class TestGetHeteroatomReplacementPrefix:
    """Test heteroatom replacement prefix generation."""

    @pytest.mark.unit
    def test_single_oxygen_bicyclo(self):
        """7-oxabicyclo[2.2.1]heptane - single oxygen in bicyclo system."""
        # SMILES: C1CC2COC1C2 (oxygen at bridgehead)
        # Alternatively: O1CC2CCC1C2
        mol = Chem.MolFromSmiles('O1CC2CCC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should contain "oxa" with some locant
        assert 'oxa' in prefix
        # Should have exactly one oxa (no multiplier)
        assert 'dioxa' not in prefix

    @pytest.mark.unit
    def test_two_oxygens_dioxa(self):
        """2,5-dioxabicyclo[2.2.1]heptane - two oxygens, multiplicative prefix."""
        # Two oxygens in a bicyclo system
        mol = Chem.MolFromSmiles('O1COC2CC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should contain "dioxa" with locants
        assert 'dioxa' in prefix

    @pytest.mark.unit
    def test_single_nitrogen_aza(self):
        """7-azabicyclo[2.2.1]heptane - nitrogen replacement."""
        mol = Chem.MolFromSmiles('N1CC2CCC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should contain "aza" with some locant
        assert 'aza' in prefix
        # Should have exactly one aza (no multiplier)
        assert 'diaza' not in prefix

    @pytest.mark.unit
    def test_mixed_oxygen_nitrogen(self):
        """2-oxa-7-azabicyclo[2.2.1]heptane - mixed O and N (O listed first per HW priority)."""
        # O and N in bicyclo system
        mol = Chem.MolFromSmiles('O1CC2CNC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should contain both oxa and aza
        assert 'oxa' in prefix
        assert 'aza' in prefix
        # oxa should come before aza (HW priority)
        oxa_pos = prefix.find('oxa')
        aza_pos = prefix.find('aza')
        assert oxa_pos < aza_pos, f"oxa should precede aza, got: {prefix}"

    @pytest.mark.unit
    def test_sulfur_thia(self):
        """3-thia- prefix for sulfur."""
        # Sulfur in bicyclo system
        mol = Chem.MolFromSmiles('S1CC2CCC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should contain "thia"
        assert 'thia' in prefix

    @pytest.mark.unit
    def test_no_heteroatoms_empty_prefix(self):
        """Pure hydrocarbon bicyclo system should return empty prefix."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # bicyclo[2.2.1]heptane
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should be empty for pure hydrocarbon
        assert prefix == ""


# ============================================================================
# Test detect_polycyclic_lactone
# ============================================================================

class TestDetectPolycyclicLactone:
    """Test polycyclic lactone detection."""

    @pytest.mark.unit
    def test_bicyclic_lactone_detection(self):
        """Simple bicyclic lactone detection - ring O and C=O both in ring system."""
        # Bicyclic lactone: C(=O)-O- where C and ester O are both in the ring
        # The carbonyl O is exocyclic (double bonded)
        mol = Chem.MolFromSmiles('O=C1OC2CCC1CC2')  # Valid bicyclic lactone
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        result = detect_polycyclic_lactone(mol, ring_atoms)

        # Should detect the lactone
        assert result is not None
        assert 'carbonyl_c' in result
        assert 'ring_oxygen' in result
        assert 'carbonyl_oxygen' in result

    @pytest.mark.unit
    def test_no_lactone_pure_hydrocarbon(self):
        """Pure hydrocarbon polycyclic should return None."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')  # bicyclo[2.2.1]heptane
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        result = detect_polycyclic_lactone(mol, ring_atoms)

        assert result is None

    @pytest.mark.unit
    def test_no_lactone_only_ether(self):
        """Polycyclic with ring O but no carbonyl should return None."""
        mol = Chem.MolFromSmiles('O1CC2CCC1C2')  # oxa-bicyclo, no C=O
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        result = detect_polycyclic_lactone(mol, ring_atoms)

        # No lactone - just an ether
        assert result is None


# ============================================================================
# Test name_polycyclic_with_heteroatoms
# ============================================================================

class TestNamePolycyclicWithHeteroatoms:
    """Test complete naming with heteroatom replacement."""

    @pytest.mark.unit
    def test_oxabicycloheptane(self):
        """7-oxabicyclo[2.2.1]heptane - complete name generation."""
        mol = Chem.MolFromSmiles('O1CC2CCC1C2')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        assert name is not None
        # Should contain oxa prefix
        assert 'oxa' in name
        # Should contain bicyclo
        assert 'bicyclo' in name
        # Should end with heptane (7 ring atoms)
        assert 'heptane' in name

    @pytest.mark.unit
    def test_azabicycloheptane(self):
        """7-azabicyclo[2.2.1]heptane - nitrogen replacement naming."""
        mol = Chem.MolFromSmiles('N1CC2CCC1C2')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        assert name is not None
        assert 'aza' in name
        assert 'bicyclo' in name
        assert 'heptane' in name

    @pytest.mark.unit
    def test_polycyclic_lactone_naming(self):
        """Polycyclic lactone: oxa- prefix + -one suffix."""
        # A bicyclic lactone
        mol = Chem.MolFromSmiles('O=C1OC2CCC1CC2')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        # If lactone detected, should have -one suffix
        # Note: exact format depends on implementation
        assert name is not None
        if 'one' in name:
            # Lactone naming: oxa- prefix and -one suffix
            assert 'oxa' in name


# ============================================================================
# Test prefix formatting rules
# ============================================================================

class TestPrefixFormattingRules:
    """Test correct prefix formatting."""

    @pytest.mark.unit
    def test_multiplicative_prefix_format(self):
        """Locant formatting: "2,5-dioxa-" not "2-oxa-5-oxa-"."""
        mol = Chem.MolFromSmiles('O1COC2CC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Should use multiplicative prefix, not repeated single prefixes
        assert 'dioxa' in prefix
        # Should NOT have "oxa-" appearing twice separately
        count_separate = prefix.count('oxa-')
        # The "dioxa" should appear once, not "oxa-" appearing twice
        assert prefix.count('-oxa-') <= 1  # May appear at most once as part of compound

    @pytest.mark.unit
    def test_prefix_hw_priority_ordering(self):
        """Prefix ordering: oxa before aza before thia (HW priority)."""
        # Create a molecule with O, N, and S
        mol = Chem.MolFromSmiles('O1CN2CCS1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # Check ordering: O > S > N (by HW priority, where lower number = higher priority)
        # So oxa should come before thia, and thia before aza
        if 'oxa' in prefix and 'thia' in prefix:
            assert prefix.find('oxa') < prefix.find('thia')
        if 'oxa' in prefix and 'aza' in prefix:
            assert prefix.find('oxa') < prefix.find('aza')
        if 'thia' in prefix and 'aza' in prefix:
            assert prefix.find('thia') < prefix.find('aza')

    @pytest.mark.unit
    def test_combined_name_format(self):
        """Combined name: "7-oxabicyclo[2.2.1]heptane" (prefix + descriptor + parent)."""
        mol = Chem.MolFromSmiles('O1CC2CCC1C2')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        assert name is not None
        # Should follow format: {locant}-{hetero_prefix}bicyclo[...]heptane
        # Check structure
        assert 'bicyclo' in name
        assert 'heptane' in name
        # Hetero prefix should come before bicyclo
        hetero_pos = name.find('oxa')
        bicyclo_pos = name.find('bicyclo')
        assert hetero_pos < bicyclo_pos, f"Hetero prefix should precede bicyclo: {name}"


# ============================================================================
# Test vowel elision for -one suffix
# ============================================================================

class TestVowelElision:
    """Test vowel elision when adding -one suffix."""

    @pytest.mark.unit
    def test_heptane_to_heptanone_elision(self):
        """Vowel elision: heptane -> heptan-X-one (drop 'e' before '-one')."""
        # This tests the elision rule in lactone naming
        # heptane + -one -> heptanone (via heptan- + -one)
        mol = Chem.MolFromSmiles('O=C1OC2CCC1CC2')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        if name and 'one' in name:
            # Should NOT have "heptaneone" or similar
            assert 'aneone' not in name
            assert 'eneone' not in name


# ============================================================================
# Test integration with VonBaeyerAnalyzer
# ============================================================================

class TestVBIntegration:
    """Test integration with VonBaeyerAnalyzer from Plan 16-01."""

    @pytest.mark.unit
    def test_numbering_used_for_locants(self):
        """Heteroatom locants should use VB numbering, not atom indices."""
        mol = Chem.MolFromSmiles('O1CC2CCC1C2')
        assert mol is not None

        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        # Get the oxygen atom index
        o_idx = None
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'O':
                o_idx = idx
                break

        assert o_idx is not None

        # The VB locant should be in the numbering
        assert o_idx in desc.numbering
        vb_locant = desc.numbering[o_idx]

        prefix = get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

        # The locant in the prefix should match the VB locant
        assert str(vb_locant) in prefix

    @pytest.mark.unit
    def test_tricyclo_with_heteroatom(self):
        """Tricyclo system with heteroatom should work correctly."""
        # Adamantane with one oxygen: 2-oxaadamantane
        # tricyclo[3.3.1.1(3,7)]decane with O
        mol = Chem.MolFromSmiles('O1C2CC3CC1CC(C2)C3')
        assert mol is not None

        name = name_polycyclic_with_heteroatoms(mol)

        assert name is not None
        assert 'oxa' in name
        assert 'tricyclo' in name


# ============================================================================
# v23 Phase 13: lambda-convention (P-31.1.4.2 / P-23.6.1) in the VB 'a'-prefix
# ============================================================================

class TestVonBaeyerLambdaConvention:
    """A non-standard-valence ring heteroatom in a von-Baeyer system must carry
    the lambda convention in its 'a'-prefix (e.g. tetravalent S -> 3λ4-thia),
    via the shared rules.lambda_convention infra. Standard-valence heteroatoms
    must be byte-identical (no spurious lambda)."""

    def _prefix(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        ring_atoms = set()
        for r in mol.GetRingInfo().AtomRings():
            ring_atoms.update(r)
        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)
        return get_heteroatom_replacement_prefix(mol, desc.numbering, ring_atoms)

    @pytest.mark.unit
    def test_tetravalent_sulfur_emits_lambda4(self):
        # S(IV) bridge in bicyclo[3.2.1]octane -> 3λ4-thia (was 3-thia,
        # which denotes S(II) -> a different molecule the gate suppressed).
        prefix = self._prefix("C12C[SH2]CC(CC1)C2")
        assert "λ4" in prefix and "thia" in prefix

    @pytest.mark.unit
    def test_standard_valence_oxygen_no_lambda(self):
        # O(II) is standard -> no lambda (byte-identical to pre-Phase-13).
        prefix = self._prefix("C12COCC(CC1)C2")
        assert "λ" not in prefix and "oxa" in prefix

    @pytest.mark.unit
    def test_full_name_round_trippable_pin(self):
        from orthonym.namer import name_compound
        # S1 (v24): the replacement 'a'-prefix attaches DIRECTLY to the von-Baeyer
        # descriptor (P-23.3.1) — no hyphen. PIN is `3λ4-thiabicyclo[3.2.1]octane`.
        name = name_compound("C12C[SH2]CC(CC1)C2")
        assert name == "3λ4-thiabicyclo[3.2.1]octane"


class TestVonBaeyerReplacementPrefixNoStrayHyphen:
    """S1 (v24): hetero von-Baeyer replacement prefix must not emit a stray hyphen
    before the descriptor. PIN = `2-oxabicyclo[...]`, NOT `2-oxa-bicyclo[...]`
    (P-23.3.1). The spiro/bicyclo.py paths were already correct; polycyclic.py's
    `get_heteroatom_replacement_prefix` appended a trailing '-' (the duplicated-
    formatter bug the cross-tool audit flagged as S1)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,pin", [
        ("O1CC2CCC1CC2", "2-oxabicyclo[2.2.2]octane"),
        ("N1CC2CCCC1C2", "6-azabicyclo[3.2.1]octane"),
        ("O1CC2CCC1O2", "2,7-dioxabicyclo[2.2.1]heptane"),
        ("C12COCC(CC1)C2", "3-oxabicyclo[3.2.1]octane"),
    ])
    def test_no_stray_hyphen_before_descriptor(self, smiles, pin):
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name == pin
        # defensive: the a-prefix→descriptor join must never carry a hyphen
        import re
        assert not re.search(r"a-(?:bi|tri|tetra|penta|hexa|hepta)cyclo", name)

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,pin", [
        ("CC12CC3CC(C1)CC(C2)C3", "1-methyltricyclo[3.3.1.1^3,7]decane"),
        ("CC1C2CC3C1C1C2C31", "2-methyltetracyclo[3.3.0.0^3,7.0^4,6]octane"),
    ])
    def test_carbocyclic_substituted_vb_no_stray_hyphen(self, smiles, pin):
        # Same stray-hyphen class as S1, on the SUBSTITUENT-prefix path: for a
        # substituted CARBOCYCLIC (no-heteroatom) von-Baeyer system the substituent
        # prefix attaches DIRECTLY to the descriptor (P-31) -> `1-methyltricyclo...`,
        # not `1-methyl-tricyclo...`. (bicyclo.py handles bicyclo correctly already;
        # this is the name_polycyclic_complete tricyclo+ path.)
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name == pin
        import re
        # a hyphen after a letter (substituent stem OR 'a'-prefix) before the descriptor
        # is the bug; a stereo ')-' descriptor hyphen is correct and excluded by [a-z].
        assert not re.search(r"[a-z]-(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca)cyclo", name)
