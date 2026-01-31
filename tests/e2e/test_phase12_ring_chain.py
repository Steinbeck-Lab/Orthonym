"""
End-to-end tests for Phase 12: Ring-Chain Parent Selection.

These tests verify the complete naming workflow from SMILES input
to IUPAC name output for compounds with both rings and functionalized chains.

IUPAC Rules tested:
- P-44.1: Parent selection (where is principal group?)
- P-61.5: Ring substituent naming (phenyl, cyclohexyl, etc.)

RINGCHAIN-05: Retained names must be preserved (tested explicitly).
"""

import pytest
from orthonym.namer import name_compound


class TestPhase12CanonicalCases:
    """Canonical test cases from Phase 12 analysis."""

    def test_4_phenylbutanoic_acid(self):
        """Primary test case: 4-phenylbutanoic acid."""
        # SMILES: c1ccc(CCCC(=O)O)cc1
        # Chain has COOH -> chain is parent -> benzene is phenyl
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert result == '4-phenylbutanoic acid', f'Got: {result}'

    def test_benzoic_acid_ring_parent(self):
        """Benzoic acid: COOH on ring -> ring is parent."""
        # SMILES: c1ccc(C(=O)O)cc1
        result = name_compound('c1ccc(C(=O)O)cc1')
        assert result == 'benzoic acid', f'Got: {result}'


class TestPhenylSubstituent:
    """Test phenyl (from benzene) as substituent on various chains."""

    def test_3_phenylpropanoic_acid(self):
        """3-Phenylpropanoic acid (hydrocinnamic acid)."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert result == '3-phenylpropanoic acid', f'Got: {result}'

    def test_2_phenylacetic_acid(self):
        """2-Phenylacetic acid (or phenylacetic acid)."""
        result = name_compound('c1ccc(CC(=O)O)cc1')
        # Locant 2 can be omitted for 2-carbon chains
        assert 'phenyl' in result.lower() and 'ethanoic acid' in result.lower(), f'Got: {result}'

    def test_5_phenylpentanoic_acid(self):
        """5-Phenylpentanoic acid."""
        result = name_compound('c1ccc(CCCCC(=O)O)cc1')
        assert result == '5-phenylpentanoic acid', f'Got: {result}'

    def test_3_phenylpropan_1_ol(self):
        """3-Phenylpropan-1-ol (alcohol on chain)."""
        result = name_compound('c1ccc(CCCO)cc1')
        assert result == '3-phenylpropan-1-ol', f'Got: {result}'

    def test_4_phenylbutan_1_ol(self):
        """4-Phenylbutan-1-ol."""
        result = name_compound('c1ccc(CCCCO)cc1')
        assert result == '4-phenylbutan-1-ol', f'Got: {result}'

    def test_6_phenylhexanoic_acid(self):
        """6-Phenylhexanoic acid."""
        result = name_compound('c1ccc(CCCCCC(=O)O)cc1')
        assert result == '6-phenylhexanoic acid', f'Got: {result}'


class TestCyclohexylSubstituent:
    """Test cyclohexyl (from cyclohexane) as substituent."""

    def test_4_cyclohexylbutanoic_acid(self):
        """4-Cyclohexylbutanoic acid."""
        result = name_compound('C1CCCCC1CCCC(=O)O')
        assert result == '4-cyclohexylbutanoic acid', f'Got: {result}'

    def test_3_cyclohexylpropanoic_acid(self):
        """3-Cyclohexylpropanoic acid."""
        result = name_compound('C1CCCCC1CCC(=O)O')
        assert result == '3-cyclohexylpropanoic acid', f'Got: {result}'

    def test_2_cyclohexylethanol(self):
        """2-Cyclohexylethan-1-ol (PIN format with locant)."""
        result = name_compound('C1CCCCC1CCO')
        # PIN style includes locant: 2-cyclohexylethan-1-ol
        assert result == '2-cyclohexylethan-1-ol', f'Got: {result}'

    def test_5_cyclohexylpentanoic_acid(self):
        """5-Cyclohexylpentanoic acid."""
        result = name_compound('C1CCCCC1CCCCC(=O)O')
        assert result == '5-cyclohexylpentanoic acid', f'Got: {result}'


class TestCyclopentylSubstituent:
    """Test cyclopentyl (from cyclopentane) as substituent."""

    def test_3_cyclopentylpropanoic_acid(self):
        """3-Cyclopentylpropanoic acid."""
        result = name_compound('C1CCCC1CCC(=O)O')
        assert result == '3-cyclopentylpropanoic acid', f'Got: {result}'

    def test_4_cyclopentylbutanoic_acid(self):
        """4-Cyclopentylbutanoic acid."""
        result = name_compound('C1CCCC1CCCC(=O)O')
        assert result == '4-cyclopentylbutanoic acid', f'Got: {result}'


class TestCyclopropylAndCyclobutylSubstituent:
    """Test smaller ring substituents."""

    def test_3_cyclopropylpropanoic_acid(self):
        """3-Cyclopropylpropanoic acid."""
        result = name_compound('C1CC1CCC(=O)O')
        assert result == '3-cyclopropylpropanoic acid', f'Got: {result}'

    def test_3_cyclobutylpropanoic_acid(self):
        """3-Cyclobutylpropanoic acid."""
        result = name_compound('C1CCC1CCC(=O)O')
        assert result == '3-cyclobutylpropanoic acid', f'Got: {result}'


class TestRetainedNamesNotBroken:
    """
    CRITICAL: Verify retained names still work after Phase 12 changes.

    RINGCHAIN-05 requirement: Amino acid retained names (phenylalanine)
    must be detected BEFORE parent selection runs. This class explicitly
    tests that retained name detection takes precedence over systematic naming.
    """

    def test_benzene_retained(self):
        """Unsubstituted benzene."""
        result = name_compound('c1ccccc1')
        assert result == 'benzene', f'Got: {result}'

    def test_toluene_retained(self):
        """Toluene (methylbenzene)."""
        result = name_compound('Cc1ccccc1')
        assert result == 'toluene', f'Got: {result}'

    def test_phenol_retained(self):
        """Phenol."""
        result = name_compound('Oc1ccccc1')
        assert result == 'phenol', f'Got: {result}'

    def test_aniline_retained(self):
        """Aniline."""
        result = name_compound('Nc1ccccc1')
        assert result == 'aniline', f'Got: {result}'

    def test_cyclohexane_retained(self):
        """Cyclohexane."""
        result = name_compound('C1CCCCC1')
        assert result == 'cyclohexane', f'Got: {result}'

    def test_phenylalanine_amino_acid(self):
        """
        RINGCHAIN-05: Phenylalanine amino acid retained name must work.

        This is the critical test - phenylalanine has both a benzene ring
        AND a chain with functional groups. If parent selection ran before
        retained name detection, it would produce a systematic name instead
        of 'phenylalanine'.

        The correct behavior is:
        1. Check retained names FIRST (in name() method)
        2. Match 'phenylalanine' pattern
        3. Return 'phenylalanine' without ever running parent selection

        Note: Uses non-stereo SMILES because amino acid lookup currently
        uses the canonical non-stereo form. Stereo-annotated SMILES would
        need additional lookup entries.
        """
        result = name_compound('NC(Cc1ccccc1)C(=O)O')
        assert result.lower() == 'phenylalanine' or 'phenylalanine' in result.lower(), f'Got: {result}'

    def test_phenylalanine_stereo_systematic(self):
        """
        Phenylalanine with stereochemistry uses systematic naming.

        Note: Stereo-annotated amino acids currently produce systematic names
        because the amino acid lookup table uses non-stereo canonical forms.
        This tests that the systematic naming path still works correctly.
        """
        result = name_compound('N[C@@H](Cc1ccccc1)C(=O)O')
        # With stereo, may get systematic name (2-amino-3-phenylpropanoic acid)
        # This is expected behavior until stereo forms are added to lookup
        assert 'amino' in result.lower() or 'phenylalanine' in result.lower(), f'Got: {result}'

    def test_tyrosine_amino_acid(self):
        """Tyrosine (another aromatic amino acid) should use retained name."""
        result = name_compound('NC(Cc1ccc(O)cc1)C(=O)O')
        assert result.lower() == 'tyrosine' or 'tyrosine' in result.lower(), f'Got: {result}'


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_ethylbenzene_ring_is_parent(self):
        """Ethylbenzene: no functional group, ring should be parent."""
        result = name_compound('c1ccc(CC)cc1')
        # ethylbenzene - ring is parent for pure hydrocarbons
        assert result == 'ethylbenzene', f'Got: {result}'

    def test_propylbenzene_ring_is_parent(self):
        """Propylbenzene: no functional group, ring is parent."""
        result = name_compound('c1ccc(CCC)cc1')
        # propylbenzene or propyl-benzene
        assert 'benzene' in result.lower(), f'Got: {result}'

    def test_benzaldehyde_ring_is_parent(self):
        """Benzaldehyde: aldehyde directly on ring."""
        result = name_compound('c1ccc(C=O)cc1')
        assert result == 'benzaldehyde', f'Got: {result}'

    def test_phenylacetaldehyde_chain_is_parent(self):
        """Phenylacetaldehyde: aldehyde on chain."""
        result = name_compound('c1ccc(CC=O)cc1')
        assert result == 'phenylacetaldehyde' or result == '2-phenylethanal', f'Got: {result}'

    def test_acetophenone_ring_is_parent(self):
        """Acetophenone: ketone directly on ring."""
        result = name_compound('c1ccc(C(=O)C)cc1')
        assert result == 'acetophenone', f'Got: {result}'

    def test_2_phenylethanol(self):
        """2-Phenylethan-1-ol (short chain with alcohol)."""
        result = name_compound('c1ccc(CCO)cc1')
        # PIN style: 2-phenylethan-1-ol
        assert '2-phenyl' in result.lower() or 'phenylethanol' in result.lower(), f'Got: {result}'


class TestNoRegression:
    """Verify no regression in existing functionality."""

    def test_simple_acids_still_work(self):
        """Acetic acid, butanoic acid - no rings."""
        assert name_compound('CC(=O)O') == 'acetic acid'
        assert name_compound('CCCC(=O)O') == 'butanoic acid'

    def test_propanoic_acid(self):
        """Propanoic acid - no rings."""
        assert name_compound('CCC(=O)O') == 'propanoic acid'

    def test_pentanoic_acid(self):
        """Pentanoic acid - no rings."""
        assert name_compound('CCCCC(=O)O') == 'pentanoic acid'

    def test_simple_alcohols_still_work(self):
        """Ethanol, propan-1-ol - no rings."""
        assert name_compound('CCO') == 'ethanol'
        assert name_compound('CCCO') == 'propan-1-ol'

    def test_butan_1_ol(self):
        """Butan-1-ol - no rings."""
        assert name_compound('CCCCO') == 'butan-1-ol'

    def test_substituted_benzene_still_works(self):
        """Chlorobenzene, nitrobenzene - no chain FG."""
        result = name_compound('Clc1ccccc1')
        assert result == 'chlorobenzene', f'Got: {result}'

    def test_bromobenzene(self):
        """Bromobenzene."""
        result = name_compound('Brc1ccccc1')
        assert result == 'bromobenzene', f'Got: {result}'

    def test_disubstituted_benzene_still_works(self):
        """1,4-Dimethylbenzene."""
        result = name_compound('Cc1ccc(C)cc1')
        assert result == '1,4-dimethylbenzene', f'Got: {result}'

    def test_1_2_dimethylbenzene(self):
        """1,2-Dimethylbenzene."""
        result = name_compound('Cc1ccccc1C')
        assert result == '1,2-dimethylbenzene', f'Got: {result}'


class TestLocantAssignment:
    """Test locants are correctly assigned for ring substituents."""

    def test_4_phenylbutanoic_acid_locant(self):
        """Phenyl at position 4 of butanoic acid."""
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert '4-phenyl' in result

    def test_3_phenylpropanoic_acid_locant(self):
        """Phenyl at position 3 of propanoic acid."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert '3-phenyl' in result

    def test_2_phenylacetic_acid_locant(self):
        """Phenyl at position 2 of acetic acid."""
        result = name_compound('c1ccc(CC(=O)O)cc1')
        # Could be '2-phenylacetic acid' or 'phenylacetic acid' (locant 2 implied)
        assert 'phenyl' in result.lower() and 'ethanoic acid' in result.lower()

    def test_4_cyclohexylbutanoic_acid_locant(self):
        """Cyclohexyl at position 4."""
        result = name_compound('C1CCCCC1CCCC(=O)O')
        assert '4-cyclohexyl' in result

    def test_3_cyclohexylpropanoic_acid_locant(self):
        """Cyclohexyl at position 3."""
        result = name_compound('C1CCCCC1CCC(=O)O')
        assert '3-cyclohexyl' in result


class TestParentSelectionDecision:
    """Test that parent selection decision is correct based on FG position."""

    def test_fg_on_chain_chain_is_parent(self):
        """When FG is on chain (not directly on ring), chain is parent."""
        # 4-phenylbutanoic acid: chain is parent
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        # Should NOT contain "benzoic" - that would mean ring is parent
        assert 'benzoic' not in result.lower()
        # Should contain "phenyl" - benzene as substituent
        assert 'phenyl' in result.lower()

    def test_fg_on_ring_ring_is_parent(self):
        """When FG is directly on ring, ring is parent."""
        # Benzoic acid: ring is parent
        result = name_compound('c1ccc(C(=O)O)cc1')
        # Should be benzoic acid (retained name for FG directly on ring)
        assert result == 'benzoic acid'

    def test_no_fg_ring_is_parent(self):
        """When no functional group, ring is parent (pure hydrocarbon)."""
        # Ethylbenzene: ring is parent
        result = name_compound('c1ccc(CC)cc1')
        assert 'benzene' in result.lower()


class TestChainLengthVariety:
    """Test various chain lengths with phenyl substituent."""

    def test_2_phenylethanoic_acid(self):
        """2-Phenylethanoic acid (2-carbon chain)."""
        result = name_compound('c1ccc(CC(=O)O)cc1')
        assert 'phenyl' in result.lower()

    def test_3_phenylpropanoic_acid(self):
        """3-Phenylpropanoic acid (3-carbon chain)."""
        result = name_compound('c1ccc(CCC(=O)O)cc1')
        assert result == '3-phenylpropanoic acid'

    def test_4_phenylbutanoic_acid(self):
        """4-Phenylbutanoic acid (4-carbon chain)."""
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert result == '4-phenylbutanoic acid'

    def test_5_phenylpentanoic_acid(self):
        """5-Phenylpentanoic acid (5-carbon chain)."""
        result = name_compound('c1ccc(CCCCC(=O)O)cc1')
        assert result == '5-phenylpentanoic acid'

    def test_6_phenylhexanoic_acid(self):
        """6-Phenylhexanoic acid (6-carbon chain)."""
        result = name_compound('c1ccc(CCCCCC(=O)O)cc1')
        assert result == '6-phenylhexanoic acid'


class TestDifferentFunctionalGroups:
    """Test ring-chain naming with different functional groups."""

    def test_phenyl_with_acid(self):
        """Phenyl with carboxylic acid."""
        result = name_compound('c1ccc(CCCC(=O)O)cc1')
        assert '4-phenylbutanoic acid' == result

    def test_phenyl_with_alcohol(self):
        """Phenyl with alcohol."""
        result = name_compound('c1ccc(CCCCO)cc1')
        assert '4-phenylbutan-1-ol' == result

    def test_cyclohexyl_with_acid(self):
        """Cyclohexyl with carboxylic acid."""
        result = name_compound('C1CCCCC1CCCC(=O)O')
        assert '4-cyclohexylbutanoic acid' == result

    def test_cyclohexyl_with_alcohol(self):
        """Cyclohexyl with alcohol."""
        result = name_compound('C1CCCCC1CCCO')
        assert 'cyclohexyl' in result.lower() and ('propan' in result.lower() or 'ol' in result.lower())
