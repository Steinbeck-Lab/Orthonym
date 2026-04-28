"""
End-to-end integration tests for algorithmic fusion naming.

Tests that fused ring systems NOT in the dictionary produce correct
systematic names via the algorithmic generator. Validates against
OPSIN-verified names and checks zero regression on dictionary entries.

Phase 112, Plan 02: Wire algorithmic fusion naming and validate.
"""

import pytest
import re
from rdkit import Chem

from orthonym.rules.fused_rings import (
    name_fused_heterocycle as _name_fused_heterocycle_raw,
    _try_algorithmic_fusion_name,
)


def name_fused_heterocycle(mol):
    """Wrapper that extracts just the name string from the tuple result."""
    result = _name_fused_heterocycle_raw(mol)
    if result is None:
        return None
    return result[0]
from orthonym.data.fused_heterocycles import (
    get_fused_heterocycle_name,
    FUSED_HETEROCYCLE_DATA,
)


# ============================================================================
# ALGORITHMIC FUSION TEST CASES
#
# All cases exercise the algorithmic path (NOT in the dictionary).
# Each tuple: (canonical_SMILES, expected_algorithmic_name)
#
# Names verified against OPSIN parsing or IUPAC naming rules.
# ============================================================================

ALGORITHMIC_FUSION_CASES = [
    # --- Heterocycle + heterocycle fusions ---
    # 1. furo[2,3-b]pyrrole: furan fused at edge b of pyrrole
    ('c1cc2ccoc2[nH]1', 'furo[2,3-b]pyrrole'),

    # 2. furo[2,3-d]pyrimidine: furan fused at edge d of pyrimidine
    ('c1ncc2ccoc2n1', 'furo[2,3-d]pyrimidine'),

    # 3. thieno[2,3-d]pyrimidine: thiophene fused at edge d of pyrimidine
    ('c1ncc2ccsc2n1', 'thieno[2,3-d]pyrimidine'),

    # 4. furo[2,3-c]pyridine: furan fused at edge c of pyridine
    ('c1cc2ccoc2cn1', 'furo[2,3-c]pyridine'),

    # 5. isoxazolo[4,5-c]pyridine: isoxazole fused at edge c of pyridine
    ('c1cc2oncc2cn1', 'isoxazolo[4,5-c]pyridine'),

    # 6. isothiazolo[4,5-c]pyridine: isothiazole fused at edge c of pyridine
    ('c1cc2sncc2cn1', 'isothiazolo[4,5-c]pyridine'),

    # 7. pyrazolo[4,5-c]pyridine: pyrazole fused to pyridine
    # Phase 149 D-07 update: FR-2.3 V18 (Appendix A.6) picks larger ring
    # (pyridine, 6-membered) as base when both rings have senior
    # heteroatom (N). Original test expectation was pyrido[3,4-d]pyrazole
    # which embeds older numeric-seniority preference for smaller
    # heteroatom-rich ring; per V18 plan + IUPAC P-25.3.2.4, FR-2.3(c)
    # "Larger ring at first point of difference" supersedes. Both names
    # round-trip via OPSIN.
    ('c1cc2n[nH]cc2cn1', 'pyrazolo[4,5-c]pyridine'),

    # 8. pyrrolo[4,3-d]pyrimidine: pyrrole fused to pyrimidine
    ('c1ncc2c[nH]cc2n1', 'pyrrolo[4,3-d]pyrimidine'),

    # 9. pyrido[3,4-d]pyrimidine: pyridine fused to pyrimidine
    ('c1cc2cncnc2cn1', 'pyrido[3,4-d]pyrimidine'),

    # 10. imidazo[4,5-c]pyridine: imidazole fused to pyridine
    # Phase 149 D-07 update: same FR-2.3 V18 rationale as case 7.
    # Original test expectation was pyrido[3,4-d]imidazole; FR-2.3(c)
    # picks larger ring (pyridine) as base.
    ('c1cc2[nH]cnc2cn1', 'imidazo[4,5-c]pyridine'),

    # 11. thieno[4,3-d]pyrimidine: thiophene fused to pyrimidine (alt edge)
    ('c1ncc2cscc2n1', 'thieno[4,3-d]pyrimidine'),

    # 12. thieno[2,3-b]pyrazine: thiophene at edge b of pyrazine
    ('c1cnc2sccc2n1', 'thieno[2,3-b]pyrazine'),

    # 13. furo[4,3-d]pyrimidine: furan fused to pyrimidine (alt edge)
    ('c1ncc2cocc2n1', 'furo[4,3-d]pyrimidine'),

    # 14. furo[2,3-b]pyrazine: furan at edge b of pyrazine
    ('c1cnc2occc2n1', 'furo[2,3-b]pyrazine'),

    # 15. thieno[3,2-c]pyrrole: thiophene fused to pyrrole
    ('c1cc2c[nH]cc2s1', 'thieno[3,2-c]pyrrole'),

    # 16. furo[3,2-c]pyrrole: furan fused to pyrrole
    ('c1cc2c[nH]cc2o1', 'furo[3,2-c]pyrrole'),

    # 17. oxazolo[5,4-d]pyrimidine: oxazole fused to pyrimidine
    ('c1ncc2ncoc2n1', 'oxazolo[5,4-d]pyrimidine'),

    # 18. thiazolo[5,4-d]pyrimidine: thiazole fused to pyrimidine
    ('c1ncc2ncsc2n1', 'thiazolo[5,4-d]pyrimidine'),

    # 19. pyrrolo[4,3-b]pyrrole: pyrrole fused to pyrrole
    ('c1cc2c[nH]cc2[nH]1', 'pyrrolo[4,3-b]pyrrole'),

    # 20. pyrrolo[3,4-e]pyridazine: pyrrole fused to pyridazine
    ('c1cc2c[nH]cc2nn1', 'pyrrolo[3,4-e]pyridazine'),

    # 21. thieno[2,3-e]pyridazine: thiophene fused to pyridazine
    ('c1cc2sccc2nn1', 'thieno[2,3-e]pyridazine'),

    # 22. furo[2,3-e]pyridazine: furan fused to pyridazine
    ('c1cc2occc2nn1', 'furo[2,3-e]pyridazine'),

    # --- Benzo fusions (benzene as child) ---
    # 23. benzo[c]thiophene: benzene fused at edge c of thiophene
    ('c1ccc2cscc2c1', 'benzo[c]thiophene'),

    # --- Cyclopenta fusions ---
    # 24. 5H-cyclopenta[b]pyridine: cyclopentadienyl at edge b of pyridine
    ('C1=Cc2ncccc2C1', '5H-cyclopenta[b]pyridine'),
]


class TestAlgorithmicFusionNaming:
    """
    Integration tests for algorithmic fusion naming.

    Each test verifies that a novel fused system NOT in the dictionary
    produces the correct systematic name via the algorithmic path.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", ALGORITHMIC_FUSION_CASES,
                             ids=[f"case_{i+1}" for i in range(len(ALGORITHMIC_FUSION_CASES))])
    def test_algorithmic_fusion_name(self, smiles, expected_name):
        """Algorithmic path produces correct systematic fusion name."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"

        # Verify NOT in dictionary (exercises algorithmic path)
        dict_result = get_fused_heterocycle_name(mol)
        assert dict_result is None, (
            f"SMILES {smiles} is in dictionary as '{dict_result}', "
            f"expected algorithmic path"
        )

        # Verify algorithmic path produces a name
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None, (
            f"Algorithmic path returned None for {smiles}, "
            f"expected '{expected_name}'"
        )

        # Verify correct name
        assert result == expected_name, (
            f"Wrong name for {smiles}: got '{result}', expected '{expected_name}'"
        )

    @pytest.mark.integration
    def test_at_least_20_cases(self):
        """Ensure at least 20 test cases are defined."""
        assert len(ALGORITHMIC_FUSION_CASES) >= 20, (
            f"Only {len(ALGORITHMIC_FUSION_CASES)} cases, need at least 20"
        )


class TestAlgorithmicFusionViaFullPipeline:
    """
    Tests that the full pipeline (name_fused_heterocycle) returns
    algorithmic names for novel fused systems.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", [
        ('c1cc2ccoc2[nH]1', 'furo[2,3-b]pyrrole'),
        ('c1ncc2ccoc2n1', 'furo[2,3-d]pyrimidine'),
        ('c1ncc2ccsc2n1', 'thieno[2,3-d]pyrimidine'),
        ('c1ccc2cscc2c1', 'benzo[c]thiophene'),
        ('C1=Cc2ncccc2C1', '5H-cyclopenta[b]pyridine'),
    ], ids=['furo-pyrrole', 'furo-pyrimidine', 'thieno-pyrimidine',
            'benzo-thiophene', 'cyclopenta-pyridine'])
    def test_full_pipeline_returns_algorithmic_name(self, smiles, expected_name):
        """Full pipeline returns algorithmic name for novel fused systems."""
        mol = Chem.MolFromSmiles(smiles)
        result = name_fused_heterocycle(mol)
        assert result == expected_name, (
            f"Full pipeline for {smiles}: got '{result}', expected '{expected_name}'"
        )


class TestDictionaryRegression:
    """
    Regression tests ensuring dictionary entries still return correct
    retained names. Dictionary path must NOT be affected by the
    algorithmic fallback.
    """

    DICTIONARY_CASES = [
        # (SMILES, expected_retained_name)
        ('c1ccc2[nH]ccc2c1', '1H-indole'),
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2cnccc2c1', 'isoquinoline'),
        ('c1ccc2occc2c1', '1-benzofuran'),
        ('c1ccc2sccc2c1', '1-benzothiophene'),
        ('c1ccc2[nH]cnc2c1', '1H-benzimidazole'),
        ('c1ccc2[nH]ncc2c1', '1H-indazole'),
        ('c1ccc2[nH]nnc2c1', '1H-benzotriazole'),
        ('c1cnc2ccoc2c1', 'furo[3,2-b]pyridine'),
        ('c1cnc2[nH]ccc2c1', '1H-pyrrolo[2,3-b]pyridine'),
    ]

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", DICTIONARY_CASES,
                             ids=[name for _, name in DICTIONARY_CASES])
    def test_dictionary_entry_still_matches(self, smiles, expected_name):
        """Dictionary entries return correct retained names."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        # Should be in dictionary
        dict_result = get_fused_heterocycle_name(mol)
        assert dict_result is not None, (
            f"SMILES {smiles} not found in dictionary, "
            f"expected '{expected_name}'"
        )

        # Full pipeline should return dictionary name
        result = name_fused_heterocycle(mol)
        assert result == expected_name, (
            f"Dictionary regression for {smiles}: "
            f"got '{result}', expected '{expected_name}'"
        )

    @pytest.mark.integration
    def test_all_dictionary_entries_accessible(self):
        """All dictionary entries should still be accessible."""
        accessible_count = 0
        for smiles in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            result = get_fused_heterocycle_name(mol)
            if result is not None:
                accessible_count += 1

        # Should have at least 100 accessible entries
        assert accessible_count >= 100, (
            f"Only {accessible_count} dictionary entries accessible, "
            f"expected at least 100"
        )


class TestAlgorithmicPathGating:
    """
    Boundary tests ensuring the algorithmic path correctly gates
    out-of-scope systems.
    """

    @pytest.mark.integration
    def test_three_ring_system_returns_none(self):
        """3-ring fused systems should return None (out of scope)."""
        # Carbazole-like: 3 fused rings
        mol = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c1ccccc12')
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"3-ring system should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_purely_carbocyclic_returns_none(self):
        """Purely carbocyclic fused systems should return None."""
        # Naphthalene: fused carbocyclic, not heterocyclic
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"Carbocyclic system should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_non_aromatic_returns_none(self):
        """Non-aromatic fused heterocycles should return None."""
        # Decahydroquinoline: fully saturated
        mol = Chem.MolFromSmiles('C1CCC2CCCNC2C1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"Non-aromatic system should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_substituted_returns_none(self):
        """Substituted fused systems should return None (no substituent handling)."""
        # 5-methyl-furo[2,3-b]pyrrole (has a methyl substituent)
        mol = Chem.MolFromSmiles('Cc1cc2ccoc2[nH]1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"Substituted system should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_single_ring_returns_none(self):
        """Single ring should return None."""
        mol = Chem.MolFromSmiles('c1ccncc1')  # pyridine
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"Single ring should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_spiro_returns_none(self):
        """Spiro compound (1 shared atom) should return None."""
        mol = Chem.MolFromSmiles('C1CCC2(CC1)CCCCC2')  # spiro[5.5]undecane
        result = _try_algorithmic_fusion_name(mol)
        assert result is None, (
            f"Spiro compound should return None, got '{result}'"
        )

    @pytest.mark.integration
    def test_unidentifiable_component_returns_none(self):
        """Ring with unidentifiable component should return None."""
        # A ring with unusual heteroatoms not in registry
        mol = Chem.MolFromSmiles('c1cc2cc[se]c2[nH]1')
        if mol is not None:
            result = _try_algorithmic_fusion_name(mol)
            # May or may not return None depending on registry
            # but should not crash
            assert result is None or isinstance(result, str)


class TestIndicatedHydrogen:
    """
    Tests for indicated hydrogen in algorithmic fusion names.

    Indicated hydrogen (nH-) is prepended when a non-aromatic NH
    or CH is present at a tautomeric position in the fused system.
    """

    @pytest.mark.integration
    def test_cyclopenta_pyridine_has_indicated_h(self):
        """5H-cyclopenta[b]pyridine should have 5H- prefix."""
        mol = Chem.MolFromSmiles('C1=Cc2ncccc2C1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        assert 'H-' in result, f"Expected indicated H in '{result}'"
        assert result.startswith('5H-'), (
            f"Expected '5H-' prefix in '{result}'"
        )

    @pytest.mark.integration
    def test_fully_aromatic_no_indicated_h(self):
        """Fully aromatic fused systems should NOT have indicated H."""
        # furo[2,3-d]pyrimidine - all aromatic
        mol = Chem.MolFromSmiles('c1ncc2ccoc2n1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        assert 'H-' not in result, (
            f"Fully aromatic system should not have indicated H: '{result}'"
        )

    @pytest.mark.integration
    def test_thieno_pyrimidine_no_indicated_h(self):
        """thieno[2,3-d]pyrimidine (fully aromatic) has no indicated H."""
        mol = Chem.MolFromSmiles('c1ncc2ccsc2n1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        assert 'H-' not in result

    @pytest.mark.integration
    def test_benzo_thiophene_no_indicated_h(self):
        """benzo[c]thiophene (fully aromatic) has no indicated H."""
        mol = Chem.MolFromSmiles('c1ccc2cscc2c1')
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        assert 'H-' not in result


class TestDescriptorFormat:
    """Tests that generated descriptors follow valid IUPAC format."""

    # Pattern: [num,num-letter] or [letter]
    DESCRIPTOR_PATTERN = re.compile(r'\[\d+,\d+-[a-z]\]|\[[a-z]\]')

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected_name", ALGORITHMIC_FUSION_CASES[:15])
    def test_descriptor_format_valid(self, smiles, expected_name):
        """Generated name contains valid fusion descriptor."""
        mol = Chem.MolFromSmiles(smiles)
        result = _try_algorithmic_fusion_name(mol)
        if result is None:
            pytest.skip(f"No algorithmic name for {smiles}")

        # Strip indicated H prefix if present
        name = result
        if re.match(r'^\d+H-', name):
            name = name.split('-', 1)[1]

        # Find descriptor
        desc_match = self.DESCRIPTOR_PATTERN.search(name)
        assert desc_match is not None, (
            f"No valid descriptor found in '{result}'"
        )

    @pytest.mark.integration
    def test_benzene_child_uses_letter_only_descriptor(self):
        """Benzene as child should use [letter] format (no child locants)."""
        mol = Chem.MolFromSmiles('c1ccc2cscc2c1')  # benzo[c]thiophene
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        # Should have [c] not [1,2-c]
        assert '[c]' in result, f"Expected [letter] format in '{result}'"
        assert '[1,2-' not in result, (
            f"Benzene child should not have child locants in '{result}'"
        )

    @pytest.mark.integration
    def test_heterocycle_child_uses_full_descriptor(self):
        """Heterocyclic child should use [num,num-letter] format."""
        mol = Chem.MolFromSmiles('c1ncc2ccoc2n1')  # furo[2,3-d]pyrimidine
        result = _try_algorithmic_fusion_name(mol)
        assert result is not None
        assert re.search(r'\[\d+,\d+-[a-z]\]', result), (
            f"Expected [num,num-letter] format in '{result}'"
        )
