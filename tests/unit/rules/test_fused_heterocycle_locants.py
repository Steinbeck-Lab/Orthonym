"""
Comprehensive validation tests for fused heterocycle IUPAC locant mappings.

Verifies:
1. Data completeness: all entries have non-empty iupac_locants
2. Locant mapping correctness against known IUPAC numbering
3. Indicated hydrogen preservation for substituted variants
4. Fallback enumeration removal
5. Locant index contiguity (no gaps in pattern indices)
6. Locant count matches parent_atoms for each entry
"""

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    _validate_all_entries,
    match_fused_heterocycle_core,
    get_fused_heterocycle_name,
)
from orthonym.namer import name_compound


# =========================================================================
# 1. Data completeness tests
# =========================================================================


@pytest.mark.unit
class TestDataCompleteness:
    """Verify every entry in FUSED_HETEROCYCLE_DATA has valid iupac_locants."""

    def test_all_entries_have_iupac_locants(self):
        """Every entry must have a non-empty iupac_locants dict."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            locants = data.get('iupac_locants')
            assert locants is not None and len(locants) > 0, (
                f"Entry '{data.get('name', smiles)}' has empty/missing iupac_locants"
            )

    def test_validate_all_entries_function(self):
        """The _validate_all_entries helper must return True."""
        assert _validate_all_entries() is True

    def test_minimum_entry_count(self):
        """There should be at least 70 fused heterocycle entries."""
        assert len(FUSED_HETEROCYCLE_DATA) >= 70, (
            f"Expected >= 70 entries, got {len(FUSED_HETEROCYCLE_DATA)}"
        )

    def test_all_entries_have_required_fields(self):
        """Every entry must have name, ring_system, parent_atoms, and iupac_locants."""
        required = {'name', 'ring_system', 'parent_atoms', 'iupac_locants'}
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            missing = required - set(data.keys())
            assert not missing, (
                f"Entry '{data.get('name', smiles)}' missing fields: {missing}"
            )


# =========================================================================
# 2. Locant index contiguity and count validation
# =========================================================================


@pytest.mark.unit
class TestLocantStructure:
    """Verify structural integrity of locant mappings."""

    def test_locant_indices_are_contiguous(self):
        """Pattern indices in iupac_locants must be 0..N with no gaps."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            locants = data.get('iupac_locants', {})
            if not locants:
                continue
            max_idx = max(locants.keys())
            for i in range(max_idx + 1):
                assert i in locants, (
                    f"Entry '{data['name']}' missing pattern_idx {i} in iupac_locants "
                    f"(max index={max_idx})"
                )

    def test_locant_count_matches_parent_atoms(self):
        """Number of locant entries must equal parent_atoms."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            locants = data.get('iupac_locants', {})
            parent_atoms = data.get('parent_atoms', 0)
            assert len(locants) == parent_atoms, (
                f"Entry '{data['name']}' has {len(locants)} locants but "
                f"parent_atoms={parent_atoms}"
            )

    def test_locant_values_are_valid_types(self):
        """Locant values must be int or str (for fusion positions like '3a')."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            locants = data.get('iupac_locants', {})
            for pattern_idx, locant_val in locants.items():
                assert isinstance(locant_val, (int, str)), (
                    f"Entry '{data['name']}' has invalid locant type at "
                    f"idx {pattern_idx}: {type(locant_val)} ({locant_val})"
                )

    def test_canonical_smiles_are_valid(self):
        """All keys must be valid SMILES parseable by RDKit."""
        for smiles in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES key: {smiles}"


# =========================================================================
# 3. Locant mapping correctness (representative sample)
# =========================================================================


@pytest.mark.unit
class TestLocantCorrectness:
    """Test locant mappings for well-known fused heterocycles against IUPAC references."""

    def test_indole_locants(self):
        """Indole: 9 atoms, positions 1-7 + 3a + 7a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2[nH]ccc2c1']
        assert data['name'] == '1H-indole'
        locants = data['iupac_locants']
        # Verify all expected IUPAC positions are present
        locant_values = set(locants.values())
        expected = {1, 2, 3, '3a', 4, 5, 6, 7, '7a'}
        assert locant_values == expected, (
            f"Indole locant values {locant_values} != expected {expected}"
        )

    def test_quinoline_locants(self):
        """Quinoline: 10 atoms, positions 1-8 + 4a + 8a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2ncccc2c1']
        assert data['name'] == 'quinoline'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, '4a', 5, 6, 7, 8, '8a'}
        assert locant_values == expected, (
            f"Quinoline locant values {locant_values} != expected {expected}"
        )

    def test_isoquinoline_locants(self):
        """Isoquinoline: 10 atoms, positions 1-8 + 4a + 8a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2cnccc2c1']
        assert data['name'] == 'isoquinoline'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, '4a', 5, 6, 7, 8, '8a'}
        assert locant_values == expected

    def test_purine_locants(self):
        """Purine: 9 atoms, positions 1-9 (IUPAC numbering)."""
        data = FUSED_HETEROCYCLE_DATA['c1ncc2nc[nH]c2n1']
        assert data['name'] == '9H-purine'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        assert locant_values == expected, (
            f"Purine locant values {locant_values} != expected {expected}"
        )

    def test_benzimidazole_locants(self):
        """Benzimidazole: 9 atoms, positions 1-7 + 3a + 7a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2[nH]cnc2c1']
        assert data['name'] == '1H-benzimidazole'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, '3a', 4, 5, 6, 7, '7a'}
        assert locant_values == expected

    def test_benzofuran_locants(self):
        """Benzofuran: 9 atoms, positions 1-7 + 3a + 7a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2occc2c1']
        assert data['name'] == '1-benzofuran'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, '3a', 4, 5, 6, 7, '7a'}
        assert locant_values == expected

    def test_benzothiophene_locants(self):
        """Benzothiophene: 9 atoms, positions 1-7 + 3a + 7a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2sccc2c1']
        assert data['name'] == '1-benzothiophene'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, '3a', 4, 5, 6, 7, '7a'}
        assert locant_values == expected

    def test_carbazole_locants(self):
        """Carbazole: 13 atoms, positions 1-9 + 4a + 4b + 8a + 9a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2c(c1)[nH]c1ccccc12']
        assert data['name'] == '9H-carbazole'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, '4a', '4b', 5, 6, 7, 8, '8a', 9, '9a'}
        assert locant_values == expected, (
            f"Carbazole locant values {locant_values} != expected {expected}"
        )

    def test_acridine_locants(self):
        """Acridine: 14 atoms, positions 1-10 + 4a + 8a + 9a + 10a."""
        data = FUSED_HETEROCYCLE_DATA['c1ccc2nc3ccccc3cc2c1']
        assert data['name'] == 'acridine'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, '4a', 5, 6, 7, 8, '8a', 9, '9a', 10, '10a'}
        assert locant_values == expected

    def test_indolizine_locants(self):
        """Indolizine: 9 atoms, bridgehead N-system (Blue Book Table 2.8 #20).

        Numbering 1,2,3,4,5,6,7,8,8a: the bridgehead N is position 4 and the
        bridgehead carbon is 8a — there is NO '3a' (that was the -flagged
        wrong stored numbering, N at 8a). 13B(a) 5/7-ring data fix re-derived
        these from OPSIN -o extendedsmi, substituted-RT verified.
        """
        data = FUSED_HETEROCYCLE_DATA['c1ccn2cccc2c1']
        assert data['name'] == 'indolizine'
        locants = data['iupac_locants']
        locant_values = set(locants.values())
        expected = {1, 2, 3, 4, 5, 6, 7, 8, '8a'}
        assert locant_values == expected


# =========================================================================
# 4. Substructure matching correctness
# =========================================================================


@pytest.mark.unit
class TestSubstructureMatching:
    """Verify match_fused_heterocycle_core returns correct mappings."""

    def test_unsubstituted_indole_matches(self):
        """Unsubstituted indole matches and returns correct core name."""
        mol = Chem.MolFromSmiles('c1ccc2[nH]ccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        assert name == '1H-indole'
        assert core_smiles == 'c1ccc2[nH]ccc2c1'

    def test_substituted_indole_matches(self):
        """5-Methylindole matches indole core."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]ccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        assert name == '1H-indole'

    def test_quinoline_matches(self):
        """Quinoline matches and returns correct mapping."""
        mol = Chem.MolFromSmiles('c1ccc2ncccc2c1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        assert name == 'quinoline'

    def test_purine_matches(self):
        """Purine matches and returns correct mapping."""
        mol = Chem.MolFromSmiles('c1ncc2nc[nH]c2n1')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        assert name == '9H-purine'

    def test_unknown_molecule_returns_none(self):
        """A simple non-fused molecule returns None."""
        mol = Chem.MolFromSmiles('CCCC')  # butane
        result = match_fused_heterocycle_core(mol)
        assert result is None

    def test_none_mol_returns_none(self):
        """None molecule returns None."""
        result = match_fused_heterocycle_core(None)
        assert result is None

    def test_largest_core_wins(self):
        """When multiple cores match, the largest is selected."""
        # Adenine contains purine core (9 atoms) which is larger than any sub-ring
        mol = Chem.MolFromSmiles('Nc1ncnc2nc[nH]c12')
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, _, _ = result
        # The purine core wins over any single ring. (It used to be a separate
        # 'adenine' catalogue entry; that name is not a Blue Book name and the entry
        # is gone: "the PIN is 7H-purine", the Blue Book.)
        assert name == '7H-purine'


# =========================================================================
# 5. Indicated hydrogen preservation (E2E tests)
# =========================================================================


@pytest.mark.unit
class TestIndicatedHydrogen:
    """Verify indicated hydrogen (1H-, 2H-, 9H-) is preserved in naming."""

    def test_indole_has_1H(self):
        """Indole name must contain '1H-'."""
        name = name_compound('c1ccc2[nH]ccc2c1')
        assert '1H-' in name, f"Expected '1H-' in '{name}'"
        assert name == '1H-indole'

    def test_methylindole_preserves_1H(self):
        """5-Methylindole must preserve '1H-' prefix."""
        name = name_compound('Cc1ccc2[nH]ccc2c1')
        assert '1H-' in name, f"Expected '1H-' in '{name}'"

    def test_purine_has_9H(self):
        """Purine name must contain '9H-'."""
        name = name_compound('c1ncc2nc[nH]c2n1')
        assert '9H-' in name, f"Expected '9H-' in '{name}'"
        assert name == '9H-purine'

    def test_benzimidazole_has_1H(self):
        """Benzimidazole name must contain '1H-'."""
        name = name_compound('c1ccc2[nH]cnc2c1')
        assert '1H-' in name, f"Expected '1H-' in '{name}'"
        assert name == '1H-benzimidazole'

    def test_methylbenzimidazole_preserves_1H(self):
        """Substituted benzimidazole must preserve '1H-'."""
        name = name_compound('Cc1ccc2[nH]cnc2c1')
        assert '1H-' in name, f"Expected '1H-' in '{name}'"

    def test_carbazole_has_9H(self):
        """Carbazole name must contain '9H-'."""
        name = name_compound('c1ccc2c(c1)[nH]c1ccccc12')
        assert '9H-' in name, f"Expected '9H-' in '{name}'"
        assert name == '9H-carbazole'

    def test_indazole_has_1H(self):
        """Indazole name must contain '1H-'."""
        name = name_compound('c1ccc2[nH]ncc2c1')
        assert '1H-' in name, f"Expected '1H-' in '{name}'"

    def test_no_indicated_H_for_quinoline(self):
        """Quinoline has no NH, so no indicated H in name."""
        name = name_compound('c1ccc2ncccc2c1')
        assert name == 'quinoline'
        assert 'H-' not in name

    def test_no_indicated_H_for_benzofuran(self):
        """Benzofuran has no NH, so no indicated H in name."""
        name = name_compound('c1ccc2occc2c1')
        # benzofuran or 1-benzofuran - no "H-" prefix
        assert 'H-' not in name


# =========================================================================
# 6. Fallback removal verification
# =========================================================================


@pytest.mark.unit
class TestFallbackRemoval:
    """Verify the fallback enumeration code path has been removed."""

    def test_all_real_entries_return_valid_results(self):
        """All entries in the data dict should produce valid match results."""
        for smiles in FUSED_HETEROCYCLE_DATA:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            result = match_fused_heterocycle_core(mol)
            assert result is not None, (
                f"Entry {smiles} ({FUSED_HETEROCYCLE_DATA[smiles]['name']}) "
                f"failed to match its own core"
            )
            name, atom_mapping, core_smiles = result
            assert len(atom_mapping) > 0, (
                f"Entry {smiles} ({name}) produced empty atom_mapping"
            )

    def test_missing_locants_would_return_none(self):
        """Verify the code path: if an entry had no locants, it returns None.

        Since all entries have locants now, we verify this by checking the
        source code no longer contains the old fallback dict comprehension
        that built incorrect sequential numbering.
        """
        import inspect
        source = inspect.getsource(match_fused_heterocycle_core)
        # The OLD fallback was: {atom_idx: locant + 1 for locant, atom_idx in enumerate(match_atoms)}
        assert 'locant + 1 for locant' not in source, (
            "Fallback enumeration dict comprehension still present in match_fused_heterocycle_core"
        )
        # The new path returns None with a warning
        assert 'return None' in source, (
            "Missing 'return None' for iupac_locants is None branch"
        )


# =========================================================================
# 7. E2E naming spot checks
# =========================================================================


@pytest.mark.unit
class TestE2ENaming:
    """End-to-end naming tests for common fused heterocycles."""

    @pytest.mark.parametrize("smiles,expected_name", [
        ('c1ccc2[nH]ccc2c1', '1H-indole'),
        ('c1ccc2ncccc2c1', 'quinoline'),
        ('c1ccc2cnccc2c1', 'isoquinoline'),
        ('c1ccc2[nH]cnc2c1', '1H-benzimidazole'),
        ('c1ccc2c(c1)[nH]c1ccccc12', '9H-carbazole'),
        ('c1ncc2nc[nH]c2n1', '9H-purine'),
    ])
    def test_unsubstituted_names(self, smiles, expected_name):
        """Unsubstituted fused heterocycles get their retained name."""
        name = name_compound(smiles)
        assert name == expected_name, f"Expected '{expected_name}', got '{name}'"

    def test_5_methylindole(self):
        """5-Methylindole should produce locant-prefix + 1H-indole."""
        name = name_compound('Cc1ccc2[nH]ccc2c1')
        assert '1H-indole' in name
        assert 'methyl' in name

    def test_5_methylbenzimidazole(self):
        """Substituted benzimidazole preserves indicated hydrogen."""
        name = name_compound('Cc1ccc2[nH]cnc2c1')
        assert '1H-benzimidazole' in name
        assert 'methyl' in name


# =========================================================================
# 8. Tautomer locant consistency
# =========================================================================


@pytest.mark.unit
class TestTautomerLocantConsistency:
    """Verify tautomer_locant is consistent with indicated H in names."""

    def test_tautomer_locant_matches_name(self):
        """If tautomer_locant is set, the name should contain nH- prefix.

        Exception: retained names (like 'coumarin') may omit the indicated H
        from the name while still tracking the tautomer locant for numbering.
        """
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            name = data['name']
            taut = data.get('tautomer_locant')
            # Skip entries with retained names that don't use nH- format
            if data.get('is_retained_name'):
                continue
            # Skip coumarin and similar retained trivial names
            if name in ('coumarin',):
                continue
            if taut is not None:
                assert 'H-' in name, (
                    f"Entry '{name}' has tautomer_locant={taut} but no 'H-' in name"
                )

    def test_no_tautomer_no_H_prefix(self):
        """If tautomer_locant is None, name should not start with nH- pattern."""
        import re
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            name = data['name']
            taut = data.get('tautomer_locant')
            # Skip retained names with =O or NH2 that might have H- in name
            if data.get('is_retained_name'):
                continue
            if taut is None:
                # Should not start with digits followed by H-
                assert not re.match(r'^\d+H-', name), (
                    f"Entry '{name}' has tautomer_locant=None but name starts with nH-"
                )
