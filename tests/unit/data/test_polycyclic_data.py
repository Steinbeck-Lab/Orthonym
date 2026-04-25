"""
Unit tests for extended PAH data module.

Tests data integrity, PAH entry validation, and substructure matching
for the polycyclic aromatic hydrocarbon lookup tables.
"""

import pytest
from rdkit import Chem

from orthonym.data.polycyclic_data import (
    POLYCYCLIC_DATA,
    get_polycyclic_by_smiles,
    get_polycyclic_by_name,
    is_polycyclic_aromatic,
    get_pah_names,
    match_polycyclic_core,
    get_pah_core_atoms,
    get_pah_substituent_positions,
)


# ============================================================================
# Data Integrity Tests
# ============================================================================

@pytest.mark.unit
class TestDataIntegrity:
    """Tests for PAH data integrity and validity."""

    def test_minimum_pah_count(self):
        """Verify we have at least 15 PAH entries (COMPLEX-06 requirement)."""
        assert len(POLYCYCLIC_DATA) >= 15, f"Expected 15+ PAH entries, got {len(POLYCYCLIC_DATA)}"

    def test_all_smiles_valid(self):
        """All canonical SMILES should produce valid RDKit molecules."""
        for name, data in POLYCYCLIC_DATA.items():
            smiles = data['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f"Invalid SMILES for {name}: {smiles}"

    def test_all_smiles_canonical(self):
        """All SMILES should be in RDKit canonical form."""
        for name, data in POLYCYCLIC_DATA.items():
            smiles = data['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            canonical = Chem.MolToSmiles(mol, canonical=True)
            assert smiles == canonical, f"{name}: SMILES not canonical ({smiles} vs {canonical})"

    def test_num_atoms_matches(self):
        """num_atoms should match actual atom count from SMILES."""
        for name, data in POLYCYCLIC_DATA.items():
            smiles = data['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            actual_atoms = mol.GetNumAtoms()
            expected_atoms = data['num_atoms']
            assert actual_atoms == expected_atoms, (
                f"{name}: num_atoms mismatch ({expected_atoms} vs actual {actual_atoms})"
            )

    def test_required_fields_present(self):
        """All PAH entries should have required fields."""
        required_fields = ['canonical_smiles', 'smarts', 'num_atoms', 'num_rings', 'substituent_positions']
        for name, data in POLYCYCLIC_DATA.items():
            for field in required_fields:
                assert field in data, f"{name} missing required field: {field}"

    def test_smarts_patterns_valid(self):
        """All SMARTS patterns should be valid."""
        for name, data in POLYCYCLIC_DATA.items():
            smarts = data['smarts']
            pattern = Chem.MolFromSmarts(smarts)
            assert pattern is not None, f"Invalid SMARTS for {name}: {smarts}"


# ============================================================================
# Lookup Function Tests
# ============================================================================

@pytest.mark.unit
class TestLookupFunctions:
    """Tests for PAH lookup functions."""

    def test_get_polycyclic_by_smiles_naphthalene(self):
        """Test naphthalene lookup by SMILES."""
        result = get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        assert result is not None
        assert result['name'] == 'naphthalene'
        assert result['num_atoms'] == 10

    def test_get_polycyclic_by_smiles_pyrene(self):
        """Test pyrene lookup by SMILES."""
        result = get_polycyclic_by_smiles('c1cc2ccc3cccc4ccc(c1)c2c34')
        assert result is not None
        assert result['name'] == 'pyrene'
        assert result['num_atoms'] == 16

    def test_get_polycyclic_by_smiles_perylene(self):
        """Test perylene lookup by SMILES."""
        result = get_polycyclic_by_smiles('c1ccc2cccc3cc4c(c1)cc1cccc4c1c23')
        assert result is not None
        assert result['name'] == 'perylene'
        assert result['num_atoms'] == 20

    def test_get_polycyclic_by_smiles_coronene(self):
        """Test coronene lookup by SMILES."""
        result = get_polycyclic_by_smiles('c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61')
        assert result is not None
        assert result['name'] == 'coronene'
        assert result['num_atoms'] == 24

    def test_get_polycyclic_by_smiles_not_found(self):
        """Non-PAH SMILES should return None."""
        result = get_polycyclic_by_smiles('c1ccccc1')  # benzene
        assert result is None

    def test_get_polycyclic_by_name_found(self):
        """Test lookup by name."""
        result = get_polycyclic_by_name('anthracene')
        assert result is not None
        assert result['name'] == 'anthracene'

    def test_get_polycyclic_by_name_not_found(self):
        """Non-existent name should return None."""
        result = get_polycyclic_by_name('notapah')
        assert result is None

    def test_is_polycyclic_aromatic(self):
        """Test is_polycyclic_aromatic function."""
        assert is_polycyclic_aromatic('c1ccc2ccccc2c1')  # naphthalene
        assert not is_polycyclic_aromatic('c1ccccc1')  # benzene

    def test_get_pah_names(self):
        """Test get_pah_names returns all PAH names."""
        names = get_pah_names()
        assert 'naphthalene' in names
        assert 'pyrene' in names
        assert 'coronene' in names
        assert len(names) >= 15


# ============================================================================
# Substructure Matching Tests
# ============================================================================

@pytest.mark.unit
class TestSubstructureMatching:
    """Tests for SMARTS-based PAH core matching."""

    def test_match_naphthalene_exact(self):
        """Exact naphthalene should match naphthalene core."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'naphthalene'
        assert len(mapping) == 10

    def test_match_methylnaphthalene(self):
        """2-methylnaphthalene should match naphthalene core."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'naphthalene'
        assert len(mapping) == 10  # Only core atoms

    def test_match_dimethylnaphthalene(self):
        """2,3-dimethylnaphthalene should match naphthalene core."""
        mol = Chem.MolFromSmiles('Cc1cc2ccccc2cc1C')
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'naphthalene'

    def test_match_methylanthracene(self):
        """9-methylanthracene should match anthracene core."""
        mol = Chem.MolFromSmiles('Cc1c2ccccc2cc2ccccc12')
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'anthracene'
        assert len(mapping) == 14

    def test_match_methylpyrene(self):
        """Methylpyrene should match pyrene core."""
        mol = Chem.MolFromSmiles('Cc1cc2ccc3cccc4ccc(c1)c2c34')
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result
        assert name == 'pyrene'

    def test_no_match_benzene(self):
        """Benzene should not match any PAH core."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        result = match_polycyclic_core(mol)
        assert result is None

    def test_match_returns_correct_mapping(self):
        """Atom mapping should correctly map core atoms."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        result = match_polycyclic_core(mol)
        assert result is not None
        name, mapping = result

        # Verify mapping contains only core atoms (not the methyl carbon)
        mol_atoms = mol.GetNumAtoms()
        assert all(idx < mol_atoms for idx in mapping.keys())
        assert len(mapping) == 10  # naphthalene has 10 atoms


# ============================================================================
# Core Atom Retrieval Tests
# ============================================================================

@pytest.mark.unit
class TestCoreAtomRetrieval:
    """Tests for get_pah_core_atoms function."""

    def test_core_atoms_naphthalene(self):
        """Get core atoms for naphthalene."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        core = get_pah_core_atoms(mol, 'naphthalene')
        assert core is not None
        assert len(core) == 10

    def test_core_atoms_substituted(self):
        """Get core atoms for substituted naphthalene."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        core = get_pah_core_atoms(mol, 'naphthalene')
        assert core is not None
        assert len(core) == 10  # Only core, not methyl

    def test_core_atoms_unknown_pah(self):
        """Unknown PAH name should return None."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        core = get_pah_core_atoms(mol, 'notapah')
        assert core is None

    def test_core_atoms_no_match(self):
        """Non-matching molecule should return None."""
        mol = Chem.MolFromSmiles('c1ccccc1')  # benzene
        core = get_pah_core_atoms(mol, 'naphthalene')
        assert core is None


# ============================================================================
# New PAH Entry Tests
# ============================================================================

@pytest.mark.unit
class TestNewPahEntries:
    """Tests for newly added PAH entries."""

    @pytest.mark.parametrize("name,expected_atoms,expected_rings", [
        ('tetracene', 18, 4),
        ('triphenylene', 18, 4),
        ('benz[a]anthracene', 18, 4),
        ('benzo[c]phenanthrene', 18, 4),
        ('pentacene', 22, 5),
        ('perylene', 20, 6),
        ('benzo[a]pyrene', 20, 5),
        ('coronene', 24, 7),
        ('9,10-dihydroanthracene', 14, 3),
        ('1,2-dihydronaphthalene', 10, 2),
    ])
    def test_new_pah_lookup(self, name, expected_atoms, expected_rings):
        """Test lookup for each new PAH entry."""
        result = get_polycyclic_by_name(name)
        assert result is not None, f"{name} not found in POLYCYCLIC_DATA"
        assert result['num_atoms'] == expected_atoms, f"{name} atom count mismatch"
        assert result['num_rings'] == expected_rings, f"{name} ring count mismatch"

    @pytest.mark.parametrize("name", [
        'tetracene',
        'triphenylene',
        'benz[a]anthracene',
        'benzo[c]phenanthrene',
        'pentacene',
        'perylene',
        'benzo[a]pyrene',
        'coronene',
    ])
    def test_new_pah_substructure_match(self, name):
        """New PAHs should match their own SMILES."""
        data = POLYCYCLIC_DATA[name]
        mol = Chem.MolFromSmiles(data['canonical_smiles'])
        result = match_polycyclic_core(mol)
        assert result is not None, f"{name} should match its own SMILES"
        matched_name, _ = result
        assert matched_name == name, f"Expected {name}, got {matched_name}"


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

@pytest.mark.unit
class TestBackwardCompatibility:
    """Tests ensuring existing PAH functionality still works."""

    def test_naphthalene_still_works(self):
        """Naphthalene lookup should still work."""
        result = get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        assert result is not None
        assert result['name'] == 'naphthalene'

    def test_anthracene_still_works(self):
        """Anthracene lookup should still work."""
        result = get_polycyclic_by_smiles('c1ccc2cc3ccccc3cc2c1')
        assert result is not None
        assert result['name'] == 'anthracene'

    def test_phenanthrene_still_works(self):
        """Phenanthrene lookup should still work."""
        result = get_polycyclic_by_smiles('c1ccc2c(c1)ccc1ccccc12')
        assert result is not None
        assert result['name'] == 'phenanthrene'

    def test_pyrene_still_works(self):
        """Pyrene lookup should still work."""
        result = get_polycyclic_by_smiles('c1cc2ccc3cccc4ccc(c1)c2c34')
        assert result is not None
        assert result['name'] == 'pyrene'

    def test_fluorene_still_works(self):
        """Fluorene lookup should still work."""
        result = get_polycyclic_by_smiles('c1ccc2c(c1)Cc1ccccc1-2')
        assert result is not None
        assert result['name'] == 'fluorene'

    def test_chrysene_still_works(self):
        """Chrysene lookup should still work."""
        result = get_polycyclic_by_smiles('c1ccc2c(c1)ccc1c3ccccc3ccc21')
        assert result is not None
        assert result['name'] == 'chrysene'


# ============================================================================
# IUPAC Numbering Correctness Tests (DATA-07)
# ============================================================================

@pytest.mark.unit
class TestIupacNumberingCorrectness:
    """Tests for IUPAC numbering map correctness in polycyclic_data.py."""

    def test_naphthalene_fusion_locants_are_strings(self):
        """Naphthalene fusion positions must use string 'a' notation, not floats."""
        numbering = POLYCYCLIC_DATA['naphthalene']['iupac_numbering']
        fusion_values = [v for v in numbering.values() if isinstance(v, str)]
        float_values = [v for v in numbering.values() if isinstance(v, float)]
        assert len(float_values) == 0, f"Found float locants: {float_values}"
        assert '4a' in fusion_values, "Missing '4a' fusion locant"
        assert '8a' in fusion_values, "Missing '8a' fusion locant"

    def test_naphthalene_has_10_entries(self):
        """Naphthalene iupac_numbering must have exactly 10 entries."""
        numbering = POLYCYCLIC_DATA['naphthalene']['iupac_numbering']
        assert len(numbering) == 10, f"Expected 10 entries, got {len(numbering)}"

    def test_naphthalene_fusion_atoms_are_degree3(self):
        """Naphthalene fusion atoms (mapped to string locants) should have degree 3."""
        data = POLYCYCLIC_DATA['naphthalene']
        mol = Chem.MolFromSmiles(data['canonical_smiles'])
        numbering = data['iupac_numbering']
        for atom_idx, locant in numbering.items():
            if isinstance(locant, str):
                atom = mol.GetAtomWithIdx(atom_idx)
                assert atom.GetDegree() == 3, (
                    f"Fusion atom idx {atom_idx} (locant {locant}) has degree "
                    f"{atom.GetDegree()}, expected 3"
                )

    def test_anthracene_numbering_nonempty(self):
        """Anthracene iupac_numbering must be populated (not empty)."""
        numbering = POLYCYCLIC_DATA['anthracene']['iupac_numbering']
        assert len(numbering) > 0, "Anthracene iupac_numbering is empty"

    def test_anthracene_has_14_entries(self):
        """Anthracene iupac_numbering must have exactly 14 entries."""
        numbering = POLYCYCLIC_DATA['anthracene']['iupac_numbering']
        assert len(numbering) == 14, f"Expected 14 entries, got {len(numbering)}"

    def test_phenanthrene_numbering_nonempty(self):
        """Phenanthrene iupac_numbering must be populated (not empty)."""
        numbering = POLYCYCLIC_DATA['phenanthrene']['iupac_numbering']
        assert len(numbering) > 0, "Phenanthrene iupac_numbering is empty"

    def test_phenanthrene_has_14_entries(self):
        """Phenanthrene iupac_numbering must have exactly 14 entries."""
        numbering = POLYCYCLIC_DATA['phenanthrene']['iupac_numbering']
        assert len(numbering) == 14, f"Expected 14 entries, got {len(numbering)}"

    def test_pyrene_numbering_nonempty(self):
        """Pyrene iupac_numbering must be populated (not empty)."""
        numbering = POLYCYCLIC_DATA['pyrene']['iupac_numbering']
        assert len(numbering) > 0, "Pyrene iupac_numbering is empty"

    def test_pyrene_has_16_entries(self):
        """Pyrene iupac_numbering must have exactly 16 entries."""
        numbering = POLYCYCLIC_DATA['pyrene']['iupac_numbering']
        assert len(numbering) == 16, f"Expected 16 entries, got {len(numbering)}"

    def test_all_fusion_locants_are_strings(self):
        """All fusion locants across all PAH entries should be strings (not floats)."""
        import re
        for name, data in POLYCYCLIC_DATA.items():
            numbering = data.get('iupac_numbering', {})
            for atom_idx, locant in numbering.items():
                if isinstance(locant, float):
                    raise AssertionError(
                        f"{name}: atom idx {atom_idx} has float locant {locant}, "
                        f"should be string like '{int(locant)}a'"
                    )
                if isinstance(locant, str):
                    assert re.match(r'^\d+[a-z]$', locant), (
                        f"{name}: string locant '{locant}' doesn't match pattern 'Na' "
                        f"(digit(s) followed by single lowercase letter)"
                    )

    def test_all_peripheral_locants_are_ints(self):
        """All peripheral locants across all PAH entries should be ints."""
        for name, data in POLYCYCLIC_DATA.items():
            numbering = data.get('iupac_numbering', {})
            for atom_idx, locant in numbering.items():
                if not isinstance(locant, str):
                    assert isinstance(locant, int), (
                        f"{name}: atom idx {atom_idx} has {type(locant).__name__} "
                        f"locant {locant}, expected int (peripheral) or str (fusion)"
                    )

    def test_numbering_count_matches_num_atoms(self):
        """For populated entries, iupac_numbering count must match num_atoms."""
        for name, data in POLYCYCLIC_DATA.items():
            numbering = data.get('iupac_numbering', {})
            if numbering:  # Only check populated entries
                assert len(numbering) == data['num_atoms'], (
                    f"{name}: iupac_numbering has {len(numbering)} entries "
                    f"but num_atoms is {data['num_atoms']}"
                )


# ============================================================================
# Phase 147 CD-04 Drift Parity Test
# ============================================================================

@pytest.mark.parametrize("pah_name", [
    "naphthalene", "anthracene", "phenanthrene", "pyrene",
])
def test_iupac_numbering_matches_canonical(pah_name):
    """Phase 147 CD-04 defensive drift test for the 4 populated PAHs.

    Each populated PAH entry's hardcoded ``iupac_numbering`` mapping
    (canonical atom index -> IUPAC locant) is parsed by the live
    ``get_polycyclic_iupac_locants(mol, name)`` wrapper. On the canonical
    molecule, ``mol.GetSubstructMatch(canonical_mol)`` is the identity
    permutation, so the wrapper's returned dict keys ARE canonical atom
    indices. Each value must equal the stored ``iupac_numbering`` value
    after string '4a' -> tuple (4, 'a') normalization (Phase 147 D-01).

    Catches the latent failure mode: a future RDKit version reorders
    canonical atom indices for one of the 4 populated PAH canonical_smiles,
    silently breaking the hardcoded mapping while leaving every other test
    green. By comparing wrapper output to stored data on the canonical mol
    itself, this test fails LOUDLY the moment that drift occurs.

    Source: Phase 147 CONTEXT CD-04 (RDKit canonical-drift guard).
    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html P-25 (PAH numbering FIXED).
    Source: Phase 147 D-01 (string fusion locant -> (int, str) tuple at boundary).
    """
    import re
    from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
    from orthonym.rules.polycyclics import get_polycyclic_iupac_locants

    entry = POLYCYCLIC_DATA[pah_name]
    mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    assert mol is not None, f"Invalid canonical_smiles for {pah_name}"

    result = get_polycyclic_iupac_locants(mol, pah_name)
    assert result is not None, (
        f"get_polycyclic_iupac_locants returned None for {pah_name} on its "
        f"own canonical SMILES — substructure match likely failed."
    )

    stored = entry['iupac_numbering']
    assert len(result) == len(stored), (
        f"{pah_name}: result has {len(result)} entries, stored has "
        f"{len(stored)}"
    )

    locant_re = re.compile(r'^(\d+)([a-z]*)$')
    for canonical_idx, stored_locant in stored.items():
        # On the canonical mol, GetSubstructMatch is identity, so the
        # wrapper's result keys ARE canonical indices.
        got = result[canonical_idx]
        if isinstance(stored_locant, int):
            assert got == stored_locant, (
                f"{pah_name}: canonical idx {canonical_idx}: "
                f"wrapper returned {got}, stored {stored_locant}"
            )
        else:
            # Stored is a string like '4a'; wrapper returns tuple (4, 'a').
            m = locant_re.match(stored_locant)
            assert m is not None, (
                f"{pah_name}: malformed stored locant {stored_locant!r}"
            )
            base = int(m.group(1))
            suffix = m.group(2)
            expected = (base, suffix) if suffix else base
            assert got == expected, (
                f"{pah_name}: canonical idx {canonical_idx}: "
                f"wrapper returned {got}, stored {stored_locant!r} "
                f"(expected {expected})"
            )
