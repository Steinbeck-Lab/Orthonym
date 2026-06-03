"""
Tests for species type detection edge cases (Plan 17-06).

Verifies:
- Large organic molecules with minor charge are reclassified as 'neutral'
- Simple ions, salts, radicals, zwitterions still correctly classified
- "no_suitable_atom" guards prevent acid suffix on non-acid molecules
- "methylidene" false positive eliminated from ion naming
- Namer routing integration works with Plan 17-03 fall-through
"""

import pytest
from rdkit import Chem

from orthonym.perception.ions import detect_species_type
from orthonym.rules.ions import name_anion, name_cation
from orthonym import name_compound


# ============================================================
# TestSpeciesTypeDetection
# ============================================================

class TestSpeciesTypeDetection:
    """Test that detect_species_type() correctly classifies molecules."""

    def test_simple_carboxylate_is_ion(self):
        """Simple carboxylate (acetate) should be classified as 'ion'."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        assert detect_species_type(mol) == 'ion'

    def test_simple_alkoxide_is_ion(self):
        """Simple alkoxide (methoxide) should be classified as 'ion'."""
        mol = Chem.MolFromSmiles('[O-]C')
        assert detect_species_type(mol) == 'ion'

    def test_ammonium_is_ion(self):
        """Ammonium should be classified as 'ion'."""
        mol = Chem.MolFromSmiles('[NH4+]')
        assert detect_species_type(mol) == 'ion'

    def test_sodium_acetate_is_salt(self):
        """Sodium acetate should be classified as 'salt'."""
        mol = Chem.MolFromSmiles('[Na+].[O-]C(C)=O')
        assert detect_species_type(mol) == 'salt'

    def test_neutral_molecule(self):
        """Ethanol should be classified as 'neutral'."""
        mol = Chem.MolFromSmiles('CCO')
        assert detect_species_type(mol) == 'neutral'

    def test_zwitterion_amino_acid(self):
        """Amino acid zwitterion should be classified as 'zwitterion'."""
        # Glycine zwitterion: [NH3+]CC([O-])=O
        mol = Chem.MolFromSmiles('[NH3+]CC([O-])=O')
        assert detect_species_type(mol) == 'zwitterion'

    def test_radical(self):
        """Methyl radical should be classified as 'radical'."""
        mol = Chem.MolFromSmiles('[CH3]')
        assert detect_species_type(mol) == 'radical'

    def test_large_organic_cation_stays_ion(self):
        """169.6-04 (Task 3): the >10-HA size-cutoff band-aid that reclassified a
        large single-charge cation as 'neutral' was REMOVED — it DROPPED the
        charge (CCCCCCCCCCCC[NH3+] -> 'dodecane', discarding the amine).
        route_charged now names it structurally (dodecan-1-aminium, RT=1), so the
        species stays 'ion'."""
        # Dodecan-1-amine protonated: 12 carbons + N + H3+ > 10 heavy atoms
        mol = Chem.MolFromSmiles('CCCCCCCCCCCC[NH3+]')
        assert detect_species_type(mol) == 'ion'

    def test_large_organic_anion_with_carboxylate_stays_ion(self):
        """Large carboxylate should remain 'ion' because carboxylate
        has retained-name tables (acetate, benzoate, etc.)."""
        # Long-chain carboxylate
        mol = Chem.MolFromSmiles('CCCCCCCCCCCC(=O)[O-]')
        assert detect_species_type(mol) == 'ion'

    def test_large_organic_anion_with_alkoxide_stays_ion(self):
        """Large alkoxide should remain 'ion' because alkoxide is a
        recognized anion type with proper naming support."""
        mol = Chem.MolFromSmiles('CCCCCCCCCCCCC[O-]')
        assert detect_species_type(mol) == 'ion'

    def test_small_organic_cation_stays_ion(self):
        """Small organic cation (<=10 atoms) should stay as 'ion'."""
        # Methylammonium: CH3-NH3+ = 2 heavy atoms
        mol = Chem.MolFromSmiles('C[NH3+]')
        assert detect_species_type(mol) == 'ion'

    def test_phenyl_hexylamine_cation_stays_ion(self):
        """169.6-04 (Task 3): an aromatic+chain organic cation stays 'ion' after
        the size-cutoff removal (was forced 'neutral' -> 'unknown organic
        compound'; route_charged now -> 6-phenylhexan-1-aminium, RT=1)."""
        # 6 ring C + 6 chain C + N = 13 heavy atoms
        mol = Chem.MolFromSmiles('c1ccc(cc1)CCCCCC[NH3+]')
        assert detect_species_type(mol) == 'ion'


# ============================================================
# TestNoSuitableAtomGuard
# ============================================================

class TestNoSuitableAtomGuard:
    """Test the name validation guards that prevent misapplied suffixes."""

    def test_valid_carboxylic_acid_anion_returns_oate(self):
        """Acetate anion should return a name ending in -ate."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        result = name_anion(mol)
        assert result  # non-empty
        assert 'ate' in result.lower()

    def test_propanoate_anion(self):
        """Propanoate should return 'propanoate'."""
        mol = Chem.MolFromSmiles('CCC(=O)[O-]')
        result = name_anion(mol)
        assert result  # non-empty
        assert 'oate' in result or 'ate' in result

    def test_molecule_without_carboxyl_no_crash(self):
        """Non-carboxylate anion naming should not crash."""
        # Simple carbanion
        mol = Chem.MolFromSmiles('[CH3-]')
        result = name_anion(mol)
        # Should return something (methanide) or empty string, not crash
        assert isinstance(result, str)

    def test_valid_amine_cation_returns_aminium(self):
        """Ammonium should return 'ammonium'."""
        mol = Chem.MolFromSmiles('[NH4+]')
        result = name_cation(mol)
        assert 'ammonium' in result

    def test_methylammonium_cation(self):
        """Methylamine cation should return name with 'ammonium'."""
        mol = Chem.MolFromSmiles('C[NH3+]')
        result = name_cation(mol)
        assert result
        assert 'ammonium' in result

    def test_non_amine_cation_no_crash(self):
        """Non-amine cation naming should not crash."""
        # Carbocation
        mol = Chem.MolFromSmiles('[CH3+]')
        result = name_cation(mol)
        # Should return methylium or similar, not crash
        assert isinstance(result, str)

    def test_retained_only_compatibility(self):
        """retained_only=True should return None for non-retained ions
        (Plan 17-03 compatibility)."""
        # Pentanoate - not in the retained ion names table
        mol = Chem.MolFromSmiles('CCCCC(=O)[O-]')
        result = name_anion(mol, retained_only=True)
        assert result is None

    def test_retained_only_returns_retained_name(self):
        """retained_only=True should return the retained name when it exists."""
        mol = Chem.MolFromSmiles('CC(=O)[O-]')
        result = name_anion(mol, retained_only=True)
        assert result == 'acetate'


# ============================================================
# TestMethylideneGuard
# ============================================================

class TestMethylideneGuard:
    """Test that 'methylidene' false positive is eliminated from ion naming."""

    def test_no_ion_name_contains_methylidene(self):
        """Ion naming should never produce a name containing 'methylidene'."""
        # Various charged molecules that could potentially trigger the issue
        test_smiles = [
            '[CH2+]',       # carbocation
            'C=[NH2+]',     # iminium
            '[CH2-]',       # carbanion
            'C[NH3+]',      # methylammonium
            'CC(=O)[O-]',   # acetate
        ]
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            # Try both anion and cation naming
            charge = Chem.GetFormalCharge(mol)
            if charge < 0:
                result = name_anion(mol)
            elif charge > 0:
                result = name_cation(mol)
            else:
                continue
            assert 'methylidene' not in (result or ''), \
                f"SMILES {smi} produced name '{result}' containing 'methylidene'"

    def test_charged_nitrogen_no_methylidene(self):
        """Charged nitrogen compounds should not produce 'methylidene'."""
        mol = Chem.MolFromSmiles('C=[NH2+]')
        result = name_cation(mol)
        assert 'methylidene' not in (result or '')

    def test_valid_methylidene_radical_preserved(self):
        """True methylidene radical should still be named correctly
        (radical naming is separate from ion naming)."""
        # [CH2] is a true divalent radical
        mol = Chem.MolFromSmiles('[CH2]')
        species_type = detect_species_type(mol)
        assert species_type == 'radical'
        # The radical naming path (not tested here) should preserve 'methylidene'
        # We only test that it's NOT an ion
        name = name_compound('[CH2]')
        assert 'methylidene' in name


# ============================================================
# TestNamerRoutingIntegration
# ============================================================

class TestNamerRoutingIntegration:
    """Test integration with the namer.py routing established in Plan 17-03."""

    def test_large_charged_organic_gets_valid_name(self):
        """Large organic molecule with charge should get a valid organic name,
        not an ion name."""
        # Protonated dodecylamine: should get organic name via normal pipeline
        mol = Chem.MolFromSmiles('CCCCCCCCCCCC[NH3+]')
        name = name_compound('CCCCCCCCCCCC[NH3+]')
        assert name  # non-empty
        # Should NOT be a simple ion name like "dodecylammonium"
        # Instead should route through normal pipeline

    def test_simple_ion_with_retained_name(self):
        """Simple ion with a retained name should return that name."""
        # Acetate has a retained name
        name = name_compound('CC(=O)[O-]')
        assert name == 'acetate'

    def test_ammonium_retained_name(self):
        """Ammonium should return 'ammonium'."""
        name = name_compound('[NH4+]')
        assert name == 'ammonium'

    def test_simple_ion_without_retained_falls_through(self):
        """Simple ion without retained name should fall through to
        aspect composition in composer.py (Plan 17-03 routing)."""
        # Butanoate - not in retained names, but small enough to stay 'ion'
        name = name_compound('CCCC(=O)[O-]')
        assert name  # non-empty
        # Should produce a valid oate name
        assert 'oate' in name or 'ate' in name
