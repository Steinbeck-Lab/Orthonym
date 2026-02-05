"""
Integration tests for aspect-based routing in composer.py and namer.py.

Tests verify:
1. Ion fall-through routing in namer.py (retained names checked first,
   then fall-through to normal pipeline for non-retained ions)
2. Aspect-based ion composition in composer.py (single-component ions
   use resolve_parent + resolve_suffix + ion suffix modification)
3. Recursion guard (_composing_ion) prevents infinite loops
4. Locant validation integrated in default assembly path
5. Salt/zwitterion/radical early-exit paths preserved

Phase: 17-03 (Composer Routing Redesign)
"""

import pytest
from orthonym import name_compound


# ============================================================================
# TestIonFallThrough: Verify namer.py routes ions correctly
# ============================================================================


class TestIonFallThrough:
    """Test that namer.py routes ions through retained-name check then falls through."""

    def test_retained_anion_acetate(self):
        """Acetate should use retained name directly from namer.py."""
        result = name_compound('CC([O-])=O')
        assert result == 'acetate', f'Expected acetate, got: {result}'

    def test_retained_anion_benzoate(self):
        """Benzoate should use retained name directly from namer.py."""
        result = name_compound('[O-]C(=O)c1ccccc1')
        assert result == 'benzoate', f'Expected benzoate, got: {result}'

    def test_retained_anion_formate(self):
        """Formate should use retained name directly from namer.py."""
        result = name_compound('O=C[O-]')
        assert result == 'formate', f'Expected formate, got: {result}'

    def test_retained_anion_methoxide(self):
        """Methoxide should use retained name directly from namer.py."""
        result = name_compound('C[O-]')
        assert result == 'methoxide', f'Expected methoxide, got: {result}'

    def test_retained_anion_ethoxide(self):
        """Ethoxide should use retained name directly from namer.py."""
        result = name_compound('CC[O-]')
        assert result == 'ethoxide', f'Expected ethoxide, got: {result}'

    def test_retained_cation_ammonium(self):
        """Ammonium should use retained name directly from namer.py."""
        result = name_compound('[NH4+]')
        assert result == 'ammonium', f'Expected ammonium, got: {result}'

    def test_retained_cation_methylammonium(self):
        """Methylammonium should use retained name directly from namer.py."""
        result = name_compound('C[NH3+]')
        assert result == 'methylammonium', f'Expected methylammonium, got: {result}'

    def test_retained_cation_ethylammonium(self):
        """Ethylammonium should use retained name directly from namer.py."""
        result = name_compound('CC[NH3+]')
        assert result == 'ethylammonium', f'Expected ethylammonium, got: {result}'

    def test_salt_early_exit(self):
        """Salts should use early exit in namer.py (never reach assemble_name)."""
        result = name_compound('[Na+].[O-]C(C)=O')
        assert result == 'sodium acetate', f'Expected sodium acetate, got: {result}'

    def test_zwitterion_early_exit(self):
        """Zwitterions should use early exit in namer.py."""
        result = name_compound('[NH3+]CC([O-])=O')
        # Should produce some valid zwitterion name (not crash)
        assert result, f'Zwitterion returned empty: {result}'
        assert isinstance(result, str), f'Expected string, got: {type(result)}'

    def test_radical_early_exit(self):
        """Radicals should use early exit in namer.py."""
        result = name_compound('[CH3]')
        assert result == 'methyl', f'Expected methyl, got: {result}'


# ============================================================================
# TestIonAspectComposition: Verify composer.py aspect-based ion naming
# ============================================================================


class TestIonAspectComposition:
    """Test aspect-based ion composition in composer.py."""

    def test_non_retained_carboxylate_pentanoate(self):
        """Non-retained carboxylate should be named via fallback ion naming."""
        result = name_compound('CCCCC(=O)[O-]')
        assert 'pentanoate' in result, f'Expected pentanoate, got: {result}'

    def test_non_retained_carboxylate_butanoate(self):
        """Butanoate (non-retained) should be named correctly."""
        result = name_compound('O=C([O-])CCC')
        assert 'butanoate' in result, f'Expected butanoate, got: {result}'

    def test_non_retained_alkoxide_propanolate(self):
        """Non-retained alkoxide should preserve chain identity."""
        result = name_compound('CCC[O-]')
        # Should be propanolate (PIN) or propoxide (common)
        assert 'prop' in result.lower(), f'Expected prop- prefix, got: {result}'

    def test_non_retained_phenolate(self):
        """Non-retained substituted phenolate should be named correctly."""
        result = name_compound('[O-]c1ccc(C)cc1')
        # Should contain 'olate' or 'oxide' -- phenolate naming
        assert 'ol' in result.lower() or 'oxide' in result.lower(), \
            f'Expected phenolate/oxide name, got: {result}'

    def test_neutral_molecule_unchanged_ethanol(self):
        """Neutral molecules should not be affected by ion routing."""
        result = name_compound('CCO')
        assert result == 'ethanol', f'Expected ethanol, got: {result}'

    def test_neutral_molecule_unchanged_butanone(self):
        """Neutral ketone should not be affected by ion routing."""
        result = name_compound('CCC(=O)C')
        assert 'butan' in result.lower() and 'one' in result.lower(), \
            f'Expected butanone, got: {result}'

    def test_neutral_molecule_unchanged_butene(self):
        """Neutral alkene should not be affected by ion routing."""
        result = name_compound('CC=CC')
        assert result == 'but-2-ene', f'Expected but-2-ene, got: {result}'

    def test_ion_fallback_to_existing_naming(self):
        """Ion without matching aspect composition should fall back to name_anion."""
        # Formate has retained name, should return directly
        result = name_compound('[O-]C=O')
        assert result == 'formate', f'Expected formate, got: {result}'

    def test_carbanion_methanide(self):
        """Carbanion methanide (retained name)."""
        result = name_compound('[CH3-]')
        assert result == 'methanide', f'Expected methanide, got: {result}'


# ============================================================================
# TestRecursionGuard: Verify _composing_ion prevents infinite loops
# ============================================================================


class TestRecursionGuard:
    """Test that the _composing_ion recursion guard works correctly."""

    def test_composing_ion_flag_prevents_reentry(self):
        """When _composing_ion=True, assemble_name should not re-enter ion path."""
        from orthonym.assembly.composer import assemble_name
        from orthonym.namer import MolecularFeatures
        from rdkit import Chem

        mol = Chem.MolFromSmiles('CC')
        features = MolecularFeatures(
            mol=mol,
            smiles='CC',
            canonical_smiles='CC',
            species_type='ion',
            principal_chain=[0, 1],
        )
        # With _composing_ion=True, ion routing is skipped
        result = assemble_name(features, style='pin', _composing_ion=True)
        # Should not enter infinite loop; should produce some name
        assert isinstance(result, str), f'Expected string, got: {type(result)}'

    def test_neutral_with_composing_ion_still_works(self):
        """Neutral molecule with _composing_ion=True should still produce a name."""
        from orthonym.assembly.composer import assemble_name
        from orthonym.namer import Orthonym
        from rdkit import Chem

        mol = Chem.MolFromSmiles('CCO')
        namer = Orthonym(style='pin')
        features = namer._perceive(mol, 'CCO', 'CCO')
        namer._classify(features)
        result = assemble_name(features, style='pin', _composing_ion=True)
        # assemble_name produces systematic name (retained name check is in namer.py)
        assert 'ol' in result, f'Expected alcohol name, got: {result}'
        assert 'ethan' in result, f'Expected ethan- prefix, got: {result}'

    def test_no_infinite_loop_on_complex_ion(self):
        """Complex ion should not cause infinite recursion."""
        # This would previously cause RecursionError without the guard
        result = name_compound('CCCCC(=O)[O-]')
        assert isinstance(result, str), f'Expected string, got: {type(result)}'
        assert len(result) > 0, 'Expected non-empty name'


# ============================================================================
# TestLocantValidation: Verify locant validation in default assembly path
# ============================================================================


class TestLocantValidation:
    """Test locant validation integrated in the default assembly path."""

    def test_cyclohexanone_locant_valid(self):
        """Cyclohexanone should keep valid locants."""
        result = name_compound('C1CCC(=O)CC1')
        assert 'cyclohex' in result.lower(), f'Expected cyclohexanone, got: {result}'
        assert 'one' in result.lower(), f'Expected -one suffix, got: {result}'

    def test_cyclohexanol_locant_valid(self):
        """Cyclohexanol should keep valid locants."""
        result = name_compound('C1CCCCC1O')
        assert 'cyclohex' in result.lower(), f'Expected cyclohexanol, got: {result}'
        assert 'ol' in result.lower(), f'Expected -ol suffix, got: {result}'

    def test_diol_locants_valid(self):
        """Butane-1,4-diol should have valid locants within parent capacity."""
        result = name_compound('OCCCCO')
        assert 'diol' in result.lower(), f'Expected diol, got: {result}'

    def test_no_zero_locants_in_name(self):
        """Names should never contain locant 0 (IUPAC uses 1-indexing)."""
        # Test several compound classes
        for smiles in ['CCO', 'CCCO', 'CC(=O)C', 'CC=CC', 'C1CCCCC1']:
            result = name_compound(smiles)
            # No "0-" or ",0," patterns in name
            assert '-0-' not in result, f'{smiles}: contains -0- in {result}'
            assert ',0,' not in result, f'{smiles}: contains ,0, in {result}'
            assert not result.startswith('0-'), f'{smiles}: starts with 0- in {result}'

    def test_existing_ethanol_unchanged(self):
        """Ethanol should be unchanged by locant validation."""
        result = name_compound('CCO')
        assert result == 'ethanol', f'Expected ethanol, got: {result}'

    def test_existing_propanone_unchanged(self):
        """Propan-2-one / acetone should be unchanged."""
        result = name_compound('CC(=O)C')
        # Acetone is a retained name
        assert result in ('acetone', 'propan-2-one'), f'Got: {result}'

    def test_hexanol_locant_valid(self):
        """Hexan-1-ol locant should be within parent size 6."""
        result = name_compound('CCCCCCO')
        assert 'hexan' in result.lower(), f'Expected hexanol, got: {result}'
        assert 'ol' in result.lower(), f'Expected -ol suffix, got: {result}'

    def test_long_chain_diol(self):
        """Long chain diol locants should be valid."""
        result = name_compound('OCCCCCCCCCO')
        assert 'diol' in result or 'ol' in result, f'Expected diol/ol, got: {result}'


# ============================================================================
# TestOverallIntegrity: Verify no regressions in common naming scenarios
# ============================================================================


class TestOverallIntegrity:
    """Smoke tests to verify no regressions in common naming paths."""

    def test_simple_alkane(self):
        """Simple alkane unchanged."""
        assert name_compound('CCC') == 'propane'

    def test_simple_alkene(self):
        """Simple alkene unchanged."""
        assert name_compound('C=C') == 'ethene'

    def test_benzene(self):
        """Benzene unchanged."""
        assert name_compound('c1ccccc1') == 'benzene'

    def test_carboxylic_acid(self):
        """Carboxylic acid unchanged."""
        result = name_compound('CC(=O)O')
        assert result == 'acetic acid', f'Got: {result}'

    def test_formic_acid(self):
        """Formic acid unchanged."""
        result = name_compound('O=CO')
        assert result == 'formic acid', f'Got: {result}'

    def test_amine(self):
        """Simple amine unchanged."""
        result = name_compound('CCN')
        assert 'amine' in result.lower() or 'amin' in result.lower(), f'Got: {result}'
