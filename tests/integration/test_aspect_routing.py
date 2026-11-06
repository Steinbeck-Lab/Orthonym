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

    def test_retained_anion_formate(self):
        """Formate should use retained name directly from namer.py."""
        result = name_compound('O=C[O-]')
        assert result == 'formate', f'Expected formate, got: {result}'

    def test_retained_anion_ethoxide(self):
        """Ethoxide should use retained name directly from namer.py."""
        result = name_compound('CC[O-]')
        assert result == 'ethoxide', f'Expected ethoxide, got: {result}'

    def test_retained_cation_ethylammonium(self):
        """CH3CH2-NH3+ is named by the substitutive aminium PIN, not 'ethylammonium'."""
        # method (1) (commit 12541211e); see test_retained_cation_methylammonium.
        result = name_compound('CC[NH3+]')
        assert result == 'ethanaminium', f'Expected ethanaminium, got: {result}'

    def test_radical_early_exit(self):
        """Radicals should use early exit in namer.py."""
        result = name_compound('[CH3]')
        assert result == 'methyl', f'Expected methyl, got: {result}'


# ============================================================================
# TestIonAspectComposition: Verify composer.py aspect-based ion naming
# ============================================================================


class TestIonAspectComposition:
    """Test aspect-based ion composition in composer.py."""

    def test_neutral_molecule_unchanged_butanone(self):
        """Neutral ketone should not be affected by ion routing."""
        result = name_compound('CCC(=O)C')
        assert 'butan' in result.lower() and 'one' in result.lower(), \
            f'Expected butanone, got: {result}'

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

    def test_no_zero_locants_in_name(self):
        """Names should never contain locant 0 (IUPAC uses 1-indexing)."""
        # Test several compound classes
        for smiles in ['CCO', 'CCCO', 'CC(=O)C', 'CC=CC', 'C1CCCCC1']:
            result = name_compound(smiles)
            # No "0-" or ",0," patterns in name
            assert '-0-' not in result, f'{smiles}: contains -0- in {result}'
            assert ',0,' not in result, f'{smiles}: contains ,0, in {result}'
            assert not result.startswith('0-'), f'{smiles}: starts with 0- in {result}'

    def test_long_chain_diol(self):
        """Long chain diol locants should be valid."""
        result = name_compound('OCCCCCCCCCO')
        assert 'diol' in result or 'ol' in result, f'Expected diol/ol, got: {result}'
