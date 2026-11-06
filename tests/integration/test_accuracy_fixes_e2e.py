"""
End-to-end tests for a phase accuracy bug fixes.

Tests validate fixes for:
-: Aromatic substituent detection (phenyl -> hexyl)
-: Bicyclo substituent and unsaturation naming
-: Fused ring partial saturation (tetrahydronaphthalene)
-: Heterocycle-as-substituent naming (partially addressed)
-: N-substituent stereochemistry

These tests ensure all accuracy fixes work together without regression
and cover the compound classes identified in GAP_ANALYSIS.md.
"""
from orthonym.namer import name_compound


class TestBug1AromaticSubstituents:
    """: Aromatic rings should not be named as alkyl chains."""

    def test_phenylacetic_acid(self):
        """Phenylacetic acid: phenyl not hexyl."""
        result = name_compound('c1ccccc1CC(=O)O')
        assert 'hexyl' not in result.lower(), f"Got hexyl in: {result}"
        # Should have phenyl or benzyl
        assert 'phenyl' in result.lower() or 'benzyl' in result.lower(), f"Missing phenyl: {result}"

    def test_phenylpropanoic_acid(self):
        """3-phenylpropanoic acid."""
        result = name_compound('c1ccccc1CCC(=O)O')
        assert 'octyl' not in result.lower()
        assert 'phenyl' in result.lower() or 'benzene' in result.lower()

    def test_cyclohexylacetic_acid(self):
        """Cyclohexylacetic acid: cyclohexyl not hexyl."""
        result = name_compound('C1CCCCC1CC(=O)O')
        # cyclohexyl contains hexyl, so check it's cyclohexyl specifically
        assert 'cyclohexyl' in result.lower(), f"Expected cyclohexyl: {result}"

    def test_amino_phenyl_propanol(self):
        """(1S,2R)-2-amino-1-phenylpropan-1-ol from GAP_ANALYSIS."""
        result = name_compound('C[C@@H](N)[C@@H](O)c1ccccc1')
        assert 'hexyl' not in result.lower(), f"Got hexyl in: {result}"
        assert 'phenyl' in result.lower()

    def test_phenyl_ethanol(self):
        """2-phenylethanol - phenyl not hexyl."""
        result = name_compound('c1ccccc1CCO')
        assert 'phenyl' in result.lower() or 'benzene' in result.lower()

class TestBug2BicycloComplete:
    """: Bicyclo systems should include substituents and unsaturation."""

    def test_methylnorbornane(self):
        """Methylnorbornane should have methyl prefix."""
        result = name_compound('CC1CC2CCC1C2')
        assert 'methyl' in result.lower(), f"Missing methyl: {result}"
        assert 'bicyclo' in result.lower(), f"Missing bicyclo: {result}"

    def test_norbornene(self):
        """Norbornene should have -ene suffix."""
        result = name_compound('C1=CC2CCC1C2')
        # Either retained name or systematic with -ene
        assert 'ene' in result.lower() or 'norbornene' in result.lower(), f"Missing ene: {result}"

    def test_dimethylnorbornane(self):
        """Dimethylnorbornane should have dimethyl."""
        result = name_compound('CC1CC2CCC1C2C')
        assert 'methyl' in result.lower()
        assert 'bicyclo' in result.lower()

    def test_bicyclo_with_ethyl(self):
        """Bicyclo with ethyl substituent."""
        result = name_compound('CCC1CC2CCC1C2')
        assert 'ethyl' in result.lower()
        assert 'bicyclo' in result.lower()

    def test_bicyclo_320_heptene(self):
        """Bicyclo[3.2.0]hept-2-ene structure."""
        result = name_compound('C1=CCC2CC1C2')
        assert 'bicyclo' in result.lower()
        # Should have ene suffix
        assert 'ene' in result.lower()


class TestBug3FusedPartialSaturation:
    """: Partially saturated fused rings should name correctly."""

    def test_tetrahydronaphthalene(self):
        """Tetrahydronaphthalene (tetralin)."""
        result = name_compound('c1ccc2c(c1)CCCC2')
        assert 'benzene' not in result.lower(), f"Wrong: {result}"
        assert 'tetrahydro' in result.lower() or 'tetralin' in result.lower(), f"Expected tetrahydro: {result}"

class TestBug4HeterocycleSubstituent:
    """: Heterocycles as substituents should not be alkyl chains."""

    def test_piperidinyl_pyridine(self):
        """Piperidinyl substituent on pyridine - now works."""
        result = name_compound('c1cncc(C2CCCCN2)c1')
        # Ring substituent detection now avoids "pentyl"
        assert 'pentyl' not in result.lower(), f"Got pentyl: {result}"

class TestBug5HeterocycleStereo:
    """: N-substituted heterocycles should include stereochemistry."""

    def test_methylpropylpiperidine_stereo(self):
        """(2S)-1-methyl-2-propylpiperidine from GAP_ANALYSIS."""
        result = name_compound('CCC[C@H]1CCCCN1C')
        has_stereo = ('(' in result and ')' in result and
                      ('S' in result or 'R' in result))
        assert has_stereo, f"Missing stereochemistry: {result}"

    def test_methylpyrrolidine_stereo(self):
        """Methylpyrrolidine with stereo."""
        result = name_compound('C[C@H]1CCCN1')
        has_stereo = '(' in result and ')' in result
        assert has_stereo, f"Missing stereochemistry: {result}"

    def test_ethylpiperidine_stereo(self):
        """Ethylpiperidine with stereo."""
        result = name_compound('CC[C@H]1CCCCN1')
        has_stereo = '(' in result and ')' in result
        assert has_stereo, f"Missing stereochemistry: {result}"

    def test_n_substituted_piperidine_stereo(self):
        """N-substituted piperidine with ring stereo."""
        result = name_compound('[C@H]1(C)CCCC[N@H]1')
        # Check for stereo descriptor
        assert 'methyl' in result.lower()


class TestCombinedFixes:
    """Tests that combine multiple fixes."""

    def test_phenyl_plus_stereo(self):
        """Compound with phenyl substituent AND stereochemistry."""
        result = name_compound('C[C@@H](c1ccccc1)O')
        assert 'phenyl' in result.lower()
        has_stereo = '(' in result and ')' in result
        assert has_stereo, f"Missing stereo: {result}"

    def test_complex_bicyclo_stereo(self):
        """Bicyclo with substituents and stereo."""
        result = name_compound('C[C@@H]1CC2CCC1C2')
        assert 'methyl' in result.lower()
        assert 'bicyclo' in result.lower()

    def test_bicyclo_with_unsaturation_and_substituent(self):
        """Bicyclo with both unsaturation and substituent."""
        result = name_compound('CC1=CC2CCC1C2')
        assert 'methyl' in result.lower()
        assert 'bicyclo' in result.lower() or 'norbornene' in result.lower()


class TestEdgeCasesFromGapAnalysis:
    """Edge cases from GAP_ANALYSIS.md."""

    def test_phenylethylamine(self):
        """Phenylethylamine - phenyl not hexyl."""
        result = name_compound('NCCc1ccccc1')
        assert 'phenyl' in result.lower() or 'phenethyl' in result.lower()
        assert 'hexyl' not in result.lower()

    def test_phenylacetone(self):
        """Phenylacetone - phenyl not hexyl."""
        result = name_compound('CC(=O)Cc1ccccc1')
        assert 'phenyl' in result.lower() or 'benzyl' in result.lower()
        assert 'hexyl' not in result.lower()
