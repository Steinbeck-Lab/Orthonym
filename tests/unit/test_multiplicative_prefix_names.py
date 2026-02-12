"""Tests for multiplicative naming with prefix-derived parent names (PEP-03).

Validates that _assemble_multiplicative_name() does not produce unparseable
"di" + saturation prefix concatenation (e.g., "ditetrahydropyran") when the
parent name starts with a modification prefix like tetrahydro, dihydro, etc.

Simple parent names (aniline, benzene, benzoic acid) must still produce
the standard di-prefix format.
"""

import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestMultiplicativeNoDiTetrahydro:
    """Multiplicative names must not produce 'ditetrahydro' concatenation."""

    def test_oxydi_tetrahydropyran_no_ditetrahydro(self):
        """4,4'-oxybis(tetrahydropyran) or substitutive form.
        SMILES: C1CCOC(C1)OC2CCOCC2
        Must NOT contain 'ditetrahydropyran' -- OPSIN cannot parse it.
        """
        name = name_compound("C1CCOC(C1)OC2CCOCC2")
        assert name is not None, "Should produce a name"
        assert "ditetrahydro" not in name, (
            f"Produced unparseable 'ditetrahydro' concatenation: {name}"
        )

    def test_methylenedi_tetrahydropyran_no_ditetrahydro(self):
        """methylenebis(tetrahydropyran) or substitutive form.
        SMILES: C1CCOC(C1)CC2CCOCC2
        Must NOT contain 'ditetrahydropyran'.
        """
        name = name_compound("C1CCOC(C1)CC2CCOCC2")
        assert name is not None, "Should produce a name"
        assert "ditetrahydro" not in name, (
            f"Produced unparseable 'ditetrahydro' concatenation: {name}"
        )


@pytest.mark.unit
class TestMultiplicativeSimpleParentsStillWork:
    """Simple parent names must still produce standard di-prefix format."""

    def test_oxydianiline_still_works(self):
        """4,4'-oxydianiline: c1cc(ccc1N)Oc2ccc(cc2)N
        Simple retained name 'aniline' should still get di-prefix.
        """
        name = name_compound("c1cc(ccc1N)Oc2ccc(cc2)N")
        assert name is not None, "Should produce a name"
        assert "dianiline" in name, (
            f"Expected 'dianiline' in name for simple parent, got: {name}"
        )

    def test_oxydibenzoic_acid_still_works(self):
        """4,4'-oxydibenzoic acid: OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1
        Space-containing parent 'benzoic acid' should still work.
        """
        name = name_compound("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1")
        assert name is not None, "Should produce a name"
        assert "dibenzoic acid" in name, (
            f"Expected 'dibenzoic acid' in name, got: {name}"
        )


@pytest.mark.unit
class TestMultiplicativeNoOtherPrefixConcat:
    """No other saturation prefix should produce 'di+prefix' concatenation."""

    def test_no_didihydro_general(self):
        """Any multiplicative compound with dihydro parent must not produce 'didihydro'.
        This is a safeguard test -- verifies the SATURATION_PREFIXES list.
        """
        from orthonym.rules.multiplicative import SATURATION_PREFIXES
        assert "dihydro" in SATURATION_PREFIXES, (
            "'dihydro' must be in SATURATION_PREFIXES list"
        )

    def test_no_dihexahydro_general(self):
        """Any multiplicative compound with hexahydro parent must not produce 'dihexahydro'.
        Verifies the SATURATION_PREFIXES list includes hexahydro.
        """
        from orthonym.rules.multiplicative import SATURATION_PREFIXES
        assert "hexahydro" in SATURATION_PREFIXES, (
            "'hexahydro' must be in SATURATION_PREFIXES list"
        )

    def test_saturation_prefixes_complete(self):
        """SATURATION_PREFIXES should include all common saturation/modification prefixes."""
        from orthonym.rules.multiplicative import SATURATION_PREFIXES
        expected = {"tetrahydro", "dihydro", "hexahydro", "perhydro", "octahydro"}
        for prefix in expected:
            assert prefix in SATURATION_PREFIXES, (
                f"'{prefix}' missing from SATURATION_PREFIXES"
            )
