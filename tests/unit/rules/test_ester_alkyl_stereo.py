"""
Tests for STER-12: Ester alkyl-side (alcohol fragment) stereo injection.

Verifies that ester naming collects and injects CIP stereodescriptors
for the alkyl (alcohol-side) fragment, mirroring the existing acid-side
stereo pattern.

Key behavior:
- sec-butyl acetate with stereo -> includes (R)- or (S)- on alkyl name
- Achiral esters (ethyl acetate, propan-2-yl acetate) -> no stereo
- Acid-side stereo continues to work (regression check)
"""

import pytest
from orthonym.namer import name_compound


class TestAlkylSideStereoPresent:
    """Esters with chiral alkyl fragments must include stereo on alkyl portion."""

    def test_sec_butyl_acetate_has_stereo(self):
        """CC(=O)O[C@@H](C)CC should have stereo on the alkyl portion."""
        name = name_compound("CC(=O)O[C@@H](C)CC")
        assert name is not None
        # The alkyl name should have a stereo prefix like "(2R)-butan-2-yl"
        # or "(R)-sec-butyl" -- just check stereo is present
        assert "(" in name and ("R" in name or "S" in name), (
            f"Expected stereo prefix for chiral alkyl fragment, got: {name}"
        )

    def test_sec_butyl_acetate_alternate_smiles(self):
        """Same molecule with different SMILES input should also include stereo."""
        name = name_compound("[C@@H](OC(=O)C)(C)CC")
        assert name is not None
        assert "(" in name and ("R" in name or "S" in name), (
            f"Expected stereo prefix for chiral alkyl fragment, got: {name}"
        )

    def test_chiral_1_methylpropyl_benzoate(self):
        """Chiral alkyl fragment on aromatic ester should have stereo."""
        name = name_compound("O=C(c1ccccc1)O[C@@H](C)CC")
        assert name is not None
        # Should include some form of stereo
        assert "R" in name or "S" in name, (
            f"Expected stereo for chiral alkyl fragment, got: {name}"
        )


class TestAlkylSideStereoAbsent:
    """Achiral alkyl fragments must NOT include stereo prefix."""

    def test_ethyl_acetate_no_stereo(self):
        """Ethyl acetate (achiral) should not have stereo on alkyl portion."""
        name = name_compound("CC(=O)OCC")
        assert name is not None
        # "ethyl acetate" -- no stereo expected
        # We check that there's no R/S in the name or no stereo prefix
        # Note: the name could be "ethyl acetate" (retained) or similar
        if "(" in name:
            # If there's a parenthesized part, it should not be a stereo prefix
            import re
            stereo_match = re.match(r'^\(\d*[RS]', name)
            assert stereo_match is None, (
                f"Achiral ethyl acetate should not have stereo prefix, got: {name}"
            )

    def test_propan_2_yl_acetate_no_stereo(self):
        """propan-2-yl acetate (achiral -- symmetric) should not have stereo."""
        name = name_compound("CC(=O)OC(C)C")
        assert name is not None
        # isopropyl/propan-2-yl -- no chiral center
        import re
        stereo_match = re.match(r'^\(\d*[RS]', name)
        assert stereo_match is None, (
            f"Achiral propan-2-yl acetate should not have stereo, got: {name}"
        )


class TestAcidSideStereoRegression:
    """Acid-side stereo must continue to work after alkyl-side changes."""

    def test_acid_side_stereo_preserved(self):
        """Acid-side stereo should still be present after alkyl-side changes."""
        name = name_compound("C[C@@H](CC)C(=O)OC")
        assert name is not None
        # Should be something like "methyl (2S)-2-methylbutanoate"
        assert "(" in name and ("R" in name or "S" in name), (
            f"Expected acid-side stereo to be preserved, got: {name}"
        )

    def test_acid_side_stereo_format(self):
        """Acid-side stereo should have the CIP prefix on the acylate portion."""
        name = name_compound("C[C@@H](CC)C(=O)OC")
        assert name is not None
        # The stereo prefix should be before the acylate name
        # "methyl (2S)-2-methylbutanoate" -- stereo is in the acylate part
        assert "2S" in name or "2R" in name, (
            f"Expected acid-side stereo with locant, got: {name}"
        )
