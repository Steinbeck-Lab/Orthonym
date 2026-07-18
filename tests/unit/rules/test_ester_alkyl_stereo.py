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


class TestClusterDPolyfunctionalEsterAlkylStereo:
    """Wave-8 P6 Cluster D (P-93.4.1.3): the alkyl-side stereo descriptor was
    dropped specifically on the POLYFUNCTIONAL-routed ester path (when the
    acid side carries a JUNIOR functional group, e.g. -OH, alongside the
    ester -- routing through name_polyfunctional_ester_via_acid instead of
    the plain name_ester). The plain-ester STER-12 collector
    (_collect_alkyl_fragment_stereo) already worked correctly (verified by
    calling name_ester directly); the polyfunctional path simply never
    called it. BB P-93.4.1.3: a stereodescriptor for a component cited as a
    separate word/prefix is placed immediately before that component, not
    hoisted to the front of the whole name.
    """

    def test_lactate_ester_both_components_carry_own_stereo(self):
        """C[C@@H](O)C(=O)O[C@@H](C)CC -- both the acid (2-hydroxypropanoate)
        and the alkyl (butan-2-yl) components are chiral; BOTH must carry
        their own descriptor."""
        from orthonym import Orthonym
        o = Orthonym(_disable_opsin_validity_gate=True)
        name = o.name("C[C@@H](O)C(=O)O[C@@H](C)CC")
        assert name == "(2S)-butan-2-yl (2R)-2-hydroxypropanoate", f"got {name!r}"

    def test_gated_matches_raw(self):
        from orthonym import Orthonym
        o_raw = Orthonym(_disable_opsin_validity_gate=True)
        o_gated = Orthonym()
        smi = "C[C@@H](O)C(=O)O[C@@H](C)CC"
        assert o_raw.name(smi) == o_gated.name(smi)
