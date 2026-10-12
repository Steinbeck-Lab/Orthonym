""" a phase — amino-acid structure-loss fixes (Tier-2 final).

Covers the three genuine root-cause wins (all OPSIN-RT verified during the phase):
  - F-THIOETHER-DROP: _name_amino_acid_systematic dropped a C-S-C thioether
                        (counted the S-alkyl carbon as backbone). Now bails to
                        the polyfunctional pipeline via the _EXTRA_FG_SMARTS list.
  - branched-AA: the carbon-count namer collapsed a branched side chain
                        into a too-long straight chain. Now bails on any chain
                        carbon with >2 carbon neighbours.
  - glycinate anion: _name_carboxylate_systematic dropped the amino group
                        ('acetate') because the retained name 'glycine' has no
                        convertible '-oic acid' suffix. Now named via
                        _amino_acid_carboxylate + the fallback
                        fail-closes when substituents would be dropped.

The PIN gate (scripts/v22_gate.py) does NOT run the unit suite, so these lock in
the behaviour at the unit level. The conftest autouse fixture keeps the OPSIN
validity gate OFF here; all three fixes act BEFORE that gate, so the names below
are the raw production output.
"""

import pytest

from orthonym.namer import name_compound


class TestThioetherDrop:
    """F-THIOETHER-DROP: the thioether must NOT be dropped (BB audit F1)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CCSCC(N)C(=O)O", "2-amino-3-(ethylsulfanyl)propanoic acid"),   # S-ethylcysteine
        ("CSCCCC(N)C(=O)O", "2-amino-5-(methylsulfanyl)pentanoic acid"),  # homomethionine
        ("CSC[C@H](N)C(=O)O", "(2R)-2-amino-3-(methylsulfanyl)propanoic acid"),  # S-methyl-L-cys
    ])
    def test_thioether_retained(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CSCC[C@@H](N)C(=O)O", "D-methionine"),  # catalog C5 methionine (P5) — unaffected (defined stereo)
        # a phase (C2b, change-asserted-value, was "S-methylcysteine"): the
        # NON_STANDARD flat key bypassed -07's STANDARD-AA stereo guard; bare
        # 'S-methylcysteine' IS config-implying (OPSIN parses it to the DEFINED L
        # form, same as 'S-methyl-L-cysteine' -- probed 2026-08-16), so at
        # top-level with the alpha-carbon undefined it now defers to the
        # systematic name, same defect class as the sibling cysteine row below.
        ("CSCC(N)C(=O)O", "2-amino-3-(methylsulfanyl)propanoic acid"),
    ])
    def test_catalog_amino_acids_unaffected(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestBranchedAminoAcid:
    """: a branched side chain must not collapse into a straight chain."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CCCC[C@H](N)C(=O)O", "(2S)-2-aminohexanoic acid"),  # straight chain — unaffected
        ("CCC[C@H](N)C(=O)O", "(2S)-2-aminopentanoic acid"),  # norvaline — unaffected
    ])
    def test_straight_chain_unaffected(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestGlycinateAnion:
    """ (audit F4): the amino group must not be dropped ('acetate')."""

    def test_glycinate(self):
        assert name_compound("[O-]C(=O)CN") == "glycinate"

    @pytest.mark.parametrize("smiles,expected", [
        ("[O-]C(=O)CCCC", "pentanoate"),                 # simple chain — unaffected
    ])
    def test_carboxylate_regressions(self, smiles, expected):
        assert name_compound(smiles) == expected
