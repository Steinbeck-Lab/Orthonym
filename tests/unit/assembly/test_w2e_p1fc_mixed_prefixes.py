"""W2E-P1FC Task 13 — P-35.5.1 (BB 18124): mixed substituent prefixes
(substitutive + additive operations combined, with enclosing-mark escalation).
Examples: (ethoxysulfinyl)amino, (acetylsulfanyl)carbonyl,
[bis(sulfanyl)phosphoryl]amino.

STATUS (2026-07-10, W2E-D2): BUILT. The heteroatom-attached mixed-prefix
N-substituents on aniline now name correctly (get_alkoxysulfinyl_prefix +
get_phosphoryl_prefix in substituent_prefix_forms.py, routed through
benzene._identify_nitrogen_group -> the 'aniline' suffix promotion). Both PINs
OPSIN-round-trip. get_sulfinyl_prefix still handles only C-linked -S(=O)-alkyl;
the new O-linked alkoxysulfinyl assembler is its complement.

OPSIN-verified target PINs:
  CCOS(=O)Nc1ccccc1  -> N-(ethoxysulfinyl)aniline
  SP(=O)(S)Nc1ccccc1 -> N-[bis(sulfanyl)phosphoryl]aniline
"""
import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestP3551MixedPrefixes:
    def test_ethoxysulfinyl_target(self):
        assert name_compound("CCOS(=O)Nc1ccccc1") == "N-(ethoxysulfinyl)aniline"

    def test_bis_sulfanyl_phosphoryl_target(self):
        assert name_compound("SP(=O)(S)Nc1ccccc1") == \
            "N-[bis(sulfanyl)phosphoryl]aniline"
