""" chalcogen-chain SUBSTITUENT maximisation (v52 N8).

§**** (the Blue Book:39325/:39327): *"Compounds with three or more
contiguous identical chalcogen atoms are treated as parent hydrides in
substitutive nomenclature"* — so a run of three-or-more contiguous divalent
sulfurs attached to a senior parent is ONE substituent chain
``<multiplier>sulfanyl`` (``trisulfanyl`` / ``tetrasulfanyl`` / …), NOT split into
a nested ``sulfanyl`` on a shorter ``disulfanyl`` chain.

Before the fix ``_name_disulfanyl_branch`` hard-coded a 2-sulfur ``disulfanyl`` and
treated everything beyond as a nested arm, so ``-S-S-S-CH2CH3`` was named
``(ethylsulfanyl)disulfanyl`` (the RIGHT molecule — both spellings round-trip
through OPSIN 2.9.0 to the identical InChIKey — but a non-PIN spelling). The fix
walks the maximal contiguous S chain. A 2-sulfur ``disulfanyl`` is unchanged
(1) /.
"""
from orthonym import name_compound


class TestChainMaximisation:
    def test_three_sulfur_substituent_is_trisulfanyl(self):
        # Was 3-[(ethylsulfanyl)disulfanyl]propanoic acid (split, non-PIN).
        assert name_compound("OC(=O)CCSSSCC") == \
            "3-(ethyltrisulfanyl)propanoic acid"

    def test_four_sulfur_substituent_is_tetrasulfanyl(self):
        assert name_compound("OC(=O)CCSSSSCC") == \
            "3-(ethyltetrasulfanyl)propanoic acid"

    def test_terminal_three_sulfur_on_arene_is_trisulfanyl(self):
        # -S-S-SH terminal run: bare trisulfanyl (no R). Was
        # (sulfanyldisulfanyl)benzene before the fix.
        assert name_compound("c1ccc(SSS)cc1") == "trisulfanylbenzene"

    def test_isotope_trisulfanyl_row(self):
        # The molecule that surfaced the defect isotope decoration over
        # the skeleton). Both spellings round-trip to the same
        # InChIKey; trisulfanyl is the PIN spelling.
        assert name_compound("CCSS[34S]CCC(=[18O])O") == \
            "3-[ethyl(1-34S)trisulfanyl](18O)propanoic acid"


class TestNoLeakTwoSulfur:
    """The 2-sulfur disulfanyl substituent must be UNCHANGED (1))."""

    def test_two_sulfur_substituent_stays_disulfanyl(self):
        assert name_compound("OC(=O)CCSSCC") == \
            "3-(ethyldisulfanyl)propanoic acid"

    def test_bare_disulfanyl_arene_unchanged(self):
        # -S-SH terminal: bare disulfanyl (BB:17954).
        assert name_compound("c1ccc(SS)cc1") == "disulfanylbenzene"

    def test_bis_disulfanyl_benzamide_unchanged(self):
        # BB:28021 verbatim PIN — two 2-sulfur runs, bis(disulfanyl).
        assert name_compound("NC(=O)c1ccc(SS)c(SS)c1") == \
            "3,4-bis(disulfanyl)benzamide"
