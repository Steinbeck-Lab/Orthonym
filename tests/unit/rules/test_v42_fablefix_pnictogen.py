""" a review — two spelling-layer defects a cross-model (a review) review found.

BLOCKER 1 — the multiplicative pnictogen-oxoacid guard was too narrow (it declined
only NAKED diaryl rings). A SUBSTITUTED diaryl phosphinic/arsinic acid, and a
diaryl phosphinate/arsinate ESTER, still shipped the wrong multiplicative name
(``1,1'-(hydroxyphosphoryl)bis(4-methylbenzene)`` etc.). The guard is now a
SENIORITY comparison, the Blue Bookff / the Blue Bookff): decline the
multiplicative name → acid/ester parent when NO ring fragment carries a group
SENIOR to the pnictogen oxoacid (class 7c). A carboxylic-acid ring (class 7a) is
senior, so ``4,4'-(hydroxyarsoryl)dibenzoic acid`` STAYS multiplicative.

BLOCKER 2 — ``phosphinolizine`` shipped bare but is the 6+6 QUINOLIZINE analogue,
so it carries indicated hydrogen exactly like its N parent (the Blue Book "the PIN is
4H-quinolizine"; the Blue Book + the Blue Book require indicated H to
be cited). The 5+6 indolizine-type analogues (arsindolizine / phosphindolizine)
are fully mancude and STAY bare.
"""
from orthonym import name_compound


class TestBlocker1PnictogenAcidEster:
    """P-41 Table 4.1 (BB:18143): a phosphinic/arsinic acid (class 7c) or its ester
    (class 9) is senior to a ring's hydroxy/amine/halogen/alkyl, so the acid/ester
    is the parent and the rings are ``bis(aryl)`` prefixes — not a multiplicative
    ``(…oryl)`` bridge name."""

    def test_bis_4_methylphenyl_phosphinic_acid(self):
        assert (name_compound("Cc1ccc(cc1)P(=O)(O)c1ccc(C)cc1")
                == "bis(4-methylphenyl)phosphinic acid")

    def test_bis_4_hydroxyphenyl_phosphinic_acid(self):
        # hydroxy (class 17) is JUNIOR to the acid -> a detachable prefix
        assert (name_compound("Oc1ccc(cc1)P(=O)(O)c1ccc(O)cc1")
                == "bis(4-hydroxyphenyl)phosphinic acid")

    def test_bis_4_aminophenyl_phosphinic_acid(self):
        # amine (class 19) is JUNIOR to the acid -> a detachable prefix
        assert (name_compound("Nc1ccc(cc1)P(=O)(O)c1ccc(N)cc1")
                == "bis(4-aminophenyl)phosphinic acid")

    def test_bis_4_chlorophenyl_phosphinic_acid(self):
        assert (name_compound("Clc1ccc(cc1)P(=O)(O)c1ccc(Cl)cc1")
                == "bis(4-chlorophenyl)phosphinic acid")

    def test_methyl_diphenylphosphinate_ester(self):
        # class 9 ester; P had no phosphinate-ester PERCEPTION before this fix
        assert (name_compound("c1ccccc1P(=O)(OC)c1ccccc1")
                == "methyl diphenylphosphinate")

    def test_methyl_diphenylarsinate_ester(self):
        # As had NO ester namer at all; the element-generic -inate namer builds it
        assert (name_compound("c1ccccc1[As](=O)(OC)c1ccccc1")
                == "methyl diphenylarsinate")


class TestBlocker1RegressionPins:
    """The boundary and the already-correct naked/di-alkyl cases must be unchanged."""

    def test_dibenzoic_acid_STAYS_multiplicative(self):
        # each ring carries a carboxylic acid (class 7a), SENIOR to the arsinic-acid
        # bridge (7c) -> the ring is the parent, multiplicative is kept (the Blue Book)
        assert (name_compound("OC(=O)c1ccc(cc1)[As](=O)(O)c1ccc(cc1)C(=O)O")
                == "4,4'-(hydroxyarsoryl)dibenzoic acid")

    def test_naked_diphenylarsinic_acid_unchanged(self):
        assert name_compound("c1ccccc1[As](=O)(O)c1ccccc1") == "diphenylarsinic acid"

    def test_naked_diphenylphosphinic_acid_unchanged(self):
        assert name_compound("c1ccccc1P(=O)(O)c1ccccc1") == "diphenylphosphinic acid"

    def test_dicyclohexylarsinic_acid_unchanged(self):
        assert (name_compound("C1CCCCC1[As](=O)(O)C1CCCCC1")
                == "dicyclohexylarsinic acid")

    def test_dimethylarsinic_acid_unchanged(self):
        assert name_compound("C[As](=O)(O)C") == "dimethylarsinic acid"

    def test_methyl_phenyl_arsinic_acid_unchanged(self):
        assert name_compound("C[As](=O)(O)c1ccccc1") == "methyl(phenyl)arsinic acid"

    def test_phenylarsonic_acid_unchanged(self):
        assert name_compound("c1ccccc1[As](=O)(O)O") == "phenylarsonic acid"

    def test_non_pnictogen_multiplicative_unchanged(self):
        # a genuine methylene-bridged dianiline must keep its multiplicative PIN
        assert (name_compound("Nc1ccc(cc1)Cc1ccc(N)cc1")
                == "4,4'-methylenedianiline")


class TestBlocker2Phosphinolizine:
    """ (the Blue Book) + (the Blue Book): indicated hydrogen is cited.
    phosphinolizine is the 6+6 quinolizine analogue (the Blue Book parallel)."""

    def test_phosphinolizine_carries_indicated_hydrogen(self):
        assert name_compound("C1=CCP2C=CC=CC2=C1") == "4H-phosphinolizine"

    def test_quinolizine_parent_parallel(self):
        # the isostructural N parent — the model for the 4H- placement
        assert name_compound("C1=CCN2C=CC=CC2=C1") == "4H-quinolizine"

    def test_arsindolizine_stays_bare(self):
        # 5+6 indolizine-type, fully mancude -> NO indicated hydrogen
        assert name_compound("C1=CC2=CC=C[As]2C=C1") == "arsindolizine"

    def test_phosphindolizine_stays_bare(self):
        # 5+6 indolizine-type, fully mancude -> NO indicated hydrogen
        assert name_compound("c1ccp2cccc2c1") == "phosphindolizine"
