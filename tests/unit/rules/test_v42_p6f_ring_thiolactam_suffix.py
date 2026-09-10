""" a phase — SATURATED ring chalcogen ketone-analogue SUFFIX (P-64.6.1).

## **P-64.6** CHALCOGEN ANALOGUES... P-64.6.1 (the Blue Book): a C=S / C=Se / C=Te on
a ring (or chain) carbon is the ketone analogue expressed with the SUFFIX
``-thione`` / ``-selone`` / ``-tellone`` (multiplied -> ``-dithione``...), NOT the
``sulfanylidene`` / ``selanylidene`` / ``tellanylidene`` substitutive PREFIX.

Seniority of the doubly-bonded chalcogen for choosing the principal characteristic
group is **C=O > C=S > C=Se > C=Te** (the Blue Book). So a ring that ALSO bears a senior
ring C=O keeps the C=O as the ``-one`` suffix and the C=S stays a ``sulfanylidene``
prefix -- that is correct and is preserved here (a project rule).

These are the SATURATED ring thiolactams (cyclic thioamides). They are the
chalcogen analogue of the lactam the lactam handler declines, and there is no
thiolactam handler, so they are named by the get_heterocycle_substituents
ring_thioamide_suffix branch (rules/heterocycles.py). The MANCUDE / added-indicated-H
ring thiones (``pyridine-2(1H)-thione``) go through name_cyclic_oxo_compound and are
covered by test_v27_p4_thione.py; both engines are exercised for the O>S seniority
guard so the two stay consistent.
"""

import pytest

from orthonym.namer import name_compound


@pytest.mark.unit
class TestRingThiolactamSuffix:
    """P-64.6.1: a saturated ring carbon's C=S/C=Se/C=Te becomes the
    -thione/-selone/-tellone SUFFIX (di- -> -dithione), not a *ylidene prefix."""

    def test_azepane_2_thione(self):
        # S=C1CCCCCN1 -- single-N monocyclic thiolactam. Was the non-PIN
        # 2-sulfanylideneazepane; PIN is the -thione suffix (P-64.6.1). NB the
        # parent 'e' is kept (thione begins with a consonant), unlike azepan-2-one.
        assert name_compound("S=C1CCCCCN1") == "azepane-2-thione"

    def test_thiazolidine_2_4_dithione(self):
        # S=C1CSC(=S)N1 -- two ring C=S, no ring C=O; both become the suffix,
        # multiplied to -dithione. Was 2,4-disulfanylidene-1,3-thiazolidine.
        assert name_compound("S=C1CSC(=S)N1") == "1,3-thiazolidine-2,4-dithione"

    def test_azepane_2_selone(self):
        # whole-class: the C=Se selenolactam analogue -> -selone suffix.
        assert name_compound("[Se]=C1CCCCCN1") == "azepane-2-selone"

    def test_azepane_2_tellone(self):
        # whole-class: the C=Te tellurolactam analogue -> -tellone suffix.
        assert name_compound("[Te]=C1CCCCCN1") == "azepane-2-tellone"

    def test_n_methyl_azepane_2_thione(self):
        # an N-substituent coexists with the -thione suffix (the branch does not
        # steal the N-substituent): CN1CCCCCC1=S -> 1-methylazepane-2-thione.
        assert name_compound("CN1CCCCCC1=S") == "1-methylazepane-2-thione"


@pytest.mark.unit
class TestOxoSeniorToThione:
    """Seniority C=O > C=S (the Blue Book): when a ring bears BOTH, the C=O wins the
    -one/-dione suffix and the C=S stays a sulfanylidene PREFIX (a project rule)."""

    def test_rhodanine_keeps_sulfanylidene_prefix(self):
        # O=C1CSC(=S)N1 (rhodanine-shaped): the senior C=O is the -4-one suffix,
        # the C=S is the 2-sulfanylidene prefix. UNCHANGED by the P-6F branch --
        # the principal group is 'secondary_amide' (not a *amide chalcogen class),
        # so ring_thioamide_suffix never fires here (the Blue Book).
        assert (name_compound("O=C1CSC(=S)N1")
                == "2-sulfanylidene-1,3-thiazolidin-4-one")

    def test_thiobarbituric_dione_keeps_sulfanylidene(self):
        # S=C1NC(=O)CC(=O)N1: two senior ring C=O -> -4,6-dione, the C=S stays
        # a 2-sulfanylidene prefix.
        assert (name_compound("S=C1NC(=O)CC(=O)N1")
                == "2-sulfanylidene-1,3-diazinane-4,6-dione")


@pytest.mark.unit
class TestP6FRegressionGuards:
    """Nothing outside the saturated ring thiolactam class is touched."""

    def test_plain_lactam_unchanged(self):
        # ordinary C=O lactam still -> azepan-2-one (lactam handler, not P-6F).
        assert name_compound("O=C1CCCCCN1") == "azepan-2-one"

    def test_exocyclic_carbothioamide_not_a_thione(self):
        # an EXOCYCLIC C(=S)N on a ring carbon is a carbothioamide, NOT a ring
        # thione: the C=S carbon is not a ring atom, so the branch (no-carbon
        # ring substituent only) never fires.
        assert name_compound("C1CCCCC1C(N)=S") == "cyclohexanecarbothioamide"

    def test_mancude_ring_thione_unchanged(self):
        # the unsaturated added-IH ring thione stays on the other engine.
        assert name_compound("S=c1cccc[nH]1") == "pyridine-2(1H)-thione"
