"""An N-substituted amine on a fused parent takes the ``-amine`` SUFFIX (P-41).

`rules/fused_rings.py`'s substituent collector routed only the BARE ``amino`` to
the amine suffix. Every N-substituted amine fell through to the generic
``functional`` branch and was cited as a PREFIX on the parent hydride, so the
senior characteristic group got no suffix at all::

    CNc1ccc2ccccc2n1   ->  2-(methylamino)quinoline      (no suffix; non-PIN)
                       =>  N-methylquinolin-2-amine

WHY THE PREFIX FORM IS NOT AVAILABLE HERE
-----------------------------------------
``anilino`` and ``alkylamino`` ARE preferred prefixes -- ``BlueBookV2.md:6371``
prints ``4-[(4-hydroxyanilino)methyl]phenol (PIN)``. But that parent bears a
PHENOL, which outranks the amine, so the amine is correctly demoted. With no
senior characteristic group present the amine IS the principal one and P-41
requires it as the suffix. The shape is ``:21610``'s
``4-methoxy-*N*-phenylaniline (PIN)``.

The monocyclic paths already did this (``N-methylaniline``,
``N-methylpyridin-2-amine``); only the fused-ring collector diverged, which is
why the defect was invisible outside fused parents.

DETECTION IS STRUCTURAL
-----------------------
``_exocyclic_amine_n_substituents`` reads the graph, never the prefix spelling --
the same lesson this module records at its anilino site, where naming a ring by
its carbon COUNT turned a quinolinyl-amine into ``heptylamino``. It declines
unless the nitrogen is a plain, acyclic, three-valent, all-single-bonded amine N
whose branches are carbon and non-acyl, so an amide/imine/nitro/N-oxide nitrogen
can never be mistaken for one.
"""

import pytest

from orthonym import name_compound


@pytest.mark.parametrize("smiles,expected", [
    # the reported defect
    ("CNc1ccc2ccccc2n1", "N-methylquinolin-2-amine"),
    # primary amine on the same parent must be unchanged
    ("Nc1ccc2ccccc2n1", "quinolin-2-amine"),
])
def test_fused_heteroaryl_amine_takes_the_suffix(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # the monocyclic controls -- these already worked and must not move
    ("CNc1ccccn1", "N-methylpyridin-2-amine"),
    ("CNc1ccccc1", "N-methylaniline"),
    ("c1ccncc1Nc1ccccc1", "N-phenylpyridin-3-amine"),
])
def test_monocyclic_controls_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # The CARBOCYCLIC half, a separate producer (rules/polycyclics.py). The
    # anilino row shipped a wrong name; the rest abstained outright.
    ("c1ccc2c(c1)cccc2Nc1ccccc1", "N-phenylnaphthalen-1-amine"),
    ("CNc1cccc2ccccc12", "N-methylnaphthalen-1-amine"),
    ("CCNc1cccc2ccccc12", "N-ethylnaphthalen-1-amine"),
    ("CN(C)c1cccc2ccccc12", "N,N-dimethylnaphthalen-1-amine"),
    # primary amine on the same parent unchanged
    ("Nc1cccc2ccccc12", "naphthalen-1-amine"),
])
def test_pah_amine_takes_the_suffix(smiles, expected):
    assert name_compound(smiles) == expected


def test_a_senior_group_keeps_the_amine_a_prefix():
    """``anilino`` IS a preferred prefix where a SENIOR characteristic group
    holds the suffix -- ``:6371``'s example sits on a phenol. With a carboxylic
    acid present the amine must NOT be promoted, and the `not suffix_groups`
    guard is what enforces it. This is the control that makes the promotion
    above safe rather than indiscriminate."""
    assert name_compound("OC(=O)c1ccc2ccccc2c1Nc1ccccc1") == \
        "1-anilinonaphthalene-2-carboxylic acid"


class TestTheGuardsHold:
    """The detector must decline anything that is not a plain amine nitrogen."""

    def test_amide_nitrogen_is_not_an_amine(self):
        """An acyl branch makes this an AMIDE (P-66.1), which is senior to and
        different from an amine; naming it `N-...-amine` would be wrong. The
        emitted name must therefore NOT be an amine suffix on the ring."""
        name = name_compound("CC(=O)Nc1ccc2ccccc2n1")
        assert name is None or "amine" not in name, name

    def test_ring_nitrogen_keeps_its_own_path(self):
        """A RING nitrogen bearing a substituent is not an exocyclic amine; the
        detector declines on `IsInRing`, so N-methylindole is untouched."""
        assert name_compound("Cn1ccc2ccccc21") == "1-methyl-1H-indole"
