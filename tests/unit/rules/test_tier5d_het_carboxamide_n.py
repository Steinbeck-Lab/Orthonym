"""
Wave2 Tier 5d(A) — N-substituted heterocycle-carboxamides.

Reproduce-first: the benzene path already handles N-substituted amides
(N,N-dimethylbenzamide works), but the heterocycle suffix detector
(`heterocycles._identify_suffix_fg`) matched only PRIMARY amides
([CX3](=O)[NX3H2]) — every N-substituted heterocycle-carboxamide fell to
'unknown'. Fix mirrors the benzene analog: secondary/tertiary amide SMARTS +
N-substituent extraction (shared `benzene._detect_n_substituents`), threaded
through the existing amine-suffix N-prefix builder (C4 mechanism).
Fail-closed: if any heavy N-neighbour fails to produce a name, the whole
suffix record declines (missing-beats-wrong).
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CNC(=O)c1ccco1", "N-methylfuran-2-carboxamide"),
        ("CN(C)C(=O)c1ccco1", "N,N-dimethylfuran-2-carboxamide"),
        ("CCN(CC)C(=O)c1ccco1", "N,N-diethylfuran-2-carboxamide"),
        ("CCN(CC)C(=O)c1cccnc1", "N,N-diethylpyridine-3-carboxamide"),
        ("CNC(=O)c1ccncc1", "N-methylpyridine-4-carboxamide"),
    ],
)
def test_tier5d_n_substituted_het_carboxamides(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # primary amides and the benzene N-substituted path must not move
        ("NC(=O)c1ccco1", "furan-2-carboxamide"),
        ("CN(C)C(=O)c1ccccc1", "N,N-dimethylbenzamide"),
        #: was "nicotinamide". This is NOT an absence argument -- both
        # Blue Book occurrences of the trivial name are NEGATIVE:
        #:32895 "2-chloropyridine-3-carboxamide (PIN) (not 2-chloronicotinamide)"
        #:33170 "pyridine-4-carbothioamide (PIN) (not thioisonicotinamide)"
        # The parent acid is retained for GENERAL nomenclature only --
        # "The following names are retained for general nomenclature with
        # functionalization but no substitution is allowed", listing "nicotinic acid
        # / pyridine-3-carboxylic acid (PIN)". The amide is built from the PIN acid
        # stem under. This also makes the three controls uniformly
        # systematic, matching the furan-2-carboxamide row above.
        ("NC(=O)c1cccnc1", "pyridine-3-carboxamide"),
    ],
)
def test_tier5d_regression_controls(smiles, expected):
    assert name_compound(smiles) == expected
