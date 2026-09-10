"""
Tests for hydrazide naming (Blue Book / sulfonohydrazide),
completing gap-fix cluster C1.

Mono-hydrazides (acetohydrazide, pentanehydrazide, cyclohexanecarbohydrazide)
already work; this cluster closes three independent sub-cases that returned
'unknown' before C1:

(a) Acyclic dihydrazide -- two terminal -C(=O)NN
      NNC(=O)CCC(=O)NN -> butanedihydrazide
    Root cause: decomposition/engine.py:_name_quality_is_acceptable multi-amide
    heuristic did not count the 'hydrazide' token, falsely rejecting the complete
    GENERAL name and forcing a wrong bond-cleavage (killed by -> unknown).

(b) Sulfonohydrazide -- R-SO2-NH-NH2 N-analogue of sulfonic acid)
      CS(=O)(=O)NN -> methanesulfonohydrazide
    Root cause: perception gap -- no sulfonohydrazide SMARTS existed.

(c)/(d) Ring-attached (carbo)hydrazide -- aromatic/ saturated ring
      O=C(NN)c1ccncc1 -> pyridine-4-carbohydrazide
      O=C(NN)c1ccc(cc1)C(=O)NN -> benzene-1,4-dicarbohydrazide
    Root cause: aromatic-ring assembly path dropped the hydrazide (oriented_ring
    is None for aromatic rings; benzene/heterocycle routers had no carbohydrazide
    support).

All expected PINs below are OPSIN-CLI-verified (opsin-cli-2.9.0).
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# (a) Acyclic dihydrazides -- previously 'unknown', GENERAL name destroyed by
# the decomposition multi-amide quality gate.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("NNC(=O)CCC(=O)NN", "butanedihydrazide"),
        ("O=C(NN)CC(=O)NN", "propanedihydrazide"),
        ("C(CCCC(=O)NN)(=O)NN", "pentanedihydrazide"),
        ("CC(C(=O)NN)CC(=O)NN", "2-methylbutanedihydrazide"),
    ],
)
def test_acyclic_dihydrazide(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# (b) Sulfonohydrazides -- previously 'unknown' (perception gap, pg=None).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CS(=O)(=O)NN", "methanesulfonohydrazide"),
        ("CCS(=O)(=O)NN", "ethanesulfonohydrazide"),
        ("CC1=CC=C(C=C1)S(=O)(=O)NN", "4-methylbenzenesulfonohydrazide"),
        ("C1(=CC=CC=C1)S(=O)(=O)NN", "benzenesulfonohydrazide"),
    ],
)
def test_sulfonohydrazide(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# (c)/(d) Ring-attached (carbo)hydrazides -- previously 'unknown' or a bogus
# substituent-prefix name (assembly gap in the aromatic-ring path).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("O=C(NN)c1ccncc1", "pyridine-4-carbohydrazide"),
        ("O=C(NN)c1cccnc1", "pyridine-3-carbohydrazide"),
        ("O=C(NN)c1ccc(cc1)C(=O)NN", "benzene-1,4-dicarbohydrazide"),
        ("C1(=CC(=CC=C1)C(=O)NN)C(=O)NN", "benzene-1,3-dicarbohydrazide"),
        ("O1C(=CC=C1)C(=O)NN", "furan-2-carbohydrazide"),
        ("C1(CCC(CC1)C(=O)NN)C(=O)NN", "cyclohexane-1,4-dicarbohydrazide"),
    ],
)
def test_ring_carbohydrazide(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# GUARD cases -- MUST remain correct (must not regress).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # mono-hydrazide (retained/general path -- already working)
        ("CC(=O)NN", "acetohydrazide"),
        ("C1CCCCC1C(=O)NN", "cyclohexanecarbohydrazide"),
        # aromatic carboxamide -- must not be caught by the new hydrazide branch
        ("O=C(N)c1ccc(cc1)C(=O)N", "benzene-1,4-dicarboxamide"),
        # acyclic diamide -- the (a) regex change must not break decomposition here
        ("NC(=O)CCC(=O)N", "butanediamide"),
        # ordinary sulfonamide -- new sulfonohydrazide SMARTS must not over-match
        ("CS(=O)(=O)N", "methanesulfonamide"),
    ],
)
def test_hydrazide_guards(smiles, expected):
    assert name_compound(smiles) == expected
