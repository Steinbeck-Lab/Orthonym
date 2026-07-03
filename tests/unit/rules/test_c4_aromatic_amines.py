"""C4 — aromatic / diaryl amines named substitutively (completes task 1.6, P-62.2.2).

The aliphatic secondary/tertiary amine case (CCCNCCC -> N-propylpropan-1-amine)
was already substitutive. This cluster fixes the AROMATIC subfamily: an amine N
that is the molecule-level principal characteristic group on a benzene / azine
ring must reclassify to the aniline (retained) or ring-amine '-amine' SUFFIX
parent with N-substituents cited as italic-N prefixes — INSTEAD of the
multiplicative "iminodibenzene" bridge or the substituent-on-ring
"(N-ethylamino)benzene" prefix path.

Root causes (per gap-spec-C4-amines.json):
  - benzene.py had no 'amine' entry in _SUFFIX_PRIORITY and no
    hydroxy->ol-style reclassification for nitrogen.
  - heterocycles.py:_identify_hetero_substituent only recognized bare -NH2
    (h_count==2), so N-substituted ring amines were dropped.
  - multiplicative.py:_try_single_atom_bridges lacked the
    _all_fragments_are_simple_carbocycles decline that its 3-unit sibling
    already applies for triphenylamine.

All expected PINs are OPSIN round-trip verified.
"""
import pytest
from orthonym.namer import name_compound


def _pin(smiles: str) -> str:
    return name_compound(smiles, style="pin")


# ---------------------------------------------------------------------------
# Acceptance — the 20 spec cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # diaryl amine — multiplicative decline -> substitutive aniline
    ("c1ccccc1Nc1ccccc1", "N-phenylaniline"),
    # N-alkyl anilines
    ("CCNc1ccccc1", "N-ethylaniline"),
    ("CNc1ccccc1", "N-methylaniline"),
    # N,N-dialkyl anilines
    ("CCN(CC)c1ccccc1", "N,N-diethylaniline"),
    ("CN(C)c1ccccc1", "N,N-dimethylaniline"),
    # ring-substituted primary anilines (bare -NH2 promoted to aniline suffix)
    ("Cc1ccc(N)cc1", "4-methylaniline"),
    ("Clc1ccc(N)cc1", "4-chloroaniline"),
    ("Nc1ccccc1Cl", "2-chloroaniline"),
    # ring-substituted + N-substituted anilines (mixed prefixes)
    ("Fc1ccc(NC)cc1", "4-fluoro-N-methylaniline"),
    ("CN(C)c1ccc(Cl)cc1", "4-chloro-N,N-dimethylaniline"),
    # N-substituted ring (azine) amines
    ("c1ccccc1Nc1ccncc1", "N-phenylpyridin-4-amine"),
    ("c1ccccc1Nc1ccccn1", "N-phenylpyridin-2-amine"),
    ("CNc1ccncc1", "N-methylpyridin-4-amine"),
])
def test_c4_aromatic_amine_acceptance(smiles, expected):
    assert _pin(smiles) == expected


# ---------------------------------------------------------------------------
# GUARDS — primary amines and amine-not-senior cases (must stay correct)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # bare aniline / bare ring amine unchanged
    ("Nc1ccccc1", "aniline"),
    ("Nc1ccncc1", "pyridin-4-amine"),
    # amine NOT senior — a senior group keeps amine as the 'amino' prefix
    ("Nc1ccc(O)cc1", "4-aminophenol"),
    ("Nc1ccc(C(=O)O)cc1", "4-aminobenzoic acid"),
    # aliphatic secondary amine (task 1.6 base case) untouched
    ("CCCNCCC", "N-propylpropan-1-amine"),
])
def test_c4_amine_guards(smiles, expected):
    assert _pin(smiles) == expected


# ---------------------------------------------------------------------------
# LOAD-BEARING multiplicative / prefix guards — MUST NOT regress
# ---------------------------------------------------------------------------

def test_c4_guard_methylenedibenzene_stays_multiplicative():
    """The CH2 (methylene) single-atom bridge must NOT be touched by the
    N-scoped multiplicative decline: 1,1'-methylenedibenzene stays."""
    assert _pin("c1ccccc1Cc1ccccc1") == "1,1'-methylenedibenzene"


def test_c4_guard_iminodibenzoic_acid_stays_multiplicative():
    """4,4'-iminodibenzoic acid: fragments carry a senior PCG (CO2H) so the
    _all_fragments_are_simple_carbocycles guard is False -> the imino
    multiplicative bridge is retained (NOT reclassified to an aniline)."""
    assert _pin("OC(=O)c1ccc(Nc2ccc(C(=O)O)cc2)cc1") == "4,4'-iminodibenzoic acid"


def test_c4_guard_amine_prefix_when_senior_group_present():
    """anilino / (N-alkylamino) must stay a PREFIX when a senior group (COOH)
    is on the ring — the amine is only promoted to the aniline suffix when it
    is the molecule-level principal group. OPSIN-verified fallback PIN."""
    assert _pin("CNc1ccc(C(=O)O)cc1") == "4-(N-methylamino)benzoic acid"
