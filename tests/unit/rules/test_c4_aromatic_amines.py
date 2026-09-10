"""C4 — aromatic / diaryl amines named substitutively (completes task 1.6,.

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
    """4,4'-azanediyldibenzoic acid: fragments carry a senior PCG (CO2H) so the
    _all_fragments_are_simple_carbocycles guard is False -> the -NH- (azanediyl)
    multiplicative bridge is retained (NOT reclassified to an aniline).
    Wave2: the divalent -NH- bridge PIN is 'azanediyl' (was 'imino')."""
    assert _pin("OC(=O)c1ccc(Nc2ccc(C(=O)O)cc2)cc1") == "4,4'-azanediyldibenzoic acid"


def test_c4_guard_amine_prefix_when_senior_group_present():
    """anilino / (N-alkylamino) must stay a PREFIX when a senior group (COOH)
    is on the ring — the amine is only promoted to the aniline suffix when it
    is the molecule-level principal group. OPSIN-verified fallback PIN."""
    assert _pin("CNc1ccc(C(=O)O)cc1") == "4-(N-methylamino)benzoic acid"


# ---------------------------------------------------------------------------
# C4b regression fix — N-substituted aromatic DIAMINES
#
# A benzene ring bearing a free primary -NH2 AND a second, N-substituted
# amino group. Both amino N's are the principal group -> 'benzene-x,y-diamine'
# parent; the N-substituent(s) on one nitrogen are cited as italic-N locant
# prefixes on that nitrogen (which takes ring position 1 so it reads plain
# N-/N,N-). C4 promoted BOTH amines to the diamine suffix but DROPPED the
# N-substituent -> emitted 'benzene-1,4-diamine' (wrong structure) which
# then suppressed to 'unknown'. All PINs OPSIN round-trip verified.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("Nc1ccc(Nc2ccccc2)cc1", "N-phenylbenzene-1,4-diamine"),
    ("Nc1ccccc1Nc1ccccc1", "N-phenylbenzene-1,2-diamine"),
    ("CNc1ccc(N)cc1", "N-methylbenzene-1,4-diamine"),
    ("Nc1ccc(NC)cc1", "N-methylbenzene-1,4-diamine"),
    ("Nc1ccc(N(C)C)cc1", "N,N-dimethylbenzene-1,4-diamine"),
    ("CNc1ccccc1N", "N-methylbenzene-1,2-diamine"),
])
def test_c4b_n_substituted_aromatic_diamine(smiles, expected):
    assert _pin(smiles) == expected


def test_c4b_pure_primary_diamine_unchanged():
    """The pure primary diamine (both -NH2, no N-substituent) must stay a valid
    diamine name and NOT regress to unknown. Its exact spelling is env-dependent
    (prod path yields '1,4-phenylenediamine'; the RT-free unit path
    yields 'benzene-1,4-diamine') — both are OPSIN-valid PINs/synonyms for the
    same molecule; the C4b fix must not touch it."""
    assert _pin("Nc1ccc(N)cc1") in {"benzene-1,4-diamine", "1,4-phenylenediamine"}


# ---------------------------------------------------------------------------
# — a COMPOUND N-substituent on the aniline parent takes its own
# enclosing marks before the italic 'N-' locant (the Blue Book,
# 4-(2-methylbutyl)-N-(3-methylbutyl)aniline). _name_substituted_aniline used
# _wrap_n_substituent alone (escalate-only), so a mark-free compound prefix
# emitted bare: 'N-3-methylbutylaniline'. The fix routes it through
# enclose_if_compound first (the same base-enclosure step the polyfunctional
# N-substituent path already applies). Simple prefixes stay bare.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # the Blue Book worked example.
    ("CCC(C)Cc1ccc(NCCC(C)C)cc1",
     "4-(2-methylbutyl)-N-(3-methylbutyl)aniline"),
    # A compound N-substituent with no ring locant still needs its marks.
    ("CC(C)CNc1ccccc1", "N-(2-methylpropyl)aniline"),
])
def test_compound_n_substituent_encloses(smiles, expected):
    assert _pin(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # Invariant 9: a SIMPLE N-substituent stays bare (no enclosure needed).
    ("CNc1ccccc1", "N-methylaniline"),
    ("CN(C)c1ccccc1", "N,N-dimethylaniline"),
    ("c1ccc(Nc2ccccc2)cc1", "N-phenylaniline"),
    ("CCNc1ccc(Cl)cc1", "4-chloro-N-ethylaniline"),
])
def test_simple_n_substituent_stays_bare(smiles, expected):
    assert _pin(smiles) == expected
