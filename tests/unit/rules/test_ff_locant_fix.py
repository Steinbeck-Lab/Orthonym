"""
Task FF-FIX (v51) — feed ring suffix atoms into numbering.

A a review cross-family review found BLOCKER-class LOCANT defects that the gate, a
Sonnet review and every fuzzer missed: they are invisible to round-trip (the
right molecule, the wrong locant). Two of them share ONE root cause.

ROOT CAUSE: ``principal_group_atoms=`` at numbering time. The ring atoms that
carry the principal characteristic group as a SUFFIX (the pseudoketone
``-one``/``-dione`` ring carbons; the ring nitrogen bearing an exocyclic
sulfur-oxoacid suffix) were never handed to
``orient_heterocycle_with_substituents`` as principal-group atoms, so the
(f) detachable-prefix tie-break decided the numbering and put the
SUBSTITUTED ring atom at locant 1. The carboxylic-acid path DOES feed the
suffix nitrogen (its ``principal_group_atoms`` is ``{N}``), which is why
``OC(=O)N1CCN(C)CC1`` -> ``4-methylpiperazine-1-carboxylic acid`` is correct.
The fix feeds the pseudoketone ring carbons and the ring-N sulfur-suffix atom the
same way, so (c) governs.

  * NUMBERING (the Blue Book): "low locants are assigned to them in
    the following decreasing order of seniority":... (c) "principal
    characteristic groups and free valences (suffixes)";... (f) "detachable
    alphabetized prefixes". Criterion (c) — the ``-2,4-dione`` / ``-1-sulfonic
    acid`` suffix set — outranks (f), the ``methyl`` prefix. For a heterocycle
    the ring heteroatoms are numbered first (change note (1), the Blue Book), then (c),
    then (f); the heteroatom set ties in both directions here, so the suffix
    decides.
"""

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# — N-substituted cyclic imide / diamide: the -dione suffix takes the
# lowest locant set ({2,4} < {2,5}) BEFORE the N-substituent (c) > (f)).
FF1_HYDANTOIN = [
    # The regressed case: the substituted N is flanked by BOTH carbonyls, so it
    # must be locant 3 (was 1-methyl...-2,5-dione).
    ("CN1C(=O)CNC1=O", "3-methylimidazolidine-2,4-dione"),
    ("CCN1C(=O)CNC1=O", "3-ethylimidazolidine-2,4-dione"),
    ("CN1C(=O)CCNC1=O", "3-methyl-1,3-diazinane-2,4-dione"),
    # The MIRROR must STAY correct (R1-FIX fixed it): the substituted N is
    # flanked by a CH2 and one carbonyl, so it is locant 1.
    ("CN1CC(=O)NC1=O", "1-methylimidazolidine-2,4-dione"),
    ("O=C1NCCN1c1ccccc1", "1-phenylimidazolidin-2-one"),
]

# — ring-N sulfur-oxoacid suffix takes the lowest locant BEFORE the
# N-substituent (was 1-methyl...-4-sulfonic acid).
FF2_RING_N_SULFUR = [
    ("OS(=O)(=O)N1CCN(C)CC1", "4-methylpiperazine-1-sulfonic acid"),
    ("OS(=O)N1CCN(C)CC1", "4-methylpiperazine-1-sulfinic acid"),
    ("NS(=O)(=O)N1CCN(C)CC1", "4-methylpiperazine-1-sulfonamide"),
    ("[O-]S(=O)(=O)N1CCN(C)CC1", "4-methylpiperazine-1-sulfonate"),
    ("OS(=O)(=O)N1CCN(CC1)c1ccccc1", "4-phenylpiperazine-1-sulfonic acid"),
    ("OS(=O)(=O)N1CCN(C)C1", "3-methylimidazolidine-1-sulfonic acid"),
]

# CONTROLS that must stay correct.
CONTROLS = [
    # Carboxylic acid on the same ring — the path that already fed the suffix N.
    ("OC(=O)N1CCN(C)CC1", "4-methylpiperazine-1-carboxylic acid"),
    # Symmetric ring — already right; must not flip.
    ("OS(=O)(=O)N1CCNCC1", "piperazine-1-sulfonic acid"),
    ("O=C1NCCN1c1ccccc1", "1-phenylimidazolidin-2-one"),
]

# v51 wins from R1/R2/R3/R1-FIX/BORON that must not regress.
V51_WINS = [
    ("OC(=O)N1CCCC1", "pyrrolidine-1-carboxylic acid"),
    ("OC(=O)N1CCCCC1", "piperidine-1-carboxylic acid"),
    ("OS(=O)N1CCCCC1", "piperidine-1-sulfinic acid"),
    ("CN1CCN(C1=O)c1ccccc1", "1-methyl-3-phenylimidazolidin-2-one"),
    ("OC(=O)N1CCNCC1", "piperazine-1-carboxylic acid"),
    ("O=C1NCCN1", "imidazolidin-2-one"),
    ("O=C1CSCN1", "1,3-thiazolidin-4-one"),
    ("B1Oc2ccccc2O1", "2H-1,3,2-benzodioxaborole"),
]


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", FF1_HYDANTOIN)
def test_ff1_hydantoin_dione_lowest_locant(smiles, expected):
    """The ring -dione suffix takes the lowest locant set (c) > (f))."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", FF2_RING_N_SULFUR)
def test_ff2_ring_n_sulfur_suffix_lowest_locant(smiles, expected):
    """The ring-N sulfur-oxoacid suffix takes the lowest locant (c))."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", CONTROLS)
def test_ff_controls_preserved(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", V51_WINS)
def test_v51_wins_preserved(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smiles,expected",
    FF1_HYDANTOIN + FF2_RING_N_SULFUR + CONTROLS,
)
def test_ff_round_trip(smiles, expected):
    """Each emitted name OPSIN-parses back to the input structure (0-wrong)."""
    assert name_compound(smiles) == expected
    result = opsin_roundtrip_check(smiles, expected)
    assert result["passed"], (
        f"OPSIN round-trip failed for {expected!r}: {result.get('error')}"
    )
