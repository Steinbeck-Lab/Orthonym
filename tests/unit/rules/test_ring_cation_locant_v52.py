"""v52 a phase, Task 4 —: the skeletal cationic ring centre takes the
LOWEST permissible locant.

**** (heading ** "Cationic characteristic groups on parent
cations"**, ``the Blue Book``; rule ``:42219``):

    "Where there is a choice, low locants for skeletal cationic centers are
    determined before considering locants for cationic suffixes. This is
    consistent with the choice of lowest locants for corresponding neutral
    compounds (see."

The Blue Book's own worked example (``:42288``) is a symmetric di-N ring —
``N,N,N,2-tetramethyl-2,6-naphthyridin-2-ium-5-aminium`` (PIN), NOT the
``…-6-ium-…`` form: the cationic ring N takes the LOWER of the symmetric ``{2,6}``
locant pair. The 1,4-diazabicyclo[2.2.1]heptane / DABCO cations below are the same
class — two equivalent bridgehead N of the neutral parent, the charge breaking the
symmetry, so the cationic N must be numbered 1 heteroatom-set
minimisation still fixes the pair to ``{1,4}``; the choice WITHIN it is the
cation's).

Before the fix the general/von-Baeyer ring-locant supplier (blind to the charge)
handed the cationic N the HIGH locant (``…heptan-4-ium``). The fix re-chooses the
automorphism-equivalent numbering that gives the skeletal cation the lowest locant,
without disturbing the heteroatom-set minimisation.

RED baseline (measured 2026-09-22, fresh process):
  ``C[N+]12CCN(CC1)C2`` -> ``4-methyl-1,4-diazabicyclo[2.2.1]heptan-4-ium``
  ``[NH+]12CCN(CC1)C2`` -> ``1,4-diazabicyclo[2.2.1]heptan-4-ium``
  ``C1C[NH+]2CCN1CC2`` -> ``1,4-diazabicyclo[2.2.2]octan-4-ium``

These rows need ``general_fallback_unverified`` + ``name_tiered`` (as the bb
conformance harness names them), or they will not reproduce.
"""
import pytest

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
             allow_aromatic_general=True)


@pytest.fixture(scope="module")
def namer():
    return Orthonym(**FLAGS)


#: the cationic ring centre gets the LOWEST locant.
LOWEST_CATION_LOCANT = [
    # (smiles, expected_PIN, note)
    # Task-4 target — quaternised (0-H) bridgehead N, methyl co-located: DEMOTE path.
    ("C[N+]12CCN(CC1)C2",
     "1-methyl-1,4-diazabicyclo[2.2.1]heptan-1-ium", "diaza[2.2.1] quaternised"),
    # Bare protonated bridgehead N (has removable H): IN-PLACE path, same class.
    ("[NH+]12CCN(CC1)C2",
     "1,4-diazabicyclo[2.2.1]heptan-1-ium", "diaza[2.2.1] protonated"),
    # DABCO monocation — higher-symmetry [2.2.2] positive, IN-PLACE path.
    ("C1C[NH+]2CCN1CC2",
     "1,4-diazabicyclo[2.2.2]octan-1-ium", "DABCO cation, [2.2.2]"),
]


# Non-regression: the cation is ALREADY at the lowest locant (or the low locant is
# fixed by a senior criterion), so the fix must leave these byte-identical.
NON_REGRESSION = [
    # Single bridgehead N — only one heteroatom, no symmetry choice; the fix must
    # not perturb it (cation naturally at 1)..
    ("C[N+]12CCC(CC1)C2.[Cl-]",
     "1-methyl-1-azabicyclo[2.2.1]heptan-1-ium chloride"),
    # Single-ring aromatic N+ (DEMOTE, single-ring supplier) — untouched.
    ("C[n+]1ccccc1", "1-methylpyridin-1-ium"),
    # Quaternary single-ring N+ — untouched.
    ("C[N+]1(C)CCCCC1", "1,1-dimethylpiperidin-1-ium"),
    # Morpholinium: O is the SENIOR heteroatom (locant 1), so there is NO choice —
    # the cationic N is correctly 4, NOT lowered. outranks.
    ("C[N+]1(C)CCOCC1", "4,4-dimethylmorpholin-4-ium"),
]


# with ANOTHER ring substituent present: the cation must STILL be
# locant 1 (heteroatom set {1,4} is independent of the substituent's position, so
# the choice within it is the cation's). The DEMOTE fragment must be built from the
# TRULY BARE ring skeleton — severing ONLY the centre's bond leaves the substituent
# on the ring, collapses the automorphism group, and re-ships the non-lowest locant
# (the review defect: `2-chloro-4-methyl-...-4-ium`). The exact substituent locant
# falls out of the chosen numbering, so we assert the `-1-ium` property + round-trip
# rather than a hardcoded string.
SUBSTITUTED_STILL_LOWEST = [
    ("C[N+]12CC(Cl)N(CC1)C2", "chloro-substituted diaza[2.2.1] quaternised"),
]


# Review minor #1 — the renumber post-pass now runs for the non-`-ium` suffixes of
# `_emit_ring_cumulative_suffix` too (via the IN-PLACE branch, whenever a heteroatom
# makes `_ring_iupac_locants` return a numbering). Spot-check `-ide`:
# * a bridge-carbon carbanion on the SYMMETRIC diaza[2.2.1] ring IS lowered to a
# low locant by the post-pass (the positive), and
# * a bridgehead carbanion opposite a single ring N keeps locant 4 — the senior
# heteroatom (N=1, has NO choice to give away, so the post-pass
# correctly does NOT force the anion below it (the guard).
# Both were RT-verified (name -> OPSIN 2.9.0 -> input InChI) when added.
NON_IUM_SUFFIX = [
    ("[CH-]1CN2CCN1C2", "1,4-diazabicyclo[2.2.1]heptan-2-ide"),
    ("[C-]12CCN(CC1)C2", "1-azabicyclo[2.2.1]heptan-4-ide"),
]


@pytest.mark.parametrize("smiles,expected,note", LOWEST_CATION_LOCANT)
def test_cation_lowest_locant(namer, smiles, expected, note):
    # (the Blue Book) — skeletal cation gets the lowest locant.
    assert namer.name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,note", SUBSTITUTED_STILL_LOWEST)
def test_substituted_ring_cation_still_lowest(namer, smiles, note):
    # (the Blue Book): cation at locant 1 even with another ring substituent;
    # assert the property + round-trip, not the (numbering-derived) substituent locant.
    name = namer.name_tiered(smiles)["name"]
    assert name is not None and "-1-ium" in name, name
    assert "-4-ium" not in name, name
    rt = opsin_roundtrip_check(smiles, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.parametrize("smiles,expected", NON_IUM_SUFFIX)
def test_non_ium_suffix_renumber(namer, smiles, expected):
    # Review minor #1 — the renumber post-pass behaves for `-ide` (and by the same
    # code, `-yl`), lowering a symmetric anion centre but never overriding a senior
    # heteroatom. RT-verified.
    name = namer.name_tiered(smiles)["name"]
    assert name == expected, name
    rt = opsin_roundtrip_check(smiles, name)
    assert rt["passed"], f"round-trip failed for {name!r}: {rt}"


@pytest.mark.parametrize("smiles,expected", NON_REGRESSION)
def test_lowest_locant_non_regression(namer, smiles, expected):
    assert namer.name_tiered(smiles)["name"] == expected
