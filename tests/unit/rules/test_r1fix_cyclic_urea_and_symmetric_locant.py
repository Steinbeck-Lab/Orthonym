"""
Task R1-FIX (v51) — two PIN defects a a review cross-family review found in R1.

a review-A [regression R1 introduced]: R1 added ``!R`` to the ``urea`` SMARTS to
reclaim the ring-N carboxamide. That ``!R`` on both nitrogens also stopped
CYCLIC ureas being perceived, which unmasked an ``oxo``-PREFIX generator on the
N-substituted ones (``1,3-dimethyl-2-oxoimidazolidine``, non-PIN). The root
cause is downstream: ``heterocycles.py``'s ``ring_ketone_suffix`` gate only fired
for ``secondary_amide`` (the N-H cyclic ureas), so an N-substituted cyclic urea
(``tertiary_amide``) fell through to the ``oxo`` prefix. A cyclic urea /
ring amide is a cyclic pseudoketone named with the ``-one`` SUFFIX.

  * (the Blue Book) "Pseudoketones": "(a) cyclic compounds in which a
    carbonyl group in a ring is bonded to one or two skeletal heteroatoms" are
    named with the ``-one`` suffix. Cf. the Blue Book ``imidazolidine-2,4-dione
    (PIN)`` — the two ring carbonyls of the cyclic diamide are ``-dione``, not
    ``2,4-dioxo`` prefixes.
  * (the Blue Book) "Lactams … Method (1) generates preferred IUPAC names"
    — the cyclic amide is a heterocyclic pseudoketone (``-one``), and the ring
    nitrogen is numbered.

a review-B [symmetric-N locant]: on a ring with two EQUIVALENT nitrogens the ring-N
suffix took the wrong (higher) locant — ``piperazine-4-carboxylic acid`` where
the suffix nitrogen should be locant 1. A saturated ring N-H was counted as an
"indicated hydrogen" and out-ranked the suffix nitrogen for locant 1. Indicated
hydrogen is a MANCUDE-ring concept; a fully saturated ring has none.

  * NUMBERING (the Blue Book) criterion (c) "principal
    characteristic groups and free valences (suffixes)" — which outranks (f)
    "detachable alphabetized prefixes" — gives ``piperazine-1-carboxylic acid``.
"""

import pytest

from orthonym import name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# a review-A — cyclic urea / ring amide -> '-one' SUFFIX, not 'oxo'.
CYCLIC_UREA_ONE = [
    # N-substituted cyclic ureas (tertiary_amide) — the regressed cases.
    ("CN1CCN(C1=O)c1ccccc1", "1-methyl-3-phenylimidazolidin-2-one"),
    ("CN1CCN(C)C1=O", "1,3-dimethylimidazolidin-2-one"),          # DMI
    ("CN1CCCN(C)C1=O", "1,3-dimethyl-1,3-diazinan-2-one"),        # DMPU
    # N-H cyclic ureas (secondary_amide) — must stay correct.
    ("O=C1NCCN1", "imidazolidin-2-one"),
    ("O=C1NCCCN1", "1,3-diazinan-2-one"),
    # N-substituted multi-heteroatom ring amides (same gate) — bonus, were 'oxo'.
    ("CN1CCOCC1=O", "4-methylmorpholin-3-one"),
    # N-H multi-heteroatom ring amide — must stay correct.
    ("O=C1CSCN1", "1,3-thiazolidin-4-one"),
]

# a review-B — ring-N suffix takes the lowest locant on a SYMMETRIC-N ring.
SYMMETRIC_N_LOCANT = [
    ("OC(=O)N1CCNCC1", "piperazine-1-carboxylic acid"),
    ("NC(=O)N1CCNCC1", "piperazine-1-carboxamide"),
    ("OC(=O)N1CCNC1", "imidazolidine-1-carboxylic acid"),
    ("OC(=O)N1CCNC(=O)C1", "3-oxopiperazine-1-carboxylic acid"),
]

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", CYCLIC_UREA_ONE)
def test_cyclic_urea_names_as_one_suffix(smiles, expected):
    """A cyclic urea / ring amide is a '-one' pseudoketone."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", SYMMETRIC_N_LOCANT)
def test_symmetric_ring_n_suffix_lowest_locant(smiles, expected):
    """The ring-N suffix takes the lowest locant on a symmetric-N ring
     NUMBERING criterion (c), the Blue Book)."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)N1CCCC1", "pyrrolidine-1-carboxylic acid"),
    ("NC(=O)N1CCCCC1", "piperidine-1-carboxamide"),
    ("OC(=O)N1CCN(C)CC1", "4-methylpiperazine-1-carboxylic acid"),
    ("OC(=O)N1CCCCC1", "piperidine-1-carboxylic acid"),
])
def test_r1_wins_preserved(smiles, expected):
    """R1's ring-N suffix wins must not regress."""
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # R2/R3 ring-S oxoacid + anion wins (separate committed tasks) — sanity.
    ("OS(=O)N1CCCCC1", "piperidine-1-sulfinic acid"),
])
def test_r2_r3_wins_preserved(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize(
    "smiles,expected",
    CYCLIC_UREA_ONE + SYMMETRIC_N_LOCANT,
)
def test_r1fix_round_trip(smiles, expected):
    """Each emitted name OPSIN-parses back to the input structure (0-wrong)."""
    assert name_compound(smiles) == expected
    result = opsin_roundtrip_check(smiles, expected)
    assert result["passed"], (
        f"OPSIN round-trip failed for {expected!r}: {result.get('error')}"
    )
