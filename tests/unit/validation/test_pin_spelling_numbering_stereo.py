"""The spelling check of reads (g) and (j) apart (``validation/spelling/models.py``).

``### **** NUMBERING`` (``the Blue Book``): (g) (``:3307``) is decided by, which leaves the
stereodescriptors out (``:3446``), so two prefixes that differ only in R and S are one prefix to it; the choice between
them is (j) (``:3346``), "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r... preferred to E, S, P,
and s", the descriptors of the parent and of the prefixes together, in order of locant, at the first point of difference
(``(2Z,4S,8R,9E)-undeca-2,9-diene-4,8-diol (PIN)``, ``:3403``, "the choice is between 'E' and 'Z' for position '2', not
between 'R' and 'S' for position '4'", ``:3405``; ``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``, ``:3399``).
The check is the only guard that keeps a name with the S prefix at the lower locant out of ``pin_verified``: each pair
below is the name that is right and the same molecule's name with a numbering that gives S, or E, the lower locant.
A unit with an isotopic descriptor has no model, so the check abstains on it ((i), ``:3336``, comes before (j)).
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling

#: (SMILES, the name the book's rule gives, the name that numbers the same molecule the other way)
PAIRS = [
    ("pyridine 2,6", "C[C@H](Cl)c1cccc([C@@H](C)Cl)n1",
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]pyridine",
     "6-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl]pyridine"),
    ("naphthalene 1,5", "C[C@H](Cl)c1cccc2c([C@@H](C)Cl)cccc12",
     "1-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]naphthalene",
     "5-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]naphthalene"),
    ("benzene 1,3 (:3399)", "CC[C@@H](C)c1cccc([C@@H](C)CC)c1",
     "1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene",
     "3-[(2R)-butan-2-yl]-1-[(2S)-butan-2-yl]benzene"),
    ("cyclohexane 1,4", "C[C@@H](Cl)C1CCC(CC1)[C@H](C)Cl",
     "1-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]cyclohexane",
     "4-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]cyclohexane"),
    ("heptane-2,6-diol: the R at 2 of the parent decides, not the prefixes",
     "C[C@@H](O)C([C@H](C)Cl)CC([C@@H](C)Cl)[C@H](C)O",
     "(2R,6S)-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptane-2,6-diol",
     "(2S,6R)-3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]heptane-2,6-diol"),
    ("undeca-2,9-diene (:3403): the Z at 2 decides, not the prefixes",
     "C/C=C\\C([C@H](C)Cl)CCCC([C@@H](C)Cl)/C=C/C",
     "(2Z,9E)-8-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]undeca-2,9-diene",
     "(2E,9Z)-4-[(1R)-1-chloroethyl]-8-[(1S)-1-chloroethyl]undeca-2,9-diene"),
    ("pentane-2,4-diol, the parent alone", "C[C@H](O)C[C@@H](C)O",
     "(2R,4S)-pentane-2,4-diol", "(2S,4R)-pentane-2,4-diol"),
    ("1,3-dichlorocyclohexane, the parent alone", "Cl[C@H]1CCC[C@@H](Cl)C1",
     "(1R,3S)-1,3-dichlorocyclohexane", "(1S,3R)-1,3-dichlorocyclohexane"),
    ("a substituent chain with its own descriptors: (2R,4S)-2,4-dichloropentan-3-yl", "C[C@H](Cl)C(C1CN1)[C@@H](C)Cl",
     "2-[(2R,4S)-2,4-dichloropentan-3-yl]aziridine", "2-[(2S,4R)-2,4-dichloropentan-3-yl]aziridine"),
    ("hepta-2,5-dienedioic acid, Z before E", "OC(=O)/C=C\\C/C=C/C(=O)O",
     "(2Z,5E)-hepta-2,5-dienedioic acid", "(2E,5Z)-hepta-2,5-dienedioic acid"),
]


def _failures(smiles, name):
    return [f for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True) if f.rule == "P-14.4"]


@pytest.mark.parametrize("why, smiles, right, wrong", PAIRS, ids=[r[0] for r in PAIRS])
def test_the_name_with_the_preferred_descriptor_at_the_lower_locant_passes(why, smiles, right, wrong):
    assert _failures(smiles, right) == []


@pytest.mark.parametrize("why, smiles, right, wrong", PAIRS, ids=[r[0] for r in PAIRS])
def test_the_name_with_the_other_descriptor_at_the_lower_locant_fails(why, smiles, right, wrong):
    failures = _failures(smiles, wrong)
    assert failures and "stereodescriptors (j)" in failures[0].detail, failures


def test_prefixes_of_identical_letters_and_other_own_locants_are_two_prefixes_to_g():
    """``****`` (``the Blue Book``): prefixes of identical letters are ordered by the locants inside their
    names: '1-chloropropoxy' is cited before '2-chloropropoxy' and takes the lower locant, whatever the descriptors say
    ((g) comes before (j))."""
    smiles = "CC[C@H](Cl)OCC(O)COC[C@@H](C)Cl"
    assert _failures(smiles, "1-[(1S)-1-chloropropoxy]-3-[(2R)-2-chloropropoxy]propan-2-ol") == []
    failures = _failures(smiles, "3-[(1S)-1-chloropropoxy]-1-[(2R)-2-chloropropoxy]propan-2-ol")
    assert failures and "stereodescriptors" not in failures[0].detail and "prefix chloropropoxy" in failures[0].detail


def test_a_numbering_a_higher_criterion_fixes_is_not_failed_for_its_descriptors():
    """The suffix at 1 and the prefix locants {2,4} (c), (f),:3256,:3301) are before (j): R is cited first
    at the higher locant (``****``, ``the Blue Book``)."""
    assert _failures("CC[C@@H](C)C1CCC(=O)C([C@@H](C)CC)C1",
                     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]cyclohexan-1-one") == []


def test_a_unit_with_an_isotopic_descriptor_is_not_judged_by_it():
    """(i) (:3336) is before (j) and is not modelled: the label's numbering decides, whatever the twins say."""
    assert _failures("c1([C@H](C)Cl)cc([C@@H](C)Cl)c([2H])cc1",
                     "3-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl](4-2H)benzene") == []


def test_the_descriptor_order_of_the_check_is_the_one_of_the_numbering():
    """The checks import no naming code, so the (j) key is held twice: the check's and ``naming_utils``'s. They rank
    the same descriptors the same way (the table of ``the Blue Book``, then chiral before pseudoasymmetric,
    ``:45465``)."""
    import itertools
    import random

    from orthonym.assembly.naming_utils import cip_descriptor_codes, cip_locant_rank_key, numbering_cip_key
    from orthonym.validation.spelling.models import _prefix_codes, _stereo_key

    rng = random.Random(3)
    codes = ["R", "S", "Z", "E", "M", "P", "r", "s"]
    lists = [[((loc, k), rng.choice(codes)) for loc in rng.sample(range(1, 7), rng.randint(0, 4))
              for k in rng.sample(range(0, 3), rng.randint(1, 2))] for _ in range(60)]
    for a, b in itertools.product(lists, repeat=2):
        ka, kb, na, nb = _stereo_key(a), _stereo_key(b), cip_locant_rank_key(a), cip_locant_rank_key(b)
        assert (ka < kb, ka == kb) == (na < nb, na == nb), (a, b)
    for name in ["(1R)-1-chloroethyl", "(2R,3S)-3-chlorobutan-2-yl", "[(1R)-1-chloroethyl]sulfanyl", "(1E)-prop-1-en-1-yl",
                 "(1r,4r)-4-methylcyclohexyl", "methyl", "(2-2H)ethyl", "{[(2S)-butan-2-yl]oxy}"]:
        assert tuple(_prefix_codes(name)) == cip_descriptor_codes(name), name
    assert _stereo_key([((2, 0), "Z"), ((3, 1), "S")]) == cip_locant_rank_key([((2, 0), "Z"), ((3, 1), "S")])
    assert numbering_cip_key([(2, "Z")], [(3, "(1S)-1-chloroethyl")]) == _stereo_key([((2, 0), "Z"), ((3, 1), "S")])
