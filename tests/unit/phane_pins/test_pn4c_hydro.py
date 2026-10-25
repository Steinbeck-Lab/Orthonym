import random

import pytest
from rdkit import Chem

from orthonym.rules import phane_pin
from orthonym.rules.phane_pin.amplificants import amplificant_hydride, hydro_split

from ._helpers import default_tier_row, mol, name_best_effort, needs_phane_opsin, substituted

SPELLINGS = 30
TERMINAL_RINGS = "c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1"      # the end of a chain of four rings and three oxygen atoms (:23959)


def _name(smiles):
    m = mol(smiles)
    return phane_pin.general_phane_name(m) or phane_pin.linear_phane_name(Chem.Mol(m))


def _respelled(smiles, seed):
    m = mol(smiles)
    order = list(range(m.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(m, order), canonical=False))


def _ring(m):
    return frozenset(a for r in m.GetRingInfo().AtomRings() for a in r)


# derived rows (the book prints none for them): the parent is the mancude ring system, the added hydrogen atoms are hydro
# prefixes and the indicated hydrogen of the parent is cited with them,:15173; (b), (e)(i),:3246,:3289)
DERIVED = [
    ("C1CCc2ccc([nH]2)CCc2ccc1[nH]2", "1^1H,4^1H-1,4(2,5)-dipyrrolacycloheptaphane"),
    ("O(c1ccc2CCCCc2c1)" + TERMINAL_RINGS,
     "1^5,1^6,1^7,1^8-tetrahydro-2,4,6-trioxa-1(2)-naphthalena-3,5(1,4),7(1)-tribenzenaheptaphane"),
    ("O(c1ccc2[nH]ccc2c1)" + TERMINAL_RINGS,
     "1^1H-2,4,6-trioxa-1(5)-indola-3,5(1,4),7(1)-tribenzenaheptaphane"),
    ("O(c1ccccc1)c1ccc(Oc2ccc([nH]2)Oc2ccc(Oc3ccccc3)cc2)cc1",
     "5^1H-2,4,6,8-tetraoxa-5(2,5)-pyrrola-1,9(1),3,7(1,4)-tetrabenzenanonaphane"),
    ("O(c1ccc2CC(O)CCc2c1)" + TERMINAL_RINGS,
     "1^5,1^6,1^7,1^8-tetrahydro-2,4,6-trioxa-1(2)-naphthalena-3,5(1,4),7(1)-tribenzenaheptaphan-1^6-ol"),
    ("O(c1ccc2CC[C@H](O)Cc2c1)" + TERMINAL_RINGS,
     "(1^7S)-1^5,1^6,1^7,1^8-tetrahydro-2,4,6-trioxa-1(2)-naphthalena-3,5(1,4),7(1)-tribenzenaheptaphan-1^7-ol"),
    ("O(c1ccc2OCOc2c1)" + TERMINAL_RINGS,
     "1^2H-2,4,6-trioxa-1(5)-[1,3]benzodioxola-3,5(1,4),7(1)-tribenzenaheptaphane"),
    # (:3557): 9H-fluorene is a mancude ring system, so the macrocycle is a cyclophane (1))
    ("C1CCCCCCCCCc2ccc3c(c2)Cc2cc1ccc2-3", "1^9H-1(2,7)-fluorenacycloundecaphane"),
    # (:14967): the amplificant is numbered by the rules of its parent name first, so the indicated hydrogen
    # takes locant 1 (1-methyl-1H-pyrazole, (b):3246) and the attachment locants are 5 and 4, not 3 and 4 of 2H
    ("c1ccc(Oc2ccc(Oc3cnn(C)c3Oc3ccccc3)cc2)cc1",
     "3^1-methyl-3^1H-2,4,6-trioxa-3(5,4)-pyrazola-1,7(1),5(1,4)-tribenzenaheptaphane"),
]
DERIVED_IDS = ["dipyrrola", "tetralin", "indole", "pyrrole_in_the_chain", "tetralinol", "tetralinol_S", "benzodioxole",
               "fluorene", "pyrazole"]

# the book's:49696: the structure of its second name; the PIN has '1^2Z' as well,:49686: more than seven nodes, all
# descriptors), for a ring double bond that no SMILES of the input can state, and OPSIN does not read it. The producer builds
# the book's second name; the name is not a PIN, so the default tier declines it (``cyclophane_descriptor_check``).
PYRIDINE_ROW = "C/1=C\\CCCC2=CC(=CC=C2)CCCCCC2CCC=C1N2"
PYRIDINE_NAME = "(2E)-1^1,1^4,1^5,1^6-tetrahydro-1(2,6)-pyridina-7(1,3)-benzenacyclododecaphan-2-ene"
PYRIDINE_PIN = "(1^2Z,2E)-1^1,1^4,1^5,1^6-tetrahydro-1(2,6)-pyridina-7(1,3)-benzenacyclododecaphan-2-ene"
#:17067 prints its PIN without a descriptor for the two ring double bonds of its two dihydropyridine amplificants, in a
# cyclophane of twelve nodes: the same class as:49696, so the same decision (the printed name is built, not certified)
DIHYDROPYRIDINES_ROW = "C1CCCCC2=NC(=CC=C2)CCCCCC2=CCC=C1N2"
UNCERTIFIED = {"bb17067"}


@pytest.mark.parametrize("line,name,smiles", substituted("indicated_hydrogen") + substituted("hydro"),
                         ids=lambda v: str(v)[:30])
def test_book_rows_of_hydro_and_indicated_hydrogen(line, name, smiles):
    assert _name(smiles) == name


@pytest.mark.parametrize("smiles,name", DERIVED, ids=DERIVED_IDS)
def test_derived_hydro_names(smiles, name):
    assert _name(smiles) == name


def test_the_hydro_prefixes_and_the_ene_are_compared_by_their_primary_locants_first():
    #:49696: the numbering with the 'ene' at 2 and hydro prefixes 1^1,1^4,1^5,1^6, not the one with 1^1,1^2,1^3,1^4 and the
    # 'ene' at 11, whose superscripts are lower
    assert _name(PYRIDINE_ROW) == PYRIDINE_NAME


@needs_phane_opsin
@pytest.mark.parametrize("line,name,smiles", substituted("indicated_hydrogen") + substituted("hydro"),
                         ids=lambda v: str(v)[:30])
def test_book_rows_read_back(line, name, smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4c"):
        assert phane_pin.verified_or_none(name, smiles) == name


@needs_phane_opsin
@pytest.mark.parametrize("smiles,name", DERIVED + [(PYRIDINE_ROW, PYRIDINE_NAME)], ids=DERIVED_IDS + ["pyridine"])
def test_derived_hydro_names_read_back(smiles, name):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4c"):
        assert phane_pin.verified_or_none(name, smiles) == name


@pytest.mark.parametrize("smiles", [DERIVED[0][0], DERIVED[2][0], DERIVED[3][0], PYRIDINE_ROW,
                                    substituted("hydro")[2][2]], ids=["dipyrrola", "indole", "pyrrole", "pyridine", "decahydro"])
def test_the_name_does_not_depend_on_the_order_of_the_atoms(smiles):
    names = {(phane_pin.general_phane_name(r) or phane_pin.linear_phane_name(Chem.Mol(r)))
             for r in (_respelled(smiles, seed) for seed in range(SPELLINGS))}
    assert len(names) == 1 and None not in names, names


def test_a_pyrrole_ring_has_one_indicated_hydrogen_and_no_hydro_prefix():
    m = mol("Cc1cc[nH]c1")
    got = amplificant_hydride(m, _ring(m))
    assert got.name == "pyrrole" and got.indicated == 1 and len(got.saturated) == 1
    assert all(len(hydro_split(got, numbering)[1]) == 0 for numbering in got.numberings)


def test_the_indicated_hydrogen_takes_the_lowest_locant_that_leaves_a_mancude_parent():
    # 3,4-dihydro-2H-pyran, not 2,3-dihydro-4H-pyran (b)): of the saturated atoms 2, 3 and 4 the indicated one is the lowest
    m = mol("C1CC=COC1")
    got = amplificant_hydride(m, _ring(m))
    best = min(((sorted(numbering[a] for a in ind), sorted(numbering[a] for a in hyd))
                for numbering in got.numberings for ind, hyd in [hydro_split(got, numbering)]))
    assert got.name == "pyran" and best == ([(2, "")], [(3, ""), (4, "")])


@pytest.mark.parametrize("smiles", [
    "CC1=CCCCC1",                   # a cycloalkene is named with 'ene', not as a hydro derivative of benzene
    "O=C1CCC=CC1",                  # an exocyclic double bond on the ring
    "C1=CC=C[NH+]=C1",              # a charge
])
def test_rings_that_are_no_hydro_amplificants_decline(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert amplificant_hydride(m, _ring(m)) is None


def test_the_indicated_hydrogen_check_reads_the_composite_locants_of_the_front_group():
    # (:3721) asks for the locant and the symbol 'H'; the locant of an atom of an amplificant is composite
    #,:15111) and its indicated hydrogen stands in front of the name,:15173): '1^1H-' is a cited group
    from orthonym.validation.spelling.checks_hydrogen import indicated_hydrogen_check
    for smiles, name in DERIVED:
        assert indicated_hydrogen_check(mol(smiles), name) is None, name
    smiles, name = DERIVED[2]
    failure = indicated_hydrogen_check(mol(smiles), name.removeprefix("1^1H-"))
    assert failure is not None and failure.rule == "P-14.7.1", failure


_DEFAULT_TIER_ROWS = [(f"bb{row[0]}", row[2], row[1]) for row in substituted("indicated_hydrogen") + substituted("hydro")]
_DEFAULT_TIER_ROWS = [r for r in _DEFAULT_TIER_ROWS if r[0] not in UNCERTIFIED] + \
    [(i, s, n) for i, (s, n) in zip(DERIVED_IDS, DERIVED)]


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [(s, n) for _, s, n in _DEFAULT_TIER_ROWS], ids=[i for i, _, _ in _DEFAULT_TIER_ROWS])
def test_the_names_with_hydro_and_indicated_hydrogen_are_the_default_tier_names(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


def test_the_indicated_hydrogen_is_numbered_before_the_attachment_locants():
    # (:14967): "Numbering of an amplificant is determined primarily by the numbering rules that apply to the
    # parent name"; (b) (:3246) gives the indicated hydrogen the lowest locants. 2H-pyran, not 6H-pyran, although
    # the numbering of 6H has the lower attachment locant (2) for an attachment at the carbon next to the oxygen
    from orthonym.rules.phane_pin.amplificants import indicated_numberings
    m = mol("C1C=CC=CO1")
    got = amplificant_hydride(m, _ring(m))
    assert {tuple(numbering[a] for a in hydro_split(got, numbering)[0]) for numbering in got.numberings} == {((2, ""),), ((6, ""),)}
    assert {tuple(numbering[a] for a in hydro_split(got, numbering)[0]) for numbering in indicated_numberings(got)} == {((2, ""),)}


def test_a_ring_without_indicated_hydrogen_keeps_every_numbering():
    from orthonym.rules.phane_pin.amplificants import indicated_numberings
    m = mol("c1ccc2ccccc2c1")
    got = amplificant_hydride(m, _ring(m))
    assert indicated_numberings(got) == tuple(got.numberings) and len(got.numberings) > 1


def test_a_ring_double_bond_of_a_hydro_amplificant_in_a_large_cyclophane_needs_a_descriptor():
    # (:49684): "When more than seven nodes are present, all descriptors must be cited"; the example (:49696) cites
    # '1^2Z' and gives the name without it as the other form. Twelve nodes here, one non-aromatic ring double bond
    from orthonym.validation.spelling.checks_phane import cyclophane_descriptor_check
    from orthonym.validation.spelling.cyclophane import large_cyclophane_ring_double_bonds
    assert len(large_cyclophane_ring_double_bonds(mol(PYRIDINE_ROW))) == 1
    assert len(large_cyclophane_ring_double_bonds(mol(DIHYDROPYRIDINES_ROW))) == 2
    failure = cyclophane_descriptor_check(mol(PYRIDINE_ROW), PYRIDINE_NAME)
    assert failure is not None and failure.rule == "P-93.5.5.3"
    assert cyclophane_descriptor_check(mol(PYRIDINE_ROW), PYRIDINE_PIN) is None      # the book's PIN cites it
    assert cyclophane_descriptor_check(mol(PYRIDINE_ROW), "1,7(2,6)-dipyridina-cyclododecane") is None   # no phane name


@pytest.mark.parametrize("smiles", [
    substituted("indicated_hydrogen")[0][2],      #:15181, 4H-pyran, six nodes: up to seven nodes no descriptor is necessary
    substituted("hydro")[0][2],                   #:21442, a linear phane: no larger ring
    "C1/C=C/c2cccc(c2)CCCc2cccc(c2)C1",           # nine nodes, benzene rings: the Kekule bonds of a mancude ring are cited by no name
], ids=["six_nodes", "linear", "mancude"])
def test_no_descriptor_is_needed_for_the_other_structures(smiles):
    from orthonym.validation.spelling.cyclophane import large_cyclophane_ring_double_bonds
    assert large_cyclophane_ring_double_bonds(mol(smiles)) == []


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [(PYRIDINE_ROW, PYRIDINE_NAME), (DIHYDROPYRIDINES_ROW, substituted("hydro")[1][1])],
                         ids=["bb49696", "bb17067"])
def test_the_name_without_the_required_descriptors_is_no_default_tier_name(smiles, name):
    # the default tier declines (no verified PIN); the best-effort tier keeps the name, which reads back, as a systematic name
    from tests.support.rt_assert import assert_full_rt
    row = default_tier_row(smiles)
    assert row["name"] != name and row["tier"] == "abstain" and [f["rule"] for f in row["spelling_failures"]] == ["P-93.5.5.3"], row
    kept = name_best_effort(smiles)
    assert kept["name"] == name and kept["tier"] == "systematic_verified", kept
    assert_full_rt(name, smiles)
