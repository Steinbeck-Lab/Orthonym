"""PIN spelling -- review fixes, a performance pass (TRIAGE.md 'Review fixes (a performance pass)' under
'PIN spelling -- four classes from the PubChem 1M job').

Every expected name below was read back by an independent OPSIN 2.9.0 call to the
input's full InChIKey (``tests/support/rt_assert``); the old spelling is the mutation
each name test rejects (``scripts/mutation_check.py``).

  RF3-3 a compound prefix is enclosed once, the Blue Book), also
         at a locant-free parent: '[(propan-2-yl)oxy]cyclohexane'.
  RF3-2 (c) (:7170, 'bis(azacyclododecane) (PIN)':7176) at the parent
         join of multiplicative nomenclature: "1,1'-methylenebis(azepane)".
  RF3-4 a completely labelled 'oxy' group before its parent cites no locants
         ,:44196): '(2H5)ethoxybenzene'.
  RF3-6 a locant-free spelling OPSIN 2.9.0 misreads keeps its locants and is
         labelled below the PIN: '(1,1,2,2,3,3,3-2H7)propylbenzene'.
  RF3-7 the ring locant of a monosubstituted pyrazine carboximidamide.
"""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


def _named(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    return name, res.get("tier")


ETHER_ONE_MARK_ROWS = [
    ("CC(C)OC1CCCCC1", "[(propan-2-yl)oxy]cyclohexane"),
    ("CC(C)OC1CCCC1", "[(propan-2-yl)oxy]cyclopentane"),
    ("CCC(C)OC1CCCCC1", "[(butan-2-yl)oxy]cyclohexane"),
    ("CC(C)OC(C)C", "2-[(propan-2-yl)oxy]propane"),
    ("CC(C)OC1CCC(C)CC1", "1-methyl-4-[(propan-2-yl)oxy]cyclohexane"),
    ("CC(C)OC(C)COC(C)C", "1,2-bis[(propan-2-yl)oxy]propane"),
    ("CC(C)Oc1ccccc1", "[(propan-2-yl)oxy]benzene"),
]


@pytest.mark.parametrize("smiles,expected", ETHER_ONE_MARK_ROWS)
def test_compound_prefix_is_enclosed_once(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


MULTIPLICATIVE_PARENT_ROWS = [
    ("C(N1CCCCCC1)N1CCCCCC1", "1,1'-methylenebis(azepane)"),
    # retained parents keep 'di'
    ("C(N1CCCC1)N1CCCC1", "1,1'-methylenedipyrrolidine"),
    # A C=O bridge between two ring nitrogens is the pseudoketone parent, the rings its
    # prefixes (b), the Blue Book;,:29370; PIN class program
    # batch 2 fix a performance pass); the multiplier rule is the same: 'bis' for the
    # Hantzsch-Widman prefix (c),:7176), 'di' for the retained ones.
    ("O=C(N1CCCCCC1)N1CCCCCC1", "bis(azepan-1-yl)methanone"),
    ("O=C(N1CCCCC1)N1CCCCC1", "di(piperidin-1-yl)methanone"),
    ("O=C(N1CCOCC1)N1CCOCC1", "di(morpholin-4-yl)methanone"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLICATIVE_PARENT_ROWS)
def test_multiplicative_parent_with_a_replacement_front(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


@pytest.mark.parametrize("name,derived", [
    ("azetidine", True), ("azepane", True), ("1,3-dioxolane", True),
    ("1H-tetrazol-5-yl", True),
    ("piperidine", False), ("morpholine", False), ("pyridine", False),
    ("1H-indazol-3-yl", False), ("thianthren-2-yl", False),
    ("phosphinolin-2-yl", False), ("phosphono", False), ("borono", False),
])
def test_replacement_front_predicate_edges(name, derived):
    from orthonym.assembly.naming_utils import opens_with_replacement_prefix
    assert opens_with_replacement_prefix(name) is derived


COMPLETE_OXY_ROWS = [
    ("[2H]C([2H])([2H])C([2H])([2H])Oc1ccccc1", "(2H5)ethoxybenzene"),
    ("[2H]C([2H])([2H])C([2H])([2H])OCCCC", "1-(2H5)ethoxybutane"),
    ("[2H]C([2H])([2H])C([2H])([2H])Oc1ccc(C)cc1", "1-(2H5)ethoxy-4-methylbenzene"),
    # a free position left: the locants stay
    ("[2H]C([2H])([2H])C([2H])([2H])O", "(1,1,2,2,2-2H5)ethan-1-ol"),
    ("[2H]c1c([2H])c([2H])c(COC(C)=O)c([2H])c1[2H]", "(2,3,4,5,6-2H5)benzyl acetate"),
]


@pytest.mark.parametrize("smiles,expected", COMPLETE_OXY_ROWS)
def test_completely_labelled_oxy_group_omits_locants(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


@pytest.mark.parametrize("smiles,expected", [
    ("[2H]C([2H])([2H])C([2H])([2H])C([2H])([2H])c1ccccc1",
     "(1,1,2,2,3,3,3-2H7)propylbenzene"),
    ("[2H]C([2H])([2H])C([2H])([2H])C([2H])([2H])C([2H])([2H])c1ccccc1",
     "(1,1,2,2,3,3,4,4,4-2H9)butylbenzene"),
])
def test_unreadable_complete_unit_spelling_is_below_the_pin(smiles, expected):
    # (:44196) would omit the locants; OPSIN 2.9.0 reads
    # '(2H7)propylbenzene' as the propan-2-yl isomer, so the locanted name ships
    # below the PIN
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier != "pin_verified", tier


RING_SUFFIX_N_PREFIX_ROWS = [
    # 'pyrazinecarboxylic acid (PIN)' (:2949); '*N*-methylbenzenecarboximidamide'
    # (:34388), '*N*-hydroxycyclohexanecarboxamide' (:30150): the N-prefix of the
    # suffix leaves the ring locant omitted
    ("NC(=N)c1cnccn1", "pyrazinecarboximidamide"),
    ("CNC(=N)c1cnccn1", "N-methylpyrazinecarboximidamide"),
    ("CNC(=O)c1cnccn1", "N-methylpyrazinecarboxamide"),
    ("CNc1cnccn1", "N-methylpyrazinamine"),
    # a ring with more than one kind of position keeps its locant
    ("CNC(=O)c1ccncc1", "N-methylpyridine-4-carboxamide"),
]


@pytest.mark.parametrize("smiles,expected", RING_SUFFIX_N_PREFIX_ROWS)
def test_ring_suffix_locant_with_an_n_prefix(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier
