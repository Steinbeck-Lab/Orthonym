"""PIN spelling -- four classes from the PubChem 1M job (TRIAGE.md section of that
title).

The PubChem 1M branch-loss job found four spellings the engine ships at the PIN
tier although the name, read back, is the right molecule: the Blue Book spells it
otherwise. Each is fixed at its producer; every expected name below was read back
by an independent OPSIN 2.9.0 call to the input's full InChIKey
(``tests/support/rt_assert``), and the old spelling is the mutation each test
rejects (``scripts/mutation_check.py``).

  1 a basic multiplier joined to an italicized structural prefix keeps the hyphen
      (d), the Blue Book, "to separate italic letters from Roman
     letters", example 'di-*tert*-butyl':6964; (b):7067-7070), and an
     ester's multiplied organyl group is formed like any multiplied component
      (a):7085 'di(propan-2-yl)'; 'di(propan-2-yl) disulfite (PIN)'
     :36921).
  2 identical prefixes are one multiplied group whatever atom bears them: an
     amine N-substituent and a ring prefix of the same name are cited once,
     'N,2-dimethyl' (b), the Blue Book; (a):7104;
     '*N*,4-dimethyl-*N*-(3-methylphenyl)benzamide (PIN)':32879, '*N*,1,4-
     triphenyl-1*H*-1,2,4-triazol-4-ium-3-aminide (PIN)':42460), the locants in
     the order (:3195, italic letters before numerals). Here the fused
     heteroring and fused hydrocarbon amines; the chain and ring-parent producers
     wait for the lane merge (TRIAGE.md).
  3 a descriptor for the atoms of a substituent prefix follows that prefix's
     attachment locant,:43718; '4-(2-14C)ethylbenzoic acid (PIN)'
     :43818, '2-(13C)methyl-3-methylpyridine (PIN)':43764).
  4 a parent with one amide / amine nitrogen cites it 'N' ('(N-2H1)acetamide
     (PIN)',:43824/:43828); numbered N locants are for the nitrogens of
     di- and polyamines, -imines and -amides,:7739); a ring nitrogen
     is located by its ring numeral.
"""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


# --------------------------------------------------------------------------
# 1 multiplied italicized prefixes: 'di-tert-butyl', never 'ditert-butyl'
# --------------------------------------------------------------------------

ITALIC_ROWS = [
    # fused heteroring prefixes (rules/fused_rings.py, _format_c_prefix)
    ("CC(C)(C)c1ccc2c(c1)[nH]c1cc(C(C)(C)C)ccc12", "2,7-di-tert-butyl-9H-carbazole"),
    ("CC(C)(C)c1ccc2c(c1)Cc1cc(C(C)(C)C)ccc12", "2,7-di-tert-butyl-9H-fluorene"),
    ("CC(C)(C)c1cc(C(C)(C)C)c2ncccc2c1", "6,8-di-tert-butylquinoline"),
    # symmetric diesters (rules/esters.py): the organyl multiplier
    ("CC(C)(C)OC(=O)C(=O)OC(C)(C)C", "di-tert-butyl oxalate"),
    ("CC(C)(C)OC(=O)CC(=O)OC(C)(C)C", "di-tert-butyl propanedioate"),
    ("CC(C)(C)OC(=O)c1ccc(C(=O)OC(C)(C)C)cc1", "di-tert-butyl benzene-1,4-dicarboxylate"),
    ("CC(C)OC(=O)C(=O)OC(C)C", "di(propan-2-yl) oxalate"),
]


@pytest.mark.parametrize("smiles,expected", ITALIC_ROWS)
def test_multiplied_italic_prefix_keeps_its_hyphen(smiles, expected):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == "pin_verified", res.get("tier")


def test_general_engine_n_substituents_keep_the_hyphen():
    """The general engine's N-substituent entries (assembly/general_engine.py): an
    'N,N-di-tert-butyl' carboxamide on a von Baeyer parent (a best-effort name)."""
    smi = "CC(C)(C)N(C(C)(C)C)C(=O)c1ccc2OCOc2c1"
    name = assert_full_rt(name_best_effort(smi).get("name"), smi)
    assert "N,N-di-tert-butyl-" in name, name
    assert "ditert" not in name, name


# --------------------------------------------------------------------------
# 2 identical prefixes on a ring carbon and on the amine nitrogen are one
# multiplied group: 'N,2-dimethyl', never 'N-methyl-2-methyl'
# --------------------------------------------------------------------------

IDENTICAL_PREFIX_ROWS = [
    # fused heteroring amines (rules/fused_rings.py, _assemble_fused_heterocycle_name)
    ("CN1C(=C2C=CC(=CC2=N1)N(C)C3=CC=CC=C3)Cl",
     "3-chloro-N,2-dimethyl-N-phenyl-2H-indazol-6-amine"),
    ("C1=CC=C(C=C1)C2=NNC3=C2C=C(C=C3)NC4=CC=CC=C4", "N,3-diphenyl-1H-indazol-5-amine"),
    ("CNc1ccc2nc(C)ccc2c1", "N,2-dimethylquinolin-6-amine"),
    # fused hydrocarbon amines (rules/polycyclics.py, name_substituted_polycyclic)
    ("CNc1ccc2cc(C)ccc2c1", "N,6-dimethylnaphthalen-2-amine"),
    ("c1ccc(Nc2ccc3-c4ccccc4C(c4ccccc4)(c4ccccc4)c3c2)cc1",
     "N,9,9-triphenyl-9H-fluoren-2-amine"),
]


@pytest.mark.parametrize("smiles,expected", IDENTICAL_PREFIX_ROWS)
def test_identical_prefixes_on_ring_and_nitrogen_are_one_group(smiles, expected):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == "pin_verified", res.get("tier")


def test_prefix_locant_order_and_grouping():
    """ (the Blue Book): italic letter locants before numerals; a
    primed or letter-suffixed numeral right after its numeral (:3193)."""
    from orthonym.assembly.composition_primitives import (
        combine_identical_prefix_groups,
        prefix_locant_order_key,
    )
    locs = [4, "N", "2'", 2, "3a", "N'", 10, "N2"]
    assert sorted(locs, key=prefix_locant_order_key) == [
        "N", "N'", "N2", 2, "2'", "3a", 4, 10]
    groups = combine_identical_prefix_groups(
        [("methyl", ["N"]), ("chloro", [3]), ("methyl", [2])])
    assert groups == [("methyl", ["N", 2]), ("chloro", [3])]


# --------------------------------------------------------------------------
# 3 a descriptor for a substituent prefix's own atoms follows that prefix's
# attachment locant: '2-(2H5)phenyl', never '(2H5)-2-phenyl'
# --------------------------------------------------------------------------

DESCRIPTOR_SLOT_ROWS = [
    ("[2H]c1c([2H])c([2H])c(-c2cc3ccccc3o2)c([2H])c1[2H]",
     "2-(2H5)phenyl-1-benzofuran"),  #:44196, a complete phenyl
    ("[2H]c1c([2H])c([2H])c(CCC(=O)O)c([2H])c1[2H]",
     "3-(2H5)phenylpropanoic acid"),  #:44196
]


@pytest.mark.parametrize("smiles,expected", DESCRIPTOR_SLOT_ROWS)
def test_descriptor_follows_the_substituent_locant(smiles, expected):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert name == expected, name
    assert res.get("tier") == "pin_verified", res.get("tier")


def test_slot_before_a_substituent_locant_is_recognised():
    """The slot in front of a substituent prefix's attachment locant is demoted;
    the slots in front of a parent's own locants (replacement / Hantzsch-Widman
    heteroatom locants, indicated hydrogen, ring-assembly locants) are not."""
    from orthonym.rules.isotopes import _slot_precedes_substituent_locant as slot
    demoted = [
        ("2-phenyl-1-benzofuran", 0),
        ("3-phenylpropanoic acid", 0),
        ("6-(dibenzo[b,d]thiophen-4-yl)-4-phenyl-1,3,5-triazin-2-yl",
         len("6-(dibenzo[b,d]thiophen-4-yl)")),
        ("2-methyl-3-methylpyridine", len("2-methyl")),
        ("2,6-di-tert-butylphenol", 0),
    ]
    for skel, off in demoted:
        assert slot(skel, off), (skel, off)
    kept = [
        ("2,4-dioxo-1,3-diazaspiro[4.7]dodecane", len("2,4-dioxo")),
        ("3-amino-6-(2,3-dichlorophenyl)-1,2,4-triazin-2-ium",
         len("3-amino-6-(2,3-dichlorophenyl)")),
        ("N-phenyl-1,5-naphthyridin-2-amine", len("N-phenyl")),
        ("1-methyl-1H-indole", len("1-methyl")),
        ("2,2'-bipyridine", 0),
        ("3,6-dimethyl-1,4-dioxane-2,5-dione", len("3,6-dimethyl")),
    ]
    for skel, off in kept:
        assert not slot(skel, off), (skel, off)


# --------------------------------------------------------------------------
# 4 the amide / amine nitrogen of a parent with ONE such nitrogen is 'N', not
# 'N1', whatever other nitrogens the molecule has; a ring nitrogen takes its
# ring numeral
# --------------------------------------------------------------------------

N_LOCANT_ROWS = [
    # a pyridine ring N elsewhere in the molecule; one amide N -> 'N'
    ("[2H]C([2H])(c1ccncc1)C(=O)N([2H])c1ccccc1", "(N,2,2-2H3)acetamide"),
    # the PubChem 1M row of the acetamide merge (a tetrazolo ring elsewhere)
    ("[2H]C([2H])(C1=NC2=C(C=CC(=C2)Cl)N3C1=NN=N3)C(=O)N([2H])C4=CC=CC=C4O[2H]",
     "(N,2,2-2H3)acetamide"),
    # a D on a ring nitrogen: its ring numeral, not 'N3'
    ("[2H]N1C([C@@H](N(C1C2=CC=CC=C2)C3=CC=CC=C3C)C)C4CCCCC4", "(3-2H)imidazolidine"),
]


@pytest.mark.parametrize("smiles,part", N_LOCANT_ROWS)
def test_single_suffix_nitrogen_is_bare_n(smiles, part):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert part in name, name
    assert "(N1" not in name and "N3-" not in name, name


def test_several_suffix_nitrogens_keep_numbered_locants():
    """ (the Blue Book): the nitrogens of a diamine are told apart by
    numbered locants; the skeleton decides, not the molecule's nitrogen count."""
    from orthonym.rules.isotopes import _parent_numbers_its_nitrogens as numbers
    assert numbers("propane-1,2-diamine")
    assert numbers("N1-methylpropane-1,2-diamine")
    assert numbers("benzene-1,3-dicarboxamide")
    assert not numbers("2-(pyridin-4-yl)-N-phenylacetamide")
    assert not numbers("N-methylpyridine-4-carboxamide")
    for smi, expected in (("[2H]NCC(N)C", "(N1-2H1)propane-1,2-diamine"),
                          ("[2H]NCCN", "(N1-2H1)ethane-1,2-diamine")):
        name = assert_full_rt(name_best_effort(smi).get("name"), smi)
        assert name == expected, name


def test_ring_nitrogen_takes_its_ring_numeral():
    """A D on the one (ring) nitrogen of a von Baeyer parent: the ring numeral of
    that skeletal atom, never the italic letter locant of a suffix nitrogen."""
    smi = "[2H]N1CC2CCCCC2C1"
    name = assert_full_rt(name_best_effort(smi).get("name"), smi)
    assert name == "(8-2H)-8-azabicyclo[4.3.0]nonane", name
