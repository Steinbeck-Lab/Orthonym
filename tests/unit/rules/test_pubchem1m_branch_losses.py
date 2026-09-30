"""PubChem 1M -- branch losses (TRIAGE.md 'PubChem 1M -- branch losses').

The paper's PubChem 1M recipe (Zenodo launch_generic.py: 30 processes, 45 s
SIGALRM, 90 s stall) run on the breadth branch lost 409 rows the paper named
round-trip exact. 62 of them the code of main 2b1455817 still names; these tests
pin the causes found on those 62 (the rows are PubChem rows, none is in
eval/splits/a holdout split.json; small analogues stand in where they reach the same
code). Every shipped name is read back by an independent OPSIN call to the input's
full InChIKey (tests/support/rt_assert).

  A multi-component drawings adducts)
     A1 OPSIN reads the em dash as a hyphen and parses greedily across it, so two
        correct component names can fuse into a word it cannot build
        ('benzene—2-aminopyridine', 'triphenylene—benzene'); the adduct then
        failed its read-back although each component reads back alone. The
        best-effort tier re-spells components (engine recovery, then floor
        spellings; the largest component first) until OPSIN reads the adduct back.
     A2 every component was named inside the whole assembly's hang budgets, so a
        mixture of four drug-size macrocycles ran out of the 500 analysis calls one
        compound gets; each component now has the budgets of one compound.
     A3 the oxo / ene valence check read the em dash as part of one word, joining
        the '8-ene' of one component with the '3-oxo' of the next.
  I isotope-labelled molecules
     I1 no insertion slot before a bracketed ring-assembly stem
        ('([1,1'-biphenyl]-4-yl)') or a bracketed substituent ring that opens with
        skeletal-replacement locants ('[9,24-dioxahexacyclo...');
     I2 the isotope round trip compared RDKit canonical SMILES, which depend on the
        Kekule form of a conjugated ring RDKit does not perceive as aromatic
        (10b,10c-dihydropyrene): the same compound (one InChI) failed;
     I3 two descriptors landing at one slot ('(2,2-2H2)' and '(N-2H1)' before
        'acetamide') failed closed; they are one descriptor '(N,2,2-2H3)'.
     Review fix: the I2 compare records no bond orders, so it accepted a
        bond-shift isomer of a labelled cyclooctatetraene; a differing parse is
        now accepted only when every alternating circuit has 4n+2 atoms.
  B speed: the clean-first offer re-derived big peptides from scratch (it is now
     capped at the naming passes the main path made), and the single-descriptor
     isotope sweep tried every locant for a nuclide that one position cannot bear.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from tests.support.rt_assert import (
    _independent_parse,
    assert_full_rt,
    name_best_effort,
)

pytestmark = pytest.mark.opsin_gate

EM = "—"


# --------------------------------------------------------------------------
# A1 component spellings OPSIN reads back in the adduct
# --------------------------------------------------------------------------

def test_fused_join_is_unreadable_and_the_suffix_form_is_not():
    """Why the adduct failed: OPSIN reads 'benzene—2-aminopyridine' as
    'benzene-2-amin...' and cannot place 'opyridine'; the same compounds with the
    amine as a suffix read back as the two components."""
    assert _independent_parse(f"benzene{EM}2-aminopyridine (1/1)") is None
    back = _independent_parse(f"benzene{EM}pyridin-2-amine (1/1)")
    assert back is not None
    assert Chem.MolToSmiles(Chem.MolFromSmiles(back)) == Chem.MolToSmiles(
        Chem.MolFromSmiles("c1ccccc1.Nc1ccccn1"))


ADDUCT_ROWS = {
    # 'N-(...-7-carbonyl)-3-chloro-4-fluoroaniline—2-amino-N-(...)-7-carboxamide'
    # fused as 'aniline-2-amin...'
    "aniline_amino": ("C.CC(C)(C)[S@@](=O)NC1CCCC2=C(N(C=C12)C)C(=O)NC3=CC(=C(C=C3)F)Cl."
                      "CN1C=C2C(CCCC2=C1C(=O)NC3=CC(=C(C=C3)F)Cl)N"),
    # '2-(3-bromonaphthalen-1-yl)triphenylene—7-bromo-...' fused as 'tri(phenylene)'
    "triphenylene": ("C1=CC=C(C=C1)C2=CC=CC=C2C3=CC=C(C=C3)Br."
                     "C1=CC=C(C=C1)C2=CC=CC3=CC(=C(C=C32)Br)C4=CC=CC=C4."
                     "C1=CC=C(C=C1)C2=CC3=CC=CC=C3C=C2Br."
                     "C1=CC=C(C=C1)C2=C(C=C3C(=C2)C=CC4=CC=CC=C43)Br."
                     "C1=CC=C2C(=C1)C=C(C=C2C3=CC4=C(C=C3)C5=CC=CC=C5C6=CC=CC=C64)Br."
                     "C1=CC(=CC=C1F)Br"),
}


@pytest.mark.parametrize("smiles", list(ADDUCT_ROWS.values()), ids=list(ADDUCT_ROWS))
def test_adduct_rows_name_rt_exact(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert EM in name
    assert res.get("tier") != "pin_verified" and not res.get("is_pin")


# Eight components, three of them two-word names at HEAD ('ethyl...propanoate',
# 'tert-butyl...carbamate', 'carbon dioxide'): OPSIN splits the words wrongly.
# The largest component is re-spelled first, so 'carbon dioxide' keeps its name
# (its only other spelling is the floor's '1,3-dioxapropa-1,2-diene').
EIGHT_ROW = ("CCC(C(=O)OCC)C(=O)OCC."
             "CCOC(=O)CC[C@H]1C[C@H](CCO1)C2=CN=C(C=C2)NCC3=CC=CC(=C3)C."
             "CCOC(=O)C(C[C@H]1C[C@H](CCO1)C2=CN=C(C=C2)NCC3=CC=CC(=C3)C)C(=O)OCC."
             "CCOC(=O)C(C[C@H]1C[C@H](CCO1)C2=CN=C(C=C2)N(CC3=CC=CC(=C3)C)"
             "C(=O)OC(C)(C)C)(C(=O)OCC)C(=O)OCC."
             "CC1=CC(=CC=C1)CNC2=NC=C(C=C2)[C@H]3CCO[C@H](C3)CCC(=O)O."
             "CC1=CC(=CC=C1)CN(C2=NC=C(C=C2)[C@H]3CCO[C@H](C3)CO)C(=O)OC(C)(C)C."
             "CC1=CC(=CC=C1)CN(C2=NC=C(C=C2)[C@H]3CCO[C@H](C3)CBr)C(=O)OC(C)(C)C."
             "C(=O)=O")


def test_small_component_keeps_its_name():
    res = name_best_effort(EIGHT_ROW)
    name = assert_full_rt(res.get("name"), EIGHT_ROW)
    assert name.endswith(f"{EM}carbon dioxide (1/1/1/1/1/1/1/1)"), name


def test_adduct_that_reads_back_is_unchanged():
    """A P-14.8 name that already reads back keeps every component's spelling (the
    Blue Book example 'benzene—pyridine (1/1)', P-14.8.1)."""
    from orthonym.rules.adducts import _readable_spellings
    ordered = [("c1ccncc1", 1), ("c1ccccc1", 1)]
    named = [("pyridine", 1), ("benzene", 1)]
    assert _readable_spellings(ordered, named, "pin", dict(
        general_fallback=True, allow_aromatic_general=True,
        general_fallback_unverified=True)) is named


# --------------------------------------------------------------------------
# A2 each component is named with the hang budgets of one compound
# --------------------------------------------------------------------------

def test_own_hang_budgets_gives_fresh_budgets_and_restores():
    assert fn.name_scope_depth() == 0
    with fn.own_hang_budgets():  # outside any name: a no-op
        assert getattr(fn._fragment_guard, "analysis_budget", None) is None
    fn.enter_name_scope()
    try:
        fn.spend_analysis_call(123)
        fn.spend_perf_work(1000)
        outer = (fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget,
                 fn._fragment_guard.work_budget)
        with fn.own_hang_budgets():
            assert fn._fragment_guard.analysis_budget == fn._ANALYSIS_CALL_BUDGET
            assert fn._fragment_guard.perf_budget == fn._PERF_BUDGET
            assert fn._fragment_guard.work_budget == fn._WORK_BUDGET
            fn.spend_analysis_call(400)
        assert (fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget,
                fn._fragment_guard.work_budget) == outer
        fn.disarm_hang_budgets()
        with fn.own_hang_budgets():  # a disarmed budget stays disarmed
            assert fn._fragment_guard.analysis_budget is None
    finally:
        fn.exit_name_scope()


# Four spiro macrocycles (296, 102, 65 and ~20 analysis calls): the fourth
# component tripped the shared 500-call budget and the drawing abstained.
MACROCYCLE_ROW = (
    "CC1=NC2=CC=CC=C2C3=C1O[C@@]4(CC3)C[C@H]5C(=O)N[C@@]6(C[C@H]6/C=C\\CCCCC[C@@H]"
    "(C(=O)N5C4)CC(=O)OC(C)(C)C)C(=O)NS(=O)(=O)C7(CC7)C."
    "CC1=NC2=CC=CC=C2C3=C1O[C@@]4(CC3)C[C@H]5C(=O)C[C@@]6(C[C@H]6/C=C\\CCCCC[C@H]"
    "(C(=O)N5C4)NC(=O)NC(C)(C)C)C(=O)NS(=O)(=O)C7(CC7)C."
    "CC1=NC2=CC=CC=C2C3=C1O[C@@]4(CC3)C[C@H]5C(=O)C[C@@]6(C[C@H]6/C=C\\CCCCC[C@H]"
    "(C(=O)N5C4)NC(=O)OC(C)(C)C)C(=O)NS(=O)(=O)C7(CC7)C."
    "CC(C)(C)OC(=O)N[C@H]1CCCCC/C=C\\[C@@H]2C[C@]2(NC(=O)[C@@H]3C[C@@]4(CCC5=C(O4)"
    "C(=NC6=CC=CC=C56)C7=CC(=CC=C7)F)CN3C1=O)C(=O)O")


def test_macrocycle_mixture_names_rt_exact():
    res = name_best_effort(MACROCYCLE_ROW)
    name = assert_full_rt(res.get("name"), MACROCYCLE_ROW)
    assert name.count(EM) == 3, name


# --------------------------------------------------------------------------
# A3 the oxo / ene valence check reads each adduct component on its own
# --------------------------------------------------------------------------

def test_oxo_ene_check_separates_adduct_components():
    from orthonym.perception.structure_conservation import oxo_ene_valence_illegal
    # each component is valid; joined into one word, the '2-ene' of the first and
    # the '2-oxo' of the second would be one carbon
    assert not oxo_ene_valence_illegal(
        f"bicyclo[2.2.2]oct-2-ene{EM}2-oxobicyclo[2.2.2]octane (1/1)")
    # an illegal component is still caught inside an adduct
    assert oxo_ene_valence_illegal(f"2-oxobicyclo[2.2.2]oct-2-ene{EM}methane (1/1)")


# --------------------------------------------------------------------------
# I isotope-labelled molecules
# --------------------------------------------------------------------------

ISOTOPE_ANALOGUES = {
    # I1: the labelled inner ring of a biphenyl-4-yl prefix
    "biphenyl_inner": ("[2H]c1c([2H])c(-c2ccccc2)c([2H])c([2H])c1-c1ccncc1",
                       "(2,3,5,6-2H4)[1,1'-biphenyl]-4-yl"),
    # I1: the labelled outer ring (primed locants)
    "biphenyl_outer": ("[2H]c1c([2H])c([2H])c(-c2ccc(-c3ccncc3)cc2)c([2H])c1[2H]",
                       "(2',3',4',5',6'-2H5)[1,1'-biphenyl]-4-yl"),
    # I3: a CD2 and the amide ND of one acetamide share one descriptor
    "acetamide_merge": ("[2H]C([2H])(c1ccccc1)C(=O)N([2H])c1ccccc1",
                        "(N,2,2-2H3)acetamide"),
}


@pytest.mark.parametrize("smiles,part", list(ISOTOPE_ANALOGUES.values()),
                         ids=list(ISOTOPE_ANALOGUES))
def test_isotope_analogues_name_rt_exact(smiles, part):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    assert part in name, name


def test_ring_assembly_and_bracketed_replacement_slots_are_offered():
    from orthonym.rules.isotopes import _insertion_offsets
    skel = "4-([1,1'-biphenyl]-4-yl)pyridine"
    assert skel.index("[") in _insertion_offsets(skel)
    skel2 = "6-[9,24-dioxahexacyclo[15.7.0.0^2,10.0^3,8.0^11,16.0^18,23]tetracosan-6-yl]-1,3,5-triazine"
    assert skel2.index("[") + 1 in _insertion_offsets(skel2)
    # a substituent's own locant after a bracket is not a replacement locant
    skel3 = "1-(2,4-dimethylphenyl)ethan-1-one"
    assert skel3.index("(") + 1 not in _insertion_offsets(skel3)


# PubChem row: 11 D on a bis-benzofuran-fused von Baeyer substituent of a triazine
DIOXA_ROW = ("[2H]C1=C(C(=C2C(=C1[2H])C3=C(C4=C2OC5=C4C(=C(C(=C5[2H])C6=NC(=NC(=N6)"
             "C7=CC8=C(C=C7)C9=CC=CC=C9C8(C)C)C1=CC=C(C=C1)C1=CC=CC=C1)[2H])[2H])"
             "OC1=C(C(=C(C(=C31)[2H])[2H])[2H])[2H])[2H])[2H]")


def test_bracketed_replacement_ring_row_names_rt_exact():
    res = name_best_effort(DIOXA_ROW)
    name = assert_full_rt(res.get("name"), DIOXA_ROW)
    # Every hydrogen position of the substituent is labelled: no locants
    #, the Blue Book, "Locants are omitted in compounds or
    # substituent groups in which all positions are completely isotopically
    # substituted"), and a hyphen before the part's own locant,:43718).
    assert "[(2H11)-9,24-dioxahexacyclo" in name, name


def test_isotope_round_trip_is_kekule_invariant_and_keeps_exchangeable_positions():
    from orthonym.rules.isotopes import _bond_order_free_key, _same_fixed_h_inchi
    # 1-methyl-10b,10c-dihydropyrene with its periphery in the two alternations
    # (the input's and OPSIN's): one compound, two canonical SMILES
    a = Chem.MolFromSmiles("CC1=C2C=CC3=CC=CC4=CC=C(C=C1)C2C34")
    b = Chem.MolFromSmiles("CC1=CC=C2C=CC3=CC=CC4=CC=C1C2C34")
    assert Chem.MolToInchiKey(a) == Chem.MolToInchiKey(b)
    assert Chem.MolToSmiles(a) != Chem.MolToSmiles(b)
    assert _same_fixed_h_inchi(a, b)
    assert _bond_order_free_key(a) == _bond_order_free_key(b)
    # D on the O or on the N of glycine: one standard InChIKey, two isotopomers
    c = Chem.MolFromSmiles("[2H]OC(=O)CN")
    d = Chem.MolFromSmiles("OC(=O)CN[2H]")
    assert Chem.MolToInchiKey(c) == Chem.MolToInchiKey(d)
    assert not _same_fixed_h_inchi(c, d)
    assert _bond_order_free_key(c) != _bond_order_free_key(d)


# PubChem row: a (2H5)phenyl on a skeleton whose '10b,10c-dihydropyren-1-yl'
# OPSIN writes in the other Kekule alternation
DIHYDROPYRENE_ROW = ("[2H]C1=C(C(=C(C(=C1[2H])[2H])C2=C3C4=C(C=CC5=C4C(=C(C=C5)C6=CC7="
                     "C(C=C6)C(C8=C7C=CC(=C8)C9=C1C=CC4=CC=CC5=CC=C(C1C54)C=C9)(C)C)O3)"
                     "C=C2)[2H])[2H]")


def test_dihydropyrene_row_names_rt_exact():
    res = name_best_effort(DIHYDROPYRENE_ROW)
    name = assert_full_rt(res.get("name"), DIHYDROPYRENE_ROW)
    # a completely labelled phenyl cites no locants,:44196)
    assert "2-(2H5)phenyl" in name and "dihydropyren" in name, name


# --------------------------------------------------------------------------
# Review fix: the isotope round trip never accepts a bond-shift isomer
# --------------------------------------------------------------------------
# The fixed-H InChI compare of I2 records no bond orders, so it also accepted
# OPSIN's parse of a labelled ring whose alternations are two compounds:
# '1-methyl(2-2H)cycloocta-1,3,5,7-tetraene' (D on the carbon double-bonded to
# C1) for the input's D on the carbon single-bonded to C1. It now accepts a
# differing parse only when every alternating circuit has 4n+2 atoms (the
# dihydropyrene periphery, 14); a 4n circuit (cyclooctatetraene 8, [12]- and
# [16]annulene, the pentalene and heptalene peripheries 8 and 12) is two
# bond-shift isomers. The full InChIKey cannot see the difference, so each name
# below is read back by an independent OPSIN call to the input's canonical
# SMILES as well.

def _reads_back_to_this_isomer(name, smiles):
    opsin = _independent_parse(name) if name else None
    got = Chem.MolFromSmiles(opsin) if opsin else None
    want = Chem.MolFromSmiles(smiles)
    return (got is not None
            and Chem.MolToInchiKey(got) == Chem.MolToInchiKey(want)
            and Chem.MolToSmiles(got) == Chem.MolToSmiles(want))


BOND_SHIFT_WITNESSES = {
    # the reviewer's three cyclooctatetraenes
    "methyl_cot": ("CC1=CC=CC=CC=C1[2H]", "1-methyl(8-2H)cycloocta-1,3,5,7-tetraene"),
    "d2_cot": ("[2H]C1=CC=CC=CC=C1[2H]", "(1,8-2H2)cycloocta-1,3,5,7-tetraene"),
    "cot_ethanol": ("C[C@H](O)C1=CC=CC=CC=C1[2H]",
                    "(1S)-1-[(8-2H)cycloocta-1,3,5,7-tetraen-1-yl]ethan-1-ol"),
    # [4n]annulenes
    "annulene12": ("[2H]C1=CC=CC=CC=CC=CC=C1C",
                   "1-methyl(12-2H)cyclododeca-1,3,5,7,9,11-hexaene"),
    "annulene16": ("[2H]C1=CC=CC=CC=CC=CC=CC=CC=C1C",
                   "1-methyl(16-2H)cyclohexadeca-1,3,5,7,9,11,13,15-octaene"),
    # the other bond-shift isomer of the first witness keeps its name
    "methyl_cot_other": ("CC1=C(C=CC=CC=C1)[2H]",
                         "1-methyl(2-2H)cycloocta-1,3,5,7-tetraene"),
}


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
@pytest.mark.parametrize("smiles,expected", list(BOND_SHIFT_WITNESSES.values()),
                         ids=list(BOND_SHIFT_WITNESSES))
def test_bond_shift_witness_gets_its_own_isomer_name(smiles, expected, tier):
    from orthonym import Orthonym
    res = (name_best_effort(smiles) if tier == "best-effort"
           else Orthonym(style="pin").name_tiered(smiles))
    name = res.get("name")
    assert name == expected, (tier, name)
    assert _reads_back_to_this_isomer(name, smiles), name
    assert res.get("tier") == "pin_verified", res.get("tier")


# Two bond-shift isomers of a mancude fusion parent with a 4n periphery: each
# form is named, reads back to itself, and the two forms get different names.
MANCUDE_4N_PAIRS = {
    "pentalene": ("[2H]C1=C2C=CC=C2C=C1", "[2H]C1=CC=C2C=CC=C12"),
    "heptalene_a": ("[2H]C1=CC2=CC=CC=CC2=CC=C1", "[2H]C1=CC=CC2=CC=CC=CC2=C1"),
    "heptalene_b": ("[2H]C1=C2C=CC=CC=C2C=CC=C1", "[2H]C1=CC=CC=C2C=CC=CC=C12"),
}


@pytest.mark.parametrize("first,second", list(MANCUDE_4N_PAIRS.values()),
                         ids=list(MANCUDE_4N_PAIRS))
def test_mancude_4n_bond_shift_isomers_get_different_names(first, second):
    names = []
    for smi in (first, second):
        name = name_best_effort(smi).get("name")
        assert _reads_back_to_this_isomer(name, smi), (smi, name)
        names.append(name)
    assert names[0] != names[1], names


def test_isotope_round_trip_refuses_a_bond_shift_isomer():
    from orthonym.rules.isotopes import (
        _isotope_round_trips,
        _kekule_forms_of_one_compound,
    )
    cot = Chem.MolFromSmiles("CC1=CC=CC=CC=C1[2H]")
    assert not _isotope_round_trips("1-methyl(2-2H1)cycloocta-1,3,5,7-tetraene", cot)
    assert _isotope_round_trips("1-methyl(8-2H1)cycloocta-1,3,5,7-tetraene", cot)
    cot2 = Chem.MolFromSmiles("[2H]C1=CC=CC=CC=C1[2H]")
    assert not _isotope_round_trips("(1,2-2H2)cycloocta-1,3,5,7-tetraene", cot2)
    assert _isotope_round_trips("(1,8-2H2)cycloocta-1,3,5,7-tetraene", cot2)
    # the stereo-blind pass refuses it too
    assert not _isotope_round_trips("1-methyl(2-2H1)cycloocta-1,3,5,7-tetraene", cot,
                                    stereo_blind=True)
    # the 14-atom dihydropyrene periphery: one compound in two alternations
    a = Chem.MolFromSmiles("CC1=C2C=CC3=CC=CC4=CC=C(C=C1)C2C34")
    b = Chem.MolFromSmiles("CC1=CC=C2C=CC3=CC=CC4=CC=C1C2C34")
    assert _kekule_forms_of_one_compound(a, b)
    # an 8-atom pentalene periphery and a cyclooctatetraene: two compounds
    for x, y in (("[2H]C1=C2C=CC=C2C=C1", "[2H]C1=CC=C2C=CC=C12"),
                 ("CC1=CC=CC=CC=C1[2H]", "CC1=C(C=CC=CC=C1)[2H]")):
        assert not _kekule_forms_of_one_compound(Chem.MolFromSmiles(x),
                                                 Chem.MolFromSmiles(y))
    # an aromatic ring in two Kekule forms is one compound
    assert _kekule_forms_of_one_compound(Chem.MolFromSmiles("[2H]c1ccccc1"),
                                         Chem.MolFromSmiles("[2H]C1=CC=CC=C1"))


# --------------------------------------------------------------------------
# B speed
# --------------------------------------------------------------------------

def test_naming_pass_cap_counts_and_restores():
    fn.enter_name_scope()
    try:
        assert fn.naming_passes() == 0
        fn.count_naming_pass()
        with fn.naming_pass_cap(2):
            fn.count_naming_pass()
            fn.count_naming_pass()
            with pytest.raises(fn.OptionalNamingCapExceeded):
                fn.count_naming_pass()
        fn.count_naming_pass()  # the cap is gone after the body
        with fn.naming_pass_cap(5):
            with fn.naming_pass_cap(1):  # the tighter cap wins
                fn.count_naming_pass()
                with pytest.raises(fn.OptionalNamingCapExceeded):
                    fn.count_naming_pass()
    finally:
        fn.exit_name_scope()
    assert not issubclass(fn.OptionalNamingCapExceeded, Exception)


# PubChem row: the clean-first offer re-derived this peptide from scratch
# (1,519 naming passes against the main path's 5, 82 s); main 2b1455817 names it
# in 4 s with the same string.
OFFER_ROW = ("C1=CC(=CC=C1C[C@H](CC(=O)CCC(=O)O)C(=O)N[C@@H](CCCCNC(=O)[C@H](CC2=CC=C"
             "(C=C2)O)NC(=O)CCC(=O)O)C(=O)N[C@@H](CCCCNC(=O)[C@H](CCCCNC(=O)[C@H](CC3=CC="
             "C(C=C3)O)NC(=O)CCC(=O)O)NC(=O)[C@H](CC4=CC=C(C=C4)O)NC(=O)CCC(=O)O)C(=O)N"
             "[C@@H](CCCCNC(=O)[C@H](CCCCNC(=O)[C@H](CCCCNC(=O)[C@H](CC5=CC=C(C=C5)O)NC"
             "(=O)CCC(=O)O)NC(=O)[C@H](CC6=CC=C(C=C6)O)NC(=O)CCC(=O)O)NC(=O)[C@H](CCCCNC"
             "(=O)[C@H](CC7=CC=C(C=C7)O)NC(=O)CCC(=O)O)NC(=O)[C@H](CC8=CC=C(C=C8)O)NC(=O)"
             "CCC(=O)O)C(=O)NCCS)O")


def test_clean_first_offer_is_capped_at_the_main_path_passes(monkeypatch):
    peak = [0]
    orig = fn.count_naming_pass

    def spy():
        orig()
        peak[0] = max(peak[0], fn.naming_passes())

    monkeypatch.setattr(fn, "count_naming_pass", spy)
    res = name_best_effort(OFFER_ROW)
    assert_full_rt(res.get("name"), OFFER_ROW)
    assert peak[0] < 100, peak[0]


def test_single_descriptor_sweep_skips_locants_no_position_can_bear(monkeypatch):
    """Four D on four aromatic CH: no one position bears four D, so no locanted
    single descriptor can round-trip; only the locant-free form is tried."""
    import orthonym.rules.isotopes as iso
    calls = []
    orig = iso._isotope_round_trips

    def spy(cand, original, *a, **k):
        calls.append(cand)
        return orig(cand, original, *a, **k)

    monkeypatch.setattr(iso, "_isotope_round_trips", spy)
    original = Chem.MolFromSmiles("[2H]c1c([2H])c([2H])c(C(=O)O)c([2H])c1")
    keys = [((2, "H"), 4, 1)]
    skel = "benzoic acid"
    cand, off, desc, rank = iso._find_best_placement(skel, keys, original, 9)
    offsets = len({0} | set(iso._insertion_offsets(skel)))
    assert len(calls) <= 2 * offsets  # the locant-free form only (two subscript forms)
    assert all("-2H" not in c for c in calls), calls


# dev2000 row (not a holdout split): its main path makes 5 naming passes and the clean
# recovery 52; the cap is never below _OFFER_MIN_PASSES, so the offer still
# replaces the floor stand-in with the engine's name.
SMALL_OFFER_ROW = "CNC(=O)[C@@H](NC(=O)[C@H](CCCc1ccccc1)[C@H](C)N(O)C=O)C(C)(C)C"


def test_offer_cap_leaves_a_small_molecule_offer_alone():
    res = name_best_effort(SMALL_OFFER_ROW)
    name = assert_full_rt(res.get("name"), SMALL_OFFER_ROW)
    assert name.endswith("benzene") and "cyclohexa" not in name, name
