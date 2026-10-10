"""A split whose assembler cannot prove its name bounds the decomposition searches of the molecule.

The ester assembler proves its group word by a round trip and returns ``None`` when no candidate
round-trips (``fragment_assembly._assemble_ester``); the carbamate assembler declines a name that
yields no monovalent group word. Before the ester assembler proved its names it returned the
string rule's name, and the search in ``engine.try_decompose`` took it as it takes any assembled
name: a split whose name passes the coverage and quality tests ENDS the search, and the read-back
downstream refused the name. The search after a ``None`` is a far larger one (every other bond is
cleaved, each remainder of up to 120 atoms is named in full, and again in every remainder) and it
spent a molecule's analysis-call budget: four 70-130-atom molecules the general pipeline names in
10-30 s were abstained (sample split, valid / complete tiers) or ran 64 s (best-effort).

The assemblers now report that they built a name they could not prove (``outcome['unproven']``,
never returned). The engine does not name a repeat of that split again, and once the molecule has
met one, its searches spend only a share of the analysis-call budget on other bonds, counted from
the start of the first unproven split (``engine._exploration_exhausted``). Where the other bonds
are cheap the search is as it was: a diacylglycerophosphoglycerol finds the ester name of its
other ester.
"""
from unittest.mock import patch

import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.decomposition import engine
from orthonym.decomposition import fragment_assembly as fa
from orthonym.decomposition.engine import _try_single_bond_decompose, try_decompose
from orthonym.decomposition.fragment_assembly import assemble_fragment_name

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_floor_between_tests():
    """The tests below arm ``analysis_budget`` by hand, outside ``enter_name_scope``, which is
    what clears the floor of the unproven-split search (``fragment_naming._fragment_guard.
    unproven_floor``, a level of that budget): clear it around every test."""
    fn._fragment_guard.unproven_floor = None
    yield
    fn._fragment_guard.unproven_floor = None


# --- the assemblers report the name they could not prove ------------------------------------

def test_ester_with_no_group_word_reports_the_string_rule_name():
    # A multiplied '-ol' is not one group word and no structure is given to build one from.
    out = {}
    assert assemble_fragment_name(
        "ester", {"acid": "acetic acid", "alkyl": "ethane-1,2-diol"}, outcome=out) is None
    assert out == {"unproven": "ethane-1,2-diol acetate"}


def test_ester_proven_name_reports_nothing():
    out = {}
    assert assemble_fragment_name(
        "ester", {"acid": "acetic acid", "alkyl": "ethanol"}, outcome=out) == "ethyl acetate"
    assert out == {}


def test_ester_whose_candidates_all_fail_the_round_trip_reports_the_string_rule_name():
    # '2-hydroxyethyl' is built from the structure and offered; the proof against the parent fails
    # for it (patched), so nothing is returned and the string rule's name is reported.
    out = {}
    with patch.object(fa, "_name_round_trips", return_value=False):
        got = assemble_fragment_name(
            "ester", {"acid": "acetic acid", "alkyl": "ethane-1,2-diol"},
            fragment_smiles={"acid": "CC(=O)O", "alkyl": "OCCO"},
            parent_smiles="CC(=O)OCCO", outcome=out)
    assert got is None and out == {"unproven": "ethane-1,2-diol acetate"}
    # the same call with the proof passing ships the group word built from the structure
    out = {}
    got = assemble_fragment_name(
        "ester", {"acid": "acetic acid", "alkyl": "ethane-1,2-diol"},
        fragment_smiles={"acid": "CC(=O)O", "alkyl": "OCCO"},
        parent_smiles="CC(=O)OCCO", outcome=out)
    assert got == "2-hydroxyethyl acetate" and out == {}


def test_carbamate_with_no_group_word_reports_the_string_rule_name():
    out = {}
    assert assemble_fragment_name(
        "carbamate", {"alkyl": "ethane-1,2-diol", "amine": "methylamine"}, outcome=out) is None
    assert out == {"unproven": "ethane-1,2-diol methylcarbamate"}
    out = {}
    assert assemble_fragment_name(
        "carbamate", {"alkyl": "ethanol"}, outcome=out) == "ethyl carbamate"
    assert out == {}


def test_assemblers_work_without_an_outcome_mapping():
    assert assemble_fragment_name(
        "ester", {"acid": "acetic acid", "alkyl": "ethane-1,2-diol"}) is None
    assert assemble_fragment_name("carbamate", {"alkyl": "ethane-1,2-diol"}) is None


# --- the engine does not name a repeat of an unproven split again ---------------------------

def _first_bond(smiles):
    from orthonym.decomposition.bond_cleavage import find_cleavable_bonds
    mol = Chem.MolFromSmiles(smiles)
    return mol, find_cleavable_bonds(mol)[0]


def _unproven_assembler(calls):
    def assembler(bond_type, fragment_names, style="pin", fragment_smiles=None,
                  parent_smiles=None, outcome=None):
        calls.append(bond_type)
        outcome["unproven"] = "ethane-1,2-diol acetate"
        return None
    return assembler


def test_unproven_split_is_not_named_again():
    mol, bond = _first_bond("CC(=O)OCC")
    calls = []
    token = memo.push_scope()
    try:
        with patch("orthonym.decomposition.fragment_assembly.assemble_fragment_name",
                   _unproven_assembler(calls)):
            assert _try_single_bond_decompose(mol, bond) is None
            assert _try_single_bond_decompose(mol, bond) is None
    finally:
        memo.pop_scope(token)
    assert calls == ["ester"]  # the second ask did not name the fragments again


def test_unproven_split_is_named_again_outside_a_naming_scope():
    mol, bond = _first_bond("CC(=O)OCC")
    calls = []
    with patch("orthonym.decomposition.fragment_assembly.assemble_fragment_name",
               _unproven_assembler(calls)):
        _try_single_bond_decompose(mol, bond)
        _try_single_bond_decompose(mol, bond)
    assert calls == ["ester", "ester"]  # no scope, no memo: a direct call keeps its behaviour


def test_proven_split_is_not_remembered():
    mol, bond = _first_bond("CC(=O)OCC")
    calls = []

    def proven(bond_type, fragment_names, style="pin", fragment_smiles=None,
               parent_smiles=None, outcome=None):
        calls.append(bond_type)
        return "ethyl acetate"

    token = memo.push_scope()
    try:
        with patch("orthonym.decomposition.fragment_assembly.assemble_fragment_name", proven):
            assert _try_single_bond_decompose(mol, bond) == "ethyl acetate"
            assert _try_single_bond_decompose(mol, bond) == "ethyl acetate"
    finally:
        memo.pop_scope(token)
    assert calls == ["ester", "ester"]


# --- the search that follows an unproven split is bounded -----------------------------------

_GOOD = "propane-1,2,3-triyl trihexadecanoate"


def _search(spend_on_first, spend_on_second, third_name=_GOOD):
    """``try_decompose`` on a molecule with three bonds: the first two are unproven splits (they
    spend ``spend_on_first`` and ``spend_on_second`` analysis calls), the third would give
    ``third_name``. Returns (result, bonds attempted)."""
    mol = Chem.MolFromSmiles("C" * 40)
    bonds = [{"bond_idx": i, "type": "ester"} for i in range(3)]
    attempted = []

    def single(m, bond, style="pin"):
        attempted.append(bond["bond_idx"])
        before = fn._fragment_guard.analysis_budget
        if bond["bond_idx"] < 2:
            fn.spend_analysis_call(spend_on_first if bond["bond_idx"] == 0 else spend_on_second)
            engine._note_unproven_split(before)
            return None
        return third_name

    token = memo.push_scope()
    saved = getattr(fn._fragment_guard, "analysis_budget", None)
    fn._fragment_guard.analysis_budget = 500
    try:
        with patch.object(fn, "_ANALYSIS_CALL_BUDGET", 500), \
                patch.object(engine, "_UNPROVEN_EXPLORATION_SHARE", 16), \
                patch("orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
                      return_value=bonds), \
                patch("orthonym.namer.name_pipeline_only", return_value=None), \
                patch("orthonym.decomposition.engine._try_single_bond_decompose",
                      side_effect=single), \
                patch("orthonym.decomposition.engine._name_quality_is_acceptable",
                      return_value=True), \
                patch("orthonym.decomposition.engine._select_best_bond", return_value=bonds[0]), \
                patch("orthonym.decomposition.engine._try_multi_bond_decompose",
                      return_value=None), \
                patch("orthonym.decomposition.engine._try_iterative_mixed_decompose",
                      return_value=None), \
                patch("orthonym.decomposition.weave.try_weave", return_value=None):
            return try_decompose(mol), attempted
    finally:
        fn._fragment_guard.analysis_budget = saved
        memo.pop_scope(token)


def test_a_cheap_search_after_an_unproven_split_is_the_search_it_was():
    # 3 + 10 calls of the 31 the share allows: the third bond is opened and gives its name
    result, attempted = _search(spend_on_first=3, spend_on_second=10)
    assert attempted == [0, 1, 2] and result == _GOOD


def test_a_search_after_an_unproven_split_ends_when_its_share_is_spent():
    # 3 + 40 calls > 31: the third bond is not opened, the molecule goes to the general pipeline
    result, attempted = _search(spend_on_first=3, spend_on_second=40)
    assert attempted == [0, 1] and result is None


def test_the_share_includes_the_cost_of_the_first_unproven_split():
    # naming the first split's remainder cost 40 > 31 calls: no other bond is opened
    result, attempted = _search(spend_on_first=40, spend_on_second=0)
    assert attempted == [0] and result is None


def test_the_share_binds_every_search_of_the_molecule_once_it_has_met_an_unproven_split():
    # the first split of the molecule was unproven and cost 40 > 31 calls; this search, of one
    # of the molecule's remainders, met no unproven split of its own and opens no other bond
    result, attempted = _search_without_unproven_split(first_split_cost=40)
    assert attempted == [0] and result is None


def test_a_molecule_that_met_no_unproven_split_has_no_share():
    result, attempted = _search_without_unproven_split(first_split_cost=None)
    assert attempted == [0, 1, 2] and result == _GOOD


def _search_without_unproven_split(first_split_cost, phases=None):
    """``try_decompose`` on three bonds (two esters and an amide, 40 atoms) none of whose splits
    is unproven, in a molecule whose first unproven split (begun at 500 calls left) has cost
    ``first_split_cost`` (None: it met none). ``phases``, a list, receives the names of the
    multi-bond phases that were opened. Returns (result, bonds attempted)."""
    mol = Chem.MolFromSmiles("C" * 40)
    bonds = [{"bond_idx": 0, "type": "ester"}, {"bond_idx": 1, "type": "amide"},
             {"bond_idx": 2, "type": "ester"}]
    attempted = []

    def single(m, bond, style="pin"):
        attempted.append(bond["bond_idx"])
        return _GOOD if bond["bond_idx"] == 2 else None

    def phase(name):
        def run(*args, **kwargs):
            if phases is not None:
                phases.append(name)
            return None
        return run

    token = memo.push_scope()
    saved = getattr(fn._fragment_guard, "analysis_budget", None)
    fn._fragment_guard.analysis_budget = 500
    try:
        with patch.object(fn, "_ANALYSIS_CALL_BUDGET", 500), \
                patch.object(engine, "_UNPROVEN_EXPLORATION_SHARE", 16), \
                patch("orthonym.decomposition.bond_cleavage.find_cleavable_bonds",
                      return_value=bonds), \
                patch("orthonym.namer.name_pipeline_only", return_value=None), \
                patch("orthonym.decomposition.engine._try_single_bond_decompose",
                      side_effect=single), \
                patch("orthonym.decomposition.engine._name_quality_is_acceptable",
                      side_effect=lambda name, m: name == _GOOD), \
                patch("orthonym.decomposition.engine._select_best_bond", return_value=bonds[0]), \
                patch("orthonym.decomposition.engine._try_iterative_mixed_decompose",
                      side_effect=phase("iterative")), \
                patch("orthonym.decomposition.engine._try_multi_bond_decompose",
                      side_effect=phase("multi-bond")), \
                patch("orthonym.decomposition.weave.try_weave", return_value=None):
            if first_split_cost is not None:
                engine._note_unproven_split(500)
                fn.spend_analysis_call(first_split_cost)
            result = try_decompose(mol)
    finally:
        fn._fragment_guard.analysis_budget = saved
        memo.pop_scope(token)
    return result, attempted


def test_the_other_phases_of_the_search_are_not_opened_once_the_share_is_spent():
    # the best bond gives nothing; the iterative and the same-type multi-bond phases follow it
    phases = []
    result, attempted = _search_without_unproven_split(first_split_cost=None, phases=phases)
    assert attempted == [0, 1, 2] and result == _GOOD  # the retry finds the third bond first
    spent = []
    result, attempted = _search_without_unproven_split(first_split_cost=40, phases=spent)
    assert attempted == [0] and result is None and spent == []
    # with the retry loop out of the way, the phases of a molecule with no share are opened
    bonds_phases = []
    with patch("orthonym.decomposition.engine.MAX_BOND_RETRY_ATTEMPTS", 1):
        result, attempted = _search_without_unproven_split(
            first_split_cost=None, phases=bonds_phases)
    assert attempted == [0] and bonds_phases == ["iterative", "multi-bond"]
    bonds_phases = []
    with patch("orthonym.decomposition.engine.MAX_BOND_RETRY_ATTEMPTS", 1):
        _search_without_unproven_split(first_split_cost=40, phases=bonds_phases)
    assert bonds_phases == []


def test_the_engine_notes_an_unproven_split_it_met():
    mol, bond = _first_bond("CC(=O)OCC")

    def assembler(cost):
        def run(bond_type, fragment_names, style="pin", fragment_smiles=None,
                parent_smiles=None, outcome=None):
            fn.spend_analysis_call(cost)
            outcome["unproven"] = "ethane-1,2-diol acetate"
            return None
        return run

    for cost, exhausted in ((40, True), (5, False)):
        token = memo.push_scope()
        saved = getattr(fn._fragment_guard, "analysis_budget", None)
        fn._fragment_guard.analysis_budget = 500
        try:
            with patch.object(fn, "_ANALYSIS_CALL_BUDGET", 500), \
                    patch.object(engine, "_UNPROVEN_EXPLORATION_SHARE", 16), \
                    patch("orthonym.decomposition.fragment_assembly.assemble_fragment_name",
                          assembler(cost)):
                assert engine._exploration_exhausted() is False  # no unproven split yet
                _try_single_bond_decompose(mol, bond)
                assert engine._exploration_exhausted() is exhausted
        finally:
            fn._fragment_guard.analysis_budget = saved
            memo.pop_scope(token)


def test_no_share_without_an_armed_budget_or_a_naming_scope():
    assert engine._exploration_exhausted() is False
    token = memo.push_scope()
    saved = getattr(fn._fragment_guard, "analysis_budget", None)
    fn._fragment_guard.analysis_budget = None  # unarmed
    try:
        engine._note_unproven_split(None)
        assert engine._exploration_exhausted() is False
    finally:
        fn._fragment_guard.analysis_budget = saved
        memo.pop_scope(token)
