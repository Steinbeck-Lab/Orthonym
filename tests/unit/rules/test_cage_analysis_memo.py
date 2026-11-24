"""The von Baeyer cage analysis is computed once per cage and context in a naming call (``rules.cage_analysis_memo``).

A hit replays what the computation did besides its result: the perf and analysis budget units it charged and the
``record_non_pin_fragment`` calls it made (a name on a cage descriptor beyond the PIN caps stays below the PIN). A
computation with any other side effect is not stored, an RWMol is never memoised, and a hit hands out a copy.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.metrics import provenance as pv
from orthonym.rules import cage_analysis_memo as cam

pytestmark = pytest.mark.unit

ADAMANTANE = "C1C2CC3CC1CC(C2)C3"


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.setattr(memo, "_MODE", "on")
    monkeypatch.setattr(fn, "_PERF_MEASURE", False)
    monkeypatch.setattr(fn._fragment_guard, "perf_budget", 10 ** 9, raising=False)
    monkeypatch.setattr(fn._fragment_guard, "analysis_budget", 10 ** 6, raising=False)
    token = memo.push_scope()
    yield
    memo.pop_scope(token)


def _counted(body):
    calls = []

    def analyze(mol, cage_atoms=None, allow_mancude=False, spiro_atom=None, substituent_atoms=None):
        calls.append(1)
        return body()
    return cam.memoised_cage_analysis(analyze), calls


def test_a_repeat_is_computed_once_and_handed_out_as_a_copy(scope):
    wrapped, calls = _counted(lambda: {"descriptor": "tricyclo[3.3.1.1^3,7]decane"})
    mol = Chem.MolFromSmiles(ADAMANTANE)
    first = wrapped(mol)
    first["descriptor"] = "edited by the caller"
    second = wrapped(Chem.MolFromSmiles(ADAMANTANE))      # the same graph, a new object
    assert len(calls) == 1 and second == {"descriptor": "tricyclo[3.3.1.1^3,7]decane"}
    assert wrapped(mol, allow_mancude=True) and len(calls) == 2      # another argument, another entry


def test_a_hit_makes_the_non_pin_records_again(scope):
    def body():
        pv.record_non_pin_fragment("tricyclo[1.1.1.1^1,3]")
        return {"descriptor": "x"}
    wrapped, calls = _counted(body)
    mol = Chem.MolFromSmiles(ADAMANTANE)
    wrapped(mol)
    log = pv.push_non_pin_log()
    try:
        wrapped(mol)
    finally:
        pv.pop_non_pin_log()
    assert len(calls) == 1 and log == ["tricyclo[1.1.1.1^1,3]"]


def test_a_hit_charges_the_budget_units_again(scope, monkeypatch):
    def body():
        fn.spend_perf_work(7)
        fn.spend_analysis_call(2)
        return {"descriptor": "x"}
    wrapped, calls = _counted(body)
    monkeypatch.setattr(fn._fragment_guard, "perf_budget", 1000, raising=False)
    monkeypatch.setattr(fn._fragment_guard, "analysis_budget", 100, raising=False)
    mol = Chem.MolFromSmiles(ADAMANTANE)
    wrapped(mol)
    wrapped(mol)
    assert len(calls) == 1
    assert (fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget) == (1000 - 14, 100 - 4)


def test_a_computation_with_another_side_effect_is_not_stored(scope):
    def body():
        pv.note_touched("general_ring_prefix")
        return {"descriptor": "x"}
    wrapped, calls = _counted(body)
    mol = Chem.MolFromSmiles(ADAMANTANE)
    wrapped(mol)
    wrapped(mol)
    assert len(calls) == 2


def test_no_memo_for_an_rwmol_or_outside_a_scope(scope):
    wrapped, calls = _counted(lambda: {"descriptor": "x"})
    rw = Chem.RWMol(Chem.MolFromSmiles(ADAMANTANE))
    wrapped(rw)
    wrapped(rw)
    assert len(calls) == 2


def test_no_memo_without_an_armed_budget(scope, monkeypatch):
    """With no budget armed the units a hit must charge are unknown, so nothing is stored."""
    monkeypatch.setattr(fn._fragment_guard, "perf_budget", None, raising=False)
    wrapped, calls = _counted(lambda: {"descriptor": "x"})
    mol = Chem.MolFromSmiles(ADAMANTANE)
    wrapped(mol)
    wrapped(mol)
    assert len(calls) == 2


def test_outside_a_scope_every_ask_computes():
    wrapped, calls = _counted(lambda: {"descriptor": "x"})
    mol = Chem.MolFromSmiles(ADAMANTANE)
    wrapped(mol)
    wrapped(mol)
    assert len(calls) == 2


def test_the_cage_analysis_is_the_memoised_one():
    from orthonym.rules import vonbaeyer_universal as vbu
    assert vbu.analyze_cage_universal.__wrapped__.__name__ == "analyze_cage_universal"


_SPELLING_PROBE = r"""
import contextlib, json, sys
from rdkit import Chem
from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from orthonym.assembly.book_prefixes import mechanical_forms
from orthonym.rules import cage_analysis_memo as cam
from orthonym.rules.vonbaeyer_universal import analyze_cage_universal
mol = Chem.MolFromSmiles("[O-][N+](=O)C1C(CCCCCCCCC)C2CCC1C2")
ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
def numbering(mechanical):
    with mechanical_forms() if mechanical else contextlib.nullcontext():
        return sorted(analyze_cage_universal(mol, cage_atoms=ring, allow_mancude=True).atom_to_locant.items())
memo._MODE = "off"
plain = {m: numbering(m) for m in (False, True)}
memo._MODE = "on"
fn._fragment_guard.perf_budget, fn._fragment_guard.analysis_budget = 10 ** 9, 10 ** 6
out = {"differ": plain[False] != plain[True], "runs": []}
for order in ((False, False, True), (True, True, False)):
    token = memo.push_scope()
    try:
        got = [numbering(m) for m in order]
        stored = sum(1 for k in memo.current_scope() if isinstance(k, tuple) and k and k[0] == cam._NS)
    finally:
        memo.pop_scope(token)
    out["runs"].append({"stored": stored, "same": got == [plain[m] for m in order]})
print(json.dumps(out))
"""


@pytest.mark.opsin_gate
def test_a_book_spelling_numbering_is_never_served_to_a_mechanical_one():
    """The nested sort-key name of the nitro group is 'nitroyl' in book spellings and a mechanical form otherwise, so
    the (g) order against 'nonyl', and with it the numbering, differs between the two runs: the key carries the
    run tags of ``memo._ck`` (review of lever K1: without them the third ask got the other run's numbering). Run in
    a child process: inside the test session the nested name is the plain 'nitro' in both runs."""
    import json
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-c", _SPELLING_PROBE], capture_output=True, text=True, timeout=600)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not out["differ"]:
        pytest.skip("the two spellings give one numbering here (reduced mode)")
    for run in out["runs"]:
        assert run["stored"] >= 1 and run["same"], out
