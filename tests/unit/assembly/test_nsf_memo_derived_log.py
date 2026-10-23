"""``substituent_naming.name_substituent_fragment`` stores a call that ran a derived non-PIN check.

``metrics.provenance.record_derived_non_pin_fragment`` reads the record of non-PIN fragments to decide
whether it records the derived form (an acid name turned into its acyl prefix). Nothing it returns
depends on that read, so a call that made it is stored with its record calls, the derived call as a typed
log item: a hit makes the call again and the record it finds decides again. Every other read of the record
still keeps a call out of the memo, and so does a log longer than ``NON_PIN_LOG_CAP`` after compaction.
The records a hit leaves are compared with the records of the same sequence of asks with the memo off.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import memo, substituent_naming as sn
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit

_DEPTH = 12


def _mol(n=_DEPTH + 2):
    return Chem.MolFromSmiles("C" * n)


class _Run:
    """Ask for fragments of one molecule in one memo scope with a fake naming; count the namings made."""

    def __init__(self, monkeypatch, body, mode="on"):
        self.uncached = 0
        self.mol = _mol()
        monkeypatch.setattr(memo, "_MODE", mode)

        def fake(mol, sub_atoms, attach_idx, parent_chain):
            self.uncached += 1
            return body(self, mol, attach_idx)
        monkeypatch.setattr(sn, "_name_substituent_fragment_uncached", fake)
        pv.clear_provenance()
        self._token = memo.push_scope()
        self._closed = False

    def ask(self, level=0, mol=None):
        mol = self.mol if mol is None else mol
        return sn.name_substituent_fragment(mol, list(range(level, _DEPTH + 2)), level, [])

    def record(self):
        return pv.get_provenance()["non_pin_fragments"]

    def close(self):
        if not self._closed:
            self._closed = True
            memo.pop_scope(self._token)
            pv.clear_provenance()


@pytest.fixture
def run(monkeypatch):
    runs = []

    def make(body, mode="on"):
        for earlier in runs:                  # one memo scope is open at a time: a second run would share its cache
            earlier.close()
        r = _Run(monkeypatch, body, mode)
        runs.append(r)
        return r
    yield make
    for r in reversed(runs):
        r.close()


def _derived_body(run_, mol, level):
    pv.record_non_pin_fragment("acid")
    pv.record_derived_non_pin_fragment("acid chloride", "acyl")
    return "name"


def test_a_call_that_ran_a_derived_check_is_stored_and_a_hit_repeats_its_records(run):
    r = run(_derived_body)
    assert r.ask() == "name"
    assert r.uncached == 1 and r.record() == ("acid", "acyl")
    pv.clear_provenance()
    assert r.ask() == "name"
    assert r.uncached == 1                      # a hit: the naming did not run again
    assert r.record() == ("acid", "acyl")       # and it made the records again


def test_a_hit_decides_the_derived_check_again_on_the_record_it_finds(run, monkeypatch):
    def body(run_, mol, level):
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")     # the fragment is recorded by the caller
        return "name"

    def sequence(mode):
        r = run(body, mode)
        out = []
        pv.record_non_pin_fragment("acid")
        r.ask(); out.append(r.record())          # fires: the record holds 'acid'
        pv.clear_provenance()
        r.ask(); out.append(r.record())          # no fragment of the source: nothing is recorded
        pv.record_non_pin_fragment("chloride")
        r.ask(); out.append(r.record())          # a fragment of the source again
        return out, r.uncached
    on, calls_on = sequence("on")
    off, calls_off = sequence("off")
    assert on == off == [("acid", "acyl"), (), ("chloride", "acyl")]
    assert calls_on == 1 and calls_off == 3


def test_the_reviewers_counterexample_at_the_memo(run):
    # the derived call finds no fragment of its source the first time; a record made after it, in the same
    # call, is a fragment of the source, and the repeat of the call records the derived form
    def body(run_, mol, level):
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        pv.record_non_pin_fragment("chloride")
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        return "name"

    def sequence(mode):
        r = run(body, mode)
        r.ask()
        first = r.record()
        pv.clear_provenance()
        r.ask()
        return first, r.record(), r.uncached
    on = sequence("on")
    off = sequence("off")
    assert on[:2] == off[:2] == (("chloride", "acyl"), ("chloride", "acyl"))
    assert on[2] == 1 and off[2] == 2


def test_a_call_that_reads_the_record_otherwise_is_not_stored(run):
    def body(run_, mol, level):
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        pv.get_provenance()
        return "name"
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 2


@pytest.mark.parametrize("read", [lambda: pv.name_carries_non_pin_part({}, "x"),
                                  lambda: pv.non_pin_label_rule_failures({}, "x"),
                                  lambda: pv.verified_field("opsin", {}, "x")])
def test_each_other_read_keeps_a_call_out_of_the_memo(run, read):
    def body(run_, mol, level):
        read()
        return "name"
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 2


def test_a_call_that_wrote_another_provenance_variable_is_not_stored(run):
    def body(run_, mol, level):
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        pv.record_non_pin_label("label")
        return "name"
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 2


def test_a_log_over_the_cap_is_not_stored_and_one_at_the_cap_is(run, monkeypatch):
    def body(run_, mol, level):
        for k in range(3):
            pv.record_derived_non_pin_fragment("acid x%d" % k, "form%d" % k)
        return "name"
    monkeypatch.setattr(pv, "NON_PIN_LOG_CAP", 2)
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 2
    r.close()
    monkeypatch.setattr(pv, "NON_PIN_LOG_CAP", 3)
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 1


def test_a_log_of_plain_records_is_stored_whatever_its_length(run, monkeypatch):
    def body(run_, mol, level):
        for k in range(5):
            pv.record_non_pin_fragment("fragment %d" % k)
        return "name"
    monkeypatch.setattr(pv, "NON_PIN_LOG_CAP", 2)
    r = run(body)
    r.ask(); r.ask()
    assert r.uncached == 1


def test_the_stored_log_is_compacted(run):
    def body(run_, mol, level):
        for _ in range(50):
            pv.record_non_pin_fragment("acid")
            pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        return "name"
    r = run(body)
    r.ask()
    stored = [entry for entry in memo._cache_var.get().values() if isinstance(entry, tuple) and len(entry) == 3]
    assert [entry[2] for entry in stored] == [("acid", ('D', "acid chloride", "acyl"))]


def _nested_body(depth):
    def body(run_, mol, level):
        pv.record_non_pin_fragment("frag-%d" % level)
        pv.record_derived_non_pin_fragment("frag-%d tail" % level, "derived-%d" % level)
        if level < depth:
            run_.ask(level + 1, mol)            # named for the sort key...
            run_.ask(level + 1, mol)            #... and again for the assembly
        pv.record_derived_non_pin_fragment("frag-%d tail" % level, "derived-%d" % level)
        return "name-%d" % level
    return body


def test_nested_hits_leave_the_record_the_unmemoised_naming_leaves(run):
    def sequence(mode):
        r = run(_nested_body(_DEPTH), mode)
        value = r.ask()
        first = r.record()
        pv.clear_provenance()
        again = r.ask()
        return value, first, again, r.record(), r.uncached
    on = sequence("on")
    off = sequence("off")
    assert on[:4] == off[:4]
    expected = tuple(x for level in range(_DEPTH + 1) for x in ("frag-%d" % level, "derived-%d" % level))
    assert on[1] == expected
    assert on[4] == _DEPTH + 1                  # each level named once
    assert off[4] == 2 * (2 ** (_DEPTH + 1) - 1)     # every ask of every level names the fragment again


def test_a_nested_hit_makes_the_derived_call_again_into_the_enclosing_log(run):
    def body(run_, mol, level):
        if level == 0:
            run_.ask(1, mol)
            run_.ask(1, mol)
            return "outer"
        pv.record_non_pin_fragment("acid")
        pv.record_derived_non_pin_fragment("acid chloride", "acyl")
        return "inner"
    r = run(body)
    r.ask(0)
    stored = {entry[1]: entry[2] for entry in memo._cache_var.get().values()
              if isinstance(entry, tuple) and len(entry) == 3}
    assert stored["inner"] == ("acid", ('D', "acid chloride", "acyl"))
    assert stored["outer"] == stored["inner"]
    assert r.uncached == 2
