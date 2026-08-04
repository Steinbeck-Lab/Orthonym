"""Contract tests for the v30 PE-1 candidate ledger.

The ledger's whole claim to be an *instrument* rather than a behaviour change rests
on two properties, and both are asserted here rather than assumed:

  1. It is OFF unless explicitly enabled, so the production path carries one
     boolean test and nothing else.
  2. Naming is **byte-identical** with it on and off.

Plus the measured design constraint from ``
§1: ``scope`` distinguishes a whole-molecule candidate from a fragment name, because
a correct fragment name can never round-trip to the whole input and treating one as a
molecule candidate manufactures a fake producer-correctness class.
"""

import pytest

from orthonym.metrics import candidate_ledger as cl


@pytest.fixture(autouse=True)
def _clean_ledger():
    """Every test starts and ends with the ledger off — it is thread-local state
    and a leaked ``enable()`` would make a later test pass for the wrong reason."""
    cl.disable()
    yield
    cl.disable()


# --------------------------------------------------------------- off by default

def test_disabled_by_default_records_nothing():
    assert cl.is_enabled() is False
    cl.record_candidate("some.site", cl.Stage.PRODUCED, "ethanol")
    assert cl.read_ledger() == []


def test_disable_drops_entries():
    cl.enable()
    cl.record_candidate("s", cl.Stage.PRODUCED, "ethanol")
    assert len(cl.read_ledger()) == 1
    cl.disable()
    assert cl.read_ledger() == []
    assert cl.is_enabled() is False


def test_enable_clears_any_previous_entries():
    cl.enable()
    cl.record_candidate("s", cl.Stage.PRODUCED, "first")
    cl.enable()
    assert cl.read_ledger() == []


# ------------------------------------------------------------------ append-only

def test_append_only_preserves_every_entry_and_its_order():
    """Unlike ``abstention.py`` (first-writer-wins, one code per session) the
    ledger must keep losers — the selection question is entirely about them."""
    cl.enable()
    cl.record_candidate("producer.a", cl.Stage.PRODUCED, "name-a")
    cl.record_candidate("gate.self01", cl.Stage.SUPPRESSED, "name-a")
    cl.record_candidate("producer.b", cl.Stage.PRODUCED, "name-b")
    entries = cl.read_ledger()
    assert [e.name for e in entries] == ["name-a", "name-a", "name-b"]
    assert [e.stage for e in entries] == ["produced", "suppressed", "produced"]


def test_duplicate_name_from_two_sites_is_kept_twice():
    """Deduplication is the consumer's job. The ledger must not hide that two
    different sites produced the same string, because *which site* is the
    build-order signal PB4 had to reconstruct by hand."""
    cl.enable()
    cl.record_candidate("site.one", cl.Stage.PRODUCED, "ethanol")
    cl.record_candidate("site.two", cl.Stage.PRODUCED, "ethanol")
    assert [e.site for e in cl.read_ledger()] == ["site.one", "site.two"]


def test_clear_ledger_keeps_recording_enabled():
    """A batch consumer names many molecules in one process and must reset
    between them without turning the ledger off."""
    cl.enable()
    cl.record_candidate("s", cl.Stage.PRODUCED, "first")
    cl.clear_ledger()
    assert cl.is_enabled() is True
    cl.record_candidate("s", cl.Stage.PRODUCED, "second")
    assert [e.name for e in cl.read_ledger()] == ["second"]


# ------------------------------------------------------------------------ scope

def test_scope_defaults_to_molecule_and_fragment_is_explicit():
    cl.enable()
    cl.record_candidate("pool.add", cl.Stage.PRODUCED, "ethanol")
    cl.record_candidate("subst.namer", cl.Stage.PRODUCED, "hydroxy",
                        scope=cl.Scope.FRAGMENT)
    by_scope = {e.name: e.scope for e in cl.read_ledger()}
    assert by_scope == {"ethanol": "molecule", "hydroxy": "fragment"}


def test_fragment_entries_are_separable_from_molecule_entries():
    """The consumer filters on this. Measured: 'N,N-diethylethanamine' is a correct
    FRAGMENT name for a row whose molecule-scope candidate was sentinel-spliced;
    round-tripping it against the whole input would count it as a wrong molecule."""
    cl.enable()
    cl.record_candidate("subst", cl.Stage.PRODUCED, "N,N-diethylethanamine",
                        scope=cl.Scope.FRAGMENT)
    cl.record_candidate("composer", cl.Stage.PRODUCED, "amino-N-substituentthing")
    mol = [e for e in cl.read_ledger() if e.scope == cl.Scope.MOLECULE]
    assert [e.name for e in mol] == ["amino-N-substituentthing"]


def test_scope_and_depth_are_resolved_automatically_at_top_level():
    """With no naming in progress the depth is 0, so an unqualified record is
    molecule-scope. This is the default every hook relies on."""
    cl.enable()
    cl.record_candidate("pool.add", cl.Stage.PRODUCED, "ethanol")
    entry = cl.read_ledger()[0]
    assert (entry.scope, entry.depth) == (cl.Scope.MOLECULE, 0)


def test_explicit_scope_overrides_the_depth_prior():
    """The substituent-cascade hook knows it produced a fragment name even when
    called at depth 0, so an explicit scope must win over the resolver."""
    cl.enable()
    cl.record_candidate("subst", cl.Stage.PRODUCED, "amino-N-ethylethyl",
                        scope=cl.Scope.FRAGMENT)
    entry = cl.read_ledger()[0]
    assert entry.scope == cl.Scope.FRAGMENT
    assert entry.depth == 0, "depth is still recorded, it is just not the scope"


def test_resolve_scope_returns_a_scope_and_an_int_depth():
    scope, depth = cl.resolve_scope()
    assert scope in (cl.Scope.MOLECULE, cl.Scope.FRAGMENT)
    assert isinstance(depth, int)


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_scope_prior_is_documented_as_imperfect_not_assumed_correct(opsin_gate):
    """The measured caveat, pinned so it cannot decay into folklore.

    ``O=C([O-])[C@H]1CCCCN1`` names its anion by naming the neutral acid in a
    NESTED component naming, so the whole-molecule candidate
    '(2R)-piperidine-2-carboxylic acid' is recorded at depth 1 and therefore
    tagged FRAGMENT. Any consumer that treated ``scope`` as ground truth would
    conclude no molecule candidate was ever built for this row. The classifier
    must resolve scope by round-trip evidence instead; this test exists so that
    requirement is visible to whoever changes the resolver.
    """
    from orthonym import Orthonym

    cl.enable()
    Orthonym().name("O=C([O-])[C@H]1CCCCN1")
    entries = cl.read_ledger()
    cl.disable()

    acid = [e for e in entries if e.name == "(2R)-piperidine-2-carboxylic acid"]
    assert acid, f"expected the neutral-acid candidate on the ledger, got {entries}"
    assert all(e.depth >= 1 and e.scope == cl.Scope.FRAGMENT for e in acid), (
        "the caveat no longer reproduces -- if the anion path stopped nesting, "
        "update candidate_ledger.resolve_scope()'s docstring and the classifier "
        "note that depends on it"
    )


# ------------------------------------------------------------- never break naming

@pytest.mark.parametrize("bad_name", [None, "", 12345, object()])
def test_record_never_raises_on_odd_input(bad_name):
    """Telemetry must never raise into naming. A ledger that throws on a None
    name would convert an instrumented run into a crash."""
    cl.enable()
    cl.record_candidate("s", cl.Stage.PRODUCED, bad_name)  # must not raise
    assert len(cl.read_ledger()) == 1


def test_read_ledger_returns_a_copy():
    """A consumer mutating the returned list must not corrupt the recorder."""
    cl.enable()
    cl.record_candidate("s", cl.Stage.PRODUCED, "ethanol")
    got = cl.read_ledger()
    got.clear()
    assert len(cl.read_ledger()) == 1


def test_as_dicts_is_json_serialisable():
    import json

    cl.enable()
    cl.record_candidate("s", cl.Stage.SUPPRESSED, "ethanol", detail="self01")
    blob = json.dumps(cl.as_dicts())
    assert "self01" in blob and "ethanol" in blob


# ------------------------------------------------------------ thread isolation

def test_ledger_is_thread_local():
    """Enabled on one thread must not record on another — the harness runs a pool
    of workers and a shared ledger would interleave molecules."""
    import threading

    cl.enable()
    cl.record_candidate("main", cl.Stage.PRODUCED, "main-name")
    seen = {}

    def worker():
        seen["enabled"] = cl.is_enabled()
        seen["entries"] = cl.read_ledger()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["enabled"] is False
    assert seen["entries"] == []
    assert len(cl.read_ledger()) == 1


# ------------------------------------------------- the load-bearing contract

# Molecules chosen from dev500 run 20260804T170501Z so the expectations are
# measured, not invented: two rt_exact rows and one that abstains. The abstaining
# row matters most -- it exercises the suppression path, which is where an
# instrument is most likely to perturb behaviour.
_BYTE_IDENTICAL_PROBES = [
    "NC(=O)C(O)CO",
    "CN1C(=O)C(Cl)(Cl)C(=O)c2ccccc21",
    "C[C@H]1CCC[C@@]2(C)CCC(O)CC12",
]


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_naming_is_byte_identical_with_ledger_enabled(opsin_gate):
    """THE contract. If enabling the ledger changes a single character of a single
    name, it is not an instrument and every number it produces is suspect.

    Runs with the OPSIN validity gate ENABLED because that is the production
    configuration and the gate is on the path this ledger instruments; with the
    gate off the suppression probe would abstain for a different reason and the
    test would pass blind (see tests/conftest.py).
    """
    from orthonym import Orthonym

    cl.disable()
    before = [Orthonym().name(s) for s in _BYTE_IDENTICAL_PROBES]

    cl.enable()
    after = [Orthonym().name(s) for s in _BYTE_IDENTICAL_PROBES]
    recorded = cl.read_ledger()
    cl.disable()

    assert after == before, (
        "enabling the candidate ledger changed a name -- the instrument is "
        f"perturbing naming.\n  ledger off: {before}\n  ledger on : {after}"
    )
    # Guard against the vacuous pass: identical output because nothing recorded
    # would satisfy the assertion above while proving nothing at all.
    assert recorded, (
        "ledger recorded ZERO entries for 3 molecules -- the byte-identical "
        "assertion above is vacuous. Check the hook sites are installed."
    )
