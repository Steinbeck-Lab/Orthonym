"""v29 P1 (fix wave 1): EVERY exit of ``Orthonym.name()`` is audited.

``name()`` has seven ``return`` statements (five originally + two added in v30
Phase 1 for producer-exception resilience: the generic ``except Exception``
routes to recovery success / descriptive fallback so a raising producer degrades
instead of crashing out of ``name()``). Before this wave only two were hooked to
the binding-proof step, and two of the three then-unhooked ones ship a real
(non-failure) name:

* the isotope-decorator exit -- the entire isotope-labeled compound class
  returned with ``ledger_stage=None`` in every mode, because that exit sits
  BEFORE the ``try:`` block that contained the hooks;
* the limit / ``--trivial`` fallback exit -- ``_apply_trivial_fallback``
  swaps a genuine ``get_general_retained_name`` hit in for the
  ``UNSUPPORTED_RING_SYSTEM`` sentinel, so it does NOT "always ship a failure
  string" (pinned by ``test_trivial_fallback_exit_ships_a_real_name`` below).

Today ``enforce`` is wired to no default path, so an unaudited exit changes
no name. From Phase 3 onward it is a hole in the fail-closed guarantee the
ledger exists to provide -- hence the AST guard, which is the test that stops
a future edit from silently reopening the whole class.
"""
import ast
import inspect
from pathlib import Path
from unittest import mock

import pytest
from rdkit import Chem

from orthonym import namer as namer_module
from orthonym.namer import (Orthonym, OrthonymLimitError, is_failure_name,
                             validate_binding_proof)
from orthonym.validation import binding_spine as bs
from orthonym.validation import proof_ledger as pl

pytestmark = pytest.mark.unit


def _best_effort(**kw):
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True, **kw)


def _spine_for(mol, tokens):
    """A trivially-true spine over ``mol``: parent token owns every atom."""
    return bs.BindingSpine(roots=(
        bs.SpineBinding(token=tokens, kind=bs.BindingKind.PARENT,
                        atom_ids=frozenset(range(mol.GetNumAtoms()))),
    ))


# Isotope-labeled molecules measured to reach the isotope-decorator exit.
ISOTOPE_PROBES = ["[2H]C([2H])([2H])O", "CC[2H]", "[13CH4]", "OC[2H]",
                  "[2H]OC([2H])([2H])[2H]"]

# A molecule with a real ``get_general_retained_name`` hit, used to drive the
# limit / --trivial exit.
TRIVIAL_SMILES = "C1CCCNCC1"
TRIVIAL_NAME = "homopiperidine"


# ---------------------------------------------------------------------------
# The class-level guard: no exit may bypass the funnel.
# ---------------------------------------------------------------------------

def _returns_in_name():
    """Every ``Return`` lexically inside ``Orthonym.name`` (not in a nested
    function), as ``(lineno, unparsed_value_or_None)``."""
    src_path = Path(inspect.getsourcefile(namer_module))
    tree = ast.parse(src_path.read_text(), filename=str(src_path))
    fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Orthonym":
            for sub in node.body:
                if (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and sub.name == "name"):
                    fn = sub
    assert fn is not None, "Orthonym.name not found in namer.py"

    found = []

    def _visit(node):
        for child in ast.iter_child_nodes(node):
            # A nested def/lambda has its OWN exits; they are not name()'s.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                found.append((child.lineno, child.value))
            _visit(child)

    _visit(fn)
    return src_path, found


def _is_finish_call(value) -> bool:
    return (isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "_finish"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "self")


def test_every_return_in_name_is_routed_through_finish():
    """THE regression guard for the whole class.

    A bare ``return <something>`` added to ``name()`` in a future edit ships a
    name that the binding proof never sees. Counting them at the AST level is
    the only check that scales past the exits that exist today.
    """
    src_path, returns = _returns_in_name()
    assert returns, "parsed no return statements -- the guard would be vacuous"

    offenders = [
        (lineno, ast.unparse(value))
        for lineno, value in returns
        if value is not None and not _is_finish_call(value)
    ]
    assert not offenders, (
        "Orthonym.name() has return statement(s) that bypass self._finish(), "
        "so the binding proof never sees the shipped name:\n"
        + "\n".join(f"  {src_path}:{lineno}: return {expr}"
                    for lineno, expr in offenders)
        + "\nRoute them through `return self._finish(<name>, smiles)`."
    )


def test_the_guard_sees_all_known_exits():
    """Pins the guard's own reach: if a refactor collapsed ``name()`` so the
    walker found (say) one return, the guard above would pass vacuously.

    Count is 7 as of v30 Phase 1: the original 5 plus the two producer-exception
    resilience exits (recovery success + descriptive fallback) added to the
    generic ``except Exception`` handler so a producer that raises degrades to
    recovery/abstain instead of crashing out of ``name()``. Both route through
    ``self._finish`` (checked by the guard above)."""
    _src, returns = _returns_in_name()
    assert len(returns) == 7, [(ln, ast.unparse(v)) for ln, v in returns]


# ---------------------------------------------------------------------------
# Exit 1/2: the isotope decorator and its fail-closed refusal.
# ---------------------------------------------------------------------------

def test_isotope_exit_finalizes_the_shipped_name():
    """The isotope exit ships ``(2H3)methanol``; a spine recorded for the
    unlabeled skeleton must be re-asserted against THAT string, not left
    unfinalized. Before this wave: ``final_name=None, ok=None``."""
    mol = Chem.MolFromSmiles("CO")

    def _deco(_smiles, _style, _namer):
        pl.record_spine(mol, _spine_for(mol, "methanol"),
                        stage="general_engine", name_at_record="methanol")
        return "(2H3)methanol"

    namer = _best_effort(binding_proof="audit")
    pl.clear_ledger()
    with mock.patch("orthonym.rules.isotopes.decorate_isotopic_name", _deco):
        out = namer.name("[2H]C([2H])([2H])O")

    assert out == "(2H3)methanol"
    rec = pl.get_proof()
    assert rec["stage"] == "general_engine", "precondition: a spine recorded"
    assert rec["final_name"] == "(2H3)methanol", (
        f"the isotope exit did not finalize the ledger: {rec}")
    assert rec["ok"] is not None, rec


def test_isotope_refusal_exit_is_routed_through_finish():
    """The decorator's fail-closed branch returns the abstention sentinel.
    ``_apply_binding_proof`` deliberately does not prove against a sentinel,
    so the observable here is the ROUTING, not a verdict."""
    namer = _best_effort(binding_proof="audit")
    with mock.patch("orthonym.rules.isotopes.decorate_isotopic_name",
                    return_value=None), \
         mock.patch.object(Orthonym, "_finish", autospec=True,
                           side_effect=Orthonym._finish) as spy:
        out = namer.name("[2H]C([2H])([2H])O")
    assert is_failure_name(out), out
    assert spy.call_count == 1, "the isotope refusal exit bypassed _finish"
    assert spy.call_args.args[1] == out


@pytest.mark.parametrize("smi", ISOTOPE_PROBES)
def test_isotope_names_are_byte_identical_off_vs_audit(smi):
    assert _best_effort().name(smi) == _best_effort(binding_proof="audit").name(smi)


# ---------------------------------------------------------------------------
# Exit 3: the limit / --trivial fallback exit.
# ---------------------------------------------------------------------------

def _forced_limit_namer(**kw):
    """A namer whose ``_name_impl`` raises the G0 ring refusal and whose late
    engine recovery declines -- the exact shape that reaches the limit exit."""
    return Orthonym(trivial_fallback=True, general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True, **kw)


def test_trivial_fallback_exit_ships_a_real_name():
    """Corrects the earlier claim that hooking this exit "would be a no-op
    because it always ships a failure string". ``UNSUPPORTED_RING_SYSTEM``'s
    default message IS the sentinel, which is precisely what makes
    ``_apply_trivial_fallback`` fire and substitute a genuine retained name."""
    mol = Chem.MolFromSmiles(TRIVIAL_SMILES)
    namer = _forced_limit_namer()

    def _boom(_s):
        raise OrthonymLimitError("UNSUPPORTED_RING_SYSTEM",
                                  "unknown organic compound")

    with mock.patch.object(type(namer), "_name_impl", staticmethod(_boom)), \
         mock.patch.object(type(namer), "_try_general_engine_recovery",
                           staticmethod(lambda _s: None)):
        out = namer.name(TRIVIAL_SMILES)

    assert mol is not None
    assert out == TRIVIAL_NAME
    assert not is_failure_name(out), (
        "precondition for the whole finding: this exit ships a REAL name")


def test_trivial_fallback_exit_finalizes_the_shipped_name():
    mol = Chem.MolFromSmiles(TRIVIAL_SMILES)

    def _boom(_s):
        pl.record_spine(mol, _spine_for(mol, "azepane"),
                        stage="general_engine", name_at_record="azepane")
        raise OrthonymLimitError("UNSUPPORTED_RING_SYSTEM",
                                  "unknown organic compound")

    namer = _forced_limit_namer(binding_proof="audit")
    pl.clear_ledger()
    with mock.patch.object(type(namer), "_name_impl", staticmethod(_boom)), \
         mock.patch.object(type(namer), "_try_general_engine_recovery",
                           staticmethod(lambda _s: None)):
        out = namer.name(TRIVIAL_SMILES)

    assert out == TRIVIAL_NAME
    rec = pl.get_proof()
    assert rec["stage"] == "general_engine", "precondition: a spine recorded"
    assert rec["final_name"] == TRIVIAL_NAME, (
        f"the limit/--trivial exit did not finalize the ledger: {rec}")
    assert rec["ok"] is not None, rec


def test_trivial_fallback_exit_is_byte_identical_off_vs_audit():
    mol = Chem.MolFromSmiles(TRIVIAL_SMILES)

    def _boom(_s):
        pl.record_spine(mol, _spine_for(mol, "azepane"),
                        stage="general_engine", name_at_record="azepane")
        raise OrthonymLimitError("UNSUPPORTED_RING_SYSTEM",
                                  "unknown organic compound")

    outs = []
    for mode in ("off", "audit"):
        namer = _forced_limit_namer(binding_proof=mode)
        pl.clear_ledger()
        with mock.patch.object(type(namer), "_name_impl",
                               staticmethod(_boom)), \
             mock.patch.object(type(namer), "_try_general_engine_recovery",
                               staticmethod(lambda _s: None)):
            outs.append(namer.name(TRIVIAL_SMILES))
    assert outs[0] == outs[1] == TRIVIAL_NAME


# ---------------------------------------------------------------------------
# _finish's own contract.
# ---------------------------------------------------------------------------

def test_finish_is_a_pure_pass_through_in_off_mode():
    namer = Orthonym()
    pl.clear_ledger()
    assert namer._finish("anything at all", "CCO") == "anything at all"
    assert pl.get_proof()["final_name"] is None, (
        "off mode must do no proof work at all")


def test_finish_never_raises(monkeypatch):
    """A bug in the audit must never turn a successful naming into a crash."""
    namer = _best_effort(binding_proof="audit")

    def _explode(_self, _name, _smiles):
        raise RuntimeError("proof bug")

    monkeypatch.setattr(type(namer), "_apply_binding_proof", _explode)
    assert namer._finish("ethanol", "CCO") == "ethanol"
    assert namer.name("CCO") == "ethanol"


def test_finish_is_inert_inside_a_recursive_frame(monkeypatch):
    """``_finish`` must reproduce the old guard exactly: the proof is asserted
    only at the TOP level, never inside a fragment/substituent frame."""
    namer = _best_effort(binding_proof="audit")
    called = []
    monkeypatch.setattr(type(namer), "_apply_binding_proof",
                        lambda _s, n, _smi: called.append(n) or n)
    monkeypatch.setattr(
        "orthonym.assembly.fragment_naming.is_top_level_naming",
        lambda: False)
    assert namer._finish("ethanol", "CCO") == "ethanol"
    assert called == [], "a nested frame must not finalize the ledger"


# ---------------------------------------------------------------------------
# The shared record helper and the CLI batch flag.
# ---------------------------------------------------------------------------

def test_record_helper_is_a_no_op_when_the_flag_is_off():
    namer = Orthonym()
    pl.clear_ledger()
    namer._record_binding_proof(Chem.MolFromSmiles("CCO"), object(),
                                stage="general_engine")
    assert pl.get_proof()["stage"] is None


def test_record_helper_never_raises_on_a_bad_producer():
    """Recording is observation; a malformed producer object must not break
    naming. ``object()`` has no ``.bindings``."""
    namer = _best_effort(binding_proof="audit")
    pl.clear_ledger()
    namer._record_binding_proof(Chem.MolFromSmiles("CCO"), object(),
                                stage="general_engine")
    assert pl.get_proof()["stage"] is None


def test_batch_mode_threads_the_binding_proof_flag(tmp_path):
    """``--binding-proof audit --batch f.txt`` used to parse cleanly and do
    nothing at all."""
    from orthonym import cli
    src = tmp_path / "in.txt"
    src.write_text("CCO\nCC(=O)O\n")
    seen = []

    def _fake(smiles, **kw):
        seen.append(kw.get("binding_proof"))
        return "x"

    with mock.patch.object(cli, "name_compound", _fake):
        rc = cli.main(["--batch", str(src), "--output",
                       str(tmp_path / "out.txt"), "--binding-proof", "audit"])
    assert rc == 0
    assert seen == ["audit", "audit"], seen


def test_batch_mode_rejects_an_invalid_binding_proof_value(tmp_path):
    """The eager ValueError must fire on the batch surface too -- the per-row
    ``except Exception`` would otherwise degrade a typo to an ERROR line."""
    from orthonym import cli
    src = tmp_path / "in.txt"
    src.write_text("CCO\n")
    with pytest.raises(ValueError):
        cli._process_batch(str(src), None, "pin", False, False, False,
                           "yes-please")


def test_validate_binding_proof_is_the_single_source_of_truth():
    assert validate_binding_proof("audit") == "audit"
    for bad in ("yes-please", "", "OFF", None):
        with pytest.raises(ValueError):
            validate_binding_proof(bad)
    with pytest.raises(ValueError):
        Orthonym(binding_proof="yes-please")
