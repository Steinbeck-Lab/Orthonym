"""The producer record stays outside the memo machinery, and every code that builds or rewrites a
name reports to it.

Two properties keep the ``producer`` key honest after the next change to the naming code:

* the record (``orthonym.metrics.producer_trail``) writes no provenance variable, so the replay
  memo, the substituent-fragment memo and the hang budgets cannot see it;
* a call of a function that builds a whole-molecule name sits inside ``noted(...)``, and a call of
  a step that rewrites a name sits inside ``carried(...)``, or it is on a reasoned allow-list.

The second check reads the source tree with ``ast``; a new unwrapped call fails it with the file and
line. A producer that is added under a new function name is not in the lists below: it shows up as
a named row whose ``producer`` is ``None`` (``tests/integration/test_producer_oracle.py``).
"""
import ast
from functools import lru_cache
from pathlib import Path

import pytest

import orthonym
from orthonym.assembly import nested_memo
from orthonym.metrics import producer_trail as pt
from orthonym.metrics import provenance

pytestmark = pytest.mark.unit

SRC = Path(orthonym.__file__).resolve().parent

# Functions whose return is a name built for a whole molecule. ``noted(producer, value, mol)``
# wraps the call; ``value`` is the second argument.
PRODUCERS = frozenset({
    "name_general", "name_t4_complete", "name_prefix_order_fallback", "try_decompose",
    "decorate_isotopic_name", "_name_multifragment_complete", "assemble_name",
    "_retained_structural_preference",
})
# Steps that turn one name into another. ``carried(name_in, name_out)`` wraps the call, so the
# label of ``name_in`` follows to ``name_out``.
STEPS = frozenset({
    "_final_stereo_check", "_final_grammar_check", "_final_opsin_validity_gate",
    "_retry_cascade_on_gate_rejection", "_exit_round_trip_check", "_maybe_prefix_order_fallback",
    "_select_rt_passing_offer_name", "_try_compose_input_stereo",
})
NOTED = frozenset({"noted", "_noted"})
CARRIED = frozenset({"carried", "_carried"})

# ``<file under src/orthonym>::<innermost enclosing function>`` -> why a reference to a producer is
# not a seam. A stale entry (one that no longer matches a reference) fails
# ``test_the_allow_lists_have_no_stale_entry``.
ALLOWED_PRODUCERS = {
    "assembly/composer.py::_try_ion_aspect_composition":
        "names the neutral form of an ion for the composer's own use (_composing_ion=True); the "
        "ion's name is what the outer assemble_name call returns, and that call's seam notes it",
    "assembly/handlers/tier_a_ring.py::_spiro_principal_suffix_preference":
        "a handler's own call to the engine, whose name the handler returns as its own result; "
        "the inner seam notes the handler's return, which is the deepest seam (decision 1)",
    "cli.py::_engine_only":
        "the --engine-only probe prints the engine's result and builds no row",
    "rules/esters.py::_name_ring_dicarboxylic_diester":
        "names the diacid that the diester is derived from, not the molecule; the handler returns "
        "its own name",
    "rules/esters.py::name_polyfunctional_ester_via_acid":
        "names the acid that the ester is derived from, not the molecule; the handler returns its "
        "own name",
    "rules/esters.py::name_polyacid_polyester":
        "names the acid that the polyester is derived from, not the molecule; the handler returns "
        "its own name",
    "rules/ions.py::_name_arylamine_quaternary_aminium":
        "names the parent amine, a part of the molecule; the handler returns the aminium name",
}
ALLOWED_STEPS = {
    "namer.py::_maybe_prefix_order_fallback":
        "compares the reordered name with its exit round-trip check and ships the reordered name "
        "or the one it was given, never a rewrite of either",
    "namer.py::_exit_round_trip_check_rt":
        "the step's own body: it runs the validity gate on the name it was given, and the step "
        "is carried by its caller",
    "namer.py::_name_with_confidence_impl":
        "not reached from name_tiered: no record is open there, so nothing is recorded or carried",
    "namer.py::_final_opsin_validity_gate":
        "the gate composes the input's stereo onto the name it was given and returns the result; "
        "the gate is a step, and every call of it sits in carried(...)",
    "namer.py::_stereo_descriptor_repair":
        "a branch of the validity gate: it composes onto the stereo-stripped form of the name the "
        "gate was given and returns the result to the gate, whose callers carry the label",
}


# --------------------------------------------------------------------------------------------
# the checker
# --------------------------------------------------------------------------------------------

class _Scan(ast.NodeVisitor):
    """References to ``names`` that are not inside the value argument of a call of ``wrappers``."""

    def __init__(self, rel, names, wrappers):
        self.rel, self.names, self.wrappers = rel, names, wrappers
        self.scope, self.wrapped, self.found = [], 0, []

    def _where(self):
        return f"{self.rel}::{self.scope[-1] if self.scope else '<module>'}"

    def _hit(self, node, label):
        if not self.wrapped:
            self.found.append((self._where(), node.lineno, label))

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node):
        if node.id in self.names:
            self._hit(node, node.id)

    def visit_Attribute(self, node):
        if node.attr in self.names:
            self._hit(node, node.attr)
        self.generic_visit(node)

    def visit_alias(self, node):
        # ``from m import name_general as ng`` hides the call from the check
        base = node.name.split(".")[-1]
        if base in self.names and node.asname and node.asname != base:
            self._hit(node, f"{base} as {node.asname}")

    def visit_Call(self, node):
        if _called(node) not in self.wrappers:
            self.generic_visit(node)
            return
        self.visit(node.func)
        for i, arg in enumerate(node.args):
            self._visit_arg(arg, i == 1)
        for kw in node.keywords:
            self._visit_arg(kw.value, kw.arg in ("value", "name_out"))

    def _visit_arg(self, node, is_value):
        if is_value:
            self.wrapped += 1
        self.visit(node)
        if is_value:
            self.wrapped -= 1


def _scan(tree, rel, names, wrappers):
    scan = _Scan(rel, names, wrappers)
    scan.visit(tree)
    return scan.found


def unreported(source, rel, names, wrappers):
    """``[(file::function, line, name)]`` for each reference to one of ``names`` in ``source``
    that is not inside the value argument of a call of one of ``wrappers``."""
    return _scan(ast.parse(source), rel, names, wrappers)


def _called(node):
    fn = node.func
    return fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None


def _seam_id_of(call):
    """('literal', id), ('prefix', 'outer:' or 'inner:') or ('other', dump) for a seam's first argument."""
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return "literal", first.value
    if (isinstance(first, ast.JoinedStr) and first.values and isinstance(first.values[0], ast.Constant)
            and first.values[0].value in ("outer:", "inner:")):
        return "prefix", first.values[0].value
    return "other", ast.dump(first)[:60]


@lru_cache(maxsize=None)
def _scan_tree():
    """One pass over ``src/orthonym`` (15 MB of source; each file is parsed once and dropped):
    the unwrapped producer references, the unwrapped step references, and the first argument of
    every ``noted(...)`` call outside the record's own module."""
    producers, steps, seams = [], [], {"literal": [], "prefix": [], "other": []}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()
        producers += _scan(tree, rel, PRODUCERS, NOTED)
        steps += _scan(tree, rel, STEPS, CARRIED)
        if path.name == "producer_trail.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args and _called(node) in NOTED:
                kind, value = _seam_id_of(node)
                seams[kind].append(value if kind != "other" else (rel, node.lineno, value))
    return tuple(producers), tuple(steps), seams


def _by_where(found):
    out = {}
    for where, line, label in found:
        out.setdefault(where, []).append((line, label))
    return out


# --------------------------------------------------------------------------------------------
# the record writes no provenance variable
# --------------------------------------------------------------------------------------------

def test_trail_module_touches_no_provenance():
    tree = ast.parse(Path(pt.__file__).read_text(encoding="utf-8"))
    forbidden_names = {"_touch", "get_provenance", "restore_provenance"}
    forbidden_modules = {"provenance", "memo", "nested_memo"}
    seen_names, seen_modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen_names.add(node.attr)
        elif isinstance(node, ast.alias):
            seen_names.add(node.name.split(".")[-1])
            seen_modules.update(node.name.split("."))
        if isinstance(node, ast.ImportFrom):
            seen_modules.update((node.module or "").split("."))
    assert not seen_names & forbidden_names, seen_names & forbidden_names
    assert not seen_modules & forbidden_modules, seen_modules & forbidden_modules


def test_provenance_keys_still_equal_the_replayed_keys():
    # the existing contract of assembly/nested_memo.py; the record adds no key to either side
    assert set(provenance.get_provenance()) == set(nested_memo._VARS)
    assert "producer" not in provenance.get_provenance()


def test_noting_and_carrying_write_no_provenance_variable():
    from rdkit import Chem
    before = provenance.get_provenance()
    provenance.push_touched_log()
    token = pt.open_trail("CCO")
    try:
        pt.noted("outer:retained_name", "ethanol", Chem.MolFromSmiles("CCO"))
        pt.carried("ethanol", "ethyl alcohol")
        with pt.suspended():
            pt.noted("inner:x", "ignored")
        assert pt.lookup("ethyl alcohol") == "outer:retained_name"
    finally:
        pt.close_trail(token)
        touched = provenance.pop_touched_log()
    assert touched == set()
    assert provenance.get_provenance() == before


# --------------------------------------------------------------------------------------------
# every producer reports, every rewriting step passes the label on
# --------------------------------------------------------------------------------------------

def _offenders(found, allowed):
    return {where: lines for where, lines in _by_where(found).items() if where not in allowed}


def test_every_producer_call_is_noted():
    offenders = _offenders(_scan_tree()[0], ALLOWED_PRODUCERS)
    assert not offenders, (
        "a call of a function that builds a name is not inside the value argument of noted(...), "
        "so the row cannot say which code built the name; wrap it as "
        "`noted(\"<id>\", <call>, <the molecule the call names>)`, or, if it names a part of the "
        "molecule or is a handler's own call, add it to ALLOWED_PRODUCERS with the reason: "
        f"{offenders}")


def test_every_rewriting_step_passes_the_label_on():
    offenders = _offenders(_scan_tree()[1], ALLOWED_STEPS)
    assert not offenders, (
        "a call of a step that rewrites a name is not inside the value argument of carried(...), "
        "so a name it changes loses its producer; wrap it as `carried(<name in>, <call>)`, or add "
        f"it to ALLOWED_STEPS with the reason: {offenders}")


def test_the_allow_lists_have_no_stale_entry():
    producers, steps, _seams = _scan_tree()
    for found, allowed in ((producers, ALLOWED_PRODUCERS), (steps, ALLOWED_STEPS)):
        live = set(_by_where(found))
        assert not set(allowed) - live, (
            f"entries that no longer match an unwrapped reference: {sorted(set(allowed) - live)}")


# What the checker says about source text, so that a change to it cannot quietly accept a call.
_SRC_WRAPPED = '''
def f(self, mol):
    a = _noted("composer", assemble_name(features, style=self.style), mol)
    b = noted("decomposition", _rescues.run(try_decompose, mol, style="pin"), mol)
    c = pt.noted("engine:t4", self._name_multifragment_complete(mol, smiles), mol)
    d = _noted("engine:general", value=name_general(mol, feats), mol=mol)
'''
_SRC_UNWRAPPED = '''
class C:
    def f(self, mol):
        return name_general(mol, feats)
    def g(self, mol):
        return self._name_multifragment_complete(mol, smiles)
    def h(self, mol):
        return _noted("composer", None, assemble_name(features))      # the molecule slot is no value
    def i(self, mol):
        return _noted("composer", len(assemble_name(features)), mol)  # a call inside the value is wrapped
    def j(self, mol):
        return _noted(assemble_name(features), "composer", mol)       # the id slot is no value
    def k(self, mol):
        return carried(assemble_name(features), like="x")             # another carried(): not a seam
    def l(self, mol):
        fn = try_decompose
        return fn(mol)
'''


def test_the_checker_accepts_a_wrapped_call():
    assert unreported(_SRC_WRAPPED, "x.py", PRODUCERS, NOTED) == []


def test_the_checker_flags_each_unwrapped_form():
    found = unreported(_SRC_UNWRAPPED, "x.py", PRODUCERS, NOTED)
    assert [(w, n) for w, _l, n in found] == [
        ("x.py::f", "name_general"),
        ("x.py::g", "_name_multifragment_complete"),
        ("x.py::h", "assemble_name"),
        ("x.py::j", "assemble_name"),
        ("x.py::k", "assemble_name"),
        ("x.py::l", "try_decompose"),
    ]


def test_the_checker_flags_an_aliased_import_and_ignores_definitions_and_plain_imports():
    src = '''
from a import name_general as ng, try_decompose
from b import assemble_name
__all__ = ["assemble_name"]
def name_general(mol): pass
class K:
    def try_decompose(self): pass
'''
    assert [(w, n) for w, _l, n in unreported(src, "x.py", PRODUCERS, NOTED)] == [
        ("x.py::<module>", "name_general as ng")]


def test_the_checker_reads_the_step_wrapper():
    src = '''
def f(self, name):
    ok = _carried(name, self._exit_round_trip_check(name, smiles))
    bad = self._exit_round_trip_check(name, smiles)
    also = _carried(_final_stereo_check(mol, name), name)    # in the name_in slot: not wrapped
'''
    assert [(w, n) for w, _l, n in unreported(src, "x.py", STEPS, CARRIED)] == [
        ("x.py::f", "_exit_round_trip_check"), ("x.py::f", "_final_stereo_check")]


# --------------------------------------------------------------------------------------------
# the ids at the seams
# --------------------------------------------------------------------------------------------

FIXED_IDS = frozenset({
    "composer", "engine:general", "engine:t4", "decomposition", "multifragment", "retained_fused",
    "isotope", "table:coordination", "table:trivial", "table:np_parent", "table:retained",
    "charged_leftovers:thiophosphate_ester_anion", "charged_leftovers:hetero_group_chalcogenide",
    "charged_leftovers:carbon_group_azanide", "charged_leftovers:multiply_charged_centre",
})


def test_every_seam_names_an_id_of_the_vocabulary():
    seams = _scan_tree()[2]
    assert not seams["other"], (
        f"a seam whose id is neither a literal nor an outer:/inner: f-string: {seams['other']}")
    assert set(seams["literal"]) <= FIXED_IDS, set(seams["literal"]) - FIXED_IDS


def test_every_fixed_id_and_both_families_have_a_seam():
    seams = _scan_tree()[2]
    assert FIXED_IDS <= set(seams["literal"]), FIXED_IDS - set(seams["literal"])
    assert {"outer:", "inner:"} <= set(seams["prefix"]), seams["prefix"]
