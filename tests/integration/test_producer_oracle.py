"""An independent oracle for the ``producer`` key of the ``name_tiered`` row.

The seams of the shipped code report to the per-call record (``metrics.producer_trail``). This
file does not use them: it wraps the producer functions at test time, in the process, and
records for each ``name_tiered`` call which producer returned which string, in call order, and
which steps rewrote a string. For every row that names a molecule, the producer the row reports
must be the first producer that returned the string that ships (a step that rewrote a string passes
its producer on), where "returned" counts only a string built for the whole input. A fragment's name, a
builder that a handler calls inside itself, and a call made by the strict twin (which names for
comparison only) are not producers of the shipped name. A named row with ``producer`` None
fails.

The rows are ``tests/fixtures/producer_oracle.jsonl`` (SMILES and a tier, no name): 240
molecules of the dev set, the Blue Book and the gold rows, chosen so that every producer id has
rows. Every case is one molecule; the cases at index 0, 10, 20,... run in the public
pull-request run and the others are ``slow`` (the nightly run).
"""
import json
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.producer_pairs import RELATIVE_STEREO, STEREO_COMPOSED

# The suite switches the OPSIN validity gate off unless a test asks for it; the producers of the
# shipped configuration are the ones with the gate on.
pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "producer_oracle.jsonl"
ROWS = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]

# Rows where the oracle and the shipped record differ by rule, as SMILES@tier -> the reason.
# Empty: a mismatch is a bug in a seam until it is shown to be a difference of rule.
KNOWN_ORDER_DIFFERENT: dict = {}

# The methods of ``Orthonym`` that build a name for the whole input and that nothing below them
# reports, as the trace saw them; a name that only one of them returned is a producer of its own.
_RESCUES = ("_try_general_engine_recovery", "_try_besteffort_clean_general_fallthrough",
            "_try_perf_budget_t4_rescue", "_try_demote_senior_group_rescue",
            "_try_alternate_parent_rescue", "_try_t4_rescue")
# The steps that turn one string into another (the name stays the same molecule's).
_STEPS = (("_retry_cascade_on_gate_rejection", 2), ("_select_rt_passing_offer_name", 0),
          ("_exit_round_trip_check", 0), ("_maybe_prefix_order_fallback", 0))
_MODULE_STEPS = (("_final_stereo_check", 1), ("_final_grammar_check", 0),
                 ("_final_opsin_validity_gate", 0), ("_try_compose_input_stereo", 0))


def _name_of(value):
    if value is None or isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


class _Oracle:
    """Wrappers around the producers; ``events`` is the list of the run that is being named."""

    def __init__(self):
        self.events = None          # the events of the current _name_tiered_impl run
        self.heavy = -1             # heavy atoms of the input of that run
        self.muted = 0              # > 0 inside the strict twin
        self.rows = {}              # id(row) -> (row, events of the run that built it)
        self._undo = []

    # -- recording -------------------------------------------------------------------------
    def seam(self, label, out, whole=True):
        if self.events is None or self.muted:
            return
        self.events.append(("seam", label, None, _name_of(out), bool(whole)))

    def step(self, label, inn, out):
        if self.events is None or self.muted:
            return
        self.events.append(("step", label, inn, _name_of(out), True))

    def whole(self, mol):
        try:
            return mol is not None and mol.GetNumHeavyAtoms() == self.heavy
        except Exception:                                       # noqa: BLE001
            return False

    # -- installing -------------------------------------------------------------------------
    def install(self, mp):
        import orthonym.assembly.general_engine as general_engine
        import orthonym.assembly.inner_dispatch as inner_dispatch
        import orthonym.assembly.t4_coverage as t4_coverage
        import orthonym.decomposition as decomposition
        import orthonym.namer as namer
        import orthonym.rules.isotopes as isotopes
        from orthonym.routing.dispatch_table import DISPATCH_TABLE
        oracle = self

        def wrap(owner, attr, fn):
            mp.setattr(owner, attr, fn(getattr(owner, attr)))

        # one run of the row builder = one record in the shipped code; the row reads its own
        def tiered_impl(orig):
            def w(self_, smiles, *a, **k):
                saved = (oracle.events, oracle.heavy)
                mol = Chem.MolFromSmiles(smiles)
                oracle.events = []
                oracle.heavy = mol.GetNumHeavyAtoms() if mol is not None else -1
                try:
                    row = orig(self_, smiles, *a, **k)
                    oracle.rows[id(row)] = (row, oracle.events)
                    return row
                finally:
                    oracle.events, oracle.heavy = saved
            return w
        wrap(Orthonym, "_name_tiered_impl", tiered_impl)

        def twin(orig):
            def w(self_, *a, **k):
                oracle.muted += 1
                try:
                    return orig(self_, *a, **k)
                finally:
                    oracle.muted -= 1
            return w
        wrap(Orthonym, "_strict_pin_twin_name", twin)

        # a row named through another drawing of the input is built by the run of that drawing; the
        # rewrite of the drawing's name into the input's name is a step
        import orthonym.assembly.halogen_drawing as halogen_drawing
        import orthonym.assembly.ocene_drawing as ocene_drawing

        def drawing(orig):
            def w(namer_, smiles, *a, **k):
                seen = set(oracle.rows)
                row = orig(namer_, smiles, *a, **k)
                runs = [v for key, v in oracle.rows.items() if key not in seen]
                if row is not None and runs:
                    built, events = runs[-1]
                    oracle.rows[id(row)] = (row, events + [("step", "drawing", built["name"], row["name"], True)])
                return row
            return w
        wrap(halogen_drawing, "iodine_drawing_row", drawing)
        wrap(ocene_drawing, "ocene_drawing_row", drawing)

        # the 53 classes: the handler of each entry of the dispatch table
        for entry in DISPATCH_TABLE.values():
            orig_handler = entry.handler
            label = "outer:" + entry.class_id.value

            def w(*a, _h=orig_handler, _label=label, **k):
                mol = a[0] if a and hasattr(a[0], "GetNumHeavyAtoms") else None
                whole = oracle.whole(mol)
                out = _h(*a, **k)
                oracle.seam(_label, out, whole)
                return out
            object.__setattr__(entry, "handler", w)
            self._undo.append((entry, orig_handler))

        # the inner handlers, as the composer reaches them
        def inner(orig):
            def w(features, mol=None, style="pin"):
                res = orig(features, mol, style)
                if res is not None:
                    m = mol if mol is not None else getattr(features, "mol", None)
                    oracle.seam("inner:" + res.handler_id, res.result, oracle.whole(m))
                return res
            return w
        wrap(inner_dispatch, "dispatch_inner", inner)

        def builder(label, mol_of, orig, *, skip_in_handler=False):
            def w(*a, **k):
                caller = sys._getframe(1).f_code.co_filename.replace("\\", "/")
                out = orig(*a, **k)
                # a handler that calls another builder inside itself returns that name as its
                # own result: the handler is the seam, not the call it made
                if not (skip_in_handler and "/assembly/handlers/" in caller):
                    oracle.seam(label, out, oracle.whole(mol_of(a, k)))
                return out
            return w
        wrap(namer, "assemble_name", lambda f: builder("composer", lambda a, k: getattr(a[0], "mol", None), f))
        wrap(general_engine, "name_general",
             lambda f: builder("engine:general", lambda a, k: a[0], f, skip_in_handler=True))
        wrap(t4_coverage, "name_t4_complete", lambda f: builder("engine:t4", lambda a, k: a[0], f))
        wrap(t4_coverage, "name_prefix_order_fallback",
             lambda f: builder("engine:t4", lambda a, k: a[0], f))
        wrap(decomposition, "try_decompose", lambda f: builder("decomposition", lambda a, k: a[0], f))
        wrap(isotopes, "decorate_isotopic_name",
             lambda f: builder("isotope", lambda a, k: Chem.MolFromSmiles(a[0]), f))

        def method_seam(label, mol_of):
            def fn(orig):
                def w(self_, *a, **k):
                    out = orig(self_, *a, **k)
                    oracle.seam(label, out, oracle.whole(mol_of(a, k)) if mol_of else True)
                    return out
                return w
            return fn
        wrap(Orthonym, "_name_multifragment_complete",
             method_seam("multifragment", lambda a, k: a[0]))
        wrap(Orthonym, "_try_retained_fused_upgrade",
             method_seam("retained_fused", lambda a, k: a[0]))
        wrap(Orthonym, "_retained_structural_preference",
             method_seam("table:retained", lambda a, k: a[0]))
        for nm in _RESCUES:
            wrap(Orthonym, nm, method_seam("rescue:" + nm, None))

        # the two lists, where name takes them as the last resort (the same functions are
        # also asked as checks, by other callers)
        def list_seam(label):
            def fn(orig):
                def w(*a, **k):
                    out = orig(*a, **k)
                    if sys._getframe(1).f_code.co_name == "_finish":
                        oracle.seam(label, out)
                    return out
                return w
            return fn
        wrap(namer, "_coordination_retained_name", list_seam("table:coordination"))
        wrap(namer, "_np_exact_list_name", list_seam("table:np_parent"))

        # the last-resort table of retained names turns a failure name into a name
        def trivial(orig):
            def w(self_, result, smiles, *a, **k):
                out = orig(self_, result, smiles, *a, **k)
                if isinstance(out, str) and out != result:
                    oracle.seam("table:trivial", out)
                return out
            return w
        wrap(Orthonym, "_apply_trivial_fallback", trivial)

        def step_of(label, pos, method):
            def fn(orig):
                def w(*a, **k):
                    args = a[1:] if method else a
                    inn = _name_of(args[pos]) if len(args) > pos else None
                    out = orig(*a, **k)
                    oracle.step(label, inn, out)
                    return out
                return w
            return fn
        for nm, pos in _STEPS:
            wrap(Orthonym, nm, step_of(nm, pos, True))
        for nm, pos in _MODULE_STEPS:
            wrap(namer, nm, step_of(nm, pos, False))

        # the cis/trans conversion of a ring pair's name: it turns one name into a
        # list of names, each of them a step from the name it was given
        import orthonym.rules.relative_stereo as relative_stereo

        def convert(orig):
            def w(name, *a, **k):
                out = orig(name, *a, **k)
                for candidate in out:
                    oracle.step("convert_pair_blocks", name, candidate)
                return out
            return w
        wrap(relative_stereo, "convert_pair_blocks", convert)

    def restore(self):
        for entry, handler in self._undo:
            object.__setattr__(entry, "handler", handler)
        self._undo.clear()

    # -- reading -------------------------------------------------------------------------------
    def expected(self, row):
        """(the id of the producer of the string that ships, every id that returned it), or
        (None, ), from the events of the run that built ``row``, in call order: the first
        whole-molecule producer to return a string keeps it, and a step that turns a string that has
        a producer into another string passes the producer on (unless another already has it)."""
        entry = self.rows.get(id(row))
        assert entry is not None and entry[0] is row, "the row was not built by a recorded run"
        producer_of = {}
        for kind, label, inn, out, whole in entry[1]:
            if not out or is_failure_name(out):
                continue
            if kind == "seam":
                if whole:
                    producer_of.setdefault(out, label)
            elif inn in producer_of and out != inn:
                producer_of.setdefault(out, producer_of[inn])
        name = row["name"]
        returned_by = tuple(dict.fromkeys(
            label for kind, label, _inn, out, whole in entry[1] if kind == "seam" and whole and out == name))
        return producer_of.get(name), returned_by


@pytest.fixture(scope="module")
def oracle():
    rec = _Oracle()
    with pytest.MonkeyPatch.context() as mp:
        rec.install(mp)
        try:
            yield rec
        finally:
            rec.restore()


@pytest.fixture(autouse=True)
def _forget_rows(oracle):
    oracle.rows.clear()
    yield
    oracle.rows.clear()


def _case(k, row):
    marks = () if k % 10 == 0 else (pytest.mark.slow,)
    return pytest.param(row["smiles"], row["tier"], id=f"{row['tier']}-{k}", marks=marks)


def _check(oracle, record_property, smiles, tier):
    """Name the row and compare its producer with the oracle's; returns the row."""
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    row = Orthonym(style="pin", **flags).name_tiered(smiles)
    if row["tier"] == "abstain" or not row.get("name") or is_failure_name(row["name"]):
        assert row["producer"] is None, row                  # nothing ships (decision 3)
        return row
    expected, returned_by = oracle.expected(row)
    record_property("ids_that_returned_the_name", len(returned_by))
    assert row["producer"] is not None, ("a named row has no producer", row, returned_by)
    if f"{smiles}@{tier}" in KNOWN_ORDER_DIFFERENT:
        return row
    assert row["producer"] == expected, (row["producer"], expected, returned_by, row["name"])
    return row


@pytest.mark.parametrize("smiles,tier", [_case(k, r) for k, r in enumerate(ROWS)])
def test_producer_is_the_first_whole_molecule_seam(oracle, record_property, smiles, tier):
    _check(oracle, record_property, smiles, tier)


# The rows whose name is a producer's string with the input's stereo composed onto it: the oracle
# sees the composition as a step (``_try_compose_input_stereo``), whichever of the emit decision and the
# validity gate calls it, and the shipped label is the one of the string the composition started from.
@pytest.mark.parametrize("smiles,tier,expected", STEREO_COMPOSED,
                         ids=[f"{expected}-{i}" for i, (_s, _t, expected) in enumerate(STEREO_COMPOSED)])
def test_producer_of_a_name_with_composed_stereo(oracle, record_property, smiles, tier, expected):
    row = _check(oracle, record_property, smiles, tier)
    assert row["producer"] == expected, row
    composed = [e for e in oracle.rows[id(row)][1] if e[0] == "step" and e[1] == "_try_compose_input_stereo"
                and e[3] == row["name"]]
    assert composed, "the shipped name is not the result of a composition: the row does not test the step"


# The rows whose name is the cis/trans name of a ring pair: the oracle sees the conversion
# as a step (``convert_pair_blocks``), and the shipped label is the one of the name with lowercase r/s
# that the conversion started from.
@pytest.mark.parametrize("smiles,tier,expected", RELATIVE_STEREO,
                         ids=[f"{expected}-{i}" for i, (_s, _t, expected) in enumerate(RELATIVE_STEREO)])
def test_producer_of_a_relative_stereo_name(oracle, record_property, smiles, tier, expected):
    row = _check(oracle, record_property, smiles, tier)
    assert row["producer"] == expected, row
    made = [e for e in oracle.rows[id(row)][1] if e[0] == "step" and e[1] == "convert_pair_blocks"
            and e[3] == row["name"]]
    assert made, "the shipped name is not the result of the conversion: the row does not test the step"


def test_the_fixture_is_what_the_plan_says():
    keys = [(r["smiles"], r["tier"]) for r in ROWS]
    assert len(keys) == 240 and len(set(keys)) == 240
    assert {t for _, t in keys} == {"pin", "best-effort"}
    assert all(set(r) == {"smiles", "tier"} for r in ROWS)
