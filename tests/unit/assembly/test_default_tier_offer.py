"""The best-effort tier does not name less than the default tier names (user decision 2026-10-20).

The validity gate withdraws, at the best-effort tier, a name OPSIN 2.9.0 cannot read, and the default tier
ships a class of them: the strict path's name whose stereo layer OPSIN rejects, when the constitution of
the stereo-stripped name is the input's and every descriptor is the input's CIP label (the gate outcome
``self_consistency_constitution_only``, label ``pin_unverified``). Six molecules of the last end-check runs
(seven rows) were named by the default tier and left unnamed by the best-effort tier above it.

``assembly.default_tier_offer`` is the rule: a molecule the best-effort tier leaves without a name gets the
row the default tier gives it, when that row is a name of this class and a name the best-effort gate
withdrew for the molecule. An offer, not a return: no name that ships changes. Parity, not a new trust path:
the row is the default tier's own, and a row of any other default-tier exception is not offered. Every entry
point agrees: ``name_tiered`` (the row), ``Orthonym.name``, ``name_compound`` and ``name_with_confidence``
(the decorator of ``name``). Only the best-effort tier takes it: the valid and complete tiers keep the
round-trip contract.

Changed asserted values: none, every row is new. The tests that assert the abstention of these molecules at
the complete tier (tests/unit/test_besteffort_full_rt_gate.py) do not run the best-effort tier and are not
changed; those through ``name_compound`` at best-effort
(tests/unit/assembly/test_besteffort_stereo_abstain.py) accept either an abstention or a name OPSIN cannot
read, and still pass.
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly import default_tier_offer
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import provenance as pv
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate

# the six molecules of the end-check inversions (the seven rows of 8 eval runs, a dev split / dev2000 /
# a holdout split / milestone1500 pairs of <split>_pin and <split>_best-effort)
SIX = [
    "Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H]1Cl",                       # (1R,2r,3S,4R,5r,6S)-...
    "Cl[C@H]1[C@H](Cl)[C@@H](Cl)[C@H](Cl)[C@H](Cl)[C@H]1Cl",
    "CN1[C@@H]2CC[C@H]1C[C@H](O)C2",                                                  # (1R,3s,5S)-tropan-3-ol
    "O=C(O)C(=O)C[C@]1(C(=O)O)C=C[C@@H](O)C=C1",                                      # (1s,4s)-...
    "O=C(/C=C/c1ccc(O)c(O)c1)O[C@H]1[C@H](O)C[C@](O)(C(=O)O)C[C@H]1O",                # chlorogenic acid
    "CC(=O)N[C@H]1[C@@H](O[C@@H]2[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]2O)O[C@H](CO)[C@@H](O)[C@@H]1O",
]

# the three molecules of tests/unit/test_rs_relative_offer.py that the conversion to cis/trans cannot name
THREE = [
    "C[C@H]1C[C@@H](O)C[C@@H](C)O1",                 # (2R,4r,6S)-2,6-dimethyloxan-4-ol
    "C[C@]1(Cl)CC[C@H](O)CC1",                       # (1r,4r)-4-chloro-4-methylcyclohexan-1-ol
    "C[C@@]1(Cl)CC[C@H](O)CC1",                      # (1s,4s)-4-chloro-4-methylcyclohexan-1-ol
]

#: the fields of a row the offer must hand over as the default tier gives them
ROW_FIELDS = ("name", "tier", "is_pin", "source", "opsin", "gates_passed", "gate_outcome", "verified",
              "spelling_failures", "stereo_unexpressed", "limit_code")


@pytest.mark.parametrize("smiles", SIX)
def test_best_effort_gives_the_row_the_default_tier_gives(smiles):
    default = default_tier_row(smiles)
    # the premise: the default tier ships it, through the stereo-layer carve-out
    assert default["tier"] == "pin_unverified", default
    assert default["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY, default
    row = name_best_effort(smiles)
    assert {k: row[k] for k in ROW_FIELDS} == {k: default[k] for k in ROW_FIELDS}, (row, default)
    assert row["opsin"] == "verified_constitution_only" and row["verified"] == "opsin_constitution", row


def test_the_complete_tier_keeps_the_round_trip_contract():
    """Only the best-effort tier takes the offer: the complete tier ships a name only when OPSIN reads it
    back in full."""
    complete = Orthonym(style="pin", **_emit_tier_flags("complete"))
    for smiles in (SIX[0], SIX[2]):
        assert complete.name_tiered(smiles)["tier"] == "abstain"


def test_a_name_that_ships_is_never_replaced_by_the_offer():
    """ (the Blue Book, `` Relative configuration: the stereodescriptors 'cis' and
    'trans'``): the best-effort tier names the pair with 'trans', a name OPSIN reads back in full. The
    default tier names it with the lowercase descriptors, a name the gate withdrew at the best-effort tier
    and holds: it is offered only to a molecule left without a name."""
    smiles = "O[C@H]1CC[C@H](O)CC1"
    assert name_best_effort(smiles)["name"] == "trans-cyclohexane-1,4-diol"
    assert default_tier_row(smiles)["name"] != "trans-cyclohexane-1,4-diol"


def test_the_row_is_deterministic():
    assert name_best_effort(SIX[2]) == name_best_effort(SIX[2])


BE_FLAGS = {k: v for k, v in _emit_tier_flags("best-effort").items() if k != "full_coverage"}


@pytest.mark.parametrize("smiles", SIX + THREE)
def test_the_caller_s_name_call_agrees_with_the_row(smiles):
    """``Orthonym.name`` at best-effort gives the name of the row (the default tier's), in the default mode of
    ``on_failure`` too (the decorator sees the failure before ``on_failure`` turns it into '')."""
    default = default_tier_row(smiles)
    assert default["tier"] == "pin_unverified", default
    engine = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert engine.name(smiles) == default["name"]


@pytest.mark.parametrize("smiles", [SIX[0], SIX[2]])
def test_every_public_entry_point_agrees(smiles):
    from orthonym import name_compound
    expected = default_tier_row(smiles)["name"]
    assert name_compound(smiles, **BE_FLAGS) == expected
    assert name_compound(smiles, on_failure="label", **BE_FLAGS) == expected
    assert name_compound(smiles, on_failure="raise", **BE_FLAGS) == expected         # no limit is raised
    assert name_compound(smiles, include_confidence=True, **BE_FLAGS)["name"] == expected
    engine = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    assert engine.name_with_confidence(smiles)["name"] == expected
    assert engine.name_tiered(smiles)["name"] == expected


def test_the_confidence_record_of_the_offered_name_is_the_default_tier_s():
    from orthonym import name_compound
    smiles = SIX[2]
    default = name_compound(smiles, include_confidence=True)
    offered = name_compound(smiles, include_confidence=True, **BE_FLAGS)
    keys = ("name", "confidence", "handler", "limit", "abstention", "verification")
    assert {k: offered[k] for k in keys} == {k: default[k] for k in keys}


@pytest.mark.parametrize("flags", ["valid", "complete"])
def test_the_valid_and_complete_tiers_keep_the_round_trip_contract_for_every_entry_point(flags):
    from orthonym import name_compound
    kwargs = {k: v for k, v in _emit_tier_flags(flags).items() if k != "full_coverage"}
    assert name_compound(SIX[2], **kwargs) == ""
    assert Orthonym(style="pin", **_emit_tier_flags(flags)).name_tiered(SIX[2])["tier"] == "abstain"


def test_a_name_that_ships_is_never_replaced_for_the_caller_s_name_call():
    from orthonym import name_compound
    assert name_compound("O[C@H]1CC[C@H](O)CC1", **BE_FLAGS) == "trans-cyclohexane-1,4-diol"


def test_the_default_tier_engine_has_the_options_of_the_engine_it_is_made_for():
    """One factory builds the strict twin of a wider tier and the engine of the offer: every construction
    option is carried, the four breadth flags are off."""
    from orthonym.assembly.default_tier_offer import default_tier_engine
    namer = Orthonym(style="pin", general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True, enable_group_splitting=True, trivial_fallback=True,
                     binding_proof="audit", _seed_excluded_dispatch_classes=frozenset({"X"}))
    attrs = ("style", "_enable_triviality_controller", "_enable_group_splitting", "_trivial_fallback",
             "_binding_proof", "_seed_excluded_dispatch_classes", "_disable_opsin_validity_gate",
             "_general_fallback", "_general_fallback_unverified", "_allow_aromatic_general", "_full_coverage")

    def options(engine):
        return {**{a: getattr(engine, a) for a in attrs}, "grammar": engine._grammar is None}
    built = default_tier_engine(namer)
    expected = {**options(namer), "_general_fallback": False, "_general_fallback_unverified": False,
                "_allow_aromatic_general": False, "_full_coverage": False}
    assert options(built) == expected
    assert options(built)["_enable_group_splitting"] and options(built)["_trivial_fallback"]
    assert options(built)["_binding_proof"] == "audit" and options(built)["_seed_excluded_dispatch_classes"] == {"X"}
    namer._strict_pin_twin_name("CCO")
    assert options(namer._pin_twin) == options(built)


# --- the offer, with a default tier that is a stub -------------------------------------------------

NAME = "(1r,4r)-cyclohexane-1,4-diol"
SMILES = "O[C@H]1CC[C@H](O)CC1"


class _Namer:
    """The attributes of an engine that ``default_tier_engine`` reads."""
    style = "pin"
    _general_fallback_unverified = True
    _trivial_fallback = False
    _grammar = object()
    _disable_opsin_validity_gate = False
    _enable_triviality_controller = False
    _enable_group_splitting = False
    _seed_excluded_dispatch_classes = frozenset()
    _binding_proof = "off"


def _default_row(**changes):
    row = {"name": NAME, "tier": "pin_unverified", "is_pin": False, "gate_outcome": pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY}
    row.update(changes)
    return row


def _stub_default(monkeypatch, row, built=None):
    from orthonym import namer as namer_module

    class Default:
        def __init__(self, **kwargs):
            if built is not None:
                built.append(kwargs)

        def name_tiered(self, smiles):
            assert smiles == SMILES
            if isinstance(row, Exception):
                raise row
            return row

    monkeypatch.setattr(namer_module, "Orthonym", Default)


@pytest.fixture
def held():
    default_tier_offer.clear_held()
    default_tier_offer.hold_withdrawn(NAME, SMILES, SMILES)
    yield
    default_tier_offer.clear_held()


ABSTAIN = {"name": None, "tier": "abstain"}


def test_the_offer_hands_over_the_default_tier_row(monkeypatch, held):
    row = _default_row()
    _stub_default(monkeypatch, row)
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, dict(ABSTAIN)) is row


@pytest.mark.parametrize("outcome", [
    "carveout:organometallic_additive",         # a known atom-drop hole: stays withdrawn at best-effort
    "carveout:inositol", "carveout:phane", "twin_verified:halido_o", pv.GATE_OUTCOME_SELF01,
    pv.GATE_OUTCOME_SUPPRESSED, pv.GATE_OUTCOME_NOT_RUN,
])
def test_only_the_stereo_layer_carve_out_is_offered(monkeypatch, held, outcome):
    _stub_default(monkeypatch, _default_row(gate_outcome=outcome))
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, dict(ABSTAIN)) == ABSTAIN


@pytest.mark.parametrize("changes", [
    {"name": "(1R,4R)-cyclohexane-1,4-diol"},   # not a name the gate withdrew for the molecule
    {"name": None, "tier": "abstain"},          # the default tier declines it
    {"name": "unknown organic compound"},       # a failure name
    {"tier": "abstain"},
])
def test_only_a_name_the_gate_withdrew_and_the_default_tier_ships_is_offered(monkeypatch, held, changes):
    _stub_default(monkeypatch, _default_row(**changes))
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, dict(ABSTAIN)) == ABSTAIN


def test_a_default_tier_that_raises_leaves_the_abstention(monkeypatch, held):
    _stub_default(monkeypatch, RuntimeError("boom"))
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, dict(ABSTAIN)) == ABSTAIN


def test_a_row_with_a_name_is_returned_without_naming_again(monkeypatch, held):
    built = []
    _stub_default(monkeypatch, _default_row(), built)
    row = {"name": "trans-cyclohexane-1,4-diol", "tier": "systematic_verified"}
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, row) is row
    assert not built


def test_the_tiers_below_best_effort_and_the_other_styles_take_no_offer(monkeypatch, held):
    built = []
    _stub_default(monkeypatch, _default_row(), built)
    complete = _Namer()
    complete._general_fallback_unverified = False
    general = _Namer()
    general.style = "general"
    for namer in (complete, general):
        assert default_tier_offer.offer_default_tier_row(namer, SMILES, dict(ABSTAIN)) == ABSTAIN
    assert not built


def test_nothing_held_nothing_offered(monkeypatch):
    built = []
    _stub_default(monkeypatch, _default_row(), built)
    default_tier_offer.clear_held()
    assert default_tier_offer.offer_default_tier_row(_Namer(), SMILES, dict(ABSTAIN)) == ABSTAIN
    assert not built


# --- what the gate holds ------------------------------------------------------------------------

def test_the_gate_holds_a_whole_molecule_name_with_a_stereo_layer_once():
    default_tier_offer.clear_held()
    hold = default_tier_offer.hold_withdrawn
    hold(NAME, SMILES, SMILES)
    hold(NAME, SMILES, SMILES)                                    # once
    hold("cyclohexane-1,4-diol", SMILES, SMILES)                  # no stereo layer: the carve-out cannot ship it
    hold("(1r)-chloroethane", "CC(Cl)Cl", SMILES)                 # the name of a fragment or another molecule
    hold("", SMILES, SMILES)
    hold(NAME, None, None)
    assert default_tier_offer.held_names() == (NAME,)
    default_tier_offer.clear_held()
    assert default_tier_offer.held_names() == ()


def test_the_held_names_are_bounded():
    default_tier_offer.clear_held()
    for n in range(2 * default_tier_offer._MAX_HELD):
        default_tier_offer.hold_withdrawn(f"({n}r)-x", SMILES, SMILES)
    assert len(default_tier_offer.held_names()) == default_tier_offer._MAX_HELD
    default_tier_offer.clear_held()


def test_a_tiered_naming_starts_without_the_names_of_the_one_before():
    from orthonym.assembly import spelling_offer
    default_tier_offer.clear_held()
    default_tier_offer.hold_withdrawn(NAME, SMILES, SMILES)
    seen = []
    row = spelling_offer.name_tiered_with_offer(
        _Namer(), "CCO", lambda smiles: seen.append(default_tier_offer.held_names()) or {"name": "ethanol", "tier": "pin_verified"})
    assert seen == [()] and row["name"] == "ethanol"


# --- the decorator of ``name``, with a stub default tier ---------------------------------------------

from orthonym.errors import OrthonymLimitError  # noqa: E402

FAILED = "unknown organic compound"


def _decorated(result, hold=(NAME,), raises=False):
    """A ``name`` that holds ``hold`` as the gate does, then fails (returns ``result`` or raises)."""
    calls = []

    @default_tier_offer.default_tier_name_scope
    def name(self, smiles, *, raise_on_limit=False):
        calls.append(smiles)
        for held_name in hold:
            default_tier_offer.hold_withdrawn(held_name, smiles, smiles)
        if raises:
            raise OrthonymLimitError("UNNAMEABLE", FAILED, None, smiles)
        return result
    return name, calls


def test_the_decorator_offers_the_default_tier_name_to_a_failed_call(monkeypatch):
    built = []
    _stub_default(monkeypatch, _default_row(), built)
    name, calls = _decorated(FAILED)
    namer = _Namer()
    assert name(namer, SMILES) == NAME and calls == [SMILES] and len(built) == 1
    assert default_tier_offer.is_offered(namer, NAME)
    assert not default_tier_offer.is_offered(namer, "(1R,4R)-cyclohexane-1,4-diol")
    complete = _Namer()
    complete._general_fallback_unverified = False
    assert not default_tier_offer.is_offered(complete, NAME)
    default_tier_offer.clear_held()
    assert not default_tier_offer.is_offered(namer, NAME)


def test_the_decorator_offers_to_a_call_that_raises_a_limit(monkeypatch):
    _stub_default(monkeypatch, _default_row())
    name, _ = _decorated(FAILED, raises=True)
    assert name(_Namer(), SMILES) == NAME
    name, _ = _decorated(FAILED, hold=(), raises=True)
    with pytest.raises(OrthonymLimitError):
        name(_Namer(), SMILES)


def test_the_decorator_leaves_a_name_and_a_withheld_failure_as_they_are(monkeypatch):
    built = []
    _stub_default(monkeypatch, _default_row(gate_outcome="carveout:organometallic_additive"), built)
    name, _ = _decorated("trans-cyclohexane-1,4-diol")
    assert name(_Namer(), SMILES) == "trans-cyclohexane-1,4-diol" and not built
    name, _ = _decorated(FAILED)
    assert name(_Namer(), SMILES) == FAILED and len(built) == 1             # the default tier's row is withheld


def test_the_decorator_starts_every_outermost_call_with_nothing_held(monkeypatch):
    _stub_default(monkeypatch, _default_row(name="(1r)-other"))
    default_tier_offer.hold_withdrawn(NAME, SMILES, SMILES)             # held by an earlier call
    name, _ = _decorated(FAILED, hold=())
    assert name(_Namer(), SMILES) == FAILED and default_tier_offer.held_names() == ()


def test_the_decorator_runs_only_the_callers_own_call_of_the_best_effort_pin_engine(monkeypatch):
    from orthonym import namer as namer_module
    from orthonym.assembly.fragment_naming import enter_name_scope, exit_name_scope
    built = []
    _stub_default(monkeypatch, _default_row(), built)
    complete, general = _Namer(), _Namer()
    complete._general_fallback_unverified = False
    general.style = "general"
    for namer in (complete, general):
        name, calls = _decorated(FAILED)
        assert name(namer, SMILES) == FAILED and calls == [SMILES]
    token = namer_module._BOOK_RETRY_OWNED.set(True)             # the call inside ``name_tiered``
    try:
        name, _ = _decorated(FAILED)
        assert name(_Namer(), SMILES) == FAILED
    finally:
        namer_module._BOOK_RETRY_OWNED.reset(token)
    enter_name_scope(SMILES)                                     # a nested call
    try:
        name, _ = _decorated(FAILED)
        assert name(_Namer(), SMILES) == FAILED
    finally:
        exit_name_scope()
    assert not built
