"""Roadmap N5, the two guards around the book-form writers.

1. A general spelling the writers can give is never labelled the PIN
   (``rules.pin_vocabulary.non_pin_vocabulary``):
   * (the Blue Book) 'benzoyl (preferred prefix) benzenecarbonyl
     oxo(phenyl)methyl'; (:30628) 'cyclohexanecarbonyl (preferred prefix)
     cyclohexylcarbonyl cyclohexyl(oxo)methyl';
   * (:30624) with 'pyrrolidine-1-carboxylic acid (PIN)' (:29892): the
     preferred prefix is '<ring>-<n>-carbonyl', not '(azetidin-1-yl)carbonyl';
   * (:32991-:33007): the amido prefix ('cyclohexanecarboxamido') in a PIN;
   * (:24412,:24414) 'tert-butyl', 'benzyl'; (:24591) 'benzyloxy'
     (preferred prefixes).
2. A book spelling never costs a name: when a molecule named with the book spellings is
   left without a verified name and a book spelling was used, ``Orthonym.name`` and
   ``Orthonym.name_tiered`` name it again with ``book_prefixes.mechanical_forms``.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import book_prefixes as bp
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.rules.pin_vocabulary import non_pin_vocabulary
from tests.support.rt_assert import name_is_rt_exact


@pytest.mark.parametrize("name", [
    "4-[(benzyloxy)carbonyl]phenol",
    "(tert-butoxycarbonyl)amino",
    "1-[(4-chlorophenyl)methoxy]-4-nitrobenzene",
    "2-({[(9H-fluoren-9-yl)methoxy]carbonyl}amino)benzoic acid",
    "(methoxycarbonyl)amino",
    "2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid",    # PIN,:33354
    "4-(azetidine-1-carbonyl)benzoic acid",
    "cyclohexanecarbonyl",
    "1,1'-[(phenylmethylene)bis(sulfanediylmethylene)]dibenzene",  # a BB PIN string
    "[(4-chlorophenyl)methylidene]amino",                          # substituted,
    "diphenylmethyl",
    "cyclohexanecarboxamido",
])
def test_a_book_spelling_is_left_alone(name):
    assert non_pin_vocabulary(name) is None


def test_a_book_spelling_is_noted():
    with bp.watch_book_forms() as fired:
        assert bp.methyl_group_name(["fluoro"] * 3) == "trifluoromethyl"
    assert fired[0] is True
    with bp.watch_book_forms() as fired:
        pass
    assert fired[0] is False


CF3_THIOUREA = "FC(F)(F)c1ccccc1NC(=S)N1CCCC1"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["valid", "best-effort"])
def test_a_failing_book_spelling_never_costs_the_name(monkeypatch, tier):
    """A writer whose book spelling describes a different group (here every methyl group
    becomes 'pentyl') makes the book-spelling candidates fail the round trip; the row
    still gets a name that reads back to the input, without the bad spelling."""
    calls = [0]

    def poisoned(branch_names, bond_order=1):
        calls[0] += 1
        bp.note_book_form()
        return {0: "pentane", 1: "pentyl", 2: "pentylidene"}.get(bond_order)

    monkeypatch.setattr(bp, "methyl_group_name", poisoned)
    o = Orthonym(style="pin", **_emit_tier_flags(tier))
    row = o.name_tiered(CF3_THIOUREA)
    assert calls[0] > 0                    # the poisoned writer was reached
    name = row.get("name")
    assert name and "pentyl" not in name, row
    assert name_is_rt_exact(name, CF3_THIOUREA), row
    plain = o.name(CF3_THIOUREA)
    assert not is_failure_name(plain) and "pentyl" not in plain
    assert name_is_rt_exact(plain, CF3_THIOUREA)


NILOTINIB = ("CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)"
             "C=1C=NC=CC1")


@pytest.mark.opsin_gate
def test_memo_verify_mode_finds_no_mismatch_with_the_mechanical_pre_pass(monkeypatch):
    """The terminal-fragment writer names a composite fragment with its mechanical
    spellings first, inside the memo scope of the book spellings
    (``terminal_fragment._same_reach_name``). The memo keys carry the switch
    (``assembly.memo.book_forms_var``), so ORTHONYM_MEMO=verify finds no stored value that
    differs from a fresh one, and the name is the same with the memo on and in verify
    mode. (The key-level known positive is ``test_nq_forms_n5f.py::
    test_the_memo_keys_carry_the_book_forms_switch``.)"""
    from orthonym.assembly import memo
    import orthonym.rules.terminal_fragment as tf
    pre_passes = [0]
    orig = tf._same_reach_name

    def spy(*args, **kwargs):
        if bp.book_forms_enabled():
            pre_passes[0] += 1
        return orig(*args, **kwargs)
    monkeypatch.setattr(tf, "_same_reach_name", spy)
    names = {}
    for mode in ("on", "verify"):
        monkeypatch.setattr(memo, "_MODE", mode)
        memo.clear_process_cache()
        memo.reset_verify_mismatches()
        names[mode] = Orthonym(style="pin", **_emit_tier_flags("valid")).name_tiered(
            NILOTINIB).get("name")
        assert memo.verify_mismatch_count() == 0, mode
    assert pre_passes[0] > 0
    assert names["on"] == names["verify"], names
    assert name_is_rt_exact(names["on"], NILOTINIB), names


def test_the_mechanical_forms_switch_every_writer_back():
    from orthonym.rules.terminal_fragment import terminal_fragment_name
    mol = Chem.MolFromSmiles("FC(F)(F)c1ccccc1")
    assert terminal_fragment_name(mol, {0, 1, 2, 3}, 1).name == "trifluoromethyl"
    with bp.mechanical_forms():
        assert terminal_fragment_name(mol, {0, 1, 2, 3}, 1).name == "1,1,1-trifluoromethyl"


# --- the retry itself: when it runs and when it must not ----------------------------------
# ``Orthonym._name_with_book_retry(run)`` calls ``run`` (one outermost ``name`` call)
# once, and a second time inside ``mechanical_forms`` when it retries. A fake run records
# whether it ran with the book spellings (True) or the mechanical ones (False).

def _fake_run(result_for):
    calls = []

    def run():
        book = bp.book_forms_enabled()
        calls.append(book)
        return result_for(book)
    return run, calls


def test_a_failed_run_with_a_book_spelling_is_named_again():
    def result_for(book):
        if book:
            bp.note_book_form()
            return "unknown organic compound"
        return "ethanol"
    run, calls = _fake_run(result_for)
    o = Orthonym(style="pin", general_fallback=True)
    assert o._name_with_book_retry(run) == "ethanol"
    assert calls == [True, False]


def test_a_run_without_a_book_spelling_is_not_named_again():
    run, calls = _fake_run(lambda book: "unknown organic compound")
    o = Orthonym(style="pin", general_fallback=True)
    assert is_failure_name(o._name_with_book_retry(run))
    assert calls == [True]


def test_no_second_run_after_a_hang_budget_trip():
    from orthonym.assembly import fragment_naming as fn

    def result_for(book):
        bp.note_book_form()
        fn._count_hang_budget_trip()
        return "unknown organic compound"
    run, calls = _fake_run(result_for)
    o = Orthonym(style="pin", general_fallback=True)
    assert is_failure_name(o._name_with_book_retry(run))
    assert calls == [True]


@pytest.mark.opsin_gate
def test_no_second_run_at_the_default_tier():
    """The default tier emits only the strict path's pin_verified names; the retry is
    for the wider tiers (``Orthonym._book_retry_applies``)."""
    def result_for(book):
        bp.note_book_form()
        return "unknown organic compound"
    run, calls = _fake_run(result_for)
    o = Orthonym(style="pin")
    assert o._book_retry_applies() is False
    o._name_with_book_retry(run)
    assert calls == [True]


def test_a_limit_error_is_raised_when_the_second_run_finds_no_name_either():
    from orthonym.errors import OrthonymLimitError

    def run():
        bp.note_book_form()
        raise OrthonymLimitError("UNNAMEABLE", "unknown organic compound")
    o = Orthonym(style="pin", general_fallback=True)
    with pytest.raises(OrthonymLimitError):
        o._name_with_book_retry(run)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier,runs", [("pin", 1), ("valid", 2), ("best-effort", 2)])
def test_name_tiered_retries_a_lost_name_at_the_wider_tiers(monkeypatch, tier, runs):
    calls = []

    def spy(self, smiles):
        calls.append(bp.book_forms_enabled())
        bp.note_book_form()
        return {"name": "unknown organic compound", "tier": "abstain",
                "limit_code": "UNNAMEABLE"}
    monkeypatch.setattr(Orthonym, "_name_tiered_scoped", spy)
    Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered("CCO")
    assert calls == [True, False][:runs]


def test_a_top_level_name_runs_its_body_at_depth_one(monkeypatch):
    """The PIN tier's promotion re-run requires ``name_scope_depth == 1`` in the body
    of the caller's call (``Orthonym._pin_promotion_eligible``); the retry must not nest
    the body one level deeper."""
    from orthonym.assembly import fragment_naming as fn
    depths = []
    orig = Orthonym._pin_promotion_eligible

    def spy(self):
        depths.append(fn.name_scope_depth())
        return orig(self)
    monkeypatch.setattr(Orthonym, "_pin_promotion_eligible", spy)
    Orthonym(style="pin").name("CCO")
    assert depths and depths[0] == 1
