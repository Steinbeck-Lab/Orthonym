"""P1 -- branch losses at corpus scale (TRIAGE.md 'P1 -- branch losses at corpus scale').

The paper's ChEBI recipe (Zenodo launch_generic.py: 30 processes, 45 s SIGALRM,
90 s stall) run on the breadth branch at a9a9fb858 lost nine big ChEBI peptides
(116-209 heavy atoms; ChEBI rows 18862-18880 and 38428) to the stall. The code
the paper's corpus runs were re-checked with names all nine; the
branch named them with the same strings, 1.7-2.5x slower (fresh process, side
by side with main 2b1455817: 102-152 s against 60-79 s).

Two whole-molecule policies the branch added ran in NESTED ``name`` frames.
Both asked ``is_top_level_naming``, which reads the fragment-recursion depth:
that depth is 0 while any ``name`` dispatches its whole molecule, and
``isolated_naming_session`` resets it to 0 for the recovery lane, the rescues
and the clean fall-through. So a helper instance created by a producer read as
top level:

  A the PIN tier's promotion re-run: helper instances in the default
     configuration -- the peptide handler's substitutive sub-namer and the
     default-tier ``name_compound`` instances -- named what their first run could
     not name a second time with the promoted producers.
  B the clean-first offer over a floor stand-in: the best-effort
     ``name_compound`` instances whose own recovery produced a stand-in re-ran a
     whole clean recovery for their sub-structure.

Both now require the outermost ``name`` (``name_scope_depth == 1``). At the
PIN tier the helpers never re-ran (the wrapper flag is set for the whole outer
call), so the PIN tier is unchanged by A; the names B contributes all come from
the outermost frame (13 of 13 dev-set rows).

  C in the outermost frame the clean-first offer re-derived every fragment of
     the molecule on a fresh memo: 12.3 s and 43.8 s on two of the nine peptides,
     for a result identical to the stand-in. It now first runs a probe: the clean
     recovery reading the molecule's memos without writing them (a throwaway copy
     of the fragment memo, a scope-memo sandbox, its budget spend dropped). The
     probe ships only the final rung's universal floor (every engine rung
     declined even with the molecule's fragment names at hand); an engine name
     built on the memo's best-effort fragment names never ships, and otherwise
     the fresh-memo recovery runs exactly as before.

No Blue Book rule is involved (engine control flow); every name is unchanged.
Witnesses are ChEBI rows and dev-set rows; none is in eval/splits/a holdout split.json.
Every name is read back by an independent OPSIN call to the input's full
InChIKey (tests/support/rt_assert).
"""
import pytest

from orthonym import Orthonym
from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

_BE = dict(general_fallback=True, general_fallback_unverified=True,
           allow_aromatic_general=True)


def _spy_promotion(monkeypatch, caller_smiles):
    """Record, at every PIN-promotion wrapper call, ``name_scope_depth`` and
    whether the molecule named is the caller's (``caller_smiles``)."""
    from rdkit import Chem
    caller = Chem.CanonSmiles(caller_smiles)
    depths = []
    orig = Orthonym._name_with_pin_promotion

    def spy(self, smiles, *, raise_on_limit):
        depths.append((fn.name_scope_depth(), Chem.CanonSmiles(smiles) == caller))
        return orig(self, smiles, raise_on_limit=raise_on_limit)

    monkeypatch.setattr(Orthonym, "_name_with_pin_promotion", spy)
    return depths


def _spy_clean_offer(monkeypatch):
    """Record, for every clean recovery the clean-first offer runs, the
    ``name_scope_depth`` of the offer and whether that run used a fresh memo."""
    runs = []
    in_offer = []
    orig_offer = Orthonym._prefer_clean_over_floor_substitute
    orig_ft = Orthonym._try_besteffort_clean_general_fallthrough

    def offer(self, smiles, floor_name):
        in_offer.append(fn.name_scope_depth())
        try:
            return orig_offer(self, smiles, floor_name)
        finally:
            in_offer.pop()

    def ft(self, smiles, *, reset_cache=True):
        if in_offer:
            runs.append((in_offer[-1], reset_cache))
        if reset_cache:  # the call the code before this fix made
            return orig_ft(self, smiles)
        return orig_ft(self, smiles, reset_cache=False)

    monkeypatch.setattr(Orthonym, "_prefer_clean_over_floor_substitute", offer)
    monkeypatch.setattr(Orthonym, "_try_besteffort_clean_general_fallthrough", ft)
    return runs


# --------------------------------------------------------------------------
# A the PIN-promotion re-run belongs to the outermost name
# --------------------------------------------------------------------------

def test_promotion_is_eligible_in_the_outermost_name_only():
    """The fragment depth cannot tell a helper's name from the caller's; the
    name-scope depth can."""
    namer = Orthonym()
    assert fn.name_scope_depth() == 0
    fn.enter_name_scope()                     # the caller's name: its _budget_scope
    try:
        assert fn.is_top_level_naming()
        assert namer._pin_promotion_eligible()
        fn.enter_name_scope()                 # a helper instance's name inside it
        try:
            assert fn.is_top_level_naming()   # same fragment depth...
            assert not namer._pin_promotion_eligible()  #... but not the caller's molecule
        finally:
            fn.exit_name_scope()
    finally:
        fn.exit_name_scope()


def test_the_pin_tier_still_promotes_the_caller_s_molecule(monkeypatch):
    """Breadth job 1 witness: the PIN tier's re-run names it, in the outermost
    name and nowhere else."""
    smiles = "NCc1csc(-c2cccs2)n1"
    depths = _spy_promotion(monkeypatch, smiles)
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == "[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine"
    # branch review fixes: a re-run name is labelled below the PIN (a breadth
    # producer built it), never pin_verified
    assert row["tier"] == "pin_unverified" and not row["is_pin"]
    assert_full_rt(row["name"], smiles)
    assert depths == [(1, True)]


# ChEBI rows whose best-effort naming made the helper re-runs (trace at 6a78ea183,
# fresh process: 1 and 15 nested promotion calls); names unchanged by the fix.
PROMOTION_HELPER_ROWS = [
    # ChEBI 45958
    ("NCCCC[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "glutaminyllysylphenylalanine"),
    # ChEBI 105164
    ("CCCC[C@@H](C)C[C@@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N(C)[C@H]"
     "(C(=O)N1C[C@@H](O)C[C@H]1C(=O)O)C(C)C)[C@@H](C)OC(C)=O",
     "(2S,4S)-1-[(2S)-2-{[(2S,3R)-3-(acetyloxy)-2-{[(2S)-2-{[(2R,4R)-2,4-dimethyl-"
     "1-oxooctyl]methylamino}-4-methyl-1-oxopentyl]amino}-1-oxobutyl]methylamino}-"
     "3-methyl-1-oxobutyl]-4-hydroxypyrrolidine-2-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", PROMOTION_HELPER_ROWS,
                         ids=["chebi-45958", "chebi-105164"])
def test_best_effort_helpers_make_no_promotion_rerun(monkeypatch, smiles, expected):
    depths = _spy_promotion(monkeypatch, smiles)
    row = name_best_effort(smiles)
    assert row["name"] == expected
    assert_full_rt(row["name"], smiles)
    # The best-effort caller is not eligible itself, and no helper frame is (a
    # strict-twin naming by name_tiered would be an outermost call, depth 1).
    assert all(depth == 1 for depth, _ in depths), depths




# --------------------------------------------------------------------------
# B the clean-first offer belongs to the outermost name
# --------------------------------------------------------------------------

def _stub_fallthrough(monkeypatch, results):
    """Replace the clean recovery with a stub that returns ``results`` in turn:
    (name, is_floor_stand_in, is_final_floor) triples, or an exception to raise."""
    calls = []

    def stub(self, smiles, *, reset_cache=True):
        calls.append({"reset_cache": reset_cache, "depth": fn.name_scope_depth()})
        out = results[len(calls) - 1]
        if isinstance(out, BaseException):
            raise out
        name, stand_in, final_floor = out
        self._last_recovery_floor_substitute = stand_in
        self._last_recovery_final_floor = final_floor
        return name

    monkeypatch.setattr(Orthonym, "_try_besteffort_clean_general_fallthrough", stub)
    return calls


_RING = "O=C(NCc1ccccn1)c1ccc(Oc2ccccc2)cc1"


def test_nested_frame_keeps_the_stand_in_without_a_clean_run(monkeypatch):
    calls = _stub_fallthrough(monkeypatch, [("clean", False, True)] * 2)
    namer = Orthonym(**_BE)
    fn.enter_name_scope()                     # the caller's name
    try:
        fn.enter_name_scope()                 # a nested best-effort name_compound
        try:
            assert fn.is_top_level_naming()
            assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "floor"
            assert calls == []
        finally:
            fn.exit_name_scope()
        assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "clean"
        assert len(calls) == 1
    finally:
        fn.exit_name_scope()


# ChEBI 105698: its best-effort naming ran the offer in a nested name (trace at
# 6a78ea183: 1 nested run); name unchanged by the fix.
NESTED_OFFER_ROW = (
    "C=C1C=CC(=O)NCC(=O)N[C@@H](C(C)C)C(=O)O[C@@H]([C@H](C)C[C@@H](C)CCC)"
    "[C@H](OC)C(=O)N1",
    "(5S,8S,9S)-8-[(1R,3S)-1,3-dimethylhexan-1-yl]-12-(methan-1-ylidene)-5-"
    "(1-methylethan-1-yl)-9-(1-oxaethan-1-yl)-3,6,10,15-tetraoxo-7-oxa-1,4,11-"
    "triazacyclopentadec-13-ene")


def test_no_clean_run_in_a_nested_frame(monkeypatch):
    runs = _spy_clean_offer(monkeypatch)
    smiles, expected = NESTED_OFFER_ROW
    row = name_best_effort(smiles)
    assert row["name"] == expected
    assert_full_rt(row["name"], smiles)
    assert all(depth == 1 for depth, _ in runs), runs


# --------------------------------------------------------------------------
# C the outermost offer reads the molecule's memos first, a fresh memo second
# --------------------------------------------------------------------------

def test_memo_reading_run_ships_the_final_floor(monkeypatch):
    """Every engine rung declined even with the molecule's fragment names: the
    probe's universal floor ships and no fresh-memo run follows."""
    calls = _stub_fallthrough(monkeypatch, [("final floor", False, True),
                                            ("fresh", False, False)])
    namer = Orthonym(**_BE)
    fn.enter_name_scope()
    try:
        assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "final floor"
    finally:
        fn.exit_name_scope()
    assert [c["reset_cache"] for c in calls] == [False]


@pytest.mark.parametrize("first", [("engine rung on the memo", False, False),
                                   ("again a stand-in", True, False),
                                   (None, False, False), fn.PerfBudgetExceeded()],
                         ids=["engine-rung", "stand-in", "no-name", "budget-trip"])
def test_fresh_memo_run_follows_a_failed_memo_reading_run(monkeypatch, first):
    calls = _stub_fallthrough(monkeypatch, [first, ("fresh", False, False)])
    namer = Orthonym(**_BE)
    fn.enter_name_scope()
    try:
        assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "fresh"
    finally:
        fn.exit_name_scope()
    assert [c["reset_cache"] for c in calls] == [False, True]


def test_both_runs_failing_keep_the_stand_in(monkeypatch):
    _stub_fallthrough(monkeypatch, [("a", True, False), ("b", True, False)])
    namer = Orthonym(**_BE)
    fn.enter_name_scope()
    try:
        assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "floor"
        assert namer._last_recovery_floor_substitute is True
    finally:
        fn.exit_name_scope()


def test_memo_reading_run_leaves_no_trace(monkeypatch):
    """The first run reads what the molecule's memos hold, and every write and
    budget unit it spends is dropped, so the fresh-memo run starts from exactly
    the state it started from before the first run existed."""
    seen = {}

    def stub(self, smiles, *, reset_cache=True):
        if not reset_cache:
            seen["read"] = fn._fragment_guard.cache.get("C")
            fn._fragment_guard.cache["CC"] = "written by the first run"
            fn.spend_analysis_call(7)
            memo.cache_or_compute("p1-probe", ("k",), lambda: "first")
            self._last_recovery_floor_substitute = True
            self._last_recovery_final_floor = False
            return "stand-in again"
        seen["fresh_budget"] = fn._fragment_guard.analysis_budget
        seen["fresh_memo"] = memo.cache_or_compute("p1-probe", ("k",), lambda: "fresh")
        self._last_recovery_floor_substitute = False
        return "fresh"

    monkeypatch.setattr(Orthonym, "_try_besteffort_clean_general_fallthrough", stub)
    namer = Orthonym(**_BE)
    token = memo.push_scope()
    fn.enter_name_scope()
    try:
        fn._fragment_guard.cache["C"] = "methane"
        budget = fn._fragment_guard.analysis_budget
        assert namer._prefer_clean_over_floor_substitute(_RING, "floor") == "fresh"
        assert seen["read"] == "methane"
        assert "CC" not in fn._fragment_guard.cache
        assert seen["fresh_budget"] == budget
        assert fn._fragment_guard.analysis_budget == budget
        assert seen["fresh_memo"] == "fresh"
    finally:
        fn.exit_name_scope()
        memo.pop_scope(token)


# dev-set rows whose floor stand-in the offer replaces with a clean engine name
# (TRIAGE 'Breadth -- PubChem losses', class C), each named by the fresh-memo
# run after the probe: the probe gives the same engine name for the first, the
# stand-in back for the next two, and for the fourth an engine name built on the
# memo's best-effort fragment names ('...-3-(2-hydroxy-1,3-dioxa-2-thiapropa-
# 1,2-dienyl)benzene'), which must not ship.
CLEAN_OFFER_ROWS = [
    # a dev split / milestone1500
    ("O=C(NCc1ccccn1)c1ccc(Oc2ccccc2)cc1",
     "2-(3-{4-[1-(cyclohexa-1,3,5-trien-1-yl)-1-oxamethyl]cyclohexa-1,3,5-trien-1-yl}"
     "-4-oxa-2-azabut-3-en-1-yl)pyridine",
     [(1, False), (1, True)]),
    # a dev split / milestone1500
    ("CO/C=C(C(=O)OC)\\C(C)=C/C=C/c1ccc(OCC=C(C)C)c(O)c1",
     "2-hydroxy-4-[(1E,3Z,5E)-6-methoxy-5-(methoxycarbonyl)-4-methylhexa-1,3,5-trien-"
     "1-yl]-1-[(3-methylbut-2-en-1-yl)oxy]benzene",
     [(1, False), (1, True)]),
    # dev2000
    ("C=CCC(=NOS(=O)(=O)O)S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
     "(2R,3S,4S,5R,6S)-6-[5,5-dioxo-2-(prop-2-en-1-yl)-4,6-dioxa-1,5λ6-dithia-3-"
     "azahex-2-en-1-yl]-3,4,5-trihydroxy-2-(hydroxymethyl)oxane",
     [(1, False), (1, True)]),
    # dev2000
    ("O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3ccccc3)cc2)c1",
     "1-(2-{4-[1-(cyclohexa-1,3,5-trien-1-yl)-1-azamethyl]cyclohexa-1,3,5-trien-1-yl}-"
     "1,2-diazaeth-1-en-1-yl)-3-(1,1-dioxo-2-oxa-1λ6-thiaethyl)benzene",
     [(1, False), (1, True)]),
]


@pytest.mark.parametrize("smiles,expected,offer_runs", CLEAN_OFFER_ROWS,
                         ids=["pyridine", "hexatrienyl", "glucosinolate", "sulfonyl"])
def test_clean_offer_names_are_kept(monkeypatch, smiles, expected, offer_runs):
    runs = _spy_clean_offer(monkeypatch)
    row = name_best_effort(smiles)
    assert row["name"] == expected
    assert row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], smiles)
    assert runs == offer_runs


# The ChEBI row the recipe lost first (ChEBI 18872, 119 heavy atoms): named
# with the published code's string, with no nested re-run and one memo-reading
# clean run (the fresh-memo run cost 12 s of the 90 s stall for the same string).
CHEBI_18872 = (
    "Cc1cccc(C[C@H](NC(=O)CNC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)CN)C2CCCCC2)C(=O)N"
    "[C@@H](CC[C@H](CN)O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)C(=O)NCC(=O)N"
    "[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N2CCC[C@H]2C(=O)N"
    "[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](C(=O)O)[C@@H](C)O)c1")
CHEBI_18872_NAME = (
    "3-{(2S,5S,11S,14S)-18-{(2S)-2-[(3S,9S,12S,13R)-3-(5-azapentan-1-yl)-12-carboxy-"
    "13-hydroxy-9-(3-hydroxy-4-oxabut-3-en-1-yl)-1,4,7,10-tetraoxo-2,5,8,11-"
    "tetraazatetradecan-1-yl]-1-azacyclopentan-1-yl}-2-[(6S,9S)-9-(cyclohexan-1-yl)-"
    "6-methyl-2,5,8,11-tetraoxo-1,4,7,10,13-pentaazatridecan-1-yl]-11-(3-hydroxy-4-"
    "oxabut-3-en-1-yl)-3,6,9,12,15-pentaoxo-14-(3-oxo-4-azabutan-1-yl)-5-[(3R)-3-"
    "{1-[(1R,3R,4R,5S,6R)-4,5,6-trihydroxy-3-(2-oxaethan-1-yl)-2-oxacyclohexan-1-yl]"
    "-1-oxamethan-1-yl}-5-azapentan-1-yl]-19-oxa-4,7,10,13,16-pentaazanonadec-18-en-"
    "1-yl}-1-methylcyclohexa-1,3,5-triene")


def test_chebi_18872_names_without_the_repeated_work(monkeypatch):
    promo = _spy_promotion(monkeypatch, CHEBI_18872)
    runs = _spy_clean_offer(monkeypatch)
    row = Orthonym(**_BE).name_tiered(CHEBI_18872)
    assert row["name"] == CHEBI_18872_NAME
    assert row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], CHEBI_18872)
    assert all(depth == 1 for depth, _ in promo), promo
    assert runs == [(1, False)]
