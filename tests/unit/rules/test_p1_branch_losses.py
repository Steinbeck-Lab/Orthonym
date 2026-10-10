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
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import assert_full_rt, name_best_effort
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "NCc1csc(-c2cccs2)n1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


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
    row = _dt_row(smiles)
    assert row["name"] == "1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine"
    # branch review fixes: a re-run name is labelled below the PIN (a breadth
    # producer built it), never pin_verified
    assert row["tier"] == "pin_unverified" and not row["is_pin"]
    assert_full_rt(row["name"], smiles)
    # every default-tier naming of the caller's molecule (the declined call and the
    # strict path's, tests/support/default_tier.py) re-runs in the outermost name
    # only
    assert depths and set(depths) == {(1, True)}, depths


# ChEBI rows whose best-effort naming made the helper re-runs (trace at 6a78ea183,
# fresh process: 1 and 15 nested promotion calls); names unchanged by the fix.
PROMOTION_HELPER_ROWS = [
    # ChEBI 45958
    ("NCCCC[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "glutaminyllysylphenylalanine"),
    # ChEBI 105164. properfix a performance pass (review I4/): the mononuclear-N amino
    # assembler (``substituent_naming._located_fg_hetero_root``) now encloses the
    # SECOND and further branch even when it is simple,
    # the Blue Book, "the second and further substituents are each enclosed
    # with parentheses even for simple substituents" -- 'methyl(phenyl)amino'
    #:26308), so the bare 'methylamino' after a compound acyl branch -- which
    # OPSIN's own grammar could misparse at the same boundary a bare 'formyl-
    # methylamino' does -- becomes '(methyl)amino'. RT-VERIFIED unchanged
    # (assert_full_rt below).
    ("CCCC[C@@H](C)C[C@@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N(C)[C@H]"
     "(C(=O)N1C[C@@H](O)C[C@H]1C(=O)O)C(C)C)[C@@H](C)OC(C)=O",
     "(2S,4S)-1-[(2S)-2-{[(2S,3R)-3-(acetyloxy)-2-{[(2S)-2-{[(2R,4R)-2,4-dimethyl-"
     "1-oxooctyl](methyl)amino}-4-methyl-1-oxopentyl]amino}-1-oxobutyl](methyl)amino}-"
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
# 6a78ea183: 1 nested run); name unchanged by the fix. Roadmap N5 (name-quality lane
# L2): 'hexyl', 'methylidene', 'methylethyl' (1), the Blue Book) and
# 'methoxy',:27667); was '...-8-[(1R,3S)-1,3-dimethylhexan-1-yl]-12-
# (methan-1-ylidene)-5-(1-methylethan-1-yl)-9-(1-oxaethan-1-yl)-...'.
NESTED_OFFER_ROW = (
    "C=C1C=CC(=O)NCC(=O)N[C@@H](C(C)C)C(=O)O[C@@H]([C@H](C)C[C@@H](C)CCC)"
    "[C@H](OC)C(=O)N1",
    "(5S,8S,9S)-8-[(1R,3S)-1,3-dimethylhexyl]-9-methoxy-5-(1-methylethyl)-"
    "12-methylidene-3,6,10,15-tetraoxo-7-oxa-1,4,11-triazacyclopentadec-13-ene")


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
    ("CO/C=C(C(=O)OC)\\C(C)=C/C=C/c1ccc(OCC=C(C)C)c(O)c1",
     "2-hydroxy-4-[(1E,3Z,5E)-6-methoxy-5-(methoxycarbonyl)-4-methylhexa-1,3,5-trien-"
     "1-yl]-1-[(3-methylbut-2-en-1-yl)oxy]benzene",
     [(1, False), (1, True)]),
    # dev2000
    # roadmap N5c (name-quality lane L2): the chain no longer ends on the S that roots
    # 'sulfanyl', the Blue Book); the hexavalent S keeps the
    # chain spelling (plan decision 5)
    ("C=CCC(=NOS(=O)(=O)O)S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
     "(2R,3S,4S,5R,6S)-6-{[4,4-dioxo-1-(prop-2-en-1-yl)-3,5-dioxa-4λ6-thia-2-"
     "azapent-1-en-1-yl]sulfanyl}-3,4,5-trihydroxy-2-(hydroxymethyl)oxane",
     [(1, False), (1, True)]),
    # The sulfonyl azo row of dev2000 ('O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3ccccc3)cc2)c1', once
    # '1-{[4-(phenylamino)phenyl]diazenyl}-3-sulfobenzene') is no longer a clean-offer row: leads
    # program L3 (43e) names it on the parent of its principal group, '3-[(4-anilinophenyl)diazenyl]
    # benzene-1-sulfonic acid', the Blue Book,:38791; '4-(phenyldiazenyl)
    # benzene-1-sulfonic acid (PIN)',:38798; 'anilino (preferred prefix)',:17800), so the main path
    # names it and the offer never runs. The row is a PIN in
    # tests/unit/assembly/test_leads_l3_43e_organyl_diazenyl.py. The two rows above still reach the offer.
]


@pytest.mark.parametrize("smiles,expected,offer_runs", CLEAN_OFFER_ROWS,
                         ids=["hexatrienyl", "glucosinolate"])
def test_clean_offer_names_are_kept(monkeypatch, smiles, expected, offer_runs):
    runs = _spy_clean_offer(monkeypatch)
    row = name_best_effort(smiles)
    assert row["name"] == expected
    assert row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], smiles)
    assert runs == offer_runs


# Roadmap N5 (name-quality lane L2): with the book spellings ('phenoxy', 'benzoyl',
# '(...)amino'; the Blue Book,:30446,:6465) the
# main path names a row of this class and the offer never runs. The offer is pinned on the
# row that still reaches it with the book spellings switched off (``mechanical_forms``): the
# glucosinolate row of ``CLEAN_OFFER_ROWS`` (the sulfonyl azo row went with leads program L3 43e,
# see ``CLEAN_OFFER_ROWS``), named with the mechanical spellings the switch keeps. Not PIN
# claims: the names are the systematic form.
MECHANICAL_OFFER_ROWS = [
    ("C=CCC(=NOS(=O)(=O)O)S[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
     "(2R,3S,4S,5R,6S)-6-[5,5-dioxo-2-(prop-2-en-1-yl)-4,6-dioxa-1,5\u03bb6-dithia-3-"
     "azahex-2-en-1-yl]-3,4,5-trihydroxy-2-(hydroxymethyl)oxane"),
]


@pytest.mark.parametrize("smiles,expected", MECHANICAL_OFFER_ROWS,
                         ids=["glucosinolate"])
def test_clean_offer_name_is_kept_mechanical_spelling(monkeypatch, smiles, expected):
    from orthonym.assembly.book_prefixes import mechanical_forms
    runs = _spy_clean_offer(monkeypatch)
    with mechanical_forms():
        row = name_best_effort(smiles)
    assert row["name"] == expected
    assert row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], smiles)
    assert runs == [(1, False), (1, True)]


def test_the_former_offer_row_with_an_n_substituted_benzamide_is_the_pin():
    # a dev split / milestone1500; once the clean-offer row "pyridine" ('2-(3-{4-[1-(cyclohexa-
    # 1,3,5-trien-1-yl)-1-oxamethyl]...}-4-oxa-2-azabut-3-en-1-yl)pyridine'): the benzamide
    # producer names the N-substituent, the Blue Book), so the strict
    # path builds the PIN and no offer runs
    smiles = "O=C(NCc1ccccn1)c1ccc(Oc2ccccc2)cc1"
    assert_pin_at_both_tiers(smiles, "4-phenoxy-N-[(pyridin-2-yl)methyl]benzamide")


# The ChEBI row the recipe lost first (ChEBI 18872, 119 heavy atoms): named
# with the published code's string, with no nested re-run and one memo-reading
# clean run (the fresh-memo run cost 12 s of the 90 s stall for the same string).
CHEBI_18872 = (
    "Cc1cccc(C[C@H](NC(=O)CNC(=O)[C@H](C)NC(=O)[C@@H](NC(=O)CN)C2CCCCC2)C(=O)N"
    "[C@@H](CC[C@H](CN)O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)C(=O)NCC(=O)N"
    "[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N2CCC[C@H]2C(=O)N"
    "[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](C(=O)O)[C@@H](C)O)c1")
# Roadmap N5 (name-quality lane L2): the carbon chains carry the heteroatoms as prefixes
#, the Blue Book), the rings take their book names:8482,
#:16290), '(aminoacetyl)amino' with its enclosing marks:7232).
# Was '3-{(2S,5S,11S,14S)-18-{(2S)-2-[(3S,9S,12S,13R)-3-(5-azapentan-1-yl)-...-19-oxa-
# 4,7,10,13,16-pentaazanonadec-18-en-1-yl}-1-methylcyclohexa-1,3,5-triene'.
CHEBI_18872_NAME = (
    "3-[(2S)-2-[(2-{[(2S)-2-({(2S)-2-[(aminoacetyl)amino]-2-cyclohexyl-1-oxoethyl}"
    "amino)-1-oxopropyl]amino}-1-oxoethyl)amino]-3-{[(1S,4R)-5-amino-1-{[(2-{[(1S)-1-"
    "({[(1S)-4-amino-1-{[(2-{(2S)-2-[(3S,9S,12S,13R)-3-(4-aminobutyl)-12-carboxy-13-"
    "hydroxy-9-(3-hydroxy-3-oxopropyl)-1,4,7,10-tetraoxo-2,5,8,11-tetraazatetradecan-"
    "1-yl]pyrrolidin-1-yl}-2-oxoethyl)amino](oxo)methyl}-4-oxobutyl]amino}(oxo)methyl)"
    "-4-hydroxy-4-oxobutyl]amino}-2-oxoethyl)amino](oxo)methyl}-4-{[(2R,3R,4S,5R,6R)-"
    "3,4,5-trihydroxy-6-(hydroxymethyl)oxan-2-yl]oxy}pentyl]amino}-3-oxopropyl]-1-"
    "methylbenzene")


def test_chebi_18872_names_without_the_repeated_work(monkeypatch):
    promo = _spy_promotion(monkeypatch, CHEBI_18872)
    runs = _spy_clean_offer(monkeypatch)
    row = Orthonym(**_BE).name_tiered(CHEBI_18872)
    assert row["name"] == CHEBI_18872_NAME
    assert row["tier"] == "systematic_verified"
    assert_full_rt(row["name"], CHEBI_18872)
    assert all(depth == 1 for depth, _ in promo), promo
    assert runs == [(1, False)]
