"""Branch review fixes (2026-09-28): the findings of the whole-branch review of
``breadth-program``, one section per class.

Every name asserted here is read back by a FRESH OPSIN call that does not go
through the engine (``tests.support.rt_assert.assert_full_rt``, full InChIKey).
Fixtures are dev-set rows (a dev split / milestone1500 / dev2000), Blue Book rows or
minimal analogues, never a holdout split rows.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.rt_assert import assert_full_rt
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
    "c1cccc(-c2ccc(cc2)CO[C@H]2CC([C@@H]([C@H]2CC/C=C\\CCC(=O)O)N2CCOCC2)=O)c1",
    "c12c(C/C=C(/CO)C)c(cc(O)c2C(=O)C[C@@H](c2cc(c(cc2)O)O)O1)O",
    "C/C(=C(/CC(=O)OC)\\C(=O)OC)/NC1CCCCC1",
    "C/C(=C\\Cc1c(O)cc(O)c2c1O[C@H](c1ccc(O)c(O)c1)CC2=O)CO",
    "CC(=S)c1ccc(C(=O)O)cc1",
    "COC(=O)CC(=C(C)N)C(=O)OC",
    "COc1c(C)cnc(CS(=O)c2nc3ccc(O)cc3[nH]2)c1C",
    "COc1ccc(/C=C2\\NC(=O)/C(=C/c3ccccc3)NC2=O)cc1",
    "COc1ccc(/C=c2\\[nH]c(=O)/c(=C/c3ccccc3)[nH]c2=O)cc1",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "COc1cccc2c1C(=O)/C(=C(\\CO)[C@@H]1OC(=O)C[C@@H]1C)O2",
    "C[C@@H](CN(C[C@@H]1CCC=CC1)C)O",
    "C[C@H](CN(C[C@@H]1CCC=CC1)C[C@@H](C)O)O",
    "NCc1csc(-c2cccs2)n1",
    "N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O",
    "O=C(CCc1cccc(OS(=O)(=O)O)c1)c1ccc(O)c(O)c1",
    "O=C(O)CC/C=C\\CC[C@H]1[C@@H](OCc2ccc(-c3ccccc3)cc2)CC(=O)[C@@H]1N1CCOCC1",
    "O=C1CC(C2CCC(=O)O2)Oc2ccccc21",
    "O=C[C@H](O)[C@@H](O)[C@H](O)COP(=O)(O)O",
    "O=S(=O)(O)OC/C=C/c1ccc(O)cc1",
    "O=S(=O)(O)Oc1ccc2cccc(O)c2n1",
    "OC(=O)Cc1c[nH]c(=O)[nH]c1=O",
    "OC(=O)Cn1cc(C)c(=O)[nH]c1=O",
    "OC(=O)Cn1ccc(=O)[nH]c1=O",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "c12c(C/C=C(/CO)C)c(cc(O)c2C(=O)C[C@@H](c2cc(c(cc2)O)O)O1)O",
    "C/C(=C(/CC(=O)OC)\\C(=O)OC)/NC1CCCCC1",
    "C/C(=C\\Cc1c(O)cc(O)c2c1O[C@H](c1ccc(O)c(O)c1)CC2=O)CO",
    "CC(=S)c1ccc(C(=O)O)cc1",
    "COC(=O)CC(=C(C)N)C(=O)OC",
    "C[C@@H](CN(C[C@@H]1CCC=CC1)C)O",
    "O=S(=O)(O)OC/C=C/c1ccc(O)cc1",
})


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _row(smiles, tier):
    if tier == "pin" and smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


# ---------------------------------------------------------------------------
# A. A name only the PIN tier's promotion re-run builds is not certified as the
# PIN: pin_unverified, is_pin False, at the PIN tier and (through the strict
# twin, which no longer runs the re-run) at best-effort. Names unchanged.
# Paper, tier labels: "The label pin_unverified means a name in PIN form that
# only a breadth producer built". The re-run's producers are the best-effort
# composition branches (allow_mancude), and its names include right-molecule
# non-PINs: '4-(1-sulfanylideneethyl)benzoic acid' for '4-(ethanethioyl)benzoic
# acid (PIN)', the Blue Book); '...methyl]sulfinyl}' for
# '...methanesulfinyl',:28090-28094); an oxolan-2-one parent beside a
# senior 1-benzopyran-4-one,:29628; (d)).
# ---------------------------------------------------------------------------

RERUN_ONLY = [
    # a dev split / milestone1500
    "COc1c(C)cnc(CS(=O)c2nc3ccc(O)cc3[nH]2)c1C",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    # dev2000
    "N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O",
    "O=S(=O)(O)OC/C=C/c1ccc(O)cc1",
    # Blue Book row and a minimal analogue
    "CC(=S)c1ccc(C(=O)O)cc1",
    "O=C1CC(C2CCC(=O)O2)Oc2ccccc21",
]

# Paper conformance (user decision 2026-09-30; Methods, "Tiers": "The label
# systematic_verified means a correct systematic name that is not the PIN"): a
# re-run name that also carries a part the code records as not the PIN is
# systematic_verified. The dev2000 row's name cites '(sulfooxy)', the ester prefix of
# a noncarbon oxoacid under a class junior to esters "Esters of
# mononuclear noncarbon oxoacids", the Blue Book). The other re-run names
# stay pin_unverified.
_RERUN_KNOWN_NON_PIN = {"O=S(=O)(O)OC/C=C/c1ccc(O)cc1"}


@pytest.mark.parametrize("smiles", RERUN_ONLY)
def test_a_rerun_name_is_labelled_below_the_pin(smiles):
    row = _row(smiles, "pin")
    assert not is_failure_name(row["name"]), row
    expected = ("systematic_verified" if smiles in _RERUN_KNOWN_NON_PIN
                else "pin_unverified")
    assert row["tier"] == expected and not row["is_pin"], row
    assert row["opsin"] == "verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "COc1c(C)cnc(CS(=O)c2nc3ccc(O)cc3[nH]2)c1C",
    "N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O",
    "O=C1CC(C2CCC(=O)O2)Oc2ccccc21",
])
def test_a_best_effort_twin_is_the_strict_path_without_the_rerun(smiles):
    row = _row(smiles, "best-effort")
    assert not is_failure_name(row["name"]), row
    assert row["tier"] != "pin_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_a_a_first_run_name_keeps_pin_verified():
    # caffeine: the strict first run names it; nothing changes
    smiles = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", row
        assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------
# B. The PIN tier's result no longer depends on ORTHONYM_MEMO. A rejected
# promotion rolled its provenance back, including the name-scoped non-PIN
# record '(R)-(cyclohex-3-en-1-yl)meth', while the scope memo kept the strings
# its computation made; a later memo hit shipped the recorded spelling as if
# nothing had been recorded (memo on: a name; memo off / verify: abstain).
# `restore_provenance` now keeps the non-PIN records made since the snapshot,
# and the ring-yl builds the descriptor itself: '[(1R)-cyclohex-3-en-1-
# yl]methyl' (the Blue Book "preceded by a numerical or letter locant...
# when such locants are present"; the free valence of a carbocyclic monocycle
# is locant 1, (c):3256, 'cyclohex-3-en-1-yl (preferred prefix)':3268).
# ---------------------------------------------------------------------------

MEMO_ROWS = [
    # minimal analogue of the milestone1500 row below
    ("C[C@@H](CN(C[C@@H]1CCC=CC1)C)O",
     "(2S)-1-({[(1R)-cyclohex-3-en-1-yl]methyl}(methyl)amino)propan-2-ol"),
    # milestone1500
    ("C[C@H](CN(C[C@@H]1CCC=CC1)C[C@@H](C)O)O",
     "(2R)-1-({[(1R)-cyclohex-3-en-1-yl]methyl}[(2R)-2-hydroxypropyl]amino)"
     "propan-2-ol"),
]


@pytest.mark.parametrize("smiles,expected", MEMO_ROWS)
def test_b_pin_tier_result_is_the_same_with_the_memo_on_and_off(monkeypatch, smiles,
                                                                  expected):
    from orthonym.assembly import memo
    seen = {}
    for mode in ("on", "off", "verify"):
        monkeypatch.setattr(memo, "_MODE", mode)
        memo.clear_process_cache()
        row = _row(smiles, "pin")
        seen[mode] = (row["name"], row["tier"], row["is_pin"])
    assert len(set(seen.values())) == 1, seen
    name, tier, is_pin = seen["on"]
    assert name == expected, seen
    # the promotion re-run built it: labelled below the PIN (section A)
    assert tier == "pin_unverified" and not is_pin, seen
    assert_full_rt(name, smiles)


def test_b_restore_provenance_keeps_the_non_pin_records_made_since_the_snapshot():
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    try:
        pv.record_non_pin_fragment("kept-before")
        snap = pv.get_provenance()
        pv.record_general_ring_prefix()
        pv.record_non_pin_fragment("(R)-(cyclohex-3-en-1-yl)meth")
        pv.restore_provenance(snap)
        prov = pv.get_provenance()
        assert prov["general_ring_prefix"] is False
        assert prov["non_pin_fragments"] == ("kept-before",
                                             "(R)-(cyclohex-3-en-1-yl)meth")
    finally:
        pv.clear_provenance()


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)N[C@@H]1CCC=CC1", "N-[(1R)-cyclohex-3-en-1-yl]acetamide"),
])
def test_b_ring_yl_free_valence_stereocentre_cites_locant_1(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_b_a_memo_hit_after_a_rejected_promotion_still_meets_its_record():
    # the class mechanism, without a molecule: a rejected promotion computed a memo
    # entry whose computation recorded a non-PIN spelling; a later hit on the entry
    # must still be read as carrying it (with the old roll-back it was not)
    from orthonym.assembly import memo
    from orthonym.metrics import provenance as pv
    from orthonym.rules.pin_vocabulary import promote_at_pin_tier

    def _compute():
        pv.record_non_pin_fragment("(R)-(cyclohex-3-en-1-yl)meth")
        return "[(R)-(cyclohex-3-en-1-yl)methyl](methyl)amino"

    scope = memo.push_scope()
    promoted = memo.pin_promotion_var.set(True)
    pv.clear_provenance()
    try:
        assert promote_at_pin_tier(
            lambda: memo.cache_or_compute("t_ns", "k", _compute)) is None
        hit = memo.cache_or_compute("t_ns", "k", lambda: "never")
        if memo._MODE == "on":
            assert hit == "[(R)-(cyclohex-3-en-1-yl)methyl](methyl)amino"
        assert pv.name_carries_non_pin_part(pv.get_provenance(), hit)
    finally:
        memo.pin_promotion_var.reset(promoted)
        memo.pop_scope(scope)
        pv.clear_provenance()


# ---------------------------------------------------------------------------
# C. The PIN tier's promotion re-run: its cost and its budget trip.
# (1) The re-run runs only when it can change the result: the first run
# reached a site whose behaviour the re-run changes and, when the only such
# sites were promotion calls, a preview of their promoted producers keeps a
# name (``rules.pin_vocabulary.promotion_could_change``). Otherwise the
# re-run would repeat the first run (same path, same failure).
# (2) The pure namespaces are not keyed apart in the re-run.
# (3) A budget trip in the re-run is "no second name": raise_on_limit=True still
# raises OrthonymLimitError (it returned the failure label instead).
# ---------------------------------------------------------------------------

def _count_reruns(monkeypatch):
    import orthonym.namer as nm
    from orthonym.assembly import memo
    calls = []
    orig = nm.Orthonym.name

    def spy(self, smiles, *a, **kw):
        if memo.pin_promotion_var.get():
            calls.append(smiles)
        return orig(self, smiles, *a, **kw)
    monkeypatch.setattr(nm.Orthonym, "name", spy)
    return calls


@pytest.mark.parametrize("smiles", [
    "CC(C)(C)C1=CC=C(C=C1)[U]",   # UNSUPPORTED_ELEMENT: no producer runs
    "CC(C)CC*",                    # WILDCARD_ATOMS
])
def test_c_a_structural_decline_is_not_named_twice(monkeypatch, smiles):
    reruns = _count_reruns(monkeypatch)
    row = _row(smiles, "pin")
    assert is_failure_name(row["name"]), row
    assert reruns == [], reruns


def test_c_a_rerun_that_would_repeat_the_first_run_is_skipped(monkeypatch):
    # dev2000-derived witness: the first run reaches promotion calls, and none of
    # their promoted producers keeps a name, so the re-run would take the first
    # run's path to the same abstention
    smiles = "CC(=O)NC(C)Cc1ccccc1"
    reruns = _count_reruns(monkeypatch)
    assert is_failure_name(_row(smiles, "pin")["name"])
    assert reruns == [], reruns
    be = _row(smiles, "best-effort")
    assert not is_failure_name(be["name"]), be
    assert_full_rt(be["name"], smiles)


def test_c_a_rerun_that_can_win_still_runs(monkeypatch):
    smiles = "NCc1csc(-c2cccs2)n1"
    reruns = _count_reruns(monkeypatch)
    row = _row(smiles, "pin")
    assert row["name"] == "[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine", row
    assert len(reruns) >= 1
    assert_full_rt(row["name"], smiles)


def test_c_pure_namespaces_are_shared_with_the_rerun():
    from orthonym.assembly import memo
    tok = memo.pin_promotion_var.set(True)
    try:
        for ns in ("fg_detect", "sugar_c_substituted", "opsin_extended_smiles",
                   "polycyclic.main_ring_struct"):
            assert memo._ck(ns, "k") == (ns, "k")
        assert memo._ck("name_substituent", "k") == ("name_substituent", "k",
                                                     "pin-promotion")
    finally:
        memo.pin_promotion_var.reset(tok)


def test_c_a_budget_trip_in_the_rerun_keeps_raise_on_limit(monkeypatch):
    import orthonym.namer as nm
    import orthonym.rules.pin_vocabulary as pv
    from orthonym.assembly import memo
    from orthonym.assembly.fragment_naming import PerfBudgetExceeded
    from orthonym.errors import OrthonymLimitError
    orig = nm.Orthonym.name

    def tripping(self, smiles, *a, **kw):
        if memo.pin_promotion_var.get():
            raise PerfBudgetExceeded("simulated trip in the re-run")
        return orig(self, smiles, *a, **kw)
    monkeypatch.setattr(nm.Orthonym, "name", tripping)
    monkeypatch.setattr(pv, "promotion_could_change", lambda probe: True)
    smiles = "CC(=O)NC(C)Cc1ccccc1"
    with pytest.raises(OrthonymLimitError):
        Orthonym().name(smiles, raise_on_limit=True)
    assert is_failure_name(Orthonym().name(smiles))


def test_c_a_real_budget_trip_still_raises(monkeypatch):
    # dev2000 cardiac glycoside: with an analysis budget of 15 calls the first run
    # (10 calls) finishes and a re-run can exhaust the rest
    from orthonym.assembly import fragment_naming as fn
    from orthonym.errors import OrthonymLimitError
    monkeypatch.setattr(fn, "_ANALYSIS_CALL_BUDGET", 15)
    smiles = ("CC1OC(OC2CCC3(CO)C4CCC5(C)C(C6=CC(=O)OC6)CCC5(O)C4CCC3(O)C2)C(O)"
              "C(O)C1OC1OC(CO)C(O)C(O)C1O")
    with pytest.raises(OrthonymLimitError):
        Orthonym().name(smiles, raise_on_limit=True)


# ---------------------------------------------------------------------------
# D. An ester of ONE polyacid is 'dimethyl...dioate',
# the Blue Book "Fully esterified acids derived from a single acid are
# systematically named by placing the name(s) of the hydroxylic component...
# in front of the name of the acid component";:31775 'dimethyl butanedioate
# (PIN)';:18875). A name that makes one ester the functional-class
# ester and cites the other as '(methoxycarbonyl)' is labelled below the PIN
# (structural: the two acid carbons share one parent); the polyester is built
# from its acid component where the strict path names that acid.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # milestone1500 and a minimal analogue: the acid component (a 2-ylidene
    # butanedioic acid) has no strict-path name, so the mono-ester name stays
    ("C/C(=C(/CC(=O)OC)\\C(=O)OC)/NC1CCCCC1",
     "methyl (3E)-4-(cyclohexylamino)-3-(methoxycarbonyl)pent-3-enoate"),
    ("COC(=O)CC(=C(C)N)C(=O)OC", "methyl 4-amino-3-(methoxycarbonyl)pent-3-enoate"),
])
def test_d_a_mono_ester_name_of_a_polyacid_ester_is_not_the_pin(smiles, expected):
    for tier in ("pin", "valid", "best-effort"):
        row = _row(smiles, tier)
        assert row["tier"] != "pin_verified" and not row["is_pin"], (tier, row)
        assert_full_rt(row["name"], smiles)
    assert _row(smiles, "pin")["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    ("COC(=O)CC(C)C(=O)OC", "dimethyl methylbutanedioate"),   #:2939
    ("CCOC(=O)C(C)C(C)C(=O)OCC", "diethyl 2,3-dimethylbutanedioate"),
    ("COC(=O)/C=C/C(=O)OC", "dimethyl (2E)-but-2-enedioate"),
    ("CCOC(=O)CC(Cc1ccccc1)C(=O)OCC", "diethyl 2-benzylbutanedioate"),
])
def test_d_the_polyester_of_one_acid_is_built_from_its_acid(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_d_different_organyl_groups_are_cited_alphanumerically():
    # dev2000; "different organyl groups are cited in alphanumerical
    # order (see " -- 'butyl' before '2-ethylhexyl' (e before b only in the
    # plain string order that counted the locant)
    smiles = "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCCCC"
    row = _row(smiles, "pin")
    assert row["name"] == "butyl 2-ethylhexyl benzene-1,2-dicarboxylate", row
    assert_full_rt(row["name"], smiles)


def test_d_a_substituted_propanedioate_is_not_built_with_locants():
    # (:3031): C-2 is the only substitutable carbon of propanedioic acid,
    # so its PIN cites no locant; the acid composer would ('2-methyl-2-phenyl...')
    smiles = "CCOC(=O)C(C)(C(=O)OCC)c1ccccc1"
    row = _row(smiles, "pin")
    assert "2-methyl-2-phenylpropanedioate" not in (row["name"] or ""), row


def test_d_the_shared_acid_parent_predicate():
    from rdkit import Chem

    from orthonym.rules.esters import (
        ester_carbonyl_atoms,
        ester_shares_its_acid_with_another_ester,
    )

    def shares(smi):
        mol = Chem.MolFromSmiles(smi)
        c = ester_carbonyl_atoms(mol)
        return ester_shares_its_acid_with_another_ester(mol, c[0])
    # one chain, one ring system, oxalate: one polyacid
    assert shares("COC(=O)CCC(=O)OC")
    assert shares("COC(=O)c1ccccc1C(=O)OC")
    assert shares("COC(=O)C(=O)OC")
    # a ring acid and a side-chain acid, an ether-linked pair: two acid parents
    # ('methyl 4-(2-methoxy-2-oxoethyl)benzoate',:31698)
    assert not shares("COC(=O)Cc1ccc(C(=O)OC)cc1")
    assert not shares("COC(=O)COCC(=O)OC")


# ---------------------------------------------------------------------------
# E. Salts. (1) The acid-salt anion words are two words, as the phosphate and
# carbonate rows already were: (the Blue Book) method (2),
# 'sodium hydrogen carbonate (PIN)' (:31623), with the Note that only inorganic
# nomenclature writes 'hydrogen' directly in front of the anion;
# (:43566) '*N*,*N*-diethylethanaminium hydrogen sulfate (PIN)';:7150
# "hydrogensulfate is an inorganic name for HSO4-". (2) A wholly inorganic
# metal compound (a metal atom, no carbon) takes the label of its naming path,
# as in the paper's measured run (user decision 2026-09-30, which replaces the
# 2026-09-28 label systematic_verified: 'sodium chloride' pin_verified). Label
# only; names unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected,tier", [
    ("CC[NH+](CC)CC.OS(=O)(=O)[O-]", "N,N-diethylethanaminium hydrogen sulfate",
     "pin_verified"),
    ("c1cc[nH+]cc1.OS(=O)(=O)[O-]", "pyridin-1-ium hydrogen sulfate", "pin_verified"),
    # carbon-free salts: the label of the naming path, as in the paper's measured
    # run (user decision 2026-09-30, 'sodium chloride' pin_verified)
    ("[K+].OS(=O)(=O)[O-]", "potassium hydrogen sulfate", "pin_verified"),
    ("[Ba+2].OS(=O)(=O)[O-].OS(=O)(=O)[O-]", "barium bis(hydrogen sulfate)",
     "pin_verified"),
    ("[Na+].OS(=O)[O-]", "sodium hydrogen sulfite", "pin_verified"),
])
def test_e_hydrogen_sulfate_is_two_words(smiles, expected, tier):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == tier and row["is_pin"] == (tier == "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,expected", [
    # A cation that states its charge fixes the ratio, so the binary name takes no
    # stoichiometric prefixes ("binary names formed by citing the name of the cation
    # followed by that of the anion", SALTS DERIVED FROM ALCOHOLS...,
    # the Blue Book); not a PIN,:4667).
    ("[Fe+3].[Cl-].[Cl-].[Cl-]", "iron(III) chloride"),
    ("[Hg+2].[Cl-].[Cl-]", "mercury(II) chloride"),
    ("O.O.[Cu+2].[O-]S(=O)(=O)[O-]", "copper(II) sulfate dihydrate"),
    ("[Na+].[Cl-]", "sodium chloride"),
])
def test_e_a_wholly_inorganic_metal_compound_keeps_the_label_of_its_naming_path(
        smiles, expected):
    # Paper conformance (user decision 2026-09-30): a carbon-free compound takes
    # the label of the paper's measured run, 'sodium chloride' pin_verified (the
    # strict PIN path built and verified it). Names unchanged.
    for tier in ("pin", "valid", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == expected, (tier, row)
        assert row["tier"] == "pin_verified" and row["is_pin"], (tier, row)
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("[Na+].OC([O-])=O", "sodium hydrogen carbonate"),   # (PIN):31623
    ("CC(=O)[O-].[Na+]", "sodium acetate"),              #:4712
    # 'tetrachlorosilane' moved to tests/integration/test_texts_labels_spelling.py:
    # a carbon-free compound has a preselected name at most,
    # the Blue Book; 'tetrachlorosilane (preselected name)',:35758),
    # so it is systematic_verified (texts, labels and spelling, 2026-09-29).
])
def test_e_salts_with_carbon_keep_their_label(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(expected, smiles)


# ---------------------------------------------------------------------------
# F. An O-attached ester of a sulfur or phosphorus oxoacid ('sulfooxy',
# 'phosphonooxy', a '(hydroxy)phosphoryl...oxy' link) cited as a prefix beside a
# suffix junior to acids and esters is not the PIN form:
# (the Blue Book) and (:36327) allow the prefix only when
# "another substituent having priority... for citation as principal group" is
# present ('3-(sulfooxy)propanoic acid (PIN)',:36488); otherwise
# (:35916-35918, 'methyl hydrogen sulfate (PIN)':35968). Label only.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "O=S(=O)(O)Oc1ccc2cccc(O)c2n1",                       # dev2000, first run
    "O=C(CCc1cccc(OS(=O)(=O)O)c1)c1ccc(O)c(O)c1",          # dev2000, first run
    "O=S(=O)(O)OC/C=C/c1ccc(O)cc1",                        # dev2000, re-run
    "O=C[C@H](O)[C@@H](O)[C@H](O)COP(=O)(O)O",             # dev2000, phosphonooxy
])
def test_f_an_oxoacid_ester_prefix_beside_a_junior_suffix_is_not_the_pin(smiles):
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert not is_failure_name(row["name"]), (tier, row)
        assert row["tier"] != "pin_verified" and not row["is_pin"], (tier, row)
        assert_full_rt(row["name"], smiles)


def test_f_the_guard_form():
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary as f
    assert f("2-(sulfooxy)quinolin-8-ol") == "sulfooxy"
    assert f("3-(phosphonooxy)propane-1,2-diol") == "phosphonooxy"
    # a senior suffix present (BB PINs), or a functional-class ester name
    for pin in ("3-(sulfooxy)propanoic acid", "(phosphonooxy)acetic acid",
                "4,4'-(hydroxyphosphoryl)dibenzoic acid",
                "methyl 3-(sulfooxy)propanoate"):
        assert f(pin) is None, pin
    # the promotion re-run's acceptance does not read this label-only form
    assert f("2-(sulfooxy)quinolin-8-ol", label_forms=False) is None


# ---------------------------------------------------------------------------
# G. A ring whose every double bond is exocyclic is saturated, although RDKit's
# aromaticity model flags the 3,6-bis(ylidene) diketopiperazine: its ring ketone
# takes the saturated parent and the retained saturated name ('piperazine-2,5-
# dione', as the mono-ylidene already had;:16928, ':28267
# piperidin-2-one (PIN)', ':29344 1,3-diazinane-2,4,6-trione (PIN)
# pyrimidine-2,4,6(1H,3H,5H)-trione'), and (g) (:3307) numbers it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # dev2000 (aromatic and Kekule spellings of one molecule)
    ("COc1ccc(/C=c2\\[nH]c(=O)/c(=C/c3ccccc3)[nH]c2=O)cc1",
     "(3Z,6Z)-3-benzylidene-6-[(4-methoxyphenyl)methylidene]piperazine-2,5-dione"),
    ("COc1ccc(/C=C2\\NC(=O)/C(=C/c3ccccc3)NC2=O)cc1",
     "(3Z,6Z)-3-benzylidene-6-[(4-methoxyphenyl)methylidene]piperazine-2,5-dione"),
    ("O=C1N/C(=C\\c2ccccc2)C(=O)N/C1=C\\c1ccccc1",
     "(3Z,6Z)-3,6-dibenzylidenepiperazine-2,5-dione"),
    ("O=C1N/C(=C/C)C(=O)N/C1=C\\c1ccccc1",
     "(3Z,6E)-3-benzylidene-6-ethylidenepiperazine-2,5-dione"),
    # the mono-ylidene control (unchanged)
    ("O=C1N/C(=C\\c2ccccc2)C(=O)NC1", "(3Z)-3-benzylidenepiperazine-2,5-dione"),
])
def test_g_a_bis_ylidene_diketopiperazine_is_a_piperazinedione(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == expected, (tier, row)
    assert_full_rt(expected, smiles)


# ---------------------------------------------------------------------------
# H. Indicated hydrogen on a six-membered azine parent ('1H-pyrimidin-3-yl') is no
# PIN form: mancude pyridine / pyrimidine / pyrazine / pyridazine need none
#, the Blue Book); a free valence there takes added hydrogen,
# (:24695) 'pyridin-1(2H)-yl (preferred prefix)'. Label only.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O",   # dev2000
    "OC(=O)Cn1ccc(=O)[nH]c1=O",             # minimal analogue
])
def test_h_uracil_1_yl_with_indicated_hydrogen_is_not_labelled_a_pin(smiles):
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert not is_failure_name(row["name"]), (tier, row)
        assert row["tier"] != "pin_verified" and not row["is_pin"], (tier, row)
        assert_full_rt(row["name"], smiles)


def test_h_the_guard_form():
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary as f
    assert f("(2,6-dioxo-1H-pyrimidin-3-yl)acetic acid") == "1H-pyrimidin"
    for pin in ("1-(2-hydroxyethyl)pyrimidine-2,4(1H,3H)-dione", "pyridin-2(1H)-one",
                "6H-pyrazino[2,3-b]carbazole", "1H-pyrrolo[2,3-b]pyridine",
                "4H-pyrido[1,2-a]pyrimidin-4-one", "2H-pyran"):
        assert f(pin) is None, pin
    assert f("(2,6-dioxo-1H-pyrimidin-3-yl)acetic acid", label_forms=False) is None


def test_h_the_uracil_control_is_unchanged():
    smiles = "OCCn1ccc(=O)[nH]c1=O"
    row = _row(smiles, "pin")
    assert row["name"] == "1-(2-hydroxyethyl)pyrimidine-2,4(1H,3H)-dione", row
    assert row["tier"] == "pin_verified", row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------
# I. A ketone on a hydro fused core takes the core's indicated hydrogen at the
# ketone position, also when the core's stem opens with its own locant
# ('1-benzopyran'): (the Blue Book) "the indicated hydrogen
# atoms are placed at peripheral atoms that will accommodate these principal
# characteristic groups... Locants for hydro prefixes are those of the
# saturated positions" -- '2,3-dihydro-4H-1-benzopyran-4-one', never
# '3,4-dihydro-2H-1-benzopyran-4-one' 's '3,4-dihydro-2H-1-benzopyran
# (PIN)',:24641, is the parent hydride's spelling).
# ---------------------------------------------------------------------------

def test_i_a_flavanone_on_the_fused_core_carries_4h_at_the_ketone():
    smiles = "C/C(=C\\Cc1c(O)cc(O)c2c1O[C@H](c1ccc(O)c(O)c1)CC2=O)CO"   # dev2000
    row = _row(smiles, "pin")
    # (the prenyl-type prefix is the section-N chain: '4-hydroxy-3-methyl')
    assert row["name"] == (
        "(2S)-2-(3,4-dihydroxyphenyl)-5,7-dihydroxy-8-[(2E)-4-hydroxy-3-methyl"
        "but-2-en-1-yl]-2,3-dihydro-4H-1-benzopyran-4-one"), row
    assert_full_rt(row["name"], smiles)
    be = _row(smiles, "best-effort")
    assert be["name"].endswith("-2,3-dihydro-4H-1-benzopyran-4-one"), be
    assert_full_rt(be["name"], smiles)


def test_i_the_plain_chromanone_is_unchanged():
    row = _row("O=C1CCOc2ccccc21", "pin")
    assert row["name"] == "2,3-dihydro-4H-1-benzopyran-4-one", row
    assert row["tier"] == "pin_verified", row


# ---------------------------------------------------------------------------
# J. A lactone ring is not the parent beside a senior ring ketone:
# (the Blue Book) "There is no seniority order difference between ketones
# and pseudoketones. When necessary, the maximum number of carbonyl groups...
# and between rings and ring systems, are considered"; (d) (:19414)
# more rings. The ketone-parent name is built where the partial-saturation
# producer reads it back ('2-(5-oxooxolan-2-yl)-2,3-dihydro-4H-1-benzopyran-4-
# one'); otherwise the oxolan-2-one name is labelled below the PIN. ':29650
# 4-(4-oxocyclohexyl)oxolan-2-one (PIN)' stays the PIN.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected,tier", [
    # minimal analogue: built inside the PIN tier's promotion re-run (section A)
    ("O=C1CC(C2CCC(=O)O2)Oc2ccccc21",
     "2-(5-oxooxolan-2-yl)-2,3-dihydro-4H-1-benzopyran-4-one", "pin_unverified"),
    # dev2000: the strict first run builds it
    ("C[C@H]1CC(=O)O[C@H]1[C@@]1(C)CC(=O)c2c(O)cccc2O1",
     "(2R)-5-hydroxy-2-methyl-2-[(2R,3S)-3-methyl-5-oxooxolan-2-yl]-2,3-dihydro-4H-"
     "1-benzopyran-4-one", "pin_verified"),
])
def test_j_the_senior_ring_ketone_is_the_parent(smiles, expected, tier):
    # the lactone producer defers to the parent when that name is built and
    # read back; the lactone ring becomes the '5-oxooxolan-2-yl' prefix
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == tier and row["is_pin"] == (tier == "pin_verified"), row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles", [
    "O=C1CC(C2CCC(=O)O2)Oc2ccccc21",                          # minimal analogue
    "C[C@H]1CC(=O)O[C@H]1[C@@]1(C)CC(=O)c2c(O)cccc2O1",       # dev2000
    "COc1cccc2c1C(=O)/C(=C(\\CO)[C@@H]1OC(=O)C[C@@H]1C)O2",   # dev2000
])
def test_j_the_senior_ring_ketone_system_is_found(smiles):
    from rdkit import Chem

    from orthonym.rules.lactones import (
        another_ring_ketone_system_is_senior,
        is_monocyclic_lactone,
    )
    mol = Chem.MolFromSmiles(smiles)
    assert another_ring_ketone_system_is_senior(mol, is_monocyclic_lactone(mol)["ring_atoms"])


def test_j_a_lactone_name_that_stays_is_labelled_below_the_pin():
    # the benzofuranone-ylidene ring ketone parent is not built: the oxolan-2-one
    # name stays, labelled below the PIN at both tiers
    smiles = "COc1cccc2c1C(=O)/C(=C(\\CO)[C@@H]1OC(=O)C[C@@H]1C)O2"
    for tier in ("pin", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"].endswith("oxolan-2-one"), (tier, row)
        assert row["tier"] != "pin_verified" and not row["is_pin"], (tier, row)
        assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("O=C1CC(C2CCC(=O)CC2)CO1", "4-(4-oxocyclohexyl)oxolan-2-one"),   # BB:29648
    ("O=C1CCC(C2CCCC2=O)O1", "5-(2-oxocyclopentyl)oxolan-2-one"),
])
def test_j_a_lactone_senior_to_a_carbocyclic_ketone_keeps_the_pin(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(expected, smiles)


def test_j_a_label_only_record_does_not_drop_a_rerun_name():
    # the label-only channel lowers the label but is not read by the re-run's
    # keep-or-drop decision
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    try:
        pv.record_non_pin_label("oxolan-2-one")
        prov = pv.get_provenance()
        assert pv.name_carries_non_pin_part(prov, "5-x-oxolan-2-one")
        assert not pv.name_carries_non_pin_part(prov, "5-x-oxolan-2-one",
                                                label_forms=False)
    finally:
        pv.clear_provenance()


# ---------------------------------------------------------------------------
# K. The ring carboxamide is assembled from its structural parts
# (``rules.amides.ring_amide_parts``: ring hydride, suffix, N-substituents),
# not by cutting name_amide's emitted string apart and splicing '-1-' into it.
# Names unchanged: (the Blue Book) one alphanumerical series of N-
# and ring prefixes; (:2869) the suffix locant with a ring prefix.
# ---------------------------------------------------------------------------

def test_k_ring_amide_parts():
    from rdkit import Chem

    from orthonym.rules.amides import ring_amide_parts
    pat = Chem.MolFromSmarts("[CX3](=O)[NX3]")

    def parts(smi, form="amide"):
        mol = Chem.MolFromSmiles(smi)
        return ring_amide_parts(mol, mol.GetSubstructMatches(pat)[0], suffix_form=form)
    p = parts("O=C(NC1CCCC1)C1CCC(F)(F)CC1")
    assert (p.ring_hydride, p.suffix, p.retained) == ("cyclohexane", "carboxamide", None)
    assert [s["name"] for s in p.n_substituents] == ["cyclopentyl"]
    assert p.parent_word() == "cyclohexanecarboxamide"
    assert p.parent_word(suffix_locant=True) == "cyclohexane-1-carboxamide"
    b = parts("CNC(=O)c1ccc(Cl)cc1")
    assert b.parent_word(suffix_locant=True) == "benzamide"
    assert parts("NC(=O)c1ccncc1") is None           # a heteroring: not spellable here
    assert not parts("NC(=O)C1CCCCC1").n_substituents


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(NC1CCCC1)C1CCC(F)(F)CC1", "N-cyclopentyl-4,4-difluorocyclohexane-1-carboxamide"),
    ("O=C(Nc1ccccc1)C1CCC(O)CC1", "4-hydroxy-N-phenylcyclohexane-1-carboxamide"),
    ("NC(=O)C1CCC(C)CC1", "4-methylcyclohexane-1-carboxamide"),
    ("CNC(=O)c1ccc(Cl)cc1", "4-chloro-N-methylbenzamide"),
    # the N- and the ring methyl are one multiplied prefix and the suffix keeps its
    # locant (b), the Blue Book;,:2869; ':32879
    # N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)'), now built by the strict path
    ("CNC(=O)C1CCC(C)CC1", "N,4-dimethylcyclohexane-1-carboxamide"),
])
def test_k_ring_amide_names_unchanged(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


# ---------------------------------------------------------------------------
# L. A ring-assembly prefix is numbered per whatever the input atom
# order (the Blue Book "Low locants are assigned to ring junctions, then
# to free valences"; '[1,1'-biphenyl]-4-yl (preferred prefix)':16118): the
# free valence sits on the unprimed ring. One dev2000 molecule came out '4-yl'
# or "4'-yl" by its SMILES spelling, both certified by the PIN tier's re-run.
# ---------------------------------------------------------------------------

BIPHENYL_SPELLINGS = [
    "O=C(O)CC/C=C\\CC[C@H]1[C@@H](OCc2ccc(-c3ccccc3)cc2)CC(=O)[C@@H]1N1CCOCC1",
    "c1cccc(-c2ccc(cc2)CO[C@H]2CC([C@@H]([C@H]2CC/C=C\\CCC(=O)O)N2CCOCC2)=O)c1",
]


def test_l_one_molecule_two_spellings_one_ring_assembly_prefix():
    names = set()
    for smi in BIPHENYL_SPELLINGS:
        row = _row(smi, "pin")
        assert "[1,1'-biphenyl]-4-yl" in row["name"], row
        assert "4'-yl" not in row["name"], row
        assert_full_rt(row["name"], smi)
        names.add(row["name"])
    assert len(names) == 1, names


@pytest.mark.parametrize("smiles,expected", [
    ("OCc1ccc(-c2ccccc2)cc1", "([1,1'-biphenyl]-4-yl)methanol"),
    ("c1ccccc1-c1ccc(CO)cc1", "([1,1'-biphenyl]-4-yl)methanol"),
    ("OCc1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "([1,1':4',1''-terphenyl]-4-yl)methanol"),
    ("c1ccc(-c2ccc(-c3ccc(CO)cc3)cc2)cc1", "([1,1':4',1''-terphenyl]-4-yl)methanol"),
])
def test_l_ring_assembly_prefix_spellings(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


# ---------------------------------------------------------------------------
# M. Best-effort names that depended on the input's atom order. (1) A decorated
# von Baeyer cage core is re-mapped through its own automorphisms at every tier
# (Kekule-preserving on an unsaturated core), so the free valence, then the
# substituents, take the lowest locants the cage allows (c) then (f),
# the Blue Book,:3301). (2) The clean-first offer's fresh-memo run names
# the canonical spelling of the molecule. Also: the bornyl formate's best-effort
# name is now the PIN tier's '1,7,7-trimethylbicyclo[2.2.1]heptan-2-yl'.
# ---------------------------------------------------------------------------

def _spellings(smiles, n=3):
    import random

    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    out = [smiles, Chem.MolToSmiles(mol)]
    rng = random.Random(0)
    for _ in range(n):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        out.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False))
    return out


@pytest.mark.parametrize("smiles", [
    "COc1c(CC(O)C=O)cc2c(c1OC)OCO2",                                   # dev2000
    "O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3ccccc3)cc2)c1",                       # dev2000
    "CO/C=C(C(=O)OC)\\C(C)=C/C=C/c1ccc(OCC=C(C)C)c(O)c1",               # a dev split
])
def test_m_best_effort_name_does_not_depend_on_the_atom_order(smiles):
    names = set()
    for spelling in _spellings(smiles):
        row = _row(spelling, "best-effort")
        assert not is_failure_name(row["name"]), (spelling, row)
        names.add(row["name"])
    assert len(names) == 1, names
    assert_full_rt(names.pop(), smiles)


def test_m_the_ordering_sort_key_memo_is_keyed_by_the_naming_context(monkeypatch):
    # milestone1500. The (g) tie of the 1,4-disubstituted piperazine is
    # decided by the prefixes' names (the one cited first, 'oxacyclopentanyl...'
    # before 'phenylsulfonyl', takes locant 1; (g) the Blue Book,
    #:3477). The sort-key names were memoised without the naming context, so the
    # clean fall-through read the best-effort context's entries: four of five
    # spellings gave '4-{...}-1-(phenylsulfonyl)piperazine', and
    # ORTHONYM_MEMO=verify raised MemoMismatch (the floor name shipped instead).
    from orthonym.assembly import memo
    smiles = "C1C[C@@H](OC1)CNC(=S)N2CCN(CC2)S(=O)(=O)C3=CC=CC=C3"
    expected = ("1-{3-[(2R)-1-oxacyclopentan-2-yl]-1-sulfanylidene-2-azapropyl}-"
                "4-(phenylsulfonyl)piperazine")
    seen = set()
    for mode in ("on", "verify", "off"):
        monkeypatch.setattr(memo, "_MODE", mode)
        for spelling in _spellings(smiles, n=2):
            memo.clear_process_cache()
            seen.add((mode, _row(spelling, "best-effort")["name"]))
    assert {name for _mode, name in seen} == {expected}, seen
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    # milestone1500
    ("CC1=C(C=C(C=C1)S(=O)(=O)N2CCCCC2)NC(=O)COC(=O)C3=CC=C(C=C3)NC(=O)C",
     "1-[(3-{2-[(4-acetamidobenzoyl)oxy]acetamido}-4-methylphenyl)sulfonyl]piperidine"),
    # dev2000
    ("Cc1c(C(=O)Nc2ccc(N3C[C@@H](C)O[C@@H](C)C3)nc2)cccc1-c1ccc(OC(F)(F)F)cc1",
     "(2R,6S)-2,6-dimethyl-4-(5-{2-methyl-3-[4-(trifluoromethoxy)phenyl]benzene-1-"
     "carboxamido}pyridin-2-yl)morpholine"),
])
def test_m_the_ordering_sort_key_memo_is_keyed_by_the_session_depth(monkeypatch, smiles,
                                                                      expected):
    # the same prefix asked at two fragment session depths is named against two depth
    # budgets; without the depth in the key ORTHONYM_MEMO=verify raised MemoMismatch
    # for these rows and the floor name shipped in place of this one
    from orthonym.assembly import memo
    monkeypatch.setattr(memo, "_MODE", "verify")
    memo.clear_process_cache()
    memo.reset_verify_mismatches()
    row = _row(smiles, "best-effort")
    assert [ns for ns, _k in memo._VERIFY_MISMATCHES
            if ns == "substituent_ordering_name"] == [], memo._VERIFY_MISMATCHES
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


def test_m_a_saturated_cage_prefix_is_numbered_as_the_pin_at_best_effort():
    smiles = "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](OC=O)C2"                       # m1500
    row = _row(smiles, "best-effort")
    assert row["name"] == "(1R,2R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-yl formate", row
    assert_full_rt(row["name"], smiles)


# ---------------------------------------------------------------------------
# N. The chain of an acyclic substituent prefix is chosen among the equally long
# ones by the chain criteria, not by the input's atom order: more multiple
# bonds, the Blue Book), then more substituents cited as
# prefixes,:21604), then lower locants,:21698) --
# '4-hydroxy-3-methylbut-2-en-1-yl' (two prefixes), not '3-(hydroxymethyl)but-
# 2-en-1-yl' (one). The same dev2000 molecule got both by its spelling.
# ---------------------------------------------------------------------------

FLAVANONE_SPELLINGS = [
    "C/C(=C\\Cc1c(O)cc(O)c2c1O[C@H](c1ccc(O)c(O)c1)CC2=O)CO",
    "c12c(C/C=C(/CO)C)c(cc(O)c2C(=O)C[C@@H](c2cc(c(cc2)O)O)O1)O",
]


def test_n_one_molecule_two_spellings_one_chain_prefix():
    names = set()
    for smi in FLAVANONE_SPELLINGS:
        row = _row(smi, "pin")
        assert "[(2E)-4-hydroxy-3-methylbut-2-en-1-yl]" in row["name"], row
        assert_full_rt(row["name"], smi)
        names.add(row["name"])
    assert len(names) == 1, names


def test_n_the_chain_key():
    from rdkit import Chem

    from orthonym.assembly.substituent_enumerator import _senior_longest_carbon_path_from
    # -CH2-CH=C(CH3)-CH2OH on a methyl stub: the chain ends at the CH2OH carbon
    mol = Chem.MolFromSmiles("CC/C=C(\\C)CO")
    frag = [1, 2, 3, 4, 5, 6]
    chain = _senior_longest_carbon_path_from(mol, set(frag), 1)
    assert chain == [1, 2, 3, 5], chain


# ---------------------------------------------------------------------------
# O. The added-hydrogen prefix of a six-membered azine ring ketone is built:
# (the Blue Book) 'pyridin-1(2H)-yl (preferred prefix)'; hydro
# prefixes at the saturated positions,:24766 "Locants for hydro
# prefixes are those of the saturated positions"), cited next to the parent;
# no indicated hydrogen on a mancude azine,:24641).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("N[C@@H](Cn1ccc(=O)[nH]c1=O)C(=O)O",                               # dev2000
     "(2S)-2-amino-3-(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)propanoic acid"),
    ("OC(=O)Cn1ccc(=O)[nH]c1=O", "(2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)acetic acid"),
    ("OC(=O)Cn1cc(C)c(=O)[nH]c1=O",
     "(5-methyl-2,4-dioxo-3,4-dihydropyrimidin-1(2H)-yl)acetic acid"),
    ("OC(=O)Cc1c[nH]c(=O)[nH]c1=O",
     "(2,4-dioxo-1,2,3,4-tetrahydropyrimidin-5-yl)acetic acid"),
    ("OC(=O)Cn1ccccc1=O", "(2-oxopyridin-1(2H)-yl)acetic acid"),
    ("OC(=O)Cn1ccc(N)nc1=O", "(4-amino-2-oxopyrimidin-1(2H)-yl)acetic acid"),
    ("OC(=O)Cc1ccc(=O)[nH]c1", "(6-oxo-1,6-dihydropyridin-3-yl)acetic acid"),
])
def test_o_azine_ring_ketone_prefixes_take_added_hydrogen(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)Cc1cccc(Cl)n1", "(6-chloropyridin-2-yl)acetic acid"),   # mancude: unchanged
    ("OC(=O)Cc1ccncc1", "(pyridin-4-yl)acetic acid"),
])
def test_o_a_mancude_azine_prefix_is_unchanged(smiles, expected):
    row = _row(smiles, "pin")
    assert row["name"] == expected, row
