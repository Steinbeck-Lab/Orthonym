"""ChEBI best-effort losses (TRIAGE.md 'ChEBI losses').

The paper's ChEBI recipe on main 171c54d5e lost 33 rows the paper named
round-trip exact. Each class below is pinned with non-a holdout split witnesses (the
ChEBI rows themselves plus small analogues; none is in
eval/splits/a holdout split.json). Every shipped name is checked by an independent
OPSIN full-InChIKey round trip plus the radical-identity and protonation-site
checks (tests/support/rt_assert.name_is_rt_exact).

Classes (causing commit -> fix):
  A multiplicative acid anion partial salts: the parent-suffix
     site count reads a multiplicative '...tetraacetate' as four sites.
  B a parent stereo block beside a substituent's own block.
  C a salt cation whose ordinary name puts the '-ium' on the wrong nitrogen
     (bfbf949fe refuses it): the per-component check now sees the protonation
     site, so the best-effort cation name with the charge on the right N ships.
     The old names are wrong molecules and must never come back.
  D the substitutive peptide-acid retry ran for every acyl branch
     and without a memo: the calcein-type potassium salts spent the analysis
     budget, and the giant peptides hung (the NATIVE_HANG rows).
  E a chain '-ylidene' double bond gets its E/Z with the chain locant
      (1)(a)); the oxa/aza floor name that carried it was
     replaced by candidates without it.
  F a diphosphoric-acid monoester is '<R> trihydrogen diphosphate'
     ; 16f45443a rightly removed the '<R> diphosphoric acid' glue.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.decomposition.fragment_assembly import _assemble_phosphodiester
from orthonym.rules.ions import _parent_acid_suffix_multiplicity
from tests.support.rt_assert import assert_full_rt, name_best_effort, name_is_rt_exact


def _be(smiles):
    res = name_best_effort(smiles)
    assert res.get("source") != "abstain", f"abstained on {smiles}"
    return res


# --------------------------------------------------------------------------
# A multiplicative acid anions,
# --------------------------------------------------------------------------

MULTIPLICATIVE_SALTS = [
    # ChEBI row 15563
    ("O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)O)CC(=O)[O-].[Na+].[Na+].[Na+]",
     "trisodium hydrogen 2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetate"),
    ("O=C([O-])CN(CCN(CC(=O)O)CC(=O)O)CC(=O)[O-].[Na+].[Na+]",
     "disodium dihydrogen 2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetate"),
    ("O=C([O-])COCC(=O)O.[Na+]", "sodium hydrogen 2,2'-oxydiacetate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", MULTIPLICATIVE_SALTS,
                         ids=[e.split()[0] + "-" + e.split()[1] for _, e in MULTIPLICATIVE_SALTS])
def test_multiplicative_acid_partial_salt(smiles, expected):
    res = _be(smiles)
    assert res["name"] == expected
    assert_full_rt(res["name"], smiles)


@pytest.mark.parametrize("name,sites", [
    ("2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetate", 4),
    ("2,2'-oxydiacetate", 2),
    ("4,4'-methylenedibenzoate", 2),
    ("3,3'-oxydipropanoate", 2),
    ("acetate", 1),
    ("propanedioate", 2),
    ("benzene-1,2-dicarboxylate", 2),
    # a primed locant set that is not one unit per prime level is not read
    ("2,2'-bipyridine-4-carboxylate", 1),
])
def test_parent_acid_suffix_multiplicity(name, sites):
    assert _parent_acid_suffix_multiplicity(name) == sites


# --------------------------------------------------------------------------
# B the parent's stereo block beside a prefix's own block
# --------------------------------------------------------------------------

MACROLIDE_92850 = (
    "COC(CCC(C)/C=C/C=C/C1CC(O)CC(O)CCCCCCCC(O)CC(O)C/C=C/C=C\\C=C\\C(O)CC(=O)CC(O)"
    "C/C=C\\C=C/C(O)CC(O)CC(O)CCC(C)C(O)CC(O)C(C)C(O)/C(C)=C\\C=C/C(C)C(O)C(C)C2OC(O)"
    "(CC(=O)O1)CC(O)C2C)C1=C(SC)C(=O)C=C(O)C1=O")


@pytest.mark.opsin_gate
def test_parent_block_beside_substituent_block():
    res = _be(MACROLIDE_92850)
    name = res["name"]
    assert name.startswith("(21E,23Z,25E,33Z,35Z,50Z,52Z)-5-{(1E,3E)-8-"), name
    assert_full_rt(name, MACROLIDE_92850)


# --------------------------------------------------------------------------
# C salt cations: the '-ium' on the nitrogen that carries the charge
# --------------------------------------------------------------------------

# (ChEBI SMILES, the paper's name: '-ium' on the WRONG nitrogen)
WRONG_SITE_SALTS = [
    ("CCOC(=O)Nc1ccc2c(c1)N(C(=O)CC[NH+]1CCOCC1)c1ccccc1S2.[Cl-]",
     "6-[(ethoxycarbonyl)amino]-9-[3-(morpholin-4-yl)-1-oxopropyl]-2-thia-9-azatricyclo"
     "[8.4.0.0^3,8]tetradeca-1(14),3(8),4,6,10,12-hexaen-9-ium chloride"),
    ("CCC[NH+]1CCCC[C@H]1C(=O)Nc1c(C)cccc1C.[Cl-]",
     "(2S)-N-(2,6-dimethylphenyl)-1-propylpiperidine-2-carboxamidium chloride"),
    ("CCC[NH+]1CCCC[C@H]1C(=O)Nc1c(C)cccc1C.O.[Cl-]",
     "(2S)-N-(2,6-dimethylphenyl)-1-propylpiperidine-2-carboxamidium chloride monohydrate"),
    ("CCN1CC(CC[NH+]2CCOCC2)C(c2ccccc2)(c2ccccc2)C1=O.[Cl-]",
     "1-ethyl-4-[2-(morpholin-4-yl)ethyl]-2-oxo-3,3-diphenylpyrrolidinium chloride"),
    ("CCN1CC(CC[NH+]2CCOCC2)C(c2ccccc2)(c2ccccc2)C1=O.O.[Cl-]",
     "1-ethyl-4-[2-(morpholin-4-yl)ethyl]-2-oxo-3,3-diphenylpyrrolidinium chloride monohydrate"),
    ("COC(=O)c1ccccc1-c1c2ccc(=[NH2+])cc-2oc2cc(N)ccc12.[Cl-]",
     "5-azaniumyl-13-imino-9-[2-(methoxycarbonyl)phenyl]-2-oxatricyclo[8.4.0.0^3,8]"
     "tetradeca-1(14),3(8),4,6,9,11-hexaene chloride"),
]


def _key(smi):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smi))


@pytest.mark.parametrize("smiles,old", WRONG_SITE_SALTS,
                         ids=[f"wrong-site-{i}" for i in range(len(WRONG_SITE_SALTS))])
def test_old_wrong_site_name_is_a_different_molecule(smiles, old):
    """The proof: OPSIN reads the old name to the input's full InChIKey (the
    key is protonation-blind), but the protonation-site check refuses it."""
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(old)
    assert parsed and _key(parsed) == _key(smiles)
    assert not name_is_rt_exact(old, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,old", WRONG_SITE_SALTS,
                         ids=[f"right-site-{i}" for i in range(len(WRONG_SITE_SALTS))])
def test_salt_cation_named_on_the_charged_nitrogen(smiles, old):
    res = _be(smiles)
    assert res["name"] != old
    assert_full_rt(res["name"], smiles)


# --------------------------------------------------------------------------
# D the substitutive peptide-acid retry: gated on a peptide name, memoised
# --------------------------------------------------------------------------

CALCEIN_K5 = ("O=C([O-])COc1cc(NC(=O)c2ccc3c(c2)C(=O)OC32c3cc(Cl)c([O-])cc3Oc3cc([O-])"
              "c(Cl)cc32)ccc1N(CC(=O)[O-])CC(=O)[O-].[K+].[K+].[K+].[K+].[K+]")  # row 30299
PEPTIDE_14 = (  # ChEBI row 19596 (NATIVE_HANG at 171c54d5e: 8,176 retries, 243 s)
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)"
    "C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H]"
    "(CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H]"
    "([C@@H](C)CC)C(=O)N[C@@H](CC(C)C)C(N)=O")


def _count_retries(monkeypatch):
    import orthonym.assembly.substituent_naming as sn
    calls = []
    real = sn._substitutive_peptide_acid_name

    def spy(frag_smi):
        calls.append(frag_smi)
        return real(frag_smi)
    monkeypatch.setattr(sn, "_substitutive_peptide_acid_name", spy)
    return calls


@pytest.mark.opsin_gate
def test_potassium_salt_names_without_peptide_retries(monkeypatch):
    calls = _count_retries(monkeypatch)
    res = _be(CALCEIN_K5)
    assert res["name"].startswith("pentapotassium 2',7'-dichloro-3',6'-dioxido-")
    assert_full_rt(res["name"], CALCEIN_K5)
    assert calls == []


@pytest.mark.opsin_gate
def test_giant_peptide_retry_is_bounded(monkeypatch):
    calls = _count_retries(monkeypatch)
    res = _be(PEPTIDE_14)
    assert_full_rt(res["name"], PEPTIDE_14)
    assert len(calls) < 100, len(calls)


@pytest.mark.opsin_gate
def test_tripeptide_keeps_its_substitutive_pin():
    """The case the retry was built for is unchanged."""
    from orthonym import Orthonym
    smi = "O=C(N[C@@H](C)C(=O)N[C@@H](C)C(=O)O)[C@@H]1CCCN1"
    res = Orthonym().name_tiered(smi)
    assert res["name"] == (
        "(2S)-2-{(2S)-2-[(2S)-pyrrolidine-2-carboxamido]propanamido}propanoic acid")
    assert_full_rt(res["name"], smi)


# --------------------------------------------------------------------------
# E a chain '-ylidene' double bond carries its E/Z (1)(a))
# --------------------------------------------------------------------------

YLIDENE_ROWS = [
    # ChEBI rows 36720 and 59088
    "COC1=CC(=O)N(C(=O)/C(C)=C\\[C@H](C)CC/C(=C/Cl)CCCN(C)C(C)=O)[C@H]1CC(C)C",
    "COC1=CC(=O)N(C(=O)/C(C)=C/[C@H](C)CC/C(=C/Cl)CCCN(C)C(C)=O)[C@H]1CC(C)C",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", YLIDENE_ROWS, ids=["36720", "59088"])
def test_chain_ylidene_stereo_rows(smiles):
    res = _be(smiles)
    assert "(4Z," in res["name"], res["name"]
    assert_full_rt(res["name"], smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,block", [
    ("OC(=O)CC/C(=C/Cl)CCC", "(4E)-"),   # was '(1E)-', a misplaced locant
    ("CC/C(=C/Cl)CCCO", "(4Z)-"),
])
def test_chain_ylidene_takes_the_chain_locant(smiles, block):
    res = _be(smiles)
    assert res["name"].startswith(block), res["name"]
    assert_full_rt(res["name"], smiles)


# --------------------------------------------------------------------------
# F diphosphoric-acid monoesters
# --------------------------------------------------------------------------

_C100 = "CC(C)=CC" + "CC(C)=CC" * 19

DIPHOSPHATE_ESTERS = [
    (_C100 + "OP(=O)(O)OP(=O)(O)O",   # ChEBI row 45229
     "3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79-icosamethyloctaconta-"
     "2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78-icosaen-1-yl "
     "trihydrogen diphosphate"),
    ("CC(C)=CCC/C(C)=C/COP(=O)(O)OP(=O)(O)O",
     "(2E)-3,7-dimethylocta-2,6-dien-1-yl trihydrogen diphosphate"),
    ("CCCCCCCCCCCCOP(=O)(O)OP(=O)(O)O", "dodecyl trihydrogen diphosphate"),
    ("CCCCCCCCCCCCOP(=O)(O)OP(=O)(O)OP(=O)(O)O", "dodecyl tetrahydrogen triphosphate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", DIPHOSPHATE_ESTERS,
                         ids=["C100", "geranyl", "dodecyl", "dodecyl-tri"])
def test_polyphosphate_monoester(smiles, expected):
    res = _be(smiles)
    assert res["name"] == expected
    assert_full_rt(res["name"], smiles)


def test_phosphodiester_assembler_polyphosphate_words():
    assert _assemble_phosphodiester(
        {"acid": "diphosphoric acid", "alkyl": "prop-2-en-1-ol"}, "pin"
    ) == "prop-2-en-1-yl trihydrogen diphosphate"
    assert _assemble_phosphodiester(
        {"acid": "diphosphoric acid", "alkyl": "ethanol"}, "pin"
    ) == "ethyl trihydrogen diphosphate"
    # the removed glue never comes back
    assert _assemble_phosphodiester(
        {"acid": "2-(phosphonooxy)ethan-1-amine", "alkyl": "ethanol"}, "pin") is None


# --------------------------------------------------------------------------
# D4 a peptide memo hit replayed the COLD budget cost (TRIAGE 'ChEBI losses
# part 2'): the two peptide replay-memos (869d8abf1, eb25cbf71) charged each
# hit the analysis calls of the first, cold computation, while the unmemoised
# repeat it stands for runs warm (145 analysis calls cold, then 0, 0, 0 for
# the whole-molecule attempt of ChEBI row 3). The 500-call budget ran out,
# PerfBudgetExceeded unwound past the retained-name fallback, and the giant
# peptides abstained. A hit now charges no budget.
# --------------------------------------------------------------------------

GIANT_PEPTIDES = [
    # ChEBI rows (paper: RT-exact), abstained at ca6005b8f
    ("CCC(C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)"
     "[C@H](Cc1ccc(O)cc1)NC(=O)CNC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C)N)C(=O)N"
     "[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccccc1)"
     "C(=O)N[C@@H](Cc1ccccc1)C(=O)O", None),
    ("C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N"
     "[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)"
     "C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H]"
     "(Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "alanylarginylglycyltyrosylserylserylphenylalanylarginyltyrosyltryptophyl"
     "phenylalanylphenylalanine"),
    # hand-made 13-mer and 12-mer (not in any split), abstained at ca6005b8f
    ("C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N"
     "[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCNC(=N)N)"
     "C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H]"
     "(Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O",
     "alanylarginylglycyltyrosylserylserylphenylalanylarginyltyrosyltryptophyl"
     "phenylalanylphenylalanylglycine"),
    ("CC(C)[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)"
     "C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]"
     "(CCCNC(=N)N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)"
     "C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
     "valylarginylglycyltyrosylserylserylphenylalanylarginyltyrosyltryptophyl"
     "phenylalanylphenylalanine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", GIANT_PEPTIDES,
                         ids=["chebi-ile-12mer", "chebi-12mer", "hand-13mer", "hand-12mer"])
def test_giant_peptide_names_inside_the_budget(smiles, expected):
    res = _be(smiles)
    assert_full_rt(res["name"], smiles)
    assert res["tier"] != "pin_verified"  # decision A: a peptide name is not a PIN
    if expected is not None:
        assert res["name"] == expected


def test_nested_memo_hit_without_budget_replay():
    """replay_budgets=False: a hit spends no budget; the default still replays."""
    from orthonym.assembly import fragment_naming as fn
    from orthonym.assembly import memo
    from orthonym.assembly.nested_memo import cached_nested_call

    def work():
        fn.spend_analysis_call(7)
        return "x"

    def spent(ns):
        before = fn._fragment_guard.analysis_budget
        assert cached_nested_call(ns, ("k",), work, **kw[ns]) == "x"
        return before - fn._fragment_guard.analysis_budget

    kw = {"replay_off": {"replay_budgets": False}, "replay_on": {}}
    old_mode = memo._MODE
    memo._MODE = "on"
    token = memo.push_scope()
    fn.enter_name_scope()
    try:
        assert spent("replay_off") == 7   # fresh: the real cost
        assert spent("replay_off") == 0   # hit: no work, no charge
        assert spent("replay_on") == 7
        assert spent("replay_on") == 7    # the default replays the recorded cost
    finally:
        fn.exit_name_scope()
        memo.pop_scope(token)
        memo._MODE = old_mode
