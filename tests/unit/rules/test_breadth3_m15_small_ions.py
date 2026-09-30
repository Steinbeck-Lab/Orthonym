"""Breadth job 3, class M15 (small ions) and the geminal-suffix class under it.

Two root causes, each a class:

1. Geminal characteristic groups (two -OH / -NH2 / -SH, or two aminium centres, on
   ONE carbon) were counted once. The chain suffix builder took a set of the suffix
   locants ('ethanamine' for CC(N)N), the prefix merge took a set of the prefix
   locants ('5,8-dihydroxy' for three OH), the generic 'alcohol' key was left
   out of the alcohol class union ('ethan-1-ol' for OC(O)CO), and the ring numbering
   read the suffix positions as a set. Each built a different molecule, which the
   round trip refused, so the PIN tier abstained. The Blue Book cites one locant per
   suffix: (the Blue Book) "the set that, when compared term by term...
   each cited in order of increasing value"; (c) (:3256) the suffixes first;
   'cyclohexane-1,1-dicarboximidamide (PIN)' (:34236) and 'cyclohexane-1,1-di[(14C)
   carboxylic acid] (PIN)' (:43774) cite the geminal locant twice; the bis(aminium)
   PINs '2-(piperidin-1-ium-3-yl)propane-1,2-bis(aminium)',:42340) and
   'N,N'-bis(trimethylsilyl)silanebis(aminium)',:41411).
2. A bare anion of a mononuclear parent hydride took the salt word ('sulfide'), which
   names no bare ion. (:40900-40902): "An anion derived formally by the
   removal of one or more hydrons from any position of a neutral parent hydride is
   preferably named by using the suffix '-ide'... Numerical prefixes 'di', 'tri',
   etc. are used to denote multiplicity"; (:40945) 'azanide', 'azanediide'.

Every name below is also read back by a FRESH OPSIN call that does not go through the
engine (tests.support.rt_assert._independent_parse) and compared by full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from tests.support.rt_assert import _independent_parse
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
    "CCNC(C)NCC",
    "CCNCNCC",
    "CN(C)CN(C)C",
    "[Fr+].[NH2-]",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


pytestmark = [pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def _pin_row(smiles):
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym().name_tiered(smiles)


def _assert_named(smiles, expected, tiers):
    row = _pin_row(smiles)
    assert row["name"] == expected, row
    assert row["tier"] in tiers, row
    assert _key(_independent_parse(expected)) == _key(smiles)


# --- 1. geminal suffixes on a chain,, -----------------

@pytest.mark.parametrize("smiles,expected", [
    ("NCN", "methanediamine"),
    ("CC(N)N", "ethane-1,1-diamine"),
    ("NC(N)CC", "propane-1,1-diamine"),
    ("CC(C)(N)N", "propane-2,2-diamine"),
    ("OCO", "methanediol"),
    ("OC(O)C", "ethane-1,1-diol"),
    ("OC(O)CCC", "butane-1,1-diol"),
    ("CC(O)(O)C", "propane-2,2-diol"),
    ("SC(S)C", "ethane-1,1-dithiol"),
    ("OC(O)CCC(O)O", "butane-1,1,4,4-tetrol"),
    #: the two OH make methanediol the parent, not benzene
    ("OC(O)c1ccccc1", "phenylmethanediol"),
    # the generic 'alcohol' key joins the class union with a primary OH
    ("OC(O)CO", "ethane-1,1,2-triol"),
    ("OCC(O)(O)C", "propane-1,2,2-triol"),
])
def test_geminal_chain_suffix(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- gem-bis(aminium): (the M15 cation witness's class) --------------------

@pytest.mark.parametrize("smiles,expected", [
    ("[NH3+]C[NH3+]", "methanebis(aminium)"),
    ("CC([NH3+])[NH3+]", "ethane-1,1-bis(aminium)"),
])
def test_geminal_bis_aminium(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- geminal prefixes stay a multiset -------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("O=C(O)CCCC(O)CCC(O)O", "5,8,8-trihydroxyoctanoic acid"),
    ("O=C(O)CCCC(O)CCC(O)(O)O", "5,8,8,8-tetrahydroxyoctanoic acid"),
    ("O=C(O)CC(N)CC(N)N", "3,5,5-triaminopentanoic acid"),
    # the non-a holdout split dev-set row of the class (milestone1500)
    ("O=C(O)CCC[C@H](O)/C=C\\C=C\\CC[C@H](O)C/C=C\\CCCCC(O)(O)O",
     "(5S,6Z,8E,12S,14Z)-5,12,20,20,20-pentahydroxyicosa-6,8,14-trienoic acid"),
])
def test_geminal_prefix_multiset(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


def test_one_group_reaching_the_merge_twice_still_counts_once():
    # the cyano of this row reaches the prefix merge as two identical strings;
    # it is ONE group ('2-cyano', not '2,2-dicyano')
    _assert_named("CCNC(=O)N/C=C(\\C#N)/C(=O)OCC",
                  "ethyl (2E)-2-cyano-3-[(ethylcarbamoyl)amino]prop-2-enoate",
                  {"pin_verified"})


# --- (the Blue Book): a multiplied FG on alkyl is a compound prefix --

@pytest.mark.parametrize("smiles,expected", [
    ("OC(O)c1ccccc1C(=O)O", "2-(dihydroxymethyl)benzoic acid"),
    ("NC(N)c1ccccc1O", "2-(diaminomethyl)phenol"),
])
def test_multiplied_compound_prefix_is_enclosed(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- geminal suffixes on a ring: numbering by the multiset (c)) --------------

@pytest.mark.parametrize("smiles,expected", [
    ("OC1(O)CCCCC1", "cyclohexane-1,1-diol"),
    ("NC1(N)CCCCC1", "cyclohexane-1,1-diamine"),
    ("SC1(S)CCCC1", "cyclopentane-1,1-dithiol"),
    # {1,1,4} is lower than {1,4,4}
    ("OC1(O)CCC(O)CC1", "cyclohexane-1,1,4-triol"),
    ("OC1CCC(O)(O)CC1", "cyclohexane-1,1,4-triol"),
    ("OC1(O)CC(O)CCC1", "cyclohexane-1,1,3-triol"),
    ("NC1CCC(N)(N)CC1", "cyclohexane-1,1,4-triamine"),
    ("OC1(O)C=CC(O)CC1", "cyclohex-2-ene-1,1,4-triol"),
    # every ring -OH is a suffix once the alcohols are the principal class, so the
    # tertiary OH is numbered as one; then the prefix takes the low locant (f))
    ("CC1(O)CCC(O)CC1", "1-methylcyclohexane-1,4-diol"),
    ("CC1(O)CCCC(O)C1", "1-methylcyclohexane-1,3-diol"),
    ("NC1(O)CCC(O)CC1", "1-aminocyclohexane-1,4-diol"),
    ("OC1(Cl)CCC(O)CC1", "1-chlorocyclohexane-1,4-diol"),
])
def test_geminal_ring_suffix_numbering(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- review fix (finding 2): every -OH of the class is a suffix, and the suffixes are
# numbered first. (c) (the Blue Book) "principal characteristic groups and
# free valences (suffixes)" take the lowest locants; (:18875) the parent has
# the "maximum number of substituents corresponding to the principal characteristic
# group". (1) Perception: a geminal pair next to a secondary or tertiary -OH was not
# perceived at all -- the subtype suppression removed every generic 'alcohol'
# match that shared ANY atom with a subtype match, and a secondary/tertiary match lists
# its neighbour carbons, so the pair split between suffix and prefix ('1,1-dihydroxy-
# cyclopentan-2-ol'). (2) Numbering: the ring's anchor set was read from the principal
# SUBTYPE only, and a wholly exocyclic -CH2OH anchored its ring atom as if it were an
# appended suffix; an -ol has no exocyclic-carbon suffix form, so the hydroxymethyl
# ring atom took locant 1 ('1-(hydroxymethyl)cyclohexan-4-ol'). The Blue Book's own
# '4-(2-hydroxyethyl)-3-(hydroxymethyl)-2-methylidenecyclopentan-1-ol (PIN)' (:26890)
# and '2-(hydroxymethyl)benzene-1,4-diol (PIN)' (:6802) number the ring suffix first.
@pytest.mark.parametrize("smiles,expected", [
    ("OC1(O)CCCC1O", "cyclopentane-1,1,2-triol"),
    ("OC1C(O)(O)C(O)C(O)C1O", "cyclopentane-1,1,2,3,4,5-hexol"),
    ("O[C@@H]1CCCC1(O)O", "(2R)-cyclopentane-1,1,2-triol"),
    ("O[C@H]1C[C@@H](O)C(O)(O)C1", "(2R,4S)-cyclopentane-1,1,2,4-tetrol"),
    ("O[C@@H]1CCC[C@H](O)C1(O)O", "(2R,6S)-cyclohexane-1,1,2,6-tetrol"),
    ("OC1(O)CCCCC1(O)O", "cyclohexane-1,1,2,2-tetrol"),
    ("OC(O)C(O)C", "propane-1,1,2-triol"),
    ("C[C@@H](O)C(O)(O)C", "(3R)-butane-2,2,3-triol"),
    ("OC(O)[C@@H](O)[C@H](O)CO", "(2S,3R)-butane-1,1,2,3,4-pentol"),
    ("CC(Cl)(O)C(C)O", "2-chlorobutane-2,3-diol"),
    ("OC(N)C(C)O", "1-aminopropane-1,2-diol"),
    ("OC(O)(C(=O)O)C(O)C(=O)O", "2,2,3-trihydroxybutanedioic acid"),
])
def test_geminal_pair_beside_a_subtype_alcohol_is_perceived(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


@pytest.mark.parametrize("smiles,expected", [
    ("OC1(O)CCC(CO)CC1", "4-(hydroxymethyl)cyclohexane-1,1-diol"),
    ("OCC1CCC(O)(O)C1", "3-(hydroxymethyl)cyclopentane-1,1-diol"),
    ("OCC1CCC(O)CC1", "4-(hydroxymethyl)cyclohexan-1-ol"),
    ("OCC1CCCCC1O", "2-(hydroxymethyl)cyclohexan-1-ol"),
    ("OCC1CCC(O)C1", "3-(hydroxymethyl)cyclopentan-1-ol"),
    ("OC1CCC(CO)(CO)CC1", "4,4-bis(hydroxymethyl)cyclohexan-1-ol"),
    ("SCC1CCC(S)CC1", "4-(sulfanylmethyl)cyclohexane-1-thiol"),
    ("OCC1=CCC(O)CC1", "4-(hydroxymethyl)cyclohex-3-en-1-ol"),
    ("OCC1CC=CC(O)C1", "5-(hydroxymethyl)cyclohex-2-en-1-ol"),
    ("C=C1C(CO)C(CCO)CC1O",                                          # BB:26890
     "4-(2-hydroxyethyl)-3-(hydroxymethyl)-2-methylidenecyclopentan-1-ol"),
    # unchanged neighbours: appended suffixes keep their exocyclic anchor
    ("OC(=O)C1CCC(CO)CC1", "4-(hydroxymethyl)cyclohexane-1-carboxylic acid"),
    ("O=CC1CCC(CO)CC1", "4-(hydroxymethyl)cyclohexane-1-carbaldehyde"),
])
def test_ring_suffix_numbered_before_an_exocyclic_hydroxymethyl(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- 2. anions of mononuclear parent hydrides -----------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("[S-2]", "sulfanediide"),        # the M15 dev-set witness (milestone1500)
    ("[O-2]", "oxidanediide"),
    ("[Se-2]", "selanediide"),
    ("[Te-2]", "tellanediide"),
    ("[NH-2]", "azanediide"),         # BB:40945 names it
    ("[N-3]", "azanetriide"),
    ("[P-3]", "phosphanetriide"),
    ("[PH-2]", "phosphanediide"),
    ("[SH-]", "sulfanide"),
])
def test_mononuclear_hydride_anion(smiles, expected):
    _assert_named(smiles, expected, {"systematic_verified", "pin_verified"})


@pytest.mark.parametrize("smiles,expected", [
    ("[OH-]", "hydroxide"),                    # preselected, (:41015)
    ("[NH2-]", "azanide"),
    ("[Na+].[Na+].[S-2]", "disodium sulfide"),  # the salt keeps its table word
])
def test_mononuclear_anion_neighbours_unchanged(smiles, expected):
    _assert_named(smiles, expected, {"systematic_verified", "pin_verified"})


# Review fix (finding 7): a salt of element ions keeps the name the default tier gives
# it at the best-effort tier too. The bare-ion word 'azanetriide' let the best-effort
# rescues join the ions in the adduct notation ('lithium(1+)—azanetriide (3/1)'), the
# format of adducts, the Blue Book,:3765), where a salt is named "by
# citing the name of the cation(s) followed by the name of the anion",
#:31563). The bare ions keep their names. The label is the one of the
# naming path, as in the paper's measured run (user decision 2026-09-30: a carbon-free
# compound, 'sodium chloride' pin_verified): the PIN path's salts pin_verified, the
# retained-table name systematic_verified.
@pytest.mark.parametrize("smiles,expected,tier", [
    ("[Li+].[Li+].[Li+].[N-3]", "lithium nitride", "pin_verified"),
    ("[Mg+2].[Mg+2].[Mg+2].[N-3].[N-3]", "magnesium nitride", "pin_verified"),
    ("[N-3].[N-3].[Sr+2].[Sr+2].[Sr+2]", "strontium nitride", "pin_verified"),
    ("[Fr+].[NH2-]", "francium amide", "systematic_verified"),  # trivial_retained
])
def test_element_ion_salt_keeps_its_name_at_best_effort(smiles, expected, tier):
    from tests.support.rt_assert import name_best_effort
    for row in (name_best_effort(smiles), _pin_row(smiles)):
        assert row["name"] == expected, row
        assert row["tier"] == tier, row
    assert _key(_independent_parse(expected)) == _key(smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("[N-3]", "azanetriide"),
    ("[S-2]", "sulfanediide"),
    ("[Na+].[SH-]", "sodium sulfanide"),
])
def test_bare_ion_names_unchanged_at_best_effort(smiles, expected):
    from tests.support.rt_assert import name_best_effort
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert _key(_independent_parse(expected)) == _key(smiles)


# --- 3. diazonium: the suffix decides the parent, ------------------
# (the Blue Book-41615) "Cations containing an -N2+ group attached to
# a parent hydride are... named... by using the suffix 'diazonium'": 'methanediazonium
# (PIN)', '2,4-dioxopentane-3-diazonium (PIN)'. A cation outranks every neutral class
#,:18169), so the parent's own groups are prefixes and the chain is numbered
# from the suffix. Appending 'diazonium' to the neutral parent's name built
# 'pentane-2,4-dionediazonium' (refused) and 'propanediazonium' (locant dropped).

@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)C(C(C)=O)[N+]#N", "2,4-dioxopentane-3-diazonium"),   # BB:41615
    ("CC(=O)CC[N+]#N", "3-oxobutane-1-diazonium"),
    ("OCC[N+]#N", "2-hydroxyethane-1-diazonium"),
    ("C=CC[N+]#N", "prop-2-ene-1-diazonium"),
    ("CC(C)[N+]#N", "propane-2-diazonium"),
    ("CCC[N+]#N", "propane-1-diazonium"),
    ("CC(C)C[N+]#N", "2-methylpropane-1-diazonium"),
    ("CC(C)(C#N)[N+]#N", "2-cyanopropane-2-diazonium"),
    ("Cc1ccc([N+]#N)cc1", "4-methylbenzene-1-diazonium"),
])
def test_diazonium_named_from_its_suffix(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


@pytest.mark.parametrize("smiles,expected", [
    ("C[N+]#N", "methanediazonium"),                    # BB:41613
    ("c1ccccc1[N+]#N", "benzenediazonium"),
    ("N#[N+]c1ccc([N+]#N)cc1", "benzene-1,4-bis(diazonium)"),   # BB:41619
])
def test_diazonium_neighbours_unchanged(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


def test_diazonium_dev_set_witness_names_at_best_effort():
    # milestone1500 M15 row: named RT-exact at best-effort, labelled below PIN
    # ('eth-1-ene' cites a locant the PIN omits); it abstained before.
    from tests.support.rt_assert import name_best_effort
    smi = "CCOC(=O)CN/C(=C\\[N+]#N)/O"
    row = name_best_effort(smi)
    assert row["name"] == ("(1E)-2-[(2-ethoxy-2-oxoethyl)amino]-2-hydroxyeth-1-ene-"
                           "1-diazonium"), row
    assert row["tier"] == "pin_unverified", row
    assert _key(_independent_parse(row["name"])) == _key(smi)


# --- 4. one systematic benzene suffix beside a prefix cites its locant ---------------
# (the Blue Book); 'sodium 4-methylbenzene-1-thiolate (PIN)' (:43599),
# '3-[(4-sulfanylphenyl)disulfanyl]benzene-1-thiol (PIN)' (:27593).

@pytest.mark.parametrize("smiles,expected", [
    ("Cc1ccc(S)cc1", "4-methylbenzene-1-thiol"),
    ("Clc1ccc(S)cc1Cl", "3,4-dichlorobenzene-1-thiol"),
    ("Cc1ccc(C(N)=S)cc1", "4-methylbenzene-1-carbothioamide"),
    ("Sc1ccccc1", "benzenethiol"),                      # alone: no locant (:6656)
    ("Cc1ccc(C#N)cc1", "4-methylbenzonitrile"),         # retained parent: none
])
def test_benzene_single_suffix_locant(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- 5. X2: polyhydroxy dicarboxylate anions take the systematic PIN ----------------
# The carbohydrate acid name ('D-mannaric acid') is a natural-product name, and "PINs
# for the natural products in Chapter are not identified" (the Blue Book);
# the systematic acid's '-oic acid' becomes '-oate',:40955).

@pytest.mark.parametrize("smiles,expected", [
    ("O=C([O-])[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)[O-]",
     "(2S,3S,4S,5S)-2,3,4,5-tetrahydroxyhexanedioate"),
    ("O=C([O-])[C@H](O)[C@H](O)[C@@H](O)C(=O)[O-]",
     "(2R,4R)-2,3,4-trihydroxypentanedioate"),
    ("O=C([O-])[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)[O-]",
     "(2R,3S,4S,5S)-2,3,4,5-tetrahydroxyhexanedioate"),
    ("O=C([O-])[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C(=O)[O-]",
     "(2R,3S,4R,5S)-2,3,4,5-tetrahydroxyhexanedioate"),
])
def test_aldarate_takes_systematic_pin(smiles, expected):
    _assert_named(smiles, expected, {"pin_verified"})


# --- N-substituted geminal diamines: spelling and label -----------------------------
# A mononuclear parent's nitrogens are told apart by primes alone ('N,N'-...',
#, the Blue Book), and two amines on one carbon carrying the same
# N-substituent are the multiplicative name's non-preferred alternative there
# ("N,N'-methylenediethanamine (PIN) N,N'-diethylmethanediamine";:23180),
# so they ship labelled below PIN: a correct systematic name that is not the PIN,
# systematic_verified (user decision 2026-09-30; Methods, "Tiers").

@pytest.mark.parametrize("smiles,expected,tier", [
    ("CCNCNCC", "N,N'-diethylmethanediamine", "systematic_verified"),
    ("CN(C)CN(C)C", "N,N,N',N'-tetramethylmethanediamine", "systematic_verified"),
    ("CCNC(C)NCC", "N1,N'1-diethylethane-1,1-diamine", "systematic_verified"),
    ("CCNCN", "N-ethylmethanediamine", "pin_verified"),
    # BB:26369 'N3-ethyl-N1,N'3-dimethylhexane-1,3,3,6-tetramine (PIN)'
    ("CCNC(CCCN)(CCNC)NC", "N3-ethyl-N1,N'3-dimethylhexane-1,3,3,6-tetramine",
     "pin_verified"),
])
def test_n_substituted_geminal_diamine(smiles, expected, tier):
    _assert_named(smiles, expected, {tier})


# Review fix (finding 3): the X2 namer that leaves the carbohydrate catalog out runs a
# whole naming of the neutral skeleton. It now runs only for a skeleton that catalog
# would name, once per molecule, and the systematic FIX 3b namer is not repeated after a
# refused carbohydrate word; on the dev-set lipid and conjugate anions both were run on
# every call of name_anion (17-26 per row).
_LIPID_DIANION = ("CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])"
                  "OC[C@H](NC(=O)CCCCCCCCCCCCCCC)C(=O)[O-]")


def test_no_carbohydrate_namer_skips_a_skeleton_the_catalog_does_not_name(monkeypatch):
    import orthonym.namer as namer_mod
    from orthonym.rules import ions
    built = []
    real_name = namer_mod.Orthonym.name

    def counting_name(self, smiles, *a, **k):
        built.append(smiles)
        return real_name(self, smiles, *a, **k)
    monkeypatch.setattr(namer_mod.Orthonym, "name", counting_name)
    assert ions._try_neutralize_and_name_without_carbohydrate(
        Chem.MolFromSmiles(_LIPID_DIANION)) == ""
    assert built == []
    mannarate = "O=C([O-])[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)[O-]"
    assert ions._try_neutralize_and_name_without_carbohydrate(
        Chem.MolFromSmiles(mannarate)) == "(2S,3S,4S,5S)-2,3,4,5-tetrahydroxyhexanedioic acid"
    assert len(built) == 1


def test_multi_anion_row_does_not_repeat_the_neutral_namings(monkeypatch):
    from orthonym.rules import ions
    calls = {"systematic": 0, "no_carbohydrate": 0}
    real_sys = ions._try_neutralize_and_name_systematic
    real_nc = ions._name_neutral_without_carbohydrate

    def sys_spy(mol):
        calls["systematic"] += 1
        return real_sys(mol)

    def nc_spy(smiles):
        calls["no_carbohydrate"] += 1
        return real_nc(smiles)
    monkeypatch.setattr(ions, "_try_neutralize_and_name_systematic", sys_spy)
    monkeypatch.setattr(ions, "_name_neutral_without_carbohydrate", nc_spy)
    Orthonym().name_tiered(_LIPID_DIANION)
    # the released code ran the systematic namer 0 times on this row
    assert calls["systematic"] == 0, calls
    # one fresh computation per breadth context the nested namings run in
    assert calls["no_carbohydrate"] <= 6, calls
