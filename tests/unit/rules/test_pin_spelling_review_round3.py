"""PIN spelling and labels -- review fixes, a performance pass (TRIAGE.md 'Review fixes (a performance pass)'
under 'PIN spelling -- four classes from the PubChem 1M job').

Every expected name below was read back by an independent OPSIN 2.9.0 call to the
input's full InChIKey (``tests/support/rt_assert``); the old spelling is the mutation
each name test rejects (``scripts/mutation_check.py``).

    every multiplier join of the producers goes through the shared
         primitive (``naming_utils.multiplied_component``): (the Blue Book
         :7085, "Parentheses... are used to enclose multiplied components that
         are:") (a):7087 'di(propan-2-yl)', (c)/(d):7104 'di(dodecyl)',
         'di(decyl)'; (:7232) marks around a compound prefix;
         (a) (:7104) 'bis' for a substituted prefix, also one named by skeletal
         replacement ('2-ethyl-1-oxabutyl'; the 'a' prefixes are nondetachable,
         ,:4789).
    the ligand tokens of mononuclear parent hydrides: the second and later
         substituent prefixes are enclosed even when simple, the multiplier stays
         outside, the Blue Book; 'ethyldi(methyl)phosphane
         (PIN)', 'ethyldi(propan-2-yl)silane (PIN)', 'tert-butyldi(methyl)
         phosphane (PIN)':16286); a substituted ligand takes 'bis' (a)).
    locants are omitted in a substituent group or compound all of whose
         positions are labelled, the Blue Book, '(2H6)benzene
         (PIN)':44200): '(2H5)ethylbenzene'; a unit with a free position keeps
         its locants,:44202).
    a count subscript the preferred form omits ('ethan-1-(2H1)ol' where the
         Blue Book writes 'ethan(2H)ol (PIN)',,:44184) is only a parse
         fallback: the name ships below the PIN.
    an amidine on a heterocycle is the '-carboximidamide' suffix,
         the Blue Book, 'cyclohexanecarboximidamide (PIN)':34206), never
         a 'carbamimidoyl' prefix on the ring.
    the fluorenamine producer cites its N-prefixes in the one alphanumerical
         series,:3448), with their marks,:7232) and the
         hyphens before a locant (a)/(b),:6938,:6944).
    a multiplied prefix group ONE of whose copies carries every label is cited
         as two groups, the labelled copy first, the Blue Book,
         '2-(13C)methyl-3-methylpyridine (PIN)':43764).
    one name, one label at both tiers: a fragment named speculatively for an
         acyl prefix leaves no producer record on the outer name.
    a binary salt whose cation states its charge takes no stoichiometric
         prefixes, the Blue Book): 'iron(III) oxide'.
"""
import pytest

from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


def _named(smiles):
    res = name_best_effort(smiles)
    name = assert_full_rt(res.get("name"), smiles)
    return name, res.get("tier")


# --------------------------------------------------------------------------
# the shared primitive at every multiplier join
# --------------------------------------------------------------------------

MULTIPLIER_JOIN_ROWS = [
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)C(N)=O", "N,N-di(dodecyl)urea"),
    # all four N-H substituted in the same way: no N locants:3007,
    # 'tetrafluorourea (PIN)':3025)
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC",
     "tetra(dodecyl)urea"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)NC(=O)c1ccccc1",
     "N',N'-di(dodecyl)benzohydrazide"),
    ("CC(C)N(C(C)C)NC(=O)c1ccccc1", "N',N'-di(propan-2-yl)benzohydrazide"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)CC(=O)O", "[di(dodecyl)amino]acetic acid"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=N)N", "N,N-di(dodecyl)guanidine"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)c1ccc(N)cc1",
     "N1,N1-di(dodecyl)benzene-1,4-diamine"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)S(=O)(=O)c1ccc(C(=O)O)cc1",
     "4-[di(dodecyl)sulfamoyl]benzoic acid"),
    ("CC(C)N(C(C)C)S(=O)(=O)c1ccc(C(=O)O)cc1",
     "4-[di(propan-2-yl)sulfamoyl]benzoic acid"),
    ("ClCCN(CCCl)S(=O)(=O)c1ccc(C(=O)O)cc1",
     "4-[bis(2-chloroethyl)sulfamoyl]benzoic acid"),
    ("CCCCCCCCCCCCc1ccc(-c2ccc(CCCCCCCCCCCC)cc2)cc1",
     "4,4'-di(dodecyl)-1,1'-biphenyl"),
    ("CC(C)c1ccc(-c2ccc(C(C)C)cc2)cc1", "4,4'-di(propan-2-yl)-1,1'-biphenyl"),
    ("CC(C)N(C#N)C(C)C", "di(propan-2-yl)cyanamide"),
    ("CC(C)N(C(C)C)C(N)=O", "N,N-di(propan-2-yl)urea"),
    ("CCCCCCCCCCCCN(N)CCCCCCCCCCCC", "1,1-di(dodecyl)hydrazine"),
    ("CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)c1ccccc1", "N,N-di(dodecyl)benzamide"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLIER_JOIN_ROWS)
def test_multiplier_join_uses_the_shared_primitive(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


@pytest.mark.parametrize("prefix,substituted", [
    ("2-ethyl-1-oxabutyl", True),
    ("2-methyl-1-oxapropyl", True),
    ("3-methyl-1-oxa-2-silabutyl", True),
    ("2-methyl-3,6-dioxaheptyl", True),
    ("1-oxabutyl", False),
    ("3,6,9-trioxadecyl", False),
    ("oxolan-2-yl", False),
    ("azanediyl", False),
    ("propan-2-yl", False),
])
def test_prefix_on_a_replacement_named_parent(prefix, substituted):
    from orthonym.assembly.naming_utils import is_substituted_substituent
    assert is_substituted_substituent(prefix) is substituted


def test_substituted_replacement_named_prefix_takes_bis():
    from orthonym.assembly.naming_utils import multiplied_component
    assert (multiplied_component(2, "2-ethyl-1-oxabutyl", "(2-ethyl-1-oxabutyl)")
            == "bis(2-ethyl-1-oxabutyl)")
    # an unsubstituted 'a' chain is simple, but opens with a skeletal
    # replacement prefix: 'bis' by (c) (the Blue Book,:7174-7178)
    assert multiplied_component(2, "1-oxabutyl", "(1-oxabutyl)") == "bis(1-oxabutyl)"


@pytest.mark.parametrize("prefix,derived", [
    ("1-oxaethyl", True), ("2,5-dioxahexyl", True), ("1,2-oxazol-3-yl", True),
    ("oxan-2-yl", True), ("oxolan-2-yl", True), ("1H-1,2,4-triazol-1-yl", True),
    ("2-azabicyclo[2.2.1]heptan-2-yl", True),
    ("furan-2-yl", False), ("thiophen-2-yl", False), ("pyridin-2-yl", False),
    ("1H-indol-3-yl", False), ("fluoranthen-3-yl", False), ("oxo", False),
    ("oxalyl", False), ("phosphono", False), ("methyl", False),
])
def test_replacement_prefix_front_takes_bis(prefix, derived):
    """ (c) (the Blue Book, "before skeletal replacement ('a')
    prefixes, such as 'aza', 'oxa', etc. that are used in the construction of
    Hantzsch-Widman names and in skeletal replacement ('a') nomenclature, to
    describe clearly the number of replacement atoms";:7178 'bis(1,2-oxazol-3-
    yl)')."""
    from orthonym.assembly.naming_utils import (get_multiplier_prefix,
                                                opens_with_replacement_prefix)
    assert opens_with_replacement_prefix(prefix) is derived
    assert get_multiplier_prefix(2, prefix) == ("bis" if derived else "di")


# found in a performance pass: the branched alkyl of an ether prefix is enclosed inside
# the compound 'oxy' prefix, "(propan-2-yl)oxy (preferred prefix)",
# the Blue Book), "(butan-2-yl)oxy (preferred prefix)" (:27687),
# "tert-butoxy (preferred prefix)" (:27679)
ETHER_PREFIX_ROWS = [
    ("CC(C)OC(C)C", "2-[(propan-2-yl)oxy]propane"),
    ("CCC(C)OC(C)CC", "2-[(butan-2-yl)oxy]butane"),
    ("CC(C)(C)OC(C)C", "2-methyl-2-[(propan-2-yl)oxy]propane"),
    ("CC(C)(C)OCC(C)C", "1-tert-butoxy-2-methylpropane"),
    ("CC(C)OC(C)COC(C)C", "1,2-bis[(propan-2-yl)oxy]propane"),
]


@pytest.mark.parametrize("smiles,expected", ETHER_PREFIX_ROWS)
def test_branched_alkoxy_prefix_marks(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# found in a performance pass: a chain amide cites its N-substituents in the one
# alphanumerical series with the acyl prefixes, the Blue Book;
#,:3477), also when no name is shared
CHAIN_AMIDE_ORDER_ROWS = [
    ("O=C(Cc1ccncc1)Nc1ccccc1", "N-phenyl-2-(pyridin-4-yl)acetamide"),
    ("CCCC(C)C(=O)NCc1ccccc1", "N-benzyl-2-methylpentanamide"),
    ("CC(C)C(=O)Nc1ccc(O)cc1", "N-(4-hydroxyphenyl)-2-methylpropanamide"),
    ("ClCC(=O)NC", "2-chloro-N-methylacetamide"),
    ("CCN(CC)C(=O)C(C)Br", "2-bromo-N,N-diethylpropanamide"),
]


@pytest.mark.parametrize("smiles,expected", CHAIN_AMIDE_ORDER_ROWS)
def test_chain_amide_n_prefixes_in_one_series(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# --------------------------------------------------------------------------
# ligand tokens of a mononuclear parent hydride
# --------------------------------------------------------------------------

LIGAND_TOKEN_ROWS = [
    ("CC(C)(C)c1ccc([SiH2]c2ccc(C(C)(C)C)cc2)cc1", "bis(4-tert-butylphenyl)silane"),
    ("C[Si](C)(c1ccc(C(C)(C)C)cc1)c1ccc(C(C)(C)C)cc1",
     "bis(4-tert-butylphenyl)di(methyl)silane"),
    ("CC[SiH](C)C", "ethyldi(methyl)silane"),
    ("CC(C)(C)[SiH](C)C", "tert-butyldi(methyl)silane"),
    ("CO[Si](C)(C)C", "methoxytri(methyl)silane"),
    ("CC[SiH](C(C)C)C(C)C", "ethyldi(propan-2-yl)silane"),
    ("CC(C)(C)[Sn](C)(C)C", "tert-butyltri(methyl)stannane"),
    ("C[Si](C)(C)C", "tetramethylsilane"),
]


@pytest.mark.parametrize("smiles,expected", LIGAND_TOKEN_ROWS)
def test_ligand_tokens_of_a_mononuclear_hydride(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# --------------------------------------------------------------------------
# a completely labelled unit cites no locants
# --------------------------------------------------------------------------

COMPLETE_UNIT_ROWS = [
    ("[2H]C([2H])(c1ccccc1)C([2H])([2H])[2H]", "(2H5)ethylbenzene"),
    ("[13CH3][13CH2]c1ccccc1", "(13C2)ethylbenzene"),
    ("[2H]C([2H])([2H])C([2H])([2H])C([2H])([2H])[2H]", "(2H8)propane"),
    ("[2H]C([2H])([2H])C([2H])([2H])OC(C)=O", "(2H5)ethyl acetate"),
    ("[2H]c1c([2H])c([2H])c(CC(=O)O)c([2H])c1[2H]", "(2H5)phenylacetic acid"),
    # a free position is left in the unit: the locants stay
    ("[2H]c1c([2H])c([2H])c(-c2ccccc2)c([2H])c1[2H]",
     "(2,3,4,5,6-2H5)-1,1'-biphenyl"),
    ("[2H]c1c([2H])c([2H])c(COC(C)=O)c([2H])c1[2H]",
     "(2,3,4,5,6-2H5)benzyl acetate"),
    ("[2H]C([2H])([2H])C([2H])([2H])O", "(1,1,2,2,2-2H5)ethan-1-ol"),
    ("[2H]C([2H])(C)c1ccccc1", "(1,1-2H2)ethylbenzene"),
    ("[2H]C([2H])([2H])Cc1ccccc1", "(2,2,2-2H3)ethylbenzene"),
]


@pytest.mark.parametrize("smiles,expected", COMPLETE_UNIT_ROWS)
def test_completely_labelled_unit_omits_locants(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


def test_unit_completely_labelled_direct():
    from rdkit import Chem

    from orthonym.rules.isotopes import _unit_completely_labelled
    full = Chem.MolFromSmiles("[2H]C([2H])(c1ccccc1)C([2H])([2H])[2H]")
    part = Chem.MolFromSmiles("[2H]c1c([2H])c([2H])c(-c2ccccc2)c([2H])c1[2H]")
    assert _unit_completely_labelled("ethylbenzene", 0, [((2, "H"), 5, 3)], full)
    assert not _unit_completely_labelled(
        "1,1'-biphenyl", 0, [((2, "H"), 5, 1)], part)


# --------------------------------------------------------------------------
# the forced count subscript is labelled below the PIN
# --------------------------------------------------------------------------

# Labels: a verified name the code records as not the PIN (a name-scoped non-PIN
# record) is systematic_verified -- the paper, Methods, "Tiers" (L73): "a correct
# systematic name that is not the PIN"; user decision 2026-09-30 (the merged label
# rule of c4bf672b8).
FORCED_SUBSCRIPT_ROWS = [
    ("CCO[2H]", "ethan-1-(2H1)ol", "systematic_verified"),
    ("CCS[2H]", "ethane-1-(2H1)thiol", "systematic_verified"),
    ("CO[2H]", "methan(2H1)ol", "systematic_verified"),
    ("CC(C)O[2H]", "propan-2-(2H1)ol", "systematic_verified"),
    # the preferred form itself: unchanged
    ("CC(=O)O[2H]", "(O-2H)acetic acid", "pin_verified"),
    ("C[13CH2]O", "(1-13C)ethan-1-ol", "pin_verified"),
]


@pytest.mark.parametrize("smiles,expected,tier", FORCED_SUBSCRIPT_ROWS)
def test_forced_count_subscript_is_below_the_pin(smiles, expected, tier):
    name, got = _named(smiles)
    assert name == expected, name
    assert got == tier, got


# --------------------------------------------------------------------------
# the ring carboximidamide suffix of a heterocycle
# --------------------------------------------------------------------------

RING_AMIDINE_ROWS = [
    ("CNC(=N)c1ccc(Cl)nc1", "6-chloro-N-methylpyridine-3-carboximidamide"),
    ("NC(=N)c1ccc(Cl)nc1", "6-chloropyridine-3-carboximidamide"),
    ("CN(C)C(=N)c1ccc(Cl)nc1", "6-chloro-N,N-dimethylpyridine-3-carboximidamide"),
    ("CNC(=N)c1ccncc1", "N-methylpyridine-4-carboximidamide"),
    ("NC(=N)c1ccco1", "furan-2-carboximidamide"),
    ("CCNC(=N)c1cccs1", "N-ethylthiophene-2-carboximidamide"),
    ("NC(=N)C1CCCCN1", "piperidine-2-carboximidamide"),
    ("NC(=N)c1cc(C(=N)N)ccn1", "pyridine-2,4-dicarboximidamide"),
    ("NC(=N)c1ccc(C#N)nc1", "6-cyanopyridine-3-carboximidamide"),
    # a senior carboxamide keeps the suffix, the amidine is the prefix
    ("NC(=N)c1ccc(C(N)=O)nc1", "5-carbamimidoylpyridine-2-carboxamide"),
]


@pytest.mark.parametrize("smiles,expected", RING_AMIDINE_ROWS)
def test_ring_amidine_is_the_suffix(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# --------------------------------------------------------------------------
# the fluorenamine N-prefixes
# --------------------------------------------------------------------------

def test_fluorenamine_n_prefixes_in_one_series():
    smiles = ("[2H]C1=C(C(=C(C(=C1[2H])[2H])C2(C3=C(C=C(C=C3)N(C4=CC=CC=C4)"
              "C5=CC=CC6=CC=CC=C65)C7=CC=CC=C72)C8=CC9=C(C=C8)SC1=CC=CC=C19)[2H])[2H]")
    name, tier = _named(smiles)
    # the labelled phenyl is cited before the unlabelled one (,
    assert name == ("9-(dibenzo[b,d]thiophen-2-yl)-N-(naphthalen-1-yl)-"
                    "9-(2H5)phenyl-N-phenyl-9H-fluoren-3-amine"), name
    assert tier == "pin_unverified", tier


def test_join_pah_prefixes_hyphenates_before_any_locant():
    from orthonym.rules.polycyclics import _join_pah_prefixes
    assert (_join_pah_prefixes(["N-(naphthalen-1-yl)", "N-phenyl", "9-phenyl"])
            == "N-(naphthalen-1-yl)-N-phenyl-9-phenyl")
    assert _join_pah_prefixes(["2-chloro", "7-methyl"]) == "2-chloro-7-methyl"


# --------------------------------------------------------------------------
# one labelled copy of a multiplied group is cited apart, first
# --------------------------------------------------------------------------

ONE_LABELLED_COPY_ROWS = [
    ("[2H]c1c([2H])c([2H])c(-c2ccc3c(c2)[nH]c2cc(-c4ccccc4)ccc23)c([2H])c1[2H]",
     "2-(2H5)phenyl-7-phenyl-9H-carbazole"),
    ("[2H]c1c([2H])c([2H])c(Nc2ccnc(Nc3ccccc3)n2)c([2H])c1[2H]",
     "N4-(2H5)phenyl-N2-phenylpyrimidine-2,4-diamine"),
    ("[2H]C([2H])([2H])Nc1ccnc(NC)n1",
     "N4-(2H3)methyl-N2-methylpyrimidine-2,4-diamine"),
    ("[2H]C([2H])([2H])NC(=O)NC", "N-(2H3)methyl-N'-methylurea"),
    ("[2H]C([2H])([2H])N(C)C(=O)c1ccccc1", "N-(2H3)methyl-N-methylbenzamide"),
    ("[13CH3]c1ncccc1C", "2-(13C)methyl-3-methylpyridine"),
    ("CC(=O)Nc1ccc2c(c1)Cc1cc(I)c([131I])cc1-2",
     "N-[6-(131I)iodo-7-iodo-9H-fluoren-2-yl]acetamide"),
    # the three splitters that existed: the labelled copy first whatever its locant
    ("[13CH3]c1cccnc1C", "3-(13C)methyl-2-methylpyridine"),
    ("C[N+](C)([11CH3])CCO", "2-hydroxy-N-(11C)methyl-N,N-dimethylethan-1-aminium"),
    ("Cc1cc([13CH3])cc(C)c1", "1-(13C)methyl-3,5-dimethylbenzene"),
]


@pytest.mark.parametrize("smiles,expected", ONE_LABELLED_COPY_ROWS)
def test_one_labelled_copy_is_cited_first(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    assert tier == "pin_verified", tier


# --------------------------------------------------------------------------
# one name, one label at both tiers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CCC(C)c1ccc(OC(=O)C(=O)Oc2ccc(C(C)CC)cc2)cc1",
     "bis[4-(butan-2-yl)phenyl] oxalate"),
    ("CCC(C)(C)c1ccc(OC(=O)C(=O)Oc2ccc(C(C)(C)CC)cc2)cc1",
     "bis[4-(2-methylbutan-2-yl)phenyl] oxalate"),
])
def test_one_label_at_both_tiers(smiles, expected):
    from orthonym import Orthonym
    name, tier = _named(smiles)
    assert name == expected, name
    pin = Orthonym(style="pin").name_tiered(smiles)
    assert pin.get("name") == expected, pin.get("name")
    assert tier == pin.get("tier") == "pin_verified", (tier, pin.get("tier"))


# --------------------------------------------------------------------------
# a stated cation charge fixes the ratio: no stoichiometric prefixes
# --------------------------------------------------------------------------

CHARGE_FIXED_RATIO_ROWS = [
    ("[Fe+3].[Fe+3].[O-2].[O-2].[O-2]", "iron(III) oxide"),
    ("[Fe+3].[Cl-].[Cl-].[Cl-]", "iron(III) chloride"),
    ("[Fe+3].[Fe+3].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-]",
     "iron(III) sulfate"),
    ("[Co+2].[Cl-].[Cl-]", "cobalt(II) chloride"),
    # OPSIN 2.9.0 reads 'copper(I) oxide' as another species: unchanged
    ("[Cu+].[Cu+].[O-2]", "bis[copper(I)] oxidanediide"),
    # no stated charge: unchanged
    ("[Na+].[Na+].[O-]S(=O)(=O)[O-]", "disodium sulfate"),
]


@pytest.mark.parametrize("smiles,expected", CHARGE_FIXED_RATIO_ROWS)
def test_stated_cation_charge_drops_the_stoichiometric_prefixes(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    # carbon-free: the label of the naming path, as in the paper's measured run (user
    # decision 2026-09-30, 'sodium chloride' pin_verified; the merged rule of
    # c4bf672b8)
    assert tier == "pin_verified", tier


def test_rescue_memo_hit_keeps_the_producer_label():
    """A fragment-rescue memo hit replays the record the cold call made, so the
    decorated general-engine skeleton is labelled as its producer at the PIN tier,
    as with the memo off (the isotope decorator names its skeleton twice).

    Default tier (the paper, Methods, "Tiers", L73: "The default configuration emits a
    name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
    declines"; user decision 2026-09-30): a general-engine name is declined there with
    NO_VERIFIED_PIN, so the producer label is read from the strict path's row with the
    emission rule switched off, and the best-effort tier names the molecule with a name
    of its own that reads back exactly (tests/support/default_tier.py)."""
    from tests.support.default_tier import declined_pin_row
    smiles = ("[2H]C1=C(C(=C(C(=C1[2H])[2H])NC(=O)C2=C(N(C(=C2C3=CC=CC=C3)C4=CC=C(C=C4)F)"
              "CCC5CC(OC(O5)(C)C)CC(=O)OC(C)(C)C)C(C)C)[2H])[2H]")
    res = declined_pin_row(smiles, best_effort_same=False)
    assert_full_rt(res.get("name"), smiles)
    assert res.get("tier") == "systematic_verified", res.get("tier")
    assert res.get("source") == "general_engine", res.get("source")


# --------------------------------------------------------------------------
# found in a performance pass (end-check renames): two spellings that are no PIN are
# labelled below it; the names stay (each read back exact)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # the legacy locant-free ring prefix: 'aziridin-1-yl' is the preferred prefix
    # (2), the Blue Book; 'naphthalen-2-yl (preferred prefix)
    # 2-naphthyl (contracted name)',:2865)
    ("O=C1C=C(N2CC2)C(=O)C(N2CC2)=C1N1CC1",
     "2,3,5-tris(aziridinyl)cyclohexa-2,5-diene-1,4-dione"),
    # the descriptor in front of the D/L affix: 'L-(4-13C,35S)methionine'
    #,:44120-44128), which OPSIN 2.9.0 does not read
    ("[13CH3][C@H](N)C(=O)O", "(3-13C)L-alanine"),
    ("[14CH2]([14C@H]([14C](=O)O)N)S", "(14C3)D-cysteine"),
])
def test_non_pin_spelling_is_labelled_below_the_pin(smiles, expected):
    name, tier = _named(smiles)
    assert name == expected, name
    # a verified name the code records as not the PIN: systematic_verified (the paper,
    # Methods, "Tiers", L73; user decision 2026-09-30, the merged rule of c4bf672b8)
    assert tier == "systematic_verified", tier


def test_unlocanted_ring_prefix_vocabulary():
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    assert non_pin_vocabulary(
        "2,3,5-tris(aziridinyl)cyclohexa-2,5-diene-1,4-dione") == "aziridinyl"
    assert non_pin_vocabulary("3-(2-pyridyl)propan-1-ol") == "pyridyl"
    # one kind of substitutable hydrogen: no locant needed,:2939;
    # '*tert*-butyldi(methyl)(oxiranylmethoxy)silane (PIN)',:18929)
    assert non_pin_vocabulary("tert-butyldi(methyl)(oxiranylmethoxy)silane") is None
    assert non_pin_vocabulary("2,3,5-tris(aziridin-1-yl)cyclohexa-2,5-diene-1,4-dione") is None


# found in a performance pass: a descriptor for a replacement-named part goes before the
# part's own heteroatom locants, hyphen-joined, the Blue Book;
# the slot of '(15N)-1H-indole (PIN)',:43790), never between the locants and
# the 'a' prefix ('2-(2H8)azatricyclo'); the hyphen after a stereodescriptor
# block stays.
@pytest.mark.parametrize("smiles,expected", [
    ('[2H]C1=C(C(=C2C(=C1[2H])C3=C(C(=C(C(=C3N2C4=CC(=CC(=C4)C5=CC(=CC=C5)[Si]6(C7=C(C8=C6C=CC=N8)N=CC=C7)C9=CC=CC=C9)C1=CC=CC=C1)[2H])[2H])[2H])[2H])[2H])[2H]',
     '2-(3-{5-[(2H8)-2-azatricyclo[7.4.0.0^3,8]trideca-1(13),3,5,7,9,11-hexaen-2-yl]-3-phenylphenyl}phenyl)-2-phenyl-7,10-diaza-2-silatricyclo[7.4.0.0^3,8]trideca-1(13),3,5,7,9,11-hexaene'),
    ("[2H][C@@]12[C@@H](O1)CCC3=CC=CC=C23",
     "(2R,4S)-(2-2H)-3-oxatricyclo[5.4.0.0^2,4]undeca-1(11),7,9-triene"),
])
def test_descriptor_before_a_replacement_part_locant(smiles, expected):
    name, _tier = _named(smiles)
    assert name == expected, name
