"""PIN spelling defects: the name round-trips but breaks the Blue Book.

Pre-existing-failures plan (docs/the workflow tooling/plans/2026-09-24-preexisting-test-failures.md),
Task 5. Every row asserts the exact Blue Book spelling AND an OPSIN round trip to the
input's full standard InChIKey (``name_is_rt_exact``, independent of the pipeline's own
validity gate). The comment on each row cites the ruling (rule id, heading, line of
``the Blue Book Blue Book``).

``test_pin_spelling`` checks the PIN tier (``name_compound``). ``test_best_effort_spelling``
checks molecules the PIN tier does not name: the best-effort tier must name them, spelled
per the Blue Book.
"""
import pytest

from orthonym import name_compound
from tests.support.rt_assert import name_best_effort, name_is_rt_exact
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
    "CCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)O",
    "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset({
    "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1",
})


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


# The OPSIN validity gate ON: these rows assert what SHIPS. With the gate off (the suite
# default) a PIN-tier benzene producer that drops a halogen would leak through.
pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


PIN_ROWS = [
    # R11: "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES":9881 "The retained
    # names adamantane and cubane are used in general nomenclature and as preferred IUPAC
    # names."; Table 1.2 "bicyclo[2.2.1]heptane (PIN)":2038. Never 'norbornane'.
    ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),
    # R13: "Functional replacement in systematic names of carboxylic acids"
    #:30215 "tautomeric groups in mixed chalcocarboxylic acids... are distinguished by
    # prefixing italic element symbols, such as O or S... to the term 'acid'";
    # "hexanethioic O-acid (PIN)":30225 (-CS-OH, determined); the designator is dropped
    # only when the chalcogen position is unknown (:31081 vs:31083). The name is decided
    # on the structure (which atom carries the H); a full InChIKey cannot tell the
    # C(=Se)O / C(=O)[SeH] tautomers apart (ledger C1-09), so the Se-acid rows below
    # pin the other tautomer.
    ("CCC(=[Se])O", "propaneselenoic O-acid"),
    ("CCC(=[Te])O", "propanetelluroic O-acid"),
    ("CCC(=O)[SeH]", "propaneselenoic Se-acid"),
    ("CCC(=O)[TeH]", "propanetelluroic Te-acid"),
    # R5: "Retained names as preferred IUPAC names":29717 "Only the following
    # five carboxylic acids retained names and are also preferred IUPAC names. All can be
    # functionalized"; "acetic acid (PIN)":29725; "ethyl acetate (PIN)":31667;
    # "9 Esters" (:18182) outrank "17 Hydroxy compounds" (:18190) and "19 Amines" (:18192).
    ("CC(=O)OCC(N)C", "2-aminopropyl acetate"),
    ("OCCOC(C)=O", "2-hydroxyethyl acetate"),
    ("CC(=O)OC(C)CO", "1-hydroxypropan-2-yl acetate"),
    # R5, formic acid (PIN):29719: "methylene acetate formate (PIN)":31840.
    ("OCCOC=O", "2-hydroxyethyl formate"),
    # R5, oxamic acid (PIN) list; "amino(oxo)acetic acid" is not the PIN).
    ("NCCOC(=O)C(N)=O", "2-aminoethyl oxamate"),
    # R26: "NUMBERING":3219; when (c) the suffix set (:3256) and (f) the prefix set
    # (:3301) tie, "(g) lowest locants for the substituent cited first as a prefix in the
    # name" (:3307), example "4-methyl-5-nitrooctanedioic acid (PIN)" (:3316). Cited first
    # by the real prefix name: hydroxy < oxo, amino < chloro, hydroxy < iodo,
    # methoxy < methyl.
    ("O=C([O-])C(=O)C[C@H](O)C(=O)[O-]", "(2S)-2-hydroxy-4-oxopentanedioate"),
    ("OC(=O)C(O)CC(=O)C(=O)O", "2-hydroxy-4-oxopentanedioic acid"),
    ("OC(=O)C(N)CC(Cl)C(=O)O", "2-amino-4-chloropentanedioic acid"),
    ("OC(=O)C(I)CC(O)C(=O)O", "2-hydroxy-4-iodopentanedioic acid"),
    ("OC(=O)C(OC)CC(C)C(=O)O", "2-methoxy-4-methylpentanedioic acid"),
    # rt-14: every -OH (primary or secondary) is the principal characteristic group
    #:18875), and gives "(c) principal characteristic groups...
    # (suffixes)" (:3256) low locants before "(e) saturation/unsaturation" (:3288).
    ("OC/C=C/C#CC#C/C=C/C=C/C(O)CCO",
     "(4E,6E,12E)-tetradeca-4,6,12-trien-8,10-diyne-1,3,14-triol"),
    ("OCC=CCC(O)CCO", "hept-5-ene-1,3,7-triol"),
    ("NCC=CCC(N)CCN", "hept-5-ene-1,3,7-triamine"),
    # rt-20 / N1,N1 class: "Superscript arabic numbers, which are the locants of
    # the parent structure, are used to differentiate the nitrogen atoms of di- and
    # polyamines" (:7739); "N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)" (:26404).
    ("CN(C)c1ccc(N)cc1", "N1,N1-dimethylbenzene-1,4-diamine"),
    ("CNc1ccc(NC)cc1", "N1,N4-dimethylbenzene-1,4-diamine"),
    ("Nc1ccc(NCCC(C)C)cc1", "N1-(3-methylbutyl)benzene-1,4-diamine"),
    # The same diamine chain the Blue Book prints, "N4-(7-chloroquinolin-4-yl)-N1,N1-
    # diethylpentane-1,4-diamine" (:4675): suffix set {1,4} < {2,5} (c):3256).
    ("CCN(CC)CCCC(C)N", "N1,N1-diethylpentane-1,4-diamine"),
    ("CCN(CC)CCCC(C)NC", "N1,N1-diethyl-N4-methylpentane-1,4-diamine"),
    # Rows 146-147: (c):3256 before (f):3301 -- the ring suffix takes 1;
    # "3-imino-2,3-dihydro-1H-isoindol-1-one (PIN)" (:29609).
    ("COc1c(C)c(O)cc2c1C(=O)N[C@H]2C",
     "(3S)-5-hydroxy-7-methoxy-3,6-dimethyl-2,3-dihydro-1H-isoindol-1-one"),
    ("CC1NC(=O)c2ccccc21", "3-methyl-2,3-dihydro-1H-isoindol-1-one"),
    # R18::28410 (the ketone on the PIN parent xanthene,:11634); the sets
    # {1,3,6,8} tie, so (g):3307 gives the first-cited prefix (hydroxy) locant 1.
    ("COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
     "1-hydroxy-6,8-dimethoxy-3-methyl-9H-xanthen-9-one"),
    # R21::33580 cyclic imides as heterocyclic pseudoketones; "2-phenyl-1H-
    # isoindole-1,3(2H)-dione (PIN)":33853;:24689 (added indicated hydrogen).
    # The pyrazolyl keeps its indicated hydrogen;:2039).
    ("Cc1cc(N2C(=O)c3ccccc3C2=O)n(C)n1",
     "2-(1,3-dimethyl-1H-pyrazol-5-yl)-1H-isoindole-1,3(2H)-dione"),
    ("O=C1NC(=O)c2ccccc21", "1H-isoindole-1,3(2H)-dione"),
    ("CN1C(=O)c2ccccc2C1=O", "2-methyl-1H-isoindole-1,3(2H)-dione"),
    ("OC(=O)Cn1cccn1", "(1H-pyrazol-1-yl)acetic acid"),
    # Rows 127, 76/84, 79/86, 65 (enclosing marks and citation of composed prefixes):
    # (a compound organyl inside its own marks, the composing suffix outside:
    # "4-[(4-carboxycyclohexyl)oxy]":23198, "4-[(3-ethoxy-3-oxopropanoyl)oxy]phenyl":31790);
    # nesting "{[({})]}" (:7446); order by the complete name (:3477);
    # (g) (:3307); "{[(2-aminoethoxy)hydroxyphosphoryl]oxy}" (:55180).
    ("N[C@H]1[C@H](OCCCCCCS)O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H]1O",
     "(2R,3R,4R,5S,6R)-3-amino-6-(hydroxymethyl)-5-(α-D-mannopyranosyloxy)-2-[(6-sulfanylhexyl)oxy]oxan-4-ol"),
    ("C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",
     "(4-bromophenyl)[2-fluoro-4-({6-[methyl(prop-2-en-1-yl)amino]hexyl}oxy)phenyl]methanone"),
    ("O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1",
     "3-({3,4-dihydroxy-5-[(3,4,5-trihydroxybenzoyl)oxy]benzoyl}oxy)-4,5-dihydroxybenzoic acid"),
    ("OC(=O)c1ccc(OC(=O)c2ccc(O)cc2)cc1", "4-[(4-hydroxybenzoyl)oxy]benzoic acid"),
    # Rows 75/85: (:18875) the ring with the most principal-group -OH is the parent.
    ("COc1cc(O)cc(C)c1Oc1cc(C)cc(O)c1O",
     "3-(4-hydroxy-2-methoxy-6-methylphenoxy)-5-methylbenzene-1,2-diol"),
    # Carry (ester as a prefix): "General methodology" (:31663) "All
    # preferred IUPAC names for esters are named by functional class nomenclature";
    # the 'alkoxycarbonyl' prefix only beside a senior group,:31698).
    ("COC(=O)c1c[nH]c2ccccc12", "methyl 1H-indole-3-carboxylate"),
    ("COC(=O)c1ccc2ccccc2c1", "methyl naphthalene-2-carboxylate"),
    # Carry (fusion parent): "Seniority criteria for selecting the parent
    # component" (:12133), "(a) a component containing at least one of the heteroatoms
    # occurring earlier in the following order: N > F >... > O..." (:12139),
    # "(b) a component containing the greater number of rings" (:12163);
    # (:11907) heteroatom locants of a component in square brackets, "[1]benzopyrano
    # [2,3-c]pyrrole (PIN)" (:12157); the fusion letter "as early in the alphabet as
    # possible" (:11911); indicated hydrogen (:14605).
    ("c1ccc2c(c1)oc1ncccc12", "[1]benzofuro[2,3-b]pyridine"),
    ("c1ccc2c(c1)oc1cccnc12", "[1]benzofuro[3,2-b]pyridine"),
    ("c1ccc2c(c1)oc1cncnc12", "[1]benzofuro[3,2-d]pyrimidine"),
    ("c1ccc2c(c1)[nH]c1ncccc12", "9H-pyrido[2,3-b]indole"),
    # fix a performance pass, item 6 (the acyclic polyfunctional substituent path; were
    # '[di(tert-butyl)amino]methyl', '3-[di(tert-butyl)amino]propyl' and
    # '[(tert-butyl)(methyl)amino]methyl'). (a) (:7033): "unsubstituted
    # prefixes, such as ethyl or tert-butyl... are multiplied by the multiplicative
    # prefixes 'di', 'tri', etc."; (d) (:6960) the hyphen, 'di-tert-butyl'
    # (:6964); (:7272) the first cited substituent of a mononuclear parent
    # takes no marks, 'tert-butyldi(methyl)phosphane (PIN)' (:16286). OPSIN 2.9.0
    # full-InChIKey round trip (with -r, radicals): exact.
    ("CC(C)(C)N(C(C)(C)C)[CH2]", "(di-tert-butylamino)methyl"),
    ("CC(C)(C)N(C(C)(C)C)CC[CH2]", "3-(di-tert-butylamino)propyl"),
    ("CC(C)(C)N(C)[CH2]", "[tert-butyl(methyl)amino]methyl"),
    # fix a performance pass (wp5): the benzene substituent path (rules/benzene.py) put italic N
    # locants on a dialkylamino prefix (were '4-(N,N-dimethylamino)benzoic acid',
    # '4-(N-ethyl-N-methylamino)benzoic acid', '4-(N-methyl-N-propan-2-ylamino)benzoic
    # acid', '4-(N,N-diethylamino)phenol'). The nitrogen of an amino prefix is a
    # mononuclear parent: (:7272) "the first cited substituent never has
    # enclosing marks unless it includes a locant. The second and further substituents
    # are each enclosed with parentheses"; 'bis(dimethylamino) (preferred prefix)';
    # '4-(dimethylamino)-2-methylbutane-2-peroxol (PIN)' (:27955); '5-methyl-2-[methyl
    # (phenyl)carbamoyl]benzoic acid (PIN)' (:32957). OPSIN 2.9.0 full-InChIKey exact.
    ("CN(C)c1ccc(C(=O)O)cc1", "4-(dimethylamino)benzoic acid"),
    ("CCN(C)c1ccc(C(=O)O)cc1", "4-[ethyl(methyl)amino]benzoic acid"),
    ("CC(C)N(C)c1ccc(C(=O)O)cc1", "4-[methyl(propan-2-yl)amino]benzoic acid"),
    ("CCN(CC)c1ccc(O)cc1", "4-(diethylamino)phenol"),
    # fix a performance pass (wp5): an N-substituted carbamoyl halide / pseudohalide (were
    # '1-(di-tert-butylamino)formyl chloride', '1-(dimethylamino)formyl chloride',
    # '1-[ethyl(methyl)amino]formyl chloride', '1-(diethylamino)formyl fluoride',
    # '1-(dimethylamino)formyl isocyanate'; the phenyl rows abstained). The acyl group of
    # carbamic acid is 'carbamoyl'; 'carbamoyl' is preferred to 'carbonyl',
    #:24613), its N-substituents cited without a locant: '4-(dimethylcarbamoyl)benzoic
    # acid (PIN)' (:30382), '5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid (PIN)'
    # (:32957); functional class name 'carbamoyl isocyanate (PIN)' (:31488). OPSIN 2.9.0
    # full-InChIKey exact.
    ("CC(C)(C)N(C(=O)Cl)C(C)(C)C", "di-tert-butylcarbamoyl chloride"),
    ("CN(C)C(=O)Cl", "dimethylcarbamoyl chloride"),
    ("CCN(C)C(=O)Cl", "ethyl(methyl)carbamoyl chloride"),
    ("CN(C(=O)Cl)c1ccccc1", "methyl(phenyl)carbamoyl chloride"),
    ("CNC(=O)Cl", "methylcarbamoyl chloride"),
    ("CCN(CC)C(=O)F", "diethylcarbamoyl fluoride"),
    ("CN(C)C(=O)N=C=O", "dimethylcarbamoyl isocyanate"),
    # fix a performance pass (wp5): a substituted mancude / unsaturated cyclic anhydride (all
    # abstained at the PIN tier: the method (2) fallback 'butanedioic anhydride' drops
    # the ring C=C and the substituent). method (1), heterocyclic pseudo-
    # ketone: '3-bromofuran-2,5-dione (PIN) bromomaleic anhydride' (:32492), 'furan-2,5-
    # dione (PIN)' (:32490). OPSIN 2.9.0 full-InChIKey exact.
    ("O=C1OC(=O)C(Br)=C1", "3-bromofuran-2,5-dione"),
    ("O=C1OC(=O)C(C)=C1", "3-methylfuran-2,5-dione"),
    ("O=C1OC(=O)C(Cl)=C1Cl", "3,4-dichlorofuran-2,5-dione"),
    ("O=C1OC(=O)c2cc(Cl)ccc21", "5-chloro-2-benzofuran-1,3-dione"),
    ("O=C1OC(=O)C2CC=CCC12", "3a,4,7,7a-tetrahydro-2-benzofuran-1,3-dione"),
    # fix a performance pass (wp5): a branched / unsaturated / ether-bearing chain on a catalog
    # PAH (all abstained at the PIN tier: the PAH identifier named alkyls by carbon count,
    # 'propyl' for propan-2-yl, and dropped an ether chain).: 'propan-2-yl
    # (preferred prefix) (not prop-2-yl)' (:15946), 'butan-2-yl (preferred prefix)'
    # (:15951); (:16334) isopropyl only in general nomenclature; 'tert-butyl
    # (preferred prefix)' (:24412); ethers cited as prefixes. OPSIN 2.9.0
    # full-InChIKey exact.
    ("CC(C)c1ccc2ccccc2c1", "2-(propan-2-yl)naphthalene"),
    ("CCC(C)c1ccc2ccccc2c1", "2-(butan-2-yl)naphthalene"),
    ("CC(C)Cc1ccc2ccccc2c1", "2-(2-methylpropyl)naphthalene"),
    ("CC(C)(C)c1ccc2ccccc2c1", "2-tert-butylnaphthalene"),
    ("C=Cc1ccc2ccccc2c1", "2-ethenylnaphthalene"),
    ("ClCc1ccc2ccccc2c1", "2-(chloromethyl)naphthalene"),
    ("COCCc1ccc2ccccc2c1", "2-(2-methoxyethyl)naphthalene"),
    ("CC(C)c1ccc2cc3ccccc3cc2c1", "2-(propan-2-yl)anthracene"),
    # the same identifier, an S-attached sulfide (abstained): (:27817) method
    # (1) substitutive, '(R)sulfanyl' prefixes, 'methylsulfanyl (preferred prefix)'
    # (:27651). OPSIN 2.9.0 full-InChIKey exact.
    ("CSc1ccc2ccccc2c1", "2-(methylsulfanyl)naphthalene"),
    ("CCSc1cccc2ccccc12", "1-(ethylsulfanyl)naphthalene"),
    # fix a performance pass (wp5): monospiro numbering (rest-of-suite s43). The small-ring
    # direction is free, 'spiro[4.5]dec-6-ene (PIN)':16721), and
    # NUMBERING (:3219) gives the suffix its lowest locant, "(c) principal
    # characteristic groups and free valences (suffixes)" (:3256), before the 'ene'
    # ending and before "(f) detachable alphabetized prefixes, all considered together"
    # (:3301). Were 'spiro[4.5]dec-1-en-3-one', '2-methylspiro[4.5]dec-1-en-3-one',
    # '6-methylspiro[4.5]decan-3-one' and the s43 name '(5S,6R,9R,10R)-10-hydroxy-9-
    # (2-hydroxypropan-2-yl)-2,6-dimethylspiro[4.5]dec-1-en-3-one'. OPSIN 2.9.0
    # full-InChIKey exact.
    ("O=C1CC2(C=C1)CCCCC2", "spiro[4.5]dec-3-en-2-one"),
    ("OC1CC2(C=C1)CCCCC2", "spiro[4.5]dec-3-en-2-ol"),
    ("CC1=CC2(CC1=O)CCCCC2", "3-methylspiro[4.5]dec-3-en-2-one"),
    ("CC1CCCCC12CCC(=O)C2", "6-methylspiro[4.5]decan-2-one"),
    ("CC1(C)CCC2(CCC(=O)C2)CC1", "8,8-dimethylspiro[4.5]decan-2-one"),
    ("CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O",
     "(5S,6R,7R,10R)-6-hydroxy-7-(2-hydroxypropan-2-yl)-3,10-dimethylspiro[4.5]dec-3-en-2-one"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS, ids=[r[0] for r in PIN_ROWS])
def test_pin_spelling(smiles, pin):
    name = _dt_name_compound(smiles)
    assert name == pin
    assert name_is_rt_exact(name, smiles), f"{name!r} does not round-trip to {smiles}"


# PIN-tier emissions that are not yet the PIN: only the spelling this task fixed is
# locked. The PIN of a phosphoric acid partial ester is the 'hydrogen phosphate'
# functional class name, (:40993) "Preferred IUPAC names of acid esters of
# inorganic acids... are formed by the method of 'hydrogen salts'" -- a named residual
# producer (TRIAGE.md, outcome). Each row: (smiles, present, absent).
SPELLING_FEATURE_ROWS = [
    # Carry (propane-2-ol): (a) (:7595) elides the parent's final 'e' before
    # a vowel-initial suffix; enclosing marks by (:7446) and order.
    ("CCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)O",
     ("propan-2-ol", "1-{[(1Z)-hexadec-1-en-1-yl]oxy}",
      "3-({[2-(trimethylazaniumyl)ethoxy]hydroxyphosphoryl}oxy)"),
     ("propane-2-ol",)),
    # Carry (ditert-butyl): (d) (:6960) a hyphen separates italic letters
    # from Roman letters, "di-tert-butyl" (:6964).
    ("CC(C)(C)N(C(C)(C)C)CC(=O)O", ("(di-tert-butylamino)",), ("ditert",)),
]


@pytest.mark.parametrize("smiles,present,absent", SPELLING_FEATURE_ROWS,
                         ids=[r[0] for r in SPELLING_FEATURE_ROWS])
def test_spelling_features(smiles, present, absent):
    name = _dt_name_compound(smiles)
    for token in present:
        assert token in name, f"{token!r} missing from {name!r}"
    for token in absent:
        assert token not in name, f"{token!r} present in {name!r}"
    assert name_is_rt_exact(name, smiles), f"{name!r} does not round-trip to {smiles}"


def test_ester_cited_as_prefix_is_not_labelled_pin():
    """ (:31663): an ester that is the principal characteristic group is
    named by functional class nomenclature in a PIN. A ring-cascade name citing it as
    an acyloxy prefix is valid but not preferred, so it never ships as pin_verified;
    it still ships (breadth), round-trip exact."""
    from orthonym import Orthonym
    smiles = "CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1"
    res = _dt_row(smiles)
    assert res.get("name") and "oxy]" in res["name"], res
    assert res.get("tier") != "pin_verified", res
    assert name_is_rt_exact(res["name"], smiles), res


def test_closing_mark_then_locant_takes_a_hyphen():
    """ (b) (:6944): a hyphen after the closing enclosing mark when a
    locant follows -- for ']' and '}' as for ')'."""
    from orthonym.rules.fused_rings import _join_fused_prefixes
    assert _join_fused_prefixes(["7-[(1R)-1-hydroxyethyl]", "3-(2-methylpropyl)"]) \
        == "7-[(1R)-1-hydroxyethyl]-3-(2-methylpropyl)-"
    assert _join_fused_prefixes(["2-{[(2S)-x]methyl}", "4-methyl"]) \
        == "2-{[(2S)-x]methyl}-4-methyl-"


def test_benzyl_alcohol_component_is_benzyl():
    """ (:24414) "C6H5-CH2- benzyl (preferred prefix) phenylmethyl"."""
    from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl
    assert _alcohol_to_alkyl("phenylmethanol") == "benzyl"


BEST_EFFORT_ROWS = [
    # The PIN tier does not name these benzene aryl ethers (its benzene vocabulary
    # stops short); the best-effort tier does, and must spell them as the Blue Book does.
    # R15: "The locant '1' is omitted:":2891 "(c) in monosubstituted
    # homogeneous monocyclic rings":2913; "bromobenzene (PIN)":2919,
    # "(cyclohexyloxy)benzene (PIN)":27768.
    ("ClCCOc1ccccc1", "(2-chloroethoxy)benzene"),
    ("FC(F)(F)Oc1ccccc1", "(trifluoromethoxy)benzene"),
    # R17: "NAMING OF STEREOISOMERS":44643 (stereodescriptors "are cited at the
    # front of the corresponding prefix"); "[(1R)-1-chloropropyl]benzene (PIN)":44668;
    #:7478 (stereo parentheses count in the nesting order).
    ("C[C@H](Cl)COc1ccccc1", "[(2S)-2-chloropropoxy]benzene"),
    ("ClCC[C@@H](Cl)Oc1ccccc1", "[(1R)-1,3-dichloropropoxy]benzene"),
    # Disubstituted: the locants stay:2869).
    ("FC(F)(F)COc1ccc(F)cc1F", "2,4-difluoro-1-(2,2,2-trifluoroethoxy)benzene"),
]


@pytest.mark.parametrize("smiles,expected", BEST_EFFORT_ROWS,
                         ids=[r[0] for r in BEST_EFFORT_ROWS])
def test_best_effort_spelling(smiles, expected):
    name = name_best_effort(smiles).get("name")
    assert name == expected
    assert name_is_rt_exact(name, smiles), f"{name!r} does not round-trip to {smiles}"
