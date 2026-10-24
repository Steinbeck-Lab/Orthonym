"""PIN class program, Task 14: parent seniority (HETERO_PARENT, CHALCOGEN_PARENT,
RING_SENIORITY, PHOSPHONIC_PARENT, DIAZENE_PARENT).

 (the Blue Book) "The senior parent structure, whether cyclic or acyclic, has
the senior atom in accordance with the seniority of classes (see expressed by the
following decreasing element order: N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga >
In > Tl > O > S > Se > Te > C. This criterion is applied to select the senior atom in parents
and to choose between rings and chains." (:18919) "A single senior atom is
sufficient to give seniority to the parent hydride."

(A) Ring seniority across a ring=CH-ring linker. (:6384) "Unsymmetrical
    substituent groups having terminal atoms with different types of free valencies":
    '(cyclohexylidenemethyl)benzene (PIN)' (:6390) "[not (phenylmethylidene)cyclohexane;
    benzene is senior to cyclohexane (see ]" (:6392). (:21033) "The
    senior ring, ring system or principal chain has the greater number of multiple bonds".
    The ring criteria of come first: a heterocycle,:19445), more rings
    ,:19461), more skeletal atoms,:19469; so cycloheptane is senior
    to benzene and 'benzylidenecycloheptane' keeps its parent). When the two rings tie,
     (:21600) "the maximum number of substituent groups cited as prefixes", then
     (:22234) "The preferred IUPAC name is the name that is earlier in alphanumerical
    order": '-ylidenemethyl' is earlier than '-ylmethylidene', the comparison of
     example (2) (:6398), whose ring pair is the one that falls the other way (the
    first cited prefix is 'ethylidene' there, so 'e' before 'i' of 'ylidene' decides, and the
    parent is not the same ring). Between and stands **** (:21698) "The
    preferred IUPAC name is based on the senior parent structure that has the lower locant or
    set of locants for substituents cited as prefixes (other than 'hydro/dehydro') to the
    parent structure" under " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT
    GROUPS" (:21596), the locant sets compared as "Lowest set of locants" (:3189)
    defines: '1-methyl-3-[(4-methylcyclohexylidene)methyl]cyclohexane' ({1,3}), not
    '1-methyl-4-[(3-methylcyclohexyl)methylidene]cyclohexane' ({1,4}). Where the sets tie and
    the groups differ, (:21791) compares the locants in the order of citation of the two
    names and (:22234) their letters ('1-[(4-ethylcyclohexyl)methylidene]-4-methylcyclohexane':
    'ethylc' before 'ethylm', ``rules/ring_tie_names.py``, tests in ``test_ring_tie_names.py``); a ring
    that is not a saturated monocycle, or names that tie, are not decided and the name is labelled
    below the PIN. The criteria of that come before are applied first: (h) the lower
    locants of the suffix,:21305; '4-[(2,4-dihydroxycyclohexyl)methylidene]cyclohexane-1,2-diol',
    the ring with the 1,2 locants, not the one with 1,3) and (k) the isotopically modified atoms of the
    parents to.6,:21477-:21523), rows in ``test_ring_tie_names.py``. The compound prefix is enclosed
    .
    ``rules/ring_selection.py:select_principal_ring_system``, ``assembly/substituent_naming.py:
    _name_polyfunctional_acyclic_substituent_impl`` (the '-ylidene' ring branch),
    ``assembly/naming_utils.py`` (enclosing marks).
(B) A mononuclear senior-atom parent hydride with substituted organyl groups. With no
    principal characteristic group the silane/germane/arsane/phosphane hub is the parent
    : 'chloro(2-silylethyl)silane (PIN)',:6245; "[not
    [2-(chlorosilyl)ethyl]silane; the parent silane of the PIN has more substituents; see
    ]"), '(1-chloro-2-methylpropan-2-yl)silane (PIN)',:16288),
    'trichloro(iodomethyl)silane (PIN; silane is a preselected name; see '
    ,:25870), 'methoxydi(methyl)[2-(trimethylgermyl)phenyl]silane (PIN) (Si is
    senior to Ge)',:38258), '(4-stibanylphenyl)arsane (PIN)',
    :39268), '[2-(1,3-dioxolan-2-yl)ethyl]tri(methyl)silane (PIN)' (:35326). The prefixes
    follow (:7272): "For mononuclear parent hydrides with two or more
    substituents the first cited substituent never has enclosing marks unless it includes a
    locant. The second and further substituents are each enclosed with parentheses even for
    simple substituents." A principal characteristic group still decides first: '3-
    (trimethoxysilyl)propane-1-thiol (PIN) [not trimethoxy(3-sulfanylpropyl)silane; the
    suffix, 'thiol', has precedence over silane)',:28164). A substituent's own
    parent carries its free valence,:15762): the -CH2-CH2-SiH3 group is
    '2-silylethyl', never 'ethylsilyl'. ``rules/mononuclear_hydrides.py:
    _name_senior_atom_hub``, ``assembly/substituent_naming.py`` (Step 5 of
    ``_name_substituent_fragment_uncached``). The organyl of a sulfide, selenide or telluride link is
    cited as '(R)sulfanyl' / '(R)selanyl' / '(R)tellanyl' "Names of chalcogen analogues of
    ethers, i.e., sulfides, selenides and tellurides",:27802; "The prefixes R′-sulfanyl, R′-selanyl,
    and R′-tellanyl are compulsory prefixes and can be attached to any atom of any parent hydride",
    :27808): 'trimethyl[2-(methylselanyl)ethyl]silane'. Where the engine does not build the hub
    parent (an organyl it cannot name, two hubs that tie, a linker the multiplicative namer does not
    take), a name on a carbon parent is correct but is not the PIN, and
    ``rules/mononuclear_hydrides.py:senior_hub_parent_expected`` labels it below the PIN
    (``assembly/inner_dispatch.py``, as for urea and guanidine).
(C) A heterogeneous chalcogen chain a[ba]n is the parent hydride. (:39403)
    "Compounds of the type a[ba]n are parent hydrides; they are discussed in ":
    'CH3-S-O-S-CH3 dimethyldithioxane (PIN)' (:39415), 'C6H5-S-O-S-CH3
    methyl(phenyl)dithioxane (PIN)' (:39417). A ring or halogen inside a terminal
    organyl leaves the chain the parent. ``rules/catenated_hydrides.py:
    name_heterochalcogen_aba``.
(D) An organyl phosphonic/arsonic/phosphinic acid is named on the acid. (:35453)
    "Acids with hydrogen atoms attached to the central atom may be substituted by organyl
    groups and preferred IUPAC names are formed in this manner": 'C2H5-P(O)(OH)2
    ethylphosphonic acid (PIN) (not ethanephosphonic acid)' (:35461). Junior groups sit in
    the organyl prefix: (:36534) '(4-arsonobutyl)phosphonic acid (PIN) (a
    phosphorus acid is senior to an arsenic acid)' (:36542); 'arsono' is a preselected
    prefix,:36030); a terminal amide on a chain is 'amino' + 'oxo'
    ,:32922). A stereodescriptor of the organyl stays on its prefix:
    (:44639) "When they relate to substituent groups, they are cited at the front of the
    corresponding prefix" (:44643). ``rules/polyfunctional.py:_pnictogen_oxoacid_parent_name``,
    ``rules/phosphorus.py:_organyl_or_junior_prefix`` and ``carbon_free_phospho_prefix``,
    ``assembly/handlers/phosphonic_acid.py`` / ``phosphinic_acid.py`` / ``pnictogen_oxoacid.py``.
(E) Diazene is the parent of an added-carbon suffix (N is senior to C,.
     "Substitution of diazene" (:38730): 'diazenedicarbaldehyde (PIN)',
    'phenyldiazenecarbonitrile (PIN)' (:38738). (:38896) "The hydrazide of
    'diazenecarboxylic acid', HN=N-CO-NH-NH2, is named systematically
    'diazenecarbohydrazide'... Chalcogen analogues are named by using the infixes and the
    prefixes 'thio', 'seleno', and 'telluro'" (:38898): 'diazenecarbothiohydrazide (PIN)'
    (:38902). ``rules/mononuclear_hydrides.py:name_diazene_added_carbon``.
"""
import pytest

from tests.support.rt_assert import name_is_rt_exact
from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)

pytestmark = pytest.mark.opsin_gate

# (A) RING_SENIORITY ---------------------------------------------------------------------
RING_ROWS = [
    ("C(=C1CCCCC1)c1ccccc1", "(cyclohexylidenemethyl)benzene"),               #:6390
    # the same rule on the class /.5 /,,
    ("C(=C1CCCC1)c1ccccc1", "(cyclopentylidenemethyl)benzene"),
    ("C1CCC(CC1)=Cc1ccccc1Cl", "1-chloro-2-(cyclohexylidenemethyl)benzene"),
    ("CC1CCC(=Cc2ccccc2)CC1", "[(4-methylcyclohexylidene)methyl]benzene"),
    ("C(=C1CCCCC1)c1ccncc1", "4-(cyclohexylidenemethyl)pyridine"),
    ("C(=C1CCCCC1)c1ccc2ccccc2c1", "2-(cyclohexylidenemethyl)naphthalene"),
    ("C(=C1CCCCC1)c1ccccc1C(=O)O", "2-(cyclohexylidenemethyl)benzoic acid"),
    ("C(=C1CCCCC1)C1CCCCC1", "(cyclohexylidenemethyl)cyclohexane"),
    ("C(=C1CCCCC1)C1CCC(C)CC1", "1-(cyclohexylidenemethyl)-4-methylcyclohexane"),
    ("CC1CCC(=CC2CCCCC2)CC1", "1-(cyclohexylmethylidene)-4-methylcyclohexane"),
    # (:21698): the parent ring that gives the lower locant set; {1,3} before {1,4}
    ("CC1CCC(CC1)=CC1CCCC(C)C1", "1-methyl-3-[(4-methylcyclohexylidene)methyl]cyclohexane"),
    ("CC1CCCC(C1)=CC1CCC(C)CC1", "1-methyl-3-[(4-methylcyclohexyl)methylidene]cyclohexane"),
    # (:21600) is applied first: three substituents against two
    ("CC1CCC(CC1)=CC1CCCC(C)C1C", "1,2-dimethyl-3-[(4-methylcyclohexylidene)methyl]cyclohexane"),
    # the two sides alike: the ylidene is cited ('-ylidenemethyl' before '-ylmethylidene',
    ("CC1CCC(CC1)=CC1CCC(C)CC1", "1-methyl-4-[(4-methylcyclohexylidene)methyl]cyclohexane"),
    # the sides differ and the locants tie: the letters of the two whole names decide,:22234)
    ("CC1CCC(CC1)=CC1CCC(CC)CC1", "1-[(4-ethylcyclohexyl)methylidene]-4-methylcyclohexane"),
    ("CC1CCC(CC1)=CC1CCC(OC)CC1", "1-[(4-methoxycyclohexyl)methylidene]-4-methylcyclohexane"),
    # (:21791): the prefix sets are 2,3 and 2,3 and the locants in the order of citation, 2,3 and 3,2, decide
    ("C1(=CC2C(C)C(O)CCC2)C(O)CCCC(C)1", "2-[(3-hydroxy-2-methylcyclohexyl)methylidene]-3-methylcyclohexan-1-ol"),
    # compares the locants of the PREFIXES: a ring that carries the principal group is numbered suffix
    # first, so the prefix sets are 2,4 and 2,6, not the free 1,2,4 and 1,2,3 of the groups with
    # the suffix counted; the -ol, -carboxylic acid and -amine alike
    ("OC1C(C)CC(=CC2C(O)C(C)CCC2)CC1", "4-[(2-hydroxy-3-methylcyclohexyl)methylidene]-2-methylcyclohexan-1-ol"),
    ("OC(=O)C1C(C)CC(=CC2C(C(O)=O)C(C)CCC2)CC1",
     "4-[(2-carboxy-3-methylcyclohexyl)methylidene]-2-methylcyclohexane-1-carboxylic acid"),
    ("NC1C(C)CC(=CC2C(N)C(C)CCC2)CC1", "4-[(2-amino-3-methylcyclohexyl)methylidene]-2-methylcyclohexan-1-amine"),
    ("C1(=CC2C(O)C(C)CCC2)C(O)CC(C)CC1", "2-[(2-hydroxy-3-methylcyclohexyl)methylidene]-5-methylcyclohexan-1-ol"),
    # (:18875): the ring with more principal groups, or the only one, is the parent before
    ("OC1CCC(CC1)=CC1CCC(O)C(O)C1", "4-[(4-hydroxycyclohexylidene)methyl]cyclohexane-1,2-diol"),
    ("OC1CCC(CC1)=CC1CCC(C)CC1", "4-[(4-methylcyclohexyl)methylidene]cyclohexan-1-ol"),
    ("OC1CCC(CC1)=CC1CCC(C)(C)CC1", "4-[(4,4-dimethylcyclohexyl)methylidene]cyclohexan-1-ol"),
    # a group on a chain ('-CH2-OH' makes the chain the parent) is no suffix of the ring
    ("OCC1CCC(CC1)=CC1CCCCC1", "[4-(cyclohexylmethylidene)cyclohexyl]methanol"),
]
# two rings of equal substituent counts and locant sets whose groups differ: and decide on the two
# names (test_ring_tie_names.py); the rings that are not saturated monocycles are not worked out, so the name is
# labelled below the PIN
RING_UNDECIDED_ROWS = [
    "CC1=CCC(CC1)=CC1CCC(CC)C=C1",
    "CC1=CCC(CC1)=CC1CCC(C)C=C1",
]
RING_NOT_PIN_ROWS = [
    ("C(=C1CCCCC1)c1ccccc1", "benzylidenecyclohexane"),
    ("C(=C1CCCCC1)c1ccncc1", "[(pyridin-4-yl)methylidene]cyclohexane"),
    ("C(=C1CCCCC1)C1CCCCC1", "cyclohexylmethylidenecyclohexane"),
    ("CC1CCC(CC1)=CC1CCCC(C)C1", "1-methyl-4-[(3-methylcyclohexyl)methylidene]cyclohexane"),  # {1,4} > {1,3}
    # the prefix sets of the -ol rows are 2,4 and 2,6; the free numbering of all the groups gave 1,2,4 and 1,2,3
    ("OC1C(C)CC(=CC2C(O)C(C)CCC2)CC1", "2-[(4-hydroxy-3-methylcyclohexylidene)methyl]-6-methylcyclohexan-1-ol"),
    ("OC(=O)C1C(C)CC(=CC2C(C(O)=O)C(C)CCC2)CC1",
     "2-[(4-carboxy-3-methylcyclohexylidene)methyl]-6-methylcyclohexane-1-carboxylic acid"),
    ("NC1C(C)CC(=CC2C(N)C(C)CCC2)CC1", "2-[(4-amino-3-methylcyclohexylidene)methyl]-6-methylcyclohexan-1-amine"),
]
RING_CONTROL_ROWS = [
    ("C(=C1CCCCCC1)c1ccccc1", "benzylidenecycloheptane"),        #
    ("OC1CCC(=Cc2ccccc2)CC1", "4-benzylidenecyclohexan-1-ol"),   # the principal group
    ("C(=C1CCc2ccccc21)c1ccccc1", "1-benzylidene-2,3-dihydro-1H-indene"),
    ("C=C1CCCCC1", "methylidenecyclohexane"),
    ("C1CCCCC1Cc1ccccc1", "(cyclohexylmethyl)benzene"),
    ("OC(=O)C=C1CCCCC1", "cyclohexylideneacetic acid"),
]


@pytest.mark.parametrize("smiles,pin", RING_ROWS + RING_CONTROL_ROWS)
def test_ring_seniority_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", RING_NOT_PIN_ROWS)
def test_ring_seniority_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles", RING_UNDECIDED_ROWS)
def test_ring_seniority_undecided_tie_is_not_pin(smiles):
    assert_declined_at_default(smiles)


# (B) HETERO_PARENT ------------------------------------------------------------------------
HETERO_ROWS = [
    ("[SiH3]CC[SiH2]Cl", "chloro(2-silylethyl)silane"),                        #:6245
    ("CC(C)([SiH3])CCl", "(1-chloro-2-methylpropan-2-yl)silane"),              #:16288
    ("Cl[Si](Cl)(Cl)CI", "trichloro(iodomethyl)silane"),                       #:25870
    ("CO[Si](C)(C)c1cccc[c]1[Ge]([CH3])([CH3])[CH3]",
     "methoxydi(methyl)[2-(trimethylgermyl)phenyl]silane"),                    #:38258
    ("[AsH2]c1cc[c]([SbH2])cc1", "(4-stibanylphenyl)arsane"),                  #:39268
    ("C[Si](C)(C)CCC1OCCO1", "[2-(1,3-dioxolan-2-yl)ethyl]tri(methyl)silane"),  #:35326
    # a one-carbon prefix cites its descriptor without a locant "Sphere I.",:45002)
    ("CO[C@]([SiH3])([GeH3])SC", "[(R)-germyl(methoxy)(methylsulfanyl)methyl]silane"),  #:45031
    # the same rule on the class
    ("ClC[SiH3]", "(chloromethyl)silane"),
    ("ClCC[SiH3]", "(2-chloroethyl)silane"),
    ("[AsH2]c1ccc(Cl)cc1", "(4-chlorophenyl)arsane"),
    ("C[Si](C)(Cl)CCl", "chloro(chloromethyl)di(methyl)silane"),
    ("C[SiH](C)c1ccccc1[SiH3]", "dimethyl(2-silylphenyl)silane"),
    ("C[Si](C)(C)c1ccc(OC)cc1", "(4-methoxyphenyl)tri(methyl)silane"),
    ("CC[Si](CC)(CC)c1ccsc1", "triethyl(thiophen-3-yl)silane"),
    ("Pc1ccc([AsH2])cc1", "(4-arsanylphenyl)phosphane"),
    # (:27802,:27808): the organyl of a selenide / telluride link is '(R)selanyl' / '(R)tellanyl'
    ("C[Si](C)(C)CC[Se]C", "trimethyl[2-(methylselanyl)ethyl]silane"),
    ("C[Si](C)(C)CC[Te]C", "trimethyl[2-(methyltellanyl)ethyl]silane"),
    ("C[Si](C)(C)CCSC", "trimethyl[2-(methylsulfanyl)ethyl]silane"),
]
HETERO_NOT_PIN_ROWS = [
    ("[SiH3]CC[SiH2]Cl", "1-(chlorosilyl)-2-silylethane"),
    ("CC(C)([SiH3])CCl", "1-chloro-2-methyl-2-silylpropane"),
    ("Cl[Si](Cl)(Cl)CI", "iodo(trichlorosilyl)methane"),
    ("CO[Si](C)(C)c1cccc[c]1[Ge]([CH3])([CH3])[CH3]",
     "1-(methoxydimethylsilyl)-2-(trimethylgermyl)benzene"),
    ("[AsH2]c1cc[c]([SbH2])cc1", "1-arsanyl-4-stibanylbenzene"),
    ("[SiH3]CC[SiH2]Cl", "[2-(chlorosilyl)ethyl]silane"),                      #:6247
    ("CO[C@]([SiH3])([GeH3])SC", "[(1R)-germyl(methoxy)(methylsulfanyl)methyl]silane"),  #:45031
    # "Compounds with groups of higher seniority" (:37465): 'trimethyl[(4,4,5,5-tetramethyl-1,3,2-
    # dioxaborolan-2-yl)methyl]stannane (PIN) (Sn is senior to B)' (:37479). The best-effort tier builds that name,
    # labelled pin_unverified (the prefix of the B and O ring has no producer at the default tier, which declines);
    # the ring-parent name is no PIN: B ranks below Sn and Si:18917, '... Pb > B > Al > Ga > In > Tl > O'),
    # so a ring of B and O leaves the hub the senior atom (the label of ``senior_hub_parent_expected``).
    ("CC1(C)OB(C[Sn](C)(C)C)OC1(C)C", "4,4,5,5-tetramethyl-2-[(trimethylstannyl)methyl]-1,3,2-dioxaborolane"),
    ("C[Si](C)(C)c1ccc(cc1)B1OC(C)(C)C(C)(C)O1",
     "4,4,5,5-tetramethyl-2-[4-(trimethylsilyl)phenyl]-1,3,2-dioxaborolane"),
    ("C[Si](C)(C)c1ccc(cc1)B1OCCCO1", "2-[4-(trimethylsilyl)phenyl]-1,3,2-dioxaborinane"),
    # (:23348): four or more heterounits in an unbranched chain with a carbon -> the 'a' name is the PIN
    ("COCCOCC[SiH2]CSC", "[2-(2-methoxyethoxy)ethyl][(methylsulfanyl)methyl]silane"),
    # (a): 'bis(4,5-dihydrothiophen-2-yl)di(methyl)germane (PIN)' (:38232); the
    # shared multiplier gives 'di' for a hydro-prefixed prefix, so the PIN tier declines.
    ("C[Ge](C)(C1=CCCS1)C1=CCCS1", "di(4,5-dihydrothiophen-2-yl)di(methyl)germane"),
    # (:18917): Si is senior to Se and Te, so the silane is the parent, not the ethane
    ("C[Si](C)(C)CC[Se]C", "1-(methylselanyl)-2-(trimethylsilyl)ethane"),
    ("C[Si](C)(C)CC[Te]C", "1-(methyltellanyl)-2-(trimethylsilyl)ethane"),
]
# Molecules whose PIN is named on the hub and that the engine does not build on it (an organyl with a
# halogenated alkyne it cannot name as a prefix, two hubs the multiplicative namer does not take, two hubs
# that differ): the carbon-parent name is correct and is not the PIN, so it is labelled below it, with the rule.
HETERO_CARBON_PARENT_ROWS = [
    ("C[Si](C)(C)CC#CCCl", "1-chloro-4-(trimethylsilyl)but-2-yne"),
    ("C[Si](C)(C)C#C[Si](C)(C)C", "1,2-bis(trimethylsilyl)ethyne"),
    ("C[Si](C)(C)C=C[Si](C)(C)C", "1,2-bis(trimethylsilyl)ethene"),
    ("C[Si](C)(C)c1ccc(cc1)[Si](C)(C)Cl", "1-(chlorodimethylsilyl)-4-(trimethylsilyl)benzene"),
]
HETERO_CONTROL_ROWS = [
    ("CO[Si](OC)(OC)CCCS", "3-(trimethoxysilyl)propane-1-thiol"),               #:28164
    ("OCC[SiH3]", "2-silylethan-1-ol"),
    ("C[Si](C)(C)C", "tetramethylsilane"),
    ("C[Si](C)(C)Cl", "chlorotri(methyl)silane"),
    ("C[Si](C)(C)O", "trimethylsilanol"),                                       #:27234
    ("[SiH3]CC[SiH3]", "(ethane-1,2-diyl)bis(silane)"),
    ("ClCCP", "(2-chloroethyl)phosphane"),
    # "Skeletal replacement ('a') nomenclature in acyclic chains" (:23346), (:23348): "Skeletal
    # replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to generate
    # preferred IUPAC names for acyclic structures when four or more heterounits are present in a unbranched chain
    # containing at least one carbon atom"; (:8026) '7,10-dioxa-2-thia-4-silaundecane (PIN)'
    ("COCCOCC[SiH2]CSC", "7,10-dioxa-2-thia-4-silaundecane"),
    # a principal characteristic group decides the parent first,:18875): the silyl chain is a prefix
    ("OC(=O)c1ccc(CC[SiH3])cc1", "4-(2-silylethyl)benzoic acid"),
    # the multiplicative namer owns two alike hubs; '(1,4-phenylene)bis(dimethylsilane) (PIN)':38260)
    ("C[Si](C)(C)c1ccc(cc1)[Si](C)(C)C", "(1,4-phenylene)bis(trimethylsilane)"),
    # N outranks Si: the ring is the parent, and a principal group decides before the atoms do
    ("C[Si](C)(C)C1CCNCC1", "4-(trimethylsilyl)piperidine"),
    ("OC(=O)c1ccc(CC[Se]C)cc1", "4-[2-(methylselanyl)ethyl]benzoic acid"),
]


@pytest.mark.parametrize("smiles,pin", HETERO_ROWS + HETERO_CONTROL_ROWS)
def test_hetero_parent_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", HETERO_NOT_PIN_ROWS)
def test_hetero_parent_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,carbon_parent", HETERO_CARBON_PARENT_ROWS)
def test_senior_hub_carbon_parent_is_labelled_below_the_pin(smiles, carbon_parent):
    default = name_default(smiles)
    assert default.get("tier") != "pin_verified" and not default.get("is_pin"), default
    breadth = name_breadth(smiles)
    assert breadth.get("tier") != "pin_verified" and not breadth.get("is_pin"), breadth
    if breadth.get("name") == carbon_parent:
        assert any(f.get("rule") == "P-44.1.2" for f in breadth.get("spelling_failures") or []), breadth


@pytest.mark.parametrize("smiles,hub_name", [
    ("CC1(C)OB(C[Sn](C)(C)C)OC1(C)C", "trimethyl[(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)methyl]stannane"),   #:37479
    ("C[Si](C)(C)c1ccc(cc1)B1OC(C)(C)C(C)(C)O1", "trimethyl[4-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)phenyl]silane"),
])
def test_a_ring_of_boron_leaves_the_hub_the_senior_atom(smiles, hub_name):
    default = name_default(smiles)
    assert default["tier"] != "pin_verified" and not default["is_pin"], default
    breadth = name_breadth(smiles)
    assert breadth["name"] == hub_name and not breadth["is_pin"], breadth     # the hub name, below the PIN label
    assert name_is_rt_exact(hub_name, smiles)


def test_senior_hub_parent_expected_class():
    from rdkit import Chem

    from orthonym.rules.mononuclear_hydrides import senior_hub_parent_expected
    expected = ["C[Si](C)(C)CC[Se]C", "C[Si](C)(C)C#C[Si](C)(C)C", "[SiH3]CC[SiH2]Cl",
                "C[Si](C)(C)c1ccc(cc1)[Si](C)(C)Cl", "C[Si](C)(C)C1CCOCC1",
                "C[Si](C)(C)c1ccc(cc1)B1OC(C)(C)C(C)(C)O1", "CC1(C)OB(C[Sn](C)(C)C)OC1(C)C"]
    other = [
        "OC(=O)c1ccc(CC[SiH3])cc1",      # a principal group decides first
        "C[Si](C)(C)C1CCNCC1",           # N is senior to Si
        "[SiH3][SiH3]",                  # a catenated hydride is another parent
        "COCCOCC[SiH2]CSC",              # skeletal replacement,
        "C[Si](C)(C)O",                  # silanol: the OH is the principal group
        "C[Si](C)(C)c1ccc(cc1)B(O)O",    # a boronic acid is the principal group
        "C[Si](C)(C)B(C)C",              # a hub bonded to B
        "CCOCC",                         # no hub
        "C1CC[SiH2]CC1",                 # a ring hub is a ring parent
    ]
    for smi in expected:
        assert senior_hub_parent_expected(Chem.MolFromSmiles(smi)), smi
    for smi in other:
        assert not senior_hub_parent_expected(Chem.MolFromSmiles(smi)), smi


def test_carbon_attached_silyl_chain_prefix():
    """'2-silylethyl', not the capped fragment's 'ethylsilyl',:15762)."""
    from rdkit import Chem

    from orthonym.assembly.substituent_naming import name_substituent_fragment
    mol = Chem.MolFromSmiles("[SiH3]CC[SiH2]Cl")
    assert name_substituent_fragment(mol, [0, 1, 2], 2, [3]) == "2-silylethyl"
    assert name_substituent_fragment(mol, [1, 2, 3, 4], 1, [0]) == "2-(chlorosilyl)ethyl"


# (C) CHALCOGEN_PARENT ---------------------------------------------------------------------
CHALCOGEN_ROWS = [
    ("CSOSc1ccccc1", "methyl(phenyl)dithioxane"),                              #:39417
    ("c1ccccc1SOSc1ccccc1", "diphenyldithioxane"),
    ("CSOSc1ccc(Cl)cc1", "(4-chlorophenyl)(methyl)dithioxane"),
    ("C1CCCCC1SOSC", "cyclohexyl(methyl)dithioxane"),
    ("ClCSOSC", "(chloromethyl)(methyl)dithioxane"),
    ("C[Se]O[Se]c1ccccc1", "methyl(phenyl)diselenoxane"),
]
CHALCOGEN_NOT_PIN_ROWS = [
    ("CSOSc1ccccc1", "([(methylsulfanyl)oxy]sulfanyl)benzene"),
]
CHALCOGEN_CONTROL_ROWS = [
    ("CSOSC", "dimethyldithioxane"),                                           #:39415
    ("CSOS", "methyldithioxane"),                                              #:39413
]


@pytest.mark.parametrize("smiles,pin", CHALCOGEN_ROWS + CHALCOGEN_CONTROL_ROWS)
def test_chalcogen_parent_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", CHALCOGEN_NOT_PIN_ROWS)
def test_chalcogen_parent_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


# (D) PHOSPHONIC_PARENT --------------------------------------------------------------------
PHOSPHONIC_ROWS = [
    ("O=P(O)(O)CCCC[As](=O)(O)O", "(4-arsonobutyl)phosphonic acid"),           #:36542
    ("OCCP(=O)(O)O", "(2-hydroxyethyl)phosphonic acid"),                      # a dev split
    ("CC(O)P(=O)(O)O", "(1-hydroxyethyl)phosphonic acid"),
    ("O[C@@H](C)P(=O)(O)O", "[(1R)-1-hydroxyethyl]phosphonic acid"),
    ("OCC[As](=O)(O)O", "(2-hydroxyethyl)arsonic acid"),
    ("OCCP(=O)(O)CC", "ethyl(2-hydroxyethyl)phosphinic acid"),
    ("NC(=O)CCP(=O)(O)O", "(3-amino-3-oxopropyl)phosphonic acid"),
    ("OC1CCC(P(=O)(O)O)CC1", "(4-hydroxycyclohexyl)phosphonic acid"),
    ("O=P(O)(O)CC[As](=O)(O)O", "(2-arsonoethyl)phosphonic acid"),
]
PHOSPHONIC_STEREO_ROWS = [
    ("C[C@@H](Cl)P(=O)(O)O", "[(1R)-1-chloroethyl]phosphonic acid"),
    ("C[C@@H](CC)P(=O)(O)O", "[(2S)-butan-2-yl]phosphonic acid"),
    ("C/C=C/P(=O)(O)O", "[(1E)-prop-1-en-1-yl]phosphonic acid"),
    ("C[C@@H](CC)[As](=O)(O)O", "[(2S)-butan-2-yl]arsonic acid"),
    ("CC[C@H](C)P(=O)(O)CC", "[(2S)-butan-2-yl](ethyl)phosphinic acid"),
    ("C[C@@H](Cl)P(O)O", "[(1R)-1-chloroethyl]phosphonous acid"),
]
PHOSPHONIC_NOT_PIN_ROWS = [
    ("C[C@@H](Cl)P(=O)(O)O", "(1R)-[1-chloroethyl]phosphonic acid"),
    ("C[C@@H](CC)P(=O)(O)O", "(2S)-[butan-2-yl]phosphonic acid"),
    ("O=P(O)(O)CCCC[As](=O)(O)O", "4-arsonobutane-1-phosphonic acid"),
    ("OCCP(=O)(O)O", "2-hydroxyethane-1-phosphonic acid"),
    ("O[C@@H](C)P(=O)(O)O", "(1R)-[1-hydroxyethyl]phosphonic acid"),
]
PHOSPHONIC_CONTROL_ROWS = [
    ("CCP(=O)(O)O", "ethylphosphonic acid"),                                  #:35461
    ("ClCCP(=O)(O)O", "(2-chloroethyl)phosphonic acid"),
    ("O=C(O)CP(=O)(O)O", "phosphonoacetic acid"),                            #:36538
    ("OC(=O)CCP(=O)(O)O", "3-phosphonopropanoic acid"),
    ("O=P(O)(O)CCP(=O)(O)O", "(ethane-1,2-diyl)bis(phosphonic acid)"),
    ("Oc1ccc(P(=O)(O)O)cc1", "(4-hydroxyphenyl)phosphonic acid"),
]


@pytest.mark.parametrize("smiles,pin",
                         PHOSPHONIC_ROWS + PHOSPHONIC_STEREO_ROWS + PHOSPHONIC_CONTROL_ROWS)
def test_phosphonic_parent_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", PHOSPHONIC_NOT_PIN_ROWS)
def test_phosphonic_parent_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


# (E) DIAZENE_PARENT -----------------------------------------------------------------------
DIAZENE_ROWS = [
    ("N=NC(=S)NN", "diazenecarbothiohydrazide"),                               #:38902
    ("O=CN=NC=O", "diazenedicarbaldehyde"),                                    #:38736
    ("N#CN=Nc1ccccc1", "phenyldiazenecarbonitrile"),                           #:38738
    ("N=NC(=[Se])NN", "diazenecarboselenohydrazide"),
    ("N=NC(N)=O", "diazenecarboxamide"),
    ("N=NC(=S)N", "diazenecarbothioamide"),
    ("NC(=O)N=NC(N)=O", "diazenedicarboxamide"),
    ("N=NC#N", "diazenecarbonitrile"),
]
DIAZENE_NOT_PIN_ROWS = [
    ("N=NC(=S)NN", "1-diazenylmethanethiohydrazide"),
    ("N=NC#N", "diazenylmethanenitrile"),
]
DIAZENE_CONTROL_ROWS = [
    ("N=NC(=O)NN", "diazenecarbohydrazide"),                                   #:38900
    ("N=NC(=O)N=N", "bis(diazenyl)methanone"),                                 #:38997
    ("NNC(N)=O", "hydrazinecarboxamide"),                                      #:32675
    ("CN=NC", "dimethyldiazene"),                                              #:38768
]


@pytest.mark.parametrize("smiles,pin", DIAZENE_ROWS + DIAZENE_CONTROL_ROWS)
def test_diazene_parent_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", DIAZENE_NOT_PIN_ROWS)
def test_diazene_parent_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
