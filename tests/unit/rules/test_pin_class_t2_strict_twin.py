"""PIN class program, Task 2: the strict-twin and default-tier non-PIN rows.

Each Blue Book PIN below was built at the best-effort tier only (labelled pin_unverified,
so the default tier declined it) or the default tier shipped a non-PIN spelling at
pin_verified. The strict path now builds the PIN, so both tiers ship it at pin_verified.

(A) Phosphanes. "Substitution of phosphanes, arsanes, and stibanes by
    organyl groups" (the Blue Book): "Alkyl, aryl, etc. groups and groups derived
    from parent hydrides containing O, S, Se, and Te atoms are always denoted by prefixes."
    (:39153); 'cyclohexylphosphane (PIN)' (:39165), 'ethyl(methyl)(phenyl)phosphane (PIN)'
    (:39171). "Retained prefixes that are preferred prefixes" (:16270):
    '*tert*-butyldi(methyl)phosphane (PIN)' (:16286). "Nonstandard bonding
    numbers" (:2756): 'triphenyl-λ5-phosphane (PIN)' (:2768). "The chirality
    symbols 'R/S'" (:46253): '(R)-methyl(phenyl)(propyl)phosphane (PIN)' (:46270). The
    organyl phosphane branch of rules/mononuclear_hydrides.py ran at the best-effort tier
    only; it now runs at every tier.

(D) Identical parent units. "Preferred IUPAC multiplicative names"
    (:23178): a multiplicative PIN needs "(2) the multiplicative groups, other than the
    central multiplicative group, are symmetrically substituted; and (3) the locants of
    all substituent groups on the identical parent structures, including suffix groups,
    are identical" (:23183-23184); "all substituent groups, including the principal
    characteristic groups must be identical and have the same locant" (:23186).
     (:6231): '4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile (PIN)' (:6235),
    '4-chloro-2-[2-(4-chloro-3-hydroxyphenyl)ethyl]phenol (PIN)' (:6237); units that
    differ only in configuration give a substitutive PIN (:22587). The multiplicative
    guards (rules/pin_vocabulary.py:multiplicative_candidate, assembly/substituent_naming
    .py:_ring_branch_may_need_other_parent) tested "same ring skeleton" only; they now
    compare the decorated units (rules/pin_vocabulary.py:identical_parent_units).
    The ring-on-carrier prefix '(3-cyanophenyl)methyl' of the:6235 PIN needs the
    multi-atom decoration fold of rules/ring_substituents.py:
    _compound_ring_on_chain_substituent at the PIN tier,:16304;
    'carboxy(4-carboxyphenyl)methylidene (preferred prefix)',:16330); the ring-yl
    itself stays built by the strict ring producers.

(B) A carboxamide on a carbon chain. (:32922): "For generation of IUPAC
    preferred names, method (1) is preferred for chains" (:32928): '5-(2-amino-2-
    oxoethyl)furan-2-carboxylic acid (PIN)' (:32940), not '(2) 5-(carbamoylmethyl)
    furan-2-carboxylic acid' (:32941). The strict path writes method (1) (the amide
    prefix conversion and the located chain namer for a chain carboxamide), and a
    '(carbamoyl...)methyl' spelling is labelled below the PIN (non_pin_vocabulary).

(C) Rows whose PIN needs a best-effort composition producer at the strict path
    (decorated ring-yl prefixes, the functional-group chain namer): xfail(strict=True).
    Opening those producers at the PIN tier relabels many other molecules pin_verified,
    some of them with non-PIN spellings (measured in the TRIAGE section of Task 2), so
    the opening is a program decision, not this task's.
"""
import pytest
from rdkit import Chem

from orthonym.rules.pin_vocabulary import identical_parent_units, multiplicative_candidate
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("CCC[P@@](C)c1ccccc1", "(R)-methyl(phenyl)(propyl)phosphane"),        #:46270
    ("CP(C)C(C)(C)C", "tert-butyldi(methyl)phosphane"),                    #:16286
    ("PC1CCCCC1", "cyclohexylphosphane"),                                  #:39165
    ("CCP(C)c1ccccc1", "ethyl(methyl)(phenyl)phosphane"),                  #:39171
    ("c1ccc([PH2](c2ccccc2)c2ccccc2)cc1", "triphenyl-λ5-phosphane"),       #:2768
    ("N#Cc1cccc(Cc2cc(Cl)ccc2C#N)c1", "4-chloro-2-[(3-cyanophenyl)methyl]benzonitrile"),  #:6235
    ("NC(=O)Cc1ccc(C(=O)O)o1", "5-(2-amino-2-oxoethyl)furan-2-carboxylic acid"),  #:32940
    # method (1), members of the class beyond the Blue Book row
    ("O=C(O)c1ccc(CC(N)=O)cc1", "4-(2-amino-2-oxoethyl)benzoic acid"),
    ("NC(=O)CCc1ccc(C(=O)O)cc1", "4-(3-amino-3-oxopropyl)benzoic acid"),
    ("NC(=O)C(C)c1ccc(C(=O)O)cc1", "4-(1-amino-1-oxopropan-2-yl)benzoic acid"),
]

_NEEDS_COMPOSITION = ("needs a best-effort composition producer at the strict path "
                      "(PIN class program Task 2 (C), TRIAGE: the measured relabels)")
XFAIL_PIN_ROWS = [
    ("O=C(O)c1ccc(CSO)cc1", "4-[(hydroxysulfanyl)methyl]benzoic acid"),    #:29962
    ("O=C1CCCCC1OOC1(O)CCCCC1OO",
     "2-[(2-hydroperoxy-1-hydroxycyclohexyl)peroxy]cyclohexan-1-one"),     #:28154
    ("CC1=CCC(C=CC(C)N(C)C)CC1",
     "N,N-dimethyl-4-(4-methylcyclohex-3-en-1-yl)but-3-en-2-amine"),       #:26269
    ("[CH3][Sn]1([CH3])[CH2]CC(B2C3CCCC2CCC3)[CH2]1",
     "3-(9-borabicyclo[3.3.1]nonan-9-yl)-1,1-dimethylstannolane"),         #:37486
    ("COc1cc2c(Oc3ccc4[nH]c(C)cc4c3F)ncnc2cc1OCCCN1CCCC1",
     "4-[(4-fluoro-2-methyl-1H-indol-5-yl)oxy]-6-methoxy-7-[3-(pyrrolidin-1-yl)propoxy]quinazoline"),  #:19437
]
# (smiles, the non-PIN spelling the default tier shipped at pin_verified at 39803cb6e)
NOT_PIN_ROWS = [
    ("CP(C)C(C)(C)C", "2-(dimethylphosphanyl)-2-methylpropane"),
    ("c1ccc([PH2](c2ccccc2)c2ccccc2)cc1", "(diphenyl-λ5-phosphanyl)benzene"),
    # an ylide: "'Ylides'" (:42499), "Method (1) is applicable to all 'ylides'
    # and leads to preferred IUPAC names" (:42509); the λ-convention name (:42530, method
    # (2)) is not the PIN, so the organyl branch keeps it at the best-effort tier
    ("CCP(=CC)(CC)CC", "triethyl(ethylidene)-λ5-phosphane"),
    ("NC(=O)Cc1ccc(C(=O)O)o1", "5-(carbamoylmethyl)furan-2-carboxylic acid"),   #:32941
    ("O=C(O)c1ccc(CC(N)=O)cc1", "4-(carbamoylmethyl)benzoic acid"),
]
# (smiles, PIN the base already ships at pin_verified at both tiers) -- protection
CONTROL_ROWS = [
    ("Brc1cccc(Cc2cccc(Cl)c2)c1", "1-bromo-3-[(3-chlorophenyl)methyl]benzene"),  #:6279
    ("CO[Si](OC)(OC)CCCS", "3-(trimethoxysilyl)propane-1-thiol"),              #:28164
    ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", "triphenylphosphane"),                   #:39157
    ("CP(C)C", "trimethylphosphane"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.xfail(strict=True, reason=_NEEDS_COMPOSITION)
@pytest.mark.parametrize("smiles,pin", XFAIL_PIN_ROWS)
def test_pin_needs_composition_producer(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_composition_rows_keep_an_rt_exact_best_effort_name():
    from tests.support.pin_tiers import assert_declined_at_default
    for smiles, _pin in XFAIL_PIN_ROWS:
        assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# (smiles, are the two ring systems identical parent units?)
UNIT_ROWS = [
    ("N#Cc1cccc(Cc2cc(Cl)ccc2C#N)c1", False),                          #:6235
    ("Oc1cc(CCc2cc(Cl)ccc2O)ccc1Cl", False),                           #:6237
    ("Brc1cccc(Cc2cccc(Cl)c2)c1", False),                              #:6279
    ("OC(=O)c1ccc(Oc2cccc(C(=O)O)c2)cc1", False),                      # locants differ
    ("C[C@H](CC)C1=CC=C(C=C1)SC1=CC=C(C=C1)[C@@H](C)CC", False),       #:22587, R and S
    ("S(C1=CC=C(C=C1)[C@H](C)CC)C1=CC=C(C=C1)[C@H](C)CC", True),       # R and R
    ("N#Cc1ccc(Cc2ccc(C#N)cc2)cc1", True),
    ("ClC(Cl)(c1ccccc1)c1ccccc1", True),     # '1,1'-(dichloromethylene)dibenzene',
    ("OC(=O)c1ccc(Cc2ccc(C(=O)O)cc2)cc1", True),  # 4,4'-methylenedibenzoic acid
]


def _ring_systems(mol):
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        r = set(ring)
        for s in [s for s in systems if s & r]:
            systems.remove(s)
            r |= s
        systems.append(r)
    return systems


@pytest.mark.parametrize("smiles,identical", UNIT_ROWS)
def test_identical_parent_units(smiles, identical):
    mol = Chem.MolFromSmiles(smiles)
    a, b = _ring_systems(mol)
    assert identical_parent_units(mol, a, b) is identical
    assert multiplicative_candidate(mol) is identical


# (dummy-parent SMILES, does the PIN tier open the method (1) chain namer for it?)
CHAIN_CARBOXAMIDE_ROWS = [
    ("[*]CC(N)=O", True),           # 2-amino-2-oxoethyl
    ("[*]CCC(N)=O", True),          # 3-amino-3-oxopropyl
    ("[*]C(C)C(N)=O", True),        # 1-amino-1-oxopropan-2-yl
    ("[*]CC(=O)N(C)C", True),       # 2-(dimethylamino)-2-oxoethyl
    ("[*]CCNC(C)=O", False),        # 2-acetamidoethyl: the reversed amide
    ("[*]CNC(C)=O", False),
    ("[*]C(=O)N", False),           # carbamoyl: one carbon, method (2)
]


@pytest.mark.parametrize("smiles,opens", CHAIN_CARBOXAMIDE_ROWS)
def test_chain_carboxamide_class(smiles, opens):
    from orthonym.assembly.substituent_naming import _is_chain_carboxamide
    mol = Chem.MolFromSmiles(smiles)
    dummy = next(a for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    assert _is_chain_carboxamide(mol, frag, dummy.GetNeighbors()[0].GetIdx()) is opens


# Review round (F1, F3). (:23180-23184): identical parent structures with
# identical substitution and locants take the multiplicative PIN; '4,4′-methylenediphenol
# (PIN) (1) 4-[(4-hydroxyphenyl)methyl]phenol',:26874) shows the substitutive
# alternative is not the PIN; esters follow it,:31790,:31806). A name
# built on any other route for such a skeleton is never labelled pin_verified
# (rules/pin_vocabulary.py:multiplicative_pin_expected); the best-effort tier keeps it.
MULT_SKELETON_NOT_PIN_ROWS = [
    ("COC(=O)Nc1ccc(Cc2ccc(NC(=O)OC)c(Cc3ccc(NC(=O)OC)cc3)c2)cc1",
     "methyl (4-{[4-[(methoxycarbonyl)amino]-3-({4-[(methoxycarbonyl)amino]phenyl}methyl)"
     "phenyl]methyl}phenyl)carbamate"),
]
MULT_SKELETON_PIN_ROWS = [
    ("Oc1ccc(Cc2ccc(O)cc2)cc1", "4,4'-methylenediphenol"),          #:26874
    ("c1ccc(Oc2ccccc2)cc1", "1,1'-oxydibenzene"),                    #:27776
    ("c1ccccc1Cc1ccccc1", "1,1'-methylenedibenzene"),                #:19360
    ("Brc1ccc(Oc2ccc(Br)cc2)cc1", "1,1'-oxybis(4-bromobenzene)"),    #:6185
    # built by the general multiplicative detector (PIN class program Task 10);
    # '2,2'-methylenedibenzonitrile (PIN)',:2648)
    ("N#Cc1ccc(Cc2ccc(C#N)cc2)cc1", "4,4'-methylenedibenzonitrile"),
    ("N#Cc1cccc(Cc2cccc(C#N)c2)c1", "3,3'-methylenedibenzonitrile"),
    ("N#Cc1ccc(CCc2ccc(C#N)cc2)cc1", "4,4'-(ethane-1,2-diyl)dibenzonitrile"),
]


@pytest.mark.parametrize("smiles,non_pin", MULT_SKELETON_NOT_PIN_ROWS)
def test_multiplicative_skeleton_substitutive_name_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)
    from tests.support.pin_tiers import assert_declined_at_default
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,pin", MULT_SKELETON_PIN_ROWS)
def test_multiplicative_skeleton_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,expected", [
    ("N#Cc1ccc(Cc2ccc(C#N)cc2)cc1", True),
    ("COC(=O)Nc1ccc(Cc2ccc(NC(=O)OC)c(Cc3ccc(NC(=O)OC)cc3)c2)cc1", True),
    ("Oc1ccc(Cc2ccc(O)cc2)cc1", True),
    ("N#Cc1cccc(Cc2cc(Cl)ccc2C#N)c1", False),        #:6235, locants differ
    ("Brc1cccc(Cc2cccc(Cl)c2)c1", False),            #:6279
    ("N#Cc1ccc(Cc2cccc(C#N)c2)cc1", False),          # 3,4'-isomer, locants differ
    ("OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1", False),    # ring assembly
    ("OC(c1ccc(Cl)cc1)c1ccc(Cl)cc1", False),         # principal group on the linker
    ("COc1ccc(Cc2ncc(Cc3ccc(OC)cc3)c(C)n2)cc1", False),  # senior parent pyrimidine
    ("C[C@H](CC)C1=CC=C(C=C1)SC1=CC=C(C=C1)[C@@H](C)CC", False),  #:22587
    ("c1ccc([PH2](c2ccccc2)c2ccccc2)cc1", False),   # phosphane parent,:2768
    ("C[Si](C)(c1ccccc1)c1ccccc1", False),          # silane parent
    ("N#Cc1ccc(OCc2ccc(C#N)cc2)cc1", False),        # unsymmetrical -O-CH2- linker
    ("c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", False),       # ring assembly, no principal group
    # (:18875): the linker chain holds as many principal groups as the two
    # units together, so the units are not shown to be the senior parent: no claim
    ("OC[C@@H](O)[C@@H](/C=C/c1ccc(O)cc1)c1ccc(O)cc1", False),
])
def test_multiplicative_pin_expected(smiles, expected):
    from orthonym.rules.pin_vocabulary import multiplicative_pin_expected
    assert multiplicative_pin_expected(Chem.MolFromSmiles(smiles)) is expected


# Review round (F2). "NUMBERING" (the Blue Book), criterion (j) (:3346): "When
# there is a choice for lower locants related to the presence of stereogenic centers or
# stereoisomers, the lower locant is assigned to CIP stereodescriptors Z, R, M, and r...
# that are preferred to E, S, P, and s"; '(2Z,5E)-hepta-2,5-dienedioic acid (PIN)' (:3354).
# The chain's stereodescriptors include those of a double bond from a chain atom to a
# substituent (an '-ylidene' prefix): its descriptor is cited at the chain locant.
EZ_YLIDENE_PIN_ROWS = [
    ("COc1cc(/C=C(C(=O)O)/C(=C\\c2ccc(O)c(OC)c2)C(=O)O)ccc1O",
     "(2Z,3E)-2,3-bis[(4-hydroxy-3-methoxyphenyl)methylidene]butanedioic acid"),
    ("C/C=C(C(=O)O)/C(=C\\C)C(=O)O", "(2Z,3E)-2,3-diethylidenebutanedioic acid"),
    ("CC/C=C(C(=O)O)/C(=C\\CC)C(=O)O", "(2Z,3E)-2,3-dipropylidenebutanedioic acid"),
    ("C/C=C(/C(=O)O)C(=C\\C)C(=O)O", "(2E)-2,3-diethylidenebutanedioic acid"),
]
EZ_YLIDENE_NOT_PIN_ROWS = [
    ("COc1cc(/C=C(C(=O)O)/C(=C\\c2ccc(O)c(OC)c2)C(=O)O)ccc1O",
     "(2E,3Z)-2,3-bis[(4-hydroxy-3-methoxyphenyl)methylidene]butanedioic acid"),
    ("C/C=C(C(=O)O)/C(=C\\C)C(=O)O", "(2E,3Z)-2,3-diethylidenebutanedioic acid"),
]


@pytest.mark.parametrize("smiles,pin", EZ_YLIDENE_PIN_ROWS)
def test_ez_ylidene_numbering_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", EZ_YLIDENE_NOT_PIN_ROWS)
def test_ez_ylidene_numbering_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


# The routes name by their own nomenclature; their labels are not read by the
# multiplicative rule (an oligosaccharide with two identical fucopyranosyl units).
P10_ROUTE_ROWS = [
    "C[C@@H]1O[C@@H](O[C@@H]2[C@@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3"
    "O[C@@H]3O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]3O)[C@@H](O)[C@@H](CO)O[C@H]2O)"
    "[C@@H](O)[C@H](O)[C@@H]1O",
]


@pytest.mark.parametrize("smiles", P10_ROUTE_ROWS)
def test_p10_route_label_unchanged(smiles):
    from tests.support.pin_tiers import name_default
    assert name_default(smiles).get("tier") == "pin_verified"


# Review round (F5). (:39151): "Alkyl, aryl, etc. groups and groups derived
# from parent hydrides containing O, S, Se, and Te atoms are always denoted by prefixes"
# (:39153): with no principal characteristic group the phosphane stays the parent when an
# organyl carries ether or sulfide groups. (:7272): "For mononuclear parent
# hydrides with two or more substituents the first cited substituent never has enclosing
# marks unless it includes a locant. The second and further substituents are each enclosed
# with parentheses even for simple substituents. When the simple substituent groups are
# accompanied by multiplicative prefixes such as 'di' and 'tri', the multiplicative prefixes
# are not included in the parentheses"; 'ethyldi(methyl)phosphane (PIN)' (:7290); also
# '(cyclohexanecarbonyl)di(methyl)oxidanium (PIN)' (:41480), 'bis(4,5-dihydrothiophen-2-yl)
# di(methyl)germane (PIN)' (:38232).
PHOSPHANE_ETHER_PIN_ROWS = [
    ("CP(C)CCOC", "(2-methoxyethyl)di(methyl)phosphane"),
    ("COCCP(CCOC)CCOC", "tris(2-methoxyethyl)phosphane"),
    ("CP(C)c1ccc(OC)cc1", "(4-methoxyphenyl)di(methyl)phosphane"),
    ("CP(C)CCSC", "dimethyl[2-(methylsulfanyl)ethyl]phosphane"),   # 'dimethyl' cited first
    ("CP(C)c1cccs1", "dimethyl(thiophen-2-yl)phosphane"),             #: P > S
]
PHOSPHANE_ETHER_NOT_PIN_ROWS = [
    ("CP(C)CCOC", "1-(dimethylphosphanyl)-2-methoxyethane"),
    ("CP(C)c1ccc(OC)cc1", "1-(dimethylphosphanyl)-4-methoxybenzene"),
    ("CP(C)c1cccs1", "2-(dimethylphosphanyl)thiophene"),
]
# a suffix group elsewhere keeps the carbon parent: unchanged controls
PHOSPHANE_SUFFIX_CONTROL_ROWS = [
    ("CP(C)CCO", "2-(dimethylphosphanyl)ethan-1-ol"),
    ("CP(C)CC(C)=O", "1-(dimethylphosphanyl)propan-2-one"),
]


@pytest.mark.parametrize("smiles,pin", PHOSPHANE_ETHER_PIN_ROWS)
def test_phosphane_ether_organyl_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", PHOSPHANE_ETHER_NOT_PIN_ROWS)
def test_phosphane_ether_organyl_not_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,pin", PHOSPHANE_SUFFIX_CONTROL_ROWS)
def test_phosphane_suffix_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
