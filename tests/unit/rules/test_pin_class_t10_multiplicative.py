"""PIN class program, Task 10: multiplicative names (MULT, MULT_REVERSE, DI_PAREN, ESTER_PARENT).

Identical parent units linked by a multiplying group
----------------------------------------------------
 "Preferred IUPAC multiplicative names" (the Blue Book): "Multiplicative
nomenclature is preferred to substitutive nomenclature for generating preferred IUPAC names
to express multiple occurrences of identical parent structures, other than alkanes when"
(:23180) "(1) the linking bonds (single or multiple) between the central substituent group of
the multiplicative group and all subsequent structural units are identical and" (:23182)
"(2) the multiplicative groups, other than the central multiplicative group, are
symmetrically substituted; and" (:23183) "(3) the locants of all substituent groups on the
identical parent structures, including suffix groups, are identical." (:23184).
'2,2'-[oxybis(ethane-2,1-diyloxy)]diacetic acid (PIN, a multiplicative name)' (:23210).

 "A maximum number of identical parent structures must be expressed by the
multiplicative name." (:23244); (:23242) and (2) (:23829): "linear
phanes consist of four or more rings or ring systems, two of which must be terminal, and
together with acyclic atoms or chains must consist of at least seven nodes";
(:23348): skeletal replacement "when four or more heterounits are present".

 (:5238): a concatenated multiplicative group is formed "by first citing the
central multiplicative substituent group, followed by a multiplicative prefix... and then, in
order, and in the direction toward the identical parent structures, the names of successive
di- or polyvalent substituent groups". (:5297): "lowest locants to the atoms
that are at the end of the component nearest to the multiplied parent structure, except where
the component has a fixed numbering. The locants attached to the multiplied parent structure
are cited last." (:6295): "Unsymmetrical central multiplicative substituent groups
are allowed if they are formed from a multivalent substituent group to which subsequent groups
are attached by identical bonds". (:6176,:6181): substituted identical units are
"treated as a compound or complex group, enclosed in parentheses, square brackets, or braces
... and designated by the appropriate numerical prefix 'bis', 'tris', 'tetrakis'".
 (:18917): "N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S >
Se > Te > C"; '(thiophene-2,5-diyl)bis(trimethylstannane) (PIN) (Sn preferred to S, see
' (:38264).

At the base the multiplicative producer (``rules/multiplicative.py:name_multiplicative``) was
entered for these molecules and returned None (narrow shapes only), so the substitutive name
shipped as the PIN ('3-(1-chloroethyl)-5-{[5-(1-chloroethyl)pyridin-3-yl]oxy}pyridine',...).
The general detector (``rules/multiplicative_general.py``) finds the units on the graph,
checks conditions (1)-(3) there and names the linker as a simple or concatenated group.

Suffix locants of a multiplied unit
-----------------------------------
 (:7085) (e): parentheses enclose "functionalized parent hydrides with locants":
'di(cyclohexane-1-carboxylic acid)'; '2,2'-sulfanediyldi(cyclopentane-1-thiol) (PIN)'
(:27591), '3,4'-disulfanediyldi(benzene-1-thiol)' (:27593).

The bridge as the parent
-------------------------------------
 (:23309): "Seniority order of classes (see is used when a choice has to be
made between a parent structure and a component of a multiplicative group."
'di(1H-imidazol-1-yl)methanethione (PIN)',:29544); a carbonyl linked to a ring
heteroatom is a pseudoketone (:1878); '4,4'-carbonyldibenzoic acid (PIN)' (:29473) and
'1,1'-carbonothioyldi(pyridin-2(1H)-one) (PIN)' (:29581) keep the multiplicative name.
The ring nitrogen may be saturated (batch 2 fix a performance pass): (b) (:28263) "an
acyclic carbonyl group is bonded... to a heteroatom of a ring or ring system. When the
heteroatom of the ring is a nitrogen atom the compound has been called an unexpressed or
'hidden amide'"; '1-(piperidin-1-yl)ethan-1-one (PIN)' (:28271); (:29370) names it
"substitutively by using the suffix 'one'";:29482 "Acyl prefixes... recommended only for
general nomenclature". So 'di(pyrrolidin-1-yl)methanone', not '1,1'-carbonyldipyrrolidine'.
A prefix with hydro prefixes takes 'bis' (batch 2 fix a performance pass): the Blue Book's one multiplied
hydro-prefixed prefix is 'bis(4,5-dihydrothiophen-2-yl)di(methyl)germane (PIN)',
:38232), so 'bis(2,3-dihydro-1H-indol-1-yl)methanone'.
When the ring prefix cannot be named at the PIN tier ('1H-indol-1-yl', 'aziridin-1-yl'), the
multiplicative name is not labelled a PIN either: the default tier declines, best-effort keeps
an RT-exact name.

Esters
------
 (:31794): esters of a multiplied acid component: "(1) all organyl
components representing the 'hydroxylic' components are cited in front of the name of the
multiplied acid component"; 'dimethyl 3,3'-oxydibenzoate (PIN)' (:31801), 'ethyl methyl
3,3'-oxydibenzoate (PIN)' (:31803). (:31809): "Esters that do not qualify
for multiplicative names... are named as monoesters"; 'methyl 2-chloro-5-[3-(ethoxy-
carbonyl)phenoxy]benzoate (PIN) (not ethyl 3-[4-chloro-3-(methoxycarbonyl)phenoxy]benzoate;
the parent structure of the PIN has more substituents)' (:31813).

Not multiplicative
------------------
'1-[(2R)-butan-2-yl]-4-({4-[(2S)-butan-2-yl]phenyl}sulfanyl)benzene (PIN) (a substitutive
name; a multiplicative name is not allowed)',:22587); '1,2-dimethoxyethane (PIN)'
(:27754, alkane units); '1-bromo-3-[(3-chlorophenyl)methyl]benzene (PIN)' (:6279, condition
(3) fails).
"""
import pytest

from tests.support.pin_tiers import (
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("O=C(O)COCCOCCOCC(=O)O", "2,2'-[oxybis(ethane-2,1-diyloxy)]diacetic acid"),      #:23210
    ("OCc1cnnc(CO)c1Oc1c(CO)cnnc1CO", "[oxydi(pyridazine-4,3,5-triyl)]tetramethanol"),  #:6112
    ("OCc1nncc(Oc2cnnc(CO)c2CO)c1CO", "[oxydi(pyridazine-5,3,4-triyl)]tetramethanol"),  #:6118
    ("CC(Cl)c1cncc(Oc2cncc(C(C)Cl)c2)c1", "3,3'-oxybis[5-(1-chloroethyl)pyridine]"),    #:6223
    ("Brc1ccc(C=CCc2ccc(Br)cc2)cc1", "1,1'-(prop-1-ene-1,3-diyl)bis(4-bromobenzene)"),  #:6301
    ("O=C(O)CCCCCCCCCc1cccnc1CCCCCCCCCC(=O)O", "10,10'-(pyridine-2,3-diyl)di(decanoic acid)"),  #:6761
    ("CC(Cl)c1ccc(Oc2ccc(C(C)Cl)c(C(C)Cl)c2)cc1C(C)Cl",
     "1,1'-oxybis[3,4-bis(1-chloroethyl)benzene]"),                                       #:22595
    ("CSSSCCSSSC", "1,1'-(ethane-1,2-diyl)bis(3-methyltrisulfane)"),                       #:23385
    ("c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1", "1,1'-[1,4-phenylenebis(oxy)]dibenzene"),          #:23925
    ("c1cc(SCCSc2cocc2SCCSc2ccoc2)co1",
     "3,3'-[furan-3,4-diylbis(sulfanediylethane-2,1-diylsulfanediyl)]difuran"),           #:23963
    ("Cc1ccc(CCCc2ccc(CCCc3ccc(C)cc3)c(CCCc3ccc(C)cc3)c2)cc1",
     "1,1',1''-[benzene-1,2,4-triyltri(propane-3,1-diyl)]tris(4-methylbenzene)"),         #:25729
    ("N#Cc1ccc(Nc2ccc(C#N)cc2)cc1", "4,4'-azanediyldibenzonitrile"),                      #:26419
    ("SC1CCCC1SC1CCCC1S", "2,2'-sulfanediyldi(cyclopentane-1-thiol)"),                    #:27591
    ("S=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanethione"),                         #:29544
    ("COC(=O)c1cccc(Oc2cccc(C(=O)OC)c2)c1", "dimethyl 3,3'-oxydibenzoate"),               #:31801
    ("CCOC(=O)c1cccc(Oc2cccc(C(=O)OC)c2)c1", "ethyl methyl 3,3'-oxydibenzoate"),          #:31803
    ("CCOC(=O)c1cccc(Oc2ccc(Cl)c(C(=O)OC)c2)c1",
     "methyl 2-chloro-5-[3-(ethoxycarbonyl)phenoxy]benzoate"),                            #:31813
    ("[CH3][Sn]([CH3])([CH3])[c]1cc[c]([Sn]([CH3])([CH3])[CH3])s1",
     "(thiophene-2,5-diyl)bis(trimethylstannane)"),                                       #:38264
    ("O=C(O)C1CCC(=NN=C2CCC(C(=O)O)CC2)CC1",
     "4,4'-hydrazinediylidenedi(cyclohexane-1-carboxylic acid)"),                         #:38617
    ("CP(C)CCP(C)C", "(ethane-1,2-diyl)bis(dimethylphosphane)"),                          #:39180
    ("c1ccc(OSOc2ccccc2)cc1", "1,1'-[sulfanediylbis(oxy)]dibenzene"),                     #:39491
    # dev2000 member; descriptors cited as in '(2S,2'S)-2,2'-{oxybis[...]}dipropanoic acid
    # (PIN)' (:50799)
    ("C[C@@H](N[C@H](C)C(=O)O)C(=O)O", "(2R,2'R)-2,2'-azanediyldipropanoic acid"),
]

# Class members (same producers, same rules).
CLASS_ROWS = [
    ("N#Cc1ccccc1Cc1ccccc1C#N", "2,2'-methylenedibenzonitrile"),                       #:2648
    ("O=C(O)C1CCCCC1COCCOCC1CCCCC1C(=O)O",
     "2,2'-[ethane-1,2-diylbis(oxymethylene)]di(cyclohexane-1-carboxylic acid)"),      #:5836
    ("Oc1ccc(CCCCCCCCCCCCCCOCCCCCCCCCCCCCCc2ccc(O)cc2)cc1",
     "4,4'-[oxydi(tetradecane-14,1-diyl)]diphenol"),                                    #:5842
    ("[SiH3]CCCCCCCCCCCCCCOCCCCCCCCCCCCCC[SiH3]",
     "[oxydi(tetradecane-14,1-diyl)]bis(silane)"),                                      #:6170
    ("F[Si](F)(F)[Si](F)(F)C[Si](F)(F)[Si](F)(F)F",
     "1,1'-methylenebis(pentafluorodisilane)"),                                         #:6190
    ("ClCc1ccc(Cc2ccc(CCl)c(Br)c2)cc1Br",
     "1,1'-methylenebis[3-bromo-4-(chloromethyl)benzene]"),                             #:6202
    ("O=C(O)c1ccccc1CCOCCOCCOCCc1ccccc1C(=O)O",
     "2,2'-[oxybis(ethane-2,1-diyloxyethane-2,1-diyl)]dibenzoic acid"),                 #:23228
    ("c1cc(OCCOc2cocc2OCCOc2ccoc2)co1",
     "3,3'-[furan-3,4-diylbis(oxyethane-2,1-diyloxy)]difuran"),                         #:23270
    ("OCCCc1ccc(CCCO)cc1", "3,3'-(1,4-phenylene)di(propan-1-ol)"),                       #:25171
    ("C[SiH](C)c1ccc([SiH](C)C)cc1", "(1,4-phenylene)bis(dimethylsilane)"),               #:38260
    ("OCc1ccc(CO)cc1", "(1,4-phenylene)dimethanol"),
    ("C[Si](C)(C)c1ccc([Si](C)(C)C)cc1", "(1,4-phenylene)bis(trimethylsilane)"),
    ("N#CCCNCCC#N", "3,3'-azanediyldipropanenitrile"),
    ("C[C@H](N[C@@H](C)C(=O)O)C(=O)O", "(2S,2'S)-2,2'-azanediyldipropanoic acid"),
    ("CCOC(=O)COCC(=O)OCC", "diethyl 2,2'-oxydiacetate"),
    ("COC(=O)c1ccc(Oc2ccc(C(=O)OC)cc2)cc1", "dimethyl 4,4'-oxydibenzoate"),
    ("O=[N+]([O-])c1ccc(Sc2ccc([N+](=O)[O-])cc2)cc1", "1,1'-sulfanediylbis(4-nitrobenzene)"),
    ("COc1ccc(Oc2ccc(OC)cc2)cc1", "1,1'-oxybis(4-methoxybenzene)"),
    ("CC(C)(C)c1ccc(Oc2ccc(C(C)(C)C)cc2)cc1", "1,1'-oxybis(4-tert-butylbenzene)"),
    ("OCCOCCOCCOCCO", "2,2'-[oxybis(ethane-2,1-diyloxy)]di(ethan-1-ol)"),                # 3 heterounits
    ("c1ccc(CCc2ccc(CCc3ccccc3)cc2)cc1", "1,1'-[1,4-phenylenedi(ethane-2,1-diyl)]dibenzene"),
    ("c1ccc(Cc2ccc(Cc3ccccc3)cc2)cc1", "1,1'-[1,4-phenylenebis(methylene)]dibenzene"),
    ("OC1CCC(CC1)OC1CCC(O)CC1", "4,4'-oxydi(cyclohexan-1-ol)"),
    ("SC1CCC(CC1)SC1CCC(S)CC1", "4,4'-sulfanediyldi(cyclohexane-1-thiol)"),
    ("OC(=O)c1cccnc1Oc1ncccc1C(=O)O", "2,2'-oxydi(pyridine-3-carboxylic acid)"),
    ("O=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanone"),
    ("S=C(n1ccnc1C)n1ccnc1C", "bis(2-methyl-1H-imidazol-1-yl)methanethione"),
    ("S=C(n1cncn1)n1cncn1", "bis(1H-1,2,4-triazol-1-yl)methanethione"),
    ("O=C(c1ccc(O)cc1)c1ccc(O)cc1", "bis(4-hydroxyphenyl)methanone"),
    # a saturated ring nitrogen (b),:28263;,:29370)
    ("O=C(N1CCCC1)N1CCCC1", "di(pyrrolidin-1-yl)methanone"),
    ("S=C(N1CCCCC1)N1CCCCC1", "di(piperidin-1-yl)methanethione"),
    ("O=C(N1CCCCC1)N1CCCCC1", "di(piperidin-1-yl)methanone"),
    ("O=C(N1CCOCC1)N1CCOCC1", "di(morpholin-4-yl)methanone"),
    ("S=C(N1CCCC1)N1CCCC1", "di(pyrrolidin-1-yl)methanethione"),
    ("O=C(N1CCc2ccccc21)N1CCc2ccccc21", "bis(2,3-dihydro-1H-indol-1-yl)methanone"),     #:38232
    ("O=C(N1CCCCCC1)N1CCCCCC1", "bis(azepan-1-yl)methanone"),                            #:7176
    ("CCOC(=O)c1ccc(Oc2ccc(Br)c(C(=O)OC)c2)cc1",
     "methyl 2-bromo-5-[4-(ethoxycarbonyl)phenoxy]benzoate"),
]

# Already the PIN at both tiers at the base (protection).
CONTROL_ROWS = [
    ("OC(=O)CSCC(=O)O", "2,2'-sulfanediyldiacetic acid"),
    ("c1ccc(Oc2ccccc2)cc1", "1,1'-oxydibenzene"),
    ("Brc1cccc(Cc2cccc(Cl)c2)c1", "1-bromo-3-[(3-chlorophenyl)methyl]benzene"),           #:6279
    ("COCCOC", "1,2-dimethoxyethane"),                                                     #:27754
    ("O=C(O)c1ccc(cc1)C(=O)c1ccc(cc1)C(=O)O", "4,4'-carbonyldibenzoic acid"),              #:29473
    ("S=C(n1c(=O)cccc1)n1c(=O)cccc1", "1,1'-carbonothioyldi(pyridin-2(1H)-one)"),         #:29581
    ("O=C(c1ccccn1)c1ccccn1", "di(pyridin-2-yl)methanone"),
    ("Clc1ccc(Oc2ccc(Cl)cc2)cc1", "1,1'-oxybis(4-chlorobenzene)"),                         # as:6185
    ("OC(=O)CCOCCC(=O)O", "3,3'-oxydipropanoic acid"),                                     #:5850
    ("NCCOCCN", "2,2'-oxydi(ethan-1-amine)"),                                              #:6124
    ("OC(=O)c1ccc(OCCOc2ccc(C(=O)O)cc2)cc1", "4,4'-[ethane-1,2-diylbis(oxy)]dibenzoic acid"),  #:23222
    ("O=C(O)CN(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),                        #:2648
    ("c1ccc(cc1)C=Cc1ccccc1", "1,1'-(ethene-1,2-diyl)dibenzene"),                          #:16615
    ("Oc1ccc(Cc2ccc(O)cc2)cc1", "4,4'-methylenediphenol"),
]

# (smiles, token of a multiplicative name that must not appear): units that differ only in
# configuration (:22587), a chain with four heterounits (skeletal replacement,:23348), a
# linear system of four rings and seven nodes (a phane,:23829).
NOT_MULTIPLICATIVE_ROWS = [
    ("CC[C@@H](C)c1ccc(Sc2ccc([C@@H](C)CC)cc2)cc1", "sulfanediyl"),
    ("C[C@@H](N[C@@H](C)C(=O)O)C(=O)O", "azanediyl"),
    ("OCCOCCOCCOCCOCCO", "diyl"),
    ("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "phenylene"),
    # four heterounits on one chain, B and O: '2,9-dimethyl-4,7-dioxa-2,9-diboradecane (PIN)
    # [ethane-1,2-diylbis(oxymethylene)]bis(dimethylborane)',:37140)
    ("CB(C)COCCOCB(C)C", "bis(dimethylborane)"),
    ("CC(O[Si](C)(C)C)C(CO[Si](C)(C)C)O[Si](C)(C)C", "tris(trimethylsilane)"),
    # a chain of alternating heteroatoms is a parent hydride,:8013):
    # 'disiloxane (preselected name)' (:7631)
    ("[SiH3]O[SiH3]", "oxybis"),
    ("Oc1ccc([Se]O[Se]c2ccc(O)cc2)cc1", "bis(selanediyl)"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS + CLASS_ROWS + CONTROL_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# (smiles, the multiplicative name): a C=O bridge between two ring nitrogens whose methanone
# parent is the PIN (b),:28263) but whose ring prefix the PIN tier does not name;
# the multiplicative name is never labelled a PIN, and best-effort stays RT-exact.
KETONE_PARENT_UNBUILT_ROWS = [
    ("O=C(n1ccc2ccccc21)n1ccc2ccccc21", "1,1'-carbonylbis(1H-indole)"),
    ("O=C(N1CC1)N1CC1", "1,1'-carbonylbis(aziridine)"),
    ("O=C(N1CCC1)N1CCC1", "1,1'-carbonylbis(azetidine)"),
]


@pytest.mark.parametrize("smiles,non_pin", KETONE_PARENT_UNBUILT_ROWS)
def test_ketone_parent_unbuilt_multiplicative_never_pin(smiles, non_pin):
    for res in (name_default(smiles), name_breadth(smiles)):
        assert not (res.get("name") == non_pin and res.get("tier") == "pin_verified"), res
    assert name_default(smiles).get("tier") != "pin_verified"
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


@pytest.mark.parametrize("smiles,token", NOT_MULTIPLICATIVE_ROWS)
def test_not_a_multiplicative_name(smiles, token):
    for res in (name_default(smiles), name_breadth(smiles)):
        assert token not in (res.get("name") or ""), res
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


@pytest.mark.xfail(strict=True, reason="needs PIN class program Task 18 (ENCLOSING_MARKS): the "
                   "compound prefix '({4-[(2S)-butan-2-yl]phenyl}sulfanyl)' is enclosed, :22587")
def test_configuration_differing_units_substitutive_pin():
    assert_pin_at_both_tiers("CC[C@@H](C)c1ccc(Sc2ccc([C@@H](C)CC)cc2)cc1",
                             "1-[(2R)-butan-2-yl]-4-({4-[(2S)-butan-2-yl]phenyl}sulfanyl)benzene")
