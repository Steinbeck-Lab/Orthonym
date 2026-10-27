"""PIN class program, the rest (lane PINC): the Blue Book PINs the engine builds now, each read back by OPSIN 2.9.0 and
named at both tiers.

Every expected name is the Blue Book's own where the book prints it (rule, heading and line below, checked in
the Blue Book), else the name its rule gives, read back to the input's full InChIKey by an independent OPSIN call
(``tests.support.rt_assert.name_is_rt_exact``, inside ``assert_pin_at_both_tiers``).

* Glyceraldehyde: "Double and triple bonds" (the Blue Book), '(2R)-2,3-dihydroxypropanal (PIN)'
  (:45194); the retained name stands in Table 10.2 of PARENT MONOSACCHARIDES (:52737), "These retained names are
  customarily used when the acyclic aldose or ketose has a carbon chain consisting of four, five, or six carbon atoms"
  ,:52733), and INTRODUCTION (:50943): "Preferred IUPAC names (PINs) are not identified for the
  compounds in this Chapter". '(2R)-2-(2H)hydroxy-3-hydroxy(1-2H)propanal (PIN)',:44219).
* Alkyl prefix with ylidene prefixes: General methodology for naming substituent groups (:15800), method (1):
  "The atom with the free valence terminates a chain and always has the locant '1', which is omitted from the name"
  (:15813); '(5-methyl-2,3-dimethylidenehexyl)cyclohexane (PIN)' (:25723, Cyclic hydrocarbons).
* P-stereogenic esters: Phosphates, phosphonates, and related compounds (:46008), "As the stereodescriptors 'R'
  and 'S' describe the entire structure, either a salt or an ester, the full name is placed in parentheses to denote the
  global configuration", 'methyl (S)-[methyl(phenyl)phosphinate] (PIN)' (:46022).
* Identical heteroaromatic parents: (:6231), '2-chloro-4-[(5-chloropyridin-2-yl)oxy]pyridine (PIN) [not
  5-chloro-2-[(2-chloropyridin-4-yl)oxy]pyridine; the locant set for the substituents '2,4' in the PIN is lower than
  '2,5'; see ]' (:6267); (:21604) and (:21698).
* Isotopically modified substituent: (:22212), '2-bromo-1-{[2-(81Br)bromopentyl]oxy}pentane (PIN) [not
  2-(81Br)bromo-1-[(2-bromopentyl)oxy]pentane]'.
* Anhydrides: (:32451), 'acetic 5-(acetylsulfanyl)-5-oxopentanoic anhydride (PIN)' (:32455), the senior
  anhydride (O > S > Se > Te) is the basis and the other linkage is named substitutively; Anhydrides of
  mononuclear noncarbon oxoacids (:35982), '(CH3)2B-O-CO-CH3 acetic dimethylborinic anhydride (PIN)' (:35990).
* Adducts: (:3759), "the individual components are cited in the order of seniority of classes (see ",
   (:18158: nitrogen compounds, class 21, before oxygen compounds, 35, before carbon compounds, 40) and
   (:19449), '1H-pyrrole (PIN) > 2H-1-benzopyran (PIN)'.
* Multiparent fusion name: Multiparent systems with one interparent component (:13536), 'furo[3,2-b:4,5-b']
  difuran (PIN, a multiparent name) [not difuro[3,2-b:2',3'-d]furan]' (:13592).
* Adduct classes (fix a performance pass): a component with no principal characteristic group stands in the class of
  of its senior skeletal atom: 21 nitrogen compounds "hydrazines" (:18197), 26 silicon "silanes" (:18202), 35 oxygen
  heterocycles (:18211), 40 "Carbon compounds: rings, chains" (:18216), 41 ethers (:18217); within a class the rings
  by and then (:21033) "The senior ring, ring system or principal chain has the greater number of
  multiple bonds": 'benzene—cyclohexane (1/1)'. '1,1'-oxydibenzene' (an ether, class 41) follows 'furan' (35), but OPSIN
  2.9.0 cannot read 'furan—1,1'-oxydibenzene (1/1)' (opsin_parse_failed), so that adduct has no verified PIN.
* Parent amide chain (fix a performance pass): (:21033) decides before (:21604) and (:21698): of two
  amide chains of equal length the one with the multiple bond is the parent, '(2E)-3-{4-[3-(methylamino)-3-oxopropyl]
  phenyl}prop-2-enamide'.
* Cations at a hydro-azine nitrogen (fix a performance pass): (:45978), '(2S)-2-(4-hydroxyphenyl)-2-phenyl-1,2,3,4-
  tetrahydroisophosphinolin-2-ium bromide (PIN)' (:45994): '1,1-dimethyl-1,2,3,4-tetrahydroquinolin-1-ium'.
* Multiplier of an anion word (fix a performance pass): General methodology (:7029), (a) simple components take 'di', 'tri',
  (c) "any component which is substituted automatically requires use of the multiplicative forms 'bis', 'tris'" (:7035);
   (e) (:7104) "functionalized parent hydrides with locants"; '(2R)-3-hydroxypropane-1,2-diyl di[(3E,5E)-hepta-3,5-
  dienoate] (PIN)' (:47993). '2-ethylhexanoate' is substituted: 'bis(...)' as 'propane-1,3-diyl bis(chloroacetate) (PIN)' (:31825).
* Salts: Salts (:31563), 'germanium tetraacetate (PIN)' (:31581), (:28178), '(C6H5-O-)4 Pb4+ lead
  tetraphenoxide (PIN)' (:28196): the cation by its element name, the anion multiplied, no oxidation number; the salts of
  a carbon-free anion keep the project's name with the oxidation number (the contributor guide, the maintainers' decision).
"""
import pytest

from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    # glyceraldehyde,:45194
    ("O=C[C@H](O)CO", "(2R)-2,3-dihydroxypropanal"),
    ("O=C[C@@H](O)CO", "(2S)-2,3-dihydroxypropanal"),
    ("O=CC(O)CO", "2,3-dihydroxypropanal"),
    ("[2H]C(=O)C(O)CO", "2,3-dihydroxy(1-2H)propanal"),          # not '(1-2H)glyceraldehyde';:44219
    # (:15813), the PIN of:25723
    ("C=C(CC(C)C)C(=C)CC1CCCCC1", "(5-methyl-2,3-dimethylidenehexyl)cyclohexane"),
    ("C=C(CC)CC1CCCCC1", "(2-methylidenebutyl)cyclohexane"),
    ("C=C(CC(C)C)C(C)C1CCCCC1", "(5-methyl-3-methylidenehexan-2-yl)cyclohexane"),   # method (2): 'propan-2-yl'
    #:46022
    ("CO[P@](C)(=O)c1ccccc1", "methyl (S)-[methyl(phenyl)phosphinate]"),
    ("CO[P@@](C)(=O)c1ccccc1", "methyl (R)-[methyl(phenyl)phosphinate]"),
    ("C[P@](=O)(OCC)OC", "ethyl methyl (R)-(methylphosphonate)"),
    ("COP(C)(=O)c1ccccc1", "methyl methyl(phenyl)phosphinate"),   # no stereocentre: no descriptor
    ("C[C@H](CC)O[P@](C)(=O)c1ccccc1", "(2R)-butan-2-yl (S)-[methyl(phenyl)phosphinate]"),
    ("C[C@@H](O)CO[P@](C)(=O)c1ccccc1", "(2R)-2-hydroxypropyl (S)-[methyl(phenyl)phosphinate]"),
    #:6267
    ("Clc1ccc(Oc2ccnc(Cl)c2)nc1", "2-chloro-4-[(5-chloropyridin-2-yl)oxy]pyridine"),
    ("Clc1ccnc(Oc2ccc(Cl)cn2)c1", "4-chloro-2-[(5-chloropyridin-2-yl)oxy]pyridine"),
    ("Brc1ccc(Sc2ccnc(Cl)c2)nc1", "4-[(5-bromopyridin-2-yl)sulfanyl]-2-chloropyridine"),
    ("Cc1ccnc(Oc2ccncc2)c1", "4-methyl-2-[(pyridin-4-yl)oxy]pyridine"),     #: more prefixes decide first
    ("Clc1ccsc1Oc1ccc(Cl)s1", "3-chloro-2-[(5-chlorothiophen-2-yl)oxy]thiophene"),
    #:22212
    ("CCCC(Br)COCC([81Br])CCC", "2-bromo-1-{[2-(81Br)bromopentyl]oxy}pentane"),
    ("CCCC([81Br])COCC(Br)CCC", "2-bromo-1-{[2-(81Br)bromopentyl]oxy}pentane"),
    #:32455,:35990
    ("CC(=O)OC(=O)CCCC(=O)SC(C)=O", "acetic 5-(acetylsulfanyl)-5-oxopentanoic anhydride"),
    ("CC(=O)SC(=O)CCCC(=O)OC(C)=O", "acetic 5-(acetylsulfanyl)-5-oxopentanoic anhydride"),
    ("CB(C)OC(C)=O", "acetic dimethylborinic anhydride"),
    ("CCB(C)OC(C)=O", "acetic ethyl(methyl)borinic anhydride"),
    ("O=C(OB(c1ccccc1)c1ccccc1)c1ccccc1", "benzoic diphenylborinic anhydride"),
    #:3769,:18197,:18216,:21033
    ("NNc1ccccc1.c1ccoc1", "phenylhydrazine—furan (1/1)"),
    ("c1ccccc1.C1CCCCC1", "benzene—cyclohexane (1/1)"),
    ("C1CCCCC1.c1ccccc1", "benzene—cyclohexane (1/1)"),
    ("c1ccccc1.C1=CCCCC1", "benzene—cyclohexene (1/1)"),
    ("c1ccccc1.C1CCOC1", "oxolane—benzene (1/1)"),
    #:21033,:21604
    ("NC(=O)/C=C/c1ccc(CCC(=O)NC)cc1", "(2E)-3-{4-[3-(methylamino)-3-oxopropyl]phenyl}prop-2-enamide"),
    #:45978,:45994
    ("C[N+]1(C)CCCc2ccccc21", "1,1-dimethyl-1,2,3,4-tetrahydroquinolin-1-ium"),
    #:3769,:19449
    ("C1=Cc2ccccc2OC1.c1cc[nH]c1", "1H-pyrrole—2H-1-benzopyran (1/1)"),
    ("c1ccccc1.c1ccncc1", "pyridine—benzene (1/1)"),
    ("c1cc[nH]c1.c1ccoc1", "1H-pyrrole—furan (1/1)"),
    ("CCO.c1ccncc1", "ethanol—pyridine (1/1)"),                           #:4661, the book's own
    ("c1ccc2c(c1)Cc1ccccc1-2.Cc1cccc2ccccc12", "9H-fluorene—1-methylnaphthalene (1/1)"),   #:3779
    #:13592
    ("c1cc2oc3ccoc3c2o1", "furo[3,2-b:4,5-b']difuran"),
    ("Clc1cc2oc3ccoc3c2o1", "2-chlorofuro[3,2-b:4,5-b']difuran"),
    #:28196,:31581
    ("[O-]c1ccccc1.[O-]c1ccccc1.[O-]c1ccccc1.[O-]c1ccccc1.[Pb+4]", "lead tetraphenoxide"),
    ("CCCCCCCCCCCCCCCCCC(=O)[O-].CCCCCCCCCCCCCCCCCC(=O)[O-].[Co+2]", "cobalt dioctadecanoate"),
    ("CC(=O)[O-].CC(=O)[O-].[Cu+2]", "copper diacetate"),
    #:7029, (e):7104,:47993: 'di' and the marks before an unsubstituted word with locants
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)[O-].CCCCCCCC/C=C\\CCCCCCCC(=O)[O-].[Zn+2]", "zinc di[(9Z)-octadec-9-enoate]"),
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)[O-].CCCCCCCC/C=C\\CCCCCCCC(=O)[O-].[Cr+2]", "chromium di[(9Z)-octadec-9-enoate]"),
    ("C=CC(=O)[O-].C=CC(=O)[O-].[Cu+2]", "copper di(prop-2-enoate)"),
    ("CCC=CC(=O)[O-].CCC=CC(=O)[O-].[Ni+2]", "nickel di(pent-2-enoate)"),
    ("CCCCC(CC)C(=O)[O-].CCCCC(CC)C(=O)[O-].[Co+2]", "cobalt bis(2-ethylhexanoate)"),   # substituted::7035; 'bis(chloroacetate)':31825
    ("CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Fe+3]", "iron triacetate"),
    ("CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Ge+4]", "germanium tetraacetate"),   #:31581
    # the classes that keep their name
    ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "zinc diacetate"),
    ("CC(=O)[O-].CC(=O)[O-].[Ca+2]", "calcium diacetate"),                #:31571
    ("[Cl-].[Cl-].[Cl-].[Fe+3]", "iron(III) chloride"),                   # carbon-free: the project's convention
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# A salt whose cation word the book spells without its charge cannot be read back as one: the name with the oxidation
# number stays, below the PIN (the PIN tier declines it, the wider tiers keep it read back).
@pytest.mark.parametrize("smiles", ["CC(=O)[O-].[Cu+]"])
def test_a_salt_that_needs_its_oxidation_number_is_no_pin(smiles):
    assert_declined_at_default(smiles)
