"""Spelling checks of enclosing marks, multiplying prefixes, hyphens and prefix order
(``validation/spelling/checks_marks.py``).

* (the Blue Book) "Enclosing marks must not be omitted from preferred IUPAC names";
   (:7446) nesting order {[({})]}, -.5 (:7465-:7509).
* (:7232) parentheses around compound and complex prefixes; (:27667) the
  simple alkoxy prefixes are methoxy, ethoxy, propoxy, butoxy, phenoxy, tert-butoxy;
  (:7368) parentheses after 'bis', 'tris'; (:7272) on a mononuclear parent every
  substituent but the first is enclosed.
* (a) (:7104) 'bis', 'tris' for substituted prefixes; (f) (:7104) 'di', 'tri'
  for simple components containing brackets ('3,5-di([1,1'-biphenyl]-3-yl)pyridine (PIN)':23913).
* (b) (:6944) a hyphen after a closing mark followed by a locant.
* /.2/.4 (:3448,:3477,:3517) alphanumerical order; (:16872) hydro prefixes
  after the alphabetized prefixes.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


def test_the_rules_are_registered():
    assert {"P-16.5", "P-16.5.1.1", "P-16.3.5", "P-16.3.4", "P-16.2.4.1", "P-14.5"} <= set(registered_rules())


@pytest.mark.parametrize("smiles,name,rule", [
    ("CC(C)OCCO", "2-[(propan-2-yl)oxy)ethan-1-ol", "P-16.5"),                    # marks do not pair
    ("CC(C)OCC(=O)O", "{[propan-2-yl]oxy}acetic acid", "P-16.5.4"),
    ("CCC(=O)[Se]OSC", "[1-[(methylsulfanyl)oxy]selanyl]propan-1-one", "P-16.5.4"),  # engine, BB:39497
    ("COC(=O)c1ccc(B(O)O)[nH]1", "(5-(methoxycarbonyl)-1H-pyrrol-2-yl)boronic acid", "P-16.5.4"),
    ("ClCC(CCl)(CCl)CCl", "2,2-bischloromethyl-1,3-dichloropropane", "P-16.5.1.10"),
    ("CCCCCOC(=O)c1ccccc1C(=O)O", "2-(pentyloxycarbonyl)benzoic acid", "P-16.5.1.1"),  # a dev split
    ("COCCNS(=O)(=O)c1cccc(C(=O)NC2(C(=O)O)CC2)c1",
     "1-[3-(2-methoxyethylsulfamoyl)benzamido]cyclopropane-1-carboxylic acid", "P-16.5.1.1"),
    ("CSC[Se][Se]C", "methyldiselanyl(methylsulfanyl)methane", "P-16.5.1.1"),     # engine, BB:18278
    ("ClC[SiH3]", "chloromethylsilane", "P-16.5.1.1"),
    ("CCCCCCCCCC(C)(C)S(=O)CC(CS(=O)C(C)(C)CCCCCCCCC)O",
     "1,3-di(2-methylundecane-2-sulfinyl)propan-2-ol", "P-16.3.5"),                # druglike2000
    ("OCc1cc(CO)cc(CO)c1", "[3,5-di(hydroxymethyl)phenyl]methanol", "P-16.3.5"),
    ("c1ccc(-c2cccc(-c3cncc(-c4cccc(-c5ccccc5)c4)c3)c2)cc1",
     "3,5-bis([1,1'-biphenyl]-3-yl)pyridine", "P-16.3.4"),                        # engine, BB:23913
    ("O=C(Nc1cnn(CC2COc3ccccc3O2)c1)c1cc2ccccc2s1",
     "N-{1-[(2,3-dihydro-1,4-benzodioxin-2-yl)methyl]-1H-pyrazol-4-yl}1-benzothiophene-2-carboxamide",
     "P-16.2.4.1"),
    ("Clc1ccc(Br)cc1", "1-chloro-4-bromobenzene", "P-14.5"),
    ("O=C(O)CC(Br)(Br)C1CCCCC1", "3-cyclohexyl-3,3-dibromopropanoic acid", "P-14.5"),   # rejected:3459
    ("COc1c(-c2cn(C)cn2)cccc1C(C)(C)C", "4-(2-methoxy-3-tert-butylphenyl)-1-methyl-1H-imidazole", "P-14.5"),
    ("CCC(C)C(C)C1(C(CC)C(C)C)CCCC1",
     "1-(3-methylpentan-2-yl)-1-(2-methylpentan-3-yl)cyclopentane", "P-14.5.4"),  # rejected:3535
    ("c1ccc2cc(Nc3ccc4c(c3)CCCC4)ccc2c1", "5,6,7,8-tetrahydrodi(2-naphthyl)amine", "P-31.2.1"),  #:26296
    # engine names labelled pin_verified on main (test_reclaim_p2_a1_iodane_suffix.py rows): a
    # substituted prefix after 'di' (a)); 'oxanylmethyl' cited before 'diphosphanyl'
    ("C=IC(OC1=CN=C(N=C1)OC2=CCNC(N=C2)N)I=C",
     "5-({5-[di(methylidene-λ3-iodanyl)methoxy]pyrimidin-2-yl}oxy)-2,7-dihydro-1H-1,3-diazepin-2-amine",
     "P-16.3.5"),
    ("C=IC1=C(C2(CCNCC2)CN1C(=O)NCC3CCOCC3)PP",
     "N-[(oxan-4-yl)methyl]-4-diphosphanyl-3-(methylidene-λ3-iodanyl)-2,8-diazaspiro[4.5]dec-3-ene-2-carboxamide",
     "P-14.5"),
])
def test_a_spelling_that_breaks_the_rule_fails(smiles, name, rule):
    assert rule in _rules(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    ("O=C(O)CC(Br)(Br)C1CCCCC1", "3,3-dibromo-3-cyclohexylpropanoic acid"),                #:3459
    ("CCCCCCC(CC(CC)CCCC)C(F)C(F)CC", "7-(1,2-difluorobutyl)-5-ethyltridecane"),           #:3483
    ("CCC(C)c1cccc(C(C)(C)C)c1", "1-(butan-2-yl)-3-tert-butylbenzene"),                    #:3507
    ("CCCCC1(C(C)(C)C)CCC(O)CC1", "4-butyl-4-tert-butylcyclohexan-1-ol"),                  #:3465
    ("CCC(C)Cc1ccc(NCCC(C)C)cc1", "4-(2-methylbutyl)-N-(3-methylbutyl)aniline"),           #:3523
    ("CCCC(C)(C)C1(C(C)(CC)CC)CCCC1", "1-(2-methylpentan-2-yl)-1-(3-methylpentan-3-yl)cyclopentane"),  #:3533
    ("CCC(C)C(C)C1(C(CC)C(C)C)CCCC1", "1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclopentane"),  #:3535
    ("N#Cc1ccc(-c2ccc(OCCCCCOC(=O)C(Cc3ccc(C(=O)O)cc3)c3ccc(C(=O)O)cc3)cc2)cc1",
     "4,4'-[3-({5-[(4'-cyano[1,1'-biphenyl]-4-yl)oxy]pentyl}oxy)-3-oxopropane-1,2-diyl]dibenzoic acid"),  #:7455
    ("N[14CH2]C1(O)CCCC1", "1-(amino[14C]methyl)cyclopentan-1-ol"),                       #:7506
    ("c1ccc(-c2cccc(-c3cncc(-c4cccc(-c5ccccc5)c4)c3)c2)cc1", "3,5-di([1,1'-biphenyl]-3-yl)pyridine"),  #:23913
    ("[CH3][Ge]([CH3])([C]1=CCCS1)[C]1=CCCS1", "bis(4,5-dihydrothiophen-2-yl)di(methyl)germane"),     #:38232
    ("CSC[Se][Se]C", "(methyldiselanyl)(methylsulfanyl)methane"),                          #:18278
    ("C[SiH2]Cl", "chloro(methyl)silane"),                                                 #:7280
    ("CCC(=O)[Se]OSC", "1-{[(methylsulfanyl)oxy]selanyl}propan-1-one"),                    #:39497
    ("c1ccc(OC2CCCCC2)cc1", "(cyclohexyloxy)benzene"),                                     #:27768
    ("O=C(O)OC(=O)c1ccccc1C(=O)O", "2-[(carboxyoxy)carbonyl]benzoic acid"),                #:31129
    ("S=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanethione"),                          #:29544
    ("O=C(O)CCOC(=O)c1cccnc1", "3-[(pyridine-3-carbonyl)oxy]propanoic acid"),              #:31723
    ("Cl[As](Cl)Cl", "trichloroarsane"),                                                    # program Q7
    ("OC(=O)Cn1ccnc1", "(1H-imidazol-1-yl)acetic acid"),                                    # program Q7
    # radicals and ions,: the rest after the prefixes is the parent radical or ion
    ("CO[B]", "methoxyboranylidene"),                                                      #:36271
    ("CO[As]=O", "methoxy(oxo)-λ5-arsanylidene"),                                          #:36283
    ("ClCN([O])CCl", "bis(chloromethyl)aminoxyl"),                                         #:40703
    ("C1(=CC=CC=C1)S[S+]", "phenyldisulfanylium"),                                         #:41668
    ("[O-][N+]#CCC1=CC=C(C=C1)C([CH2])[CH2]",
     "2-{4-[2-(oxidoazaniumylidyne)ethyl]phenyl}propane-1,3-diyl"),                        #:43386
    ("CCCCC[O]", "pentyloxyl"),                          # engine; 'butoxyl (PIN)':40713,
    ("[B]C", "methylboranylidene"),                      # engine; 'benzylsilylidene (PIN)':40490
    # an isotope descriptor with an element locant:43718, '(N-2H1)acetamide (PIN)'
    #:43828) is no prefix
    ("[2H]NCc1ccccc1", "1-phenyl(N-2H1)methanamine"),
    # a Hantzsch-Widman ring of iodine ('ioda' + 'ocine', is the parent
    ("CC1=CC(=CI=C(CC1)C)C(F)(F)I=C",
     "7-[difluoro(methylidene-λ3-iodanyl)methyl]-2,5-dimethyl-3,4-dihydro-1λ3-iodocine"),
])
def test_book_pins_pass(smiles, name):
    assert _rules(smiles, name) == []


def test_a_run_of_terminal_prefixes_on_a_mononuclear_parent_is_not_read():
    # (:7272) wants the second and further substituents enclosed, but the book also
    # prints '(R)-bromo(chloro)fluoromethane (PIN)' (:44645); the gate protects
    # 'bromodichlorofluoromethane' (gold_pins.json STEREO-06): the check abstains on such a run
    assert _rules("C(Br)(Cl)(Cl)F", "bromodichlorofluoromethane") == []


def test_prefixes_differing_only_in_stereodescriptors_are_not_ordered():
    # (:3446) "the principles of alphanumerical order do not include... stereochemical
    # descriptors"; the book orders such prefixes by (:45829)
    smiles = ("C[C@H](Cl)[C@H](Cl)N([C@H](Cl)[C@@H](C)Cl)[C@](N)(O[C@@H](Cl)[C@@H](C)Cl)"
              "[C@@H](Cl)[C@H](C)Cl")
    name = ("(s)-{[(1R,2R)-1,2-dichloropropyl][(1S,2R)-1,2-dichloropropyl]amino}"
            "{[(1R,2S)-1,2-dichloropropyl][(1S,2S)-1,2-dichloropropyl]amino}methanol")
    assert not {"P-14.5", "P-14.5.4"} & set(_rules(smiles, name))
