"""Spelling checks of locants and numbering (``validation/spelling/checks_locants.py``).

* (the Blue Book) low locants in the order (a) heteroatoms, (b) indicated hydrogen,
  (c) principal characteristic groups and free valences, (d) added hydrogen, (e) hydro/ene/yne,
  then double bonds, (f) detachable prefixes together, (g) the prefix cited first; decided for
  parents whose numbering symmetries the name gives (chains, monocycles, naphthalene,
  anthracene). An element locant ranks below a numeral,:3195), but the PIN
  'N4,2-dimethylpentane-2,4-diamine' (:26371) needs the other ranking: the check fails a name only
  when both rankings agree.
* (:2869) all locants of a unit or none ('2-chloroethan-1-ol' is the PIN, not
  '2-chloroethanol'); a multiplied simple prefix with another number of locants.
* (:2877),.2 (:2891),.3 (:2939),.5 (:3007),.6 (:3031) omitted locants.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


def test_the_rules_are_registered():
    assert {"P-14.4", "P-14.3.3", "P-14.3.4"} <= set(registered_rules())


@pytest.mark.parametrize("smiles,name,rule", [
    # roadmap N8 (b): str-path names labelled pin_verified that are not the PIN
    ("CCc1ccc(OC2CCCCC2C#N)cc1", "6-(4-ethylphenoxy)cyclohexane-1-carbonitrile", "P-14.4"),
    ("OC(COC1CCCc2ccccc21)CN1CCN(c2ccccc2)CC1",
     "1-(1-phenylpiperazin-4-yl)-3-[(1,2,3,4-tetrahydronaphthalen-1-yl)oxy]propan-2-ol", "P-14.4"),
    ("C1(C#N)C(n2nc(C)c(Cl)c2C)CC(CCC)CC1",
     "6-(4-chloro-3,5-dimethyl-1H-pyrazol-1-yl)-4-propylcyclohexane-1-carbonitrile", "P-14.4"),
    ("C#CC[CH]CC=C", "hept-6-en-1-yn-4-yl", "P-14.4"),        # PIN prefix 'hept-1-en-6-yn-4-yl' (:17256)
    # the book's rejected spellings
    ("Cc1ccc([N+](=O)[O-])c2ccccc12", "4-methyl-1-nitronaphthalene", "P-14.4"),        #:3318
    ("CCC(CC)CC(C)C", "3-ethyl-5-methylhexane", "P-14.4"),                              #:4867
    ("CCC(C)CC(CC)CC", "5-ethyl-3-methylheptane", "P-14.4"),                            #:4871
    ("CC=CC(O)CC(C)C", "2-methylhept-5-en-4-ol", "P-14.4"),                             #:4903
    ("C#CC(O)C(=C)C", "4-methylpent-4-en-1-yn-3-ol", "P-14.4"),                         #:3299
    ("C1CCCSCCCSCCC1", "1,9-dithiacyclododecane", "P-14.4"),                            #:8798
    ("CC1CCC(C)C1(C)C", "1,2,2,3-tetramethylcyclopentane", "P-14.4"),                   #:25510
    ("Nc1nc(NC2CC2)nc(-n2ccnc2)n1", "N4-cyclopropyl-6-(1H-imidazol-1-yl)-1,3,5-triazine-2,4-diamine",
     "P-14.4"),                                                    # lower under both rankings
    ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropane-1,3-diamide", "P-14.3.4.1"),            #:2889
    ("O=C(O)C(F)F", "2,2-difluoroacetic acid", "P-14.3.4.6"),                           #:3037
    ("O=C(O)CC(=O)O", "2-carboxyacetic acid", "P-14.3.4.6"),                            #:4973
    ("CCO", "ethan-1-ol", "P-14.3.4.2"),
    ("OC1CCCCC1", "cyclohexan-1-ol", "P-14.3.4.2"),
    ("FC(F)(F)c1ccccc1Cl", "1-chloro-2-(1,1,1-trifluoromethyl)benzene", "P-14.3.4.5"),   #:3007
    ("O=C1CCC(C2COC(=O)C2)CC1", "4-(2-oxooxolan-4-yl)cyclohexanone", "P-14.3.3"),     #:29650
    ("OCCCl", "2-chloroethanol", "P-14.3.3"),                                           #:2869
    ("CC1CCC(C(=O)O)CC1", "4-methylcyclohexanecarboxylic acid", "P-14.3.3"),
    ("CC(C)(Cl)Cl", "2-dichloropropane", "P-14.3.3"),
    ("C1CSC(C(F)(F)C(F)(F)C(F)(F)F)(N2CCOCC2)S1", "4-(4-heptafluoropropyl-1,3-dithiolan-4-yl)morpholine",
     "P-14.3.3"),                                                  # pubchem10k
    # (b) (:2891,:2907); (:40382) "'ethanide' not 'ethyl anion' for CH3-CH2"
    ("[CH2-]C", "ethan-1-ide", "P-14.3.4.2"),
])
def test_a_spelling_that_breaks_the_rule_fails(smiles, name, rule):
    assert rule in _rules(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    ("Cc1ccc([N+](=O)[O-])c2ccccc12", "1-methyl-4-nitronaphthalene"),                  #:3318
    ("CCC(CC)CC(C)C", "4-ethyl-2-methylhexane"),                                        #:4867
    ("CCC(C)CC(CC)CC", "3-ethyl-5-methylheptane"),                                      #:4871
    ("CC=CC(O)CC(C)C", "6-methylhept-2-en-4-ol"),                                       #:4903
    ("C#CC(O)C(=C)C", "2-methylpent-1-en-4-yn-3-ol"),                                   #:3299
    ("C1CCCSCCCSCCC1", "1,5-dithiacyclododecane"),                                      #:8798
    ("CC1CCC(C)C1(C)C", "1,1,2,5-tetramethylcyclopentane"),                             #:25510
    ("CCCCC(CC(C)CC)C(CCC)CC(CC)CC", "5-butyl-8-ethyl-3-methyl-6-propyldecane"),        #:22164
    ("CNC(C)CC(C)(C)N", "N4,2-dimethylpentane-2,4-diamine"),                            #:26371
    ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropanediamide"),                                #:2889
    ("O=C(O)C(F)F", "difluoroacetic acid"),                                             #:3037
    ("CCO", "ethanol"),                                                                 #:2907
    ("SC1CCCCC1", "cyclohexanethiol"),                                                  #:2917
    ("CCl", "chloromethane"),                                                           #:2897
    ("C=C", "ethene"),                                                                  #:2925
    ("C1CCC2CCCCC2C1", "decahydronaphthalene"),                                         #:3013
    ("O=C(O)C(F)(F)C(F)(F)C(F)(F)F", "heptafluorobutanoic acid"),                      #:3017
    ("OCC(F)(F)C(F)(F)F", "2,2,3,3,3-pentafluoropropan-1-ol"),                          #:3019
    ("FC(F)(F)C(F)(F)c1ccccc1Cl", "1-chloro-2-(pentafluoroethyl)benzene"),              #:3023
    ("O=C(O)c1cnccn1", "pyrazinecarboxylic acid"),                                      #:2949
    ("O=CC1CCC(C(=O)O)CC1", "4-formylcyclohexane-1-carboxylic acid"),                   #:35009
    ("Nc1ccc(S(=O)(=O)O)cc1", "4-aminobenzene-1-sulfonic acid"),                        #:31174
    ("NCCO", "2-aminoethan-1-ol"),                                                      #:28156
    ("CCCCC1(C(C)(C)C)CCC(O)CC1", "4-butyl-4-tert-butylcyclohexan-1-ol"),               #:3465
    ("CC(=O)O", "acetic acid"),                                                         # program Q7
    ("NNC(N)=O", "hydrazinecarboxamide"),                                               # program Q7
    ("[CH2-]C", "ethanide"),                                                            #:40382
    ("[CH3-]", "methanide"),                                                            #:40876
    # two kinds of suffix in one unit: ranked by rules the numbering model does not hold
    ("C[N+]1(CC[N+]CC1)C", "4,4-dimethylpiperazin-4-ium-1-ylium"),                      #:42205
    ("C1=CC=CC=2[C]C3=CC=CC=C3[CH]C12", "anthracen-9(10H)-yl-10-ylidene"),              #:24711
    # a stereodescriptor on an acetic acid unit: the book prints the locants (:46729) and omits
    # them (:45731)
    ("Cl[C@@H](C(=O)O)C1=CC=C(C=C1)[C@@](C)(CC)O",
     "(2R)-2-chloro-2-{4-[(2R)-2-hydroxybutan-2-yl]phenyl}acetic acid"),                #:46729
])
def test_book_pins_pass(smiles, name):
    assert _rules(smiles, name) == []


@pytest.mark.parametrize("name", ["N4-ethyl-2-methylpentane-2,4-diamine", "N2-ethyl-4-methylpentane-2,4-diamine"])
def test_an_element_locant_against_a_numeral_abstains(name):
    # the two numberings differ only in how 'N4' ranks against '2':3195 against the
    # PIN:26371): neither spelling is failed
    assert "P-14.4" not in _rules("CCNC(C)CC(C)(C)N", name)


def test_a_prefix_with_an_inherent_multiplying_syllable_is_not_read_as_multiplied():
    # 'trioxidanyl' is -OOOH, not three 'oxidanyl': '1-trioxidanylpropan-1-one (PIN)' (:39389)
    assert _rules("CCC(=O)OOO", "1-trioxidanylpropan-1-one") == []


@pytest.mark.parametrize("smiles,name", [
    # book PINs as the conformance report holds them: text extraction split a superscript off
    # its element locant with a space ('N1,N'4-dimethyl...':34059, 'N3-ethyl-N1,N3-dimethyl...'
    #:26369), which would read '4-dimethyl' and '3-dimethyl' as one locant for two prefixes
    ("CNNC(=O)c1ccc(C(=O)N(C)N)c2ccccc12", "N1,N' 4-dimethylnaphthalene-1,4-dicarbohydrazide"),
    ("CCNC(CCCN)(CCNC)NC", "N3-ethyl-N1,N^ 3-dimethylhexane-1,3,3,6-tetramine"),
])
def test_a_locant_set_split_by_an_extraction_space_is_not_read(smiles, name):
    assert "P-14.3.3" not in _rules(smiles, name)


@pytest.mark.parametrize("name,kind", [
    ("3-[(2H3)methyl]cyclohexanol", "isotope"),          # isotopically substituted,:43718
    ("3-([2H3]methyl)cyclohexanol", "isotope_br"),       # specifically labelled,:44361
])
def test_a_name_with_an_isotope_descriptor_is_not_read_by_p14_3_3(name, kind):
    """ abstains on a name that holds an isotope descriptor: the locants of isotopically
    modified names follow (:44176-:44186), which this check does not model. The descriptor
    sits in the substituent here, so the outer unit alone would be read (the suffix locant '1' is
    missing); the abstention is a recall bound, not a PIN claim."""
    from orthonym.validation.spelling.lexer import enclosures, normalise
    assert kind in {e.kind for e in enclosures(normalise(name))}
    assert "P-14.3.3" not in _rules("OC1CCCC(C([2H])([2H])[2H])C1", name)
