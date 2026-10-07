"""Spelling checks of the parent choice (``validation/spelling/checks_parent.py``).

* (the Blue Book, Table 4.1): the suffix cites the senior class present; esters
  (class 9, "functional class names are given to noncyclic esters":18182) outrank amides,
  ketones, alcohols and amines, so a carbamate R-NH-CO-O-R' is cited as the ester; "cyclic amides
  are named as heterocycles" (:18184).
* (:18875) "The senior parent structure has the maximum number of substituents
  corresponding to the principal characteristic group (suffix)"; (:24096) "for the same
  number of characteristic groups cited as the principal characteristic group, a ring is always
  selected as the parent hydride".
* (heading:18462):18470, Table 4.4 (:18597): a class prefix beside a junior suffix of its
  class ('carbamoyl' beside 'sulfonamide', 'oxo' beside 'thione') fails at the choice of the
  principal characteristic group, before counts anything.
* (:20916) the principal chain is the longer one, length before unsaturation (:18865;
  '2-ethylideneoctanoic acid (PIN) [not 2-hexylbut-2-enoic acid...]':29938); (:21604)
  then the parent with the maximum number of prefixes ('4-methoxy-N-phenylaniline (PIN) [not
  N-(4-methoxyphenyl)aniline...]':21610).
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


def test_the_rules_are_registered():
    assert {"P-41", "P-44.1.1", "P-45.2.1"} <= set(registered_rules())


@pytest.mark.parametrize("smiles,name,rule", [
    # the book's rejected spellings
    ("O=C(O)c1ccccc1C(=O)O", "2-carboxybenzoic acid", "P-44.1.1"),                       #:4975
    ("O=C(O)c1ccc(C(=O)O)c(C(=O)O)c1", "2,4-dicarboxybenzoic acid", "P-44.1.1"),         #:4977
    ("OCC(O)c1ccccc1O", "2-(1,2-dihydroxyethyl)phenol", "P-44.1.1"),                      #:28144
    ("O=C(O)c1ccccc1O", "2-carboxyphenol", "P-41"),                                       #:4983
    ("O=CNC=O", "diformylamine", "P-41"),                                                 #:33101
    ("CC(=O)NC(=O)c1ccccc1", "acetyl(benzoyl)amine", "P-41"),                             #:33111
    ("NCCO", "ethanolamine", "P-41"),                                                     #:28156
    ("CC=C(CCCCCC)C(=O)O", "2-hexylbut-2-enoic acid", "P-44.3.2"),                        #:29938
    ("C=C(C=O)CCCC", "2-butylprop-2-enal", "P-44.3.2"),                                   #:35068
    ("COc1ccc(Nc2ccccc2)cc1", "N-(4-methoxyphenyl)aniline", "P-45.2.1"),                  #:21610
    # roadmap N8 (b): (:21604, example:21624)
    ("COC(=O)C(C)Cc1c(C)nn(-c2ccccc2Cl)c1C",
     "methyl 2-{[1-(2-chlorophenyl)-3,5-dimethyl-1H-pyrazol-4-yl]methyl}propanoate", "P-45.2.1"),
    ("N(C(=O)C(C)Cc1ccc(OC)cc1)C(C)CO",
     "N-(1-hydroxypropan-2-yl)-2-[(4-methoxyphenyl)methyl]propanamide", "P-45.2.1"),
    ("C[CH]CCC(CC(C)Cl)C(Cl)C(C)Cl", "7-chloro-5-(1,2-dichloropropyl)octan-2-yl", "P-45.2.1"),  #:22746
    # engine names of the eval sets labelled pin_verified on main
    ("CC(C)C[C@@H](C(=O)NC1CCCCC1)NC(=O)OC(C)(C)C",
     "(2S)-2-[(tert-butoxycarbonyl)amino]-N-cyclohexyl-4-methylpentanamide", "P-41"),
    ("CC(C)C(=O)c1cncc(=O)[nH]1", "2-methyl-1-(6-oxo-1,6-dihydropyrazin-2-yl)propan-1-one", "P-52.2.8"),
    ("OCCc1ccc(O)cc1", "2-(4-hydroxyphenyl)ethan-1-ol", "P-52.2.8"),
])
def test_a_parent_the_rules_do_not_choose_fails(smiles, name, rule):
    assert rule in _rules(smiles, name)


#: (structure, a name whose suffix is a junior suffix of the class of one of its prefixes, the
#: spelling with the senior suffix); Table 4.4 order: carboxamide (16,:18759) > thioamide
#: (:18762-:18763) > sulfonamide (19,:18769); 'one' > 'thione' (48,:18831-:18833); 'ol' >
#: 'selenol' (49,:18836-:18839). Each name reads back to the structure (OPSIN 2.9.0, full InChIKey).
JUNIOR_SUFFIX_ROWS = [
    # pubchem10k, labelled pin_verified on main
    ("Cc1ccc(NS(=O)(=O)c2ccc(C)c(C(N)=O)c2)c(C)c1",
     "3-carbamoyl-N-(2,4-dimethylphenyl)-4-methylbenzene-1-sulfonamide",
     "5-[(2,4-dimethylphenyl)sulfamoyl]-2-methylbenzamide"),
    ("NC(=O)c1ccc(S(N)(=O)=O)cc1", "4-carbamoylbenzene-1-sulfonamide", "4-sulfamoylbenzamide"),
    ("CC(=O)CC(C)=S", "4-oxopentane-2-thione", "4-sulfanylidenepentan-2-one"),
    ("OCCC[SeH]", "3-hydroxypropane-1-selenol", "3-selanylpropan-1-ol"),
    # the substituent branch (a ring substituent carries the senior group)
    ("NC(=O)c1ccc(CS(N)(=O)=O)cc1", "(4-carbamoylphenyl)methanesulfonamide", "4-(sulfamoylmethyl)benzamide"),
    ("NC(=O)CCC(N)=S", "3-carbamoylpropanethioamide", None),
]


@pytest.mark.parametrize("smiles,name,senior", JUNIOR_SUFFIX_ROWS)
def test_a_class_prefix_beside_a_junior_suffix_of_its_class_fails_p43(smiles, name, senior):
    rules = _rules(smiles, name)
    assert "P-43" in rules, rules
    assert not {"P-44.1.1", "P-52.2.8"} & set(rules), rules     # no group is counted before
    if senior:
        assert _rules(smiles, senior) == []


def test_a_class_prefix_beside_the_suffix_it_expresses_fails_p44_1_1():
    # the same pattern with the suffix the prefix expresses: 'carbamoyl' beside 'amide'
    assert _rules("NC(=O)c1cccc(C(N)=O)c1", "3-carbamoylbenzamide") == ["P-44.1.1"]
    assert _rules("NC(=O)c1cccc(C(N)=O)c1", "benzene-1,3-dicarboxamide") == []


@pytest.mark.parametrize("smiles,name", [
    ("O=C(O)c1ccccc1C(=O)O", "benzene-1,2-dicarboxylic acid"),                           #:4975
    ("CC(O)C(CCO)CCCCCl", "3-(4-chlorobutyl)pentane-1,4-diol"),                           #:18883
    ("O=C(O)c1ccccc1O", "2-hydroxybenzoic acid"),                                         #:4983
    ("CC=C(CCCCCC)C(=O)O", "2-ethylideneoctanoic acid"),                                  #:29938
    ("C=C(C=O)CCCC", "2-methylidenehexanal"),                                             #:35068
    ("COc1ccc(Nc2ccccc2)cc1", "4-methoxy-N-phenylaniline"),                               #:21610
    ("CCCC(CC)C(C)C", "3-ethyl-2-methylhexane"),                                          #:21649
    ("CCCC(CCC(=O)O)CC(C)C", "6-methyl-4-propylheptanoic acid"),                          #:21653
    ("NNc1ccccn1", "2-hydrazinylpyridine"),                                               #:19376
    ("CC(O)COC(=O)NCCN", "2-hydroxypropyl (2-aminoethyl)carbamate"),                      #:30766
    ("NCCO", "2-aminoethan-1-ol"),                                                        #:28156
    ("c1ccc(-c2cncc(-c3ccccc3)c2)cc1", "3,5-diphenylpyridine"),                           #:23909
    ("CCc1ccc(OC2CCCCC2C#N)cc1", "2-(4-ethylphenoxy)cyclohexane-1-carbonitrile"),
    ("COC(=O)C(C)Cc1c(C)nn(-c2ccccc2Cl)c1C",
     "methyl 3-[1-(2-chlorophenyl)-3,5-dimethyl-1H-pyrazol-4-yl]-2-methylpropanoate"),
    ("N(C(=O)C(C)Cc1ccc(OC)cc1)C(C)CO", "N-(1-hydroxypropan-2-yl)-3-(4-methoxyphenyl)-2-methylpropanamide"),
    ("C[CH]CCC(CC(C)Cl)C(Cl)C(C)Cl", "6,7-dichloro-5-(2-chloropropyl)octan-2-yl"),        #:22746
    ("OCCc1ccc(O)cc1", "4-(2-hydroxyethyl)phenol"),
    ("O=C(O)CCOC(=O)c1cccnc1", "3-[(pyridine-3-carbonyl)oxy]propanoic acid"),             #:31723
    # 'oxa' + 'ocine' is the Hantzsch-Widman parent 'oxocine', not an 'oxo' prefix
    ("O=C1C=CC=CC=CO1", "2H-oxocin-2-one"),
])
def test_book_pins_pass(smiles, name):
    assert _rules(smiles, name) == []
