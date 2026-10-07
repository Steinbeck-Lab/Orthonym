"""Spelling check: indicated hydrogen (``validation/spelling/checks_hydrogen.py``).

 (the Blue Book) "In general nomenclature, indicated hydrogen may be omitted... and
1H-pyrrole can be called just 'pyrrole'. However, in a preferred IUPAC name a locant and the symbol
'H' must be cited." (:8318-:8320): after the maximum number of noncumulative double
bonds, a ring atom joined by single bonds only and carrying hydrogen is indicated hydrogen.
 (:24639) "in preferred IUPAC names indicated hydrogen must always be cited when present
in the corresponding structure". The count comes from the RDKit structure; the name's front
``nH-`` groups must cover it. The round trip cannot see an omission: OPSIN reads
'pyrido[1,2-b]pyridazin-6-one' back to the input.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


def test_the_rule_is_registered():
    assert "P-14.7.1" in registered_rules()


@pytest.mark.parametrize("smiles,name", [
    # the program's Q1 row; the PIN is '6H-pyrido[1,2-b]pyridazin-6-one',:24768)
    ("O=c1ccn2ncccc2c1", "pyrido[1,2-b]pyridazin-6-one"),
    # the book's rejected spellings: "6H,6'H-2,2'-bipyran (PIN) (not 2,2'-bi-6H-pyran)":15599,
    # "1H,1'H-1,1'-biindene (PIN) (not 1,1'-bi-1H-indene)":15605 (one group per assembly)
    ("C1=CCOC(C2=CC=CCO2)=C1", "2,2'-bi-6H-pyran"),
    ("C1=CC(C2C=Cc3ccccc32)c2ccccc21", "1,1'-bi-1H-indene"),
    # engine names of the eval sets labelled pin_verified on main
    ("COc1ccc(Cl)cc1-n1nnc(C#N)c1CC(C)C",
     "1-(5-chloro-2-methoxyphenyl)-5-(2-methylpropyl)-1,2,3-triazole-4-carbonitrile"),
    ("COc1cc(=O)oc2c1CO[C@@](C)(OC)[C@@]2(C)Br",
     "(7R,8S)-8-bromo-4,7-dimethoxy-7,8-dimethyl-5H-pyrano[4,3-b]pyran-2-one"),
    ("C1=CC2=C(C=CC(=O)C=C2)N=C1", "cyclohepta[b]pyridin-7-one"),
])
def test_an_omitted_indicated_hydrogen_fails(smiles, name):
    assert "P-14.7.1" in _rules(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    ("O=c1ccn2ncccc2c1", "6H-pyrido[1,2-b]pyridazin-6-one"),
    ("c1cc[pH]c1", "1H-phosphole"),                                            #:16930
    ("[AsH]1cccc1", "1H-arsole"),                                              # program Q7
    ("Cc1cc2cn(CC(=O)O)cc2cc1C", "(5,6-dimethyl-2H-isoindol-2-yl)acetic acid"),  #:6728
    ("S=C(n1ccnc1)n1ccnc1", "di(1H-imidazol-1-yl)methanethione"),              #:29544
    ("O=C(O)Cn1ccc2ccccc21", "(1H-indol-1-yl)acetic acid"),                    #:2039
    ("C1=CCOC(C2=CC=CCO2)=C1", "6H,6'H-2,2'-bipyran"),                         #:15599
    ("C1=CC(C2C=Cc3ccccc32)c2ccccc21", "1H,1'H-1,1'-biindene"),                #:15605
    ("C1CCC2CCCCC2C1", "decahydronaphthalene"),                                #:3013 (saturated)
    ("c1ccc2ncccc2c1", "quinoline"),                                           #:1996 (needs none)
])
def test_book_pins_pass(smiles, name):
    assert "P-14.7.1" not in _rules(smiles, name)


def test_a_charged_ring_abstains():
    # 'benzo[7]annulenylium (PIN)' (:43552) is formed from the neutral parent by an operation the
    # count does not model; the check reads no charged ring
    assert "P-14.7.1" not in _rules("c1ccc2ccc[cH+]c-2cc1", "benzo[7]annulenylium")


def test_a_structure_without_a_kekule_form_is_not_read_as_saturated():
    """``kekule_ring_double`` counts the endocyclic double bonds of the Kekule form; a structure
    that has none (an aromatic five-ring of carbons only, unsanitized) raises instead of counting
    its aromatic bonds as single, which would read the mancude ring as saturated."""
    from orthonym.validation.spelling.indicated_hydrogen import kekule_ring_double
    assert kekule_ring_double(Chem.MolFromSmiles("C1=CC=CC1"), range(5)) == 2
    assert kekule_ring_double(Chem.MolFromSmiles("c1ccccc1"), range(6)) == 3
    broken = Chem.MolFromSmiles("c1cccc1", sanitize=False)
    broken.UpdatePropertyCache(strict=False)
    with pytest.raises(Chem.KekulizeException):
        kekule_ring_double(broken, range(5))
