"""The 'oxime' of functional class nomenclature is not a preferred IUPAC name.

 'Oximes' (the Blue Book): "For general nomenclature they are named according to the
principles of functional class nomenclature by placing the class name 'oxime' as a separate word after the
name of the aldehyde or ketone. Preferred IUPAC names are formed substitutively as N-hydroxy derivatives of
imines";:38462 "In these recommendations preferred IUPAC names for oximes are generated substitutively as
N-hydroxy derivatives of imines rather than by functional class nomenclature as in previous
recommendations" (the example:38468: 'N-hydroxypentan-2-imine (PIN)... pentan-2-one oxime').

The oxime handler builds the substitutive name for a parent whose imine name it can cite 'N-hydroxy' in
(a plain or substituted alkane or cycloalkane parent, with the configuration of the C=N, one oxime or two:
``test_oxime_substitutive_pin``) and falls back to the functional class name for the rest (a bicyclic parent,
whose imine has no PIN at the default tier; a one-carbon parent with two aryl groups, whose order of prefixes
the book prints against (:47666); an unsaturated parent). That fallback was labelled pin_verified: camphor
oxime '(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one oxime'. It is recorded as not the PIN: the default
tier declines it, the best-effort tier gives it unchanged (systematic_verified), and every name reads back
to the input's full InChIKey (OPSIN 2.9.0, independent of the engine).
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: (what, SMILES, the functional class name the engine gives it at the best-effort tier)
FUNCTIONAL_CLASS = [
    ("bicyclic ketoxime with stereocentres (camphor oxime)", "C[C@@]12CC[C@@H](C1(C)C)CC2=NO",
     "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one oxime"),
    ("one-carbon parent with two aryl groups", "O/N=C(/c1ccccc1)c1ccc(Cl)cc1",
     "(Z)-(4-chlorophenyl)(phenyl)methanone oxime"),
    ("unsaturated parent", "CC(=NO)C=C", "but-3-en-2-one oxime"),
]


@pytest.mark.parametrize("what, smiles, name", FUNCTIONAL_CLASS, ids=[c[0] for c in FUNCTIONAL_CLASS])
def test_a_functional_class_oxime_is_declined_at_the_default_tier(what, smiles, name):
    d = name_default(smiles)
    assert d.get("tier") == "abstain" and d.get("limit_code") == "NO_VERIFIED_PIN", d
    assert d.get("is_pin") is not True, d


@pytest.mark.parametrize("what, smiles, name", FUNCTIONAL_CLASS, ids=[c[0] for c in FUNCTIONAL_CLASS])
def test_a_functional_class_oxime_is_kept_below_the_pin_at_best_effort(what, smiles, name):
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier"), b.get("is_pin")) == (name, "systematic_verified", False), b
    assert_full_rt(b["name"], smiles)


@pytest.mark.parametrize("smiles, pin", [
    ("CCCC(C)=NO", "N-hydroxypentan-2-imine"),
    ("CCC=NO", "N-hydroxypropan-1-imine"),
    ("C1CCCCC1=NO", "N-hydroxycyclohexan-1-imine"),
    ("CC1CCCCC1=NO", "N-hydroxy-2-methylcyclohexan-1-imine"),
])
def test_the_substitutive_oxime_keeps_its_pin(smiles, pin):
    """ (:38462,:38468): the PIN of an oxime is the N-hydroxy imine."""
    for res in (name_default(smiles), name_breadth(smiles)):
        assert (res.get("name"), res.get("tier"), res.get("is_pin")) == (pin, "pin_verified", True), res
    assert_full_rt(pin, smiles)


def test_camphor_oxime_has_one_label_in_every_spelling():
    """The canonical spelling got the false PIN label and the given spelling did not: one molecule, two
    labels. Every spelling is named from RDKit's canonical SMILES now and none is a PIN."""
    given = "C[C@@]12CC[C@@H](C1(C)C)CC2=NO"
    mol = Chem.MolFromSmiles(given)
    spellings = list(dict.fromkeys([given, Chem.MolToSmiles(mol), *Chem.MolToRandomSmilesVect(mol, 5, randomSeed=11)]))
    assert len(spellings) >= 3
    labels = {(name_default(s).get("tier"), name_breadth(s).get("name"), name_breadth(s).get("tier"))
              for s in spellings}
    assert labels == {("abstain", "(1R,4R)-1,7,7-trimethylbicyclo[2.2.1]heptan-2-one oxime",
                       "systematic_verified")}, labels
