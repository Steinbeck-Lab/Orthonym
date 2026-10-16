""": the ring is the parent of an amine whose nitrogen is bonded to a ring and a chain (SL lane G, G2).

A round trip proves the molecule, never the spelling: OPSIN reads 'N-phenyl-1-(phenylsulfanyl)butan-2-
amine' back to the structure of the preferred 'N-[1-(phenylsulfanyl)butan-2-yl]aniline'.
 "Selection of the principal chain or senior ring system in secondary and tertiary amines"
(the Blue Book): "When a choice for parent hydride is possible between a ring and a chain, the
ring is preferred." Its examples: 'N-cyclohexylaniline (PIN)' (:26288), 'N-butylcyclopropanamine (PIN)'
(:26292, not 'N-cyclopropylbutan-1-amine'). (:19340): within the same class a ring is senior
to a chain. The check reads the structure (one amine nitrogen, bonded to a ring and to a chain) and the end
of the name (a chain stem and the suffix 'amine').
"""
import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps

pytestmark = pytest.mark.unit


def _rules(smiles, name):
    return [f.rule for f in ps.check_pin_spelling(Chem.MolFromSmiles(smiles), name)]


@pytest.mark.parametrize("smiles,name", [
    ("CCC(CSc1ccccc1)Nc1ccccc1", "N-phenyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CCC(Cc1ccccc1)Nc1ccccc1", "N,1-diphenylbutan-2-amine"),
    ("CCCCNC1CC1", "N-cyclopropylbutan-1-amine"),                      # the book's 'not' name (:26292)
    ("CCC(CSc1ccccc1)NC1CCCCC1", "N-cyclohexyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CCC(CSc1ccccc1)N(C)c1ccccc1", "N-methyl-N-phenyl-1-(phenylsulfanyl)butan-2-amine"),
    ("CC(C)NC1CCCCC1", "N-cyclohexylpropan-2-amine"),
    ("CC(C1=CC=CS1)NC2CCC3=CC=CC=C3C2",
     "N-(1,2,3,4-tetrahydronaphthalen-2-yl)-1-(thiophen-2-yl)ethan-1-amine"),
    ("CC=CCNc1ccccc1", "N-phenylbut-2-en-1-amine"),                    # an unsaturated chain stem
])
def test_the_chain_as_parent_of_an_amine_with_a_ring_on_the_nitrogen_fails(smiles, name):
    assert "P-62.2.2.2" in _rules(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # the book's own PINs (:26288,:26292)
    ("c1ccccc1NC1CCCCC1", "N-cyclohexylaniline"),
    ("CCCCNC1CC1", "N-butylcyclopropanamine"),
    ("c1ccccc1Nc1cccnc1", "N-phenylpyridin-3-amine"),                  # heteroring over benzene, ring parents
    # ring parents with a prefixed chain
    ("CCC(CSc1ccccc1)Nc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    ("CC(C)Nc1ccccc1", "N-(propan-2-yl)aniline"),
    # the nitrogen is bonded to chains only: the chain is the parent
    ("CCN(C)Cc1ccccc1", "N-benzyl-N-methylethan-1-amine"),
    ("CNCc1ccccc1", "N-methyl-1-phenylmethanamine"),
    # a second amine nitrogen: counts the groups before the ring is asked
    ("NCCCNc1ccccc1", "N1-phenylpropane-1,3-diamine"),
    # an amide, an amine that is a prefix, an alcohol parent
    ("CCC(C)NC(=O)c1ccccc1", "N-(butan-2-yl)benzamide"),
    ("CCC(CO)Nc1ccccc1", "2-anilinobutan-1-ol"),
    # an amine on a chain ring name: 'cyclobutanamine' and a replacement name are no chain stems
    ("CCCCNC1CCC1", "N-butylcyclobutanamine"),
])
def test_names_without_the_defect_pass(smiles, name):
    assert "P-62.2.2.2" not in _rules(smiles, name)


def test_the_message_names_the_chain():
    fails = [f for f in ps.check_pin_spelling(
        Chem.MolFromSmiles("CCC(Cc1ccccc1)Nc1ccccc1"), "N,1-diphenylbutan-2-amine")
        if f.rule == "P-62.2.2.2"]
    assert fails and "'but'" in fails[0].detail
