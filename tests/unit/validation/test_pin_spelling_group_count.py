""": the parent of a name carries the maximum number of principal groups (SL lane G, review fix 1).

A round trip proves the molecule, never the parent: OPSIN reads 'N-(2-aminoethyl)aniline' back to the
structure of the preferred 'N1-phenylethane-1,2-diamine'. "The senior parent structure has the
maximum number of substituents corresponding to the principal characteristic group (suffix) or senior
parent hydride in accord with the seniority of classes and the seniority of suffixes "
(the Blue Book). (:26375): "Complex polyamines, i.e., compounds in which a choice between
two or more parent hydrides must be made... The senior parent amine structure is chosen in accordance with
the choice of a principal chain or a senior ring or ring system, as described in "; its example
'N1-(aminomethyl)ethane-1,2-diamine (PIN)' has the diamine for parent. A secondary or tertiary amine nitrogen
is bonded to a carbon of every parent it can have,:26247), so it is a group of each of them.
The check counts the groups of the senior class that each candidate parent (every ring system, every chain of
acyclic carbons) carries, on the structure, and reads the count the name's parent cites.
"""
import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps

pytestmark = pytest.mark.unit


def _rules(smiles, name):
    return [f.rule for f in ps.check_pin_spelling(Chem.MolFromSmiles(smiles), name)]


@pytest.mark.parametrize("smiles,name", [
    # a diamine chain is the parent, not the aniline
    ("NCCNc1ccccc1", "N-(2-aminoethyl)aniline"),
    ("NCCNc1ccc(Cl)cc1", "N-(2-aminoethyl)-4-chloroaniline"),
    ("CC(N)CNc1ccccc1", "N-(2-aminopropyl)aniline"),
    ("NCCCNc1ccccc1", "N-(3-aminopropyl)aniline"),
    ("NCc1ccc(NCCN)cc1", "N-(2-aminoethyl)-4-(aminomethyl)aniline"),
    ("CCN(CC)CCNc1ccccc1", "N-[2-(diethylamino)ethyl]aniline"),
    # the same class on a heteroring and a fused ring (pin_verified on main)
    ("NCCNc1ccncc1", "N-(2-aminoethyl)pyridin-4-amine"),
    ("NCCNc1cccc2ccccc12", "N-(2-aminoethyl)naphthalen-1-amine"),
    # the diamide chain, not the butanamide with an acylamino prefix (d1914, marimastat)
    ("CNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)[C@H](O)C(=O)NO)C(C)(C)C",
     "(2S)-2-{(2R)-2-[(1S)-1-hydroxy-2-(hydroxyamino)-2-oxoethyl]-4-methylpentanamido}-N,3,3-trimethylbutanamide"),
    # the ring that carries two groups, not the one that carries one
    ("Nc1ccc(Nc2ccccc2)cc1", "4-amino-N-phenylaniline"),
    ("N#Cc1cc(C#N)ccc1", "3-cyanobenzonitrile"),
    ("Oc1cc(O)cc(-c2ccccc2)c1", "5-phenyl-3-hydroxyphenol"),
])
def test_a_parent_that_carries_fewer_groups_than_another_candidate_fails(smiles, name):
    assert "P-44.1.1" in _rules(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    ("NCCNc1ccccc1", "N1-phenylethane-1,2-diamine"),
    ("NCCNc1ccncc1", "N1-(pyridin-4-yl)ethane-1,2-diamine"),
    ("CNC(=O)[C@@H](NC(=O)CC)C(C)(C)C", "N,3,3-trimethyl-2-propanamidobutanamide"),
    ("Nc1ccc(Nc2ccccc2)cc1", "N1-phenylbenzene-1,4-diamine"),
    ("N#Cc1cc(C#N)ccc1", "benzene-1,3-dicarbonitrile"),
    # one amine nitrogen on a ring and a chain: the count ties and the ring is the parent
    ("CCC(CSc1ccccc1)Nc1ccccc1", "N-[1-(phenylsulfanyl)butan-2-yl]aniline"),
    ("c1ccccc1Nc1ccccc1", "N-phenylaniline"),
    ("NCc1ccccc1N", "2-(aminomethyl)aniline"),
    ("NCCCN", "propane-1,3-diamine"),
    ("OCC(O)CO", "propane-1,2,3-triol"),
    # the names the book prints with several groups of a retained or multiplicative name
    ("c1(O)c(O)c(O)c(O)c(O)c1O", "benzenehexol"),
    ("NC(=O)C(N)=O", "oxamide"),
    ("N#CC#N", "oxalonitrile"),
    ("CCNCNCC", "N,N'-methylenediethanamine"),
    ("OC[C@@H](O)[C@H](O)[C@@H](O)CO", "xylitol"),
])
def test_a_parent_that_carries_the_most_groups_passes(smiles, name):
    assert "P-44.1.1" not in _rules(smiles, name)


def test_a_name_of_another_class_than_the_senior_one_is_left_to_the_class_checks():
    # the alcohol is the senior class: the name that cites the amine is 's to fail, not this check's
    assert "P-44.1.1" not in _rules("NCCNc1ccccc1CO", "2-{[(2-aminoethyl)amino]}benzyl alcohol")


def test_the_message_counts_the_groups():
    fails = [f for f in ps.check_pin_spelling(
        Chem.MolFromSmiles("NCCNc1ccccc1"), "N-(2-aminoethyl)aniline") if f.rule == "P-44.1.1"]
    assert fails and "carries 1 'amine' group(s)" in fails[0].detail and "carries 2" in fails[0].detail
