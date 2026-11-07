"""Mixed anhydrides of a sulfonic or sulfinic acid "Mixed anhydrides", the Blue Book).

"Anhydrides derived from different monobasic acids are named by citing in alphabetical order the names of the two
acids, substituted or unsubstituted, without the class name 'acid' followed by the class name 'anhydride'":
'benzenesulfinic ethanesulfonic anhydride (PIN)' (:32261), 'chloroacetic 4-nitrobenzene-1-sulfonic anhydride (PIN)'
(:32270). An acyl group on the oxygen of a sulfonic acid is the other acid of an anhydride, not the organyl group of an
ester ('acetyl methanesulfonate' is no name of.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("O=S(c1ccccc1)OS(=O)(=O)CC", "benzenesulfinic ethanesulfonic anhydride"),                      #:32261
    ("[O-][N+](=O)c1ccc(cc1)S(=O)(=O)OC(=O)CCl", "chloroacetic 4-nitrobenzene-1-sulfonic anhydride"),  #:32270
    ("CC(=O)OS(C)(=O)=O", "acetic methanesulfonic anhydride"),
    ("CC(=O)OS(=O)(=O)c1ccc(C)cc1", "acetic 4-methylbenzene-1-sulfonic anhydride"),
    ("O=C(OS(=O)(=O)C(F)(F)F)c1ccccc1", "benzoic trifluoromethanesulfonic anhydride"),
    ("O=COS(C)(=O)=O", "formic methanesulfonic anhydride"),
    ("CC(=O)OS(=O)C", "acetic methanesulfinic anhydride"),
    ("CS(=O)(=O)OS(C)(=O)=O", "methanesulfonic anhydride"),                                        # unchanged
    ("ClCCS(=O)OS(=O)CCCl", "bis(2-chloroethane-1-sulfinic) anhydride"),                           #:32251
    ("CS(=O)OS(C)=O", "methanesulfinic anhydride"),
    # "Di- and polyanhydrides" (:32378): "two or more -CO-O-CO- or related groups, such as -SO2-O-SO2-"
    ("CC(=O)OS(=O)(=O)CS(=O)(=O)OC(=O)CC", "acetic methanedisulfonic propanoic dianhydride"),          #:32388
    ("CC(=O)OC(=O)CCC(=O)OS(C)(=O)=O", "acetic butanedioic methanesulfonic dianhydride"),
    ("CC(=O)OS(=O)(=O)c1ccc(cc1)S(=O)(=O)OC(C)=O", "diacetic benzene-1,4-disulfonic dianhydride"),
    # method (1) with a sulfonyl link (:32417)
    ("CC(=O)OC(=O)CCC(=O)OS(=O)(=O)CS(=O)(=O)OC(C)=O",
     "acetic butanedioic [(acetyloxy)sulfonyl]methanesulfonic dianhydride"),
])
def test_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    # (:44642): a configuration at the sulfinic sulfur stands in front of its acid word, which this namer does
    # not write: no PIN rather than '(R)-acetic methanesulfinic anhydride'
    "C[S@](=O)OC(C)=O",
    "C[S@](=O)OC(=O)c1ccccc1",
    # "Di- and polyanhydrides" (:32380): two links are a dianhydride; a residue linked at a carboxylic C and
    # at a sulfonic S would cite one link as a prefix ('diacetic 3-sulfopropanoic dianhydride'), a form the book never
    # gives (as a carboxy prefix,:32433)
    "CC(=O)OS(=O)(=O)CCC(=O)OC(C)=O",
    # a configuration at a sulfinic S of a dianhydride
    "C[S@](=O)OS(=O)(=O)CS(=O)(=O)OC(C)=O",
    "O=NOC(=O)CCS(=O)(=O)OC(C)=O",
])
def test_no_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] != "pin_verified"
