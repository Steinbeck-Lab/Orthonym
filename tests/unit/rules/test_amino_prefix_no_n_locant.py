"""An amino prefix cites its N-substituent without the locant 'N-'.

 (the Blue Book): "Preferred IUPAC names for prefixes corresponding to -NHR,
-NRR', or -NR2 are formed by prefixing the names of the groups R and R' to the prefix 'amino',
for example 'methylamino' for -NH-CH3"; '4,4-bis(methylamino)butanoic acid (PIN)' (:26318);
'6-[(methylamino)sulfinyl]naphthalene-2-carboxylic acid (PIN)' (:32989); a compound alkyl
prefix is enclosed, '2-[di(butan-2-yl)amino]butan-2-ol (PIN)' (:28170). The ring namer wrote
'4-(N-methylamino)benzoic acid' and '4-[N-(methoxysulfinyl)amino]benzoic acid' (pin_verified).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccc(NC)cc1", "4-(methylamino)benzoic acid"),
    ("OC(=O)c1ccc(NCC)cc1", "4-(ethylamino)benzoic acid"),
    ("Oc1ccc(NC)cc1", "4-(methylamino)phenol"),
    ("OC(=O)c1ccc(NC(C)C)cc1", "4-[(propan-2-yl)amino]benzoic acid"),
    ("OC(=O)c1ccc(NC)c(Cl)c1", "3-chloro-4-(methylamino)benzoic acid"),
    ("COS(=O)Nc1ccc(C(=O)O)cc1", "4-[(methoxysulfinyl)amino]benzoic acid"),
    ("CNc1ccccc1", "N-methylaniline"),      # the amine as the suffix keeps 'N-'
])
def test_amino_prefix_without_n(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
