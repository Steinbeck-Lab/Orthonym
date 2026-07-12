"""W2F-P8 Task 2: P-45.6.2 guard — multiplicative nomenclature is configuration
dependent. The stereo-differing (R vs S) diaryl ether is NOT two identical units,
so a multiplicative name must decline (BlueBookV2.md:22281/22585); the PIN is the
substitutive name built by Tasks 3-4. Pins the correct decline so a later change
cannot make multiplicative fire on stereo-differing arms. No code change.
"""
from rdkit import Chem

from orthonym.rules.multiplicative import name_multiplicative


def test_stereo_differing_ether_declines():
    # P-45.6.2: R vs S arms are NOT identical -> multiplicative must return None
    assert name_multiplicative(Chem.MolFromSmiles(
        "C[C@H](Cl)c1ccc(Oc2ccc(cc2)[C@@H](C)Cl)cc1")) is None


def test_identical_arms_multiplicative_kept():
    # regression: identical (achiral, symmetric) arms still use the multiplicative PIN
    import orthonym
    assert orthonym.name_compound("Cc1ccc(Oc2ccc(C)cc2)cc1", style="pin") == \
        "1,1'-oxybis(4-methylbenzene)"
