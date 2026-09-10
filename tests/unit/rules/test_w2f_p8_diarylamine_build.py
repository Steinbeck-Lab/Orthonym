"""W2F-P8 Task 5: N-(substituted-phenyl) citation for substituted diarylamines.

A substituted N-aryl ring is cited as `N-(2-chlorophenyl)` (via the shared
decorated-ring core namer), promoting the current benzene ring to the retained
`aniline` parent /. The symmetric case isolates the builder
from the parent choice and anchors determinism.

Also: the azanediyl MULTIPLICATIVE name must NOT win when the amine bridges two
prefix-only carbocycles — the amine is then the principal characteristic group,
so the substitutive aniline is the PIN (the Blue Book.
"""
import orthonym
from rdkit import Chem


def test_symmetric_diarylamine():
    # both rings identical (2-chlorophenyl) -> same string either way; the
    # substitutive aniline is the PIN, not 1,1'-azanediylbis(2-chlorobenzene).
    assert orthonym.name_compound("Clc1ccccc1Nc1ccccc1Cl", style="pin") == \
        "2-chloro-N-(2-chlorophenyl)aniline"


def test_diphenylamine_regression():
    # bare N-phenyl path unchanged
    assert orthonym.name_compound("c1ccc(Nc2ccccc2)cc1", style="pin") == \
        "N-phenylaniline"


def test_symmetric_diarylamine_determinism():
    mol = Chem.MolFromSmiles("Clc1ccccc1Nc1ccccc1Cl")
    names = {
        orthonym.name_compound(Chem.MolToSmiles(mol, doRandom=True), style="pin")
        for _ in range(10)
    }
    assert names == {"2-chloro-N-(2-chlorophenyl)aniline"}, names
