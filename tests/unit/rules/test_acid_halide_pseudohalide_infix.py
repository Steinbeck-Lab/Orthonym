"""A pseudohalide beside a halide on an -ic acid is cited as an infix; the halide keeps the class name.

IUPAC 2013 (the Blue Book), "names are formed on the basis of the senior class":
'(CH3)2N-P(O)(NCO)Cl N,N-dimethylphosphoramidisocyanatidic chloride (PIN)' (:35727). An -ous acid keeps the class
names, 'diphenylstibinous isocyanate (PIN)' (:35714).
"""

import pytest

from orthonym import name_compound

pytestmark = pytest.mark.unit


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,name", [
    ("CN(C)P(=O)(Cl)N=C=O", "N,N-dimethylphosphoramidisocyanatidic chloride"),
    ("CCP(=O)(Cl)N=C=O", "ethylphosphonisocyanatidic chloride"),
    ("O=C=N[Sb](c1ccccc1)c1ccccc1", "diphenylstibinous isocyanate"),
])
def test_names(smiles, name):
    assert name_compound(smiles) == name
