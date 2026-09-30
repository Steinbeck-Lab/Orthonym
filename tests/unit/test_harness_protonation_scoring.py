"""eval/harness.py scores a name that puts a hydron or a charge on another atom as a
wrong molecule.

The standard InChIKey's /p layer does not place a hydron, so the partial salts of
one polybasic acid, and the cations of one base protonated on different atoms,
share it. An equal full key with the hydron elsewhere is 'protonation_mismatch'
(validation/protonation_identity.py), never 'rt_exact'.
"""
import sys
from pathlib import Path

import pytest

from tests.support.jars import jar_or_skip

_EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(_EVAL))
pytestmark = pytest.mark.skipif(not (_EVAL / "harness.py").exists(),
                                reason="eval/harness.py is not in this checkout")


@pytest.mark.parametrize("smiles,name,outcome", [
    # the salt of the CH2 carboxylate named as the ring carboxylate
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "sodium 2-(carboxymethyl)benzoate",
     "protonation_mismatch"),
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "sodium (2-carboxyphenyl)acetate", "rt_exact"),
    ("O=C(O)Cc1ccccc1C(=O)[O-].[Na+]", "sodium 2-(carboxymethyl)benzoate", "rt_exact"),
    # the hydron on the methylated N-1 instead of N-3
    ("C[N+]1=CNc2ccccc21", "1-methyl-1H-benzimidazol-1-ium", "protonation_mismatch"),
    ("C[N+]1=CNc2ccccc21", "1-methyl-1H-benzimidazol-3-ium", "rt_exact"),
    # a zwitterion named as the neutral amino acid keeps its treatment
    ("[NH3+]CC(=O)[O-]", "glycine", "rt_exact"),
    # a method (2) 'dihydrogen' name leaves the site among equivalent P open;
    # a substitutive name places it
    ("[Na+].[Na+].OP(=O)([O-])OP(=O)([O-])O", "disodium dihydrogen diphosphate", "rt_exact"),
    ("O=P([O-])(O)C(Cl)(Cl)P(=O)([O-])O", "(dichloro-phosphonomethyl)phosphonate",
     "protonation_mismatch"),
])
def test_protonation_scoring(smiles, name, outcome):
    jar_or_skip()
    import harness
    rows, _ = harness.score([{"smiles": smiles, "name": name, "tier": "pin_verified"}], {})
    assert rows[0]["outcome"] == outcome
