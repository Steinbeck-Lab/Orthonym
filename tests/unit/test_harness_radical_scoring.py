"""eval/harness.py scores radical names soundly.

A radical input's name is parsed with OPSIN -r; a closed-shell input's is not.
An equal full InChIKey with a different radical graph is 'radical_mismatch'
(a wrong molecule), never 'rt_exact' -- the key encodes neither radical
electrons nor bond order.
"""
import sys
from pathlib import Path

import pytest

from tests.support.jars import jar_or_skip

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))


@pytest.mark.parametrize("smiles,name,outcome", [
    ("C[O]", "methoxyl", "rt_exact"),
    ("CC1(C)CCCC(C)(C)N1[O]", "(2,2,6,6-tetramethylpiperidin-1-yl)oxyl", "rt_exact"),
    ("[CH2][CH2]", "ethene", "radical_mismatch"),        # closed-shell name, radical input
    ("C=C", "1λ3,2λ3-ethane", "radical_mismatch"),       # radical parse, closed-shell input
    ("CCO", "ethanol", "rt_exact"),
])
def test_radical_scoring(smiles, name, outcome):
    jar_or_skip()
    import harness
    rows, _ = harness.score([{"smiles": smiles, "name": name, "tier": "pin_verified"}], {})
    assert rows[0]["outcome"] == outcome
