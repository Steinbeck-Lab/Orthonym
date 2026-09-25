""" radical names that used to ship as the right molecule but the wrong spelling.

Each row cites the Blue Book line that fixes the spelling; each name must also
pass a strict OPSIN -r round trip (canonical SMILES, radical dots included).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from tests.support.jars import jar_or_skip


def _strict_rt(name: str, smiles: str) -> bool:
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


ROWS = [
    # (the Blue Book): "the IUPAC preferred name for HO. is 'hydroxyl'
    #... and... for HOO. is 'hydroperoxyl'" (was 'oxyl').
    ("[OH]", "hydroxyl"),
    ("[O]O", "hydroperoxyl"),
    # (:40681): the retained list ends at butoxyl and tert-butoxyl
    # (was 'tert-butyloxyl'); a longer chain takes method (1), '<R>oxyl' (was 'pentoxyl').
    ("CC(C)(C)[O]", "tert-butoxyl"),
    ("CCC[O]", "propoxyl"),
    ("CCCCC[O]", "pentyloxyl"),
    # (:40694): '(chloroacetyl)oxyl (PIN)' (was '(2-chloro-1-oxoethyl)oxyl').
    ("ClCC(=O)[O]", "(chloroacetyl)oxyl"),
    ("CCCC(=O)[O]", "butanoyloxyl"),
    # "Acyl radicals" (:40600-40612): named from the acid.
    ("C[P](=O)C", "dimethylphosphinoyl"),          # was 'dimethyl-λ5-phosphanonyl'
    ("CCCCC[C]=O", "hexanoyl"),
    ("C[C]=S", "ethanethioyl"),
    ("CCC[C]=N", "butanimidoyl"),
    ("O=[C]C1CCCCC1", "cyclohexanecarbonyl"),
    ("CC(C)[C]=O", "2-methylpropanoyl"),
    ("N[C]=O", "carbamoyl"),                        # 'carbamoyl (preferred prefix)' (:17776)
    ("O=[C]OO", "carbonoperoxoyl"),                 # (:30192) "used in preferred IUPAC names"
    # (:40634): '(CH3)3P=N. trimethyl-λ5-phosphaniminyl (PIN)' (was 'P,P,P-...').
    ("CP(=[N])(C)C", "trimethyl-λ5-phosphaniminyl"),
    # 'boranylidene (preselected prefix) (not borylidene)' (:15884) (was 'methylborylidene').
    ("[B]C", "methylboranylidene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_radical_pin_spelling(smiles, expected):
    with jvm_slots(1, purpose="test-radical-b8"):
        row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected
    assert row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)
