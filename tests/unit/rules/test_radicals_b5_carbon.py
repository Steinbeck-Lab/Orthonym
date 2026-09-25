"""Carbon-centred radicals,.

A monovalent carbon radical formed by removing one H is named like the substituent
group of the whole molecule attached at the radical carbon. Contracted preferred
prefixes are NOT radical names (1), the Blue Book): (CH3)3C. is
'2-methylpropan-2-yl (PIN)' (:40453). Each name must match exactly and pass a strict
OPSIN -r round trip (canonical SMILES, radical dots included).
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
    ("[c]1ccc2ccccc2c1", "naphthalen-2-yl"),          # (PIN):40459 (was abstain)
    ("[c]1cccc2ccccc12", "naphthalen-1-yl"),          # was 'naphthyl'
    ("[CH]1C=CC=C1", "cyclopenta-2,4-dien-1-yl"),     # (PIN):40457 (guard)
    ("C1=CC[CH]C=C1", "cyclohexa-2,4-dien-1-yl"),
    ("C[CH]C(=O)O", "1-carboxyethyl"),                # (:40526): groups as prefixes
    ("[CH2]CO", "2-hydroxyethyl"),
    ("[c]1ccncc1", "pyridin-4-yl"),
    ("C[CH]c1ccccc1", "1-phenylethyl"),
    ("[CH]1C=Cc2ccccc21", "1H-inden-1-yl"),
    ("[CH2]C#N", "cyanomethyl"),                      # was 'cyanomethanyl'
    ("C[C](C)C", "2-methylpropan-2-yl"),              # (PIN):40453, not 'tert-butyl'
    ("CC[CH]CC", "pentan-3-yl"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_carbon_radical(smiles, expected):
    with jvm_slots(1, purpose="test-radical-b5"):
        row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)
