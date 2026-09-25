"""Radicals with several centres,,,.

Blue Book PIN examples, byte-identical, each verified by a strict OPSIN -r round
trip -- which also proves the Kekulé-twin diradicals get their RADICAL name, never a
closed-shell one (p-benzoquinone for [O]c1ccc([O])cc1, xylene for the dimethyl).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from tests.support.jars import jar_or_skip


def _strict_rt(name, smiles):
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


ROWS = [
    ("[c]1cc[c]cc1", "benzene-1,4-diyl"),                                  # (PIN):40540
    ("O=[C]c1ccc(cc1)[C]=O", "benzene-1,4-dicarbonyl"),                    # (PIN):40618
    ("O=[S]c1ccc(cc1)[S]=O", "benzene-1,4-disulfinyl"),                    # (PIN):40614
    ("[N]=C=[N]", "methanebis(iminyl)"),                                   # (PIN):40664
    ("O=C([NH])c1ccccc1C(=O)[NH]", "benzene-1,2-bis(carboxamidyl)"),       # (PIN):40667
    ("[NH]C(=O)CCC(=O)[NH]", "butanebis(amidyl)"),                         # (PIN):40670
    ("[CH2]C1CC1[CH2]", "(cyclopropane-1,2-diyl)dimethyl"),                # (PIN):40745
    ("CC(C)([O])CC(C)(C)[O]", "(2,4-dimethylpentane-2,4-diyl)bis(oxyl)"),  # (PIN):40749
    ("[O]OC1CC(C1)O[O]", "(cyclobutane-1,3-diyl)bis(peroxyl)"),            # (PIN):40751
    ("[CH2][CH2]", "ethane-1,2-diyl"),                                     # guard:40530
    ("[CH2][CH][CH2]", "propane-1,2,3-triyl"),                             # guard:40536
    ("[CH2]C=C[CH2]", "but-2-ene-1,4-diyl"),                               # guard
    ("[O]c1ccc([O])cc1", "(benzene-1,4-diyl)bis(oxyl)"),                   # Kekulé twin of p-benzoquinone
    ("[CH2]c1ccccc1[CH2]", "(benzene-1,2-diyl)dimethyl"),                  # Kekulé twin
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ROWS)
def test_multicentre(smiles, expected):
    with jvm_slots(1, purpose="test-radical-b6"):
        row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)
