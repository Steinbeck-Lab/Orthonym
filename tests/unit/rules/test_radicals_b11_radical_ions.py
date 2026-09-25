"""Radical ions from mononuclear ionic parent hydrides.

the Blue Book-43422: the ionic parent hydride's name + 'yl'/'ylidene', "with
elision of the final letter 'e'". '-ylium' cations are NOT radical ions even
though RDKit counts their empty orbital as two radical electrons; they keep their
names (user decision Q2 = A: a separate build).
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


def _name(smiles):
    with jvm_slots(1, purpose="test-radical-b11"):
        return Orthonym(style="pin").name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[B-](C)C", "trimethylboranuidyl"),                    # (PIN):43430
    ("COC(=O)[C-]", "(methoxycarbonyl)methanidylidene"),      # (PIN):43439
    ("[CH2+]", "methyliumyl"),                                # (PIN):43443
    ("CCC[OH+]", "propyloxidaniumyl"),                        # (PIN):43515
    ("CC(=O)[N-]", "acetylazanidyl"),                         # (PIN):43517
    ("C[NH2+]", "methylazaniumyl"),
])
def test_radical_ion(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected and row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C[O+]", "methoxylium"),          # retained PIN, not 'methyloxidaniumylidene'
    ("[NH2+]", "azanylium"),           # gold row
    ("[O+]c1ccccc1", "phenoxylium"),
    ("[NH3+]C", "methanaminium"),      # closed-shell control
])
def test_ylium_and_closed_shell_cations_unchanged(smiles, expected):
    assert _name(smiles)["name"] == expected
