"""Methane-centred radicals and radicals on unbranched homogeneous heteroatom chains.

'diphenylmethylidene (PIN)', 'trisilan-2-yl (PIN)', 'hydrazine-1,2-diyl (PIN)' are
Blue Book rows of the radical pack (benchmarks/bb_conformance/radical_pack.json).
Every name passes a strict OPSIN -r round trip.
"""
import pytest

from tests.support.default_tier import assert_default_tier_declines, strict_path_row
from tests.support.pin_tiers import is_carbon_free
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


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C1(=CC=CC=C1)[C]C1=CC=CC=C1", "diphenylmethylidene"),   # BB (PIN)
    ("[C](c1ccccc1)(c1ccccc1)c1ccccc1", "triphenylmethyl"),
    ("[CH](c1ccccc1)C1CCCCC1", "cyclohexyl(phenyl)methyl"),
    ("[SiH3][SiH][SiH3]", "trisilan-2-yl"),                     # BB (PIN)
    ("[NH][NH]", "hydrazine-1,2-diyl"),                         # BB (PIN)
])
def test_named(smiles, expected):
    with jvm_slots(1, purpose="test-radical-methane-chain"):
        if is_carbon_free(smiles):
            # a preselected name, no PIN (user decision 2026-11-02; ``pin_tiers.CARBON_FREE_RULE``,
            # the Blue Book,:2058): the default tier declines it, the strict path builds the name
            assert_default_tier_declines(smiles)
            row = strict_path_row(smiles)
        else:
            row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == expected and row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)
