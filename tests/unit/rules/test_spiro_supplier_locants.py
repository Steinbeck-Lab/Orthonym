"""The mixed spiro/fused locant supplier hands the cascade only ``int`` / ``(int, str)`` locants.

The spiro namers carry a lettered fusion locant of the primed component as ``('7a', "'")`` (``7'a``,
 "Primed locants are placed immediately after the corresponding unprimed: 4, 4', 4a, 4'a").
``get_mixed_spiro_fused_iupac_locants`` is the cascade-step-6 supplier (declared ``_Locant = int |
(int, str)``); it leaked that shape, ``p44_scorer._candidate_locants`` then compared ``'7a'`` with
``4``, raised TypeError, and every molecule with such a spiro ring system abstained.
"""
import pytest
from rdkit import Chem

from orthonym.rules.locants import cascade_locant
from orthonym.rules.spiro import get_mixed_spiro_fused_iupac_locants
from tests.support.rt_assert import assert_tier_contract

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("given, expected", [
    (3, 3),
    ((4, "a"), (4, "a")),
    ("7a", (7, "a")),
    (("7a", "'"), (7, "a'")),
    (("10a", "''"), (10, "a''")),
    ((2, "'"), (2, "'")),
    (("5", "'"), (5, "'")),
])
def test_cascade_locant_shape(given, expected):
    assert cascade_locant(given) == expected


def test_primed_lettered_locants_order_as_numbers():
    # 10a' must not sort before 7a' (string order); orders the numeric part first
    assert sorted([cascade_locant(("10a", "'")), cascade_locant(("7a", "'"))]) == [(7, "a'"), (10, "a'")]


def test_supplier_dict_is_contract_shaped():
    # spiro[azetidine-3,5'-pyrrolo[1,2-a]imidazole]: the second component has fusion atoms 3'a / 7'a
    mol = Chem.MolFromSmiles("C1CC2=NC=CN2C13CNC3")
    locants = get_mixed_spiro_fused_iupac_locants(mol)
    assert locants
    for loc in locants.values():
        assert isinstance(loc, int) or (isinstance(loc, tuple) and isinstance(loc[0], int) and isinstance(loc[1], str)), loc
    assert any(isinstance(v, tuple) and v[1].startswith("a") for v in locants.values())


# each abstained at best-effort with TypeError('<' not supported between 'int' and 'str') in the parent pool
@pytest.mark.parametrize("smiles", [
    # N-acyl amine on a spiro[pyrrolo[1,2-a]imidazole-azetidine] with a ring-bearing N-substituent
    "C1CC2=CC(=NN2C1)CN3CC4(C3)CC(C5=NC=CN45)NC(=O)CC(F)(F)F",
    # spiro[indene-cyclopropane] ester
    "CC(=C)C(=O)OC1C2=C(CC13CC3)C=C(C=C2)C4CCCC(C4)C5CC6=CC=CC=C6C5OC(=O)C(C)(C)C",
    # spiro[isoquinoline-cyclobutane] substituent on a tetrahydroisoquinoline
    "CCC1=CC2=C(CN(C(C2)CO)C)C(=C1)CC(C)C3=CC4=C(C=C3)C5(CCC5)NCC4",
])
def test_spiro_with_lettered_primed_fusion_locants_is_named(smiles):
    assert_tier_contract(smiles)
