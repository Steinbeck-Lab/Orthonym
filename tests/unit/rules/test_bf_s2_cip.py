""" (j) (the Blue Book): "When there is a choice for lower locants related to the
presence of stereogenic centers or stereoisomers, the lower locant is assigned to CIP
stereodescriptors Z, R, M, and r... that are preferred to E, S, P, and s, respectively".
The meso 2,3-diols of 1,2,3,4-tetrahydro-1,4-methanonaphthalene: the mirror numbering of
the parent maps the constitution onto itself, so and (a)-(g) tie and
only the descriptors differ. OPSIN 2.9.0 reads both spellings to the input's full
InChIKey: only these tests hold the PIN spelling."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import _cip_key, build
from tests.support.rt_assert import name_is_rt_exact

CIP_ROWS = [
    ("O[C@@H]1[C@H](O)[C@H]2C[C@@H]1c1ccccc12",
     "(1R,2S,3R,4S)-1,2,3,4-tetrahydro-1,4-methanonaphthalene-2,3-diol"),   # not (1S,2R,3S,4R)
    ("O[C@@H]1[C@H](O)[C@@H]2C[C@H]1c1ccccc12",
     "(1R,2R,3S,4S)-1,2,3,4-tetrahydro-1,4-methanonaphthalene-2,3-diol"),   # not (1S,2S,3R,4R)
]


@pytest.mark.parametrize("smiles,name", CIP_ROWS)
def test_the_cip_tie_is_broken_by_r_at_the_first_point_of_difference(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


def test_cip_key_orders_r_before_s_and_z_before_e():
    a = _cip_key([(1, "R"), (2, "S")])
    b = _cip_key([(1, "S"), (2, "R")])
    assert a < b
    assert _cip_key([(2, "Z")]) < _cip_key([(2, "E")])
    assert _cip_key([(1, "R"), (2, "?")]) is None


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", CIP_ROWS)
def test_the_cip_tie_ships_the_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
