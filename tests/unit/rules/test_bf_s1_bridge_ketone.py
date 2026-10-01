"""A ketone on a saturated bridge atom of a bridged fused parent (slice S1, Task S1.8b).

 (the Blue Book): "As the formation of ketones is achieved by the
conversion of a methylene, >CH2, group into a >C=O group, the suffix 'one' with appropriate
locants can be added to the name of parent hydrides having such groups." A carbon of a
saturated bridge (methano, ethano) is such a >CH2 group, so no added hydrogen is needed
('1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one'). A ketone at a ring position of the
fused parent needs added hydrogen (:28388,:24693): slice S3, declined here.
Every expected name was read back by OPSIN 2.9.0 to the input's full InChIKey (S1.8b)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import build
from tests.support.rt_assert import name_is_rt_exact

BRIDGE_KETONES = [
    ("O=C1C2CCC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one"),
    ("O=C1[C@@H]2CC[C@@H]1c1c(F)c(F)c(F)c(F)c12",                       # the pubchem10k row
     "(1R,4R)-5,6,7,8-tetrafluoro-1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one"),
    ("O=C1CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethanoanthracen-11-one"),   # C11 next to C10
    ("O=C1C(=O)C2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethanoanthracene-11,12-dione"),
    ("O=C1C2CCC1C1CCCCC12", "decahydro-1,4-methanonaphthalen-9-one"),
    ("O=C1C2C=CC1c1ccccc12", "1,4-dihydro-1,4-methanonaphthalen-9-one"),
    ("O=C1CC2C=CC1c1ccccc12", "1,4-dihydro-1,4-ethanonaphthalen-9-one"),         # (j): the ethano reading
    ("O=C1CC23C=CC=CC12C=CC=C3", "4a,8a-ethanonaphthalen-9-one"),                 # C9 next to 8a
    ("O=C1C2CCC1c1c2c2ccc1o2", "5,6,7,8-tetrahydro-1,4-epoxy-5,8-methanonaphthalen-9-one"),
    ("CC(=O)c1ccc2c(c1)C1CCC2C1=O", "6-acetyl-1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one"),
    ("C[C@@]12CC[C@@H](C1=O)c1ccccc21", "(1S,4R)-1-methyl-1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-one"),
]
RING_KETONES = [   # added hydrogen needed at the ketone position: slice S3, declined
    "O=C1CC2CC1c1ccccc12",           # the ring C2 of 1,2,3,4-tetrahydro-1,4-methanonaphthalene
    "O=C1CC2CCC1c1ccccc12",          # two ethano readings; (c) puts the ketone in the ring (2)
    "O=C1C(=O)C2CCC1c1ccccc12",      # the same for the dione
    "O=C1C=CC(=O)C2C3CCC(C3)C12",    # ketones on the fused parent's ring
]


@pytest.mark.parametrize("smiles,name", BRIDGE_KETONES)
def test_bridge_ketone_name(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


@pytest.mark.parametrize("smiles", RING_KETONES)
def test_ring_ketone_is_declined(smiles):
    assert build(Chem.MolFromSmiles(smiles)) is None


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s1"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", BRIDGE_KETONES[:5])
def test_bridge_ketone_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_a_ring_ketone_keeps_its_von_baeyer_name_at_best_effort():
    smiles = "O=C1CC2CCC1c1ccccc12"
    assert _row(smiles, "pin")["tier"] == "abstain"
    be = _row(smiles, "best-effort")
    assert be["tier"] == "systematic_verified" and "cyclo[" in be["name"], be
    assert name_is_rt_exact(be["name"], smiles)
