"""Slice S2 targets: bridged fused PINs on every parent the fused namers can number.

Every expected name was read back by OPSIN 2.9.0 to the input's full InChIKey before it
was written here (S2 planning notes, ledger). Blue Book rows (the Blue Book)::19855
'4,7-methanoazulene (PIN)',:14653 '1H-3a,7-ethanoazulene (PIN)',:14253
'1,5-methanoindole (PIN)',:14648 '2H,7H-4a,7-ethano-1-benzopyran (PIN)',:24653
'1H,3H-3a,7a-methano-2-benzofuran (PIN)',:14193 '1H-1,4-ethanothioxanthene (PIN)',
:14187 '1,4:6,9-dimethanooxanthrene (PIN)',:14189 '6,9-epoxy-1,4-methanobenzo[8]annulene
(PIN)',:19831 '2H-1,4:5,8-dimethanobenzo[7]annulene (PIN)'."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

BB_TARGETS = [
    ("C1=CC2=CC3=CC=C(C3)C2=C1",
     "4,7-methanoazulene"),
    ("C1=CC23C=CCC2=CC(=C1)CC3",
     "1H-3a,7-ethanoazulene"),
    ("c1cc2c3ccn2Cc1c3",
     "1,5-methanoindole"),
    ("C1=CC23C=CC(C=C2OC1)CC3",
     "2H,7H-4a,7-ethano-1-benzopyran"),
    ("C1=CC23COCC2(C=C1)C3",
     "1H,3H-3a,7a-methano-2-benzofuran"),
    ("C1=CC2CCC1=C1Sc3ccccc3C=C12",
     "1H-1,4-ethanothioxanthene"),
    ("C1=C2CC(=C1)C1=C2OC2=C(O1)C1=CC=C2C1",
     "1,4:6,9-dimethanooxanthrene"),
    ("C1=CC2=C3C=c4ccc(o4)=CC3=C1C2",
     "6,9-epoxy-1,4-methanobenzo[8]annulene"),
    ("C1=CC2=C3C4=CCC(=C3C=C1C2)C4",
     "2H-1,4:5,8-dimethanobenzo[7]annulene"),
]
DEV_TARGETS = [  # milestone1500 (2) and dev2000 (1); abstain at the PIN tier at the base
    ("C1CC2CC3C1CC(C3)C2O",
     "octahydro-1H-2,5-methanoinden-8-ol"),
    ("C1CC[C@@]2([C@H](C1)C[C@H]3CCC[C@@H]2[C@H]3O)O",
     # slice S3: the 4a-ol sits on a fusion atom with no hydrogen in the mancude parent,
     # so it takes 'added indicated hydrogen', the Blue Book)
     "(4aS,5R,9R,10aR,11S)-decahydro-5,9-methanobenzo[8]annulene-4a,11(2H)-diol"),
    ("CC1(C)CCC[C@]2(C)[C@@H]3[C@H]1[C@@H](O)[C@@]2(C)C[C@H]3O",
     "(1S,3R,3aS,4R,8aR,9R)-1,5,5,8a-tetramethyldecahydro-1,4-methanoazulene-3,9-diol"),
]
CLASS_TARGETS = [
    ("C1=CC2C3C=CC(C3)C2C1",
     "3a,4,7,7a-tetrahydro-1H-4,7-methanoindene"),  # dicyclopentadiene
    ("C1CC2C3CCC(C3)C2C1",
     "octahydro-1H-4,7-methanoindene"),
    ("C12C=CC(C1)c1ccc3ccccc3c12",
     "1,4-dihydro-1,4-methanophenanthrene"),
    ("C12=CC=C(C3=C4C=5C6=CC=C(C5C(=C13)C4)C6)C2",
     "1,4:5,8:9,10-trimethanoanthracene"),
    ("C1=CC23C=CC=CC2(C=C1)C=CC3",
     "4a,8a-prop[1]enonaphthalene"),
    ("C1=CC2OCC1c1ccccc12",
     "1,4-dihydro-1,4-(epoxymethano)naphthalene"),
    ("c1n[nH]c2c1C1CCC2C1",
     "4,5,6,7-tetrahydro-1H-4,7-methanoindazole"),
    ("O[C@@H]1[C@H](O)[C@H]2C[C@@H]1c1ccccc12",
     "(1R,2S,3R,4S)-1,2,3,4-tetrahydro-1,4-methanonaphthalene-2,3-diol"),
    ("C12CCC(C3=CC=CC=C13)C2CCO",
     "2-(1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-yl)ethan-1-ol"),
    ("N#Cc1ccc2c(c1)C1CCC2C1",
     "1,2,3,4-tetrahydro-1,4-methanonaphthalene-6-carbonitrile"),
]
ALL_TARGETS = BB_TARGETS + DEV_TARGETS + CLASS_TARGETS



def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s2_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
