"""Slice S1 targets: bridged fused PINs on naphthalene and anthracene parents, in any
hydrogenation state, with prefixes, suffixes and stereodescriptors.

Every expected name was read back by OPSIN 2.9.0 to the input's full InChIKey before it
was written here (the S1 plan, S1_NOTES.md ledger). Blue Book rows: the Blue Book:14249
'4a,8a-ethanonaphthalene (PIN)',:14587 '1,4-epoxy-4a,8a-ethanonaphthalene (PIN)',:19964
'1,2,3,4,4a,9,9a,10-octahydro-9,10-ethanoanthracene (PIN)',:19966
'1,2,3,4,4a,8a,9,9a,10,10a-decahydro-9,10-ethenoanthracene (PIN)'."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

BB_TARGETS = [
    ("C1=CC23C=CC=CC2(C=C1)CC3", "4a,8a-ethanonaphthalene"),
    ("C1=CC23CCC2(C=C1)c1ccc3o1", "1,4-epoxy-4a,8a-ethanonaphthalene"),
    ("c1ccc2c(c1)C1CCC2C2CCCCC12", "1,2,3,4,4a,9,9a,10-octahydro-9,10-ethanoanthracene"),
    ("C1=CC2C(C=C1)C1C=CC2C2CCCCC12",
     "1,2,3,4,4a,8a,9,9a,10,10a-decahydro-9,10-ethenoanthracene"),
]
DEV_TARGETS = [  # a dev split and milestone1500 (the same two rows in both)
    ("CC1(C)C2=CC=CC(C)(C)C23C=CC1C3",
     "1,1,5,5-tetramethyl-1,5-dihydro-2H-2,4a-methanonaphthalene"),
    ("C[C@@]12CCC[C@@]3(C)[C@@H](C1)[C@@](O)(CO)CC[C@@]23C",
     "(1R,2R,4aS,5R,8aS)-2-(hydroxymethyl)-4a,5,8a-trimethyldecahydro-1,5-methanonaphthalen-2-ol"),
]
BARE_CLASS_TARGETS = [
    ("CC1=CC2CC1c1ccccc12", "2-methyl-1,4-dihydro-1,4-methanonaphthalene"),
    ("Cc1ccc2c(c1)C1CCC2C1", "6-methyl-1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
    ("OC(=O)C1CC2c3ccccc3C1c1ccccc12", "9,10-dihydro-9,10-ethanoanthracene-11-carboxylic acid"),
    ("C1=CC2CCC1c1ccccc12", "1,4-dihydro-1,4-ethanonaphthalene"),
    ("C1CC2CC1C1CCCCC12", "decahydro-1,4-methanonaphthalene"),
]
MORE_TARGETS = [
    ("C1CC23CCCCC2(CC1)CC3", "octahydro-4a,8a-ethanonaphthalene"),                 # total hydrogenation, (d) bridge
    ("CC1CC23CCCCC2(CC1)CC3", "2-methyloctahydro-4a,8a-ethanonaphthalene"),
    ("Cc1ccc2c3ccc(C=C3)c2c1", "6-methyl-1,4-ethenonaphthalene"),                   # all-aromatic routing
    ("Cc1ccc2c3ccc(o3)c2c1", "6-methyl-1,4-epoxynaphthalene"),
    ("C12=CC=C(C3=CC=4C5=CC=C(C4C=C13)C5)C2", "1,4:5,8-dimethanoanthracene"),     # identical bridges,
    ("C1CCc2c3ccc(C3)c2C1", "5,6,7,8-tetrahydro-1,4-methanonaphthalene"),          # (a) before hydro
    ("OC1CC2CC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-2-ol"),
    ("NC1CC2CC1c1ccccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-2-amine"),
    ("c1ccc(cc1)C1CC2CC1c1ccccc12", "2-phenyl-1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
]
ALL_TARGETS = BB_TARGETS + DEV_TARGETS + BARE_CLASS_TARGETS + MORE_TARGETS


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s1"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ALL_TARGETS)
def test_s1_target_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
