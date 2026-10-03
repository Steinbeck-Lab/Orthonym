"""A bridged fused ring system as a substituent prefix (S1 controller decision 4).

 (the Blue Book): free-valence locants are "as low as is consistent with any
established numbering of the parent hydride"; (:17326): "low locants go first to
the fixed numbering of the system, then indicated hydrogen, followed by free valence
suffix, and finally 'hydro' prefixes". The Blue Book prints no bridged fused '-yl' prefix,
so the bridge locants are fixed first and the free valence takes the place of a suffix in
 (c) (:3256). Both spellings of each pair below read back to the input's full
InChIKey with OPSIN 2.9.0: only these tests hold the spelling."""
import logging

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import build_substituent
from orthonym.rules.bridged_fused_pin.selection import ring_system
from orthonym.rules.ring_substituents import (
    _bridged_fused_substituent_name,
    _extract_ring_submol,
    _polycyclic_substituent_name,
)
from tests.support.rt_assert import name_is_rt_exact


def _split(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = tuple(sorted(ring_system(mol)))
    attach = next(a for a in ring for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                  if nb.GetIdx() not in ring)
    return mol, ring, attach


def _prefix(smiles):
    mol, ring, attach = _split(smiles)
    sub, attach_sub = _extract_ring_submol(mol, ring, attach)
    return build_substituent(sub, attach_sub)


@pytest.mark.parametrize("smiles,prefix", [
    ("C12CCC(C3=CC=CC=C13)C2CCO", "1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-yl"),
    # the bridge locants are fixed first: '...-5-yl', not '5,8-dihydro-5,8-methano...-1-yl'
    ("CC(=O)Nc1cccc2c1C1C=CC2C1", "1,4-dihydro-1,4-methanonaphthalen-5-yl"),
    ("C12C(=CC(C3=CC=CC=C13)C2)CC(=O)O", "1,4-dihydro-1,4-methanonaphthalen-2-yl"),
    ("OC(c1ccc(F)cc1)(c1ccc(F)cc1)C1CC2c3ccccc3C1c1ccccc12",
     "9,10-dihydro-9,10-ethanoanthracen-11-yl"),
])
def test_the_bridged_prefix(smiles, prefix):
    assert _prefix(smiles) == prefix


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    ("C12CCC(C3=CC=CC=C13)C2CCO", "2-(1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-yl)ethan-1-ol"),
    ("CC(=O)Nc1cccc2c1C1C=CC2C1", "N-(1,4-dihydro-1,4-methanonaphthalen-5-yl)acetamide"),
    ("C12C(=CC(C3=CC=CC=C13)C2)CC(=O)O", "(1,4-dihydro-1,4-methanonaphthalen-2-yl)acetic acid"),
    ("OC(c1ccc(F)cc1)(c1ccc(F)cc1)C1CC2c3ccccc3C1c1ccccc12",
     "(9,10-dihydro-9,10-ethanoanthracen-11-yl)bis(4-fluorophenyl)methanol"),
    # amide N-substituents on two more S2 parents (a heterocycle and indene): the
    # bond split hands the ring system to the substituent namer (LEFTOVERS B4)
    ("CC(=O)Nc1cccc2c1C1C=CC2COC1", "N-(1,2,4,5-tetrahydro-1,5-etheno-3-benzoxepin-6-yl)acetamide"),
    ("CC(=O)NC1C=CC2C3C=CC(C3)C12", "N-(3a,4,7,7a-tetrahydro-1H-4,7-methanoinden-1-yl)acetamide"),
])
def test_the_whole_name_is_the_verified_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    # the free valence is on a ring that is one ring of a polycyclic system: the amide
    # N-substituent enricher read the system's other ring atoms as substituents of that
    # ring ('2-phenyl1,2,3,4-tetrahydro-1,4-ethanonaphthalen-2-yl', '4-ethan-2-ylbicyclo
    # [2.2.1]heptan-2-yl': different molecules, suppressed, the PIN tier abstained)
    ("CC(=O)NC1CC2CCC1c1ccccc12", "N-(1,2,3,4-tetrahydro-1,4-ethanonaphthalen-2-yl)acetamide"),
    ("CC(=O)NC1CC2CC1c1ccccc12", "N-(1,2,3,4-tetrahydro-1,4-methanonaphthalen-2-yl)acetamide"),
    ("CC(=O)NC1CC2CCC1C2", "N-(bicyclo[2.2.1]heptan-2-yl)acetamide"),
    ("CC(=O)NC1CCC2(CC1)CCCC2", "N-(spiro[4.5]decan-8-yl)acetamide"),
    ("CC(=O)NC1CCCC2CCCCC12", "N-(decahydronaphthalen-1-yl)acetamide"),
])
def test_an_amide_n_substituent_ring_system_is_named_whole(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_a_tautomer_sensitive_hetero_parent_n_substituent_is_declined():
    # N-acetyl on C3 of 4,5,6,7-tetrahydro-1H-4,7-methanoindazole: the indicated hydrogen
    # is on a ring N (spec section 8), so build_substituent declines; the PIN tier
    # abstains (as at the S2 base) and best-effort keeps an RT-exact name
    smiles = "CC(=O)Nc1n[nH]c2c1C1CCC2C1"
    assert _prefix(smiles) is None
    assert _row(smiles, "pin")["tier"] == "abstain"
    row = _row(smiles, "best-effort")
    assert row["tier"] != "abstain" and name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles", [
    "c1ccc2ccccc2c1",                        # a fused ring system: not bridged
    "C1CC2CCC1C2",                           # norbornane: von Baeyer
])
def test_build_substituent_declines_outside_its_class(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert build_substituent(mol, 0) is None


def test_a_stereocentre_declines():
    sub = Chem.MolFromSmiles("C1C[C@H]2C[C@@H]1c1ccccc12")
    attach = next(a.GetIdx() for a in sub.GetAtoms() if a.GetTotalNumHs() == 2 and a.GetDegree() == 2)
    assert build_substituent(sub, attach) is None


def test_an_error_inside_the_substituent_builder_declines_and_is_logged(monkeypatch, caplog):
    # the ring-substituent namer reaches the builder through build_substituent: an internal
    # error declines (the namer then tries its next namer, as for any decline) and is logged
    # at ERROR level with its traceback, as on the parent route (test_bf_s2_entry.py); a
    # deliberate refusal (OrthonymLimitError) still propagates out of the entry
    import orthonym.rules.bridged_fused_pin as pkg
    from orthonym.errors import unsupported_ring_system
    caplog.set_level(logging.DEBUG, logger="orthonym.rules.bridged_fused")

    def errors():
        return [r for r in caplog.records
                if r.name == "orthonym.rules.bridged_fused" and r.levelno >= logging.ERROR
                and r.exc_info and "bridged fused PIN builder raised" in r.getMessage()]

    mol, ring, attach = _split("C12CCC(C3=CC=CC=C13)C2CCO")
    sub, attach_sub = _extract_ring_submol(mol, ring, attach)
    expected = "1,2,3,4-tetrahydro-1,4-methanonaphthalen-9-yl"
    assert _bridged_fused_substituent_name(sub, attach_sub) == expected
    assert _polycyclic_substituent_name(mol, ring, attach) == expected
    assert errors() == []

    def raising(exc):
        def build_substituent(mol, attach):
            raise exc
        return build_substituent

    monkeypatch.setattr(pkg, "build_substituent", raising(RuntimeError("a defect inside the builder")))
    assert _bridged_fused_substituent_name(sub, attach_sub) is None
    assert _polycyclic_substituent_name(mol, ring, attach) != expected
    logged = errors()
    assert len(logged) == 2 and all(r.exc_info[0] is RuntimeError for r in logged), logged
    caplog.clear()
    monkeypatch.setattr(pkg, "build_substituent", raising(unsupported_ring_system()))
    with pytest.raises(type(unsupported_ring_system())):
        _bridged_fused_substituent_name(sub, attach_sub)
    assert errors() == []
