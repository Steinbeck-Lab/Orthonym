"""Entry points of slice S2: every bridged fused ring system goes to the bridged fused PIN
builder, and the all-aromatic routing exemption asks the builder too.

The older bare-system excisions behind ``name_bridged_fused_pin`` named a naphthalene
parent wherever one existed, missing (b) (the Blue Book, "include the
maximum number of skeletal atoms"; ex.:14277): they shipped
'1,2,3,4-tetrahydro-1,4-propanonaphthalene' pin_verified. The builder reproduces every
other name they built on the 1345 bare ring systems of the S2 census, sample and Blue Book
corpora (S2 planning notes, the differential run); those 47 names are locked here."""
import logging

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused import has_aromatic_mancude_bridge, name_bridged_fused_pin
from tests.support.rt_assert import name_is_rt_exact

#: bare ring systems the retired excisions named; the builder gives the same name
REPRODUCED = [
    ("C1CC2CC1=C1C3CCC(C3)C12", "1,2,3,4,4a,5,6,7-octahydro-1,4:5,8-dimethanonaphthalene"),
    ("C1=CCC2C3CCC(CC3)C2C1", "1,2,3,4,4a,5,8,8a-octahydro-1,4-ethanonaphthalene"),
    ("C1=CC2CC1C1C3CCC(C3)C21", "1,2,3,4,4a,5,8,8a-octahydro-1,4:5,8-dimethanonaphthalene"),
    ("C1=C2OC3CCCC2C3CC1", "1,2,3,4,4a,7,8,8a-octahydro-1,5-epoxynaphthalene"),
    ("C1=CC2C(C=C1)C1C=CC2C2CCCCC12", "1,2,3,4,4a,8a,9,9a,10,10a-decahydro-9,10-ethenoanthracene"),
    ("C1=CC2C3CCC(C3)C2C=C1", "1,2,3,4,4a,8a-hexahydro-1,4-methanonaphthalene"),
    ("c1ccc2c(c1)CC1CCC3CC1C2O3", "1,2,3,4,4a,9,9a,10-octahydro-2,9-epoxyanthracene"),
    ("c1ccc2c(c1)C1CCC2C2CCCCC12", "1,2,3,4,4a,9,9a,10-octahydro-9,10-ethanoanthracene"),
    ("C1CCC2=C(C1)C1CCC2O1", "1,2,3,4,5,6,7,8-octahydro-1,4-epoxynaphthalene"),
    ("C1CC2CC1C1=C2C2CCC1C2", "1,2,3,4,5,6,7,8-octahydro-1,4:5,8-dimethanonaphthalene"),
    ("c1ccc2c(c1)CC13CCCCC1(C2)O3", "1,2,3,4,9,10-hexahydro-4a,9a-epoxyanthracene"),
    ("c1ccc2c(c1)C1CCC2CC1", "1,2,3,4-tetrahydro-1,4-ethanonaphthalene"),
    ("c1ccc2c(c1)C1CCC2C1", "1,2,3,4-tetrahydro-1,4-methanonaphthalene"),
    ("C1=CC2=CCC3CCC(C1)C2C3", "1,2,6,7,8,8a-hexahydro-1,7-ethanonaphthalene"),
    ("C1=CC23CCC(CC2)CC3CC1", "1,3,4,7,8,8a-hexahydro-2H-2,4a-ethanonaphthalene"),
    ("c1ccc2c3cc(cc2c1)C3", "1,3-methanonaphthalene"),
    ("C1=CC2CCC1C1CCCCC21", "1,4,4a,5,6,7,8,8a-octahydro-1,4-ethanonaphthalene"),
    ("C1=CC2CCC1C1=CCCCC12", "1,4,4a,5,6,7-hexahydro-1,4-ethanonaphthalene"),
    ("C1=CC2C3C=CC(CC3)C2CC1", "1,4,4a,5,6,8a-hexahydro-1,4-ethanonaphthalene"),
    ("C1=CC2C3C=CC(C3)C2CC1", "1,4,4a,5,6,8a-hexahydro-1,4-methanonaphthalene"),
    ("C1=CCC2C3C=CC(C3)C2C1", "1,4,4a,5,8,8a-hexahydro-1,4-methanonaphthalene"),
    ("C1=CC2CC1C1Cc3ccccc3CC21", "1,4,4a,9,9a,10-hexahydro-1,4-methanoanthracene"),
    ("C1=CC2CCC1c1cc3ccccc3cc12", "1,4-dihydro-1,4-ethanoanthracene"),
    ("C1=CC2CCC1c1ccccc12", "1,4-dihydro-1,4-ethanonaphthalene"),
    ("C1=CC23CCC2(C=C1)c1ccc3o1", "1,4-epoxy-4a,8a-ethanonaphthalene"),
    ("C1=C2CC(=C1)c1c2c2ccc1o2", "1,4-epoxy-5,8-methanonaphthalene"),
    ("c1ccc2c3ccc(o3)c2c1", "1,4-epoxynaphthalene"),
    ("C1=C2CC(=C1)c1cc3c4ccc(c3cc12)CC4", "1,4-ethano-5,8-methanoanthracene"),
    ("c1ccc2c3ccc(c2c1)CC3", "1,4-ethanonaphthalene"),
    ("C1=C2CC(=C1)c1ccccc12", "1,4-methanonaphthalene"),
    ("C1=CC23CCCC=C2CC1CC3", "1,5,6,7-tetrahydro-2H-2,4a-ethanonaphthalene"),
    ("C1=CCC23C=CC(CC2=C1)C3", "1,5-dihydro-2H-2,4a-methanonaphthalene"),
    ("C1=CC23C=CC=CC2(C=C1)CC3", "4a,8a-ethanonaphthalene"),
    ("c1c2cc3cc4c5ccc(o5)c4cc3c1C2", "5,8-epoxy-1,3-methanoanthracene"),
    ("c1ccc2c(c1)C1OC2c2ccccc21", "9,10-dihydro-9,10-epoxyanthracene"),
    ("c1ccc2c(c1)C1CCC2c2ccccc21", "9,10-dihydro-9,10-ethanoanthracene"),
    ("c1ccc2c3c4ccccc4c(c2c1)CC3", "9,10-ethanoanthracene"),
    ("c1ccc2c(c1)C1CCN2c2ccccc21", "9H-9,10-ethanoacridine"),
    ("C1CCC2C3CCC(C3)C2C1", "decahydro-1,4-methanonaphthalene"),
    ("C1CC2OC1C1C3CCC(O3)C21", "decahydro-1,4:5,8-diepoxynaphthalene"),
    ("C1CC2CC1C1C3CCC(C3)C21", "decahydro-1,4:5,8-dimethanonaphthalene"),
    ("C1CC2CC3CCCC2C3C1", "decahydro-1,5-methanonaphthalene"),
    ("C1CC2CC3CCC2C(C1)N3", "decahydro-1,6-epiminonaphthalene"),
    ("C1CC2CCC3CC2C(C1)O3", "decahydro-1,7-epoxynaphthalene"),
    ("C1C2CC3C(CC4CC3C1N4)N2", "decahydro-1,7:3,5-diepiminonaphthalene"),
    ("C1CC2CCC3CC1CC2C3", "decahydro-2,7-methanonaphthalene"),
    ("C1CCC23CCC(CC2C1)C3", "octahydro-2H-2,4a-methanonaphthalene"),
]

#: (b): the larger parent (benzo[7]/[8]annulene) beats naphthalene + a longer
#: bridge; OPSIN 2.9.0 reads both spellings of each row to the same full InChIKey
CORRECTED = [
    ("C1CC2CCCC1c1ccccc12", "6,7,8,9-tetrahydro-5H-5,9-ethanobenzo[7]annulene"),
    ("C1=CC2CCCC1c1ccccc12", "6,7,8,9-tetrahydro-5H-5,9-ethenobenzo[7]annulene"),
    ("c1ccc2c(c1)CC1CCCC2C1", "5,6,7,8,9,10-hexahydro-5,9-methanobenzo[8]annulene"),
    ("C12=CC=C(C3=CC=CC=C13)C=CC=C2", "5,10-ethenobenzo[8]annulene"),   # not 1,4-buta[1,3]dieno...
]


@pytest.mark.parametrize("smiles,name", REPRODUCED + CORRECTED)
def test_name_bridged_fused_pin_is_the_builder(smiles, name):
    res = name_bridged_fused_pin(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC2=CC3=CC=C(C=C3)C2=C1", True),     # 4,7-ethenoazulene (all atoms RDKit-aromatic)
    ("c1cc2c3ccc(C=C3)c2nc1", True),          # 5,8-ethenoquinoline
    ("c1cc2ccc3cccc4ccc(c1)c2c34", False),    # pyrene: a fused ring system
    ("c1ccc2ccccc2c1", False),
])
def test_the_all_aromatic_exemption_asks_the_builder(smiles, expected):
    assert has_aromatic_mancude_bridge(Chem.MolFromSmiles(smiles)) is expected


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", CORRECTED + [
    ("C1=CC2=CC3=CC=C(C=C3)C2=C1", "4,7-ethenoazulene"),
    ("c1cc2c3ccc(C=C3)c2nc1", "5,8-ethenoquinoline"),
])
def test_engine_names(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)


def _raises(exc):
    def build(mol):
        raise exc
    return build


def _builder_errors(caplog):
    """The ERROR records with a traceback that the bridged fused entry points logged."""
    return [r for r in caplog.records
            if r.name == "orthonym.rules.bridged_fused" and r.levelno >= logging.ERROR
            and r.exc_info and "bridged fused PIN builder raised" in r.getMessage()]


def test_an_error_inside_the_builder_declines_and_does_not_reach_routing(monkeypatch, caplog):
    # name_bridged_fused_pin and the all-aromatic routing predicate both call the builder:
    # an internal error declines (no name, never a different one) and is logged at ERROR
    # level with its traceback (the eval harness workers switch off logging up to WARNING,
    # eval/harness.py), so a defect does not look like an ordinary decline; a deliberate
    # refusal (OrthonymLimitError) still propagates
    import orthonym.rules.bridged_fused_pin as pkg
    from orthonym.errors import unsupported_ring_system
    caplog.set_level(logging.DEBUG, logger="orthonym.rules.bridged_fused")
    mol = Chem.MolFromSmiles("C1=CC2=CC3=CC=C(C=C3)C2=C1")       # 4,7-ethenoazulene
    assert name_bridged_fused_pin(mol)[0] == "4,7-ethenoazulene"
    assert has_aromatic_mancude_bridge(mol) is True
    assert _builder_errors(caplog) == []
    monkeypatch.setattr(pkg, "build", _raises(RuntimeError("a defect inside the builder")))
    assert name_bridged_fused_pin(mol) is None
    assert has_aromatic_mancude_bridge(mol) is False
    logged = _builder_errors(caplog)
    assert len(logged) == 2 and all(r.exc_info[0] is RuntimeError for r in logged), logged
    caplog.clear()
    monkeypatch.setattr(pkg, "build", _raises(unsupported_ring_system()))
    with pytest.raises(type(unsupported_ring_system())):
        name_bridged_fused_pin(mol)
    assert _builder_errors(caplog) == []
