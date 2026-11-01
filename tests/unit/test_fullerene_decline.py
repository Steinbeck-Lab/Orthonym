"""Fullerene cages are declined at once, with the limit code FULLERENE.

A fullerene row used to cost 11-18 s (and often a timeout) before the same decline. The
detector is perception.fullerene (12 pentagons, V/2 - 10 hexagons, every cage atom with
three cage neighbours, at least 60 atoms); the C20 dodecahedrane cage is not a fullerene
(Blue Book and keeps the von Baeyer name best-effort gives it.

Every cage is declined at the PIN, valid and complete tiers. At the best-effort tier the C60
skeleton with at most one addend on a [6,6] bond is named (its own tests: tests/unit/rules/
test_fullerene_c60_names.py); every other cage is declined there as well.
"""
from __future__ import annotations

import time

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import (LIMIT_CATALOG, OrthonymLimitError, classify_failure_limit,
                             is_failure_name)
from orthonym.perception.fullerene import fullerene_cages

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

FULLERENES = {
    "C60": 'c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41',
    "C70": 'c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c%12c%13c%14c%15c%16c%17c%14c%14c%18c%13c%11c1c1c5c%10c5c(c%14c%10c%17c%11c%13c%16c%14c%16c%15c%12c%12c%16c(c2c7c%126)c2c8c3c(c%13c%142)c2c9c4c5c%10c%112)c%181',
    "C59N cage (heterofullerene)": 'C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12[N+]5=C%11C4=C3C3=C5C(=C81)C%10=C23',
    "C60 dihydro": 'C12C3C4C5C6C7C8C9C%10C%11C%12C%13C%14=C%10C8=C8C%10C7=C5C1C1=C5C2C2C7C%15C5C5=C%16C%15=C%15C%17=C%18C(=C%12C%12=C%11C%11=C%19C(=C4C4=C%20C%19=C%12C%18=C%20C(=C2C34)C7%17)C6=C9%11)C%15=C%13C%16=C%14C8C5C%101',
    "cyclopropa adduct of C60": 'CCOC(=O)C1(C23C14C5=C6C7=C8C9=C1C%10=C%11C%12=C%13C%14=C%10C%10=C1C1=C%15C%16=C%17C%18=C%19C%20=C%21C%22=C%23C%24=C%25C%26=C(C7=C9C%11=C%26C%12=C%24C%22=C%13C%20=C%14C%18=C%10%16)C7=C%25C9=C(C4=C76)C4=C2C(=C%17C3=C%15C5=C81)C%19=C%21C4=C%239)C=C',
    "C70 adduct": 'C12C3C4C5=C6C1C7(C8=C9C2(C1=C2C3(C3=C%10C4(C4=C%11C%12(C%10=C%10C%13=C3C2=C2C3=C%13C%13=C%14C%15=C%16C%17=C%18C(=C3%15)C2(C1C9C%18(C1=C8C2=C3C(=C%171)C1=C%16C%14(C8C(C%13%10C(F)(F)F)C%12C9=C%10C8C1(C1C%10C8(C(C9%11C(F)(F)F)C(C54C(F)(F)F)C4=C6C7=C2C(=C48)C13C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F)C(F)(F)F',
    "multi-cage": 'C1C2=C(CC3=C1CC45C6=C7C8=C9C1=C%10C%11=C%12C%13=C%14C%15=C%11C1=C1C%11=C%16C%17=C%18C(=C%15%11)C%14=C%11C%14=C%13C%13=C%12C%12=C%10C8=C8C%12=C%10C%13=C%12C%14=C%13C%11=C%18C%11=C%17C%14=C(C4(C3)C3=C%14C%16=C1C9=C73)C1=C%11C%13=C3C%12=C%10C(=C86)C5=C31)CC1=C(C2)CC23C4=C5C6=C7C8=C9C%10=C%11C%12=C%13C%14=C%10C8=C8C%10=C%15C%16=C%17C(=C%14%10)C%13=C%10C%13=C%12C%12=C%11C%11=C9C6=C6C%11=C9C%12=C%11C%13=C%12C%10=C%17C%10=C%16C%13=C(C2(C1)C1=C%13C%15=C8C7=C51)C1=C%10C%12=C2C%11=C9C(=C64)C3=C21',
}
#: the C60 skeletons with at most one addend: named at the best-effort tier only
NAMED_AT_BEST_EFFORT = ("C60", "cyclopropa adduct of C60")
# 1 s of warm-up work is not charged: the bound is generous (the decline takes well under 1 s)
LIMIT_S = 5.0
TIERS = ("pin", "valid", "complete", "best-effort")


def _declined_at(label, tier):
    return not (label in NAMED_AT_BEST_EFFORT and tier == "best-effort")

# Names recorded at HEAD before the early decline existed.
CONTROLS_BOTH_TIERS = {
    "coronene": ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "coronene"),
    "benzo[ghi]perylene": ("c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56", "benzo[ghi]perylene"),
    "adamantane": ("C1C2CC3CC1CC(C2)C3", "adamantane"),
    "cubane": ("C12C3C4C1C5C2C3C45", "cubane"),
}
CONTROLS_BEST_EFFORT = {
    "corannulene": ("c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51", 'hexacyclo[11.5.2.0^4,17.0^7,16.0^10,15.0^14,18]icosa-1(19),2,4(17),5,7,9,11,13(20),14(18),15-decaene'),
    "sumanene": ("c1cc2c3c4c1Cc1ccc5c(c14)c1c(ccc(c31)C2)C5", 'heptacyclo[12.6.1.0^4,19.0^6,18.0^9,17.0^11,16.0^15,20]henicosa-1(20),2,4(19),6(18),7,9(17),11(16),12,14-nonaene'),
    "dodecahedrane": ("C12C3C4C5C1C1C6C2C2C3C3C4C4C5C1C1C6C2C3C41", 'undecacyclo[9.9.0.0^2,9.0^3,7.0^4,20.0^5,18.0^6,16.0^8,15.0^10,14.0^12,19.0^13,17]icosane'),
}


def test_limit_code_is_catalogued_and_is_a_failure_label():
    assert "FULLERENE" in LIMIT_CATALOG
    assert is_failure_name(LIMIT_CATALOG["FULLERENE"]["message"])


@pytest.mark.parametrize("label", sorted(FULLERENES))
@pytest.mark.parametrize("tier", TIERS)
def test_fullerene_declines_fast_with_its_code(label, tier):
    if not _declined_at(label, tier):
        pytest.skip("named at the best-effort tier")
    smiles = FULLERENES[label]
    start = time.time()
    with pytest.raises(OrthonymLimitError) as err:
        name_compound(smiles, emit_tier=tier, raise_on_limit=True)
    assert time.time() - start < LIMIT_S
    assert err.value.code == "FULLERENE"
    label_text = name_compound(smiles, emit_tier=tier, on_failure="label")
    assert is_failure_name(label_text)


@pytest.mark.parametrize("label", sorted(FULLERENES))
def test_detector_and_failure_classifier_agree(label):
    mol = Chem.MolFromSmiles(FULLERENES[label])
    assert fullerene_cages(mol)
    assert classify_failure_limit(mol).code == "FULLERENE"


@pytest.mark.parametrize("label", sorted(CONTROLS_BOTH_TIERS))
@pytest.mark.parametrize("tier", TIERS)
def test_non_fullerene_controls_keep_their_names(label, tier):
    smiles, expected = CONTROLS_BOTH_TIERS[label]
    assert name_compound(smiles, emit_tier=tier) == expected


@pytest.mark.parametrize("label", sorted(CONTROLS_BEST_EFFORT))
def test_non_fullerene_cages_keep_their_best_effort_names(label):
    smiles, expected = CONTROLS_BEST_EFFORT[label]
    assert not fullerene_cages(Chem.MolFromSmiles(smiles))
    assert name_compound(smiles, emit_tier="best-effort") == expected
