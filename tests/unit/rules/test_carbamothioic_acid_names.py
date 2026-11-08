"""The free N-substituted thio acids of carbamic acid (``assembly.carbamothioic_acid_names``).

 (the Blue Book): functional replacement is applied to the retained name 'carbamic acid', and the tautomers
take an italic element locant: 'carbamothioic O-acid (PIN)' (:30774), 'carbamothioic S-acid (PIN)' (:30826); the
N-substituents are prefixes with no italic N, as in 'dimethylcarbamic acid (PIN)':30762) and
'diethylcarbamothioic di(propan-2-yl)stibinous thioanhydride (PIN)' (:35996). "Substitution of formic acid is not
recommended":30756): the rows were '(diethylamino)methane(dithioic acid)' at best-effort and no name at the PIN
tier.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("CCN(CC)C(=S)S", "diethylcarbamodithioic acid"),
    ("CCN(C)C(=S)S", "ethyl(methyl)carbamodithioic acid"),
    ("c1ccccc1NC(=S)S", "phenylcarbamodithioic acid"),
    ("OCCN(CCO)C(=S)S", "bis(2-hydroxyethyl)carbamodithioic acid"),
    ("NCCNC(=S)S", "(2-aminoethyl)carbamodithioic acid"),
])
def test_the_dithioic_acid_is_the_pin_at_every_tier(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


@pytest.mark.parametrize("smiles,pin,readable", [
    ("CNC(=S)O", "methylcarbamothioic O-acid", "methylcarbamothioic acid"),
    ("CNC(=O)S", "methylcarbamothioic S-acid", "methylcarbamothioic acid"),
    ("Cc1ccc(NC(=S)O)cc1", "(4-methylphenyl)carbamothioic O-acid", "(4-methylphenyl)carbamothioic acid"),
])
def test_a_mixed_acid_carries_its_locant_at_the_pin_tier(smiles, pin, readable):
    # OPSIN 2.9.0 reads no 'O-acid' word: best-effort ships the readable word below the PIN (user decision 2026-10-22)
    assert (tier_row(smiles, "pin")["name"], tier_row(smiles, "pin")["tier"]) == (pin, "pin_verified")
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (readable, "systematic_verified"), row


def test_a_ring_nitrogen_is_no_carbamic_acid():
    # piperidine-1-carbodithioic acid is a ring carboxylic analogue, not carbamic acid: this producer leaves it alone
    assert "carbam" not in tier_row("C1CCN(CC1)C(=S)S", "best-effort")["name"]


@pytest.mark.parametrize("smiles", [
    "SC(=O)NCCC(=S)S",          # a carbodithioic acid class 7a,:18172) outranks the carbamic one (7b,:18174)
    "OS(=O)(=O)CCNC(=S)S",      # beside a sulfonic acid this module ranks nothing
])
def test_beside_another_acid_no_carbamic_parent_is_claimed(smiles):
    row = tier_row(smiles, "pin")
    assert "carbam" not in row["name"] and row["tier"] != "pin_verified", row


def test_a_multiplied_substituent_keeps_its_multiplier_outside_the_marks():
    # (the Blue Book) "the multiplicative prefixes are not included in the parentheses";
    # 'diethylcarbamothioic di(propan-2-yl)stibinous thioanhydride (PIN)' (:35996); was '[di(propan-2-yl)]carbamic acid'
    for smiles, name in (("CC(C)N(C(C)C)C(=S)S", "di(propan-2-yl)carbamodithioic acid"),
                         ("CC(C)N(C(C)C)C(=O)O", "di(propan-2-yl)carbamic acid")):
        assert (tier_row(smiles, "pin")["name"], tier_row(smiles, "pin")["tier"]) == (name, "pin_verified")
