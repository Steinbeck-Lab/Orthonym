"""A lactam with a separate acid, ester or amide: the ring C=O is the prefix 'oxo'.

 "Cyclic anhydrides, esters and amides are named as pseudoketones; the resulting names are preferred
IUPAC names." (the Blue Book) and "Lactams and lactims" (:33218): "Method (1) generates
preferred IUPAC names", method (1) being "as heterocyclic pseudoketones". "SENIORITY ORDER FOR CLASSES"
Table 4.1 (:18189) puts "16 Ketones..., pseudoketones..." below the acids, esters, acid halides and amides.
 (:29477): "In cyclic pseudoketones, the prefix 'oxo' and/or acyl group prefixes are used to denote
a carbonyl group". The lactone handler already yields to a senior group on that ground,:32112:
"A lactone, as a pseudoketone, ranks lower in the seniority of classes than an acid or an ester"); the lactam
handler did not, so '5-oxopyrrolidine-2-carboxylic acid' was named '5-carboxypyrrolidin-2-one' and the PIN tier
declined every such molecule.

The book prints no lactam with a carboxylic acid; the names are what these rules construct, and the OPSIN
read-back (``opsin_gate``) confirms each to the full InChIKey.

A thioamide, a hydrazide, an amidine, a nitrile, an aldehyde and an amide of the lactam's own class are senior to the
pseudoketone as well (classes 11 to 15 of Table 4.1,:18162-18188). They were not seen as senior: the order of
``rules.seniority.SENIORITY_ORDER`` ranks a ring amide among the amides, so the principal group was the lactam and
'1-(2-hydrazinyl-2-oxoethyl)pyrrolidin-2-one' (a hydrazide demoted to a prefix) shipped as a PIN; the ring amides now
step aside for a group senior to the ketone (``seniority._head_principal_group``). Names such as
'2-(2-oxopyrrolidin-1-yl)acetohydrazide', '2-(2-oxopyrrolidin-1-yl)ethanethioamide' and '5-oxopyrrolidine-2-carbonitrile'
are read back by OPSIN to the full InChIKey.

Changed asserted values, none: every row is new.
"""
import pytest

import orthonym.assembly.handlers.lactone as lactone_handler
from orthonym import Orthonym, name_compound

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _tiered(smiles: str):
    row = Orthonym().name_tiered(smiles)
    return row["name"], row["tier"]


@pytest.mark.parametrize("smiles, expected", [
    # the acid is the principal group, the ring C=O is 'oxo'
    ("OC(=O)C1CCC(=O)N1", "5-oxopyrrolidine-2-carboxylic acid"),
    ("OC(=O)[C@@H]1CCC(=O)N1", "(2S)-5-oxopyrrolidine-2-carboxylic acid"),
    ("OC(=O)C1CC(=O)N1", "4-oxoazetidine-2-carboxylic acid"),
    ("OC(=O)C1CCCC(=O)N1", "6-oxopiperidine-2-carboxylic acid"),
    ("OC(=O)C1CCCCC(=O)N1", "7-oxoazepane-2-carboxylic acid"),
    ("O=C1NCCC1C(=O)O", "2-oxopyrrolidine-3-carboxylic acid"),
    ("OC(=O)C1CCC(=O)NC1", "6-oxopiperidine-3-carboxylic acid"),
    # the substituted lactam nitrogen
    ("OC(=O)C1CCC(=O)N1C", "1-methyl-5-oxopyrrolidine-2-carboxylic acid"),
    ("O=C1CCC(C(=O)O)N1C(=O)C", "1-acetyl-5-oxopyrrolidine-2-carboxylic acid"),
    # the acid on a chain: the chain is the parent and the lactam a substituent
    ("OC(=O)CN1CCCC1=O", "(2-oxopyrrolidin-1-yl)acetic acid"),
    ("OC(=O)CCC1NC(=O)CC1", "3-(5-oxopyrrolidin-2-yl)propanoic acid"),
    # the ester and the amide are senior to the pseudoketone as well
    ("O=C1CCC(C(=O)OC)N1", "methyl 5-oxopyrrolidine-2-carboxylate"),
    ("CCOCC(COC(=O)C1CCC(=O)N1)OCC", "2,3-diethoxypropyl 5-oxopyrrolidine-2-carboxylate"),
    ("NC(=O)C1CCC(=O)N1", "5-oxopyrrolidine-2-carboxamide"),
    ("O=C1CCCN1CC(=O)NC", "N-methyl-2-(2-oxopyrrolidin-1-yl)acetamide"),
])
def test_a_separate_senior_group_owns_the_suffix(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    # the lactam is the principal group: no group of the molecule is senior to it
    ("O=C1CCCN1", "pyrrolidin-2-one"),
    ("O=C1CCCCCN1", "azepan-2-one"),
    ("O=C1CCC(C)N1C", "1,5-dimethylpyrrolidin-2-one"),
    ("O=C1CCCN1C(=O)C", "1-acetylpyrrolidin-2-one"),
    # an alcohol and an amine rank below the pseudoketone, 17 and 19)
    ("O=C1CCC(CO)N1", "5-(hydroxymethyl)pyrrolidin-2-one"),
    ("O=C1CCC(N)N1", "5-aminopyrrolidin-2-one"),
])
def test_the_lactam_stays_the_parent_when_nothing_is_senior(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


@pytest.mark.parametrize("smiles, expected", [
    # classes 11 and 12: amides of every chalcogen, hydrazides
    ("NNC(=O)CN1CCCC1=O", "2-(2-oxopyrrolidin-1-yl)acetohydrazide"),
    ("NC(=S)CN1CCCC1=O", "2-(2-oxopyrrolidin-1-yl)ethanethioamide"),
    ("NC(=[Se])CN1CCCC1=O", "2-(2-oxopyrrolidin-1-yl)ethaneselenoamide"),
    ("CNC(=S)CN1CCCC1=O", "N-methyl-2-(2-oxopyrrolidin-1-yl)ethanethioamide"),
    ("NNC(=S)CN1CCCC1=O", "2-(2-oxopyrrolidin-1-yl)ethanethiohydrazide"),
    ("NNC(=O)C1CCC(=O)N1", "5-oxopyrrolidine-2-carbohydrazide"),
    ("NC(=S)C1CCC(=O)N1", "5-oxopyrrolidine-2-carbothioamide"),
    ("CNC(=O)CN1CCCC1=O", "N-methyl-2-(2-oxopyrrolidin-1-yl)acetamide"),
    ("CN(C)C(=O)CN1CCCC1=O", "N,N-dimethyl-2-(2-oxopyrrolidin-1-yl)acetamide"),
    ("O=C1CCC(C(=O)NC)N1", "N-methyl-5-oxopyrrolidine-2-carboxamide"),
    # the amidine, the sulfonamide, the nitrile (class 14: the numbering follows the nitrile)
    ("O=C1CCC(C(=N)N)N1", "5-oxopyrrolidine-2-carboximidamide"),
    ("O=C1CCC(S(N)(=O)=O)N1", "5-oxopyrrolidine-2-sulfonamide"),
    ("O=C1CCC(C#N)N1", "5-oxopyrrolidine-2-carbonitrile"),
    ("O=C1CC(C#N)CN1", "5-oxopyrrolidine-3-carbonitrile"),
    ("N#CC1CCCCC(=O)N1", "7-oxoazepane-2-carbonitrile"),
    ("O=C1CCCN1CCC#N", "3-(2-oxopyrrolidin-1-yl)propanenitrile"),
    # class 15: the aldehyde on the nitrogen's chain
    ("O=C1CCCN1CC=O", "(2-oxopyrrolidin-1-yl)acetaldehyde"),
])
def test_a_group_of_classes_11_to_15_owns_the_suffix(smiles, expected):
    assert _tiered(smiles) == (expected, "pin_verified")


def test_an_aldehyde_on_the_ring_is_not_a_false_pin():
    """'5-oxopyrrolidine-2-carbaldehyde' is the name; the strict path builds 'pyrrolidine-2,5-dicarbaldehyde' for it
    (the lactam C=O read as a second aldehyde), which the read-back refuses, so only the best-effort tier names it."""
    name, tier = _tiered("O=C1CCC(C=O)N1")
    assert tier != "pin_verified", name
    assert name_compound("O=C1CCC(C=O)N1") == "5-oxopyrrolidine-2-carbaldehyde"


def test_the_lactone_guard_is_unchanged():
    """The lactone handler shares the senior-group scan: its rows keep their names."""
    assert _tiered("OC(=O)C1CCC(=O)O1") == ("5-oxooxolane-2-carboxylic acid", "pin_verified")
    assert _tiered("O=C1CCCO1") == ("oxolan-2-one", "pin_verified")


def test_the_tests_fail_without_the_guard(monkeypatch):
    """The mutation check: with the scan switched off the lactam parent takes the acid as 'carboxy' again."""
    monkeypatch.setattr(lactone_handler, "has_senior_group_outside", lambda *a, **k: False)
    assert _tiered("OC(=O)C1CCC(=O)N1")[1] == "abstain"
