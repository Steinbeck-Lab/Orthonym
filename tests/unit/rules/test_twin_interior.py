"""Fused parents the Blue Book prints with an interior-atom locant, at the default tier.

 Numbering of interior carbon atoms (the Blue Book), (:12658): the locant of
an interior atom is that of the peripheral atom it is nearest to with a superscript number of bonds; the
interior atom of phenalene is '3a^1'. (heading LIMITATIONS OF FUSION NOMENCLATURE,
:14456; the rule:14470): the heteroatoms of such a system are named by skeletal replacement on the
hydrocarbon, whose numbering "is not altered by the 'a' prefixes"; '1,3a^1,4,9-tetraazaphenalene (PIN)'
(:14491). (:23474): '1,3a^1,4-triazaphenalene (PIN)' (:23486). (:25319, heading
 Seniority of heteroatoms over suffixes:25297): '1,3a^1,4,9-tetraazaphenalene-3-carboxylic acid
(PIN)' (:25350).

OPSIN 2.9.0 has no grammar for the superscript locant, so the default tier ships the name through its
twin spelling with the old locant ('9b',: "The previous rule... recommended that interior
atoms follow the highest numbered peripheral atom"); a wider tier keeps the von Baeyer name it has.
"""
import re

import pytest

from orthonym.data.fused_interior_pins import INTERIOR_PIN_PARENTS
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _canon, _independent_parse, name_best_effort

pytestmark = pytest.mark.opsin_gate

TETRAAZA = "N1=CC=C2N=CC=C3C=CN=C1N23"
TRIAZA = "N1=CC=C2N=CC=C3C=CC=C1N23"
ACID = "N1=CC(=C2N=CC=C3C=CN=C1N23)C(=O)O"
PRINTED_ROWS = [
    (TETRAAZA, "1,3a^1,4,9-tetraazaphenalene", "1,9b,4,9-tetraazaphenalene"),                       #:14491
    (ACID, "1,3a^1,4,9-tetraazaphenalene-3-carboxylic acid", "1,9b,4,9-tetraazaphenalene-3-carboxylic acid"),   #:25350
    (TRIAZA, "1,3a^1,4-triazaphenalene", "1,9b,4-triazaphenalene"),                                 #:23486
]


@pytest.mark.parametrize("smiles,name,twin", PRINTED_ROWS)
def test_default_tier_gives_the_printed_name(smiles, name, twin):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin" and row["gate_outcome"] == "twin_verified:interior_phenalene", row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles,name,twin", PRINTED_ROWS)
def test_the_twin_of_the_printed_name_is_the_structure_of_the_input(smiles, name, twin):
    from orthonym.validation.twin_spellings import twin_of
    assert twin_of(name) == (twin, ("interior_phenalene",))
    assert _independent_parse(name, smiles) is None
    assert _canon(_independent_parse(twin, smiles)) == _canon(smiles)


def test_best_effort_keeps_the_von_baeyer_label():
    # the printed parents are not read at a wider tier: the name keeps its von Baeyer form and its label
    for smiles in (TETRAAZA, ACID):
        row = name_best_effort(smiles)
        assert row["name"].startswith("2,6,12,13-tetraazatricyclo[7.3.1.0^5,13]") and row["tier"] == "systematic_verified", row
        assert row["is_pin"] is False, row


def test_interior_locant_string_order():
    """ Lowest set of locants (:3189): "3a is lower than 3a^1" (:3213); the book prints the set
    '1,3a^1,4,9' (:14491): a numeral, then the letter, then the superscript, in ascending order."""
    def key(loc):
        m = re.fullmatch(r"(\d+)([a-z]?)(?:\^(\d+))?", loc)
        return int(m[1]), m[2], int(m[3] or 0)
    for smiles, name, _twin in PRINTED_ROWS:
        locs = re.match(r"([\d,a^]+)-", default_tier_row(smiles)["name"])[1].split(",")
        assert locs == sorted(locs, key=key), name
    assert key("3a") < key("3a^1") < key("4")


def test_a_substituent_on_the_printed_parent_takes_the_entry_numbering():
    # 3-carboxylic acid: the parent has no symmetry that keeps its heteroatom locants, so the numbering is
    # the entry's; a methyl on the same atom is a prefix of the same parent
    smiles = "N1=CC(C)=C2N=CC=C3C=CN=C1N23"
    row = default_tier_row(smiles)
    assert row["name"] == "3-methyl-1,3a^1,4,9-tetraazaphenalene", row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin_twin", row


def test_other_interior_systems_keep_their_behaviour():
    # two skeletons the table does not hold (a single interior nitrogen: fusion nomenclature names those,
    #: the default tier declines and the best-effort tier gives the von Baeyer name
    for smiles, be in (("C1=CC2=CC=CC3=CC=CC(=C1)N23", "13-azatricyclo[7.3.1.0^5,13]trideca-1(12),2,4,6,8,10-hexaene"),
                       ("C1=CC2=CC=CN3C=CC=C(C1)C23", "1-azatricyclo[7.3.1.0^5,13]trideca-2,4,6,9,11-pentaene")):
        assert default_tier_row(smiles)["limit_code"] == "NO_VERIFIED_PIN"
        assert name_best_effort(smiles)["name"] == be


def test_a_fused_parent_of_the_shared_catalogue_is_unchanged():
    assert default_tier_row("C1=Cc2cccc3cccc(c23)C1")["name"] == "1H-phenalene"
    assert default_tier_row("O=C1C=Cc2cccc3cccc1c23")["name"] == "1H-phenalen-1-one"


@pytest.mark.parametrize("smiles,entry", sorted(INTERIOR_PIN_PARENTS.items()))
def test_the_entry_numbers_the_atoms_as_the_twin_does(smiles, entry):
    """The entry writes its SMILES in the order of its locants; OPSIN's reading of the twin gives the same
    atom order and the same locant on every atom (the interior atom is '9b' there, '3a^1' in the book)."""
    import subprocess

    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    from orthonym.validation.twin_spellings import twin_of
    twin = twin_of(entry["name"])[0]
    proc = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", _find_opsin_jar("2.9.0"), "-o", "extendedsmi"],
                          input=twin + "\n", capture_output=True, text=True, timeout=120)
    smi, labels = proc.stdout.strip().split(" |$_AV:")
    assert smi == smiles
    assert labels.rstrip("$|").split(";") == [str(loc).replace("3a^1", "9b") for loc in entry["locants"]]
