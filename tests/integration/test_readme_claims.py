"""P2 -- public claims (2026-09-28): the README's examples and tier statements.

Every name the README prints (the Python examples, the Quick start, the round-trip
picture and the specimen picture that ``docs/readme/build.py`` draws from the engine's
own output) is pinned here, and each one is read back by a FRESH OPSIN call that does
not go through the engine (``tests.support.rt_assert._independent_parse``) and compared
by full standard InChIKey. The README's statements about the tiers are checked on
witnesses of each class:

* "pin_verified (the strict PIN path built it, certified it as the PIN, and OPSIN read
  it back)" -- verified 'opsin' and an independent full-key read-back;
* "The default tier makes a few exceptions for names OPSIN cannot read in full...
  `--provenance` marks each of them. The wider tiers ship none of them, except the
  metal-complex list names" -- a grammar carve-out (thioperoxol), a constitution-only
  name (cis-cyclohexane-1,4-diol) and the heme b list name.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import _independent_parse
from tests.unit.rules.test_d1_coordination_v36 import HEME_B

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]


def _key(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return inchi.MolToInchiKey(mol) if mol is not None else ""


def _row(smiles, tier="pin"):
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


# README "The same from Python" and "Quick start"
QUICK_START = [
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid"),
    ("CCO", "ethanol"),
    ("CC(=O)Oc1ccccc1C(=O)O", "2-(acetyloxy)benzoic acid"),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    ("C/C=C/C", "(2E)-but-2-ene"),
    ("C[C@H](O)CC", "(2S)-butan-2-ol"),
    ("c1ccccc1", "benzene"),
]


@pytest.mark.parametrize("smiles,name", QUICK_START, ids=[n for _, n in QUICK_START])
def test_quick_start_example_prints_its_name(smiles, name):
    assert name_compound(smiles) == name
    assert _key(_independent_parse(name)) == _key(smiles)


def test_round_trip_picture_caffeine():
    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    row = _row(caffeine)
    assert row["name"] == "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert _key(caffeine) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    assert _key(_independent_parse(row["name"])) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"


# README "See it working": the five specimen commands and what they print
SPECIMEN = [
    ("CCO", "pin", "ethanol", "pin_verified"),
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "pin",
     "(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid", "pin_verified"),
    ("CCCCCCCCCCCCC/C=C/[C@H]([C@H](CO)N)O", "valid",
     "(2S,3R,4E)-2-aminooctadec-4-ene-1,3-diol", "pin_unverified"),
    ("CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C", "valid",
     "(2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-en-1-yl)"
     "nona-2,4,6,8-tetraen-1-ol", "pin_unverified"),
]


@pytest.mark.parametrize("smiles,tier,name,label", SPECIMEN,
                         ids=["ethanol", "ibuprofen", "sphingosine", "retinol"])
def test_specimen_row(smiles, tier, name, label):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == label and row["verified"] == "opsin", row
    assert _key(_independent_parse(name)) == _key(smiles)


def test_specimen_uranium_trioxide_declines():
    row = _row("O=[U](=O)=O")
    assert row["name"] == "inorganic compound (not supported)", row
    assert row["tier"] == "abstain" and row["limit_code"] == "UNSUPPORTED_ELEMENT", row


# README "Output tiers" / "Checked.": the default tier's marked exceptions
def test_default_tier_exceptions_are_marked_and_not_shipped_at_the_wider_tiers():
    carve = _row("CSO")                       # a name form outside OPSIN's grammar
    assert carve["name"] == "methane-SO-thioperoxol", carve
    assert carve["verified"] == "unverified", carve
    assert carve["gate_outcome"] == "carveout:thioperoxol", carve
    assert carve["tier"] != "pin_verified", carve
    assert _independent_parse(carve["name"]) is None
    diol = _row("O[C@H]1CC[C@@H](O)CC1")      # stereodescriptors OPSIN cannot parse
    assert diol["name"] == "(1s,4s)-cyclohexane-1,4-diol", diol
    assert diol["verified"] == "opsin_constitution", diol
    assert diol["tier"] != "pin_verified", diol
    assert _independent_parse(diol["name"]) is None
    heme = _row(HEME_B["smiles"])             # a metal-complex list name
    # the label of its naming path, as in the paper's measured run (user decision
    # 2026-09-30, replacing the 2026-09-28 label systematic_verified)
    assert heme["verified"] == "identity" and heme["tier"] == "pin_verified", heme
    for tier in ("valid", "best-effort"):
        wide = {s: _row(s, tier) for s in ("CSO", "O[C@H]1CC[C@@H](O)CC1")}
        for s, row in wide.items():
            assert row["name"] not in (carve["name"], diol["name"]), row
            if row["name"]:
                assert row["verified"] == "opsin", row
                assert _key(_independent_parse(row["name"])) == _key(s), row
        assert _row(HEME_B["smiles"], tier)["name"] == heme["name"]
