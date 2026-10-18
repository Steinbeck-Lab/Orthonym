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
  metal-complex list names" -- a grammar carve-out (inositol), a constitution-only
  name (cis-cyclohexane-1,4-diol) and the heme b list name;
* "PIN spellings that OPSIN's grammar lacks and that OPSIN reads back through a twin
  spelling of the same structure" -- 1,4-dihydro-1,4-sulfanonaphthalene, ``verified``
  'opsin_twin': a fresh OPSIN call reads nothing for the name and reads its twin to the
  input; a wider tier names the structure in a spelling OPSIN reads.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym, name_compound
from orthonym.cli import _emit_tier_flags
from orthonym.validation.twin_spellings import twin_of
from tests.support.rt_assert import _independent_parse

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
    carve = _row("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O")  # a name form outside OPSIN's grammar
    assert carve["name"] == "scyllo-inositol", carve
    assert carve["verified"] == "unverified", carve
    assert carve["gate_outcome"] == "carveout:inositol", carve
    assert carve["tier"] != "pin_verified", carve
    assert _independent_parse(carve["name"]) is None
    diol = _row("O[C@H]1CC[C@@H](O)CC1")      # stereodescriptors OPSIN cannot parse
    assert diol["name"] == "(1s,4s)-cyclohexane-1,4-diol", diol
    assert diol["verified"] == "opsin_constitution", diol
    assert diol["tier"] != "pin_verified", diol
    assert _independent_parse(diol["name"]) is None
    # The third exception, the heme b metal-complex list name, and its label at every tier
    # are asserted by test_p2_labels_public_claims.py::
    # test_coordination_list_name_keeps_the_label_of_its_naming_path.
    for tier in ("valid", "best-effort"):
        wide = {s: _row(s, tier) for s in ("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "O[C@H]1CC[C@@H](O)CC1")}
        for s, row in wide.items():
            assert row["name"] not in (carve["name"], diol["name"]), row
            if row["name"]:
                assert row["verified"] == "opsin", row
                assert _key(_independent_parse(row["name"])) == _key(s), row


def test_default_tier_twin_checked_pin_spelling_is_read_back_through_its_twin():
    sulfano = "C12C=CC(C3=CC=CC=C13)S2"        # naphthalene, C1 and C4 bridged by -S-
    name, twin = "1,4-dihydro-1,4-sulfanonaphthalene", "1,4-dihydro-1,4-epithionaphthalene"
    row = _row(sulfano)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin", row
    assert row["gate_outcome"] == "twin_verified:bridge_sulfano", row
    assert "twin_round_trip" in row["gates_passed"], row
    assert twin_of(name) == (twin, ("bridge_sulfano",))
    assert _independent_parse(name, sulfano) is None              # OPSIN has no grammar for the name
    assert _key(_independent_parse(twin, sulfano)) == _key(sulfano)   # and reads its twin to the input
    for tier in ("valid", "best-effort"):                         # a wider tier ships the twin's form
        wide = _row(sulfano, tier)
        assert wide["name"] == twin, wide
        assert wide["verified"] == "opsin" and wide["tier"] != "pin_verified", wide
