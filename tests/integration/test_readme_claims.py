"""P2 -- public claims (2026-09-28): the README's examples and tier statements.

Every name the README prints (the Python examples, the Quick start, the round-trip
picture, the specimen picture and the gallery that ``docs/readme/build.py`` draws from the engine's
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
from orthonym.opsin_capabilities import opsin_capabilities
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
     "nona-2,4,6,8-tetraen-1-ol", "pin_verified"),
]


@pytest.mark.parametrize("smiles,tier,name,label", SPECIMEN,
                         ids=["ethanol", "ibuprofen", "sphingosine", "retinol"])
def test_specimen_row(smiles, tier, name, label):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == label and row["verified"] == "opsin", row
    assert _key(_independent_parse(name)) == _key(smiles)


# README gallery ("What it names"): the seven cards and the tier each one draws
GALLERY = [
    ("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", "pin",
     "(4R,4aR,7S,7aR,12bS)-3-methyl-2,3,4,4a,7,7a-hexahydro-1H-4,12-methano[1]benzofuro"
     "[3,2-e]isoquinoline-7,9-diol", "pin_verified"),
    ("c1ccc(cc1)I(=O)=O", "pin", "iodylbenzene", "pin_verified"),
    ("c1ccc(cc1)[P@]1CCC[C@@H]1c1ccccc1", "pin", "(1S,2R)-1,2-diphenylphospholane", "pin_verified"),
    ("[2H][C@@H](C)C[C@H]([3H])C", "pin", "(2R,4S)-(4-2H1,2-3H1)pentane", "pin_verified"),
    ("CC(=O)[O-].CC(=O)[O-].[Zn+2]", "pin", "zinc diacetate", "pin_verified"),
    ("C1CC2CC3CCC4CC5CC1C2C3C45", "best-effort",
     "pentacyclo[9.3.1.0^3,14.0^6,13.0^8,12]pentadecane", "systematic_verified"),
    ("C1CCC2(CC1)CCCC2", "pin", "spiro[4.5]decane", "pin_verified"),
]


@pytest.mark.parametrize("smiles,tier,name,label", GALLERY,
                         ids=["morphine", "iodylbenzene", "phospholane", "isotopes", "zinc", "cage", "spiro"])
def test_gallery_card(smiles, tier, name, label):
    row = _row(smiles, tier)
    assert row["name"] == name, row
    assert row["tier"] == label, row
    assert _key(_independent_parse(name)) == _key(smiles)


def test_specimen_xenon_difluoride_declines():
    row = _row("F[Xe]F")
    assert row["name"] == "inorganic compound (not supported)", row
    assert row["tier"] == "abstain" and row["limit_code"] == "UNSUPPORTED_ELEMENT", row


# README "Output tiers" / "Checked.": the default tier's marked exceptions
def test_default_tier_exceptions_are_marked_and_not_shipped_at_the_wider_tiers():
    # an inositol (a name form outside OPSIN's grammar) is no PIN:50943, relabel 2026-10-31): the
    # default tier declines it and the best-effort tier ships it on its exact InChIKey (guide/tiers.md)
    inositol = "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    assert _row(inositol)["limit_code"] == "NO_VERIFIED_PIN"
    carve = _row(inositol, "best-effort")
    assert carve["name"] == "scyllo-inositol", carve
    assert carve["verified"] == "unverified", carve
    assert carve["gate_outcome"] == "carveout:inositol", carve
    assert carve["tier"] != "pin_verified" and carve["is_pin"] is False, carve
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
        wide = {s: _row(s, tier) for s in (inositol, "O[C@H]1CC[C@@H](O)CC1")}
        for s, row in wide.items():
            if s == inositol and tier == "best-effort":
                continue                             # asserted above
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


# README "Checked." / "Output tiers": the twin families the engine has no other spelling for ship at
# every tier (validation.twin_spellings.general_tier_twin_round_trip). (the Blue Book)
# 'diphenyliodanuide (PIN)'; (:39785) the halido ligand word of an additive organometallic name, a class
# gives no PIN. The last column is the capability of the OPSIN in use (``opsin_capabilities``) that reads the
# name itself: OPSIN's main branch reads a halogen -uide name, so the name needs no twin there and its label is
# ``opsin``; no OPSIN reads the halido ligand word.
TWIN_EVERY_TIER = [
    ("c1ccc(cc1)[I-]c1ccccc1", "diphenyliodanuide", "diphenyl-λ3-iodanide", "pin_verified", "halogen_uide"),
    ("C(C)[Ni]Br", "bromido(ethyl)nickel", "bromo(ethyl)nickel", "systematic_verified", None),
]


@pytest.mark.parametrize("smiles,name,twin,tier_label,capability", TWIN_EVERY_TIER,
                         ids=[n for _, n, _, _, _ in TWIN_EVERY_TIER])
def test_a_twin_family_with_no_other_spelling_ships_through_its_twin_at_every_tier(
        smiles, name, twin, tier_label, capability):
    reads_itself = bool(capability and opsin_capabilities()[capability])
    if reads_itself:
        assert _key(_independent_parse(name, smiles)) == _key(smiles)   # the OPSIN in use has a grammar for the name
    else:
        assert _independent_parse(name, smiles) is None                 # OPSIN has no grammar for the name
    assert _key(_independent_parse(twin, smiles)) == _key(smiles)     # and reads its twin to the input
    for tier in ("pin", "valid", "best-effort"):
        row = _row(smiles, tier)
        if tier == "pin" and tier_label != "pin_verified":
            assert row["tier"] == "abstain", row                     # the default tier ships PINs only
            continue
        assert row["name"] == name, row
        assert row["tier"] == tier_label and row["verified"] == ("opsin" if reads_itself else "opsin_twin"), row


# README "Checked." / "Output tiers": a thioperoxol is named in the form OPSIN reads at a wider tier (user decision
# 2026-10-22, "Best-effort ships readable"); its PIN spelling, THE SUFFIXES 'SO-THIOPEROXOL', AND CHALOGEN
# ANALOGUES (the Blue Book) 'CH3-S-OH methane-SO-thioperoxol (PIN)' (:24326), ships through its twin at the
# default tier, and at a wider tier only for a molecule that has no name OPSIN reads (the anion,,:41035).
def test_a_thioperoxol_is_named_in_the_form_opsin_reads_at_a_wider_tier():
    smiles, pin, readable = "CSO", "methane-SO-thioperoxol", "(hydroxysulfanyl)methane"
    assert _independent_parse(pin, smiles) is None                    # OPSIN has no grammar for the PIN spelling
    assert _key(_independent_parse(readable, smiles)) == _key(smiles)  # and reads the name by prefixes
    row = _row(smiles, "pin")
    assert row["name"] == pin and row["tier"] == "pin_verified" and row["verified"] == "opsin_twin", row
    for tier in ("valid", "complete", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == readable and row["verified"] == "opsin", row
        assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row


def test_the_pin_spelling_of_the_thioperoxolate_anion_ships_through_its_twin_at_every_tier():
    smiles, name = "CS[O-]", "methane(SO-thioperoxolate)"
    assert _independent_parse(name, smiles) is None
    for tier in ("pin", "valid", "complete", "best-effort"):
        row = _row(smiles, tier)
        assert row["name"] == name, row
        assert row["tier"] == "pin_verified" and row["verified"] == "opsin_twin", row


def _hydrolysed_keys(smiles):
    """The InChIKeys of the acids an anhydride gives on hydrolysis, by RDKit alone: every
    C(=O)-O-C(=O) oxygen keeps one acyl carbon and the other gets a new OH."""
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    links = {m[2]: m[3] for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX2][CX3]=O"))}
    for o, c in links.items():
        mol.RemoveBond(o, c)
        mol.AddBond(c, mol.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    return sorted(_key(Chem.MolToSmiles(f)) for f in Chem.GetMolFrags(out, asMols=True))


def test_default_tier_poly_anhydride_is_read_back_through_its_acids():
    # Di- and polyanhydrides (the Blue Book): 'diacetic butanedioic dianhydride (PIN)' (:32386)
    smiles = "CC(=O)OC(=O)CCC(=O)OC(=O)C"
    name = "diacetic butanedioic dianhydride"
    row = _row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin_acids", row
    assert row["gate_outcome"] == "acid_split_verified:anhydride_acids", row
    assert "acid_split_round_trip" in row["gates_passed"], row
    assert _independent_parse(name, smiles) is None                   # OPSIN has no grammar for the name
    acids = [_key(_independent_parse(a)) for a in ("acetic acid", "acetic acid", "butanedioic acid")]
    assert _hydrolysed_keys(smiles) == sorted(acids)                  # the acids it cites are the input's
    for tier in ("valid", "best-effort"):                             # a wider tier ships a name OPSIN reads
        wide = _row(smiles, tier)
        assert wide["name"] == "1,4-bis(acetyloxy)-1,4-dioxobutane", wide
        assert wide["verified"] == "opsin" and wide["tier"] != "pin_verified", wide
        assert _key(_independent_parse(wide["name"], smiles)) == _key(smiles), wide
