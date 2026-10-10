"""Leads program L6, item 43a: an acylated non-suffix nitrogen is cited as the acylamino prefix.

When the decomposition cuts an amide whose nitrogen is not a suffix nitrogen of the amine
parent (the amine fragment's name cites it as an 'amino' prefix: the alpha-amino N of a
mercapturic acid, the aminophenoxy N of an acetaminophen glycoside, an amino sugar), the
acyl group was floated in front of the whole name:
'N-acetyl(2R)-2-amino-3-[(2,3-dihydroxypropanoyl)sulfanyl]propanoic acid'.

Blue Book, 'Substituents of the types -NH-CO-R and -NH-SO2-R'
(the Blue Book): "When a group having preference for citation as a principal
characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted amide
is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter
'e' in the complete name of the amide to 'o'" (:32993,:32995); "Method (1) generates preferred
IUPAC names." (:32998). The stereodescriptor goes with the parent: 'Naming of
stereoisomers' (:44643): "They are placed at the front of the complete name when related to the
parent structure".

``engine.nacyl_float_as_acylamino`` cites the nitrogen as 'acetamido' at the locant of the
amine fragment's 'amino' where the rule applies: the amine parent's principal class is senior
to the amide (an acid: the premise of the rule), each candidate proven by an OPSIN round trip
against the whole molecule and clean under the registered spelling checks (a prefix that now
sorts out of its alphanumerical place,, is not offered). The name is labelled below the
PIN (it is built by a string assembler), so the default tier does not claim it. The float stays
the fallback where no candidate qualifies; an amino sugar or a ceramide, whose amide is the
principal group ('N-[...]acetamide'), keeps it.

Mutation check (run once, recorded in internal notes): making
``nacyl_float_as_acylamino`` return None fails the unit tests and the molecule tests below.
"""
import pytest

from orthonym.decomposition import engine
from orthonym.metrics import provenance as pv
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

MERCAPTURATE = "CC(=O)N[C@@H](CSC(=O)C(O)CO)C(=O)O"
MERCAPTURATE_AMINE = "N[C@@H](CSC(=O)C(O)CO)C(=O)O"
MERCAPTURATE_AMINE_NAME = "(2R)-2-amino-3-[(2,3-dihydroxypropanoyl)sulfanyl]propanoic acid"
MERCAPTURATE_PIN_FORM = "(2R)-2-acetamido-3-[(2,3-dihydroxypropanoyl)sulfanyl]propanoic acid"

GLUCURONIDE = "CC(=O)Nc1ccc(O[C@@H]2O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]2O)cc1"
GLUCURONIDE_AMINE = "Nc1ccc(O[C@@H]2O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]2O)cc1"
GLUCURONIDE_AMINE_NAME = "(2S,3S,4S,5R,6S)-6-(4-aminophenoxy)-3,4,5-trihydroxyoxane-2-carboxylic acid"
GLUCURONIDE_ACYLAMINO = "(2S,3S,4S,5R,6S)-6-(4-acetamidophenoxy)-3,4,5-trihydroxyoxane-2-carboxylic acid"

TERPENE_CONJUGATE = (
    "C=C/C(C)=C/C[C@@H]1[C@@]2(C)C[C@@H](O)CC(C)(C)[C@@H]2C[C@H](O)[C@@]1(O)CSC[C@H](NC(C)=O)C(=O)O")


# ---------------------------------------------------------------------------
# The conversion (OPSIN-proven, fail-closed)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("amine,amine_name,acid_name,parent,expected", [
    (MERCAPTURATE_AMINE, MERCAPTURATE_AMINE_NAME, "acetic acid", MERCAPTURATE,
     MERCAPTURATE_PIN_FORM),
    (GLUCURONIDE_AMINE, GLUCURONIDE_AMINE_NAME, "acetic acid", GLUCURONIDE,
     GLUCURONIDE_ACYLAMINO),
    ("NCC(=O)O", "aminoacetic acid", "acetic acid", "CC(=O)NCC(=O)O", "acetamidoacetic acid"),
    ("NCC(=O)O", "aminoacetic acid", "propanoic acid", "CCC(=O)NCC(=O)O",
     "propanamidoacetic acid"),
    ("NCC(=O)O", "aminoacetic acid", "benzoic acid", "O=C(NCC(=O)O)c1ccccc1",
     "benzamidoacetic acid"),
])
def test_the_amino_prefix_becomes_the_acylamino_prefix_at_its_locant(
        amine, amine_name, acid_name, parent, expected):
    got = engine.nacyl_float_as_acylamino(amine, amine_name, acid_name, parent)
    assert got == expected
    assert name_is_rt_exact(got, parent), got       # an independent read-back, not the engine's


def test_a_candidate_that_does_not_read_back_is_not_offered(monkeypatch):
    monkeypatch.setattr(engine, "_round_trips_to", lambda smiles, name: False)
    assert engine.nacyl_float_as_acylamino(
        MERCAPTURATE_AMINE, MERCAPTURATE_AMINE_NAME, "acetic acid", MERCAPTURATE) is None


def _spelling_clean(monkeypatch):
    """The logic tests below feed names that are not the names of their structures: the
    registered spelling checks are told to find nothing (they are tested on real names below)."""
    import orthonym.validation.pin_spelling as ps
    monkeypatch.setattr(ps, "check_pin_spelling", lambda mol, name, **kw: [])


def test_two_candidates_that_read_back_are_not_told_apart(monkeypatch):
    """Which 'amino' is the nitrogen is not read from the name: with two proven candidates the
    conversion declines and the float keeps its place."""
    _spelling_clean(monkeypatch)
    monkeypatch.setattr(engine, "_round_trips_to", lambda smiles, name: True)
    assert engine.nacyl_float_as_acylamino(
        "NCC(=O)O", "2-amino-3-amino-propanoic acid", "acetic acid", "CC(=O)NCC(=O)O") is None
    assert engine.nacyl_float_as_acylamino(
        "NCC(=O)O", "2-amino-propanoic acid", "acetic acid", "CC(=O)NCC(=O)O") == (
        "2-acetamido-propanoic acid")


def test_only_the_amino_prefix_of_a_name_is_a_candidate(monkeypatch):
    """'dimethylamino' and 'benzenamine' end in 'amino'/'amine' as part of a longer word."""
    seen = []

    def fake(smiles, name):
        seen.append(name)
        return True

    _spelling_clean(monkeypatch)
    monkeypatch.setattr(engine, "_round_trips_to", fake)
    engine.nacyl_float_as_acylamino(
        "NCC(=O)O", "3-(dimethylamino)-2-amino-propanoic acid", "acetic acid", "CC(=O)NCC(=O)O")
    assert seen == ["3-(dimethylamino)-2-acetamido-propanoic acid"]


CERAMIDE = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCC(C)C"
CERAMIDE_AMINE = "N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCC(C)C"
CERAMIDE_AMINE_NAME = "(2S,3R,4E)-2-amino-15-methylhexadec-4-ene-1,3-diol"


def test_a_parent_junior_to_the_amide_keeps_the_float(monkeypatch):
    """: the amide of a ceramide is senior to the alcohols of its sphingoid base, so it is
    the principal group ('N-[...]tetracosanamide'); an acylamino PREFIX on the diol parent is
    not what licenses: it opens "When a group having preference for citation as a
    principal characteristic group is present" (the Blue Book), i.e. a group senior to the
    amide. The seniority alone declines, whatever the read-back and the spelling checks say."""
    import orthonym.validation.pin_spelling as ps
    monkeypatch.setattr(ps, "check_pin_spelling", lambda mol, name, **kw: [])
    monkeypatch.setattr(engine, "_round_trips_to", lambda smiles, name: True)
    assert engine.nacyl_float_as_acylamino(
        CERAMIDE_AMINE, CERAMIDE_AMINE_NAME, "tetracosanoic acid", CERAMIDE) is None
    # the same fragment shape with an acid as the principal group is converted
    assert engine.nacyl_float_as_acylamino(
        "NCC(=O)O", "2-amino-acetic acid", "tetracosanoic acid",
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCC(=O)O") == "2-tetracosanamido-acetic acid"


@pytest.mark.opsin_gate
def test_a_prefix_that_would_sort_out_of_its_place_is_not_offered():
    """'acetyloxy' sorts before 'amino', and after 'acetamido': 'acetam' < 'acetyl'): the
    name 'acetyloxy'... 'acetamido' reads back to the molecule and breaks the alphanumerical
    order, so the float keeps its place. The parent here is an acid (senior to the amide)."""
    parent = "CC(=O)N[C@@H](COC(C)=O)C(=O)O"
    candidate = "(2S)-3-(acetyloxy)-2-acetamidopropanoic acid"
    assert name_is_rt_exact(candidate, parent)
    from rdkit import Chem
    from orthonym.validation.pin_spelling import check_pin_spelling
    assert [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(parent), candidate)] == ["P-14.5"]
    assert engine.nacyl_float_as_acylamino(
        "N[C@@H](COC(C)=O)C(=O)O", "(2S)-3-(acetyloxy)-2-aminopropanoic acid", "acetic acid",
        parent) is None


@pytest.mark.parametrize("amine,amine_name", [
    ("NCC(=O)O", "glycine"),                               # no 'amino' prefix to replace
    ("CNCC(=O)O", "(methylamino)acetic acid"),             # a secondary amine nitrogen
    ("OC(=O)C1CCCN1", "pyrrolidine-2-carboxylic acid"),    # a ring nitrogen
    ("NCCN", "2-aminoethan-1-amine"),                      # two acylatable nitrogens
    ("c1ccc2[nH]ccc2c1", "1H-indole"),                     # an aromatic nitrogen
])
def test_it_declines_where_the_nitrogen_is_not_one_primary_acyclic_amino(amine, amine_name):
    assert engine.nacyl_float_as_acylamino(amine, amine_name, "acetic acid", "CC(=O)NCC(=O)O") is None


def test_it_declines_without_opsin(monkeypatch):
    """Fail closed: a name nothing has read is not shipped."""
    import orthonym.validation.opsin_roundtrip as rt
    monkeypatch.setattr(rt, "_find_opsin_jar", lambda *a, **k: None)
    assert engine.nacyl_float_as_acylamino(
        MERCAPTURATE_AMINE, MERCAPTURATE_AMINE_NAME, "acetic acid", MERCAPTURATE) is None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

FLOAT = "N-acetyl" + MERCAPTURATE_AMINE_NAME


def _gate(**kw):
    return engine.gate_nonsuffix_nacyl_float(
        MERCAPTURATE_AMINE, FLOAT, MERCAPTURATE_AMINE_NAME, **kw)


@pytest.mark.opsin_gate
def test_the_gate_cites_the_acylamino_prefix_and_labels_it_below_the_pin():
    pv.clear_provenance()
    got = _gate(acid_name="acetic acid", parent_smiles=MERCAPTURATE)
    assert got == MERCAPTURATE_PIN_FORM
    assert MERCAPTURATE_PIN_FORM in pv.get_provenance()["non_pin_fragments"]
    assert FLOAT not in pv.get_provenance()["non_pin_fragments"]
    pv.clear_provenance()


def test_the_gate_keeps_the_float_as_a_demotion_where_nothing_is_proven(monkeypatch):
    monkeypatch.setattr(engine, "_round_trips_to", lambda smiles, name: False)
    pv.clear_provenance()
    assert _gate(acid_name="acetic acid", parent_smiles=MERCAPTURATE) == FLOAT
    assert FLOAT in pv.get_provenance()["non_pin_fragments"]
    pv.clear_provenance()
    # and without the acid and the parent it knew nothing about: as before the change
    assert _gate() == FLOAT
    pv.clear_provenance()


def test_the_refusal_scope_is_untouched():
    """Inside _handle_peptide's systematic attempt the float is withheld so the composer can
    build the substitutive name (decision A): the conversion is for the demotion only."""
    refusals = []
    token = engine._NACYL_FLOAT_REFUSALS.set(refusals)
    try:
        assert _gate(acid_name="acetic acid", parent_smiles=MERCAPTURATE) is None
    finally:
        engine._NACYL_FLOAT_REFUSALS.reset(token)
    assert refusals == [FLOAT]


def test_a_suffix_nitrogen_keeps_its_float():
    """'N-methylacetamide'-shaped names: the N is the amide / amine suffix of the parent."""
    assert engine.gate_nonsuffix_nacyl_float(
        "NC1CCCCC1", "N-acetylcyclohexanamine", "cyclohexanamine",
        acid_name="acetic acid", parent_smiles="CC(=O)NC1CCCCC1") == "N-acetylcyclohexanamine"


# ---------------------------------------------------------------------------
# The molecules (production: the OPSIN validity gate on; read back independently)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected_start", [
    (MERCAPTURATE, MERCAPTURATE_PIN_FORM),
    (GLUCURONIDE, GLUCURONIDE_ACYLAMINO),
    (TERPENE_CONJUGATE, "(2R)-2-acetamido-3-[({(1S,2R,3R,4S,6S,9S)-3,4,9-trihydroxy-1,7,7-trimethyl-2-"),
], ids=["mercapturate", "acetaminophen-glucuronide", "terpene-conjugate"])
def test_the_float_is_gone_from_the_breadth_tier(smiles, expected_start):
    res = name_best_effort(smiles)
    name = res.get("name")
    assert name and res.get("tier") != "abstain", res
    assert name.startswith(expected_start), name
    assert "N-acetyl(" not in name and "N-acetyl[" not in name, name
    assert name_is_rt_exact(name, smiles), name
    # pin_verified only where the strict path builds the PIN itself: since the leads integration
    # (L6-acylsulfanyl.patch) the mercapturate's acylsulfanyl prefix is built by the prefix writer,
    # the ester cited as a prefix of the acid 'Esters cited as prefixes',
    # the Blue Book), so its PIN form is certified like any other; never an assembled name.
    if res.get("tier") == "pin_verified":
        assert (name, res.get("source")) == (MERCAPTURATE_PIN_FORM, "pin_path"), res


@pytest.mark.opsin_gate
def test_the_default_tier_does_not_claim_the_assembled_name():
    from tests.support.pin_tiers import name_default
    res = name_default(MERCAPTURATE)
    assert res.get("tier") != "pin_verified" or res.get("name") == MERCAPTURATE_PIN_FORM, res
    if res.get("name") and res.get("tier") != "abstain":
        assert name_is_rt_exact(res["name"], MERCAPTURATE), res
