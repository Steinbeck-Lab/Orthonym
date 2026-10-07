"""D3 (user decision 2026-09-28): the best-effort prefix-order fallback.

The round trip (OPSIN's SMILES of the name, read by RDKit, full InChIKey) of the
P(III) stereocentre of the PubChem-500k row below depends on the citation order of
the prefixes (TRIAGE.md 'Breadth -- PubChem losses', class D): the '(2S,4R,5S)-'
name reproduces the input's key only when the P-substituent '2-{...}' is cited
first; in the order (the Blue Book, "Alphanumerical order is
used to establish the order of citation of detachable substituent prefixes") it
gives the other P configuration, so the Blue-Book-ordered name never round-trips
and the row had no name. The name's meaning does not depend on the order: OPSIN's
own StdInChIKey is the same for both. OPSIN's SMILES carries a ring-closure digit on
the three-coordinate P in one order only, and RDKit reads a lone-pair stereocentre
that carries a ring-closure digit with the opposite configuration from the
OpenSMILES reading (CDK, OPSIN) -- as it reads the input, which carries one too.
The user chose to ship the paper's spelling (prefixes out of order),
built as a class rule: at the best-effort tier only, as the last resort for a
molecule about to be left unnamed, when the -ordered floor name fails its
round trip at the stereo layer only, the parent's prefixes are tried in other
orders (each moved to the front in turn, at most three orders) and the first
spelling that passes every check ships -- best_effort, never a PIN, with
``prefix_order_fallback`` True in the provenance row. One more check keeps the
RDKit reading from choosing the descriptors: the engine's CIP labels must equal
CDK's reading of the input string as written (``_cip_labels_match_input_as_written``);
without it, an input written without the ring-closure digit at P was given the
other P configuration's descriptors. Every shipped name here is
read back by an independent OPSIN call to the input's full InChIKey
(tests/support/rt_assert).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import (
    _full_inchikey,
    _independent_parse,
    assert_full_rt,
    name_best_effort,
)

pytestmark = pytest.mark.opsin_gate

# PubChem 500k row 493267 (not in eval/splits/a holdout split.json).
P_ROW = ("Cc1cn([C@H]2CC(O[P@]3O[C@](C)(c4ccccc4)[C@@H]4CCCN43)[C@@H](CO)O2)"
         "c(=O)[nH]c1=O")

# The paper's name (Zenodo source_data/pubchem500k__orthonym.csv, row 493267,
# round-trip exact): the P-substituent cited first.
PAPER_NAME = (
    "(2S,4R,5S)-2-{1-[(2R,4R)-4-(5-methyl-2,4-dioxo-1,3-diazacyclohex-5-en-1-yl)"
    "-2-(2-oxaethan-1-yl)-3-oxacyclopentan-1-yl]-1-oxamethan-1-yl}"
    "-4-(cyclohexa-1,3,5-trien-1-yl)-4-methyl"
    "-3-oxa-1-aza-2-phosphabicyclo[3.3.0]octane")

# The same descriptors in the order (the floor's first spelling).
ORDERED_NAME = (
    "(2S,4R,5S)-4-(cyclohexa-1,3,5-trien-1-yl)-4-methyl"
    "-2-{1-[(2R,4R)-4-(5-methyl-2,4-dioxo-1,3-diazacyclohex-5-en-1-yl)"
    "-2-(2-oxaethan-1-yl)-3-oxacyclopentan-1-yl]-1-oxamethan-1-yl}"
    "-3-oxa-1-aza-2-phosphabicyclo[3.3.0]octane")

# PubChem 500k row 3 of 'Breadth -- PubChem losses': a floor name with E/Z
# descriptors whose spelling round-trips.
EZ_FLOOR_ROW = "C/C=C(\\CC(=C\\C)/C(C)=C/Nc1ccc(-c2csc(C(=O)N(C)C)c2)cn1)OC"


# Roadmap N5 (name-quality lane L2): with the book spellings the strict path names the
# row by a substitutive name whose order round-trips, and the general engine names
# it at the complete tier; the fallback is not reached for it (both read back by OPSIN
# 2.9.0 to the full InChIKey). The fallback rule itself is pinned on this witness with the
# book spellings switched off (``book_prefixes.mechanical_forms``), which gives the
# writers' spellings of the paper's run.
BOOK_NAME = (
    "1-[(2R,5R)-5-(hydroxymethyl)-4-{[(2S,4R,5S)-4-methyl-4-phenyl-3-oxa-1-aza-"
    "2-phosphabicyclo[3.3.0]octan-2-yl]oxy}oxolan-2-yl]-5-methylpyrimidine-"
    "2,4(1H,3H)-dione")
COMPLETE_BOOK_NAME = (
    "(2S,4R,5S)-2-{[(2R,5R)-2-(hydroxymethyl)-5-(5-methyl-2,4-dioxo-1,3-diazacyclohex-"
    "5-en-1-yl)oxolan-3-yl]oxy}-4-methyl-4-phenyl-3-oxa-1-aza-2-phosphabicyclo[3.3.0]"
    "octane")


def test_the_book_spellings_name_the_witness_without_the_fallback(monkeypatch):
    import orthonym.assembly.t4_coverage as t4
    calls = []
    real = t4.name_prefix_order_fallback
    monkeypatch.setattr(t4, "name_prefix_order_fallback",
                        lambda mol: calls.append(1) or real(mol))
    res = name_best_effort(P_ROW)
    assert res.get("name") == BOOK_NAME
    assert_full_rt(res.get("name"), P_ROW)
    assert res.get("tier") == "systematic_verified"
    assert res.get("prefix_order_fallback") is False
    assert calls == []


def test_witness_ships_the_paper_spelling_at_best_effort():
    from orthonym.assembly.book_prefixes import mechanical_forms
    with mechanical_forms():
        res = name_best_effort(P_ROW)
    assert res.get("name") == PAPER_NAME
    assert_full_rt(res.get("name"), P_ROW)
    assert res.get("tier") == "best_effort"
    assert res.get("is_pin") is False
    assert res.get("source") == "t4_floor"
    assert res.get("prefix_order_fallback") is True
    assert res.get("verified") == "opsin"


def test_witness_is_deterministic_in_one_process():
    first = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(P_ROW)
    again = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(P_ROW)
    assert (first["name"], first["tier"], first["prefix_order_fallback"]) == (
        again["name"], again["tier"], again["prefix_order_fallback"])


def test_p145_ordered_spelling_is_read_as_another_stereoisomer():
    """The round trip the rule works around: same descriptors, order,
    same constitution, other full key (OPSIN's SMILES read by RDKit)."""
    back = _independent_parse(ORDERED_NAME)
    assert back is not None
    assert _full_inchikey(back) != _full_inchikey(P_ROW)
    assert _full_inchikey(back)[:14] == _full_inchikey(P_ROW)[:14]


@pytest.mark.parametrize("tier", ["pin", "complete"])
def test_pin_and_complete_tiers_are_unchanged(tier, monkeypatch):
    import orthonym.assembly.t4_coverage as t4
    calls = []
    real = t4.name_prefix_order_fallback
    monkeypatch.setattr(t4, "name_prefix_order_fallback",
                        lambda mol: calls.append(1) or real(mol))
    namer = (Orthonym(style="pin") if tier == "pin"
             else Orthonym(style="pin", **_emit_tier_flags(tier)))
    res = namer.name_tiered(P_ROW)
    if tier == "pin":
        assert res.get("tier") == "abstain"
    else:
        # roadmap N5 (name-quality lane L2): the general engine names the row with
        # the book spellings (it abstained before); never through the fallback
        assert res.get("name") == COMPLETE_BOOK_NAME
        assert res.get("tier") == "systematic_verified"
        assert_full_rt(res.get("name"), P_ROW)
    assert res.get("prefix_order_fallback") is False
    assert calls == []


def test_fallback_is_not_reached_by_a_molecule_that_names(monkeypatch):
    import orthonym.assembly.t4_coverage as t4
    calls = []
    real = t4.name_prefix_order_fallback
    monkeypatch.setattr(t4, "name_prefix_order_fallback",
                        lambda mol: calls.append(1) or real(mol))
    res = name_best_effort(EZ_FLOOR_ROW)
    assert_full_rt(res.get("name"), EZ_FLOOR_ROW)
    assert res.get("prefix_order_fallback") is False
    assert calls == []


def test_floor_keeps_the_p145_order_when_it_round_trips():
    """The fallback must not fire where the order works: armed or not, the
    floor gives the same name, and the fallback helper gives nothing."""
    from orthonym.assembly.t4_coverage import name_prefix_order_fallback
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    mol = Chem.MolFromSmiles(EZ_FLOOR_ROW)
    plain = name_universal_substitutive(mol)
    armed = name_universal_substitutive(mol, prefix_order_fallback=True)
    assert armed.name == plain.name
    assert armed.prefix_order_fallback is False
    assert_full_rt(armed.name, EZ_FLOOR_ROW)
    assert name_prefix_order_fallback(mol) is None


def test_floor_fallback_builds_the_paper_spelling():
    # roadmap N5 (name-quality lane L2): the fallback rule is pinned on the paper's
    # spelling with the book spellings switched off (the spellings it was built on).
    from orthonym.assembly.book_prefixes import mechanical_forms
    from orthonym.assembly.t4_coverage import name_prefix_order_fallback
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    mol = Chem.MolFromSmiles(P_ROW)
    with mechanical_forms():
        assert name_universal_substitutive(mol).prefix_order_fallback is False
        assert name_prefix_order_fallback(mol) == PAPER_NAME


def test_stereo_layer_only_gate():
    from orthonym.assembly.universal_substituent import _fails_at_stereo_layer_only
    mol = Chem.MolFromSmiles(P_ROW)
    assert _fails_at_stereo_layer_only(mol, ORDERED_NAME) is True
    # a different constitution (4-ethyl for 4-methyl) is not a stereo-only failure
    assert _fails_at_stereo_layer_only(
        mol, ORDERED_NAME.replace("4-methyl-", "4-ethyl-")) is False
    # a name that round-trips does not fail at all
    assert _fails_at_stereo_layer_only(mol, PAPER_NAME) is False


# dev2000 row: the floor's full candidate omits the E/Z descriptor of the
# ring-to-ring ylidene bond; it fails at the stereo layer only, but no citation
# order restores an omitted descriptor, so no other order is even built.
OMISSION_ROW = "CC(C)=C1NC(=O)/C(=C2/C=CC[C@H](C)O2)C1=O"


def test_an_omitted_descriptor_is_not_reordered(monkeypatch):
    import orthonym.assembly.universal_substituent as us
    from orthonym.assembly.t4_coverage import name_prefix_order_fallback
    rebuilds = []
    real = us._name_component

    def spy(*args, **kwargs):
        if kwargs.get("top_prefix_front") is not None:
            rebuilds.append(kwargs["top_prefix_front"])
        return real(*args, **kwargs)
    monkeypatch.setattr(us, "_name_component", spy)
    assert name_prefix_order_fallback(Chem.MolFromSmiles(OMISSION_ROW)) is None
    assert rebuilds == []


def test_orders_tried_are_capped(monkeypatch):
    import orthonym.assembly.universal_substituent as us
    # roadmap N5 (name-quality lane L2): the fallback rule is pinned on the paper's
    # spelling with the book spellings switched off (the spellings it was built on).
    from orthonym.assembly.book_prefixes import mechanical_forms
    from orthonym.assembly.t4_coverage import name_prefix_order_fallback
    rebuilds = []
    real = us._name_component

    def spy(*args, **kwargs):
        if kwargs.get("top_prefix_front") is not None:
            rebuilds.append(kwargs["top_prefix_front"])
        return real(*args, **kwargs)
    monkeypatch.setattr(us, "_name_component", spy)
    with mechanical_forms():
        assert name_prefix_order_fallback(Chem.MolFromSmiles(P_ROW)) == PAPER_NAME
    # the P-substituent (index 2, on the P stereocentre) is reached after the
    # 4-methyl prefix (index 1, on the C-4 stereocentre); both on stereo atoms
    assert rebuilds == [1, 2]
    assert len(rebuilds) <= us._MAX_PREFIX_ORDERS


def _opsin_own_stdinchikey(name):
    """OPSIN 2.9.0's StdInChIKey of ``name`` computed by OPSIN itself (``-o
    stdinchikey``, a fresh java -jar run): no SMILES, no RDKit."""
    import subprocess
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar("2.9.0")
    out = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-o", "stdinchikey"],
                         input=name + "\n", capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


def test_opsin_reads_both_orders_as_one_molecule():
    """The name's meaning does not depend on the order involves no
    stereodescriptor, the Blue Book): OPSIN's own key is the same."""
    both = {_opsin_own_stdinchikey(PAPER_NAME), _opsin_own_stdinchikey(ORDERED_NAME)}
    assert len(both) == 1 and next(iter(both)).startswith("DAXAGMLYVBTTGA-")


# The row-1 molecule of the four corpus rows ('CC(C)O[P@]1NC[C@@H]2CCCN21')
# written without a ring-closure digit at P, in both P configurations. RDKit reads
# these as written (its own labeller and CDK agree on P), but RDKit's canonical
# SMILES puts the ring-closure digit on P, so the engine's CIP label (centres on that
# SMILES) is the other configuration's. Before the guard the fallback shipped
# '(2S,5S)-...' for the first one, whose OPSIN key is an InChIKey
# while the input is an InChIKey: a different stereoisomer.
UNCLOSED_P_ROWS = ["C1(=CC=CC=C1)N1[P@@](N2CCC[C@H]2C1)OC(C)C",
                   "C1(=CC=CC=C1)N1[P@](N2CCC[C@H]2C1)OC(C)C"]
CORPUS_P_ROW = "CC(C)O[P@]1N(c2ccccc2)C[C@@H]2CCCN21"


@pytest.mark.parametrize("smiles", UNCLOSED_P_ROWS, ids=["P-R", "P-S"])
def test_fallback_never_ships_descriptors_the_input_does_not_have(smiles):
    from orthonym.namer import _cip_labels_match_input_as_written
    assert _cip_labels_match_input_as_written(smiles) is False
    res = name_best_effort(smiles)
    assert res.get("prefix_order_fallback") is False
    name = res.get("name")
    if name:
        assert_full_rt(name, smiles)
        assert _opsin_own_stdinchikey(name) == _full_inchikey(smiles)


def test_corpus_p_row_labels_are_those_as_written_and_it_ships():
    from orthonym.namer import _cip_labels_match_input_as_written
    assert _cip_labels_match_input_as_written(CORPUS_P_ROW) is True
    assert _cip_labels_match_input_as_written(P_ROW) is True
    # the fallback rule, with the book spellings switched off (the spellings it was
    # built on)
    from orthonym.assembly.book_prefixes import mechanical_forms
    with mechanical_forms():
        res = name_best_effort(CORPUS_P_ROW)
    assert res.get("name") == (
        "(2R,5S)-2-(2-methyl-1-oxapropan-1-yl)-3-(cyclohexa-1,3,5-trien-1-yl)"
        "-1,3-diaza-2-phosphabicyclo[3.3.0]octane")
    assert_full_rt(res.get("name"), CORPUS_P_ROW)
    assert res.get("prefix_order_fallback") is True
    # roadmap N5 (name-quality lane L2): with the book spellings ('1-methylethoxy',
    # the Blue Book; 'phenyl',:16290) the order
    # round-trips and the general engine names the row without the fallback
    res = name_best_effort(CORPUS_P_ROW)
    assert res.get("name") == (
        "(2R,5S)-2-(1-methylethoxy)-3-phenyl-1,3-diaza-2-phosphabicyclo[3.3.0]octane")
    assert_full_rt(res.get("name"), CORPUS_P_ROW)
    assert res.get("prefix_order_fallback") is False
