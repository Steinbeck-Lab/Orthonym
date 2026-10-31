"""DT3 step A: the PIN tier's promotion re-run also runs over a first-run name the default tier declines.

``Orthonym._name_with_pin_promotion`` returned the first run's name before the default tier had judged it, so a
name that tier then declined (a general-engine name, a strict-path name with a recorded non-PIN part) never got the
re-run that can answer with a name every piece of which a strict producer built (``assembly.strict_ring_prefix``).
The molecules below are declined at the PIN tier before the change and shipped ``pin_verified`` after it, with
the name the best-effort tier gives; the hand verdicts are those of `internal notes` (derived from
the book: parent, numbering, order of prefixes). Governing rule of the esters: "All preferred IUPAC
names for esters are named by functional class nomenclature" (the Blue Book).

The class guard (``assembly.pin_rerun.rerun_class_guard``) keeps the re-run from shipping a name for the
molecules the book gives no PIN "INTRODUCTION",:50943: not identified for Chapter compounds), or a
name that breaks a spelling rule of the PIN: a phospholipid, a glycoside, a boronic acid whose descriptors sit
at the front of the name "NAMING OF STEREOISOMERS",:44643).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.assembly.pin_rerun import first_run_declined, first_run_state, rerun_class_guard
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: molecules the PIN tier declined and now names: (SMILES, name). Hand verdict PIN in the plan (rows 5, 129, 213,
#: 220, 277, 280, 281, 310, 335 of ``dt3_judged.tsv``)
GAINS = [
    ("COc1ccc2[nH]cc(CCNC(C)=O)c2c1", "N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide"),      # melatonin
    ("CC(=O)OCCC1CC=C(C)C1(C)C", "2-(2,2,3-trimethylcyclopent-3-en-1-yl)ethyl acetate"),
    ("CC(=O)OCCc1scnc1C", "2-(4-methyl-1,3-thiazol-5-yl)ethyl acetate"),
    ("CC1=C(C(CCC1)(C)C)CCC(=O)CC(=O)OCC=C", "prop-2-en-1-yl 3-oxo-5-(2,6,6-trimethylcyclohex-1-en-1-yl)pentanoate"),
    ("CC1=CC(=O)CC1(C)CCCC/C=C/C(=O)OC", "methyl (2E)-7-(1,2-dimethyl-4-oxocyclopent-2-en-1-yl)hept-2-enoate"),
    ("CCOC(=O)CCC1=CC[C@@H](CC1)C(=C)C", "ethyl 3-[(4R)-4-(prop-1-en-2-yl)cyclohex-1-en-1-yl]propanoate"),
    ("COC(=O)CCCCCSC1CCCC1O", "methyl 6-[(2-hydroxycyclopentyl)sulfanyl]hexanoate"),
    ("B1(OC(C(O1)(C)C)(C)C)C(=C)CCCOC(=O)C(C)(C)C",
     "4-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)pent-4-en-1-yl 2,2-dimethylpropanoate"),
]

#: molecules whose re-run name must not ship over the declined first run: the class guard stops them
STAY_DECLINED = [
    ("phospholipid", "CCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCCC"),
    ("glycoside", "C=C(C)[C@@H]1CC=C(CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)CC1"),
    ("boronic acid", "B(/C=C/C[C@H](C)O)(O)O"),
    # atropine left this list with DT4 F2: the first run builds its functional-class ester name on the von Baeyer
    # ring, a fully systematic name, the Blue Book), not a stereoparent name
    ("Boc-peptide ester",      # peptide-like: the substitutive name is not promoted (DT3 lead decision 1)
     "CCOC(=O)/C=C/[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)OC(C)(C)C)C(C)C"),
]


def _pin_tier():
    return Orthonym(style="pin", **_emit_tier_flags("pin"))


@pytest.mark.parametrize("smiles, expected", GAINS)
def test_a_declined_first_run_gets_the_rerun_name_at_the_pin_tier(smiles, expected):
    row = _pin_tier().name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (expected, "pin_verified", True)
    assert name_compound(smiles, emit_tier="pin") == expected
    assert name_compound(smiles, emit_tier="best-effort") == expected      # the best-effort name, byte for byte


@pytest.mark.parametrize("what, smiles", STAY_DECLINED)
def test_the_class_guard_keeps_a_name_the_book_gives_no_pin_declined(what, smiles):
    row = _pin_tier().name_tiered(smiles)
    assert (row["tier"], row["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), what
    assert name_compound(smiles, emit_tier="pin") == ""


def test_the_name_does_not_depend_on_how_the_input_is_spelled():
    """Determinism: a respelled input gives the same name (the decision of the re-run reads the structure)."""
    namer = _pin_tier()
    for smiles, expected in GAINS[:3]:
        mol = Chem.MolFromSmiles(smiles)
        spellings = {Chem.MolToSmiles(mol, doRandom=True, canonical=False) for _ in range(4)}
        for spelling in spellings:
            assert namer.name_tiered(spelling)["name"] == expected, spelling


def test_a_name_the_default_tier_emits_is_not_named_again(monkeypatch):
    """Byte for byte: a first run the default tier emits ends the wrapper before the re-run is considered."""
    from orthonym.rules import pin_vocabulary
    seen = []
    real = pin_vocabulary.promotion_could_change
    monkeypatch.setattr(pin_vocabulary, "promotion_could_change", lambda probe: seen.append(1) or real(probe))
    assert name_compound("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", emit_tier="pin") == \
        "(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid"
    assert seen == []


def test_the_rerun_is_not_tried_for_a_first_run_that_reached_no_promotion_site(monkeypatch):
    """``first_run_declined`` is read only when the first run reached a site the re-run changes."""
    from orthonym.assembly import pin_rerun
    calls = []
    monkeypatch.setattr(pin_rerun, "first_run_declined", lambda *a: calls.append(a) or False)
    assert name_compound("CCO", emit_tier="pin") == "ethanol"
    assert calls == []


def test_first_run_declined_reads_the_default_tier_rule_on_the_first_name(monkeypatch):
    from orthonym.assembly import pin_rerun
    seen = []
    real = pin_rerun.first_run_declined

    def spy(namer, smiles, name):
        seen.append((name, real(namer, smiles, name)))
        return seen[-1][1]

    monkeypatch.setattr(pin_rerun, "first_run_declined", spy)
    assert name_compound("COc1ccc2[nH]cc(CCNC(C)=O)c2c1", emit_tier="pin") == \
        "N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide"
    assert len(seen) == 1 and seen[0][1] is True          # the first run's name was one the tier declines
    assert seen[0][0] != "N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide"


def test_a_row_that_cannot_be_derived_is_no_decline():
    class _Namer:
        _trivial_fallback = False

        def _tier_row_and_pin_form(self, *args):
            raise RuntimeError("no row")

    assert first_run_declined(_Namer(), "CC", "ethane") is False


def test_first_run_state_is_put_back_only_for_a_named_first_run():
    class _Namer:
        _last_dispatch_class = "first"
        _last_selected_offer = "offer"
        _prefix_order_fallback_name = None
        _offers = ["x"]

    namer = _Namer()
    restore = first_run_state(namer, True)
    namer._last_dispatch_class, namer._last_selected_offer, namer._offers = "second", None, []
    restore()
    assert (namer._last_dispatch_class, namer._last_selected_offer, namer._offers) == ("first", "offer", ["x"])
    skip = first_run_state(namer, False)
    namer._last_dispatch_class = "second"
    skip()
    assert namer._last_dispatch_class == "second"


@pytest.mark.parametrize("smiles, name, reason", [
    ("C=C(C)[C@@H]1CC=C(CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)CC1",
     "[(4S)-4-(prop-1-en-2-yl)cyclohex-1-en-1-yl]methyl beta-D-glucopyranoside", "sugar ring"),
    ("CCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCCC",
     "2-aminoethyl (2R)-2-(heptadecanoyloxy)-3-{[(9Z)-hexadec-9-enoyl]oxy}propyl hydrogen phosphate",
     "lipid backbone"),
    ("B(/C=C/C[C@H](C)O)(O)O", "(1E,4S)-(4-hydroxypent-1-en-1-yl)boronic acid", "P-91.3"),
    ("CN1C2CCC1CC(C2)OC(=O)C(CO)c1ccccc1", "8-methyl-8-azabicyclo[3.2.1]octan-3-yl 3-hydroxy-2-phenylpropanoate",
     "natural-product class alkaloid"),
    ("CCOC(=O)/C=C/[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](CO)NC(=O)OC(C)(C)C)C(C)C",
     "ethyl (2E,4S)-4-{(2S)-2-[(2S)-2-{(2S)-2-[(tert-butoxycarbonyl)amino]-3-hydroxypropanamido}-3-methylbutan"
     "amido]-4-methylpentanamido}-5-[(3S)-2-oxopyrrolidin-3-yl]pent-2-enoate", "peptide-like"),
])
def test_the_class_guard_names_its_reason(smiles, name, reason):
    assert reason in rerun_class_guard(Chem.MolFromSmiles(smiles), name)


def test_the_class_guard_lets_a_plain_name_through():
    assert rerun_class_guard(Chem.MolFromSmiles("COc1ccc2[nH]cc(CCNC(C)=O)c2c1"),
                             "N-[2-(5-methoxy-1H-indol-3-yl)ethyl]acetamide") is None
    assert rerun_class_guard(None, "x") == "no structure"


def test_a_guard_that_cannot_read_the_molecule_lets_nothing_through(monkeypatch):
    from orthonym.perception import lipids

    def boom(mol):
        raise RuntimeError("reader defect")

    monkeypatch.setattr(lipids, "detect_lipid_backbone", boom)
    reason = rerun_class_guard(Chem.MolFromSmiles("CCO"), "ethanol")
    assert reason is not None and "could not read" in reason
