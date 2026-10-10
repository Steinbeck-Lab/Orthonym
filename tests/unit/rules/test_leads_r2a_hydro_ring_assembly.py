"""Leads R2A: a ring assembly of two identical skeletons that differ in hydrogenation is named with
'hydro' prefixes (``rules.ring_assemblies.detect_hydro_ring_assembly``).

``### **** Ring assemblies composed of monocyclic components`` (the Blue Book),
(a):17079: "Low locants are assigned to 'hydro' prefixes in accordance with the fixed numbering of each
assembly. In biphenyl and polyphenyl assemblies, one benzene ring must remain in the assembly;
otherwise, the starting parent hydride is the saturated assembly and the ending 'ene' is used to denote
unsaturation (see. Furthermore, when a modified ring assembly of two rings consists of a
benzene ring and a cyclohexane ring substitutive nomenclature is preferred (see.", example
:17083 "2,3-dihydro-1,1'-biphenyl (PIN)... (cyclohexa-1,3-dien-1-yl)benzene"; (b):17112 "Low locants are
assigned to junctions between rings, then to indicated hydrogen, if any, and finally to 'hydro'
prefixes" and ``### **** Ring assemblies composed of polycyclic compounds`` (:17153),
:17161 "1,2',3',4-tetrahydro-2,2'-binaphthalene (PIN)"; ``## **** UNSATURATION IN RING ASSEMBLIES
COMPOSED OF MONOCYCLIC MANCUDE AND SATURATED RINGS`` (:24151),:24153 "When assemblies of otherwise
identical rings contain both mancude and saturated rings, the use of hydro prefixes is preferred, except
in the case of a two ring assembly consisting of one benzene ring and a cyclohexane ring",:24157
"cyclohexylbenzene (PIN)",:24159 "1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)"; ``### ****
NUMBERING`` (:3219) (b) indicated hydrogen, (c) principal characteristic groups, (e)(i) "low locants are
given to hydro/dehydro prefixes", (f) the detachable alphabetized prefixes; ``****`` (:16872)
"in names, they are cited immediately at the front of the name of the parent hydride, after alphabetized
prefixes"; ``### **** General methodology`` (:16878),:16880 "Indicated hydrogen atoms have
priority over 'hydro' prefixes for low locants. If indicated hydrogen atoms are present in a name, the
'hydro' prefixes precede them."; ``****`` (:15593) the junction positions count for the maximum
number of noncumulative double bonds ('2H-1,2'-bipyridine (PIN)',:15634).

Before this producer a benzene ring joined to a cyclohexene ring was named '(cyclohex-1-en-1-yl)benzene',
which the check (``validation/spelling/checks_assembly.py``) withdraws from the PIN tier, so the
molecule had no PIN at all. Round-trip exactness is no evidence of the spelling: OPSIN reads the names
below, their rivals with higher locants and the substitutive names back to one structure.

The tests of the engine (``needs_wiring``) need the call of ``detect_hydro_ring_assembly`` in
``namer.py`` (``proposals/R2A-namer-hydro-assembly.patch``); without it they are expected failures and
the producer is exercised on its own.

An isotope-labelled molecule is named through its isotope-stripped skeleton, and the label is spliced into
the skeleton's name, the Blue Book "before the part of the compound that is isotopically
substituted"). The splice cannot enter a name of this producer (the labels of the primed component need a
primed locant, and there is no slot between the hydro prefixes and a locant set that carries a prime), so the
namer runs the decorator inside ``hydro_assembly_withheld`` and the molecule keeps the producers it had
(``needs_label_wiring``: the second hunk of the same patch).
"""
import random

import pytest
from rdkit import Chem, RDLogger

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.perception.rings import get_ring_systems
from orthonym.rules import ring_assemblies
from orthonym.rules.ring_assemblies import detect_hydro_ring_assembly
from orthonym.validation.opsin_roundtrip import OpsinUnavailable
from tests.support.hydro_wiring import needs_label_wiring, needs_wiring
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact

RDLogger.DisableLog("rdApp.*")

#: (id, SMILES, the PIN, where the book decides it). The first five names are printed in the book.
BOOK = [
    ("dihydro-biphenyl", "c1ccccc1C1=CC=CCC1", "2,3-dihydro-1,1'-biphenyl",
     "P-31.2.3.3.5.1 (a) :17083 '2,3-dihydro-1,1'-biphenyl (PIN)'"),
    ("hexahydro-2,2'-bipyridine", "c1ccc(nc1)C1CCCCN1", "1,2,3,4,5,6-hexahydro-2,2'-bipyridine",
     "P-54.3 :24159 '1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)'"),
    ("tetrahydro-1,2'-binaphthalene", "c1ccc2c(c1)CCC(c1cccc3ccccc13)C2",
     "1',2',3',4'-tetrahydro-1,2'-binaphthalene", "P-31.2.2 :16884 (PIN)"),
    ("dihydro-2,2'-binaphthalene", "C1=CC(c2ccc3ccccc3c2)Cc2ccccc21", "1,2-dihydro-2,2'-binaphthalene",
     "P-54.4.3.1 :24254 (PIN)"),
    ("tetrahydro-2,2'-binaphthalene", "C1=C(C2C=c3ccccc3=CC2)Cc2ccccc2C1",
     "1,2',3',4-tetrahydro-2,2'-binaphthalene", "P-31.2.3.3.5.2 :17161 (PIN), both rings hydro"),
]

#: names the same rules give for molecules the book prints no name of (each read back by OPSIN below)
DERIVED = [
    # the gold row SUBST-01a: (a):17079 with (e)(i): the hydro prefixes take the lowest locants,
    # {2,3,4,5} (the double bond C1=C6) rather than {3,4,5,6} (C1=C2)
    ("cyclohexenylbenzene", "c1ccccc1C1=CCCCC1", "2,3,4,5-tetrahydro-1,1'-biphenyl"),
    #:15593 with:15634 '2H-1,2'-bipyridine (PIN)': the junction nitrogen carries no double bond
    ("piperidin-1-yl-pyridine", "c1ccc(N2CCCCC2)nc1", "3,4,5,6-tetrahydro-2H-1,2'-bipyridine"),
    # (e)(i) before (f): the hydro prefixes {1,2,3,4} (substituent at 6) before {1,4,5,6} (at 2)
    ("butyl-on-hydro-ring", "CCCCC1=CCCCC1c1ccccc1", "6-butyl-1,2,3,4-tetrahydro-1,1'-biphenyl"),
    #:16872: alphabetized prefixes first, the hydro prefixes right before the parent; the
    # prefixes of the benzene ring (primed) take {2',5'} (not {3',6'})
    ("methoxy-methyl", "CC1=C(C=C(C=C1)OC)C2=CCCCC2", "5'-methoxy-2'-methyl-2,3,4,5-tetrahydro-1,1'-biphenyl"),
    ("chloro", "Clc1ccc(cc1)C1=CCCCC1", "4'-chloro-2,3,4,5-tetrahydro-1,1'-biphenyl"),
    # the nitro group is a prefix (its charges are RDKit's drawing of it), (f) with (e)(i) as above
    ("nitro", "[O-][N+](=O)c1ccc(cc1)C1=CCCCC1", "4'-nitro-2,3,4,5-tetrahydro-1,1'-biphenyl"),
    #:15567 "Lowest possible locants must be used to denote the positions of attachment": (3,4')
    # before (4,3'), so the pyridine ring is unprimed
    ("3,4'-bipyridine", "c1ccncc1C1CCNCC1", "1',2',3',4',5',6'-hexahydro-3,4'-bipyridine"),
    ("bifuran", "c1ccoc1C1CCCO1", "2,3,4,5-tetrahydro-2,2'-bifuran"),
    # indicated hydrogen before hydro prefixes (:16880), cited at the front of the assembly (:15593)
    ("bipyrrole", "c1cc[nH]c1C1CCCN1", "2,3,4,5-tetrahydro-1H,1'H-2,2'-bipyrrole"),
    ("biindole", "c1ccc2[nH]c(C3Cc4ccccc4N3)cc2c1", "2,3-dihydro-1H,1'H-2,2'-biindole"),
    ("biquinoline", "c1ccc2nc(C3CCc4ccccc4N3)ccc2c1", "1,2,3,4-tetrahydro-2,2'-biquinoline"),
]

ROWS = [(r[0], r[1], r[2]) for r in BOOK] + DERIVED
IDS = [r[0] for r in ROWS]

#: names of the same molecules with higher locants: OPSIN reads each back, so only the rule picks
RIVALS = [
    ("c1ccccc1C1=CCCCC1", ["3,4,5,6-tetrahydro-1,1'-biphenyl", "2',3',4',5'-tetrahydro-1,1'-biphenyl"]),
    ("c1ccccc1C1=CC=CCC1", ["5,6-dihydro-1,1'-biphenyl"]),
    ("c1ccc(nc1)C1CCCCN1", ["1',2',3',4',5',6'-hexahydro-2,2'-bipyridine"]),
    ("c1ccncc1C1CCNCC1", ["1,2,3,4,5,6-hexahydro-4,3'-bipyridine"]),
]


def _detect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return detect_hydro_ring_assembly(mol, get_ring_systems(mol))


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    with jvm_slots(1, purpose="leads-R2A-test"):
        yield


# ------------------------------------------------------------------ the producer on its own
@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_producer_builds_the_name_and_certifies_it(_id, smiles, pin):
    info = _detect(smiles)
    assert info is not None and info["hydro"] is True
    assert info["name"] == pin
    assert info["count"] == 2 and info["double_bond_junction"] is False
    # independent of the producer's own check: a fresh OPSIN call, full standard InChIKey
    assert_full_rt(pin, smiles)


@pytest.mark.parametrize("smiles,rivals", RIVALS)
def test_a_rival_with_higher_locants_is_a_name_of_the_molecule_but_not_the_pin(smiles, rivals):
    chosen = _detect(smiles)["name"]
    for rival in rivals:
        assert rival != chosen
        assert_full_rt(rival, smiles)


@pytest.mark.parametrize("_id,smiles,pin", ROWS, ids=IDS)
def test_the_name_does_not_depend_on_the_spelling_of_the_smiles(_id, smiles, pin):
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(20261010)
    for _ in range(12):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        shuffled = Chem.RenumberAtoms(mol, order)
        spelled = Chem.MolToSmiles(shuffled, canonical=False, doRandom=True)
        again = Chem.MolFromSmiles(spelled)
        info = detect_hydro_ring_assembly(again, get_ring_systems(again))
        assert info is not None and info["name"] == pin, spelled


def test_the_book_row_of_the_gold_set_is_named():
    # SUBST-01a: 'c1ccccc1C1=CCCCC1' expects "2,3,4,5-tetrahydro-1,1'-biphenyl" (packs/rings_numbering.json)
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "benchmarks/pin_oracle"
    rows = json.loads((root / "packs/rings_numbering.json").read_text())["rows"]
    row = next(r for r in rows if r["def_id"] == "SUBST-01a" and r["smiles"] == "c1ccccc1C1=CCCCC1")
    assert _detect(row["smiles"])["name"] == row["expected_pin"]


# ------------------------------------------------------------------ what is declined
DECLINED = [
    #:24153,:24157: the one two-ring assembly that is named substitutively, substituted or not
    ("cyclohexylbenzene", "C1CCC(CC1)c1ccccc1"),
    ("methylcyclohexylbenzene", "Cc1ccc(cc1)C1CCCCC1"),
    # identical rings: detect_ring_assembly names them
    ("biphenyl", "c1ccc(cc1)-c1ccccc1"),
    ("bipyridine", "c1ccc(nc1)-c1ccccn1"),
    # two mancude components, one in an indicated-hydrogen state: the path of detect_ring_assembly
    ("2H-1,2'-bipyridine", "C1C=CC=CN1c1ccccn1"),
    # no mancude ring left: for rings the saturated assembly and 'ene' (:17079)
    ("cyclohexene-cyclohexane", "C1=CCCCC1C1CCCCC1"),
    # different skeletons are no ring assembly:15550)
    ("cyclohexenylpyridine", "C1=CCCCC1c1ccccn1"),
    ("cyclohexylpyridine", "C1CCCCC1c1ccccn1"),
    # three ring systems: composite locants:24078), not built here
    ("terphenyl-like", "c1ccc(cc1)C1=CCCCC1c1ccccc1"),
    # no single bond joins two ring systems
    ("spiro", "C1CCC2(CC1)CCCCC2"),
    ("fused", "C1CCC2CCCCC2C1"),
    # stereo, isotope, charge, an oxo group, and substituents that are a suffix or not simple
    ("stereo", "c1ccc(nc1)[C@@H]1CCCCN1"),
    ("isotope", "[2H]C1=C(c2ccccc2)CCCC1"),
    ("charged", "[CH+]1CCCC=C1c1ccccc1"),
    ("oxo", "O=C1CCCC=C1c1ccccc1"),
    ("hydroxy", "Oc1ccc(cc1)C1=CCCCC1"),
    ("carboxylic acid", "OC(=O)c1ccc(cc1)C1=CCCCC1"),
    ("amino", "Nc1ccc(cc1)C1=CCCCC1"),
    ("branched alkyl", "CC(C)c1ccc(cc1)C1=CCCCC1"),
    ("ester", "CC(=O)OCC1=CCCCC1c1ccccc1"),
]


@pytest.mark.parametrize("_id,smiles", DECLINED, ids=[d[0] for d in DECLINED])
def test_the_producer_declines(_id, smiles):
    assert _detect(smiles) is None


@pytest.mark.parametrize("smiles", ["C1CCC(CC1)c1ccccc1", "Cc1ccc(cc1)C1CCCCC1"])
def test_the_benzene_cyclohexane_exception_does_not_need_aromatic_flags(smiles):
    # the exception of (:24153) is read from the Kekule structure as well, so a mol without
    # aromatic flags is no hexahydro-biphenyl
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    assert ring_assemblies._hydro_name(mol, get_ring_systems(mol)) is None


def test_a_molecule_of_several_fragments_is_declined():
    assert _detect("c1ccccc1C1=CCCCC1.O") is None


# ------------------------------------------------------------------ nothing ships that no round trip confirms
def test_a_name_that_opsin_cannot_read_back_is_not_offered(monkeypatch):
    import orthonym.validation.opsin_roundtrip as rt

    def unavailable(name, jar_version="2.9.0"):
        raise OpsinUnavailable("test")

    monkeypatch.setattr(rt, "extended_smiles_or_unavailable", unavailable)
    assert _detect("c1ccccc1C1=CCCCC1") is None


def test_a_name_that_opsin_reads_as_another_structure_is_not_offered(monkeypatch):
    import orthonym.validation.opsin_roundtrip as rt

    monkeypatch.setattr(rt, "extended_smiles_or_unavailable",
                        lambda name, jar_version="2.9.0": "c1ccccc1C1=CC=CCC1 |$_AV:1;2$|")
    assert _detect("c1ccccc1C1=CCCCC1") is None


def test_a_name_that_opsin_rejects_is_not_offered(monkeypatch):
    import orthonym.validation.opsin_roundtrip as rt

    monkeypatch.setattr(rt, "extended_smiles_or_unavailable", lambda name, jar_version="2.9.0": None)
    assert _detect("c1ccccc1C1=CCCCC1") is None


def test_the_detector_never_raises_on_a_structure_it_cannot_read(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("test")

    monkeypatch.setattr(ring_assemblies, "_hydro_name", boom)
    assert _detect("c1ccccc1C1=CCCCC1") is None


def test_the_consumers_of_detect_ring_assembly_never_see_a_hydro_assembly():
    # the prefix builders of ring fragments name every ring system alike; a hydro assembly must not reach them
    mol = Chem.MolFromSmiles("c1ccccc1C1=CCCCC1")
    assert ring_assemblies.detect_ring_assembly(mol, get_ring_systems(mol)) is None
    info = _detect("c1ccccc1C1=CCCCC1")
    assert ring_assemblies.name_ring_assembly_prefix(mol, info, 6) is None
    assert ring_assemblies.name_ring_assembly(mol, info, None) == "2,3,4,5-tetrahydro-1,1'-biphenyl"


# ------------------------------------------------------------------ isotope labels (the producer)
def test_nothing_is_offered_while_the_hydro_assembly_is_withheld():
    # the decorator of isotopic names names the stripped skeleton here (see the comment at
    # ``ring_assemblies.hydro_assembly_withheld``): no offer, whatever the structure
    for _id, smiles, pin in ROWS:
        assert _detect(smiles)["name"] == pin
        with ring_assemblies.hydro_assembly_withheld():
            assert _detect(smiles) is None, _id
        assert _detect(smiles)["name"] == pin, _id


def test_the_withheld_state_is_restored_on_every_exit_and_nests():
    smiles = "c1ccccc1C1=CCCCC1"
    with pytest.raises(RuntimeError):
        with ring_assemblies.hydro_assembly_withheld():
            assert _detect(smiles) is None
            raise RuntimeError("test")
    assert _detect(smiles) is not None
    with ring_assemblies.hydro_assembly_withheld():
        with ring_assemblies.hydro_assembly_withheld():
            assert _detect(smiles) is None
        assert _detect(smiles) is None      # the inner exit restores the outer state, not the default
    assert _detect(smiles) is not None


# ------------------------------------------------------------------ the engine
pytestmark = pytest.mark.opsin_gate           # the labels are the production ones only with the gate on

ENGINE = [r for r in ROWS if r[0] in (
    "dihydro-biphenyl", "hexahydro-2,2'-bipyridine", "tetrahydro-1,2'-binaphthalene", "cyclohexenylbenzene",
    "piperidin-1-yl-pyridine", "butyl-on-hydro-ring", "methoxy-methyl", "chloro", "bifuran", "bipyrrole")]


@needs_wiring
@pytest.mark.parametrize("_id,smiles,pin", ENGINE, ids=[r[0] for r in ENGINE])
def test_the_engine_ships_the_pin_at_both_tiers(_id, smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_cyclohexylbenzene_is_unchanged_and_the_exception_stays_substitutive():
    assert_pin_at_both_tiers("C1CCC(CC1)c1ccccc1", "cyclohexylbenzene")


@needs_wiring
def test_the_gate_row_function_matches_the_gold_row_subst_01a():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    import pin_conformance_eval as gate
    packs = gate.load_packs(root / "benchmarks/pin_oracle/packs", [])
    row = next(r for rows in (packs.values() if isinstance(packs, dict) else (p[1] for p in packs))
               for r in rows if r.get("def_id") == "SUBST-01a" and r["smiles"] == "c1ccccc1C1=CCCCC1")
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    record = gate._name_row("rings_numbering", row, 120, False, Orthonym(), None, opsin_roundtrip_check)
    assert record["shipped_name"] == "2,3,4,5-tetrahydro-1,1'-biphenyl"
    assert record["verdict"] == "MATCH"


@pytest.mark.parametrize("smiles", [
    "O=C1CCCC=C1c1ccccc1", "Oc1ccc(cc1)C1=CCCCC1", "OC(=O)c1ccc(cc1)C1=CCCCC1",
    "CC(=O)OCC1=CCCCC1c1ccccc1"])
def test_a_molecule_the_producer_declines_keeps_a_name_that_reads_back(smiles):
    # the group is out of scope for the producer: the producers the molecule had still name it at the
    # best-effort tier, and the name still reads back to the structure (best-effort never goes down)
    from orthonym.cli import _emit_tier_flags
    best = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert best["name"] and name_is_rt_exact(best["name"], smiles), best


#: isotope-labelled molecules of the class. Named through the stripped skeleton; each is named at the
#: best-effort tier today (v1.0.6 and leads-int), by the producers that stay open to a label.
LABELLED_RING = [
    ("D on the benzene ring", "[2H]c1ccccc1C1=CCCCC1"),
    ("D on the cyclohexene ring", "[2H]C1=C(c2ccccc2)CCCC1"),
    ("D on the pyridine ring", "[2H]c1ccc(nc1)C1CCCCN1"),
    ("13C on the cyclohexadiene ring", "C1=CC=CC[13CH]1c1ccccc1"),
]
LABELLED_CHAIN = [("13C on a substituent chain", "[13CH3]CCc1ccc(cc1)C1=CCCCC1")]


def _assert_labelled_molecule_is_named(smiles):
    # the name that ships carries the label: the full InChIKey of the read-back, isotope layer included, is
    # the input's
    from orthonym.cli import _emit_tier_flags
    from orthonym.errors import is_failure_name
    best = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert best["name"] and not is_failure_name(best["name"]), best
    assert_full_rt(best["name"], smiles)
    pin = Orthonym(style="pin").name_tiered(smiles)
    if pin["name"] and not is_failure_name(pin["name"]):
        assert_full_rt(pin["name"], smiles)    # what the PIN tier ships is the molecule too


@needs_label_wiring
@pytest.mark.parametrize("_id,smiles", LABELLED_RING, ids=[x[0] for x in LABELLED_RING])
def test_a_ring_label_keeps_a_name_at_the_best_effort_tier(_id, smiles):
    # the skeleton of a labelled molecule is named without the hydro producer: a name of the producer takes
    # no descriptor for a ring atom, and the molecule would be an abstention (best-effort never goes down)
    _assert_labelled_molecule_is_named(smiles)


@pytest.mark.parametrize("_id,smiles", LABELLED_CHAIN, ids=[x[0] for x in LABELLED_CHAIN])
def test_a_substituent_label_keeps_a_name_at_the_best_effort_tier(_id, smiles):
    _assert_labelled_molecule_is_named(smiles)


@needs_wiring
def test_a_labelled_molecule_does_not_change_what_the_next_molecule_is_named():
    # the withheld state is the decorator's alone: it is not left set (a name never depends on what the
    # process named before), and no classification of the labelled skeleton is kept for the unlabelled one
    pin = "2,3,4,5-tetrahydro-1,1'-biphenyl"
    namer = Orthonym()
    assert namer.name("c1ccccc1C1=CCCCC1") == pin
    namer.name_tiered("[2H]c1ccccc1C1=CCCCC1")
    assert namer.name("c1ccccc1C1=CCCCC1") == pin
    fresh = Orthonym()
    fresh.name_tiered("[2H]C1=C(c2ccccc2)CCCC1")
    assert fresh.name("c1ccccc1C1=CCCCC1") == pin
    assert ring_assemblies._HYDRO_WITHHELD.get() is False


def test_the_gold_row_is_on_the_floor_again():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "benchmarks/pin_oracle"
    data = json.loads((root / "baseline_passing_rows.json").read_text())
    assert ["SUBST-01a", "c1ccccc1C1=CCCCC1"] in data["passing_target_rows"]
    assert len(data["passing_target_rows"]) == data["_meta"]["n_rows"] == 1662
