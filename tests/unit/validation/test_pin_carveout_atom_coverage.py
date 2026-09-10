""" a phase — PIN carve-out decoration-mutation atom-coverage certificate.

``namer._final_opsin_validity_gate`` (:1081-1353) has 9 hard-gated ``return name``
carve-outs (:1152-1237) that ship a PIN WITHOUT the /OPSIN round-trip,
because the correct PIN for each class is OPSIN-unparseable by construction
(OPSIN's *generation* grammar has no rule for these suffix/class-word families):
``thioperoxol``, ``inositol``, ``np_stereoparent``, ``dianhydride``,
``chalcogen_dianhydride``, ``polyol_polyester``, ``organometallic_additive``,
``phane``, ``halogen_uide``. Each is commented as "emitted ONLY by a hard-gated /
atom-conservation-vetoed" internal producer, i.e. the atom-coverage claim for
these classes rests entirely on trusting that producer, with no external check
of any kind — see `internal notes` Gap 2.

The historical failure mode this guards against (fixed for cholesteryl sulfate
-> ``cholest-5-ene``, but never adversarially re-probed): a recogniser fires on
a molecule that CONTAINS its recognised core plus extra decoration atoms, and
names only the bare core -> a silent atom-drop that the carve-out would wave
through unexamined (no ever runs on that branch).

This module is a REGRESSION CERTIFICATE, not a source change: for each of the
9 classes it (1) locks a representative PIN molecule shipping via that exact
carve-out (``gate_outcome == "carveout:<class>"``), then (2) builds a DECORATED
mutant -- the same recognised core plus one small extra substituent that
changes the molecular formula -- and proves the decoration is never silently
dropped: the decorated molecule must never reproduce the bare-core name.

Three honest outcomes for the decorated mutant, all acceptable:
  (a) VERIFIED -- the decoration broke the carve-out's own hard gate
      (regex / exact-SMILES / exact-atom-count match), the molecule routed to
      an ordinary producer, and OPSIN's round-trip independently
      proved the emitted name's skeleton matches the input -- the strongest
      possible proof, and observed for inositol + dianhydride below.
  (b) HONEST ABSTAIN -- the decorated molecule correctly fails closed
      (``errors.is_failure_name`` true: 'unknown...' / '...(not supported)')
      rather than guessing -- observed for np_stereoparent, chalcogen_dianhydride,
      organometallic_additive, phane, halogen_uide below.
  (c) SAME-CARVEOUT, ATOM-COMPLETE -- the decorated molecule still matches the
      carve-out's own regex (thioperoxol / polyol_polyester below), because
      OPSIN can NEVER round-trip ANY member of these two classes, decorated or
      not -- that unparseability is the entire reason the carve-out exists, so
      "OPSIN-verify the decorated name" is not an available check for this
      residual set. The producer nonetheless computed a real substitutive name
      (new locants, a real substituent prefix for the added atoms), so the
      textual proxy checked here is: the decorated name is NOT the bare name,
      AND the added substituent's IUPAC marker token (e.g. "chloro") appears
      in it.

The ONE failure shape this certificate exists to catch, and did not find live
in any of the 9 cases (see the module docstring in the phase report for the
one adjacent-but-different anomaly that WAS found and is reported, not
xfailed, because it is a locant-formatting defect, not an atom drop):
decorated_name == bare_core_name, i.e. the recogniser fired on the bigger
molecule and reproduced the smaller molecule's name byte-for-byte.

Scope discipline: this file touches NO ``src/`` module. It cannot regress the
1652 PIN gate. Run in isolation only (never the whole suite -- the OPSIN pipe
deadlocks):
    .venv/bin/python -m pytest tests/unit/validation/test_pin_carveout_atom_coverage.py -v
"""
import pytest

from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym

# This whole module is ABOUT the OPSIN validity gate's carve-out branches, so
# it must run gate-ON (tests/conftest.py defaults the gate OFF for speed and
# would otherwise be green-but-blind -- see conftest.py's "never silently
# blind" block). `pytest_runtest_call` there skips these tests outright if no
# OPSIN jar is reachable, rather than passing them unverified.
pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# ---------------------------------------------------------------------------
# The 9 carve-out classes. Each representative was independently confirmed
# (2026-08-14) to ship via `gate_outcome == "carveout:<carveout>"` -- i.e. it
# is the code path at namer.py:1152-1237, not merely a name that happens to
# look similar.
#
# `outcome_kind` documents which of the three honest decorated-mutant outcomes
# ((a) self01 / (b) abstain / (c) same-carveout-marker) was OBSERVED for the
# constructed mutant, so the assertion helper knows which proof to demand.
# ---------------------------------------------------------------------------
CASES = [
    dict(
        carveout="thioperoxol",
        bare_smiles="CSO",
        bare_name="methane-SO-thioperoxol",
        # BB verbatim: CH3-S-OH -> methane-SO-thioperoxol (PIN).
        decorated_smiles="ClCCSO",  # Cl-CH2-CH2-S-OH: same -S-OH core, +Cl +C.
        outcome_kind="carveout_marker",
        decoration_marker="chloro",
    ),
    dict(
        carveout="inositol",
        bare_smiles="O[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        bare_name="myo-inositol",
        # is_inositol_skeleton hard-gates on EXACTLY 12 heavy atoms (C6H12O6);
        # an O-methyl ether on one ring position is 13 heavy atoms -> the hard
        # gate structurally cannot fire, so this exercises the "recogniser
        # correctly declines, molecule routes elsewhere" path.
        decorated_smiles="CO[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        outcome_kind="self01",
    ),
    dict(
        carveout="np_stereoparent",
        bare_smiles="CC(C)[C@H]1CC[C@H]2[C@@H](CC[C@H]3C(C)(C)CCC[C@]23C)C1",
        bare_name="abietane",
        # get_natural_product_name keys on EXACT canonical SMILES; a
        # hydroxymethyl in place of one terminal methyl cannot match the
        # abietane entry, so name_inositol-style exact lookup structurally
        # cannot fire either -> exercises the "falls through, general engine
        # cannot handle this fused polycyclic substituent, fails closed"
        # path (the general-engine gap that surfaces is unrelated to this
        # carve-out and is NOT this test's concern -- what matters is it
        # never re-emits the bare "abietane").
        decorated_smiles="OCC(C)[C@H]1CC[C@H]2[C@@H](CC[C@H]3C(C)(C)CCC[C@]23C)C1",
        outcome_kind="abstain",
    ),
    dict(
        carveout="dianhydride",
        bare_smiles="CC(=O)OC(=O)CCC(=O)OC(C)=O",
        bare_name="diacetic butanedioic dianhydride",
        # _DIANHYDRIDE_PIN_RE requires a straight '...(di|tri|...)anhydride'
        # tail; a chloro substituent on the central succinic chain breaks
        # _name_dianhydride's own pattern, so the molecule falls through to
        # the ordinary mixed-anhydride/acyloxy substitutive namer -- a
        # DIFFERENT, OPSIN-parseable name that verifies directly.
        decorated_smiles="CC(=O)OC(=O)C(Cl)CC(=O)OC(C)=O",
        outcome_kind="self01",
    ),
    dict(
        carveout="chalcogen_dianhydride",
        bare_smiles="CC(=O)SC(=O)CCC(=O)SC(C)=O",
        bare_name="diacetic butanedioic bis(thioanhydride)",
        # Same chloro-on-the-central-chain decoration as dianhydride above,
        # sulfur analogue; here the fallback substitutive path has no
        # acylthio-ester namer wired, so it fails closed rather than
        # producing a parseable alternative -- still an honest abstain, never
        # the bare thioanhydride name.
        decorated_smiles="CC(=O)SC(=O)C(Cl)CC(=O)SC(C)=O",
        outcome_kind="abstain",
    ),
    dict(
        carveout="polyol_polyester",
        bare_smiles="CC(=O)OCC(COC(C)=O)OC(=O)CC",
        bare_name="propane-1,2,3-triyl 1,3-diacetate 2-propanoate",
        # A chloro substituent on the (unsubstituted-in-the-bare-case)
        # propanoate acyl group: _POLYOL_POLYESTER_PIN_RE only requires the
        # multi-ester SHAPE (triyl parent + >=2 locant-prefixed '...ate'
        # words), which the decorated acyl word still satisfies, so this is
        # the other OPSIN-can-never-verify-this-class residual case.
        decorated_smiles="CC(=O)OCC(COC(C)=O)OC(=O)C(Cl)C",
        outcome_kind="carveout_marker",
        decoration_marker="chloro",
    ),
    dict(
        carveout="organometallic_additive",
        bare_smiles="C[Ti](Cl)(Cl)Cl",
        bare_name="trichlorido(methyl)titanium",
        # Swapping the methyl ligand for 2-chloroethyl exceeds the additive
        # ligand namer's substituent-recursion depth (observed: "
        # substituent_skip: reason=recursion_depth_fallback") -> honest
        # generic-metal descriptive fallback, never the bare methyl name.
        decorated_smiles="ClCC[Ti](Cl)(Cl)Cl",
        outcome_kind="abstain",
    ),
    dict(
        carveout="phane",
        bare_smiles="C1Cc2ccc(cc2)CCc2ccc1cc2",
        bare_name="1,4(1,4)-dibenzenacyclohexaphane",
        # build_phane_pin requires "all-identical-benzene amplificants"; a
        # bromo substituent on one ring breaks that symmetry requirement, so
        # the recogniser correctly declines rather than naming the bare
        # unsubstituted phane and dropping the Br.
        decorated_smiles="C1Cc2cc(Br)ccc2CCc2ccc1cc2",
        outcome_kind="abstain",
    ),
    dict(
        carveout="halogen_uide",
        bare_smiles="[I-](c1ccccc1)c1ccccc1",
        bare_name="diphenyliodanuide",
        # A 4-methyl on one phenyl: _emit_group13_uide DOES compute a real,
        # atom-complete candidate here ("(4-methylphenyl)phenyliodanuide" --
        # confirmed via direct inspection of the pre-gate warning), so the
        # methyl is NOT silently dropped at the producer. It then fails
        # closed to an honest abstain only because _HALOGEN_UIDE_PIN_RE
        # requires the name to START with a bare `[a-z]` and this candidate
        # starts with '(' -- a separate, narrower, reportable coverage gap
        # in the carve-out's OWN regex (fails closed, not open: no wrong
        # molecule ships). See the phase report for this named separately;
        # it is not the atom-drop shape this certificate targets.
        decorated_smiles="[I-](c1ccccc1)c1ccc(C)cc1",
        outcome_kind="abstain",
    ),
]

_IDS = [c["carveout"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_carveout_baseline_ships_expected_pin(namer, case):
    """Locks the carve-out still fires: the representative ships its exact
    expected PIN, and the gate outcome is the carve-out itself (not some
    other path that happens to produce the same string)."""
    result = namer.name_tiered(case["bare_smiles"])
    assert result["name"] == case["bare_name"], (
        f"{case['carveout']}: baseline SMILES {case['bare_smiles']!r} no longer "
        f"ships the expected carve-out PIN. Got {result['name']!r}."
    )
    assert result.get("gate_outcome") == f"carveout:{case['carveout']}", (
        f"{case['carveout']}: baseline shipped the right STRING but not via "
        f"its carve-out (gate_outcome={result.get('gate_outcome')!r}) -- the "
        f"representative no longer proves this carve-out is live."
    )


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_carveout_decoration_mutation_no_atom_drop(namer, case):
    """The real test: a decorated (core + extra atoms) variant must NEVER
    reproduce the bare-core carve-out name -- that would mean the recogniser
    fired on the bigger molecule and silently dropped the decoration, exactly
    the historical cholesteryl-sulfate-> cholest-5-ene failure mode this
    certificate exists to catch."""
    result = namer.name_tiered(case["decorated_smiles"])
    name = result["name"]
    gate_outcome = result.get("gate_outcome")

    # The one invariant that matters for every case, regardless of outcome
    # kind: never re-emit the bare core name for a strictly bigger molecule.
    assert name != case["bare_name"], (
        f"LIVE ATOM-DROP in {case['carveout']!r}: decorated molecule "
        f"{case['decorated_smiles']!r} (a superset of the {case['bare_smiles']!r} "
        f"core plus extra atoms) shipped the BARE carve-out name "
        f"{case['bare_name']!r} byte-for-byte -- the decoration was silently "
        f"dropped. gate_outcome={gate_outcome!r}."
    )

    kind = case["outcome_kind"]

    if kind == "abstain":
        assert is_failure_name(name), (
            f"{case['carveout']}: decorated mutant did not abstain honestly "
            f"and did not name the whole molecule either -- got {name!r} "
            f"(gate_outcome={gate_outcome!r}). Expected an 'unknown...' / "
            f"'...(not supported)' fail-closed signal."
        )
        return

    if kind == "self01":
        # Strongest proof available: (InChIKey-skeleton compare
        # against the ORIGINAL input) already ran and passed -- this is
        # exactly E1's atom-coverage guarantee, just derived from (mol, name)
        # instead of a TokenBinding partition (see module docstring/audit).
        assert not is_failure_name(name), (
            f"{case['carveout']}: expected a real self01-verified name, got "
            f"an abstain {name!r} instead (gate_outcome={gate_outcome!r})."
        )
        assert gate_outcome in (
            "self_consistency_verified", "self_consistency_constitution_only",
        ), (
            f"{case['carveout']}: decorated mutant emitted a real name "
            f"{name!r} but it was NOT independently OPSIN/SELF-01 verified "
            f"(gate_outcome={gate_outcome!r}) -- atom coverage is unproven."
        )
        return

    assert kind == "carveout_marker"
    # OPSIN can never round-trip ANY member of this class (that unparseability
    # is the whole reason the carve-out exists) -- decorated or not. The best
    # available proof without touching src is: it is a REAL name (not an
    # abstain), it fired via the SAME carve-out (confirming the producer, not
    # some unrelated path, computed it), and the added substituent's IUPAC
    # marker token is textually present -- i.e. the decoration is accounted
    # for in the string, not silently absent.
    assert not is_failure_name(name), (
        f"{case['carveout']}: decorated mutant abstained ({name!r}) rather "
        f"than emitting a real name; expected the producer to still compute "
        f"a decorated PIN via the same OPSIN-unparseable class."
    )
    assert gate_outcome == f"carveout:{case['carveout']}", (
        f"{case['carveout']}: decorated mutant's real name {name!r} did not "
        f"ship via the expected carve-out (gate_outcome={gate_outcome!r})."
    )
    marker = case["decoration_marker"]
    assert marker in name, (
        f"LIVE ATOM-DROP in {case['carveout']!r}: decorated molecule "
        f"{case['decorated_smiles']!r} shipped {name!r}, which does not even "
        f"textually mention the added {marker!r} substituent -- the "
        f"decoration appears to have been dropped."
    )
