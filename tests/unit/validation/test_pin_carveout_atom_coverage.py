""" a phase — PIN carve-out decoration-mutation atom-coverage certificate.

``namer._final_opsin_validity_gate`` (:1081-1353) has hard-gated ``return name``
carve-outs (:1152-1237) that ship a PIN WITHOUT the /OPSIN round-trip (``halogen_uide`` ships
through its twin read-back since the HU topic: ``validation.uide_twin``,
``tests/unit/rules/test_halogen_uide_names.py``; with the CO topic ``thioperoxol`` ships through
``validation.thioperoxol_twin`` (``tests/unit/validation/test_thioperoxol_twin.py``) and ``dianhydride`` and
``chalcogen_dianhydride`` when ``validation.anhydride_acids`` reads the name back through its acids
(``tests/unit/rules/test_dianhydride_names.py``)),
because the correct PIN for each class is OPSIN-unparseable by construction
(OPSIN's *generation* grammar has no rule for these suffix/class-word families):
``inositol``, ``np_stereoparent``, ``polyol_polyester``,
``organometallic_additive``, ``phane``. Each is commented as "emitted ONLY by a hard-gated /
atom-conservation-vetoed" internal producer, i.e. the atom-coverage claim for
these classes rests entirely on trusting that producer, with no external check
of any kind — see `internal notes` Gap 2.

The historical failure mode this guards against (fixed for cholesteryl sulfate
-> ``cholest-5-ene``, but never adversarially re-probed): a recogniser fires on
a molecule that CONTAINS its recognised core plus extra decoration atoms, and
names only the bare core -> a silent atom-drop that the carve-out would wave
through unexamined (no ever runs on that branch).

This module is a REGRESSION CERTIFICATE, not a source change: for each of the
7 classes it (1) locks a representative PIN molecule shipping via that exact
carve-out (``gate_outcome == "carveout:<class>"``), then (2) builds a DECORATED
mutant -- the same recognised core plus one small extra substituent that
changes the molecular formula -- and proves the decoration is never silently
dropped: the decorated molecule must never reproduce the bare-core name.

Three honest outcomes for the decorated mutant, all acceptable:
  (a) VERIFIED -- the decoration broke the carve-out's own hard gate
      (regex / exact-SMILES / exact-atom-count match), the molecule routed to
      an ordinary producer, and OPSIN's round-trip independently
      proved the emitted name's skeleton matches the input -- the strongest
      possible proof, and observed for inositol, dianhydride + np_stereoparent below.
  (b) HONEST ABSTAIN -- the decorated molecule correctly fails closed
      (``errors.is_failure_name`` true: 'unknown...' / '...(not supported)')
      rather than guessing -- observed for chalcogen_dianhydride,
      organometallic_additive, phane below.
  (c) SAME-CARVEOUT, ATOM-COMPLETE -- the decorated molecule still matches the
      carve-out's own regex (polyol_polyester below), because
      OPSIN can NEVER round-trip ANY member of this class, decorated or
      not -- that unparseability is the entire reason the carve-out exists, so
      "OPSIN-verify the decorated name" is not an available check for this
      residual set. The producer nonetheless computed a real substitutive name
      (new locants, a real substituent prefix for the added atoms), so the
      textual proxy checked here is: the decorated name is NOT the bare name,
      AND the added substituent's IUPAC marker token (e.g. "chloro") appears
      in it.

The ONE failure shape this certificate exists to catch, and did not find live
in any of the 7 cases (see the module docstring in the phase report for the
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
from tests.support.opsin_marks import release_opsin_only
from tests.support.np_producer import np_producer_only  # noqa: F401

# NPL 2026-11-01: the subject is the natural-product producer's own name; the route now offers the systematic
# name first INTRODUCTION, the Blue Book), so the producer is named with that offer off.

# This whole module is ABOUT the OPSIN validity gate's carve-out branches, so
# it must run gate-ON (tests/conftest.py defaults the gate OFF for speed and
# would otherwise be green-but-blind -- see conftest.py's "never silently
# blind" block). `pytest_runtest_call` there skips these tests outright if no
# OPSIN jar is reachable, rather than passing them unverified.
pytestmark = [pytest.mark.opsin_gate, pytest.mark.usefixtures("np_producer_only")]


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# ---------------------------------------------------------------------------
# The 7 carve-out classes. Each representative was independently confirmed
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
        # cannot fire either. The chain is then the parent and the decorated
        # fused ring system its '-yl' prefix (DT4 F2b): '2-[(2S,4aS,4bR,8aS,
        # 10aS)-4b,8,8-trimethyltetradecahydrophenanthren-2-yl]propan-1-ol',
        # which OPSIN reads back to the full input InChIKey -- what matters is
        # it never re-emits the bare "abietane".
        decorated_smiles="OCC(C)[C@H]1CC[C@H]2[C@@H](CC[C@H]3C(C)(C)CCC[C@]23C)C1",
        outcome_kind="self01",
    ),
    dict(
        carveout="polyol_polyester",
        bare_smiles="CC(=O)OCC(COC(C)=O)OC(=O)CC",
        bare_name="propane-1,2,3-triyl 1,3-diacetate 2-propanoate",
        # A chloro substituent on the (unsubstituted-in-the-bare-case)
        # propanoate acyl group. The name is OPSIN-unparseable (no multi-anion
        # grammar) and, since the CO topic, ships through its acyloxy twin
        # (validation.polyester_twin: '1,3-bis[acetyloxy]-2-[propanoyloxy]propane',
        # the substitutive spelling of the same ester, method (2),
        # the Blue Book,:31840) read back to the FULL InChIKey of the
        # input: the gate outcome is the twin's, and the decorated name is
        # proven complete by the same read-back (the marker check stays).
        decorated_smiles="CC(=O)OCC(COC(C)=O)OC(=O)C(Cl)C",
        outcome_kind="carveout_marker",
        decoration_marker="chloro",
        gate="twin_verified:polyester_acyloxy",
    ),
    dict(
        carveout="organometallic_additive",
        bare_smiles="C=C.C=C.[Ni]",
        bare_name="bis(\u03b7\u00b2-ethene)nickel",
        # Since the CO topic the '-ido' names ('trichlorido(methyl)titanium') ship through their 'chloro'
        # twin (validation.halido_twin; tests/unit/rules/test_halido_twin.py) and the carve-out holds the
        # names with an eta, kappa or mu descriptor only, for which no spelling OPSIN reads exists.
        # Replacing one ethene by chloroethene is not an eta-ethene ligand the additive namer names
        # (observed: "nickel compound (not supported)") -> honest abstain, never the bare name.
        decorated_smiles="ClC=C.C=C.[Ni]",
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
]

#: the phane carve-out ships the phane names OPSIN 2.9.0 cannot read; an OPSIN that reads them (main) ships the same
#: name through the self-consistency read-back, so the carve-out representative describes 2.9.0 only.
_PARAMS = [
    pytest.param(c, marks=release_opsin_only, id=c["carveout"]) if c["carveout"] == "phane"
    else pytest.param(c, id=c["carveout"])
    for c in CASES
]


@pytest.mark.parametrize("case", _PARAMS)
def test_carveout_baseline_ships_expected_pin(namer, case):
    """Locks the carve-out still fires: the representative ships its exact
    expected PIN, and the gate outcome is the carve-out itself (not some
    other path that happens to produce the same string)."""
    if case["carveout"] == "inositol":
        #:50943 (relabel 2026-10-31): the PIN tier declines an inositol; the carve-out ships it at best-effort
        from orthonym.cli import _emit_tier_flags
        namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    result = namer.name_tiered(case["bare_smiles"])
    assert result["name"] == case["bare_name"], (
        f"{case['carveout']}: baseline SMILES {case['bare_smiles']!r} no longer "
        f"ships the expected carve-out PIN. Got {result['name']!r}."
    )
    assert result.get("gate_outcome") == case.get("gate", f"carveout:{case['carveout']}"), (
        f"{case['carveout']}: baseline shipped the right STRING but not via "
        f"its carve-out (gate_outcome={result.get('gate_outcome')!r}) -- the "
        f"representative no longer proves this carve-out is live."
    )


@pytest.mark.parametrize("case", _PARAMS)
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
    assert gate_outcome == case.get("gate", f"carveout:{case['carveout']}"), (
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
