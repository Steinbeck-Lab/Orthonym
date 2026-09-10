""" no-abstain Phase A: wire `validation.reconstruct.verify_or_none` onto
the best-effort/T4 UNVERIFIED emission path (`namer.py`
``Orthonym._try_general_engine_recovery``, ~:3696-3745).

Before this task, when the general engine's OWN certified name (NOT the T4
producer -- ``_cand_from_t4`` is False) was OPSIN-UNPARSEABLE and
``general_fallback_unverified`` was on, the recovery ladder shipped it
COMPLETELY UNVERIFIED (`opsin_status` stayed ``"unverified"`` with zero
proof of any kind). Measured live on a dev split (task-A-report.md A.1): 3
witnesses shipped this way, e.g.

    O=C1C=CC(=O)[C@@]23O[C@@]12C(=O)C(Cl)=CC31Oc2cccc3cccc(c23)O1
    -> (1S,6R)-8-chloro-2,5,7-trioxospiro[...pentaene] (OPSIN cannot parse it)

The fix routes this exact branch through ``verify_or_none`` before shipping:
non-None -> ship (opsin_status="verified" -- fix-round-1 Finding 3 corrected
this from the originally-shipped "verified_reconstructor", since name_facts=None
means the only reachable success is a genuine OPSIN match); None -> abstain.
``name_facts`` is None at this call site (no name->NameFacts extractor exists
yet for an arbitrary general-engine name string -- see task-A-report.md A.3),
so with the real ``verify_or_none`` this branch always abstains today; most
tests here monkeypatch ``verify_or_none`` itself to prove the WIRING (call +
ship/abstain contract) fast and deterministically, independent of that scope
limit. One test (``test_real_dev500_witness_now_abstains``) uses the REAL,
unmocked oracle against an actual a dev split witness for end-to-end confirmation.
"""
from unittest import mock

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def _force_opsin_rejects(monkeypatch):
    """Simulate 'jar present, OPSIN definitively rejects the candidate name'
    -- the exact condition the wired branch fires under -- without needing a
    live JVM."""
    import orthonym.namer as namer_mod
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(namer_mod, "_validity_gate_name_to_smiles",
                        lambda name: None)
    monkeypatch.setattr(namer_mod, "_validity_gate_status",
                        lambda name: "rejected")


def test_opsin_unparseable_engine_name_now_abstains(monkeypatch):
    """The live gap: an OPSIN-unparseable, non-T4 engine name used to ship
    with opsin_status='unverified'. With verify_or_none unable to confirm it
    (the real oracle on an unparseable name+name_facts=None, simulated here so
    the test is fast/deterministic and does not depend on a live JVM) it must
    now abstain (None), never ship bare.

    Note: `verify_or_none`'s OPSIN branch uses its OWN independent oracle
    (`validation.atom_coverage.validate_atom_coverage`), not the namer's
    `_validity_gate_*` singleton -- so faking the OUTER ladder's OPSIN check
    alone (`_force_opsin_rejects`) does not make the INNER verify_or_none call
    agree (a real name like '2-chlorobutane' genuinely parses either way).
    Mocking `verify_or_none` itself isolates exactly the wiring contract this
    test exists to check."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod
    _force_opsin_rejects(monkeypatch)
    monkeypatch.setattr(
        recon_mod, "verify_or_none",
        lambda name, input_smiles, name_facts=None: None)
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
    out = nm._try_general_engine_recovery("CC(Cl)CC")
    assert out is None, (
        f"OPSIN-unparseable engine name shipped unverified instead of "
        f"abstaining: {out!r}")


def test_opsin_unparseable_engine_name_ships_when_verify_or_none_confirms(
        monkeypatch):
    """Wiring contract: a non-None verify_or_none return DOES ship, tagged
    opsin='verified' (fix-round-1 Finding 3: with name_facts=None the only
    reachable success is a genuine OPSIN verification, never the bare
    'unverified' claim, and never mislabeled 'verified_reconstructor')."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod
    _force_opsin_rejects(monkeypatch)
    monkeypatch.setattr(
        recon_mod, "verify_or_none",
        lambda name, input_smiles, name_facts=None: name)

    from orthonym.metrics.provenance import clear_provenance, get_provenance
    clear_provenance()
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
    out = nm._try_general_engine_recovery("CC(Cl)CC")
    assert out is not None, "verify_or_none confirmed but the name did not ship"
    assert get_provenance()["opsin"] == "verified"


def test_verify_or_none_called_with_the_candidate_and_input_smiles(monkeypatch):
    """The wired call passes the CANDIDATE name and the INPUT smiles (never
    the input's graph) -- name_facts=None at this site (no extractor yet)."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod
    _force_opsin_rejects(monkeypatch)
    seen = {}

    def _spy(name, input_smiles, name_facts=None):
        seen["name"] = name
        seen["input_smiles"] = input_smiles
        seen["name_facts"] = name_facts
        return None

    monkeypatch.setattr(recon_mod, "verify_or_none", _spy)
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
    nm._try_general_engine_recovery("CC(Cl)CC")
    assert seen.get("input_smiles") == "CC(Cl)CC"
    assert seen.get("name_facts") is None
    assert isinstance(seen.get("name"), str) and seen["name"]


@pytest.mark.opsin_gate
def test_engine_name_that_opsin_can_parse_still_ships_unaffected(monkeypatch):
    """Regression guard: the NEW branch only fires when OPSIN cannot parse the
    candidate at all. When OPSIN parses and round-trips, behaviour (and the
    'verified' tag) is byte-identical to before this task."""
    smi = "O=C1CCC2CCCCC2C1"  # decalin-2-one (existing wiring test's witness)
    from orthonym.assembly.general_engine import name_general
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
    mol = Chem.MolFromSmiles(smi)
    feats = nm._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    nm._classify(feats)
    expected = name_general(mol, feats).name

    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="test-verify-or-none-regression"):
        out = nm._try_general_engine_recovery(smi)
    assert out == expected, out


@pytest.mark.opsin_gate
def test_real_dev500_witness_now_abstains():
    """End-to-end, UNMOCKED confirmation using a real a dev split witness (measured
    in task-A-report.md A.1): before this task this exact SMILES shipped

        (1S,6R)-8-chloro-2,5,7-trioxospiro[11-oxatricyclo[4.4.0.1^1,6]undeca-
        3,8-diene-2,3'-2,4-dioxatricyclo[7.3.1.0^5,13]trideca-1(12),5(13),6,8,
        10-pentaene]

    at tier=T4/source=general_engine despite being genuinely OPSIN-unparseable
    (measured via eval/harness.py --tier best-effort, outcome=opsin_parse_fail).
    With the real verify_or_none wired (name_facts=None, no extractor for this
    shape), it must now abstain rather than ship it bare."""
    smi = "O=C1C=CC(=O)[C@@]23O[C@@]12C(=O)C(Cl)=CC31Oc2cccc3cccc(c23)O1"
    from orthonym.jvm_budget import jvm_slots
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
    with jvm_slots(1, purpose="test-verify-or-none-real-witness"):
        out = nm._try_general_engine_recovery(smi)
    assert out is None, (
        f"real dev500 witness still shipped an OPSIN-unparseable name: {out!r}")
