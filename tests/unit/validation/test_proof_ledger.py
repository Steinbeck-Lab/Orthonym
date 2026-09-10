""" a phase: the proof ledger asserts the spine on the FINAL name."""
import pytest
from rdkit import Chem

from orthonym.validation import binding_spine as bs
from orthonym.validation import proof_ledger as pl

pytestmark = pytest.mark.unit

ETHANOL = Chem.MolFromSmiles("CCO")


def _good_spine():
    return bs.BindingSpine(roots=(
        bs.SpineBinding(token="eth", kind=bs.BindingKind.PARENT,
                        atom_ids=frozenset([0, 1])),
        bs.SpineBinding(token="ol", kind=bs.BindingKind.SUFFIX,
                        atom_ids=frozenset([2])),
    ))


def setup_function(_):
    pl.clear_ledger()


def test_finalize_without_a_record_returns_none():
    assert pl.finalize("ethan-1-ol") is None


def test_finalize_verifies_against_the_final_name():
    pl.record_spine(ETHANOL, _good_spine(), stage="general_engine",
                    name_at_record="ethan-1-ol")
    proof = pl.finalize("ethan-1-ol")
    assert proof is not None and proof.ok, proof.findings


def test_post_processing_that_mangles_a_token_is_detected():
    pl.record_spine(ETHANOL, _good_spine(), stage="general_engine",
                    name_at_record="ethan-1-ol")
    proof = pl.finalize("propan-1-ol")   # a post-processor changed the stem
    assert not proof.ok
    assert bs.TOKEN_ABSENT in proof.codes()


def test_get_proof_reports_both_strings_for_diagnosis():
    pl.record_spine(ETHANOL, _good_spine(), stage="general_engine",
                    name_at_record="ethan-1-ol")
    pl.finalize("propan-1-ol")
    rec = pl.get_proof()
    assert rec["stage"] == "general_engine"
    assert rec["name_at_record"] == "ethan-1-ol"
    assert rec["final_name"] == "propan-1-ol"
    assert rec["ok"] is False


def test_clear_ledger_resets_everything():
    pl.record_spine(ETHANOL, _good_spine(), stage="general_engine",
                    name_at_record="ethan-1-ol")
    pl.clear_ledger()
    assert pl.finalize("ethan-1-ol") is None
    assert pl.get_proof()["stage"] is None


def test_clear_ledger_forgets_the_verdict_not_only_the_record():
    """Added by mutation check M3: clearing only ``_RECORD`` left the previous
    molecule's verdict and final name readable, so a molecule that records
    nothing would report the PREVIOUS molecule's proof -- the cross-molecule
    contamination class the per-molecule reset block exists to prevent. The
    brief's clear-ledger test could not see it (it asserts only ``stage``)."""
    pl.record_spine(ETHANOL, _good_spine(), stage="general_engine",
                    name_at_record="ethan-1-ol")
    pl.finalize("propan-1-ol")
    assert pl.get_proof()["ok"] is False, "precondition: a verdict exists"

    pl.clear_ledger()
    rec = pl.get_proof()
    assert rec["ok"] is None
    assert rec["codes"] == ()
    assert rec["final_name"] is None
    assert rec["name_at_record"] is None


def test_allow_charged_is_carried_from_the_record_into_the_verification():
    """Added by mutation check M4: hard-coding ``allow_charged=False`` in
    ``finalize`` passed every brief test, yet would raise a spurious
    NET_CHARGE_OUT_OF_SCOPE on every charged best-effort emission. Asserted
    differentially so the flag must actually be threaded, not defaulted."""
    cation = Chem.MolFromSmiles("C[NH3+]")
    spine = bs.BindingSpine(roots=(
        bs.SpineBinding(token="methan", kind=bs.BindingKind.PARENT,
                        atom_ids=frozenset([0])),
        bs.SpineBinding(token="aminium", kind=bs.BindingKind.SUFFIX,
                        atom_ids=frozenset([1])),
    ))
    pl.record_spine(cation, spine, stage="general_engine",
                    name_at_record="methanaminium", allow_charged=True)
    permitted = pl.finalize("methanaminium")
    assert bs.NET_CHARGE_OUT_OF_SCOPE not in permitted.codes()

    pl.clear_ledger()
    pl.record_spine(cation, spine, stage="general_engine",
                    name_at_record="methanaminium", allow_charged=False)
    refused = pl.finalize("methanaminium")
    assert bs.NET_CHARGE_OUT_OF_SCOPE in refused.codes()


def test_finalize_never_raises_on_an_internal_error():
    class Exploding:
        @property
        def roots(self):
            raise RuntimeError("boom")
    pl.record_spine(ETHANOL, Exploding(), stage="general_engine",
                    name_at_record="x")
    proof = pl.finalize("x")
    assert proof is not None and not proof.ok
    assert "PROOF_INTERNAL_ERROR" in proof.codes()
