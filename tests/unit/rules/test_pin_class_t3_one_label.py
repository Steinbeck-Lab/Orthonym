"""PIN class program, Task 3: one name, one label at both tiers (a declined charged route).

 "Peroxides, disulfides, diselenides, and ditellurides" (the Blue Book):
"Method (1), substitutive nomenclature, gives preferred IUPAC names" (:27866);
'1-[(propan-2-yl)diselanyl]propane (PIN)' (:27876). The default tier ships it at
pin_verified. The best-effort tier built the same string on the same path and labelled
it systematic_verified, source general_engine: naming the prefix, the radical
'CC(C)[Se][Se]' entered the charged route (``rules/charged_router.py:route_charged``),
whose re-entry of the neutral 'CC(C)[Se][SeH]' reached the general engine at that tier
and recorded it as the producer; the route then declined ('') and the radical namer
(``rules/radicals.py:name_radical``) built '(propan-2-yl)diselanyl' itself, but the
record of the discarded re-entry stayed on the call. A declined charged route now
leaves the producer record as it found it.

 (:31619): "Preferred IUPAC names of acid salts of organic derivatives of
polybasic inorganic oxoacids (including carbonic) are named by method (2)";
'potassium hydrogen methylphosphonate (PIN)' (:31625). Already pin_verified at both
tiers at the task's base (protection).
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("CCC[Se][Se]C(C)C", "1-[(propan-2-yl)diselanyl]propane"),           #:27876
]

CONTROL_ROWS = [
    ("CP(=O)([O-])O.[K+]", "potassium hydrogen methylphosphonate"),      #:31625
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _route_with_reentry(monkeypatch, neutral_name):
    """``route_charged`` on 'CC[NH3+]' with a re-entry that records the general engine
    as the producer (as the best-effort tier's re-entry does) and returns
    ``neutral_name``; the caller's record before the route is 'trivial_retained'."""
    from orthonym.metrics import provenance as pv
    from orthonym.rules import charged_router

    def reentry(neutral_smi, style):
        pv.record_source("general_engine", opsin="verified")
        return neutral_name

    monkeypatch.setattr(charged_router, "_reenter", reentry)
    pv.clear_provenance()
    pv.record_source("trivial_retained")
    try:
        name = charged_router.route_charged(Chem.MolFromSmiles("CC[NH3+]"))
        return name, pv.producer_record()
    finally:
        pv.clear_provenance()


def test_declined_charged_route_leaves_the_producer_record(monkeypatch):
    assert _route_with_reentry(monkeypatch, "") == ("", ("trivial_retained", None))


def test_charged_route_that_names_keeps_the_record_of_its_reentry(monkeypatch):
    """'ethanaminium' is built on the re-entered 'ethanamine', so the re-entry's
    producer record stays."""
    assert _route_with_reentry(monkeypatch, "ethanamine") == (
        "ethanaminium", ("general_engine", "verified"))
