"""v30 RISK 5 Class 1 — a ring ester (lactone) must be named as the oxa-heterocycle bearing a
ring '-one', not the invalid '<ring>-carboxylate' (impossible O-locant + anion suffix on a
neutral) the ester producer emitted.

Root cause: `name_general_monocycle` trusted the perceived 'ester' principal group for a
ring-INTERNAL ester and routed it to the carboxylate suffix. The lactone's carbonyl C and ester
O are both in the ring, so the exocyclic =O is a ring ketone (-one) and the ring O is the
heteroatom of the parent (furan/pyran). Learned from /(both decompose the ring
ester at perception: ring O -> ring heteroatom, exocyclic =O -> oxo/-one) — refR5 consult.

Targets verified by OPSIN round-trip:
  O=C1OCC=C1                -> 2,5-dihydrofuran-2-one  (== furan-2(5H)-one)
  O=C1OC=CC=C1              -> pyran-2-one             (== 2H-pyran-2-one)
  CC(O)C1C=CC(=O)O1         -> 5-(1-hydroxyethyl)... furan-2-one form
"""
import pytest
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

# The lactone reclassification is exercised through the recovery lane / SELF-01
# suppression of the (wrong) saturated PIN name, which needs the OPSIN validity
# gate ENABLED (it is disabled suite-wide by default — see tests/conftest.py).
pytestmark = pytest.mark.opsin_gate


def _best_effort(smi):
    r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smi)
    return r["name"] if isinstance(r, dict) else r


def _rt_ok(smi, name):
    import sys; sys.path.insert(0, ".")
    from eval.harness import opsin_batch, _inchikey
    if not name:
        return False
    op = opsin_batch([name])[0]
    return bool(op) and _inchikey(smi) == _inchikey(op)


def test_butenolide_not_carboxylate():
    nm = _best_effort("O=C1OCC=C1")
    assert nm is not None
    assert "carboxylate" not in nm, nm
    assert "-one" in nm, nm


def test_butenolide_round_trips():
    smi = "O=C1OCC=C1"
    nm = _best_effort(smi)
    assert _rt_ok(smi, nm), nm


def test_pyranone_round_trips():
    smi = "O=C1OC=CC=C1"
    nm = _best_effort(smi)
    assert _rt_ok(smi, nm), nm


def test_substituted_butenolide_round_trips():
    smi = "CC(O)C1C=CC(=O)O1"
    nm = _best_effort(smi)
    assert _rt_ok(smi, nm), nm


def test_saturated_lactone_unchanged():
    # gamma-butyrolactone already names correctly via the PIN path -> must not regress.
    r = Orthonym(style="pin").name_tiered("O=C1CCCO1")
    assert (r["name"] if isinstance(r, dict) else r) == "oxolan-2-one"
