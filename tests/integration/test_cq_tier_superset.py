"""Task E guard (fable RISK 8) — best-effort is a SUPERSET of the complete tier.

The CQ3 spy found (at base 8ce7f804) that turning on ``general_fallback_unverified``
(best-effort) SUPPRESSED an already-working, complete-tier, OPSIN-RT-verified name
down to ``unknown organic compound`` on 2 witnesses (contextvar contamination of the
recursion). Task A's clean fall-through (commit 11599af9) closed that gap as a side
effect, so Task E landed 0 commits — but the property was left with NO regression
test. This file PINS the property (not the implementation): best-effort must never
emit a WORSE result than the complete tier where the complete tier produced an
RT-verified name. If a later change re-narrows the fall-through gate, this fails.

See ``.superpowers/sdd/CQ1-IMPL-PLAN/task-E-report.md`` and CQ1-CQ5-FABLE-REVIEW.md RISK 8.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = [pytest.mark.integration, pytest.mark.roundtrip,
              pytest.mark.opsin_gate]

# The 2 CQ3-SPY witnesses where best-effort used to suppress a complete-tier name.
WITNESSES = [
    "Cc1ccc(-c2cc(C(F)(F)F)nc(N/N=C/c3ccn(CC(F)(F)F)n3)n2)s1",   # hydrazone
    "CSCc1cccc(CNC(=NCCF)N2CC[C@@H](Cc3cnn(C)c3)C2)c1",           # guanidine
]


def _complete():
    return Orthonym(general_fallback=True, allow_aromatic_general=True)


def _besteffort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def _rt_ok(name, smiles):
    if is_failure_name(name):
        return False
    osmi = opsin_parse(name)
    return osmi is not None and _inchikey(osmi) == _inchikey(smiles)


@pytest.mark.parametrize("smiles", WITNESSES)
def test_besteffort_never_loses_complete_tier_name(smiles):
    """SUPERSET property: where the complete tier yields an RT-verified name,
    best-effort must also yield an RT-verified name (never abstain/worse)."""
    complete = _complete().name(smiles)
    best = _besteffort().name(smiles)
    if _rt_ok(complete, smiles):
        assert not is_failure_name(best), (
            f"best-effort SUPPRESSED a complete-tier name on {smiles!r}: "
            f"complete={complete!r} (RT-ok) but best-effort={best!r}")
        assert _rt_ok(best, smiles), (
            f"best-effort shipped {best!r} which does not RT-verify, while "
            f"complete={complete!r} does — best-effort must be a superset")
    else:
        # Complete tier didn't produce an RT-verified name here; the superset
        # property is vacuous, but best-effort must still be 0-wrong (either a
        # RT-verified name or an honest abstain, never a wrong molecule).
        assert is_failure_name(best) or _rt_ok(best, smiles), (
            f"best-effort shipped a non-RT-verifying name {best!r} on {smiles!r}")
