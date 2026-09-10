# tests/unit/assembly/test_no_abstain_cd_checkpoint.py
""" Phase C + Phase D close-out — lightweight checkpoint regressions.

internal notes (read-only execution checkpoint,
jar-PRESENT, default config) found:

- Phase D ("decorated hetero-monocycle" witnesses) is ALREADY SUBSUMED: both
  a trace-table witnesses emit and round-trip at PIN via the existing small-
  ring/HW-nameable path, with no new code needed.
- Phase C ("name_general returns ENGINE_NONE on a path B4 doesn't cover") is
  REDUNDANT: both of its own named live rescue classes (acyclic hetero-chain,
  sulfate-decorated acyclic acid) now emit through the shipped B4 universal
  rung (`t4_coverage.py`'s final cascade rung, `name_universal_substitutive`)
  with zero `name_general` changes.

These tests guard that both findings hold at HEAD — no abstain on either
witness set — using the `Orthonym(general_fallback=True,
general_fallback_unverified=True)` construction the other best-effort tests
in this suite use (see `tests/unit/test_general_fallback_wiring.py`). No
producer/source code changed for this checkpoint; these are read-only
regression guards.
"""
import pytest

from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def _best_effort_name(smi: str):
    return Orthonym(general_fallback=True,
                      general_fallback_unverified=True).name(smi)


# ---------------------------------------------------------------------------
# Phase D — decorated hetero-monocycle witnesses (PHASE-CD-CHECKPOINT.md)
# ---------------------------------------------------------------------------

_PHASE_D_WITNESSES = [
    "ClC1CCSC1",              # -> 3-chlorothiolane (pin_path)
    "ClC1OC(Br)C(F)C1I",      # -> 5-bromo-2-chloro-4-fluoro-3-iodooxolane (T1)
]


@pytest.mark.parametrize("smi", _PHASE_D_WITNESSES)
def test_phase_d_hetero_monocycle_never_abstains(smi):
    """Both Phase D a trace-table witnesses must EMIT a non-abstain name."""
    out = _best_effort_name(smi)
    assert out is not None and not is_failure_name(out), (smi, out)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", _PHASE_D_WITNESSES)
def test_phase_d_hetero_monocycle_roundtrips(smi):
    """The emitted name OPSIN-round-trips to the input's InChIKey (real gate)."""
    from rdkit import Chem
    from orthonym.validation.opsin_roundtrip import opsin_parse

    out = _best_effort_name(smi)
    assert out is not None and not is_failure_name(out), (smi, out)
    parsed = opsin_parse(out)
    assert parsed, (smi, out, "emitted name must OPSIN-parse")
    want = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    got = Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert got == want, (smi, out, parsed)


# ---------------------------------------------------------------------------
# Phase C — redundancy witnesses (PHASE-CD-CHECKPOINT.md): both of Phase
# C's own named live classes now emit via the B4 universal rung
# (`name_universal_substitutive`, the final rung of `t4_coverage.py`'s
# cascade) rather than needing a separate `name_general` restructure.
# ---------------------------------------------------------------------------

def test_phase_c_acyclic_hetero_chain_never_abstains():
    """`C=CN=C` (Class A: acyclic hetero-chain) reaches the B4 universal rung
    and emits `2-azabuta-1,3-diene` instead of an ENGINE_NONE abstain."""
    smi = "C=CN=C"
    out = _best_effort_name(smi)
    assert out is not None and not is_failure_name(out), (smi, out)
