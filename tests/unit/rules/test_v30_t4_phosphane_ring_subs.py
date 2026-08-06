"""v30 T4 degrade-floor slice 1 — ring-substituted phosphane parent.

The phosphane parent namer handles simple alkyl phosphanes (triethylphosphane, PIN) but
ring-substituted phosphanes (tricyclododecylphosphane) fell through: `name_phosphine` is never
reached for a ring-bearing molecule (ring routing bypasses it) and `name_mononuclear_hydride`
deferred pure-organyl phosphanes to it (`'silyl' in subs` gate). `_classify_phosphane_subs`
already yields the ring substituents, so best-effort (T4) emits the substitutive phosphane.

T4-SCOPED: guarded by `best_effort_ctx`, so the PIN path is byte-identical (unit test below drives
the contextvar directly, independent of the OPSIN gate which behaves differently under pytest).
0-wrong: RT identity below + SELF-01 in production.
"""
from rdkit import Chem
from rdkit.Chem import inchi
import pytest

from orthonym import namer as N
from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
from orthonym.metrics.provenance import best_effort_ctx


def _ik(s):
    m = Chem.MolFromSmiles(s)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


def _rt(smi, name):
    o = N._validity_gate_name_to_smiles(name)
    return o is not None and _ik(smi) == _ik(o)


RING_PHOSPHANES = [
    ("C1CCCCCC(CCCCC1)P(C2CCCCCCCCCCC2)C3CCCCCCCCCCC3", "tricyclododecylphosphane"),  # real DEGRADE row
    ("P(C1CCCCC1)(C1CCCCC1)C1CCCCC1", "tricyclohexylphosphane"),
]


@pytest.mark.parametrize("smi,expected", RING_PHOSPHANES)
def test_ring_phosphane_t4_only(smi, expected):
    mol = Chem.MolFromSmiles(smi)
    # PIN path (contextvar False) -> must abstain (None): byte-identical, no regression.
    tok = best_effort_ctx.set(False)
    try:
        assert name_mononuclear_hydride(mol) is None, "PIN path must not emit (T4-scoping broken)"
    finally:
        best_effort_ctx.reset(tok)
    # best-effort (contextvar True) -> emits the substitutive phosphane, RT-correct.
    tok = best_effort_ctx.set(True)
    try:
        out = name_mononuclear_hydride(mol)
    finally:
        best_effort_ctx.reset(tok)
    assert out == expected, out
    assert out.endswith("phosphane")
    assert _rt(smi, out), f"wrong molecule: {out!r}"


def test_simple_phosphane_unchanged_pin():
    # name_phosphine owns acyclic phosphanes; T4 branch must not shadow it in PIN.
    from orthonym import Orthonym
    assert Orthonym().name("CCP(CC)CC") == "triethylphosphane"
    assert Orthonym().name("CCCP(CCC)CCC") == "tripropylphosphane"
