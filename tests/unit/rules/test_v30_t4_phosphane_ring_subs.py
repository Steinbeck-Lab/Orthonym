"""Ring-substituted phosphane parent (slice 1; PIN class program Task 2).

The phosphane parent namer handles simple alkyl phosphanes (triethylphosphane, PIN) but
ring-substituted phosphanes (tricyclododecylphosphane) fell through: `name_phosphine` is never
reached for a ring-bearing molecule (ring routing bypasses it). `_classify_phosphane_subs`
yields the ring substituents, so `name_mononuclear_hydride` emits the substitutive phosphane.

This branch ran at the best-effort tier only. "Substitution of phosphanes,
arsanes, and stibanes by organyl groups" (the Blue Book): "Alkyl, aryl, etc. groups...
are always denoted by prefixes" (:39153), 'cyclohexylphosphane (PIN)' (:39165): the organyl
phosphane is the PIN, so the branch now runs at every tier (the unit test below drives the
contextvar directly, independent of the OPSIN gate which behaves differently under pytest).
0-wrong: RT identity below + in production.
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
def test_ring_phosphane_named_at_every_tier(smi, expected):
    mol = Chem.MolFromSmiles(smi)
    for tier_flag in (False, True):   # the PIN path and the best-effort path
        tok = best_effort_ctx.set(tier_flag)
        try:
            out = name_mononuclear_hydride(mol)
        finally:
            best_effort_ctx.reset(tok)
        assert out == expected, (tier_flag, out)
    assert out.endswith("phosphane")
    assert _rt(smi, out), f"wrong molecule: {out!r}"


def test_simple_phosphane_unchanged_pin():
    # name_phosphine owns acyclic phosphanes; branch must not shadow it in PIN.
    from orthonym import Orthonym
    assert Orthonym().name("CCP(CC)CC") == "triethylphosphane"
    assert Orthonym().name("CCCP(CCC)CCC") == "tripropylphosphane"
