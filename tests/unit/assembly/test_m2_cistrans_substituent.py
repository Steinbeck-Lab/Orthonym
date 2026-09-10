""" M2.3 — best-effort relative cis/trans on a ring-as-SUBSTITUENT.

When the general-engine best-effort floor names a molecule whose ring appears
AS A SUBSTITUENT with exactly two same-ring stereocentres (>=1 pseudoasymmetric,
lowercase ``_CIPCode``), the ring's absolute pseudoasymmetric block (e.g.
``(1s,3R)-``) is OPSIN-unparseable, so today the whole stereo candidate fails
round-trip and the molecule ABSTAINS. M2.3 instead OFFERS a relative
``cis-``/``trans-`` variant of that substituent (both senses) into the existing
full-InChIKey round-trip gate; whichever sense round-trips ships, at 0-wrong by
construction (a wrong sense simply fails the gate).

These tests pin the six Task-1 witnesses (``_m2_witnesses.WITNESSES``) plus the
BLOCKER-2 PIN gold row (a default-path regression guard that travels with this
task): the override MUST NOT change the PIN path.

Governing rule: (relative configuration ``cis``/``trans`` of two ring
substituents); the descriptor is the RELATION of two substituents on ONE ring,
so it is emitted only for a saturated 3-6-ring with exactly two same-ring
stereocentres (3+ -> r/c/t, out of scope).
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.validation.reconstruct import verify_or_none
from tests.unit.assembly._m2_witnesses import WITNESSES

# The best-effort tier ships with the OPSIN validity gate ON in production
# (``namer._DISABLE_VALIDITY_GATE`` defaults False); the suite's autouse fixture
# turns it OFF unless a test opts back in. Without the gate, an OPSIN-unparseable
# pseudoasymmetric ``(1r,4R)-`` general-engine candidate leaks through BEFORE the
# floor is reached, so this file MUST run gate-ON to observe the real behaviour
# M2.3 changes (skipped when the OPSIN jar is absent, per conftest).
pytestmark = pytest.mark.opsin_gate


def _name_best_effort(smiles: str):
    """Name *smiles* at the best-effort (T4) tier, in-process, EXACTLY as the
    CLI does (``--emit-tier best-effort``): the flag triple comes from
    ``cli._emit_tier_flags`` so this test cannot drift from the shipped tier.

    Returns the ``name_tiered`` row dict (``name`` is ``None`` on abstain).
    """
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import Orthonym

    flags = _emit_tier_flags("best-effort")
    namer = Orthonym(
        style="pin",
        general_fallback=flags["general_fallback"],
        general_fallback_unverified=flags["general_fallback_unverified"],
        allow_aromatic_general=flags["allow_aromatic_general"],
        full_coverage=flags.get("full_coverage", False),
    )
    return namer.name_tiered(smiles)


def _expected_relword(ref_name: str) -> str:
    """The cis/trans word the reference (RT-verified) name carries. The shipped
    sense must match it -- both name the same molecule, and the gate picks the
    RT-correct sense."""
    # ``trans`` before ``cis`` so a name containing neither substring is caught.
    if "trans" in ref_name:
        return "trans"
    assert "cis" in ref_name, ref_name
    return "cis"


@pytest.mark.parametrize("w", WITNESSES, ids=[w.wid for w in WITNESSES])
def test_witness_ships_relative_cistrans_and_round_trips(w):
    """Each witness: best-effort emits a name that (a) is not an abstain,
    (b) carries the expected ``cis``/``trans`` word, and (c) full-InChIKey
    round-trips through OPSIN to the input (0-wrong)."""
    row = _name_best_effort(w.smiles)
    name = row["name"]
    assert name is not None, (
        f"{w.wid}: abstained at best-effort (tier={row['tier']}, "
        f"limit={row.get('limit_code')}); M2.3 must reclaim it")

    relword = _expected_relword(w.ref_name)
    assert relword in name, (
        f"{w.wid}: shipped name lacks the '{relword}' descriptor: {name!r}")

    # 0-wrong: the shipped name must FULL-InChIKey round-trip to the input.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(w.smiles))
    assert verify_or_none(name, can) is not None, (
        f"{w.wid}: shipped name does NOT round-trip to the input: {name!r}")


def test_blocker2_gold_row_byte_identical():
    """BLOCKER-2 PIN gold row (def_id W4-S2): the default (PIN) path stays
    byte-identical -- the override never fires there (``relative_override is
    None``), so the pseudoasymmetric ``(1r,4r)-`` block is preserved.

    BB 48117 cites ``bis[(1r,4r)-4-methylcyclohexyl]phosphane``; this gold row
    exercises the identical ``(1r,4r)-4-methylcyclohexyl`` construction."""
    assert name_compound("C[C@@H]1CC[C@H](CC1)c1ccccc1") == \
        "[(1r,4r)-4-methylcyclohexyl]benzene"
