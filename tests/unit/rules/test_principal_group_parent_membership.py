"""Principal-group occurrences must belong to the chosen parent (atom-partition invariant).

Root cause of a 25/67 share of the costly atom-drop class (v30, 2026-08-05,
): our perception
gathers ALL senior-group occurrences molecule-wide into `principal_group_atoms` with no
parent-membership filter, so a substituent's group (e.g. the OH of an isopropanol arm on a ring)
is wrongly claimed as a parent suffix — double-assigning the atom and producing a wrong,
atom-short parent (`cyclohexane-1,2-diol` for a mono-ol ring). prevent this
by deciding suffix-vs-prefix on atom→parent membership.


This test pins the invariant: after `_classify`, every principal-group occurrence's locant-bearing
atom lies inside the chosen parent (chain or ring). It FAILS today (documents the bug) and must
PASS after the membership filter lands. Perception-level, no OPSIN, so it stays in the fast suite.
"""

import sys
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthonym.cli import _emit_tier_flags  # noqa: E402
from orthonym.namer import Orthonym  # noqa: E402
from orthonym.rules.parent_selection import _pg_attachment_atoms  # noqa: E402


def _classify(smiles):
    flags = _emit_tier_flags("best-effort")
    nm = Orthonym(
        general_fallback=flags["general_fallback"],
        general_fallback_unverified=flags["general_fallback_unverified"],
        allow_aromatic_general=flags["allow_aromatic_general"],
    )
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol))
    nm._classify(feats)
    return feats


def _parent_atoms(feats):
    chain = set(feats.principal_chain or [])
    rings = set()
    for rs in feats.ring_systems or []:
        rings |= set(rs)
    if getattr(feats, "chain_is_parent", False) and chain:
        return chain
    if rings:
        return rings
    return chain


def _off_parent_occurrences(feats):
    parent = _parent_atoms(feats)
    if not parent or not feats.principal_group or not feats.principal_group_atoms:
        return []
    off = []
    for match in feats.principal_group_atoms:
        loc = _pg_attachment_atoms(feats.principal_group, tuple(match))
        if loc and not (set(loc) & parent):
            off.append((tuple(match), loc))
    return off


def test_isopropanol_arm_oh_is_not_claimed_as_a_ring_suffix():
    """The exact defect: C[C@H]1CC[C@H](C(C)(C)O)[C@@H](O)C1 — the arm's tertiary OH (atom 8,
    on arm carbon 5, not a ring atom) must NOT be a principal-group ring suffix."""
    feats = _classify("C[C@H]1CC[C@H](C(C)(C)O)[C@@H](O)C1")
    off = _off_parent_occurrences(feats)
    assert off == [], (
        f"principal-group occurrence(s) attached off-parent (the partition bug): {off}. "
        f"principal_group={feats.principal_group}, atoms={feats.principal_group_atoms}"
    )


def test_ring_ol_and_chain_ol_stay_on_parent():
    """Controls that must ALREADY pass: a genuine ring -ol and a genuine chain -ol are on-parent,
    so the filter must not strip them."""
    for smi in ("OC1CCCCC1", "CCCO", "CCC(=O)O"):
        feats = _classify(smi)
        assert _off_parent_occurrences(feats) == [], (
            f"{smi}: a legitimate parent suffix was flagged off-parent — the filter would over-strip"
        )
