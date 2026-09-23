"""Universal coverage-floor: a monovalent halogen is a substituent PREFIX, never
a chain-spine atom.

Root cause (measured 2026-08-30, spiro-followup Task A cascade-trace): the
universal best-effort core's chain-spine walk (``_tree_neighbors`` /
``_name_chain_spine``) treated a terminal halogen (F/Cl/Br/I) as a heavy
chain-continuation atom, so a fluoroalkyl branch was absorbed into the parent
skeleton -- a PHANTOM extra carbon plus a dropped halogen:

    ``c1ccccc1C(F)F`` (CHF2, difluoromethyl) -> ``1-fluoroethan-1-yl...`` WRONG
    ``c1ccccc1C(F)(F)F`` (CF3) -> ``1,1-difluoroethan-1-yl`` WRONG
    ``CCC(F)F`` / ``CC(F)(F)F`` -> None (whole molecule voids)

A halogen is monovalent: it is ALWAYS a terminal substituent and NEVER a
skeletal chain/replacement atom, so it must be excluded from the chain-spine
walk exactly as nitro-N, phosphinate-P and charged terminal atoms already are
(``_tree_neighbors``). It is then discovered as an off-spine branch and named
by the existing ``_leaf_shortcut`` (``fluoro``/``chloro``/``bromo``/``iodo``).

Reachable ONLY on the best-effort coverage floor (this whole module is
``_general_fallback``-gated in namer.py), so PIN/default output is byte-identical
-- proven by the fast gate. IUPAC 2013 / (halogen substituent
prefixes).

Targeted-file run only (avoid the OPSIN-pipe deadlock of a full pytest run):
    .venv/bin/python -m pytest tests/unit/assembly/test_universal_halogen_leaf.py -q
"""
from __future__ import annotations

from rdkit import Chem

from orthonym.assembly.universal_substituent import name_universal_substitutive
from orthonym.validation.reconstruct import verify_or_none


def _heavy(smiles):
    m = Chem.MolFromSmiles(smiles)
    return frozenset(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1)


def _name_and_verify(smiles):
    m = Chem.MolFromSmiles(smiles)
    r = name_universal_substitutive(m)
    assert r is not None, f"expected a name for {smiles!r}, got None"
    assert r.covers == _heavy(smiles), (
        f"coverage gap for {smiles!r}: covers={r.covers} heavy={_heavy(smiles)}")
    return r, verify_or_none(r.name, smiles)


# --- fluoroalkyl branches on a ring parent (the residual's dominant shape) ---

def test_difluoromethyl_on_ring_roundtrips():
    """CHF2 must stay a 1-carbon methane leaf, not a phantom ``ethan-1-yl``
    (2 carbons, 1 F). The bug rendered ``1-fluoroethan-1-yl``."""
    r, verified = _name_and_verify("c1ccccc1C(F)F")
    assert verified == r.name, f"{r.name!r} did not round-trip"
    assert "difluoromethan" in r.name, f"CHF2 mis-rendered: {r.name!r}"
    assert "fluoroethan" not in r.name, f"phantom carbon leaked: {r.name!r}"


def test_trifluoromethyl_on_ring_roundtrips():
    r, verified = _name_and_verify("c1ccccc1C(F)(F)F")
    assert verified == r.name, f"{r.name!r} did not round-trip"
    assert "trifluoromethan" in r.name, f"CF3 mis-rendered: {r.name!r}"
    assert "fluoroethan" not in r.name, f"phantom carbon leaked: {r.name!r}"


# --- fluoroalkyl on a chain parent (was a hard VOID before the fix) ---

def test_difluoropropane_names_and_roundtrips():
    r, verified = _name_and_verify("CCC(F)F")
    assert verified == r.name, f"{r.name!r} did not round-trip"


def test_trifluoroethane_names_and_roundtrips():
    r, verified = _name_and_verify("CC(F)(F)F")
    assert verified == r.name, f"{r.name!r} did not round-trip"


# --- the other halogens, chain terminus ---

def test_chloropropane_names_and_roundtrips():
    r, verified = _name_and_verify("CCCCl")
    assert verified == r.name, f"{r.name!r} did not round-trip"
    assert "chloro" in r.name, f"expected a chloro prefix: {r.name!r}"


def test_bromo_and_iodo_terminus_roundtrip():
    for smi, tok in (("CCCCBr", "bromo"), ("CCCI", "iodo")):
        r, verified = _name_and_verify(smi)
        assert verified == r.name, f"{r.name!r} did not round-trip"
        assert tok in r.name, f"expected {tok} in {r.name!r}"


# --- regression guards: plain alkyl and ring-halogen unchanged ---

def test_plain_methyl_unchanged():
    r, verified = _name_and_verify("Cc1ccccc1")
    assert verified == r.name
    assert "methyl" in r.name  # CH3 stays a 1-carbon leaf, spelled per (D3)


def test_ring_halogen_unchanged():
    r, verified = _name_and_verify("Clc1ccccc1")
    assert verified == r.name
    assert "chloro" in r.name
