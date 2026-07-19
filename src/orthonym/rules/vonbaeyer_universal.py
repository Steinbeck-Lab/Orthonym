"""v25 G2: universal von-Baeyer cage analysis.

Names ANY bridged/fused polycyclic cage — including AROMATIC cages — by
kekulizing a canonical copy and expressing every former-aromatic bond as an
explicit ene locant (P-23 unsaturation). The emitted ``...-polyene`` cage
re-parses (OPSIN) to a kekule structure whose canonicalization re-aromatizes
to the SAME molecule, so structural fidelity is preserved; this is the
universal T3 ring fallback for the opt-in general engine ONLY. The default
PIN path's aromatic-cage refusal (polycyclic.py:2789) is deliberately
untouched.

Determinism: kekulization is atom-order dependent, so we ALWAYS canonical-
reparse first and work in canonical indices; results map back to original
indices via GetSubstructMatch.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

MAX_CAGE_ATOMS = 40
MAX_CAGE_RINGS = 8


@dataclass(frozen=True)
class UniversalCage:
    descriptor: str                 # e.g. "bicyclo[2.2.1]"
    total_atoms: int
    hetero_prefix: str              # "" | "7-oxa-" | "2,5-diaza-" ...
    unsaturation: dict              # {'double_bonds': [...], 'triple_bonds': [...]}
    cage_atoms: Tuple[int, ...]     # ORIGINAL mol indices
    atom_to_locant: Dict[int, int]  # ORIGINAL idx -> VB locant
    canon_match: Tuple[int, ...]    # canon idx -> orig idx
    is_mancude: bool = False        # aromatic/mancude cage (always False here:
                                    # mancude cages are refused, G5-A)


def analyze_cage_universal(mol, cage_atoms=None) -> Optional[UniversalCage]:
    """Deterministic universal cage analysis; None on any refusal."""
    from .polycyclic import (
        VonBaeyerAnalyzer, _get_largest_connected_ring_component,
        get_heteroatom_replacement_prefix, get_polycyclic_unsaturation,
    )

    if mol is None:
        return None

    canon_smiles = Chem.MolToSmiles(mol, canonical=True)
    canon = Chem.MolFromSmiles(canon_smiles)
    if canon is None:
        return None
    match = mol.GetSubstructMatch(canon)
    if len(match) != mol.GetNumAtoms():
        logger.info("vonbaeyer_universal: no whole-mol canon match; refuse")
        return None
    orig_to_canon = {orig: c for c, orig in enumerate(match)}

    kek = Chem.RWMol(canon)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception as e:  # kekulization failure -> fail closed
        logger.info("vonbaeyer_universal: kekulize failed: %s", e)
        return None

    ri = kek.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    if not ring_atoms:
        return None

    if cage_atoms is not None:
        cage_canon = {orig_to_canon[i] for i in cage_atoms}
        if not cage_canon <= ring_atoms:
            return None
    else:
        cage_canon = _get_largest_connected_ring_component(kek, ring_atoms)

    if len(cage_canon) > MAX_CAGE_ATOMS:
        return None
    n_rings = sum(1 for ring in ri.AtomRings() if set(ring) <= cage_canon)
    if n_rings < 2 or n_rings > MAX_CAGE_RINGS:
        return None

    analyzer = VonBaeyerAnalyzer()
    if len(analyzer._find_all_bridgeheads(kek, cage_canon)) < 2:
        return None  # spiro / degenerate: out of G2 scope
    try:
        desc = analyzer.analyze(kek, cage_canon)
    except Exception as e:
        logger.info("vonbaeyer_universal: analyze failed: %s", e)
        return None
    if not desc or not desc.numbering:
        return None
    # AUDIT: the numbering must cover the exact cage (no silent shrink).
    if set(desc.numbering.keys()) != set(cage_canon):
        logger.info("vonbaeyer_universal: numbering!=cage; refuse")
        return None

    hetero = get_heteroatom_replacement_prefix(kek, desc.numbering, cage_canon)
    unsat = get_polycyclic_unsaturation(kek, cage_canon, desc.numbering)

    # v25 G5-A: cite each ring double bond with the von-Baeyer COMPOUND locant
    # n(m) when its two atoms are NOT consecutively numbered (a fusion/bridge
    # ene, e.g. octalin 1(6)); plain n when m == n+1. The bare min(n,m) model
    # mislabels non-consecutive enes (and, adjacent to an oxo, fabricates the
    # 5-bond-carbon valence clash). ``double_bond_pairs`` (raw (low,high) VB
    # locants) is retained for the engine's valence guard. Scoped to THIS cage
    # payload — the default polycyclic path is untouched (it uses ints directly).
    _pairs = []
    for _b in kek.GetBonds():
        _i, _j = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()
        if (_i in desc.numbering and _j in desc.numbering
                and _b.GetBondTypeAsDouble() == 2.0):
            _lo, _hi = sorted((desc.numbering[_i], desc.numbering[_j]))
            _pairs.append((_lo, _hi))
    _pairs.sort()
    unsat['double_bond_pairs'] = _pairs
    unsat['double_bonds'] = [
        str(lo) if hi == lo + 1 else f"{lo}({hi})" for lo, hi in _pairs
    ]

    cage_orig = tuple(sorted(match[c] for c in cage_canon))
    atom_to_locant = {match[c]: loc for c, loc in desc.numbering.items()}

    # v25 G5-A: von-Baeyer is the PIN only for genuinely SATURATED / non-mancude
    # cages. A cage carrying an AROMATIC ring atom (original-mol perception) is
    # mancude -- its PIN is a fused/retained parent (P-25) + added/indicated H
    # (P-58.2.2), NOT a von-Baeyer polyene. Refuse (fail-closed); positive
    # mancude naming is a future G5-B build. Isolated ring double bonds
    # (norbornadiene) and saturated hetero cages (quinuclidine) are NOT aromatic
    # -> still named. This is the whole-class fix for the caffeine oxo/ene
    # valence-clash and the non-PIN-polyene defects (ledger SCOUT VERDICT).
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in cage_orig):
        logger.info("vonbaeyer_universal: mancude/aromatic cage -> refuse (G5-B)")
        return None

    return UniversalCage(
        descriptor=desc.descriptor_string,
        total_atoms=desc.total_atoms,
        hetero_prefix=hetero or "",
        unsaturation=unsat,
        cage_atoms=cage_orig,
        atom_to_locant=atom_to_locant,
        canon_match=tuple(match),
    )
