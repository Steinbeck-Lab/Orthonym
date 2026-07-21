"""v27 P5: general fusion-nomenclature PARENT producer (P-25.3), a PIN-quality
upgrade over the von-Baeyer polyene the complete-tier engine ships for mancude
fused ring systems.

Design (0-wrong-by-construction): this module NEVER trusts a
computed fusion descriptor blindly. Every candidate fusion word is either
(a) an EXACT structural match against a vetted catalog (canonical-SMILES keyed,
Java-free) or (b) put through an AFFIRMATIVE OPSIN round-trip (the fusion word ->
structure must equal the input). The round-trip fails CLOSED when Java/OPSIN is
unavailable (unlike the pipeline's SELF-01, which fails OPEN), so a wrong fusion
descriptor -- e.g. the fusion machinery's known benzo[f]/benzo[h] orientation
slips -- is NEVER emitted in any environment. On any refusal the caller keeps the
(correct, non-PIN) von-Baeyer polyene form. This makes the phase strictly
non-regressing: it only ever replaces a VB-polyene with a STRUCTURALLY-VERIFIED
fusion PIN.

Scope: the BARE mancude fused parent (no principal-group suffix, no ring
substituents) -- the common polycyclic-aromatic-hydrocarbon / fused-heteroarene
case. Substituted / suffixed fusion parents (which need the fusion NUMBERING
threaded through the substituent+suffix tail) fall through to the VB polyene and
are the documented bounded remainder.
"""
from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

_CATALOG_CACHE: Optional[dict] = None


def _build_catalog() -> dict:
    """canonical-SMILES -> fusion PIN, from the two vetted ring catalogs. Both
    are hand-verified (stored names), so an exact SMILES match is 0-wrong without
    any OPSIN consultation."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    cat: dict = {}
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        for smi, entry in FUSED_HETEROCYCLE_DATA.items():
            name = entry.get('name')
            if not name:
                continue
            try:
                cs = Chem.CanonSmiles(smi)
            except Exception:
                continue
            cat.setdefault(cs, name)
    except Exception:
        pass
    try:
        from ..data.polycyclic_data import POLYCYCLIC_DATA
        for name, entry in POLYCYCLIC_DATA.items():
            cs_raw = entry.get('canonical_smiles')
            if not cs_raw:
                continue
            try:
                cs = Chem.CanonSmiles(cs_raw)
            except Exception:
                continue
            cat.setdefault(cs, name)
    except Exception:
        pass
    _CATALOG_CACHE = cat
    return cat


def _affirmative_roundtrip(name: str, target_smiles: str) -> bool:
    """True ONLY if OPSIN parses ``name`` to a structure equal to
    ``target_smiles`` (InChI match). Fails CLOSED (False) when OPSIN/Java is
    unavailable or on any mismatch -- the Java-free 0-wrong floor."""
    try:
        from ..validation.opsin_roundtrip import opsin_roundtrip_check
        res = opsin_roundtrip_check(target_smiles, name)
        return bool(res.get('passed'))
    except Exception:
        return False


_FUSION_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def _benzo_annulation_name(mol, target_smiles: str) -> Optional[str]:
    """Generate-and-verify a ``benzo[<letter>]<base>`` fusion PIN for a mancude
    system that decomposes as a named base + ONE peripheral ortho-fused benzo
    ring (e.g. ``benzo[h]quinoline`` = quinoline + benzo).

    This is a CONSTRUCTIVE search, not a descriptor derivation: the base is named
    by the vetted machinery, then every plausible attachment side-letter is
    enumerated and the one whose OPSIN structure equals the input is returned
    (0-wrong; the correct side-letter is picked by round-trip, sidestepping the
    fusion-orientation derivation entirely). Fail-closed (None) without OPSIN, on
    no clean single-benzo decomposition, or if no candidate round-trips.
    """
    from ..rules.vonbaeyer_universal import _extract_spiro_submol
    from ..rules.fused_rings import name_fused_heterocycle

    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if len(rings) < 2:
        return None
    for i, ring in enumerate(rings):
        if len(ring) != 6:
            continue
        if any(mol.GetAtomWithIdx(a).GetSymbol() != 'C'
               or not mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        others = set().union(*[rings[j] for j in range(len(rings)) if j != i])
        shared = ring & others
        unique = ring - others
        # a terminal ortho-fused benzo shares exactly one edge (2 atoms) and
        # contributes 4 unique atoms.
        if len(shared) != 2 or len(unique) != 4:
            continue
        base_atoms = set(range(mol.GetNumAtoms())) - unique
        base, _map = _extract_spiro_submol(mol, base_atoms)
        if base is None:
            continue
        res = name_fused_heterocycle(base)
        base_name = res[0] if isinstance(res, tuple) else res
        if not base_name or not isinstance(base_name, str):
            continue
        # a base that already carries its own fusion brackets would need nested
        # locants — out of this single-level scope.
        if '[' in base_name:
            continue
        for L in _FUSION_LETTERS[:12]:
            cand = f"benzo[{L}]{base_name}"
            if _affirmative_roundtrip(cand, target_smiles):
                return cand
    return None


def name_fusion_parent(mol, cage_atoms) -> Optional[str]:
    """Return a verified P-25.3 fusion-PIN word for the BARE mancude fused parent
    whose ring atoms are exactly ``cage_atoms``, or None (fail-closed) to keep the
    VB-polyene form.

    Only fires when the cage IS the whole molecule (a bare parent: no suffix, no
    substituents) -- the caller enforces that. Candidate sources: (1) exact
    catalog match (Java-free, 0-wrong); (2) the existing ``name_fused_heterocycle``
    machinery, AFFIRMATIVE-RT-verified. Never emits an unverified fusion word.
    """
    if mol is None or not cage_atoms:
        return None
    cage_set = set(cage_atoms)
    # bare-parent guard: the cage must be the whole molecule (only ring atoms,
    # nothing pendant) so the parent WORD alone is the complete name.
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if heavy != cage_set:
        return None
    try:
        target = Chem.MolToSmiles(mol)
    except Exception:
        return None

    # (1) exact catalog match — 0-wrong by construction, no OPSIN needed.
    try:
        cs = Chem.CanonSmiles(target)
        cat = _build_catalog()
        hit = cat.get(cs)
        if hit:
            return hit
    except Exception:
        pass

    # (2) computed candidate via the existing fusion machinery, but shipped ONLY
    # if it AFFIRMATIVELY round-trips (catches the orientation slips; fail-closed
    # without Java).
    try:
        from ..rules.fused_rings import name_fused_heterocycle
        res = name_fused_heterocycle(mol)
        cand = res[0] if isinstance(res, tuple) else res
        if cand and isinstance(cand, str) and _affirmative_roundtrip(cand, target):
            return cand
    except Exception:
        pass

    # (3) constructive benzo-annulation, generate-and-verify (fixes the fusion
    # machinery's orientation slips, e.g. benzo[h]quinoline the plain path
    # mis-orients to benzo[f]; the correct side-letter is chosen by round-trip).
    try:
        ann = _benzo_annulation_name(mol, target)
        if ann:
            return ann
    except Exception:
        pass

    return None
