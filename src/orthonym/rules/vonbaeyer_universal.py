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
    is_mancude: bool = False        # aromatic/mancude cage; True only under the
                                    # opt-in complete tier (allow_mancude), where
                                    # the cage emits as a kekulized VB polyene


def audit_von_baeyer_descriptor(
    mol, cage_atoms, numbering: Dict[int, int], bridge_info_list,
) -> bool:
    """Java-free structural correctness floor for a von-Baeyer descriptor.

    Reconstruct the EXACT set of ring/bridge bonds the descriptor + its
    numbering ASSERTS -- every main-ring branch, the main bridge and each
    secondary bridge is a numbered path ``start_bh -> atoms... -> end_bh``,
    so the union of its consecutive-atom edges is precisely the skeleton the
    emitted name encodes -- and require SET-EQUALITY with the actual
    molecular-graph ring/bridge bond set over the cage. Any mismatch (a cage
    bond the name fails to assert, or a numbering that breaks path adjacency)
    -> ``False`` so the caller fails closed.

    This is intentionally independent of OPSIN: the downstream SELF-01
    (name->structure round-trip) fails OPEN when Java/OPSIN is unavailable, so
    a structurally-wrong descriptor would otherwise ship unchecked. Every
    bridge path is a real graph walk, so the asserted set is always a subset
    of the graph set; equality therefore reduces to COMPLETE coverage of the
    cage bonds (no ring bond silently dropped). Locant space is used on both
    sides (relabelled through ``numbering``), which additionally rejects a
    numbering that maps two graph-adjacent atoms to the same locant.
    """
    cage = set(cage_atoms)
    asserted = set()
    for bridge in bridge_info_list:
        path = [bridge.start_bh] + list(bridge.atoms) + [bridge.end_bh]
        for a, b in zip(path, path[1:]):
            if a not in numbering or b not in numbering:
                return False
            la, lb = numbering[a], numbering[b]
            if la == lb:
                return False
            # each asserted edge must be a REAL molecular bond (guards a
            # bridge path that jumps a non-bonded pair)
            if mol.GetBondBetweenAtoms(a, b) is None:
                return False
            asserted.add((min(la, lb), max(la, lb)))
    actual = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in cage and j in cage:
            if i not in numbering or j not in numbering:
                return False
            li, lj = numbering[i], numbering[j]
            actual.add((min(li, lj), max(li, lj)))
    return asserted == actual


def analyze_cage_universal(
    mol, cage_atoms=None, allow_mancude: bool = False,
) -> Optional[UniversalCage]:
    """Deterministic universal cage analysis; None on any refusal.

    v26 P2: when ``allow_mancude`` is True the aromatic/mancude-cage refusal
    below is LIFTED -- the cage is kekulized (already done above) and every
    former-aromatic bond is emitted as an explicit von-Baeyer polyene ene
    locant (P-23 unsaturation). When False (the default / PIN path) the
    refusal fires exactly as before, so that path is byte-identical.

    Every returned descriptor is put through ``audit_von_baeyer_descriptor``
    (a Java-free skeleton edge-audit): the descriptor+numbering must assert
    exactly the molecular-graph ring/bridge bond set or the cage is discarded
    (fail-closed). This floor holds for the saturated/valid tier too and is
    what makes the mancude polyene emission trustworthy independent of OPSIN.
    """
    from .polycyclic import (
        VonBaeyerAnalyzer, _get_largest_connected_ring_component,
    )
    from .ring_replacement import build_replacement_prefix
    from .ring_unsaturation import render_ring_unsaturation

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
    # AUDIT: descriptor+numbering must assert the exact molecular ring/bridge
    # bond set (Java-free skeleton floor; SELF-01 fails open without Java).
    if not audit_von_baeyer_descriptor(
            kek, cage_canon, desc.numbering, desc.bridge_info_list):
        logger.info("vonbaeyer_universal: descriptor edge-audit failed; refuse")
        return None

    # v29 Phase 2 T2a: skeletal-replacement TOTALITY. The prefix builder can only
    # spell the elements in its table; every other skeletal ring atom used to be
    # skipped while the stem kept counting it, so ``C1CC2CC[Hg]C2C1`` named as
    # ``bicyclo[3.3.0]octane`` — a hydrocarbon name for a mercury ring. SELF-01
    # fails OPEN with no OPSIN jar (a supported mode), so nothing downstream
    # caught it. Refuse whenever a skeletal atom is left unexpressed: never a
    # ring stem that counts an atom no morpheme in the name spells.
    replacement = build_replacement_prefix(kek, desc.numbering, cage_canon)
    if replacement.unexpressed:
        logger.info(
            "vonbaeyer_universal: %d skeletal atom(s) unexpressible by the "
            "replacement-prefix table; refuse", len(replacement.unexpressed))
        return None
    hetero = replacement.prefix

    # v25 G5-A / v29 Phase 2 T1: cite each ring double bond with the von-Baeyer
    # COMPOUND locant n(m) when its two atoms are NOT consecutively numbered (a
    # fusion/bridge ene, e.g. octalin 1(6)); plain n when m == n+1. The bare
    # min(n,m) model mislabels non-consecutive enes (and, adjacent to an oxo,
    # fabricates the 5-bond-carbon valence clash). ``double_bond_pairs`` (raw
    # (low,high) VB locants) is retained for the engine's valence guard.
    #
    # T1 moved this to ``rules/ring_unsaturation.py`` so it covers BOTH bond
    # orders. Before T1 the triple-bond locants came straight from
    # ``get_polycyclic_unsaturation``'s ``min(loc1, loc2)`` — a bare locant with
    # no composite form and NO GUARD, i.e. a wrong-bond citation waiting for a
    # non-consecutively-numbered yne. The primitive fails closed on that state
    # instead (it is unreachable for standard bonding numbers, so reaching it
    # means the NUMBERING is wrong). Scoped to THIS cage payload — the default
    # polycyclic path is untouched (it uses ints directly).
    ring_unsat = render_ring_unsaturation(kek, desc.numbering)
    if ring_unsat is None:
        logger.info("vonbaeyer_universal: ring unsaturation not citable; refuse")
        return None
    unsat = {
        'double_bonds': list(ring_unsat.double_locants),
        'double_bond_pairs': list(ring_unsat.double_pairs),
        'triple_bonds': list(ring_unsat.triple_locants),
    }

    cage_orig = tuple(sorted(match[c] for c in cage_canon))
    atom_to_locant = {match[c]: loc for c, loc in desc.numbering.items()}

    # v25 G5-A / v26 P2: a cage carrying an AROMATIC ring atom (original-mol
    # perception) is mancude. On the DEFAULT / PIN path (allow_mancude=False)
    # its PIN is a fused/retained parent (P-25) + added/indicated H (P-58.2.2),
    # NOT a von-Baeyer polyene, so we still refuse (fail-closed, byte-identical
    # to pre-P2). Under the opt-in complete tier (allow_mancude=True) the cage
    # was kekulized above and every former-aromatic bond is already captured in
    # ``unsat`` as an explicit ene locant (P-23) -- express it as the kekulized
    # von-Baeyer polyene instead of refusing. Isolated ring double bonds
    # (norbornadiene) and saturated hetero cages (quinuclidine) are NOT aromatic
    # -> named on both paths. The oxo/ene valence guard in the engine
    # (general_engine.name_general_ring, suffix_core=='one') still fires.
    is_mancude = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in cage_orig)
    if is_mancude and not allow_mancude:
        logger.info("vonbaeyer_universal: mancude/aromatic cage -> refuse (default path)")
        return None

    return UniversalCage(
        descriptor=desc.descriptor_string,
        total_atoms=desc.total_atoms,
        hetero_prefix=hetero or "",
        unsaturation=unsat,
        cage_atoms=cage_orig,
        atom_to_locant=atom_to_locant,
        canon_match=tuple(match),
        is_mancude=is_mancude,
    )


# =====================================================================
# v27 P3 (P-24.2): general SPIRO analysis — a sibling to the von-Baeyer
# cage engine above. ``analyze_cage_universal`` deliberately refuses spiro
# (``<2 bridgeheads`` at :156); this analyzer names monospiro / linear &
# branched polyspiro / heterocyclic spiro ring SYSTEMS, returning the SAME
# dataclass shape (``SpiroSystem`` mirrors ``UniversalCage``) so the general
# engine's suffix+substituent+stereo emission tail consumes it unchanged.
# It composes the audited building blocks in ``rules/spiro.py`` (do not
# re-derive) and runs every result through ``audit_spiro_descriptor`` — the
# mandatory Java-free structural floor (SELF-01 fails OPEN without Java).
# =====================================================================


@dataclass(frozen=True)
class SpiroSystem:
    """Same field shape as ``UniversalCage`` so ``_emit_ring_from_analysis``
    (the shared general-engine ring tail) is analysis-form-agnostic."""
    descriptor: str                 # e.g. "spiro[4.5]" / "dispiro[3.2.3.2]"
    total_atoms: int
    hetero_prefix: str              # "" | "6-oxa" | "2-oxa-6-thia" ...
    unsaturation: dict              # {'double_bonds':[...], 'triple_bonds':[...],
                                    #  'double_bond_pairs':[...]}
    cage_atoms: Tuple[int, ...]     # ORIGINAL (input-mol) indices
    atom_to_locant: Dict[int, int]  # ORIGINAL idx -> spiro locant
    canon_match: Tuple[int, ...]    # canon idx -> orig idx (identity here)
    is_mancude: bool = False


def audit_spiro_descriptor(
    mol, cage_atoms, numbering: Dict[int, int], spiro_atoms, descriptor: str,
) -> bool:
    """Java-free structural correctness floor for a spiro descriptor+numbering.

    Mirrors ``audit_von_baeyer_descriptor``'s set-equality contract, adapted to
    spiro topology. Fail-closed (``False``) on any of:
      (bijection)  numbering is not a 1-1 map of the cage atoms onto {1..N}
                   (rejects a numbering that maps two atoms to the same locant);
      (coverage)   the SSSR rings contained in the cage do not union to exactly
                   the cage atom set (a ring atom silently dropped);
      (pure-spiro) two cage rings share >1 atom (fused/bridged mis-routed here),
                   a shared atom is not a declared spiro atom, a spiro atom is
                   not in exactly 2 cage rings, a non-spiro atom is not in
                   exactly 1, or n_rings != n_spiro + 1;
      (simple-cycle) any cage atom has != 2 ring-neighbours within one of its
                   rings (a real bond the descriptor's cycle would assert is
                   missing, or a bridge edge is present);
      (arithmetic) the descriptor's segment integers + n_spiro != N (a wrong
                   ring-size descriptor for the audited skeleton).
    """
    import re
    cage = set(cage_atoms)
    n = len(cage)
    if n == 0:
        return False
    # (bijection)
    if set(numbering.keys()) != cage:
        return False
    if sorted(numbering.values()) != list(range(1, n + 1)):
        return False
    ri = mol.GetRingInfo()
    cage_rings = [set(r) for r in ri.AtomRings() if set(r) <= cage]
    if not cage_rings:
        return False
    # (coverage)
    union: set = set()
    for r in cage_rings:
        union |= r
    if union != cage:
        return False
    spiro_set = set(spiro_atoms)
    # (pure-spiro) pairwise ring intersections are single spiro atoms
    for i in range(len(cage_rings)):
        for j in range(i + 1, len(cage_rings)):
            inter = cage_rings[i] & cage_rings[j]
            if len(inter) > 1:
                return False
            if len(inter) == 1 and next(iter(inter)) not in spiro_set:
                return False
    ring_count: Dict[int, int] = {a: 0 for a in cage}
    for r in cage_rings:
        for a in r:
            ring_count[a] += 1
    for a in cage:
        if a in spiro_set:
            if ring_count[a] != 2:
                return False
        elif ring_count[a] != 1:
            return False
    if len(cage_rings) != len(spiro_set) + 1:
        return False
    # (simple-cycle) every atom has exactly 2 ring-neighbours inside each of
    # its rings — the cycle edges the descriptor asserts are all real bonds.
    for r in cage_rings:
        for a in r:
            nb = [x.GetIdx() for x in mol.GetAtomWithIdx(a).GetNeighbors()
                  if x.GetIdx() in r]
            if len(nb) != 2:
                return False
    # (arithmetic) descriptor segment sum + n_spiro == N
    try:
        body = descriptor[descriptor.index('[') + 1:descriptor.index(']')]
    except ValueError:
        return False
    segs = []
    for tok in body.split('.'):
        m = re.match(r'^(\d+)', tok)
        if not m:
            return False
        segs.append(int(m.group(1)))
    if sum(segs) + len(spiro_set) != n:
        return False
    return True


def _extract_spiro_submol(mol, cage_set):
    """Build a standalone submol of exactly ``cage_set`` (ring atoms), bonds
    among them preserved, broken parent/substituent bonds left as implicit H.
    Returns ``(submol, submol_to_mol)`` where ``submol_to_mol[k]`` is the
    original index of submol atom ``k`` — or ``(None, None)`` on failure.
    Atoms are added in sorted-index order for determinism."""
    order = sorted(cage_set)
    mol_to_sub = {m: k for k, m in enumerate(order)}
    rw = Chem.RWMol()
    for m in order:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(m).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in mol_to_sub and j in mol_to_sub:
            rw.AddBond(mol_to_sub[i], mol_to_sub[j], b.GetBondType())
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None, None
    return sub, order


def analyze_spiro_universal(
    mol, cage_atoms=None, allow_mancude: bool = False, free_valence_atoms=None,
) -> Optional[SpiroSystem]:
    """Deterministic universal spiro analysis; None on any refusal.

    ``cage_atoms`` selects the spiro ring system inside ``mol`` (defaults to all
    ring atoms). ``free_valence_atoms`` (original indices) biases the monospiro
    numbering to give the free valence the lowest locant (P-29.3, substituent
    use). ``allow_mancude`` lifts the aromatic-spiro refusal, emitting the
    kekulized ene-locant (P-23) form; when False an aromatic spiro fails closed
    (deferring to the retained-ring-name PIN path). Every result is put through
    ``audit_spiro_descriptor`` (fail-closed on any structural mismatch).
    """
    from ..perception.rings import get_spiro_atoms
    from .spiro import (
        generate_spiro_descriptor, get_spiro_numbering,
        _get_polyspiro_numbering, _build_hetero_prefix,
    )
    from .polycyclic import _get_alkane_name  # noqa: F401 (parity import)

    if mol is None:
        return None

    ring_atoms_mol = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms_mol:
        return None
    cage_set = set(cage_atoms) if cage_atoms is not None else set(ring_atoms_mol)
    if not cage_set <= ring_atoms_mol:
        return None
    if len(cage_set) > MAX_CAGE_ATOMS:
        return None

    # cage-only submol (reuses the spiro.py blocks unchanged, isolated from any
    # substituent rings on the parent).
    sub, sub_to_mol = _extract_spiro_submol(mol, cage_set)
    if sub is None:
        return None
    mol_to_sub = {m: k for k, m in enumerate(sub_to_mol)}

    ri = sub.GetRingInfo()
    n_rings = ri.NumRings()
    if n_rings < 2 or n_rings > MAX_CAGE_RINGS:
        return None
    spiro_sub = get_spiro_atoms(sub)
    if not spiro_sub:
        return None
    # pure spiro: n_spiro + 1 rings, and every ring-ring share is a spiro atom
    if n_rings != len(spiro_sub) + 1:
        return None

    descriptor = generate_spiro_descriptor(sub)
    if not descriptor:
        return None

    fv_sub = None
    if free_valence_atoms:
        fv_sub = {mol_to_sub[a] for a in free_valence_atoms if a in mol_to_sub}

    if len(spiro_sub) == 1:
        center = next(iter(spiro_sub))
        numbering = get_spiro_numbering(
            sub, center, suffix_ring_atoms=fv_sub)
    else:
        numbering = _get_polyspiro_numbering(sub, spiro_sub)
    # numbering must cover exactly the (ring-only) submol atoms
    if not numbering or set(numbering.keys()) != set(range(sub.GetNumAtoms())):
        return None

    if not audit_spiro_descriptor(
            sub, set(range(sub.GetNumAtoms())), numbering, spiro_sub, descriptor):
        logger.info("analyze_spiro_universal: descriptor audit failed; refuse")
        return None

    # --- unsaturation (ene/yne locants on the fixed spiro numbering) ---
    is_mancude = any(sub.GetAtomWithIdx(i).GetIsAromatic()
                     for i in range(sub.GetNumAtoms()))
    if is_mancude and not allow_mancude:
        return None
    kek = Chem.RWMol(sub)
    if is_mancude:
        try:
            Chem.Kekulize(kek, clearAromaticFlags=True)
        except Exception:
            return None
    # v29 Phase 2 T1: one shared producer for both bond orders (this block and the
    # cage sibling's computed the same thing twice, and only this copy guarded the
    # yne). The blanket non-consecutive-yne refusal that used to live here is now
    # the primitive's, so both paths refuse identically.
    from .ring_unsaturation import render_ring_unsaturation
    ring_unsat = render_ring_unsaturation(kek, numbering)
    if ring_unsat is None:
        logger.info("analyze_spiro_universal: ring unsaturation not citable; "
                    "refuse")
        return None
    unsat = {
        'double_bond_pairs': list(ring_unsat.double_pairs),
        'double_bonds': list(ring_unsat.double_locants),
        'triple_bonds': list(ring_unsat.triple_locants),
    }

    # --- heteroatom skeletal-replacement prefix (P-24.2.4) ---
    hetero_prefix = ""
    if any(sub.GetAtomWithIdx(i).GetSymbol() != 'C'
           for i in range(sub.GetNumAtoms())):
        # v29 Phase 2 T2a: the SAME skeletal-replacement totality rule the cage
        # sibling applies. ``_build_hetero_prefix`` was NOT fail-closed here: it
        # reaches ``polycyclic_bridged.get_heteroatom_prefix``, whose fallback is
        # ``symbol.lower() + 'a'``, so an off-table skeletal element got an
        # INVENTED morpheme (``3-znaspiro[5.5]undecane``, ``3-feaspiro[…]``,
        # ``3-alaspiro[…]``) rather than being refused. ``build_replacement_prefix``
        # is consulted for ``.unexpressed`` ONLY — the spelling stays with
        # ``_build_hetero_prefix`` so this path's strings are byte-identical for
        # the in-table elements (the two builders differ in their λ source and
        # their >20-multiplier handling).
        from .ring_replacement import build_replacement_prefix
        totality = build_replacement_prefix(
            sub, numbering, set(range(sub.GetNumAtoms())))
        if totality.unexpressed:
            logger.info(
                "analyze_spiro_universal: %d skeletal atom(s) unexpressible by "
                "the replacement-prefix table; refuse",
                len(totality.unexpressed))
            return None
        hp = _build_hetero_prefix(
            sub, set(spiro_sub), set(range(sub.GetNumAtoms())))
        if not hp:
            return None  # hetero present but prefix underivable -> fail closed
        hetero_prefix = hp

    total_atoms = sub.GetNumAtoms()
    cage_orig = tuple(sorted(cage_set))
    atom_to_locant = {sub_to_mol[k]: loc for k, loc in numbering.items()}

    return SpiroSystem(
        descriptor=descriptor,
        total_atoms=total_atoms,
        hetero_prefix=hetero_prefix,
        unsaturation=unsat,
        cage_atoms=cage_orig,
        atom_to_locant=atom_to_locant,
        canon_match=tuple(cage_orig),
        is_mancude=is_mancude,
    )
