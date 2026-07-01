"""Unified substituent enumeration module with ReplaceCore-based extraction.

Provides a single enumeration path for all substituents on both ring and chain
parent structures. Replaces the previously fragmented three-path system that
caused silent drops, double-counting, and wrong locants.

Architecture:
  - Ring parents: ReplaceCore(mol, core_from_ring_atoms) -> fragment mols
  - Chain parents: Branch-point enumeration from features.substituents dict
  - All fragments: classify -> name via existing naming infrastructure

Public API:
  - discover_substituents(mol, parent_atoms, parent_type, ...) [Phase 84]
  - extract_ring_substituents(mol, ring_atoms, oriented_ring)
  - extract_chain_substituents(mol, principal_chain, substituents_dict)
  - classify_and_name_fragment(mol, frag_info, parent_atoms, features=None)
  - collect_substituent_atom_set(substituent_infos)

References:
    IUPAC 2013 P-31.1 (detachable prefixes)
    IUPAC 2013 P-44 (parent selection determines what's a substituent)
"""

import logging
from collections import deque, namedtuple

# Phase 160.2 Plan-04-02 WR-04 closure: removed unused
# ``from typing import List, Optional, Set, Dict`` — none of the four
# names are referenced anywhere in 1608 LOC (verified via AST scan;
# they appeared only in docstrings, not annotations). Re-add narrowly
# scoped imports here when real annotations are added.

from rdkit import Chem
from rdkit.Chem import RWMol

from .naming_utils import get_alkyl_name, SIMPLE_MULTIPLIERS
from .substituent_naming import name_substituent_fragment, _name_aryl_methyl_ether
from .substituent_prefix_forms import _check_substituent_prefix_form
from ..rules.seniority import get_prefix

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

SubstituentInfo = namedtuple(
    'SubstituentInfo',
    ['frag_mol', 'locant', 'attach_mol_idx', 'frag_atoms']
)
"""
Represents a single substituent on a parent structure.

Fields:
    frag_mol: RDKit Mol of the isolated fragment (with dummy atom at attachment),
              or None for chain-parent substituents where fragment is inline.
    locant: IUPAC locant (1-indexed integer) on the parent.
    attach_mol_idx: Original mol atom index of the attachment point on parent.
    frag_atoms: Set of original mol atom indices belonging to this substituent.
"""


# ============================================================================
# Universal Substituent Discovery (Phase 84)
# ============================================================================


def discover_substituents(
    mol,
    parent_atoms,
    parent_type="auto",
    oriented_ring=None,
    principal_chain=None,
    atom_to_locant=None,
):
    """Discover ALL substituents on a parent structure.

    Universal entry point that replaces six parallel substituent discovery
    systems. Every non-parent, non-hydrogen atom in mol is assigned to
    exactly one SubstituentInfo. No silent drops, no size limits.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.
        parent_type: ``"ring"``, ``"chain"``, or ``"auto"`` (auto-detects).
        oriented_ring: Ring atom indices in IUPAC order (required for ring parents).
        principal_chain: Chain atom indices in order (required for chain parents).
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.

    Returns:
        List[SubstituentInfo] with one entry per substituent fragment.
    """
    parent_set = set(parent_atoms)

    if parent_type == "auto":
        parent_type = _detect_parent_type(mol, parent_set)

    if parent_type == "ring":
        results = extract_ring_substituents(
            mol, tuple(parent_set), oriented_ring
        )
    else:
        results = _discover_chain_substituents(
            mol, parent_set, principal_chain, atom_to_locant
        )

    _verify_completeness(mol, parent_set, results)

    return results


def _detect_parent_type(mol, parent_atoms):
    """Auto-detect whether parent_atoms represent a ring or chain parent.

    Checks if any complete ring in the molecule is a subset of parent_atoms.
    If so, returns ``"ring"``; otherwise ``"chain"``.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.

    Returns:
        ``"ring"`` or ``"chain"``.
    """
    parent_set = set(parent_atoms)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if set(ring).issubset(parent_set):
            return "ring"
    return "chain"


def _discover_chain_substituents(mol, parent_atoms, principal_chain,
                                  atom_to_locant=None):
    """BFS-based substituent discovery for chain parents.

    For each atom on the principal chain, finds non-parent neighbors and
    BFS-collects complete substituent fragments. Walks through ALL atom
    types (no carbon-only restriction). No size limit.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices in the parent chain.
        principal_chain: List of atom indices in chain order.
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.

    Returns:
        List[SubstituentInfo] namedtuples.
    """
    results = []
    parent_set = set(parent_atoms)
    assigned = set()

    if principal_chain is None:
        principal_chain = sorted(parent_set)

    for chain_pos, chain_atom_idx in enumerate(principal_chain):
        chain_atom = mol.GetAtomWithIdx(chain_atom_idx)
        for nbr in chain_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in parent_set or nbr_idx in assigned:
                continue
            if nbr.GetAtomicNum() == 1:
                continue

            frag_atoms = _bfs_collect_fragment(
                mol, nbr_idx, parent_set, assigned
            )
            if not frag_atoms:
                continue
            assigned.update(frag_atoms)

            locant = chain_pos + 1
            if atom_to_locant and chain_atom_idx in atom_to_locant:
                locant = atom_to_locant[chain_atom_idx]

            results.append(SubstituentInfo(
                frag_mol=None,
                locant=locant,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frozenset(frag_atoms),
            ))

    return results


def _bfs_collect_fragment(mol, start_idx, parent_set, already_assigned):
    """BFS from start_idx, collecting all non-parent heavy atoms.

    CRITICAL: Does NOT stop at heteroatoms. Collects O, N, S, P and all
    atoms reachable through them. This ensures FG-containing substituents
    are discovered as compound fragments (USUB-04). No size limit.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index to start BFS from.
        parent_set: Set of parent atom indices (BFS boundary).
        already_assigned: Set of atom indices already claimed by another
            substituent.

    Returns:
        Set of atom indices in the collected fragment.
    """
    visited = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in visited or idx in parent_set or idx in already_assigned:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            continue
        visited.add(idx)
        for nbr in atom.GetNeighbors():
            queue.append(nbr.GetIdx())
    return visited


def _verify_completeness(mol, parent_atoms, substituents):
    """Assert that all non-parent heavy atoms are accounted for.

    Every non-parent, non-hydrogen atom must be assigned to exactly one
    SubstituentInfo. Reports double-assigned and missed atoms via assert.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of parent atom indices.
        substituents: List of SubstituentInfo namedtuples.

    Raises:
        AssertionError: If atoms are double-assigned or missed.
    """
    parent_set = set(parent_atoms)
    all_sub_atoms = set()
    for sub in substituents:
        overlap = all_sub_atoms & set(sub.frag_atoms)
        assert not overlap, (
            f"Double-assigned atoms: {overlap}"
        )
        all_sub_atoms.update(sub.frag_atoms)

    expected = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in parent_set:
            expected.add(atom.GetIdx())

    missed = expected - all_sub_atoms
    assert not missed, f"Unassigned atoms: {missed}"
    extra = all_sub_atoms - expected
    assert not extra, f"Extra atoms not in molecule: {extra}"


# ============================================================================
# Universal Substituent Naming (Phase 85)
# ============================================================================


def name_substituent(mol, frag_atoms, attach_idx):
    """Name any substituent fragment. Never returns None.

    Five-tier naming cascade:
      1. Retained substituent names (isopropyl, phenyl, etc.) -- IUPAC preferred
      2. Static fragment cache (FRAGMENT_NAME_CACHE) -- O(1) lookup
      3. Linear alkyl fast path (chain_prefixes table)
      4. Recursive compound naming (name_substituent_fragment)
      5. Descriptive fallback (guaranteed non-None)

    Args:
        mol: RDKit Mol of the full molecule.
        frag_atoms: Set/list of atom indices belonging to the substituent.
        attach_idx: Atom index WITHIN frag_atoms that bonds to the parent.

    Returns:
        str: IUPAC prefix name (always non-None, always non-empty).

    References:
        IUPAC 2013 P-31.1 (detachable prefixes)
        Phase 85 design: five-tier cascade with guaranteed fallback
    """
    import re as _re
    from .fragment_naming import FRAGMENT_NAME_CACHE
    from .substituent_naming import (
        parent_to_prefix,
        _check_retained_substituent,
        _is_linear_alkyl,
        _add_substituent_stereo,
    )

    frag_atoms_set = set(frag_atoms)

    # Edge case: empty fragment
    if not frag_atoms_set:
        return "substituent"

    # ---- D-08 (Phase 177 WSB-02): stereo-dropping tier double-apply guard ----
    # Tiers 0.5/1/1.5/1.6/2/3 build their prefix from a canonicalised fragment
    # (e.g. Tier-2's MolFragmentToSmiles strips @/@@ before the cache lookup),
    # so they SHORT-CIRCUIT Tier-4 — the only tier that natively reaches
    # _add_substituent_stereo. Without this guard a stereogenic substituent that
    # resolves via an early tier ships descriptor-less. _stereo_route() routes a
    # tier return through _add_substituent_stereo IFF the fragment carries CIP
    # stereo AND the candidate prefix does not ALREADY carry a "(...)" stereo
    # block (the re.match double-apply guard — RESEARCH Q#2: a fragment-scope
    # check, NOT a needs_stereo_injection call).
    _STEREO_BLOCK_RE = _re.compile(r'^\(\d*[a-z]?[RSrsEZez](,\d*[a-z]?[RSrsEZez])*\)-')

    def _frag_has_cip_stereo() -> bool:
        for _i in frag_atoms_set:
            _a = mol.GetAtomWithIdx(_i)
            if _a.HasProp('_CIPCode'):
                return True
        for _b in mol.GetBonds():
            if (_b.GetBeginAtomIdx() in frag_atoms_set
                    and _b.GetEndAtomIdx() in frag_atoms_set
                    and _b.HasProp('_CIPCode')):
                return True
        return False

    def _stereo_route(prefix: str) -> str:
        # Route a stereo-dropping tier's return through the substituent stereo
        # emitter, unless the prefix already carries a leading "(...)" descriptor
        # (double-apply guard) or the fragment has no CIP stereo. `attach_idx` is
        # threaded so the emitter can derive the located descriptor (PIN name +
        # attachment locant) for an acyclic-alkyl substituent from STRUCTURE.
        if not prefix or _STEREO_BLOCK_RE.match(prefix):
            return prefix
        if not _frag_has_cip_stereo():
            return prefix
        return _add_substituent_stereo(
            mol, list(frag_atoms_set), prefix, attach_idx=attach_idx
        )

    # ---- Tier 0.5 (Phase 160.1 D-04): IUPAC P-65 / P-66 prefix-form check ----
    # PURE read-only check. Returns the IUPAC-canonical prefix form for any
    # fragment that ENTIRELY contains one of the 14 non-principal functional
    # groups (ester, ether, amide, sulfoxide, sulfone, thioether, nitrile,
    # carbamate, urea, isocyanate, isothiocyanate). Short-circuits Tier-1..5
    # for FG-bearing fragments, eliminating the 'methyl formatyl' /
    # 'hydroxymethyl' bug per RESEARCH §3 root-cause fix.
    try:
        prefix_form = _check_substituent_prefix_form(
            mol, frag_atoms_set, attach_idx
        )
        if prefix_form is not None:
            return _stereo_route(prefix_form)
    except Exception:
        # Defensive: any unexpected SMARTS / RDKit error falls through to Tier-1
        pass

    # ---- Tier 1: Retained substituent names ----
    # Checked first per IUPAC: retained names (phenyl, isopropyl, etc.)
    # are the preferred forms and must take priority over cache-derived
    # parent-to-prefix conversions (e.g., "phenyl" not "benzenyl").
    try:
        retained = _check_retained_substituent(
            mol, list(frag_atoms_set), attach_idx
        )
        if retained:
            return _stereo_route(retained)
    except Exception:
        pass

    # ---- Tier 1.5 (Phase 173.5 L1): monocyclic heteroaryl PIN locant ----
    # A heteroaryl ring substituent (pyridine, imidazole, furan, ...) takes
    # free-valence numbering — pyridin-3-yl, 1H-imidazol-5-yl — instead of the
    # locant-less parent_to_prefix form (pyridinyl / imidazolyl) that the cache
    # (Tier 2) or recursive namer (Tier 4) would otherwise emit. Guarded:
    # returns None (so we fall through unchanged) unless the locant is provably
    # PIN-correct (IUPAC P-31.1.4.3.4).
    if attach_idx is not None:
        try:
            from ..rules.ring_substituents import pin_heteroaryl_substituent_name
            for ring in mol.GetRingInfo().AtomRings():
                if attach_idx in ring and set(ring) <= frag_atoms_set:
                    pin = pin_heteroaryl_substituent_name(mol, ring, attach_idx)
                    if pin is not None:
                        return _stereo_route(pin)
                    break
        except Exception:
            pass

    # ---- Tier 1.6 (WS-A.2): decorated monocyclic ring substituent ----
    # A ring fragment carrying its own substituents must keep them with
    # attachment-correct numbering ('2-oxocyclohexyl'), instead of the
    # cache/recursive parent_to_prefix form that keeps the PARENT numbering
    # ('1-oxocyclohexyl' — structurally impossible) or drops the group.
    # Guarded: returns None (fall through unchanged) unless the ring is a
    # supported simple monocycle AND the decorated name covers EXACTLY the
    # fragment atoms (P-14.4 numbering; see rules/ring_substituents.py).
    if attach_idx is not None:
        try:
            from ..rules.ring_substituents import decorated_ring_substituent_name
            for ring in mol.GetRingInfo().AtomRings():
                if attach_idx in ring and set(ring) <= frag_atoms_set:
                    dec = decorated_ring_substituent_name(
                        mol, ring, attach_idx, expected_atoms=frag_atoms_set)
                    if dec is not None:
                        return _stereo_route(dec)
                    break
        except Exception:
            pass

    # ---- Tier 1.7 (DD2 Fix B, Phase D): peroxy / disulfanyl substituent ----
    # A -O-O-R (peroxy) / -S-S-R (disulfanyl) substituent: the attach atom is a
    # divalent chalcogen bonded to a second like chalcogen inside the fragment.
    # Named (R)peroxy / (R)disulfanyl per P-63.3.1(1). Placed before the cache /
    # recursive tiers, which otherwise mangle the -O-O-/-S-S- into a bogus
    # 'peroxyl'/'dithioperoxyl' fragment. Reachable from EVERY caller (the chain
    # GENERAL path, the benzene/ring-substituent path), so the substitutive
    # peroxide/disulfide is named identically wherever it appears.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        _attach_atom = mol.GetAtomWithIdx(attach_idx)
        _sym = _attach_atom.GetSymbol()
        if _sym in ('O', 'S') and any(
            n.GetSymbol() == _sym and n.GetIdx() in frag_atoms_set
            for n in _attach_atom.GetNeighbors()
        ):
            _parent_atoms = set(range(mol.GetNumAtoms())) - frag_atoms_set
            _frag_list = list(frag_atoms_set)
            _chal = (
                _name_peroxy_branch(mol, _frag_list, attach_idx, _parent_atoms)
                if _sym == 'O'
                else _name_disulfanyl_branch(mol, _frag_list, attach_idx, _parent_atoms)
            )
            if _chal:
                return _stereo_route(_chal)

    # ---- Tier 1.8 (DD5 RC-6 / SEN-04): located acyclic alkyl ----
    # A BRANCHED or INTERNALLY-attached acyclic all-carbon saturated alkyl
    # substituent is named by its OWN principal chain numbered from the free
    # valence (hexan-2-yl, pentan-3-yl, 3-methylbutyl) per P-29.2 / P-46. This
    # MUST precede the fragment cache (Tier 2) and the linear fast path (Tier 3),
    # both of which name the fragment as a FREE molecule and lose the attachment
    # (-> 'hexyl', 'pentyl', '2-methylbutyl' — a wrong locant or constitution).
    # Scoped to the cases where the located form DIFFERS from the plain alkyl
    # (internal attachment OR a branch): a TERMINAL unbranched chain keeps the
    # fast path byte-identical. Returns None for rings / heteroatoms / unsaturated
    # -> falls through unchanged.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import (
                _located_acyclic_alkyl_name,
                _attach_is_chain_terminus,
                _is_linear_alkyl,
            )
            _frag_list = list(frag_atoms_set)
            _terminal_linear = (
                _is_linear_alkyl(mol, _frag_list)
                and _attach_is_chain_terminus(mol, _frag_list, attach_idx)
            )
            if not _terminal_linear:
                _located = _located_acyclic_alkyl_name(mol, _frag_list, attach_idx)
                if _located is not None:
                    return _stereo_route(_located[0])
        except Exception:
            pass

    # ---- Tier 1.9 (v22 C-T2 / V-3): ether-substituted carbon chain ----
    # A saturated all-carbon chain bearing ether -O-R substituent(s), numbered
    # from the free valence, named (R-oxy)alkyl per P-63.2.2.2 (phenoxymethyl,
    # 2-phenoxyethyl, methoxymethyl). MUST precede the cache (Tier 2): the cache
    # maps the capped fragment SMILES to a whole-molecule retained name
    # (COc1ccccc1 -> 'anisole') which parent_to_prefix then mangles to 'anisolyl'
    # — a DIFFERENT constitution (free valence on the ring). Reachable from every
    # caller so the ether substituent is named identically wherever it appears.
    # Returns None (fall through) for anything not this narrow class (fail-closed).
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_ether_substituted_chain
            _ether = _name_ether_substituted_chain(
                mol, list(frag_atoms_set), attach_idx, set()
            )
            if _ether:
                return _stereo_route(_ether)
        except Exception:
            pass

    # ---- Tier 1.92 (v23 Phase 8, P-68.2.2): Group-14 silyl/germyl substituent --
    # A monovalent Si/Ge substituent is named (prefixes)silyl / (prefixes)germyl,
    # with the substituents on the Si/Ge centre cited as prefixes. MUST precede the
    # Tier-2 cache and Tier-4 recursive namer, which drop a bare -SiH3 ('substituent')
    # or keep an -OH as a parent-hydride -ol suffix (-Si(OH)3 -> 'silanetriolyl').
    # Gated on the attach atom being Si/Ge (rare -> contained blast radius);
    # fail-closed (None -> fall through unchanged) for every other case.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        if mol.GetAtomWithIdx(attach_idx).GetSymbol() in ('Si', 'Ge'):
            try:
                from .substituent_naming import _name_group14_substituent
                _g14 = _name_group14_substituent(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _g14:
                    return _stereo_route(_g14)
            except Exception:
                pass

    # ---- Tier 1.95 (Phase 4 SUBST-01): ring-system substituent chokepoint ----
    # A ring-bearing fragment is named by the trustworthy ring engine
    # (get_ring_substituent_name + _compound_ring_on_chain_substituent) BEFORE the
    # Tier-2 cache, which would otherwise (a) drop ene/yne locants
    # ('cyclohexenyl' for cyclohex-1-en-1-yl via parent_to_prefix), or (b) name a
    # ring-on-chain as a different molecule ('methylcyclohexyl' for
    # cyclohexylmethyl). allow_enumerator_fallback=False makes the chokepoint
    # return None on a decline instead of re-entering this cascade (recursion
    # guard); we then fall through to the existing tiers unchanged.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            _ri = mol.GetRingInfo()
            if any(_ri.NumAtomRings(a) > 0 for a in frag_atoms_set):
                from ..rules.ring_substituents import name_ring_system_substituent
                _ring_nm = name_ring_system_substituent(
                    mol, sorted(frag_atoms_set), attach_idx,
                    allow_enumerator_fallback=False,
                )
                if _ring_nm:
                    return _stereo_route(_ring_nm)
                # P-32.2.2 CONSTITUTION GUARD (Tier 1.95 / recursive-path gate):
                # Parallel guard to substituent_naming.py Step 1c.  When the ring-
                # system namer declines a fused multi-ring fragment in the
                # fused-heterocycle catalog under a tautomer_locant name (e.g.
                # '1H-indene') AND the fragment has non-aromatic partial unsaturation
                # (endocyclic C=C that is not aromatic), the Tier-2 cache and Tier-4
                # recursive namer would independently produce the same wrong-
                # constitution name.  Return None (sentinel) so the Tier-4 recursive
                # path is bypassed and the caller emits 'unknown' instead.
                _rings_in_frag = [r for r in _ri.AtomRings() if set(r) <= frag_atoms_set]
                if len(_rings_in_frag) > 1:
                    from rdkit import Chem as _Chem_enum
                    _frag_smi = _Chem_enum.MolFragmentToSmiles(
                        mol, sorted(frag_atoms_set), canonical=True
                    )
                    if _frag_smi:
                        try:
                            from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
                            _fhe = FUSED_HETEROCYCLE_DATA.get(_frag_smi)
                            if _fhe and _fhe.get('tautomer_locant') is not None:
                                _fmol = _Chem_enum.MolFromSmiles(_frag_smi)
                                if _fmol is not None and any(
                                    b.GetBondTypeAsDouble() == 2.0
                                    and not b.GetIsAromatic()
                                    for b in _fmol.GetBonds()
                                ):
                                    return None
                        except ImportError:
                            pass
        except Exception:
            pass

    # ---- Tier 1.96 (v23 SL): acyclic substituent with detachable prefixes ----
    # A saturated acyclic carbon chain bearing >=1 simple detachable prefixes
    # (carboxy/amino/hydroxy/oxo/halogen) is named from STRUCTURE, numbered from
    # the free valence. MUST precede the Tier-2 cache, which maps the H-capped
    # fragment to a whole-molecule retained name (serine-O -CH2CH(NH2)COOH caps
    # to 'alanine' -> 'alaninyl'; -CH2COOH caps to 'acetic acid' -> 'acetyl', a
    # different molecule), and the Tier-4 recursive path, which lets
    # parent_to_prefix DROP secondary prefixes ('(R)-2-carboxyethyl', amino lost)
    # or inherit the parent's lowest-locant numbering (-CH2CH2CH2OH ->
    # '1-hydroxypropyl', wrong end). Fail-closed (None -> fall through) for
    # rings / branched / unsaturated / amides / esters / ethers / bare-acyl.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_polyfunctional_acyclic_substituent
            _poly = _name_polyfunctional_acyclic_substituent(
                mol, list(frag_atoms_set), attach_idx, set()
            )
            if _poly:
                return _stereo_route(_poly)
        except Exception:
            pass

    # ---- Tier 2: Static fragment cache (O(1)) ----
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, list(frag_atoms_set))
        if frag_smiles:
            canonical = Chem.CanonSmiles(frag_smiles)
            if canonical:
                cached = FRAGMENT_NAME_CACHE.get(canonical)
                if cached:
                    # Convert parent name to prefix form
                    carbon_count = sum(
                        1 for i in frag_atoms_set
                        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                    )
                    prefix = parent_to_prefix(cached, chain_length=carbon_count)
                    if prefix:
                        return _stereo_route(prefix)
    except Exception:
        pass  # Cache miss is fine, continue to next tier

    # ---- Tier 3: Linear alkyl fast path (attached at a chain TERMINUS) ----
    # DD5 RC-6 / SEN-04: a linear chain attached at an INTERNAL carbon
    # (pentan-3-yl, hexan-2-yl) is NOT a terminal alkyl — defer to Tier 4's
    # located deriver so the free valence becomes the numbering basis.
    try:
        from .substituent_naming import _attach_is_chain_terminus
        if _is_linear_alkyl(mol, list(frag_atoms_set)) and _attach_is_chain_terminus(
            mol, list(frag_atoms_set), attach_idx
        ):
            carbon_count = sum(
                1 for i in frag_atoms_set
                if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            )
            if carbon_count > 0:
                return _stereo_route(get_alkyl_name(carbon_count))
    except Exception:
        pass

    # ---- Tier 4: Recursive compound naming ----
    # Skip Tier 4 for single-atom non-carbon fragments (halogens, -OH, -NH2,
    # =O, etc.) where the recursive namer produces garbled results like
    # "ammoniayl" or "unknown organic compoundyl". The descriptive fallback
    # (Tier 5) handles these correctly.
    _skip_tier4 = False
    if len(frag_atoms_set) == 1:
        _single_atom = mol.GetAtomWithIdx(next(iter(frag_atoms_set)))
        if _single_atom.GetAtomicNum() != 6:
            _skip_tier4 = True

    if not _skip_tier4:
        try:
            result = name_substituent_fragment(
                mol, list(frag_atoms_set), attach_idx, []
            )
            if result and "unknown" not in result.lower():
                return result
        except Exception:
            pass

    # ---- Tier 5: Descriptive fallback (guaranteed non-None) ----
    return _descriptive_fallback(mol, frag_atoms_set, attach_idx)


def _descriptive_fallback(mol, frag_atoms, attach_idx):
    """Produce a compositional description for unnameable fragments.

    Analyzes fragment atoms directly to build a best-effort prefix name.
    For simple fragments (1-3 atoms), produces specific names like
    "hydroxy", "amino", "methyl". For complex unnameable fragments,
    returns "substituent" as absolute last resort.

    Args:
        mol: RDKit Mol object.
        frag_atoms: Set of atom indices in the fragment.
        attach_idx: Attachment atom index.

    Returns:
        str: Always non-None, always non-empty.
    """
    if not frag_atoms:
        return "substituent"

    # Analyze fragment composition
    carbons = 0
    heteroatoms = {}
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        anum = atom.GetAtomicNum()
        if anum == 1:
            continue
        if anum == 6:
            carbons += 1
        else:
            sym = atom.GetSymbol()
            heteroatoms[sym] = heteroatoms.get(sym, 0) + 1

    # Single-atom fragments
    if len(frag_atoms) == 1:
        idx = next(iter(frag_atoms))
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        total_hs = atom.GetTotalNumHs()

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        # Oxygen
        if sym == 'O':
            return 'hydroxy' if total_hs >= 1 else 'oxo'
        # Nitrogen
        if sym == 'N':
            if total_hs >= 2:
                return 'amino'
            elif total_hs == 1:
                return 'imino'
            else:
                return 'azanyl'
        # Sulfur
        if sym == 'S':
            return 'sulfanyl' if total_hs >= 1 else 'sulfanylidene'
        # Single carbon
        if sym == 'C':
            return 'methyl'

    # Carbon-only fragments: use alkyl names
    if carbons > 0 and not heteroatoms:
        try:
            return get_alkyl_name(carbons)
        except (ValueError, KeyError):
            pass

    # Multi-atom heteroatom-only fragments
    if carbons == 0 and heteroatoms:
        symbols = sorted(heteroatoms.keys())
        # -NO2 (nitro)
        if symbols == ['N', 'O'] and heteroatoms.get('N', 0) == 1 and heteroatoms.get('O', 0) == 2:
            return 'nitro'
        # -N3 (azido)
        if symbols == ['N'] and heteroatoms.get('N', 0) == 3:
            return 'azido'
        # Single heteroatom type
        if len(symbols) == 1:
            sym = symbols[0]
            if sym == 'O':
                return 'hydroxy'
            if sym == 'N':
                return 'amino'
            if sym == 'S':
                return 'sulfanyl'
            if sym in _HALOGEN_MAP:
                return _HALOGEN_MAP[sym]

    # ---- Compound substituents: carbon + heteroatom combinations ----
    # IUPAC P-31.1.3: compound prefix names built from
    # heteroatom-prefix + alkyl-stem (e.g., hydroxymethyl, aminoethyl).
    # Restricted to avoid positional ambiguity.

    if carbons > 0 and heteroatoms:
        # Cyano: exactly 1C + 1N with triple bond (nitrile substituent)
        if carbons == 1 and heteroatoms == {'N': 1}:
            for idx in frag_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in frag_atoms and nbr.GetSymbol() == 'N':
                            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                            if bond and bond.GetBondTypeAsDouble() == 3.0:
                                return 'cyano'

        # 2-HA fragments (1 carbon + 1 heteroatom): unambiguous position
        total_ha = carbons + sum(heteroatoms.values())
        if total_ha <= 2 and carbons == 1:
            hetero_prefix = None
            for idx in frag_atoms:
                atom = mol.GetAtomWithIdx(idx)
                sym = atom.GetSymbol()
                if sym == 'O':
                    hetero_prefix = 'hydroxy' if atom.GetTotalNumHs() >= 1 else 'oxo'
                    break
                elif sym == 'N':
                    hetero_prefix = 'amino' if atom.GetTotalNumHs() >= 2 else 'imino'
                    break
                elif sym == 'S':
                    hetero_prefix = 'sulfanyl' if atom.GetTotalNumHs() >= 1 else 'thio'
                    break
                elif sym in _HALOGEN_MAP:
                    hetero_prefix = _HALOGEN_MAP[sym]
                    break

            if hetero_prefix:
                try:
                    alkyl_stem = get_alkyl_name(carbons)
                    return f"{hetero_prefix}{alkyl_stem}"
                except (ValueError, KeyError):
                    pass

    # Absolute last resort
    return "substituent"


# ============================================================================
# Halogen Name Map (for fg_only classification)
# ============================================================================

_HALOGEN_MAP = {
    'F': 'fluoro',
    'Cl': 'chloro',
    'Br': 'bromo',
    'I': 'iodo',
}


# ============================================================================
# Ring Substituent Extraction (ReplaceCore-based)
# ============================================================================


def extract_ring_substituents(mol, ring_atoms, oriented_ring):
    """Extract all substituent fragments from a ring parent using ReplaceCore.

    Builds a core mol from ring_atoms, calls ReplaceCore to extract all
    non-ring fragments as separate mol objects with isotope-labeled dummy
    atoms indicating attachment points.

    Args:
        mol: RDKit Mol object.
        ring_atoms: Tuple or list of ring atom indices (from principal_ring).
        oriented_ring: List of ring atom indices in IUPAC numbering order.

    Returns:
        List of SubstituentInfo namedtuples, one per substituent fragment.
        Empty list if ring has no substituents.
    """
    ring_set = set(ring_atoms)

    # Build core mol from ring atoms
    core = RWMol()
    idx_map = {}  # original mol idx -> core mol idx
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = core.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        # Preserve aromaticity for correct matching
        core.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
        idx_map[atom_idx] = new_idx

    # Copy bonds between ring atoms
    added_bonds = set()
    for atom_idx in ring_atoms:
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in ring_set and end in ring_set:
                bond_key = (min(begin, end), max(begin, end))
                if bond_key not in added_bonds:
                    core.AddBond(
                        idx_map[begin], idx_map[end], bond.GetBondType()
                    )
                    added_bonds.add(bond_key)

    core_mol = core.GetMol()

    # Match tuple maps core atom index -> original mol atom index
    match = tuple(ring_atoms)

    # ReplaceCore: removes core, returns fragments with isotope-labeled dummies
    frags = Chem.ReplaceCore(mol, core_mol, match, labelByIndex=True)
    if frags is None:
        return []

    # Split into individual fragment mols
    subfgs = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
    if not subfgs:
        return []

    # Pre-compute per-branch atom sets for geminal substituent disambiguation
    # Key: ring_atom_idx -> list of frozensets (one per separate branch from that atom)
    branch_map = _compute_branch_map(mol, ring_set)

    # Track which branches have been claimed (for geminal disambiguation)
    claimed_branches = set()  # set of (attach_idx, branch_id) tuples

    results = []
    for frag in subfgs:
        # Find the dummy atom(s) to determine attachment point
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom
                # CRITICAL: isotope 0 is valid (maps to match position 0)
                core_pos = atom.GetIsotope()
                if core_pos < len(match):
                    mol_atom_idx = match[core_pos]

                    # Map to IUPAC locant via oriented_ring
                    locant = _get_locant_from_oriented_ring(
                        mol_atom_idx, oriented_ring
                    )

                    # Collect original mol atom indices for this fragment
                    # Use per-branch disambiguation for geminal substituents
                    frag_atoms = _collect_frag_atoms_for_fragment(
                        mol, ring_set, mol_atom_idx, frag,
                        branch_map, claimed_branches
                    )

                    if locant is not None:
                        results.append(SubstituentInfo(
                            frag_mol=frag,
                            locant=locant,
                            attach_mol_idx=mol_atom_idx,
                            frag_atoms=frag_atoms,
                        ))
                break  # only process first dummy atom per fragment

    return results


def _get_locant_from_oriented_ring(mol_atom_idx, oriented_ring):
    """Map a mol atom index to its IUPAC locant via oriented_ring.

    Args:
        mol_atom_idx: Atom index in the original mol.
        oriented_ring: List of atom indices in IUPAC numbering order.

    Returns:
        1-indexed IUPAC locant, or None if not found.
    """
    for pos, ring_atom in enumerate(oriented_ring):
        if ring_atom == mol_atom_idx:
            return pos + 1
    return None


def _compute_branch_map(mol, ring_set):
    """Compute per-branch atom sets for each ring atom.

    For geminal substituents (two substituents on the same ring atom), each
    separate branch needs its own atom set. This function BFS-es from each
    individual non-ring neighbor of each ring atom to produce separate sets.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.

    Returns:
        Dict mapping ring_atom_idx -> list of frozensets (one per branch).
    """
    branch_map = {}
    for ring_idx in ring_set:
        branches = []
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in ring_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set:
                continue
            # BFS from this specific neighbor to collect its branch
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx in ring_set:
                    continue
                visited.add(idx)
                for nbr2 in mol.GetAtomWithIdx(idx).GetNeighbors():
                    nbr2_idx = nbr2.GetIdx()
                    if nbr2_idx not in visited and nbr2_idx not in ring_set:
                        stack.append(nbr2_idx)
            if visited:
                branches.append(frozenset(visited))
        if branches:
            branch_map[ring_idx] = branches
    return branch_map


def _collect_frag_atoms_for_fragment(mol, ring_set, attach_ring_idx, frag_mol,
                                      branch_map, claimed_branches):
    """Collect original mol atom indices for a specific fragment.

    Handles geminal substituents by matching frag_mol composition against
    individual branches from branch_map, and tracking which branches have
    already been claimed.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index the fragment is attached to.
        frag_mol: RDKit Mol of the isolated fragment.
        branch_map: Dict from _compute_branch_map.
        claimed_branches: Mutable set of (attach_idx, branch_idx) tuples already used.

    Returns:
        Frozenset of original mol atom indices.
    """
    branches = branch_map.get(attach_ring_idx, [])

    if len(branches) <= 1:
        # Single substituent at this position -- use all non-ring neighbors
        return branches[0] if branches else frozenset()

    # Multiple branches (geminal): match fragment composition to find the right one
    # Count non-dummy heavy atoms in frag_mol
    frag_heavy = _count_frag_heavy_atoms(frag_mol)

    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key in claimed_branches:
            continue

        # Count heavy atoms in this branch from original mol
        branch_heavy = {}
        for idx in branch_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum != 1:  # skip hydrogen
                sym = atom.GetSymbol()
                branch_heavy[sym] = branch_heavy.get(sym, 0) + 1

        if branch_heavy == frag_heavy:
            claimed_branches.add(key)
            return branch_atoms

    # Fallback: if no exact match, claim the first unclaimed branch
    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key not in claimed_branches:
            claimed_branches.add(key)
            return branch_atoms

    # Last resort: return union of all branches (shouldn't happen)
    all_atoms = set()
    for b in branches:
        all_atoms.update(b)
    return frozenset(all_atoms)


def _count_frag_heavy_atoms(frag_mol):
    """Count non-dummy, non-H atoms in a fragment mol by element symbol.

    Returns:
        Dict of {symbol: count} for heavy atoms in the fragment.
    """
    counts = {}
    if frag_mol is None:
        return counts
    for atom in frag_mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 0 or anum == 1:
            continue
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _collect_frag_original_atoms(mol, ring_set, attach_ring_idx):
    """BFS from a ring atom to collect all non-ring substituent atoms.

    Note: This collects ALL substituent atoms from the ring atom. For geminal
    substituents, use _collect_frag_atoms_for_fragment instead.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index that the substituent is attached to.

    Returns:
        Frozenset of original mol atom indices belonging to the substituent.
    """
    visited = set()
    stack = []

    # Start from neighbors of the ring atom that are NOT in the ring
    ring_atom = mol.GetAtomWithIdx(attach_ring_idx)
    for nbr in ring_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx not in ring_set:
            stack.append(nbr_idx)

    while stack:
        idx = stack.pop()
        if idx in visited or idx in ring_set:
            continue
        visited.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_set:
                stack.append(nbr_idx)

    return frozenset(visited)


# ============================================================================
# Chain Substituent Extraction
# ============================================================================


def extract_chain_substituents(mol, principal_chain, substituents_dict):
    """Convert a substituents dict to a list of SubstituentInfo namedtuples.

    Takes the existing features.substituents dict (position -> list of
    atom index lists) and wraps each entry as a SubstituentInfo for
    unified downstream processing.

    Args:
        mol: RDKit Mol object.
        principal_chain: List of atom indices in the principal chain.
        substituents_dict: Dict mapping chain position (1-indexed) to list
            of substituent atom index lists.

    Returns:
        List of SubstituentInfo namedtuples.
    """
    if not substituents_dict:
        return []

    chain_set = set(principal_chain)
    results = []

    for position, sub_atom_lists in substituents_dict.items():
        for sub_atoms in sub_atom_lists:
            if not sub_atoms:
                continue

            # The first atom in sub_atoms is the one bonded to the chain
            attach_idx = sub_atoms[0] if isinstance(sub_atoms, (list, tuple)) else sub_atoms

            # Compute the actual atom index on the chain that this attaches to
            # position is 1-indexed into principal_chain
            if isinstance(position, int) and 1 <= position <= len(principal_chain):
                chain_atom_idx = principal_chain[position - 1]
            else:
                chain_atom_idx = attach_idx

            frag_atoms = frozenset(sub_atoms) if isinstance(sub_atoms, (list, tuple)) else frozenset([sub_atoms])

            results.append(SubstituentInfo(
                frag_mol=None,  # Chain substituents use atom indices, not frag mols
                locant=position,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frag_atoms,
            ))

    return results


# ============================================================================
# Fragment Classification and Naming
# ============================================================================


def classify_and_name_fragment(mol, frag_info, parent_atoms, features=None):
    """Classify a substituent fragment and produce its IUPAC prefix name.

    Routes each fragment to the appropriate naming function based on its
    composition:
      - fg_only: No carbon atoms (halogens, -OH, -NH2, -NO2, etc.)
      - pure_alkyl: Only carbon atoms (methyl, ethyl, etc.)
      - compound: Carbon + heteroatoms (trifluoromethyl, hydroxymethyl, etc.)

    Phase 160.1 D-04: Tier-0.5 prefix-form check runs FIRST so that any
    fragment matching the 14-row IUPAC P-65/P-66 prefix-form table is
    named via the canonical prefix form (e.g., -C(=O)OCH3 -> methoxycarbonyl)
    before falling through to compound/pure_alkyl/fg_only classification.
    This eliminates the polyfunctional-path duplicate-name bug
    (hydroxymethyl + methoxycarbonyl on the same ester atoms) per
    RESEARCH §3 root-cause fix.

    Args:
        mol: RDKit Mol object of the full molecule.
        frag_info: SubstituentInfo namedtuple for this substituent.
        parent_atoms: Set of atom indices in the parent structure.
        features: Optional features object (for additional context).

    Returns:
        IUPAC prefix name string (e.g., "methyl", "hydroxy", "trifluoromethyl"),
        or None if naming fails (with WARNING logged).
    """
    frag_mol = frag_info.frag_mol
    frag_atoms = frag_info.frag_atoms

    # ---- Tier 0.5 (Phase 160.1 D-04): IUPAC P-65 / P-66 prefix-form check ----
    # Pure read-only check. Applies to fragments that entirely contain one of
    # the 14 non-principal functional groups. The polyfunctional handler routes
    # substituent fragments here (via _name_compound_substituent fallback);
    # without this gate the compound-substituent path generates "hydroxymethyl"
    # for the methyl-ester fragment per RESEARCH §3 bug trace.
    try:
        frag_atom_set = set(frag_atoms) if not isinstance(frag_atoms, set) else frag_atoms
        attach_idx = getattr(frag_info, "attach_mol_idx", None)
        if attach_idx is None and frag_atom_set:
            attach_idx = next(iter(frag_atom_set))
        prefix_form = _check_substituent_prefix_form(
            mol, frag_atom_set, attach_idx
        )
        if prefix_form is not None:
            return prefix_form
    except Exception:
        pass

    # WS-A task 9: ring-containing fragments go to the single
    # ring-substituent chokepoint (P-29.2 free-valence locant:
    # naphthalen-2-yl, pyridin-2-yl, ...) — composition-based naming below
    # would count a ring's carbons as a chain. Recursion-safe:
    # name_ring_system_substituent only uses get_ring_substituent_name and
    # the name_substituent cascade, never this router.
    try:
        _ri = mol.GetRingInfo()
        if attach_idx is not None and any(
                _ri.NumAtomRings(a) > 0 for a in frag_atom_set):
            from ..rules.ring_substituents import name_ring_system_substituent
            _ring_nm = name_ring_system_substituent(
                mol, sorted(frag_atom_set), attach_idx
            )
            if _ring_nm:
                return _ring_nm
    except Exception:
        pass

    # Classify the fragment by composition
    category = _classify_fragment(mol, frag_mol, frag_atoms)

    if category == 'fg_only':
        return _name_fg_only(mol, frag_mol, frag_atoms)
    elif category == 'pure_alkyl':
        return _name_pure_alkyl(mol, frag_info, parent_atoms)
    elif category == 'compound':
        return _name_compound_substituent(mol, frag_info, parent_atoms)
    else:
        # Unknown -- log warning, never silently drop
        _frag_smiles = _get_frag_smiles(mol, frag_atoms)
        logger.warning(
            "Unrecognized substituent fragment at locant %s: %s",
            frag_info.locant, _frag_smiles
        )
        return None


def _classify_fragment(mol, frag_mol, frag_atoms):
    """Classify a fragment as fg_only, pure_alkyl, compound, or unknown.

    Uses the fragment mol if available (ring parent), otherwise uses
    original mol atom indices.

    Args:
        mol: RDKit Mol of the full molecule.
        frag_mol: RDKit Mol of the isolated fragment (may be None).
        frag_atoms: Set/frozenset of original mol atom indices.

    Returns:
        String category: 'fg_only', 'pure_alkyl', 'compound', or 'unknown'.
    """
    carbons = 0
    heteroatoms = 0

    if frag_mol is not None:
        # Use fragment mol (from ReplaceCore)
        for atom in frag_mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum == 0:
                continue  # skip dummy atom
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1
    else:
        # Use original mol indices
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1

    if carbons == 0 and heteroatoms > 0:
        return 'fg_only'
    elif carbons > 0 and heteroatoms == 0:
        return 'pure_alkyl'
    elif carbons > 0 and heteroatoms > 0:
        return 'compound'
    else:
        return 'unknown'


def _name_fg_only(mol, frag_mol, frag_atoms):
    """Name a fragment that is a pure functional group (no carbon).

    Checks halogens by symbol, then common FG patterns.

    Args:
        mol: RDKit Mol of full molecule.
        frag_mol: RDKit Mol of isolated fragment (may be None).
        frag_atoms: Set of original mol atom indices.

    Returns:
        IUPAC prefix name string, or None if unrecognized.
    """
    # Collect non-dummy, non-hydrogen atoms
    if frag_mol is not None:
        atoms_info = []
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue  # skip dummy
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })
    else:
        atoms_info = []
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                continue
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })

    if len(atoms_info) == 0:
        return None

    # Single atom cases
    if len(atoms_info) == 1:
        ai = atoms_info[0]
        sym = ai['symbol']

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]

        # -OH (oxygen with 1 H)
        if sym == 'O' and ai['total_hs'] >= 1:
            return 'hydroxy'

        # =O (oxo -- oxygen with no H, double bonded)
        if sym == 'O' and ai['total_hs'] == 0:
            return 'oxo'

        # -NH2 (nitrogen with 2 H)
        if sym == 'N' and ai['total_hs'] >= 2:
            return 'amino'

        # =NH (imino -- nitrogen with 1 H)
        if sym == 'N' and ai['total_hs'] == 1:
            return 'imino'

        # -SH (sulfanyl)
        if sym == 'S' and ai['total_hs'] >= 1:
            return 'sulfanyl'

        # =S (sulfanylidene)
        if sym == 'S' and ai['total_hs'] == 0:
            return 'sulfanylidene'

    # Multi-atom FG-only patterns
    if len(atoms_info) >= 2:
        symbols = sorted(ai['symbol'] for ai in atoms_info)

        # -NO2 (nitro): N + 2O
        if symbols == ['N', 'O', 'O']:
            return 'nitro'

        # -N3 (azido): 3 N atoms
        if symbols == ['N', 'N', 'N']:
            return 'azido'

    # Try seniority.get_prefix as fallback
    # For FG-only fragments on ring parents, try SMARTS matching
    if frag_mol is not None:
        from ..perception.functional_groups import FUNCTIONAL_GROUP_SMARTS
        for fg_name, smarts_str in FUNCTIONAL_GROUP_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern and frag_mol.HasSubstructMatch(pattern):
                prefix = get_prefix(fg_name)
                if prefix:
                    return prefix

    # Last resort for single-atom halogens on original mol
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        if sym == 'O' and atom.GetTotalNumHs() >= 1:
            return 'hydroxy'
        if sym == 'O' and atom.GetTotalNumHs() == 0:
            return 'oxo'
        if sym == 'N' and atom.GetTotalNumHs() >= 2:
            return 'amino'
        if sym == 'S' and atom.GetTotalNumHs() >= 1:
            return 'sulfanyl'

    logger.warning(
        "Could not name fg_only fragment with atoms: %s",
        [ai['symbol'] for ai in atoms_info]
    )
    return None


def _name_pure_alkyl(mol, frag_info, parent_atoms):
    """Name a pure alkyl substituent (carbon-only).

    Uses name_substituent_fragment for retained names (isopropyl, etc.)
    and recursive naming, with get_alkyl_name as fallback.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Alkyl prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom within the fragment
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        return name

    # Fallback: count carbons
    carbon_count = sum(
        1 for idx in frag_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )
    if carbon_count > 0:
        try:
            return get_alkyl_name(carbon_count)
        except (ValueError, KeyError):
            pass

    return None


def _name_compound_substituent(mol, frag_info, parent_atoms):
    """Name a compound substituent (carbon + heteroatoms).

    Handles haloalkyl (trifluoromethyl), hydroxyalkyl, aminoalkyl,
    alkoxy, sulfanylalkyl, and other compound types.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Compound prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    frag_set_for_chalcogen = set(frag_atoms)

    # DD2 Fix B (Phase D, P-63.3.1(1)): peroxy branch -O-O-R -> (R)peroxy.
    # Checked BEFORE the alkoxy branch because both attach through a divalent O;
    # only the peroxide case has a second O on the far side of the attach O.
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'O' and attach_atom.GetDegree() == 2 and any(
            n.GetSymbol() == 'O' and n.GetIdx() in frag_set_for_chalcogen
            and n.GetIdx() not in parent_atoms
            for n in attach_atom.GetNeighbors()
        ):
            name = _name_peroxy_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # DD2 Fix B (Phase D): disulfanyl branch -S-S-R -> (R)disulfanyl, before the
    # thioether (sulfanyl) branch (both attach through a divalent S).
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'S' and any(
            n.GetSymbol() == 'S' and n.GetIdx() in frag_set_for_chalcogen
            and n.GetIdx() not in parent_atoms
            for n in attach_atom.GetNeighbors()
        ):
            name = _name_disulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Special case: O-attached branches (ether substituents)
    # -O-R -> "alkoxy" (e.g., methoxy, ethoxy, phenoxy)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'O' and attach_atom.GetDegree() == 2:
            name = _name_alkoxy_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Special case: S-attached branches (thioether substituents)
    # -S-R -> "alkylsulfanyl" (e.g., methylsulfanyl, ethylsulfanyl)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'S' and attach_atom.GetTotalNumHs() == 0:
            name = _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Special case: N-attached branches (amino substituents)
    # -NH-R -> "alkylamino" (e.g., methylamino, phenylamino/anilino)
    # -NH-C(=O)-R -> "acylamino" (e.g., acetylamino, benzoylamino)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'N':
            name = _name_amino_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        # Reject garbled names from S/P-attached branches where
        # name_substituent_fragment doesn't handle the root heteroatom
        if attach_idx is not None:
            attach_sym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            if attach_sym in ('S', 'P') and 'thiyl' in name:
                pass  # fall through to warning
            else:
                return name

    # Fallback: recursive naming for ring-containing compound fragments.
    # Use name_fragment_recursively() which has cycle detection via visited set.
    # This handles cases where the fragment is a ring system with heteroatoms
    # that the simpler naming paths above cannot handle (DROP-18/19/24/25).
    # Guard: only for moderately-sized fragments (<=25 atoms).
    if len(frag_atoms) <= 25:
        frag_smiles = _get_frag_smiles(mol, frag_atoms)
        if frag_smiles and frag_smiles != "unknown":
            try:
                from .fragment_naming import name_fragment_recursively
                from .substituent_naming import parent_to_prefix
                frag_name = name_fragment_recursively(frag_smiles)
                if frag_name:
                    # Convert parent name to prefix form (e.g., "benzoic acid" -> not useful,
                    # but "pyridine" -> "pyridinyl", "cyclohexanone" -> "oxocyclohexyl")
                    carbon_count = sum(
                        1 for idx in frag_atoms
                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                    )
                    prefix_name = parent_to_prefix(frag_name, chain_length=carbon_count)
                    if prefix_name:
                        logger.debug(
                            "DROP-18/19 fallback: recursive naming for %s -> %s",
                            frag_smiles, prefix_name,
                        )
                        return prefix_name
            except Exception:
                pass  # Keep falling through to warning

    # If naming infrastructure couldn't handle it, log warning
    frag_smiles = _get_frag_smiles(mol, frag_atoms)
    logger.warning(
        "Could not name compound substituent at locant %s: %s",
        frag_info.locant, frag_smiles
    )
    return None


def _name_alkoxy_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name an alkoxy-attached branch: -O-R -> alkoxy.

    IUPAC P-63.2.3: Ether substituents named as alkoxy when the oxygen
    is the attachment point to the parent. Examples:
      -O-CH3 -> methoxy
      -O-C2H5 -> ethoxy
      -O-phenyl -> phenoxy
      -O-CH2-phenyl -> benzyloxy

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the O attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Alkoxy prefix name, or None if not a simple case.
    """
    from ..data.chain_names import get_chain_prefix

    frag_set = set(frag_atoms)
    o_atom = mol.GetAtomWithIdx(attach_idx)

    # Find non-parent neighbor of O (the R group)
    alkyl_start = None
    for nbr in o_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set:
            alkyl_start = nbr_idx
            break

    if alkyl_start is None:
        # Bare -O- with no R group (shouldn't happen for compound substituent)
        return None

    alkyl_atom = mol.GetAtomWithIdx(alkyl_start)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if alkyl_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if alkyl_start in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxy"
        return "phenoxy"

    # Case B: O -> CH(aryl)n -> benzyloxy (1 aryl) / diphenylmethoxy (2 phenyl).
    # HYG-04 (Phase 167): single shared aryl-count helper (was inline benzyloxy here).
    _aryl_ether = _name_aryl_methyl_ether(mol, alkyl_start, attach_idx)
    if _aryl_ether is not None:
        return _aryl_ether

    # Case C: O -> simple alkyl chain -> "methoxy", "ethoxy", etc.
    # Count carbons in the alkyl part (BFS from alkyl_start excluding O)
    visited = set()
    stack = [alkyl_start]
    carbon_count = 0
    has_heteroatom = False
    has_ring = False
    ring_info = mol.GetRingInfo()

    while stack:
        idx = stack.pop()
        if idx in visited or idx == attach_idx:
            continue
        if idx not in frag_set:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            carbon_count += 1
        elif atom.GetAtomicNum() != 1:
            has_heteroatom = True
        if ring_info.NumAtomRings(idx) > 0:
            has_ring = True
        for n in atom.GetNeighbors():
            if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                stack.append(n.GetIdx())

    # Only name as alkoxy if the R group is a pure alkyl chain (no heteroatoms, no rings)
    if has_heteroatom or has_ring or carbon_count == 0:
        return None

    ALKOXY_NAMES = {
        1: "methoxy", 2: "ethoxy", 3: "propoxy", 4: "butoxy",
        5: "pentyloxy", 6: "hexyloxy", 7: "heptyloxy", 8: "octyloxy",
        9: "nonyloxy", 10: "decyloxy",
    }

    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    elif carbon_count > 10:
        return get_chain_prefix(carbon_count) + "yloxy"
    return None


def _name_amino_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name an N-attached branch: -NH-R -> alkylamino, phenylamino/anilino.

    IUPAC P-62.2.3: Amine substituents named as amino when nitrogen is the
    attachment point. Also handles acylamino (-NH-C(=O)-R).

    Key patterns:
      -NH2 -> amino (handled as fg_only, not here)
      -NH-CH3 -> methylamino
      -NH-phenyl -> anilino (retained name for phenylamino)
      -NH-C(=O)-R -> acylamino (e.g., acetylamino, hexanoylamino)
      -N(CH3)2 -> dimethylamino

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the N attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Amino prefix name, or None if pattern not recognized.
    """
    from ..data.chain_names import get_chain_prefix

    frag_set = set(frag_atoms)
    n_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect non-parent, non-N neighbors
    branches = []
    for nbr in n_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set:
            branches.append(nbr_idx)

    if not branches:
        return None

    ring_info = mol.GetRingInfo()

    # Check for acylamino: -NH-C(=O)-R
    for branch_start in branches:
        branch_atom = mol.GetAtomWithIdx(branch_start)
        if branch_atom.GetSymbol() != 'C':
            continue
        # Check if this C has a =O (carbonyl)
        has_carbonyl = False
        for nbr in branch_atom.GetNeighbors():
            if nbr.GetIdx() == attach_idx:
                continue
            if nbr.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(branch_start, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() == 2.0:
                    has_carbonyl = True
                    break
        if has_carbonyl:
            # Count carbons in the acyl R-group (excluding the carbonyl C and =O)
            acyl_carbons = 0
            visited = set()
            stack_c = [branch_start]
            while stack_c:
                idx = stack_c.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                visited.add(idx)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    acyl_carbons += 1
                for n in atom.GetNeighbors():
                    if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                        stack_c.append(n.GetIdx())
            if acyl_carbons >= 1:
                # acyl_carbons includes the carbonyl C
                # IUPAC acyl nomenclature: chain_prefix + "anoyl" + "amino"
                # e.g., 2C = ethanoyl + amino, 3C = propanoyl + amino
                # Special case: 1C = formyl (methanoyl), but formylamino is rare
                acyl_name = get_chain_prefix(acyl_carbons) + "anoylamino"
                return acyl_name

    # Check for anilino: -NH-phenyl (isolated benzene ring directly on N)
    for branch_start in branches:
        branch_atom = mol.GetAtomWithIdx(branch_start)
        if branch_atom.GetIsAromatic() and branch_atom.GetSymbol() == 'C':
            for ring in ring_info.AtomRings():
                if branch_start in ring and len(ring) == 6:
                    all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                    all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                    if all_arom and all_c:
                        # Verify isolated (not fused)
                        ring_set_check = set(ring)
                        is_fused = any(
                            set(other) != ring_set_check and set(other) & ring_set_check
                            for other in ring_info.AtomRings()
                        )
                        if not is_fused:
                            return "anilino"

    # Simple amino: -NH-alkyl or -N(alkyl)2
    if len(branches) == 1:
        # BFS from branch to count carbons
        visited = set()
        stack_c = [branches[0]]
        carbon_count = 0
        has_hetero = False
        has_ring = False
        while stack_c:
            idx = stack_c.pop()
            if idx in visited or idx == attach_idx:
                continue
            if idx not in frag_set:
                continue
            visited.add(idx)
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'C':
                carbon_count += 1
            elif atom.GetAtomicNum() != 1:
                has_hetero = True
            if ring_info.NumAtomRings(idx) > 0:
                has_ring = True
            for n in atom.GetNeighbors():
                if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                    stack_c.append(n.GetIdx())
        if carbon_count > 0 and not has_hetero and not has_ring:
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                return f"{alkyl_name}amino"
            except (ValueError, KeyError):
                pass

    # -N(alkyl)2+ -> dialkylamino (e.g. dimethylamino). HYG-04 site#2 (Phase 167):
    # the disubstituted case the docstring promised but was never implemented, so
    # N,N-dialkylamino substituents on a chain parent fell to `return None` and were
    # mis-walked into a spurious amino+alkylamino split. Mirror the WORKING
    # principal-amine multiplicity (composer._assemble_amine_name: Counter +
    # SIMPLE_MULTIPLIERS) as a PREFIX form (no N- locants). Principal-amine path
    # (_assemble_amine_name) is untouched (Pitfall 3).
    if len(branches) >= 2:
        from collections import Counter

        branch_names = []
        all_pure = True
        for branch_start in branches:
            visited = set()
            stack_c = [branch_start]
            carbon_count = 0
            has_hetero = False
            has_ring = False
            while stack_c:
                idx = stack_c.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                visited.add(idx)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    carbon_count += 1
                elif atom.GetAtomicNum() != 1:
                    has_hetero = True
                if ring_info.NumAtomRings(idx) > 0:
                    has_ring = True
                for n in atom.GetNeighbors():
                    if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                        stack_c.append(n.GetIdx())
            if carbon_count <= 0 or has_hetero or has_ring:
                all_pure = False
                break
            try:
                branch_names.append(get_alkyl_name(carbon_count))
            except (ValueError, KeyError):
                all_pure = False
                break
        if all_pure and branch_names:
            counts = Counter(branch_names)
            parts = []
            # Alphabetical by alkyl stem (di-/tri- are ignored for ordering,
            # matching the principal-amine analog's sorted assembly).
            for nm in sorted(counts.keys()):
                count = counts[nm]
                if count == 1:
                    parts.append(nm)
                else:
                    mult = SIMPLE_MULTIPLIERS.get(count, str(count))
                    parts.append(f"{mult}{nm}")
            return f"{''.join(parts)}amino"

    return None


def _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name a sulfanyl-attached branch: -S-R -> alkylsulfanyl.

    For simple -S-alkyl branches, produces IUPAC substitutive prefix names:
    -S-CH3 -> methylsulfanyl
    -S-C2H5 -> ethylsulfanyl

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the S attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Sulfanyl prefix name, or None if not a simple case.
    """
    frag_set = set(frag_atoms)
    s_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect carbon atoms bonded to S (excluding parent)
    alkyl_atoms = []
    for nbr in s_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set and nbr.GetSymbol() == 'C':
            # BFS from this C to collect all connected carbons in fragment
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    visited.add(idx)
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                            stack.append(n.GetIdx())
            alkyl_atoms.extend(visited)

    if not alkyl_atoms:
        return None

    # Check for non-C non-H atoms in the alkyl portion
    has_hetero_in_alkyl = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in alkyl_atoms
    )
    if has_hetero_in_alkyl:
        return None  # Complex case, defer

    carbon_count = len(alkyl_atoms)
    if carbon_count == 0:
        return None

    try:
        alkyl_name = get_alkyl_name(carbon_count)
        return f"{alkyl_name}sulfanyl"
    except (ValueError, KeyError):
        return None


_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})  # O, S, Se, Te


def _is_divalent_chalcogen_atom(atom) -> bool:
    """True iff *atom* is a neutral, non-aromatic divalent chalcogen with only
    single bonds — the ``-O-``/``-S-`` ether-oxidation state of a peroxide /
    disulfide / thioperoxol linkage.

    CR-01 guard (Phase D code review): without this, a higher-oxidation-state S
    (sulfinyl ``-S(=O)-`` / sulfonyl ``-S(=O)(=O)-`` / thiosulfonate) was claimed
    as a disulfide and its ``=O`` atoms dropped, producing a parseable name for a
    DIFFERENT molecule. Mirrors ``skeletal_replacement._dichalcogen_bond_set`` and
    the ``benzene.py`` ring-substituent guard.
    """
    if atom.GetAtomicNum() not in _CHALCOGEN_ATOMIC_NUMS:
        return False
    if atom.GetFormalCharge() != 0 or atom.GetIsAromatic():
        return False
    from rdkit import Chem as _Chem
    return all(b.GetBondType() == _Chem.BondType.SINGLE for b in atom.GetBonds())


def _name_peroxy_or_disulfanyl_R(mol, r_start, boundary, frag_set):
    """Name the R group of a -O-O-R / -S-S-R substituent as a substituent prefix.

    Collects the R fragment (atoms in ``frag_set`` reachable from ``r_start``
    without crossing the two-chalcogen ``boundary``) and delegates to the shared
    ``name_substituent`` cascade so alkyl (methyl/ethyl), aryl (phenyl), and
    branched R groups are all named via one root-cause path. Returns None if R
    is empty.
    """
    visited = set()
    stack = [r_start]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in boundary or idx not in frag_set:
            continue
        visited.add(idx)
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = n.GetIdx()
            if nidx not in visited and nidx not in boundary:
                stack.append(nidx)
    if not visited:
        return None
    r_name = name_substituent(mol, sorted(visited), r_start)
    if not r_name:
        return None
    # WR-01 (Phase D code review): a COMPOUND R needs enclosing marks before the
    # outer peroxy/disulfanyl suffix is appended, so '[(methylperoxy)methyl]peroxy'
    # not the ambiguous '(methylperoxy)methylperoxy'. apply_enclosing_marks does the
    # ()->[]->{} nesting; a simple/retained R (methyl, phenyl) is returned bare.
    from .naming_utils import is_complex_substituent, apply_enclosing_marks
    _needs_marks = (
        is_complex_substituent(r_name)
        or '(' in r_name or '[' in r_name  # embedded enclosing marks (nested peroxy/disulfanyl)
    )
    if _needs_marks and not (
        r_name.startswith('(') and r_name.endswith(')')
        and r_name.count('(') == 1
    ):
        r_name = apply_enclosing_marks(r_name, depth=-1)
    return r_name


def _name_peroxy_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """DD2 Fix B (Phase D, P-63.3.1(1)): name a peroxy branch -O-O-R -> (R)peroxy.

    The substituent attaches to the parent through a divalent O whose other
    bond is to a second O (the peroxide linkage). The far side R is named as a
    substituent prefix and suffixed with ``peroxy``:
      -O-O-CH3   -> methylperoxy
      -O-O-C2H5  -> ethylperoxy

    Returns None if the attach atom is not a peroxide O or R cannot be named.
    """
    frag_set = set(frag_atoms)
    o1 = mol.GetAtomWithIdx(attach_idx)
    if o1.GetSymbol() != 'O' or not _is_divalent_chalcogen_atom(o1):
        return None
    o2_idx = None
    for nbr in o1.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms or nbr_idx not in frag_set:
            continue
        if nbr.GetSymbol() == 'O' and _is_divalent_chalcogen_atom(nbr):
            o2_idx = nbr_idx
            break
    if o2_idx is None:
        return None
    o2 = mol.GetAtomWithIdx(o2_idx)
    r_start = None
    for nbr in o2.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == attach_idx or nbr_idx not in frag_set:
            continue
        r_start = nbr_idx
        break
    if r_start is None:
        return None
    r_name = _name_peroxy_or_disulfanyl_R(mol, r_start, {attach_idx, o2_idx}, frag_set)
    if not r_name:
        return None
    return f"{r_name}peroxy"


def _name_disulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """DD2 Fix B (Phase D, P-63.3.1(1) / P-35.2.2): name a disulfanyl branch
    -S-S-R -> (R)disulfanyl; terminal -S-SH -> disulfanyl.

    Mirrors ``_name_peroxy_branch`` for the S-S linkage:
      -S-S-CH3 -> methyldisulfanyl
      -S-SH    -> disulfanyl (terminal; the H-bearing S carries no R)
    """
    frag_set = set(frag_atoms)
    s1 = mol.GetAtomWithIdx(attach_idx)
    if s1.GetSymbol() != 'S' or not _is_divalent_chalcogen_atom(s1):
        return None
    s2_idx = None
    for nbr in s1.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms or nbr_idx not in frag_set:
            continue
        if nbr.GetSymbol() == 'S' and _is_divalent_chalcogen_atom(nbr):
            s2_idx = nbr_idx
            break
    if s2_idx is None:
        return None
    s2 = mol.GetAtomWithIdx(s2_idx)
    r_start = None
    for nbr in s2.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == attach_idx or nbr_idx not in frag_set:
            continue
        r_start = nbr_idx
        break
    if r_start is None:
        # Terminal -S-SH: no R group -> bare disulfanyl (P-35.2.2).
        return "disulfanyl"
    r_name = _name_peroxy_or_disulfanyl_R(mol, r_start, {attach_idx, s2_idx}, frag_set)
    if not r_name:
        return None
    return f"{r_name}disulfanyl"


def _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms):
    """Find the fragment atom that is bonded to the parent structure.

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        parent_atoms: Set of atom indices in the parent.

    Returns:
        Atom index of the attachment atom, or None.
    """
    frag_set = set(frag_atoms)
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in parent_atoms:
                return idx
    return None


def _get_frag_smiles(mol, frag_atoms):
    """Get SMILES for a fragment defined by atom indices.

    Args:
        mol: RDKit Mol.
        frag_atoms: Collection of atom indices.

    Returns:
        SMILES string, or "unknown" if extraction fails.
    """
    try:
        atoms = list(frag_atoms)
        if atoms:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
            return smi if smi else "unknown"
    except Exception:
        pass
    return "unknown"


# ============================================================================
# Atom Set Collection (for deduplication)
# ============================================================================


def collect_substituent_atom_set(substituent_infos):
    """Collect the union of all substituent atom indices.

    Used by callers to prevent double-counting: FGs whose atoms are
    entirely within a named branch should be skipped in standalone
    FG prefix generation.

    Args:
        substituent_infos: List of SubstituentInfo namedtuples.

    Returns:
        Frozenset of all original mol atom indices covered by substituents.
    """
    all_atoms = set()
    for info in substituent_infos:
        if info.frag_atoms:
            all_atoms.update(info.frag_atoms)
    return frozenset(all_atoms)
