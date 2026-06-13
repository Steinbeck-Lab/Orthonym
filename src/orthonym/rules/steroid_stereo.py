"""Steroid ring-face α/β configurational descriptors (IUPAC P-101.2.6, Phase 181 WSC-02).

Generate ring-face α/β descriptors (``3beta``, ``5alpha``) for steroid scaffolds by
INVERTING OPSIN's own forward parser. OPSIN parses α/β names by applying a parity to a
ring stereocentre from its ``alphaBetaClockWiseAtomOrdering`` (ABO) — see
``opsin StereochemistryHandler.applyAlphaBetaStereochemistryToStereoCentre`` (Java
842-918) and ``SMILESWriter.atomParityToSmiles`` (880-952). This module runs that
algorithm backwards: read the molecule's RDKit chiral parity at each ABO ring locant
and map it back to α/β. Because it inverts OPSIN's own data + algorithm, the emitted
descriptor round-trips through OPSIN by construction (D-01).

Sign convention (D-02): empirically pinned this build — OPSIN parity ``+1 → beta``,
``-1 → alpha``. The 6-structure parity unit test is the tripwire; if a future RDKit
upgrade flips neighbour-ordering semantics it fails loudly and ``_SIGN`` is flipped
ONCE, globally — never per-molecule.

Root-cause-only (D-09): no postprocessor, no regex on the existing ``(3R,5S,...)`` string,
no seniority/dispatch edit. All logic is parity arithmetic + dict lookups + set math.
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

# Empirically pinned sign convention (this session): OPSIN parity +1 → beta, −1 → alpha.
_SIGN = {1: 'beta', -1: 'alpha'}
BRIDGEHEADS = {8, 9, 10, 13, 14}

_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _perm_parity(perm):
    """Bubble-sort swap count mod 2 (mirrors opsin swapsRequiredToSort, Java 1061-1082)."""
    s = list(perm)
    swaps = 0
    for i in range(len(s) - 1, 0, -1):
        for j in range(i):
            if s[j] > s[j + 1]:
                s[j], s[j + 1] = s[j + 1], s[j]
                swaps += 1
    return swaps % 2


def alpha_beta_at(mol, locant, loc2idx, idx2loc, ringorder):
    """Return 'alpha'/'beta' for the ring stereocentre at IUPAC ``locant``, or None if
    it is not a tetrahedral centre / unresolvable (→ triggers no-mix fallback if cited).

    Inverts opsin StereochemistryHandler.applyAlphaBetaStereochemistryToStereoCentre:
    atomRefs4 = [prev-in-ABO, classified-neighbour, classified-neighbour, next-in-ABO];
    parity +1 → beta, −1 → alpha.
    """
    if locant not in ringorder or locant not in loc2idx:
        return None
    atom = mol.GetAtomWithIdx(loc2idx[locant])
    tag = atom.GetChiralTag()
    if tag not in _TETRAHEDRAL:
        return None  # sp2 / aromatic (e.g. estradiol or Δ5 C-5) / undefined — NOT a centre
    p = ringorder.index(locant)
    a0 = loc2idx.get(ringorder[p - 1])                       # atomRefs4[0] = previous in clockwise order
    a3 = loc2idx.get(ringorder[(p + 1) % len(ringorder)])    # atomRefs4[3] = next
    if a0 is None or a3 is None:
        return None
    nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
    numH = atom.GetTotalNumHs()
    remaining = [x for x in nbrs if x not in (a0, a3)]
    if numH == 1:
        remaining = remaining + [-1]                          # sentinel for the implicit H
    if len(remaining) != 2:
        return None                                           # not a clean 4-coordinate centre
    r1, r2 = remaining
    in_ring = lambda x: x != -1 and x in idx2loc and idx2loc[x] in ringorder
    # OPSIN neighbour classification: ring-in-order > H > acyclic/substituent (Java 857-880)
    if in_ring(r1):
        a1, a2 = r1, r2
    elif in_ring(r2):
        a1, a2 = r2, r1
    elif r1 == -1 != r2:
        a1, a2 = r2, r1                                       # a1 = the non-H
    elif r2 == -1 != r1:
        a1, a2 = r1, r2
    else:
        a1, a2 = r1, r2                                       # substituent fallthrough (e.g. 17β-yl)
    order = [a0, a1, a2, a3]
    # RDKit reference order for the tag: implicit-H FIRST when degree-3 + 1 H.
    ref = ([-1] + nbrs) if (len(nbrs) == 3 and numH == 1) else nbrs
    pos = {v: i for i, v in enumerate(ref)}
    try:
        perm = [pos[v] for v in order]
    except KeyError:
        return None
    base = 1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1
    parity = base if _perm_parity(perm) == 0 else -base
    return _SIGN[parity]


def _ring_wiring(scaffold_info):
    """Return (ringorder, loc2idx, idx2loc) for the detected scaffold, or None.

    ringorder is the ABO in IUPAC-locant space; loc2idx/idx2loc bridge it to the
    target molecule's atom indices. Returns None when the ABO or numbering map is
    absent — NO ad-hoc ordering is ever synthesized (D-03).
    """
    from ..data.opsin_imports.natural_products_opsin import OPSIN_NATURAL_PRODUCTS
    from .natural_products import _build_target_to_iupac

    scaffold_smiles = scaffold_info.get("scaffold_smiles")
    entry = OPSIN_NATURAL_PRODUCTS.get(scaffold_smiles)
    if entry is None:
        return None
    abo = entry.get("alphaBetaClockWiseAtomOrdering")
    if abo is None:
        return None
    try:
        ringorder = [int(x) for x in abo.split("/")]
    except (ValueError, AttributeError):
        return None
    idx2loc = _build_target_to_iupac(scaffold_info)          # {target_atom_idx: IUPAC locant}
    if not idx2loc:
        return None
    loc2idx = {loc: idx for idx, loc in idx2loc.items()}
    return ringorder, loc2idx, idx2loc


def _reference_profile(scaffold_smiles, ringorder):
    """Compute the implied-stereoparent α/β profile over the catalogued scaffold (D-05).

    The catalogued NATURAL_PRODUCT_SCAFFOLDS SMILES carry the natural @/@@ configuration,
    so running the SAME recipe over the reference mol yields the parent's natural face at
    each ring locant. A target ring stereocentre is cited as INVERTED only where it differs
    from this reference (D-04c). Read ONLY for the diff — the emitted value is always the
    target's own parity (D-15). Returns {} (conservative: nothing inverted) on any failure.
    """
    from ..data.natural_products import get_scaffold_numbering
    from ..perception.stereo import assign_stereochemistry

    ref_mol = Chem.MolFromSmiles(scaffold_smiles)
    if ref_mol is None:
        return {}
    numbering = get_scaffold_numbering(scaffold_smiles)      # {query_pos: locant}; query_pos == atom idx
    if not numbering:
        return {}
    try:
        assign_stereochemistry(ref_mol)
    except Exception:
        return {}
    ref_idx2loc = {idx: loc for idx, loc in numbering.items()}
    ref_loc2idx = {loc: idx for idx, loc in ref_idx2loc.items()}
    profile = {}
    for loc in ringorder:
        ab = alpha_beta_at(ref_mol, loc, ref_loc2idx, ref_idx2loc, ringorder)
        if ab is not None:
            profile[loc] = ab
    return profile


def collect_steroid_alpha_beta(mol, scaffold_info, numbering):
    """Return {'ring_ab': {locant: 'alpha'/'beta'}, 'side_rs': [(locant, cip), ...]} for a
    steroid, or None to signal the per-molecule no-mix fallback (D-08).

    - ring_ab cites: C-5 when chiral (D-04a) ∪ substituent/suffix-bearing ring stereocentres
      (D-04b) ∪ ring stereocentres inverted vs the implied parent (D-04c). Natural-config
      fixed stereocentres (bridgeheads C-8/9/10/13/14, C-17, etc.) are suppressed.
    - side_rs keeps acyclic side-chain stereocentres (C-20/22/24/25 — NOT in the ABO) as R/S,
      a separate leading block (D-06); ring atoms are never passed into this collection.
    - Returns None (whole-molecule R/S fallback) iff any DEFINED ring stereocentre cannot be
      resolved to α/β — never mixes ring α/β with ring R/S. An sp2 C-5 (Δ5) or sp2 ketone
      carbon is not a defined stereocentre → never a fallback trigger (Pitfall 1).
    """
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors

    assign_stereochemistry(mol)

    wiring = _ring_wiring(scaffold_info)
    if wiring is None:
        return None                                          # D-03/D-08: fall back to R/S
    ringorder, loc2idx, idx2loc = wiring
    ringset = set(ringorder)

    # Defined ring stereocentres = ABO locants whose target atom carries a tetrahedral chiral tag.
    ring_stereo_locants = [
        loc for loc in ringorder
        if loc in loc2idx and mol.GetAtomWithIdx(loc2idx[loc]).GetChiralTag() in _TETRAHEDRAL
    ]

    # NO-MIX FALLBACK (D-08, Pitfall 2): every DEFINED ring stereocentre must resolve to α/β,
    # else discard all α/β and emit the whole-graph R/S string.
    target_ab = {}
    for loc in ring_stereo_locants:
        ab = alpha_beta_at(mol, loc, loc2idx, idx2loc, ringorder)
        if ab is None:
            return None
        target_ab[loc] = ab

    matched = set(scaffold_info.get("matched_atoms", ()))
    # Decorated ring stereocentre: bears an exocyclic neighbour outside the scaffold skeleton.
    decorated = set()
    for loc in target_ab:
        atom = mol.GetAtomWithIdx(loc2idx[loc])
        if any(n.GetIdx() not in matched for n in atom.GetNeighbors()):
            decorated.add(loc)

    ref = _reference_profile(scaffold_info.get("scaffold_smiles"), ringorder)

    cited = set()
    if 5 in target_ab:                                       # D-04a: C-5 always when chiral
        cited.add(5)
    cited |= decorated                                       # D-04b: substituent/suffix-bearing

    # D-04c / D-08: every IMPLIED (non-cited) ring stereocentre — bridgeheads C-8/9/10/13/14
    # and other fixed centres — must match the implied-stereoparent reference. If any differs,
    # we cannot represent it with the suppressed natural config, so fall back to the whole-graph
    # R/S string (which represents ANY configuration). This (a) names genuinely inverted steroids
    # correctly via R/S and (b) guards against substructure-match orientation ambiguity at the
    # symmetric bridgeheads — we NEVER emit a wrong α/β bridgehead (the documented RT-breaker).
    # Cited centres (C-5 + decorated) are oriented unambiguously by their unique substituents.
    for loc in target_ab:
        if loc in cited:
            continue
        if ref.get(loc) is not None and target_ab[loc] != ref[loc]:
            return None

    ring_ab = {loc: target_ab[loc] for loc in cited}

    # SIDE-CHAIN R/S (D-06, Pitfall 3): stereocentres whose locant is NOT in the ABO.
    side_map = {idx: loc for idx, loc in numbering.items() if loc not in ringset}
    side_rs: List[Tuple[int, str]] = collect_stereodescriptors(mol, side_map) if side_map else []

    return {"ring_ab": ring_ab, "side_rs": side_rs}
