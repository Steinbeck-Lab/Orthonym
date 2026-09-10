"""Ring-chalcogen oxide namer /, Wave-2 completion).

A NEUTRAL ring sulfur/selenium/tellurium bearing 1-2 exocyclic terminal =O
is named additively on the intact ring parent — the S-oxide sibling of the
established ``pyridine 1-oxide`` N-oxide path::

    O=S1c2ccccc2-c2ccccc21 -> dibenzo[b,d]thiophene 5-oxide
    O=S1(=O)c2ccccc2-c2ccccc21 -> dibenzo[b,d]thiophene 5,5-dioxide
    O=S1CCCC1 -> thiolane 1-oxide

SCOPE (fail-closed, accuracy-first): a single oxidised ring chalcogen; every
non-ring heavy atom of the molecule is one of its oxide oxygens (bare ring
system otherwise — substituted variants cascade onward); the de-oxidised base
ring must resolve BOTH a name and an authoritative IUPAC locant for the
chalcogen. Anything else returns None — never a wrong name.

Graph/atom classifier (no SMARTS broadening); pure — no mol mutation.
"""

from typing import Optional

from rdkit import Chem

from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)

_CHALCOGENS = {'S', 'Se', 'Te'}

_HW_RING_SIZES = frozenset(range(3, 11))


def _name_sultone(mol) -> Optional[str]:
    """ sultone (method 1 = PIN): the intramolecular ester of a
    hydroxy sulfonic acid — a saturated monocyclic ring with an ester O adjacent
    to a ring S(=O)2 — named on the Hantzsch-Widman oxathiolane/oxathiane parent
    bearing the λ6 convention on the S and a ``-2,2-dione`` suffix::

        O=S1(=O)CCCO1 -> 1,2lambda6-oxathiolane-2,2-dione
        CC1CCCS(=O)(=O)O1 -> 3-methyl-1,2lambda6-oxathiane-2,2-dione

    This is the PREFERRED form; the additive functional-class '1,2-oxathiolane
    2,2-dioxide' (method 3) is BB's explicit non-PIN alternative.

    Fail-closed scope (accuracy-first): one saturated monocyclic ring (size
    3-10) of only C + one ester O + one S; the S bears exactly two exocyclic
    terminal =O (λ6), one ring-O ester neighbour and one ring-C neighbour.
    Declines the aromatic thiophene S-dioxides (no ring ester O; kept as the
    additive 'dioxide') and the sultams (ring N, not O)."""
    from .heterocycles import build_hw_name
    from .lambda_convention import nonstandard_bonding_number

    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return None
    ring = list(ri.AtomRings()[0])
    n = len(ring)
    if n not in _HW_RING_SIZES:
        return None
    ring_set = set(ring)

    # Saturated ring skeleton (the S=O bonds are exocyclic).
    for b in bonds_of(mol):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.SINGLE:
                return None

    s_idx = None
    ester_o = None
    oxide_os = []
    for i in ring:
        at = mol.GetAtomWithIdx(i)
        sym = at.GetSymbol()
        if sym == 'S':
            if s_idx is not None:
                return None
            exo_o = []
            for nb in at.GetNeighbors():
                if nb.GetIdx() in ring_set:
                    continue
                bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
                if (nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                        and nb.GetTotalNumHs() == 0
                        and bond.GetBondType() == Chem.BondType.DOUBLE):
                    exo_o.append(nb.GetIdx())
                else:
                    return None  # other exocyclic group on S -> not this class
            if len(exo_o) != 2:
                return None
            s_idx, oxide_os = i, exo_o
        elif sym == 'O':
            if at.GetTotalNumHs() != 0:
                return None
            nbrs = [nb.GetIdx() for nb in at.GetNeighbors()]
            if len(nbrs) != 2 or any(x not in ring_set for x in nbrs):
                return None
            if ester_o is not None:
                return None  # >1 ring O -> not a simple sultone
            ester_o = i
        elif sym == 'C':
            continue
        else:
            return None  # ring N (sultam) / other heteroatom -> not a sultone
    if s_idx is None or ester_o is None:
        return None

    # Ester linkage: S adjacent to the ring O and to a ring C; λ6 on the S.
    s_ring_nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(s_idx).GetNeighbors()
                   if nb.GetIdx() in ring_set]
    if len(s_ring_nbrs) != 2 or ester_o not in s_ring_nbrs:
        return None
    other = next(x for x in s_ring_nbrs if x != ester_o)
    if mol.GetAtomWithIdx(other).GetSymbol() != 'C':
        return None
    lam = nonstandard_bonding_number(mol, s_idx)
    if lam != 6:
        return None

    # Numbering: ester O = 1, S = 2 (O has HW priority over S and they are
    # adjacent, so this set {1,2} is forced), carbons continue from the S.
    order = [ester_o, s_idx]
    prev, cur = ester_o, s_idx
    while True:
        nxts = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                if nb.GetIdx() in ring_set and nb.GetIdx() != prev
                and nb.GetIdx() not in order]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    if len(order) != n:
        return None
    loc = {a: k + 1 for k, a in enumerate(order)}

    parent = build_hw_name(
        heteroatoms=[(1, 'O'), (2, 'S')], ring_size=n,
        is_saturated=True, is_aromatic=False, lambda_by_locant={2: lam},
    )
    if not parent:
        return None
    core = f"{parent}-2,2-dione"  # 'dione' is consonant-initial: no elision

    # Substituents on the ring carbons (fail-closed).
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    from ..assembly.substituent_enumerator import name_substituent
    oxide_set = set(oxide_os)
    subs = []
    for a in order:
        if a in (ester_o, s_idx):
            continue
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring_set or j in oxide_set or nb.GetAtomicNum() <= 1:
                continue
            frag, seen, stack = [], set(ring_set) | oxide_set, [j]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                frag.append(x)
                stack.extend(nb2.GetIdx() for nb2 in mol.GetAtomWithIdx(x).GetNeighbors()
                             if nb2.GetIdx() not in seen)
            nm = name_substituent(mol, frag, j)
            if not nm:
                return None  # unnameable substituent -> fail closed
            subs.append((loc[a], nm))

    if not subs:
        return core
    groups = {}
    for locant, nm in subs:
        groups.setdefault(nm, []).append(locant)
    parts = []
    for nm in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[nm])
        m = get_multiplier_prefix(len(locs), nm) if len(locs) > 1 else ""
        parts.append(f"{','.join(str(x) for x in locs)}-{m}{nm}")
    prefix = "-".join(parts)
    sep = "-" if core[:1].isdigit() else ""
    return f"{prefix}{sep}{core}"


def name_ring_chalcogen_oxide(mol) -> Optional[str]:
    """Return the additive ring-chalcogen oxide name, else ``None``."""
    if mol is None:
        return None

    # sultone (cyclic ester of a hydroxy sulfonic acid): the
    # λ6-dione PIN takes precedence over the additive '...dioxide' form.
    sultone = _name_sultone(mol)
    if sultone is not None:
        return sultone

    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return None
    ring_atoms = set()
    for r in ri.AtomRings():
        ring_atoms.update(r)

    # Locate the oxidised ring chalcogen and its terminal =O oxygens.
    site = None
    oxide_oxygens = []
    for atom in atoms_of(mol):
        if atom.GetSymbol() not in _CHALCOGENS:
            continue
        if atom.GetIdx() not in ring_atoms:
            continue
        oxos = []
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if (nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                    and nb.GetTotalNumHs() == 0
                    and bond.GetBondType() == Chem.BondType.DOUBLE):
                oxos.append(nb.GetIdx())
        if not oxos:
            continue
        if site is not None:
            return None  # two oxidised sites — not built, fail closed
        if len(oxos) > 2:
            return None
        site = atom.GetIdx()
        oxide_oxygens = oxos
    if site is None:
        return None

    # Bare ring system: every non-ring heavy atom must be an oxide oxygen.
    non_ring = {a.GetIdx() for a in atoms_of(mol)} - ring_atoms
    if non_ring != set(oxide_oxygens):
        return None

    # De-oxidise: remove the =O atoms; the base must sanitize and name.
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp('__rco_orig', a.GetIdx())
    for idx in sorted(oxide_oxygens, reverse=True):
        rw.RemoveAtom(idx)
    base = rw.GetMol()
    try:
        Chem.SanitizeMol(base)
        base_smiles = Chem.MolToSmiles(base, canonical=True)
    except Exception:
        return None

    from ..assembly.fragment_naming import name_fragment_recursively
    try:
        base_name = name_fragment_recursively(base_smiles)
    except Exception:
        return None
    if not base_name or 'unknown' in base_name:
        return None

    # Authoritative locant of the chalcogen under the BASE ring numbering
    # (dibenzothiophene S = 5). Fail closed when the ring-info cascade offers
    # no full iupac_locants map — a guessed locant would be a wrong name.
    base_site = next(
        (i for i in range(base.GetNumAtoms())
         if base.GetAtomWithIdx(i).GetIntProp('__rco_orig') == site), None)
    if base_site is None:
        return None
    try:
        from ..namer import _build_ring_info_for_parent_selection, compute_features
        feats = compute_features(base)
        rinfo = _build_ring_info_for_parent_selection(feats)
    except Exception:
        return None
    iupac = (rinfo or {}).get('iupac_locants')
    if not iupac or base_site not in iupac:
        # Single-ring base: the chalcogen is locant 1 by HW convention
        # (thiolane 1-oxide) — accept ONLY the unambiguous one-heteroatom
        # monocycle; everything else fails closed.
        base_ri = base.GetRingInfo()
        if base_ri.NumRings() == 1:
            ring = base_ri.AtomRings()[0]
            hetero_in_ring = [i for i in ring
                              if base.GetAtomWithIdx(i).GetAtomicNum()
                              not in (1, 6)]
            if hetero_in_ring == [base_site]:
                loc = 1
            else:
                return None
        else:
            return None
    else:
        loc = iupac[base_site]
        if isinstance(loc, tuple):
            loc = loc[0]
        if not isinstance(loc, int):
            return None

    if len(oxide_oxygens) == 1:
        return f"{base_name} {loc}-oxide"
    return f"{base_name} {loc},{loc}-dioxide"


__all__ = ["name_ring_chalcogen_oxide"]
