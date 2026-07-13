"""
Sulfur compound naming rules per IUPAC 2013.

IUPAC P-63.6: Sulfur-containing functional groups:
- Thiols (-SH): suffix -thiol, prefix sulfanyl- (P-63.6.1.1)
- Sulfides (R-S-R'): functional class naming (P-63.6.2.1)
- Sulfoxides (R-SO-R'): functional class naming (P-63.6.3.1)
- Sulfones (R-SO2-R'): functional class naming (P-63.6.3.2)
- Sulfonic acids (-SO3H): suffix -sulfonic acid, prefix sulfo- (P-65.3.1.2)
"""

from typing import Optional, Tuple, List
from collections import deque
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name


def name_thiol(mol, thiol_atoms: Tuple[int, ...], parent_name: str, locant: Optional[int] = None) -> str:
    """
    Name a thiol compound with -thiol suffix.

    Args:
        mol: RDKit Mol object
        thiol_atoms: Atom indices from SMARTS match (S, C)
        parent_name: Parent chain/ring name without suffix
        locant: Position of thiol group (None if implied)

    Returns:
        Name like "methanethiol", "propane-1-thiol"
    """
    # Terminal thiols: locant is 1, often omitted for 1-2 carbon chains
    if locant is None or locant == 1:
        # For methane/ethane, no locant needed
        if parent_name in ("methan", "ethan"):
            return f"{parent_name}ethiol"
        # For longer chains, include locant in PIN style
        return f"{parent_name}e-1-thiol"
    else:
        return f"{parent_name}e-{locant}-thiol"


def name_sulfide(mol, sulfur_idx: int) -> Optional[str]:
    """
    Name a sulfide (thioether) using functional class nomenclature.

    IUPAC P-63.6.2.1 prefers functional class for simple sulfides:
    - Symmetric: "dimethyl sulfide", "diethyl sulfide"
    - Asymmetric: "ethyl methyl sulfide" (alphabetical order, P-14.4)

    Args:
        mol: RDKit Mol object
        sulfur_idx: Index of sulfur atom

    Returns:
        Functional class name, or None if not a simple sulfide
    """
    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # A RING sulfur is never an acyclic functional-class sulfide ("R R' sulfide",
    # P-63.6.2.1) — it is a skeletal heteroatom named by the ring system (thiophene,
    # thiane, the epithio bridge of a bridged-fused parent, ...). Characterising its
    # two ring branches as substituent groups LINEARISES the ring into a phantom
    # chain (e.g. the S-bridged 1,4-epithio-1,4-dihydronaphthalene -> "didecyl
    # sulfide" for some SMILES spellings — an order-dependent WRONG name). Decline so
    # the ring/heterocycle path names it. (The thioether handler's ring_type guard is
    # spelling-fragile; this chemical-logic guard is spelling-independent.)
    if sulfur.IsInRing():
        return None

    # Get carbon neighbors
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Characterize each substituent (aryl or alkyl)
    sub_names = []
    for neighbor in neighbors:
        name, count = _characterize_sulfur_substituent(mol, neighbor.GetIdx(), {sulfur_idx})
        if name is None:
            return None  # Unrecognized substituent
        sub_names.append(name)

    # Sort alphabetically
    sub_names.sort()

    # Check for symmetry
    if sub_names[0] == sub_names[1]:
        return f"di{sub_names[0]} sulfide"
    else:
        return f"{sub_names[0]} {sub_names[1]} sulfide"


def name_sulfoxide(mol, sulfoxide_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a sulfoxide using functional class nomenclature.

    IUPAC P-63.6.3.1 prefers functional class for simple sulfoxides:
    - Symmetric: "dimethyl sulfoxide"
    - Asymmetric: "ethyl methyl sulfoxide" (alphabetical order, P-14.4)

    Args:
        mol: RDKit Mol object
        sulfoxide_atoms: Atom indices from SMARTS match

    Returns:
        Functional class name, or None if not a simple sulfoxide
    """
    # Find the sulfur atom (has =O and 2 C neighbors)
    sulfur_idx = None
    for idx in sulfoxide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'S':
            sulfur_idx = idx
            break

    if sulfur_idx is None:
        return None

    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # Get carbon neighbors (exclude oxygen)
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Characterize each substituent (aryl or alkyl)
    sub_names = []
    for neighbor in neighbors:
        name, count = _characterize_sulfur_substituent(mol, neighbor.GetIdx(), {sulfur_idx})
        if name is None:
            return None
        sub_names.append(name)

    # Sort alphabetically
    sub_names.sort()

    # Check for symmetry
    if sub_names[0] == sub_names[1]:
        return f"di{sub_names[0]} sulfoxide"
    else:
        return f"{sub_names[0]} {sub_names[1]} sulfoxide"


def name_sulfone(mol, sulfone_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a sulfone using functional class nomenclature.

    IUPAC P-63.6.3.2 prefers functional class for simple sulfones:
    - Symmetric: "dimethyl sulfone"
    - Asymmetric: "ethyl methyl sulfone" (alphabetical order, P-14.4)

    Args:
        mol: RDKit Mol object
        sulfone_atoms: Atom indices from SMARTS match

    Returns:
        Functional class name, or None if not a simple sulfone
    """
    # Find the sulfur atom
    sulfur_idx = None
    for idx in sulfone_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'S':
            sulfur_idx = idx
            break

    if sulfur_idx is None:
        return None

    sulfur = mol.GetAtomWithIdx(sulfur_idx)

    # Get carbon neighbors (exclude oxygens)
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:
        return None

    # Characterize each substituent (aryl or alkyl)
    sub_names = []
    for neighbor in neighbors:
        name, count = _characterize_sulfur_substituent(mol, neighbor.GetIdx(), {sulfur_idx})
        if name is None:
            return None
        sub_names.append(name)

    # Sort alphabetically
    sub_names.sort()

    # Check for symmetry
    if sub_names[0] == sub_names[1]:
        return f"di{sub_names[0]} sulfone"
    else:
        return f"{sub_names[0]} {sub_names[1]} sulfone"


def name_sulfonic_acid(mol, sulfonic_atoms: Tuple[int, ...], parent_name: str) -> str:
    """
    Name a sulfonic acid with -sulfonic acid suffix.

    Args:
        mol: RDKit Mol object
        sulfonic_atoms: Atom indices from SMARTS match
        parent_name: Parent chain/ring name

    Returns:
        Name like "methanesulfonic acid", "benzenesulfonic acid"
    """
    # Sulfonic acid always at chain/ring end, no locant needed
    return f"{parent_name}sulfonic acid"


def _classify_oxide_side(mol, c_idx: int, sulfur_idx: int):
    """Classify one R side of R-S(=O)x-R' for substitutive P-63.6 naming.

    Wave2 T3b. Returns ``(stem, kind, atoms)`` where ``stem`` is the
    parent-hydride name used both for the acid-form prefix
    ('{stem}sulfinyl': methanesulfinyl / benzenesulfinyl / cyclohexanesulfinyl)
    and as the parent name when this side wins parent selection; ``kind`` is
    'chain' or 'ring'; ``atoms`` is the side's full atom set. Returns None
    for any side that is not one of the honestly-nameable shapes (linear
    terminal saturated all-C chain; plain benzene; plain saturated
    cycloalkane) — the caller then falls back to functional class, never
    fabricating a name from a carbon count.
    """
    from ..data.chain_names import get_chain_prefix

    # Full side BFS (never cross S)
    side = set()
    q = deque([c_idx])
    while q:
        i = q.popleft()
        if i in side or i == sulfur_idx:
            continue
        side.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            ni = nb.GetIdx()
            if ni not in side and ni != sulfur_idx:
                q.append(ni)

    # all-carbon only
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in side):
        return None

    ri = mol.GetRingInfo()
    in_ring = [i for i in side if ri.NumAtomRings(i) > 0]
    if in_ring:
        # the side must be EXACTLY one plain unsubstituted monocycle
        if set(in_ring) != side:
            return None
        rings = [set(r) for r in ri.AtomRings() if set(r) == side]
        if len(rings) != 1:
            return None
        n = len(side)
        atoms = [mol.GetAtomWithIdx(i) for i in side]
        if n == 6 and all(a.GetIsAromatic() for a in atoms):
            return ('benzene', 'ring', side)
        if all(not a.GetIsAromatic() for a in atoms) and all(
            mol.GetBondBetweenAtoms(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            .GetBondTypeAsDouble() == 1.0
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in side and b.GetEndAtomIdx() in side
        ):
            try:
                return (f"cyclo{get_chain_prefix(n)}ane", 'ring', side)
            except (ValueError, KeyError):
                return None
        return None

    # acyclic: linear unbranched saturated chain attached at its terminus
    path = [c_idx]
    seen = {c_idx}
    cur = c_idx
    while True:
        nxt = [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
            if nb.GetIdx() in side and nb.GetIdx() not in seen
        ]
        if len(nxt) > 1:
            return None
        if not nxt:
            break
        b = mol.GetBondBetweenAtoms(cur, nxt[0])
        if b is None or b.GetBondTypeAsDouble() != 1.0:
            return None
        cur = nxt[0]
        seen.add(cur)
        path.append(cur)
    if len(path) != len(side):
        return None
    try:
        return (f"{get_chain_prefix(len(side))}ane", 'chain', side)
    except (ValueError, KeyError):
        return None


def name_chalcogen_oxide_substitutive(
    mol, match_atoms: Tuple[int, ...], oxide_kind: str,
) -> Optional[str]:
    """P-63.6 substitutive PIN for R-S(=O)-R' / R-S(=O)(=O)-R' (Wave2 T3b).

    BB-verbatim targets: '(methanesulfinyl)methane' (46154, DMSO),
    '1-(ethanesulfinyl)butane' (28094), '(ethanesulfonyl)ethane' (28115),
    '(methanesulfinyl)benzene', "1,1'-sulfinyldibenzene" (28110, symmetric
    diaryl multiplicative; 'Multiplication of acyclic hydrocarbons is not
    permitted' — identical chains stay substitutive).

    Args:
        mol: RDKit Mol.
        match_atoms: sulfoxide/sulfone SMARTS match (S first).
        oxide_kind: 'sulfinyl' (one =O) or 'sulfonyl' (two =O).

    Returns:
        The substitutive name, or None (caller keeps functional class).
    """
    from ..assembly.naming_utils import should_omit_locant_one

    sulfur_idx = None
    for idx in match_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'S':
            sulfur_idx = idx
            break
    if sulfur_idx is None:
        return None
    c_nbrs = [
        n.GetIdx() for n in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors()
        if n.GetSymbol() == 'C'
    ]
    if len(c_nbrs) != 2:
        return None

    sides = []
    for c in c_nbrs:
        s = _classify_oxide_side(mol, c, sulfur_idx)
        if s is None:
            return None
        sides.append(s)

    (stem_a, kind_a, atoms_a), (stem_b, kind_b, atoms_b) = sides

    # Symmetric diaryl -> multiplicative (P-63.6 form (3) is the PIN).
    if kind_a == 'ring' and kind_b == 'ring':
        if stem_a == stem_b == 'benzene':
            return f"1,1'-{oxide_kind}dibenzene"
        # identical/different non-benzene ring pairs: the general
        # multiplicative ring machinery lands in Tier 5a — fail through.
        return None

    # Ring + chain: ring is the senior parent (P-44.1.2.2).
    if kind_a == 'ring' or kind_b == 'ring':
        ring_stem = stem_a if kind_a == 'ring' else stem_b
        chain_stem = stem_b if kind_a == 'ring' else stem_a
        chain_atoms = atoms_b if kind_a == 'ring' else atoms_a
        # W3-P04 (P-65.3.2.2.2 / P-14.3.4): the acyl-from-sulfonic substituent is
        # located on the CHAIN carbon it derives from — 'propane-1-sulfonyl'
        # (BB @31396 '(propane-1-sulfonyl)benzene (PIN)'). The '-1-' is cited for
        # C3+ chains and omitted for methane/ethane (BB @302 '(ethanesulfonyl)
        # ethane', DMSO '(methanesulfinyl)benzene'), exactly like the two-chain
        # branch below.
        chain_n = len(chain_atoms)
        if should_omit_locant_one(
            context="prefix", chain_length=chain_n, is_monosubstituted=True,
        ):
            return f"({chain_stem}{oxide_kind}){ring_stem}"
        return f"({chain_stem}-1-{oxide_kind}){ring_stem}"

    # Two chains: the longer chain is the parent (P-44.3); tie -> either
    # (identical stems for the symmetric case).
    n_a, n_b = len(atoms_a), len(atoms_b)
    if n_a >= n_b:
        parent_stem, parent_n, sub_stem = stem_a, n_a, stem_b
    else:
        parent_stem, parent_n, sub_stem = stem_b, n_b, stem_a
    prefix = f"({sub_stem}{oxide_kind})"
    if should_omit_locant_one(
        context="prefix", chain_length=parent_n, is_monosubstituted=True,
    ):
        return f"{prefix}{parent_stem}"
    return f"1-{prefix}{parent_stem}"


def chalcogen_oxide_fc_covers_molecule(
    mol, match_atoms: Tuple[int, ...],
) -> bool:
    """Wave2 T3b conservation guard for the functional-class sulfoxide/
    sulfone namers: their name describes EXACTLY R-S(=O)x-R', so it is only
    honest when S + its =O oxygens + both full side fragments account for
    every heavy atom in the molecule. 'CSCCS(=O)C' used to emit 'ethyl
    methyl sulfoxide' — the alkyl walk stopped at the second S, silently
    dropping -S-CH3 (a different molecule). On False the handler declines
    and the polyfunctional path names the whole structure.
    """
    sulfur_idx = None
    for idx in match_atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'S':
            sulfur_idx = idx
            break
    if sulfur_idx is None:
        return False
    covered = {sulfur_idx}
    for nb in mol.GetAtomWithIdx(sulfur_idx).GetNeighbors():
        if nb.GetSymbol() == 'O' and nb.GetDegree() == 1:
            covered.add(nb.GetIdx())
            continue
        # full side BFS
        q = deque([nb.GetIdx()])
        while q:
            i = q.popleft()
            if i in covered or i == sulfur_idx:
                continue
            covered.add(i)
            for nb2 in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb2.GetIdx() not in covered and nb2.GetIdx() != sulfur_idx:
                    q.append(nb2.GetIdx())
    return len(covered) == mol.GetNumHeavyAtoms()


def _characterize_sulfur_substituent(mol, start_idx: int, exclude: set):
    """Characterize a substituent attached to sulfur as aryl or alkyl.

    Returns:
        Tuple of (name, carbon_count) where name is "phenyl"/"naphthyl" for aryl
        or the alkyl name string. Returns (None, 0) if uncharacterizable.
    """
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check for aryl groups (phenyl, naphthyl)
    if start_atom.GetIsAromatic() and start_atom.GetSymbol() == 'C':
        aromatic_atoms = set()
        aq = deque([start_idx])
        while aq:
            ai = aq.popleft()
            if ai in aromatic_atoms or ai in exclude:
                continue
            a = mol.GetAtomWithIdx(ai)
            if a.GetIsAromatic() and a.GetSymbol() == 'C':
                aromatic_atoms.add(ai)
                for nb in a.GetNeighbors():
                    ni = nb.GetIdx()
                    if ni not in aromatic_atoms and ni not in exclude:
                        aq.append(ni)
        ar_count = len(aromatic_atoms)
        if ar_count == 6:
            return ("phenyl", 6)
        elif ar_count == 10:
            return ("naphthyl", 10)
        return (None, 0)

    # Alkyl group — Wave2 T3b conservation: the old C-only BFS silently
    # flattened branched / ring / hetero-bearing sides into a linear alkyl
    # count ('CSCCS(=O)C' -> 'ethyl methyl sulfoxide', the -S-CH3 dropped;
    # a benzyl side became 'heptyl'). Route through the strict side
    # classifier: only a linear terminal saturated all-C chain earns an
    # alkyl name; everything else -> (None, 0) so the caller declines.
    chalcogen_idx = next(iter(exclude), None)
    if chalcogen_idx is None:
        return (None, 0)
    side = _classify_oxide_side(mol, start_idx, chalcogen_idx)
    if side is None or side[1] != 'chain':
        return (None, 0)
    count = len(side[2])
    if count == 0 or count > 10:
        return (None, 0)
    return (get_alkyl_name(count), count)


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """Count carbon atoms in an alkyl group via BFS (backward-compatible)."""
    _, count = _characterize_sulfur_substituent(mol, start_idx, exclude)
    return count


def get_sulfur_prefix(fg_name: str) -> Optional[str]:
    """
    Get prefix form for sulfur functional groups.

    Returns:
        Prefix string, or None if group uses functional class naming
    """
    SULFUR_PREFIXES = {
        "thiol": "sulfanyl",  # IUPAC P-63.6.1.1 (2013), not "mercapto"
        "sulfonic_acid": "sulfo",
        "sulfinic_acid": "sulfino",
        # These use functional class naming, no prefix:
        "thioether": None,
        "sulfide": None,
        "sulfoxide": None,
        "sulfone": None,
    }
    return SULFUR_PREFIXES.get(fg_name)
