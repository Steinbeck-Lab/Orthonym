"""
Ring-as-substituent naming according to IUPAC 2013 (Blue Book).

Implements IUPAC P-61.5: Standard substituent names for rings when they
become substituents on a chain parent structure.

Examples:
- benzene -> phenyl (4-phenylbutanoic acid)
- cyclohexane -> cyclohexyl (4-cyclohexylbutanoic acid)
- naphthalene -> naphthyl (position-specific: 1-naphthyl, 2-naphthyl)
- pyridine -> pyridyl (position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl)
"""

from typing import Dict, List, Optional, Set, Tuple

# Chain length prefixes - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix


# IUPAC P-61.5: Standard substituent names for rings
# Maps ring system name to substituent prefix name
RING_SUBSTITUENT_NAMES: Dict[str, str] = {
    # Carbocyclic aromatic
    'benzene': 'phenyl',
    'naphthalene': 'naphthyl',  # Position-specific: 1-naphthyl, 2-naphthyl
    'anthracene': 'anthryl',
    'phenanthrene': 'phenanthryl',
    'pyrene': 'pyrenyl',
    'fluorene': 'fluorenyl',
    'acenaphthylene': 'acenaphthylenyl',
    'chrysene': 'chrysenyl',
    'fluoranthene': 'fluoranthenyl',

    # Carbocyclic saturated (cycloalkanes)
    'cyclopropane': 'cyclopropyl',
    'cyclobutane': 'cyclobutyl',
    'cyclopentane': 'cyclopentyl',
    'cyclohexane': 'cyclohexyl',
    'cycloheptane': 'cycloheptyl',
    'cyclooctane': 'cyclooctyl',

    # Heterocyclic aromatic
    'pyridine': 'pyridyl',  # Position-specific: 2-pyridyl, 3-pyridyl, 4-pyridyl
    'furan': 'furyl',
    'thiophene': 'thienyl',
    'pyrrole': 'pyrrolyl',
    'imidazole': 'imidazolyl',
    'pyrimidine': 'pyrimidinyl',
    'pyrazine': 'pyrazinyl',
    'pyridazine': 'pyridazinyl',

    # Heterocyclic saturated (6-membered)
    'oxane': 'oxanyl',
    'piperidine': 'piperidinyl',
    'morpholine': 'morpholinyl',
    'piperazine': 'piperazinyl',

    # Heterocyclic saturated (5-membered)
    'oxolane': 'oxolanyl',
    'pyrrolidine': 'pyrrolidinyl',

    # Heterocyclic saturated (4-membered)
    'oxetane': 'oxetanyl',
    'azetidine': 'azetidinyl',

    # Heterocyclic saturated (3-membered)
    'oxirane': 'oxiranyl',
    'aziridine': 'aziridinyl',

    # Fused heterocyclic (Phase 139 ARCH-02)
    'indole': 'indolyl',
    'quinoline': 'quinolinyl',
    'isoquinoline': 'isoquinolinyl',
    'benzofuran': 'benzofuranyl',
    'benzothiophene': 'benzothienyl',
    'benzimidazole': 'benzimidazolyl',
    'purine': 'purinyl',
    'carbazole': 'carbazolyl',
}

# Rings that need position-specific names based on attachment point
# Maps ring name -> {ring_position: substituent_name}
POSITION_SPECIFIC_RINGS: Dict[str, Dict[int, str]] = {
    'naphthalene': {
        1: '1-naphthyl',
        2: '2-naphthyl',
    },
    'pyridine': {
        2: '2-pyridyl',
        3: '3-pyridyl',
        4: '4-pyridyl',
    },
    'quinoline': {
        2: '2-quinolinyl',
        3: '3-quinolinyl',
        4: '4-quinolinyl',
        5: '5-quinolinyl',
        6: '6-quinolinyl',
        7: '7-quinolinyl',
        8: '8-quinolinyl',
    },
    'isoquinoline': {
        1: '1-isoquinolinyl',
        3: '3-isoquinolinyl',
        4: '4-isoquinolinyl',
        5: '5-isoquinolinyl',
        6: '6-isoquinolinyl',
        7: '7-isoquinolinyl',
        8: '8-isoquinolinyl',
    },
}


def identify_ring_system(mol, ring_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Identify ring system name from molecular structure.

    Examines the ring atoms to determine what type of ring system it is
    based on size, aromaticity, and heteroatom composition.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring

    Returns:
        Ring system name string (e.g., 'benzene', 'cyclohexane', 'pyridine'),
        or None if the ring cannot be identified
    """
    ring_size = len(ring_atoms)
    ring_set = set(ring_atoms)

    # --- Single-ring guard (Phase 4 SUBST-01) -------------------------------
    # identify_ring_system describes ONE ring by its size + heteroatom set. If
    # the atom set spans more than one ring (a fused/bridged/spiro polycyclic
    # system) the size-based branches below would mislabel it as a monocycle of
    # that atom count (norbornane[7]->'cycloheptane', decalin/tetralin[10]->
    # 'cyclodecane') — a DIFFERENT molecule. Decline so the caller routes the
    # polycyclic fragment to the dedicated von-Baeyer / spiro / partial-hydro
    # namers instead of guessing a monocycle.
    _atom_rings = mol.GetRingInfo().AtomRings()
    _contained = [r for r in _atom_rings if set(r) <= ring_set]
    if len(_contained) != 1:
        return None

    # Check aromaticity of all ring atoms
    is_aromatic = all(
        mol.GetAtomWithIdx(idx).GetIsAromatic()
        for idx in ring_atoms
    )

    # Get heteroatom list (symbols of non-carbon atoms)
    heteroatoms = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            heteroatoms.append(symbol)

    # Sort for consistent comparison
    heteroatoms.sort()

    # === 6-membered rings ===
    if ring_size == 6:
        if is_aromatic:
            if not heteroatoms:
                return 'benzene'
            elif heteroatoms == ['N']:
                return 'pyridine'
            elif heteroatoms == ['N', 'N']:
                # Distinguish pyridazine (1,2), pyrimidine (1,3), pyrazine (1,4)
                # by checking the topological relationship between N atoms
                n_indices = [idx for idx in ring_atoms
                             if mol.GetAtomWithIdx(idx).GetSymbol() == 'N']
                if len(n_indices) == 2:
                    n1, n2 = n_indices
                    # Check if N atoms are directly bonded (pyridazine = 1,2-diazine)
                    n1_nbrs = {nbr.GetIdx() for nbr in mol.GetAtomWithIdx(n1).GetNeighbors()}
                    if n2 in n1_nbrs:
                        return 'pyridazine'
                    # Check shortest path between N atoms in the ring
                    # Pyrimidine (1,3): 1 C between N's on shorter side
                    # Pyrazine (1,4): 2 C between N's on both sides (symmetric)
                    ring_list = list(ring_atoms)
                    pos1 = ring_list.index(n1)
                    pos2 = ring_list.index(n2)
                    sep = abs(pos1 - pos2)
                    min_sep = min(sep, ring_size - sep)
                    if min_sep == 3:  # para = pyrazine
                        return 'pyrazine'
                    else:  # min_sep == 2 = meta = pyrimidine
                        return 'pyrimidine'
                return 'pyrimidine'  # fallback
            elif heteroatoms == ['N', 'N', 'N']:
                # Distinguish triazine isomers by counting adjacent N-N bonds:
                #   0 adjacent pairs -> 1,3,5-triazine (sym-triazine)
                #   1 adjacent pair  -> 1,2,4-triazine (as-triazine PIN)
                #   2 adjacent pairs -> 1,2,3-triazine (v-triazine)
                n_indices = [idx for idx in ring_atoms
                             if mol.GetAtomWithIdx(idx).GetSymbol() == 'N']
                adj_count = sum(
                    1 for i in range(len(n_indices))
                    for j in range(i + 1, len(n_indices))
                    if mol.GetBondBetweenAtoms(n_indices[i], n_indices[j]) is not None
                )
                if adj_count == 0:
                    return '1,3,5-triazine'
                elif adj_count == 1:
                    return '1,2,4-triazine'
                else:
                    return '1,2,3-triazine'
        else:
            # Saturated 6-membered
            if not heteroatoms:
                return 'cyclohexane'
            elif heteroatoms == ['O']:
                return 'oxane'
            elif heteroatoms == ['S']:
                return 'thiane'
            elif heteroatoms == ['N']:
                return 'piperidine'
            elif heteroatoms == ['N', 'O']:
                return 'morpholine'
            elif heteroatoms == ['N', 'N']:
                return 'piperazine'

    # === 5-membered rings ===
    elif ring_size == 5:
        if is_aromatic:
            if not heteroatoms:
                # Aromatic 5-membered all carbon = cyclopentadienyl anion
                # Usually not encountered, but handle gracefully
                return 'cyclopentadiene'
            elif heteroatoms == ['O']:
                return 'furan'
            elif heteroatoms == ['S']:
                return 'thiophene'
            elif heteroatoms == ['N']:
                return 'pyrrole'
            elif heteroatoms == ['N', 'N']:
                return 'imidazole'
        else:
            # Saturated 5-membered
            if not heteroatoms:
                return 'cyclopentane'
            elif heteroatoms == ['O']:
                return 'oxolane'
            elif heteroatoms == ['S']:
                return 'thiolane'
            elif heteroatoms == ['N']:
                return 'pyrrolidine'

    # === 3-membered rings ===
    elif ring_size == 3:
        if not heteroatoms:
            return 'cyclopropane'
        elif heteroatoms == ['O']:
            return 'oxirane'
        elif heteroatoms == ['N']:
            return 'aziridine'

    # === 4-membered rings ===
    elif ring_size == 4:
        if not heteroatoms:
            return 'cyclobutane'
        elif heteroatoms == ['O']:
            return 'oxetane'
        elif heteroatoms == ['N']:
            return 'azetidine'

    # === 7-membered rings ===
    elif ring_size == 7:
        if not heteroatoms:
            if is_aromatic:
                return 'cycloheptatriene'
            else:
                return 'cycloheptane'

    # === 8-membered and larger ===
    elif ring_size == 8 and not heteroatoms:
        return 'cyclooctane'

    # Fallback: generic cycloalkane for all-carbon saturated rings
    if not heteroatoms and not is_aromatic:
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}ane'

    return None


# IUPAC P-31.1.4.3.4 / P-29.3.5: PIN substituent stems for monocyclic
# heteroarenes (ring name with the trailing 'e' removed). Only these rings are
# claimed by pin_heteroaryl_substituent_name(); anything else guards to None.
_PIN_HETEROARYL_STEMS: Dict[str, str] = {
    'pyridine': 'pyridin',
    'pyrimidine': 'pyrimidin',
    'pyrazine': 'pyrazin',
    'pyridazine': 'pyridazin',
    'furan': 'furan',
    'thiophene': 'thiophen',
    'pyrrole': 'pyrrol',
    'imidazole': 'imidazol',
    # WS-A task 9: fully-SATURATED retained monocycles take the same
    # free-valence numbering (P-29.2): pyrrolidin-1-yl, morpholin-4-yl,
    # piperidin-1-yl, piperazin-1-yl.
    'pyrrolidine': 'pyrrolidin',
    'morpholine': 'morpholin',
    'piperidine': 'piperidin',
    'piperazine': 'piperazin',
    # Phase 4 SUBST-01 (d): saturated O/S monocycles take the same free-valence
    # numbering (heteroatom = locant 1): oxan-2-yl, oxolan-3-yl, thian-2-yl,
    # thiolan-3-yl. Extends the path beyond the N-rings above.
    'oxane': 'oxan',
    'oxolane': 'oxolan',
    'thiane': 'thian',
    'thiolane': 'thiolan',
    # P-31.1.4.2: triazine PIN stems
    '1,2,4-triazine': '1,2,4-triazin',
    '1,2,3-triazine': '1,2,3-triazin',
    '1,3,5-triazine': '1,3,5-triazin',
}

# Element seniority for assigning low locants to heteroatoms (IUPAC Table 28:
# F > Cl > Br > I > O > S > Se > Te > N > P > ...). Lower value = senior.
#
# v22 Phase E1 / DD4: DERIVED from the single source of truth
# ``locants.ELEMENT_NUMBERING_SENIORITY`` (the full P-15.4.1.2 order) instead of
# a hand-maintained 4th copy. Densely re-ranking that order restricted to the
# elements that appear as ring heteroatoms reproduces the original
# {F:0,...,B:17} mapping byte-identically (At/Po/C and the Al..Tl tail are not
# ring-heteroatom-relevant here), so behaviour is unchanged — the divergence is
# eliminated for the numbering consumer. A unit test pins the equality.
from .locants import ELEMENT_NUMBERING_SENIORITY as _ELEMENT_NUMBERING_SENIORITY

_HETEROATOM_SENIORITY: Dict[str, int] = {
    sym: rank
    for rank, sym in enumerate(
        sorted(
            ('F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te',
             'N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B'),
            key=lambda x: _ELEMENT_NUMBERING_SENIORITY[x],
        )
    )
}


def pin_heteroaryl_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
) -> Optional[str]:
    """PIN substituent name for a monocyclic heteroaromatic ring substituent.

    Computes free-valence-aware numbering per IUPAC 2013 P-31.1.4.3.4: when a
    ring is detached as a substituent it is renumbered so low locants go, in
    order of decreasing priority, to (1) the heteroatoms as a set, (2) the
    heteroatoms in element-seniority order, (3) the indicated hydrogen, and
    (4) the free valence (point of attachment). The result is
    ``{nH-}{stem}-{locant}-yl``.

    Example: ``c1cnc[nH]1`` attached at the carbon next to the NH numbers as
    ``1H-imidazol-5-yl`` — the indicated H at locant 1 outranks the lower
    free-valence locant that the alternative ``3H-imidazol-4-yl`` numbering
    would give.

    Deliberately *guarded*: returns ``None`` (so the caller keeps its existing
    locant-less form, guaranteeing zero regression) whenever the PIN locant is
    not provably correct —
      * the ring is not one of the supported monocyclic heteroarenes;
      * it is a pyrazole (which identify_ring_system() reports as 'imidazole');
      * the ring carries any substituent other than the single attachment;
      * the ring is not a simple aromatic monocycle;
      * ``attachment_atom`` is not a ring atom.

    Args:
        mol: RDKit Mol of the whole molecule.
        ring_atoms: Atom indices forming the candidate ring.
        attachment_atom: Ring atom index bearing the free valence (bonded to
            the parent structure).

    Returns:
        The PIN substituent name, or None when not provably PIN-correct.
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)
    ring_set = set(ring_list)

    # --- Guard: attachment must be a ring atom -------------------------------
    if attachment_atom not in ring_set:
        return None

    # --- Guard: supported, correctly-identified heteroarene ------------------
    ring_name = identify_ring_system(mol, tuple(ring_list))
    stem = _PIN_HETEROARYL_STEMS.get(ring_name)
    if stem is None:
        return None

    # --- Guard: aromatic OR fully saturated simple monocycle -----------------
    # WS-A task 9: fully-saturated heterocyclic monocycles (pyrrolidine,
    # morpholine, piperidine, piperazine) use the SAME free-valence numbering
    # cascade — minus indicated hydrogen, an aromatic-only concept. Mixed
    # saturation stays guarded out (None -> legacy form).
    _is_aromatic_ring = all(
        mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list
    )
    if not _is_aromatic_ring:
        from rdkit import Chem as _Chem
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list):
            return None
        for i in ring_list:
            for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
                if nbr.GetIdx() in ring_set:
                    bond = mol.GetBondBetweenAtoms(i, nbr.GetIdx())
                    if bond.GetBondType() != _Chem.BondType.SINGLE:
                        return None
    ring_adj: Dict[int, List[int]] = {}
    for idx in ring_list:
        nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None  # fusion atom / spiro / not a simple ring
        ring_adj[idx] = nbrs

    het_atoms = [i for i in ring_list
                 if mol.GetAtomWithIdx(i).GetSymbol() != 'C']

    # --- Guard: imidazole vs pyrazole ----------------------------------------
    # identify_ring_system() reports BOTH 5-membered N,N arenes as 'imidazole'.
    # Genuine imidazole has its two ring N's NON-adjacent (1,3); pyrazole has
    # them adjacent (1,2). Refuse the adjacent-N case so we never emit an
    # imidazol-*-yl name for a pyrazole.
    if ring_name == 'imidazole':
        n_idx = [i for i in het_atoms
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) != 2 or mol.GetBondBetweenAtoms(n_idx[0], n_idx[1]) is not None:
            return None

    # --- Guard: no ring substituent other than the single attachment ---------
    for idx in ring_list:
        heavy_exo = sum(
            1 for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        )
        if idx == attachment_atom:
            if heavy_exo < 1:
                return None  # no parent bond at the claimed attachment
        elif heavy_exo:
            return None  # an extra ring substituent we do not number

    # --- Indicated-hydrogen atom (the ring NH, if any; AROMATIC only) --------
    if _is_aromatic_ring:
        indicated_h_atoms = [
            i for i in het_atoms
            if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1
        ]
        if len(indicated_h_atoms) > 1:
            return None  # ambiguous indicated H; do not guess
        indicated_h_atom = indicated_h_atoms[0] if indicated_h_atoms else None
    else:
        indicated_h_atom = None  # saturated rings have no indicated H

    # --- Build the cyclic atom order -----------------------------------------
    order = [ring_list[0], ring_adj[ring_list[0]][0]]
    while len(order) < n:
        prev, curr = order[-2], order[-1]
        nxt = [x for x in ring_adj[curr] if x != prev]
        if not nxt:
            return None
        order.append(nxt[0])
    if len(order) != n:
        return None

    # --- Enumerate all 2n numberings; take the lexicographic-min key ---------
    best_key = None
    best = None  # (free_valence_locant, indicated_h_locant)
    for start in range(n):
        for direction in (1, -1):
            atom_to_pos = {
                order[(start + direction * p) % n]: p + 1
                for p in range(n)
            }
            het_locs = tuple(sorted(atom_to_pos[i] for i in het_atoms))
            seniority_locs = tuple(
                atom_to_pos[i] for i in sorted(
                    het_atoms,
                    key=lambda a: (
                        _HETEROATOM_SENIORITY.get(
                            mol.GetAtomWithIdx(a).GetSymbol(), 99),
                        atom_to_pos[a],
                    ),
                )
            )
            ih_loc = (atom_to_pos[indicated_h_atom]
                      if indicated_h_atom is not None else 0)
            fv_loc = atom_to_pos[attachment_atom]

            key = (het_locs, seniority_locs, ih_loc, fv_loc)
            if best_key is None or key < best_key:
                best_key = key
                best = (fv_loc, ih_loc)

    if best is None:
        return None
    attach_locant, ih_locant = best
    prefix = f"{ih_locant}H-" if indicated_h_atom is not None else ""
    return f"{prefix}{stem}-{attach_locant}-yl"


# Multiplier stems for skeletal-unsaturation infixes (di/tri/tetra...).
_UNSAT_MULT: Dict[int, str] = {
    2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa', 7: 'hepta', 8: 'octa',
}


def _fmt_locant(loc) -> str:
    """Render a numbering value (int or lettered fusion locant like '4a')."""
    return str(loc)


def _build_ene_yne_infix(stem: str, ene: List[int], yne: List[int]) -> Optional[str]:
    """Assemble ``<stem>[a]-<locs>-[mult]en[-<locs>-[mult]yn]`` WITHOUT the
    trailing 'e', so the caller appends ``-<loc>-yl``.

    ``cyclohex`` + ene=[1]      -> ``cyclohex-1-en``
    ``cyclohex`` + ene=[1,3]    -> ``cyclohexa-1,3-dien``   (euphonic 'a' before di)
    Per IUPAC 2013 P-31.1.4: the parent-hydride stem keeps its 'a' before a
    multiplied unsaturation suffix (hexa-1,3-diene) but elides it before a single
    'ene' (hex-1-ene).
    """
    if not ene and not yne:
        return None
    has_mult = len(ene) >= 2 or len(yne) >= 2
    out = stem + ('a' if has_mult else '')
    if ene:
        m = _UNSAT_MULT.get(len(ene), '') if len(ene) > 1 else ''
        out += '-' + ','.join(str(x) for x in ene) + '-' + m + 'en'
    if yne:
        m = _UNSAT_MULT.get(len(yne), '') if len(yne) > 1 else ''
        out += '-' + ','.join(str(x) for x in yne) + '-' + m + 'yn'
    return out


def _unsaturated_carbocyclic_substituent(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: int
) -> Optional[str]:
    """P-31.1.4.3.4 substituent name for a monocyclic, all-carbon, non-aromatic
    ring carrying >=1 skeletal multiple bond and no other decoration:
    ``cyclohex-1-en-1-yl``, ``cyclohex-2-en-1-yl``, ``cyclohexa-2,4-dien-1-yl``.

    The free valence takes locant 1 (lowest for a carbocyclic monocyclic
    substituent, P-29.2); the ring is numbered in the direction giving the lowest
    locants to the skeletal multiple bonds. Returns None (fail-closed) for
    anything outside this narrow class — heteroatoms, charge, fused/spiro atoms,
    saturated rings, or any extra exocyclic decoration.
    """
    from .cycloalkanes import ring_double_bond_locant
    ring_set = set(ring_atoms)
    n = len(ring_set)
    if attachment_atom not in ring_set or n < 3:
        return None
    ri = mol.GetRingInfo()
    adj: Dict[int, List[int]] = {}
    for i in ring_atoms:
        a = mol.GetAtomWithIdx(i)
        if (a.GetSymbol() != 'C' or a.GetIsAromatic()
                or a.GetFormalCharge() != 0):
            return None
        if ri.NumAtomRings(i) != 1:
            return None  # fused / spiro atom -> not a simple monocycle
        nbrs = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None
        adj[i] = nbrs
    # No exocyclic heavy decoration except the single parent bond.
    for i in ring_atoms:
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            if nbr.GetIdx() in ring_set or nbr.GetAtomicNum() <= 1:
                continue
            if i == attachment_atom:
                continue
            return None
    multibonds: List[Tuple[int, int, int]] = []
    for i in ring_atoms:
        for j in adj[i]:
            if i < j:
                order = mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()
                if order >= 2.0:
                    multibonds.append((i, j, int(order)))
    if not multibonds:
        return None  # saturated -> not this path
    order0 = [attachment_atom, adj[attachment_atom][0]]
    while len(order0) < n:
        prev, cur = order0[-2], order0[-1]
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            return None
        order0.append(nxt[0])
    if len(order0) != n:
        return None
    candidates = [order0, [order0[0]] + order0[:0:-1]]
    best: Optional[Tuple[list, List[int], List[int]]] = None
    for seq in candidates:
        pos = {atom: idx + 1 for idx, atom in enumerate(seq)}
        ene: List[int] = []
        yne: List[int] = []
        for a, b, o in multibonds:
            loc = ring_double_bond_locant(pos[a] - 1, pos[b] - 1, n)
            (ene if o == 2 else yne).append(loc)
        key = sorted(ene) + sorted(yne)
        if best is None or key < best[0]:
            best = (key, sorted(ene), sorted(yne))
    _, ene, yne = best
    stem = 'cyclo' + _get_chain_prefix(n)
    infix = _build_ene_yne_infix(stem, ene, yne)
    if infix is None:
        return None
    return f'{infix}-1-yl'


# Wave2 T6b: hydro-prefix multipliers for the heteromonocyclic substituent
# emitter (hydro counts are always even — each hydrogenated mancude double
# bond contributes two positions).
_HYDRO_MULTIPLIERS = {2: 'di', 4: 'tetra', 6: 'hexa', 8: 'octa', 10: 'deca'}


def _unsaturated_heteromonocyclic_substituent(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: int
) -> Optional[str]:
    """Wave2 T6b (P-32.2.1): substituent name for a NON-AROMATIC monocyclic
    ring with exactly ONE heteroatom carrying >=1 skeletal double bond, named
    on its mancude parent with hydro prefixes and indicated hydrogen:
    ``3,4-dihydro-2H-pyran-2-yl``, ``2,3-dihydrofuran-...-yl``.

    Locant selection follows P-32.2.1: heteroatom lowest (fixed = 1), then
    indicated hydrogen, then the free-valence suffix, and finally 'hydro'
    prefixes. The mancude double-bond patterns are enumerated as maximum
    matchings of the ring cycle (O/S/Se/Te excluded from double bonds); the
    actual double-bond set must be a subset of the chosen pattern, the
    hydrogenated pattern bonds become the hydro positions, and the unmatched
    atom (if any) is the indicated hydrogen. Returns None (fail-closed) for
    anything outside this narrow class — multiple heteroatoms, aromatic or
    fused rings, triple bonds, exocyclic decoration, attachment on the
    heteroatom (would need added indicated hydrogen), or an unnameable
    mancude parent.
    """
    from itertools import combinations

    ring_set = set(ring_atoms)
    n = len(ring_set)
    if attachment_atom not in ring_set or n < 3 or n > 10:
        return None
    ri = mol.GetRingInfo()

    het_idx = None
    adj: Dict[int, List[int]] = {}
    for i in ring_atoms:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic() or a.GetFormalCharge() != 0:
            return None
        if ri.NumAtomRings(i) != 1:
            return None  # fused / spiro atom -> not a simple monocycle
        sym = a.GetSymbol()
        if sym != 'C':
            if sym not in ('N', 'O', 'S', 'Se', 'Te') or het_idx is not None:
                return None  # unsupported element or >1 heteroatom
            het_idx = i
        nbrs = [x.GetIdx() for x in a.GetNeighbors() if x.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None
        adj[i] = nbrs
    if het_idx is None:
        return None  # carbocycle -> _unsaturated_carbocyclic_substituent
    if attachment_atom == het_idx:
        return None  # N-attachment needs added indicated hydrogen -> decline

    # No exocyclic heavy decoration; the single parent bond at the attachment
    # atom must be a single bond (ylidene -> decline).
    for i in ring_atoms:
        atom_i = mol.GetAtomWithIdx(i)
        for nbr in atom_i.GetNeighbors():
            j = nbr.GetIdx()
            if j in ring_set or nbr.GetAtomicNum() <= 1:
                continue
            if i != attachment_atom:
                return None
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond.GetBondTypeAsDouble() != 1.0:
                return None

    # Ring bonds: actual double bonds (A) and double-bond-eligible bonds.
    ring_bonds: List[Tuple[int, int]] = []
    actual_dbl: Set[Tuple[int, int]] = set()
    db_eligible: List[Tuple[int, int]] = []
    for i in ring_atoms:
        for j in adj[i]:
            if i >= j:
                continue
            order = mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()
            if order not in (1.0, 2.0):
                return None  # triple / aromatic / kekulization oddity
            pair = (i, j)
            ring_bonds.append(pair)
            if order == 2.0:
                actual_dbl.add(pair)
            if all(
                mol.GetAtomWithIdx(k).GetSymbol() in ('C', 'N') for k in pair
            ):
                db_eligible.append(pair)
    if not actual_dbl:
        return None  # saturated -> the ordinary saturated stems apply
    if not actual_dbl <= set(db_eligible):
        return None  # double bond on a divalent chalcogen — not mancude-based

    # Maximum matchings of the ring cycle over the eligible bonds = the
    # candidate mancude double-bond patterns.
    max_size = 0
    matchings: List[Set[Tuple[int, int]]] = []
    for size in range(len(db_eligible) // 2 + 1, 0, -1):
        for combo in combinations(db_eligible, size):
            atoms_seen: Set[int] = set()
            ok = True
            for x, y in combo:
                if x in atoms_seen or y in atoms_seen:
                    ok = False
                    break
                atoms_seen.add(x)
                atoms_seen.add(y)
            if ok:
                matchings.append(set(combo))
        if matchings:
            max_size = size
            break
    if not matchings or max_size == 0:
        return None
    candidates_m = [m for m in matchings if actual_dbl <= m]
    if not candidates_m:
        return None  # actual pattern incompatible with any mancude parent

    # Two numbering directions, heteroatom fixed at locant 1.
    def _ring_order(start_nbr: int) -> Optional[List[int]]:
        seq = [het_idx, start_nbr]
        while len(seq) < n:
            prev, cur = seq[-2], seq[-1]
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt:
                return None
            seq.append(nxt[0])
        return seq

    best = None  # (ih_locs, yl_loc, hydro_locs, name_parts)
    for start in adj[het_idx]:
        seq = _ring_order(start)
        if seq is None or len(seq) != n:
            continue
        pos = {atom: idx + 1 for idx, atom in enumerate(seq)}
        for m in candidates_m:
            matched_atoms = {k for pair in m for k in pair}
            # Indicated hydrogen sites are unmatched sp3 C/N skeletal atoms of
            # the mancude parent; an unmatched divalent chalcogen (O/S/Se/Te)
            # bears no H and is NOT an indicated-H position (furan/pyran O).
            ih_locs = sorted(
                pos[a] for a in ring_set
                if a not in matched_atoms
                and mol.GetAtomWithIdx(a).GetSymbol() in ('C', 'N')
            )
            if len(ih_locs) > 1:
                continue  # multi-indicated-H parent — out of scope
            hydro_locs = sorted(
                pos[k] for pair in (m - actual_dbl) for k in pair
            )
            yl_loc = pos[attachment_atom]
            key = (ih_locs, yl_loc, hydro_locs)
            if best is None or key < best[0]:
                best = (key, ih_locs, yl_loc, hydro_locs)
    if best is None:
        return None
    _, ih_locs, yl_loc, hydro_locs = best

    # Mancude parent stem via the heterocycle parent namer on a detached,
    # mancude-ized copy of the ring (any maximum matching gives the same
    # stem; the indicated-H token, if emitted, is stripped — the substituent
    # form re-derives it from the selected numbering above).
    stem = _mancude_monocycle_stem(mol, ring_atoms, candidates_m[0])
    if stem is None:
        return None

    hydro_part = ''
    if hydro_locs:
        mult = _HYDRO_MULTIPLIERS.get(len(hydro_locs))
        if mult is None:
            return None
        hydro_part = f"{','.join(str(l) for l in hydro_locs)}-{mult}hydro"
    ih_part = f"{ih_locs[0]}H-" if ih_locs else ''
    if hydro_part and ih_part:
        core = f"{hydro_part}-{ih_part}{stem}"
    elif hydro_part:
        core = f"{hydro_part}{stem}"
    else:
        core = f"{ih_part}{stem}"
    if stem.endswith('e'):
        core = core[:-1]
    return f"{core}-{yl_loc}-yl"


def _mancude_monocycle_stem(
    mol, ring_atoms: Tuple[int, ...], matching: Set[Tuple[int, int]]
) -> Optional[str]:
    """Name the mancude parent of a detached heteromonocycle (double bonds
    set to ``matching``) via the heterocycle parent namer and return the bare
    stem with any leading indicated-H token stripped ('2H-pyran' -> 'pyran').
    Fail-closed: None unless the result is a single plain word."""
    import re
    from rdkit import Chem
    keep = set(ring_atoms)
    try:
        rw = Chem.RWMol(mol)
        for idx in sorted((a.GetIdx() for a in mol.GetAtoms()
                           if a.GetIdx() not in keep), reverse=True):
            rw.RemoveAtom(idx)
        sub = rw.GetMol()
        # Re-map original indices -> submol indices (removal preserves the
        # relative order of the kept atoms).
        old_order = sorted(keep)
        old_to_new = {old: new for new, old in enumerate(old_order)}
        for bond in sub.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        for x, y in matching:
            b = sub.GetBondBetweenAtoms(old_to_new[x], old_to_new[y])
            if b is None:
                return None
            b.SetBondType(Chem.BondType.DOUBLE)
        for a in sub.GetAtoms():
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    try:
        from .heterocycles import name_heterocycle
        raw = name_heterocycle(sub, tuple(range(sub.GetNumAtoms())))
    except Exception:
        return None
    if not raw or not isinstance(raw, str):
        return None
    stem = re.sub(r'^\d+H-', '', raw)
    if not re.fullmatch(r'[a-z]+', stem):
        return None  # locants / hyphens survived — not a bare mancude stem
    return stem


def _extract_ring_submol(mol, ring_atoms, attachment_atom):
    """Return ``(submol, attach_idx_in_submol)`` for the ring system as a
    standalone molecule, the broken parent bond at the attachment atom left as an
    implicit hydrogen (so the fragment is a valid neutral ring the parent namers
    can perceive). Returns ``(None, None)`` on failure."""
    from rdkit import Chem
    keep = set(ring_atoms)
    if attachment_atom not in keep:
        return None, None
    try:
        rw = Chem.RWMol(mol)
        rw.GetAtomWithIdx(attachment_atom).SetAtomMapNum(990017)
        for idx in sorted((a.GetIdx() for a in mol.GetAtoms()
                           if a.GetIdx() not in keep), reverse=True):
            rw.RemoveAtom(idx)
        sub = rw.GetMol()
        Chem.SanitizeMol(sub)
    except Exception:
        return None, None
    attach_sub = None
    for a in sub.GetAtoms():
        if a.GetAtomMapNum() == 990017:
            attach_sub = a.GetIdx()
            a.SetAtomMapNum(0)
            break
    if attach_sub is None:
        return None, None
    return sub, attach_sub


def _vonbaeyer_substituent_name(sub, attach_sub) -> Optional[str]:
    """``bicyclo[2.2.1]heptan-2-yl`` etc. for a detached carbocyclic von-Baeyer
    ring system, reusing the bicyclo parent namer (P-23) with the free valence
    treated as the lowest-locant feature (P-31.1.4.3.4). A ring double bond is
    cited with the ``-<loc>-en-`` infix (P-32.1.3: free valence gets the lowest
    locant first, THEN the unsaturation) -> ``bicyclo[2.2.2]oct-5-en-2-yl``.
    Carbocyclic only; heteroatom von-Baeyer stems are out of this phase's
    scope -> None."""
    from rdkit import Chem
    from . import bicyclo
    try:
        if not bicyclo.is_bicyclo_system(sub):
            return None
        if any(a.GetSymbol() != 'C' for a in sub.GetAtoms()):
            return None
        desc = bicyclo.generate_bicyclo_descriptor(sub)
        if not desc:
            return None
        numbering = bicyclo.get_bicyclo_numbering(
            sub, suffix_ring_atoms={attach_sub})
        if not numbering or attach_sub not in numbering:
            return None
        loc = numbering[attach_sub]
        stem = bicyclo._get_alkane_name(sub.GetNumAtoms())  # 'heptane'

        # Ring double bonds (skeletal, non-aromatic). Cite each as an 'ene' at
        # the lower of its two atom locants (P-31.1.4). Fail closed if a ring
        # double-bond atom is not in the numbering (should not happen).
        ene: List[int] = []
        for b in sub.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE or b.GetIsAromatic():
                continue
            a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a1 not in numbering or a2 not in numbering:
                return None
            ene.append(min(numbering[a1], numbering[a2]))
        ene.sort()

        if ene:
            # stem 'octane' -> 'oct' base; _build_ene_yne_infix appends the
            # ene suffix WITHOUT the trailing 'e', then '-<loc>-yl'.
            base = stem[:-3] if stem.endswith('ane') else (
                stem[:-1] if stem.endswith('e') else stem)
            infix = _build_ene_yne_infix(base, ene, [])
            if infix is None:
                return None
            return f'{desc}{infix}-{loc}-yl'  # bicyclo[2.2.2]oct-5-en-2-yl

        parent = desc + stem  # 'bicyclo[2.2.1]heptane'
        if parent.endswith('e'):
            parent = parent[:-1]
        return f'{parent}-{loc}-yl'
    except Exception:
        return None


def _spiro_substituent_name(sub, attach_sub) -> Optional[str]:
    """``spiro[4.5]decan-<loc>-yl`` for a detached carbocyclic monospiro ring
    system, reusing the spiro parent namer (P-24). Carbocyclic only -> None
    otherwise."""
    from . import spiro
    try:
        if not spiro.is_spiro_system(sub):
            return None
        if any(a.GetSymbol() != 'C' for a in sub.GetAtoms()):
            return None
        desc = spiro.generate_spiro_descriptor(sub)
        if not desc:
            return None
        ri = sub.GetRingInfo()
        centers = [a.GetIdx() for a in sub.GetAtoms()
                   if ri.NumAtomRings(a.GetIdx()) >= 2]
        if len(centers) != 1:
            return None  # only monospiro in scope
        numbering = spiro.get_spiro_numbering(
            sub, centers[0], suffix_ring_atoms={attach_sub})
        if not numbering or attach_sub not in numbering:
            return None
        loc = numbering[attach_sub]
        stem = _get_chain_prefix(sub.GetNumAtoms()) + 'ane'  # 'decane'
        parent = desc + stem
        if parent.endswith('e'):
            parent = parent[:-1]
        return f'{parent}-{loc}-yl'
    except Exception:
        return None


def _fused_hydro_substituent_name(sub, attach_sub) -> Optional[str]:
    """``5,6,7,8-tetrahydronaphthalen-1-yl`` etc. for a detached partially
    saturated fused carbocycle, reusing the partial-saturation parent perception
    but numbered with the FREE VALENCE first (P-31.1.4.3.4): low locant to the
    attachment, then to the hydro positions, then to any residual ring double
    bond. Also names the FULLY-saturated (perhydro) case with the count-form
    prefix (``decahydronaphthalen-2-yl``; D-FOLLOWON item 3). Returns None
    (fail-closed) for anything the mancude parent skeleton cannot number."""
    from rdkit import Chem
    from . import partial_saturation as ps
    from .partial_saturation import _locant_key
    from ..data.polycyclic_data import POLYCYCLIC_DATA
    try:
        ring_atoms = set(range(sub.GetNumAtoms()))
        det = ps.detect_carbocyclic_partial_saturation(sub, ring_atoms)
        if not det:
            return None
        is_perhydro = bool(det.get('is_perhydro'))
        parent = det['parent_name']
        prefix = det['prefix']
        sp3 = set(det['saturated_indices'])
        entry = POLYCYCLIC_DATA.get(parent)
        numbering = entry.get('iupac_numbering') if entry else None
        csmiles = entry.get('canonical_smiles') if entry else None
        if not numbering or not csmiles:
            return None
        cmol = Chem.MolFromSmiles(csmiles)
        if cmol is None:
            return None
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        params.adjustDegree = False
        query = Chem.AdjustQueryProperties(cmol, params)
        ring_db: List[Tuple[int, int]] = []
        for b in sub.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic():
                a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if a1 in ring_atoms and a2 in ring_atoms:
                    ring_db.append((a1, a2))
        best = None
        for match in sub.GetSubstructMatches(query, uniquify=False):
            a2l = {match[c]: loc for c, loc in numbering.items() if c < len(match)}
            if set(a2l) != ring_atoms or attach_sub not in a2l:
                continue
            attach_loc = _locant_key(a2l[attach_sub])
            hydro = sorted(_locant_key(a2l[a]) for a in sp3)
            ene = sorted(min(_locant_key(a2l[i]), _locant_key(a2l[j]))
                         for i, j in ring_db)
            key = (attach_loc, hydro, ene)
            if best is None or key < best[0]:
                best = (key, a2l)
        if best is None:
            return None
        a2l = best[1]
        stem = parent[:-1] if parent.endswith('e') else parent  # naphthalen
        if is_perhydro:
            # Fully-saturated fused carbocycle: cite the COUNT-form prefix
            # (P-31.1.4.3.4 / the explicit-hydrogen convention this project uses
            # for the PARENT, e.g. 'decahydronaphthalene'), with NO per-position
            # hydro locants, then the free-valence locant ->
            # 'decahydronaphthalen-2-yl'. Fail-closed if the count is non-standard.
            count_prefix = ps.get_saturation_prefix(det.get('hydrogen_count', len(sp3)))
            if not count_prefix:
                return None
            return f'{count_prefix}{stem}-{_fmt_locant(a2l[attach_sub])}-yl'
        hydro_locs = sorted((a2l[a] for a in sp3), key=_locant_key)
        hydro_str = ','.join(_fmt_locant(x) for x in hydro_locs)
        return f'{hydro_str}-{prefix}{stem}-{_fmt_locant(a2l[attach_sub])}-yl'
    except Exception:
        return None


def _polycyclic_substituent_name(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: Optional[int]
) -> Optional[str]:
    """Name a MULTI-RING substituent (von-Baeyer, spiro, or partially hydro fused
    carbocycle) by routing the detached ring system through the existing parent
    namers with free-valence numbering. Returns the ``...-<loc>-yl`` name, or None
    (fail-closed) for any polycyclic the routed namers cannot number — never a
    monocycle size-guess (the old ``cyclo{N}yl`` corruption)."""
    if attachment_atom is None or attachment_atom not in set(ring_atoms):
        return None
    sub, attach_sub = _extract_ring_submol(mol, ring_atoms, attachment_atom)
    if sub is None:
        return None
    for namer in (_vonbaeyer_substituent_name,
                  _spiro_substituent_name,
                  _fused_hydro_substituent_name):
        try:
            name = namer(sub, attach_sub)
        except Exception:
            name = None
        if name:
            return name
    return None


def _phthalimido_substituent_name(mol, frag_set: Set[int],
                                  attach_idx: int) -> Optional[str]:
    """P-66.2.2 (BB 33859/55900): the phthalimido fragment
    O=C1N([*])C(=O)c2ccccc21 -> '1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl'
    (the BB preferred prefix). Matched by SUBGRAPH (attach N carries the
    two flanking carbonyls of a benzo-fused 5-ring imide) with EXACT
    fragment coverage. Returns None for anything else (fail-closed)."""
    from rdkit import Chem
    if attach_idx not in frag_set:
        attach_idx = next(
            (n.GetIdx() for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
             if n.GetIdx() in frag_set), None)
        if attach_idx is None:
            return None
    a = mol.GetAtomWithIdx(attach_idx)
    if a.GetAtomicNum() != 7:
        return None
    patt = Chem.MolFromSmarts("O=C1N([*])C(=O)c2ccccc21")
    if patt is None:
        return None
    for match in mol.GetSubstructMatches(patt):
        # match atoms: O, C, N, [*], C, O, c, c, c, c, c, c  (12 atoms)
        # the [*] (index 3) is the attachment substituent, OUTSIDE the frag.
        n_in_match = match[2]
        star = match[3]
        if n_in_match != attach_idx:
            continue
        # every matched atom except the [*] wildcard must be IN the fragment,
        # and the fragment must contain NOTHING else (exact coverage).
        core = set(match) - {star}
        if core == frag_set:
            return "1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl"
    return None


def name_ring_system_substituent(
    mol,
    frag_atoms,
    attach_idx: int,
    allow_enumerator_fallback: bool = True,
) -> Optional[str]:
    """Name a RING-CONTAINING substituent fragment (WS-A task 9 chokepoint).

    The single delegate used by every ring-parent path (fused-heterocycle
    parents, PAH parents) when a substituent fragment contains ring atoms.
    Chooses the producer:

    - fragment IS exactly one ring system rooted at a ring atom ->
      ``get_ring_substituent_name`` (PIN free-valence locant, P-29.2:
      naphthalen-2-yl, pyridin-2-yl, 1H-indol-2-yl, ...);
    - anything else (ring + chain linker, chain-rooted) -> the universal
      ``substituent_enumerator.name_substituent``.

    Returns None when no trustworthy name can be produced — callers must
    treat None as "do not emit", never fabricate a carbon-count alkyl name
    for a ring fragment (the historical phenyl->'hexyl' corruption).
    """
    ring_info = mol.GetRingInfo()
    frag_atoms = list(frag_atoms)
    frag_set = set(frag_atoms)

    # Normalize: some callers pass the PARENT-side attachment atom
    # (SubstituentInfo.attach_mol_idx). The free-valence atom must be the
    # FRAGMENT-side atom bonded to it.
    if attach_idx not in frag_set:
        attach_idx = next(
            (n.GetIdx()
             for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
             if n.GetIdx() in frag_set),
            None,
        )
        if attach_idx is None:
            return None

    name: Optional[str] = None
    # P-66.2.2 (BB 33859/55900, W2E-P1FG Task 15 L2): the exact phthalimido
    # fragment — a benzo-fused 5-ring imide N-attached, with the two ring
    # carbons flanking the N each bearing an exocyclic =O -> the BB preferred
    # prefix '1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl'. Matched by subgraph
    # (not a SMILES string compare) with full-fragment coverage; anything else
    # falls through to the generic producers (fail-closed).
    _phth = _phthalimido_substituent_name(mol, frag_set, attach_idx)
    if _phth is not None:
        return _phth

    frag_ring_atoms = {a for a in frag_atoms if ring_info.NumAtomRings(a) > 0}
    if frag_ring_atoms == frag_set and ring_info.NumAtomRings(attach_idx) > 0:
        try:
            name = get_ring_substituent_name(mol, tuple(frag_atoms), attach_idx)
        except Exception:
            name = None
    elif frag_ring_atoms and ring_info.NumAtomRings(attach_idx) == 0:
        # Chain-ROOTED fragment carrying a ring system, e.g. -CH2-naphthalene.
        # P-29.1.2 compound substituent: '(naphthalen-2-yl)methyl'. The
        # generic recursive namer mis-roots these (it names the fragment as a
        # molecule and appends -yl: '2-methylnaphthalenyl'), so build the
        # carrier+ring form here under tight guards; anything more complex
        # falls through to the universal producer (status quo).
        name = _compound_ring_on_chain_substituent(
            mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info
        )
    if not name and allow_enumerator_fallback:
        # Recursion guard: the public ``name_substituent`` cascade routes
        # ring-bearing fragments back here (Tier 1.95). When called from there,
        # the caller passes allow_enumerator_fallback=False so a decline returns
        # None instead of re-entering the cascade (no infinite loop).
        from ..assembly.substituent_enumerator import name_substituent
        name = name_substituent(mol, frag_atoms, attach_idx)
    if name and name != 'substituent' and ' ' not in name:
        return name
    return None


def _compound_ring_on_chain_substituent(
    mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info
) -> Optional[str]:
    """Build '(ring-yl)alkyl' for an unbranched all-carbon carrier rooted at
    the attachment with exactly ONE ring system hanging off it. Returns None
    (caller falls back) whenever any guard fails — never guesses."""
    # P-29.6.2.1 (BB 16304): a substituent ON the ring (e.g. -Cl at para of a
    # benzyl's phenyl) belongs to the ring-yl name '(4-chlorophenyl)', NOT to
    # the carrier. Fold single non-H heavy atoms that hang off a ring atom (and
    # are NOT part of the ring themselves) INTO frag_ring_atoms BEFORE computing
    # the carrier, so a halogen/alkyl decoration is never mistaken for a
    # non-carbon carrier atom (which would decline at the all-carbon guard).
    # name_ring_system_substituent renders the widened single-substituent phenyl
    # as '4-chlorophenyl' and fails closed on anything it cannot fully describe,
    # so no wrong name leaks; multi-substituent rings still trip the coverage
    # guard below (frag_ring_atoms names as bare 'phenyl' -> caller declines).
    ring_substituent_atoms = set()
    for rc in list(frag_ring_atoms):
        for n in mol.GetAtomWithIdx(rc).GetNeighbors():
            ni = n.GetIdx()
            if not (ni in frag_set and ni not in frag_ring_atoms
                    and n.GetAtomicNum() > 1
                    and ring_info.NumAtomRings(ni) == 0):
                continue
            # A ring SUBSTITUENT is terminal to the substituent fragment: it is
            # NOT the free-valence attachment atom, and it touches nothing
            # outside the fragment (the CARRIER, by contrast, bridges the ring
            # to the parent and so is bonded to a non-fragment atom). This
            # cleanly separates the para-Cl (folds in) from the -CH2- carrier
            # (stays carrier).
            if ni == attach_idx:
                continue
            if any(nn.GetIdx() not in frag_set
                   for nn in n.GetNeighbors()):
                continue
            if n.GetDegree() != 1:
                continue
            ring_substituent_atoms.add(ni)
    frag_ring_atoms = set(frag_ring_atoms) | ring_substituent_atoms
    raw_carrier = frag_set - frag_ring_atoms
    # w2f p1 (P-29.6.2.1, BB 16322 'bromo(4-methylphenyl)methyl (preferred
    # prefix)'): an α-HALOGEN on the carrier is a carrier DECORATION cited
    # as a prefix, not a carrier atom (it used to land in `carrier` and
    # fail the all-carbon guard at the return-None below). v1 whitelist:
    # halogens ONLY — compulsory prefix-only groups (Table 5.1), valence-1,
    # locant-free on a mononuclear carrier; they can never re-parent the
    # substituent nor form a PCG. Everything else (nitro / OH / NH2 / =O)
    # stays fail-closed or is owned by parent selection (research §3.C).
    _halogen_z = {9, 17, 35, 53}
    decorations = set()
    for _a in raw_carrier:
        _atom = mol.GetAtomWithIdx(_a)
        if _atom.GetAtomicNum() not in _halogen_z:
            continue
        if _atom.GetDegree() != 1 or _atom.GetFormalCharge() != 0:
            continue
        _bond = _atom.GetBonds()[0]
        _nbr = _bond.GetOtherAtom(_atom)
        if (_bond.GetBondTypeAsDouble() == 1.0
                and _nbr.GetIdx() in raw_carrier
                and _nbr.GetSymbol() == 'C'):
            decorations.add(_a)
    carrier = raw_carrier - decorations
    if decorations and (len(carrier) != 1 or len(decorations) > 2):
        # v1: decorated MONONUCLEAR carriers with 1-2 halogens only (the
        # P-29.6.2.1 benzylic α class). Decorated LONGER carriers need
        # decoration locants — not built; stays fail-closed as today.
        return None
    # Guards: all-carbon, fully SATURATED, undecorated carrier; attach is a
    # carrier atom. The saturation guard is load-bearing: get_alkyl_name
    # cannot express -ene/-yne, so an unsaturated carrier here silently
    # described a DIFFERENT molecule (canary rt75_0430: (2E)-prop-2-enyl
    # emitted as 'propyl', RT True->False).
    if attach_idx not in carrier:
        return None
    for a in carrier:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != 'C':
            return None
        for b in atom.GetBonds():
            if (b.GetOtherAtom(atom).GetIdx() in frag_set
                    and b.GetBondTypeAsDouble() != 1.0):
                return None
    # Carrier must be a simple path rooted at attach.
    adj = {
        a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors()
            if n.GetIdx() in carrier]
        for a in carrier
    }
    if len(adj.get(attach_idx, [])) > 1:
        return None
    path = [attach_idx]
    while True:
        nxt = [n for n in adj[path[-1]] if n not in path]
        if not nxt:
            break
        if len(nxt) > 1:
            return None
        path.append(nxt[0])
    if len(path) != len(carrier):
        return None
    # Exactly one ring system, attached to exactly one carrier atom.
    ring_attach_positions = []
    ring_side_atoms = []
    for pos, c in enumerate(path, start=1):
        for n in mol.GetAtomWithIdx(c).GetNeighbors():
            if n.GetIdx() in frag_ring_atoms:
                ring_attach_positions.append(pos)
                ring_side_atoms.append(n.GetIdx())
    if len(ring_attach_positions) != 1:
        return None
    # Coverage: every fragment atom is carrier, ring (ring substituents
    # were folded into frag_ring_atoms at function entry), or an admitted
    # α-halogen decoration (w2f p1) — anything else declines.
    if carrier | frag_ring_atoms | decorations != frag_set:
        return None
    if decorations:
        # P-45.1.1 identical-units decline (research §3.C.4 — LOAD-BEARING,
        # the item's #1 hazard): when the in-fragment ring unit equals the
        # parent side, the PIN is MULTIPLICATIVE
        # (1,1'-(dichloromethylene)dibenzene). Shapes name_multiplicative
        # CAN build never reach here (dispatch @900); shapes it cannot yet
        # build (gem-dihalo, classifier requires 1 sub + 1 H) MUST
        # refuse — the substitutive '[dichloro(phenyl)methyl]benzene' is
        # RT-valid but non-PIN (a SELF-01-invisible wrong name, leak proof
        # research §3.D).
        from rdkit import Chem
        try:
            _unit_a = Chem.MolFragmentToSmiles(
                mol, atomsToUse=sorted(frag_ring_atoms))
            _complement = [a.GetIdx() for a in mol.GetAtoms()
                           if a.GetIdx() not in frag_set]
            _unit_b = Chem.MolFragmentToSmiles(mol, atomsToUse=_complement)
            if (not _unit_a or not _unit_b
                    or Chem.CanonSmiles(_unit_a) == Chem.CanonSmiles(_unit_b)):
                return None
        except Exception:
            return None
    ring_name = name_ring_system_substituent(
        mol, sorted(frag_ring_atoms), ring_side_atoms[0]
    )
    if not ring_name:
        return None
    try:
        from ..assembly.naming_utils import get_alkyl_name
        alkyl = get_alkyl_name(len(path))
    except Exception:
        return None
    if not alkyl:
        return None
    loc = ring_attach_positions[0]
    if decorations:
        # w2f p1 (P-29.6.2.1 / P-16.5.1.3.1, BB 7272): cite ALL carrier
        # substituents (halo prefixes + ring-yl) in ONE alphanumerical
        # sequence by the letters-only key; first-cited bare UNLESS it
        # includes a locant; every further item parenthesized; multiplying
        # prefixes outside parens. OPSIN-verified surfaces (research §3.D):
        # bromo(phenyl)methyl / dichloro(phenyl)methyl /
        # chloro(fluoro)(phenyl)methyl / (4-bromophenyl)(chloro)methyl.
        # An α-decorated carrier NEVER takes retained 'benzyl' (P-29.6.2.1:
        # any α-substitution kills the retained form in PINs) — the tail
        # below (incl. the benzyl gate) is unreachable from here.
        from collections import Counter
        from ..assembly.naming_utils import alpha_sort_key
        _halo_name = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
        _halo_counts = Counter(
            _halo_name[mol.GetAtomWithIdx(_d).GetAtomicNum()]
            for _d in decorations
        )
        if any(_k > 1 for _k in _halo_counts.values()) and \
                any(_ch.isdigit() for _ch in ring_name):
            # gem-dihalo + locant-bearing ring-yl: no OPSIN-verified
            # surface (research §3.C.5) — refuse rather than guess.
            return None
        _items = []
        for _hname, _k in _halo_counts.items():
            _rendered = _hname if _k == 1 else f'di{_hname}'
            _items.append((alpha_sort_key(_rendered), _rendered, False))
        _items.append((alpha_sort_key(ring_name), ring_name, True))
        _items.sort(key=lambda _t: _t[0])
        _parts = []
        for _pos_i, (_key, _rendered, _is_ring) in enumerate(_items):
            if _pos_i == 0:
                # first-cited: bare unless it includes a locant
                _parts.append(f'({_rendered})'
                              if any(_ch.isdigit() for _ch in _rendered)
                              else _rendered)
            else:
                if not _is_ring and _rendered.startswith('di'):
                    return None  # unreachable in v1 scope; fail closed
                _parts.append(f'({_rendered})')
        # Returned BARE (no spaces -> passes the space-guard); the citation
        # layer (naming_utils.format_substituent_prefix, Task 6) escalates
        # the outer mark to brackets (P-16.5.2.4). Mononuclear carrier
        # cites no locant.
        return f"{''.join(_parts)}{alkyl}"
    # P-16.3.3: enclose the ring-yl in marks only when it is itself complex
    # (carries locants/parens, e.g. '(naphthalen-2-yl)methyl'); a simple ring-yl
    # is concatenated bare ('cyclohexylmethyl', 'phenylmethyl' is retained
    # 'benzyl' elsewhere). The old unconditional parens produced the non-PIN
    # '(cyclohexyl)methyl'.
    from ..assembly.naming_utils import is_complex_substituent
    inner = f'({ring_name})' if is_complex_substituent(ring_name) else ring_name
    if len(path) == 1:
        # P-29.6.1 (BB 16270): the UNSUBSTITUTED -CH2-C6H5 group is the retained
        # PREFERRED prefix 'benzyl', not 'phenylmethyl'. Gated tightly to the
        # bare phenyl case (ring_name == 'phenyl' means the ring carries no
        # substituent — a substituted phenyl comes back as '4-chlorophenyl' and
        # correctly stays the systematic (4-chlorophenyl)methyl per P-29.6.2.1).
        # The single -CH2- carrier is already guaranteed by len(path)==1 plus
        # the saturated/undecorated carrier guards above.
        if ring_name == 'phenyl':
            return 'benzyl'
        return f'{inner}methyl'
    return f'{loc}-{inner}{alkyl}'


def get_ring_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_point: Optional[int] = None
) -> Optional[str]:
    """
    Get the substituent name for a ring when it becomes a substituent on a chain.

    This is the main entry point for ring-as-substituent naming.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_point: Optional ring atom index where the ring attaches to chain.
                         Used for position-specific names (e.g., 2-pyridyl vs 4-pyridyl).

    Returns:
        Substituent name string (e.g., 'phenyl', 'cyclohexyl', '2-pyridyl'), or
        None when the ring system cannot be named by a provable rule (Phase 4
        SUBST-01: fail-closed — never a monocycle size-guess for a polycyclic).
    """
    # Identify the ring system
    ring_name = identify_ring_system(mol, ring_atoms)

    if ring_name is None:
        # Multi-ring system (fused polycyclic): check retained names
        # identify_ring_system() only handles single rings (up to 8 atoms).
        # For multi-ring systems (naphthalene=10, anthracene=14, etc.),
        # extract the fragment SMILES and look up retained names.
        from rdkit import Chem
        frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
        if frag_smi:
            from ..data import get_retained_name
            retained = get_retained_name(frag_smi)
            # WS-A task 9: the fused-heterocycle catalog name carries the
            # indicated-hydrogen / numbering prefix the plain retained table
            # drops ('1-benzofuran' vs 'benzofuran', '1H-indole' vs
            # 'indole') — prefer it for PIN substituent stems.
            try:
                from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
                _fh_entry = FUSED_HETEROCYCLE_DATA.get(frag_smi)
                if _fh_entry and _fh_entry.get('name'):
                    retained = _fh_entry['name']
            except ImportError:
                pass
            if retained:
                # Convert retained name to substituent form:
                # naphthalene -> naphthalen-{locant}-yl
                # anthracene -> anthracen-{locant}-yl
                # phenanthrene -> phenanthren-{locant}-yl
                # General rule: drop trailing 'e', add '-yl'
                stem = retained.rstrip('e') if retained.endswith('e') else retained
                # Determine attachment locant for position-specific naming
                if attachment_point is not None:
                    attach_locant = _get_polycyclic_attachment_locant(
                        mol, ring_atoms, attachment_point, ring_name=retained
                    )
                    if attach_locant is not None:
                        return f'{stem}-{attach_locant}-yl'
                    # Position matters but the retained-stem numberer cannot place
                    # it -- e.g. a saturated/hydro retained ring such as
                    # 'decahydronaphthalene' (_get_polycyclic_attachment_locant
                    # only numbers mancude PAH stems). Do NOT emit the locant-less
                    # '{stem}-yl': it mis-names a non-symmetric attachment and
                    # fails OPSIN-RT (SELF-01 then suppresses it to unknown).
                    # Fall through to _polycyclic_substituent_name below, which
                    # numbers the free valence via the mancude parent
                    # (D-FOLLOWON item 3 -> 'decahydronaphthalen-2-yl').
                else:
                    return f'{stem}-yl'

        # Phase 4 SUBST-01: multi-ring substituent with no retained name —
        # route the detached system through the von-Baeyer / spiro / partial-hydro
        # parent namers with free-valence numbering (bicyclo[2.2.1]heptan-2-yl,
        # 5,6,7,8-tetrahydronaphthalen-1-yl, spiro[4.5]decan-2-yl).
        poly = _polycyclic_substituent_name(mol, ring_atoms, attachment_point)
        if poly:
            return poly

        # Fail-closed: an unidentifiable ring system (the old generic
        # ``cyclo{N}yl`` size-guess named a DIFFERENT molecule — norbornane as
        # cycloheptyl, tetralin as cyclodecyl). Decline so the caller emits a
        # locant-less form or an honest ``unknown`` rather than a wrong name.
        return None

    # Phase 4 SUBST-01 (a): a monocyclic, all-carbon, non-aromatic ring carrying
    # a skeletal multiple bond keeps its ene/yne locants as a substituent
    # (cyclohex-1-en-1-yl) — identify_ring_system reports the saturated stem
    # ('cyclohexane'), dropping the unsaturation. Free-valence-first numbering.
    if attachment_point is not None:
        _unsat = _unsaturated_carbocyclic_substituent(
            mol, ring_atoms, attachment_point)
        if _unsat is not None:
            return _unsat

    # Wave2 T6b (P-32.2.1): heteromonocyclic analog — mancude parent + hydro
    # prefixes + indicated hydrogen (3,4-dihydro-2H-pyran-2-yl).
    if attachment_point is not None:
        _h_unsat = _unsaturated_heteromonocyclic_substituent(
            mol, ring_atoms, attachment_point)
        if _h_unsat is not None:
            return _h_unsat

    # Wave2 T6b fail-closed: a NON-AROMATIC heteromonocycle with a skeletal
    # multiple bond must never fall through to its SATURATED dictionary stem —
    # 'oxanyl' for a dihydropyranyl fragment names a DIFFERENT molecule (the
    # ring double bond is silently dropped). If the dedicated emitter above
    # declined, decline entirely; the caller emits an honest unknown.
    _ring_set_t6 = set(ring_atoms)
    _t6_has_het = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_atoms
    )
    if _t6_has_het and not all(
        mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms
    ):
        for _i in ring_atoms:
            for _nbr in mol.GetAtomWithIdx(_i).GetNeighbors():
                _j = _nbr.GetIdx()
                if _j in _ring_set_t6 and _i < _j:
                    _b = mol.GetBondBetweenAtoms(_i, _j)
                    if _b.GetBondTypeAsDouble() >= 2.0:
                        return None

    # IUPAC P-31.1.4.3.4: a monocyclic heteroaryl substituent takes
    # free-valence numbering (pyridin-3-yl, 1H-imidazol-5-yl, furan-2-yl, ...),
    # which supersedes both the legacy POSITION_SPECIFIC forms (3-pyridyl) and
    # the locant-less dictionary forms (imidazolyl). Guarded — returns None and
    # falls through to the legacy forms below whenever the locant is not
    # provably PIN-correct (benzene, saturated rings, substituted rings, etc.).
    if attachment_point is not None:
        pin_name = pin_heteroaryl_substituent_name(
            mol, ring_atoms, attachment_point
        )
        if pin_name is not None:
            return pin_name

    # Check for position-specific name
    if ring_name in POSITION_SPECIFIC_RINGS and attachment_point is not None:
        # Determine the position in the ring
        ring_position = _get_ring_position_for_attachment(
            mol, ring_atoms, attachment_point, ring_name
        )
        if ring_position in POSITION_SPECIFIC_RINGS[ring_name]:
            return POSITION_SPECIFIC_RINGS[ring_name][ring_position]

    # Look up standard substituent name
    if ring_name in RING_SUBSTITUENT_NAMES:
        return RING_SUBSTITUENT_NAMES[ring_name]

    # Fallback: convert ring name to substituent form
    # Generally: remove 'e' and add 'yl' (benzene -> benzyl, but benzene -> phenyl is special)
    if ring_name.endswith('ane'):
        if ring_name.startswith('cyclo'):
            # P-29.2: carbocyclic cycloalkanes drop '-ane' entirely:
            # cyclononane -> cyclononyl, cyclododecane -> cyclododecyl
            # (sizes 3-8 hit RING_SUBSTITUENT_NAMES above; >=9 land here —
            # the old [:-1] kept 'an': 'cyclododecanyl', WS-A task 9).
            return ring_name[:-3] + 'yl'
        # Hantzsch-Widman '-ane' heterocycles elide only the final 'e':
        # azepane -> azepanyl, oxocane -> oxocanyl
        return ring_name[:-1] + 'yl'
    elif ring_name.endswith('ene'):
        return ring_name[:-1] + 'yl'  # cyclohexene -> cyclohexenyl
    elif ring_name.endswith('ine'):
        return ring_name[:-1] + 'yl'  # pyridine -> pyridinyl

    # Wave2 T3c: benzene -> 'phenyl' (never the invalid 'benzenyl' token). The
    # RING_SUBSTITUENT_NAMES table above already returns 'phenyl' for an
    # unsubstituted benzene, so this is belt-and-suspenders — it only guarantees
    # the malformed 'benzenyl' can never escape if any future path reaches this
    # fallback with ring_name=='benzene'. Substituted-benzene decoration is
    # handled upstream by decorated_ring_substituent_name (stem 'phenyl').
    if ring_name == 'benzene':
        return 'phenyl'

    return ring_name + 'yl'


def _get_polycyclic_attachment_locant(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
    ring_name: Optional[str] = None
) -> Optional[int]:
    """
    Determine the IUPAC locant for an attachment point on a polycyclic ring system.

    Uses the retained name's canonical IUPAC numbering. For polycyclic aromatics,
    we use the IUPAC numbering by checking the fused_heterocycles data or by
    using the RDKit canonical atom ordering as a proxy.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring system
        attachment_atom: The ring atom that connects to the chain

    Returns:
        IUPAC locant (1-indexed), or None if cannot determine
    """
    from rdkit import Chem

    # Try to get IUPAC numbering from fused heterocycle data.
    # WS-A task 9: 'iupac_locants' is keyed by the atom indices of the DATA
    # ENTRY's reference SMILES, NOT this molecule's indices. The old direct
    # `attachment_atom in iupac_locants` lookup compared across index spaces
    # and returned whatever locant collided ('1H-indol-3-yl' for a C2
    # attachment). Translate via substructure match of the reference onto
    # THIS ring system, minimized over automorphisms (P-29.2 lowest
    # free-valence locant).
    frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        if frag_smi in FUSED_HETEROCYCLE_DATA:
            entry = FUSED_HETEROCYCLE_DATA[frag_smi]
            iupac_locants = entry.get('iupac_locants', {})
            ref = Chem.MolFromSmiles(frag_smi) if iupac_locants else None
            if ref is not None:
                ring_atom_set = set(ring_atoms)
                best_het_locant = None
                for match in mol.GetSubstructMatches(ref, uniquify=False):
                    if set(match) != ring_atom_set:
                        continue
                    for ref_idx, mol_idx in enumerate(match):
                        if mol_idx != attachment_atom:
                            continue
                        loc = iupac_locants.get(ref_idx)
                        if isinstance(loc, int) and (
                                best_het_locant is None
                                or loc < best_het_locant):
                            best_het_locant = loc
                if best_het_locant is not None:
                    return best_het_locant
    except ImportError:
        pass

    # WS-A task 9: carbocyclic polycyclic aromatics (naphthalene, anthracene,
    # ...) have authoritative IUPAC numbering in the PAH machinery. P-29.2:
    # the free valence takes the LOWEST locant the numbering allows, so
    # minimize over ALL automorphic substructure matches of THIS ring system
    # (naphthalen-2-yl, never the symmetry-equivalent -6-yl).
    if ring_name is not None:
        try:
            from .polycyclics import POLYCYCLIC_DATA, _map_pah_atoms_to_iupac
            pah_data = POLYCYCLIC_DATA.get(ring_name)
            if pah_data:
                pattern = Chem.MolFromSmarts(pah_data['smarts'])
                if pattern is not None:
                    ring_atom_set = set(ring_atoms)
                    best_locant: Optional[int] = None
                    for match in mol.GetSubstructMatches(pattern, uniquify=False):
                        if set(match) != ring_atom_set:
                            continue
                        mapping = _map_pah_atoms_to_iupac(
                            mol, ring_name, list(match)
                        )
                        loc = mapping.get(attachment_atom) if mapping else None
                        if isinstance(loc, int) and (
                                best_locant is None or loc < best_locant):
                            best_locant = loc
                    if best_locant is not None:
                        return best_locant
        except ImportError:
            pass

    # For monocyclic heterocycles with retained names (e.g. 1,3-dioxolane),
    # use proper IUPAC numbering: start from highest-priority heteroatom,
    # go in direction giving lowest locant set for remaining heteroatoms.
    # For all-carbon rings or polycyclic aromatics without explicit data,
    # fall back to sorted atom order.
    ring_set = set(ring_atoms)

    # Collect heteroatoms in the ring
    het_indices = []
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            het_indices.append(idx)

    if het_indices and len(ring_atoms) <= 8:
        # Monocyclic heterocycle: build a ring traversal path and number
        # according to IUPAC Hantzsch-Widman rules (heteroatom gets pos 1,
        # direction chosen to give lowest locant set for other heteroatoms).

        # Build adjacency within ring
        ring_adj: dict = {idx: [] for idx in ring_atoms}
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_set:
                    ring_adj[idx].append(nidx)

        # Heteroatom priority: O > S > N (Hantzsch-Widman)
        _het_priority = {'O': 0, 'S': 1, 'Se': 2, 'N': 3}
        het_indices_sorted = sorted(
            het_indices,
            key=lambda i: _het_priority.get(mol.GetAtomWithIdx(i).GetSymbol(), 99)
        )
        start_atom = het_indices_sorted[0]  # Highest priority heteroatom = position 1

        # Try both directions around the ring from start_atom
        def _traverse_ring(start, first_next, adj, ring_size):
            """Walk around ring from start via first_next, return ordered path."""
            path = [start, first_next]
            while len(path) < ring_size:
                prev = path[-2]
                curr = path[-1]
                nexts = [n for n in adj[curr] if n != prev and n in ring_set]
                if not nexts:
                    break
                path.append(nexts[0])
            return path

        neighbors_of_start = ring_adj[start_atom]
        best_path = None
        best_het_locants = None

        for next_atom in neighbors_of_start:
            path = _traverse_ring(start_atom, next_atom, ring_adj, len(ring_atoms))
            if len(path) != len(ring_atoms):
                continue
            # Compute heteroatom locant set (excluding position 1 which is always a het)
            het_locs = tuple(sorted(
                path.index(hi) + 1 for hi in het_indices if hi != start_atom
            ))
            if best_het_locants is None or het_locs < best_het_locants:
                best_het_locants = het_locs
                best_path = path

        if best_path and attachment_atom in best_path:
            return best_path.index(attachment_atom) + 1  # 1-indexed

    # Fallback: canonical atom order within the ring system. ONLY valid for
    # monocyclic fragments (any rotation is a legal numbering start there up
    # to direction). For MULTI-ring systems this would fabricate a locant
    # from the arbitrary atom-index order — a wrong locant is worse than a
    # locant-less name, so return None and let the caller emit the bare
    # '{stem}-yl' form (WS-A task 9).
    ri_local = mol.GetRingInfo()
    n_rings_in_fragment = sum(
        1 for r in ri_local.AtomRings() if set(r) <= ring_set
    )
    if n_rings_in_fragment > 1:
        return None

    ring_list = sorted(ring_atoms)
    if attachment_atom in ring_list:
        return ring_list.index(attachment_atom) + 1  # 1-indexed

    return None


def _get_ring_position_for_attachment(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
    ring_name: str
) -> Optional[int]:
    """
    Determine the ring position number for an attachment point.

    For position-specific substituents like pyridine (2-pyridyl, 3-pyridyl, 4-pyridyl),
    we need to determine which position the attachment is at based on IUPAC numbering.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_atom: The ring atom that connects to the chain
        ring_name: Name of the ring system

    Returns:
        Ring position number (1-indexed), or None if cannot determine
    """
    # For pyridine: N is at position 1, so we number relative to N
    # Try both ring directions and pick the one giving the lowest locant
    # at the attachment point (IUPAC lowest locant rule).
    if ring_name == 'pyridine':
        # Find the nitrogen
        n_idx = None
        for idx in ring_atoms:
            if mol.GetAtomWithIdx(idx).GetSymbol() == 'N':
                n_idx = idx
                break

        if n_idx is None:
            return None

        # Try both ring traversal directions from N
        ring_set = set(ring_atoms)
        n_atom = mol.GetAtomWithIdx(n_idx)
        ring_neighbors = [
            nbr.GetIdx() for nbr in n_atom.GetNeighbors()
            if nbr.GetIdx() in ring_set
        ]

        best_pos = None
        for start_nbr in ring_neighbors:
            path = _build_ring_path_from_start_via(
                mol, ring_atoms, n_idx, start_nbr
            )
            if attachment_atom in path:
                pos = path.index(attachment_atom) + 1  # 1-indexed
                if best_pos is None or pos < best_pos:
                    best_pos = pos

        return best_pos

    # For naphthalene: determine alpha (1,4,5,8) vs beta (2,3,6,7) position
    # Alpha positions are adjacent to the fusion bond; beta are farther
    if ring_name == 'naphthalene':
        ring_set = set(ring_atoms)
        # Find fusion atoms: ring atoms bonded to 3 other ring atoms
        # (shared atoms between the two 6-membered rings in naphthalene)
        fusion_atoms = set()
        for idx in ring_atoms:
            atom = mol.GetAtomWithIdx(idx)
            ring_nbr_count = sum(
                1 for nbr in atom.GetNeighbors() if nbr.GetIdx() in ring_set
            )
            if ring_nbr_count == 3:
                fusion_atoms.add(idx)
        if len(fusion_atoms) == 2 and attachment_atom in ring_set:
            # Alpha positions: atoms adjacent to a fusion atom (but not fusion atoms themselves)
            is_alpha = any(
                mol.GetBondBetweenAtoms(attachment_atom, fa) is not None
                for fa in fusion_atoms
            )
            if attachment_atom in fusion_atoms:
                return None  # Fusion atom itself: use generic naphthyl
            return 1 if is_alpha else 2  # 1-naphthyl (alpha) or 2-naphthyl (beta)
        return None

    return None


def _build_ring_path_from_start(
    mol,
    ring_atoms: Tuple[int, ...],
    start_idx: int
) -> List[int]:
    """
    Build an ordered path around the ring starting from a given atom.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        start_idx: Starting atom index

    Returns:
        List of atom indices in order around the ring
    """
    ring_set = set(ring_atoms)
    path = [start_idx]
    visited = {start_idx}

    current = start_idx
    while len(path) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)

        # Find next ring neighbor not yet visited
        next_idx = None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                next_idx = nbr_idx
                break

        if next_idx is None:
            break

        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx

    return path


def _build_ring_path_from_start_via(
    mol,
    ring_atoms: Tuple[int, ...],
    start_idx: int,
    first_neighbor: int
) -> List[int]:
    """Build an ordered path around the ring starting from start_idx going
    through first_neighbor.

    This allows choosing the direction of traversal around the ring, which
    is needed for IUPAC lowest-locant rule: try both directions and pick
    the one giving lower locants.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        start_idx: Starting atom index (position 1)
        first_neighbor: The neighbor of start_idx to visit first (direction)

    Returns:
        List of atom indices in order around the ring
    """
    ring_set = set(ring_atoms)
    path = [start_idx, first_neighbor]
    visited = {start_idx, first_neighbor}

    current = first_neighbor
    while len(path) < len(ring_atoms):
        atom = mol.GetAtomWithIdx(current)
        next_idx = None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in ring_set and nbr_idx not in visited:
                next_idx = nbr_idx
                break
        if next_idx is None:
            break
        path.append(next_idx)
        visited.add(next_idx)
        current = next_idx

    return path


def get_ring_attachment_locant(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int],
    atom_to_locant: Dict[int, int]
) -> int:
    """
    Find which chain position the ring is attached to.

    The ring connects to the chain via a bond between a ring atom and a chain atom.
    This function finds that chain atom and returns its locant.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain
        atom_to_locant: Mapping from chain atom index to locant (1-indexed)

    Returns:
        Locant (1-indexed position) where the ring attaches to the chain

    Raises:
        ValueError: If no connection found between ring and chain
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    # Find the bond connecting ring to chain
    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Check if neighbor is on the chain
            if nbr_idx in chain_set:
                # Found the connection
                if nbr_idx in atom_to_locant:
                    return atom_to_locant[nbr_idx]

    # If no direct connection found, raise error
    raise ValueError(
        "Could not find connection between ring and chain. "
        f"Ring atoms: {ring_atoms}, Chain atoms: {chain_atoms}"
    )


def get_ring_attachment_atom(
    mol,
    ring_atoms: Tuple[int, ...],
    chain_atoms: List[int]
) -> Optional[int]:
    """
    Find the ring atom that attaches to the chain.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        chain_atoms: List of atom indices in the principal chain

    Returns:
        Ring atom index that connects to chain, or None if not found
    """
    ring_set = set(ring_atoms)
    chain_set = set(chain_atoms)

    for ring_idx in ring_atoms:
        ring_atom = mol.GetAtomWithIdx(ring_idx)

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            if nbr_idx in chain_set:
                return ring_idx

    return None


def _ring_atom_simple_substituents(mol, ring_atom_idx: int,
                                   ring_atom_set: Set[int],
                                   skip_atoms: Set[int]):
    """Detect the supported simple substituents anchored at one ring atom.

    Returns (prefixes, covered_atoms) where prefixes is a list of prefix
    strings and covered_atoms the exocyclic atom indices they account for —
    or None if the atom carries ANY exocyclic group outside the supported
    table (the caller must then guard out to its legacy form; emitting a
    partially-decorated name would be structurally wrong).

    v1 table (per the WS-A.2 flip-population scope — covers the head of the
    distribution): oxo, cyano, halogen, hydroxy, methoxy/simple n-alkoxy,
    amino (-NH2), nitro, unbranched pure alkyl. Deliberately NOT supported
    (guard out): esters/amides/acyl, sulfonyl, nested ring substituents,
    branched alkyl, anything charged or exotic.
    """
    _HAL = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    _ALKYL = {1: 'methyl', 2: 'ethyl', 3: 'propyl', 4: 'butyl',
              5: 'pentyl', 6: 'hexyl', 7: 'heptyl', 8: 'octyl'}
    _ALKOXY = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy', 4: 'butoxy'}

    def _linear_carbon_chain(start_idx, exclude):
        """Length of a pure, UNBRANCHED, saturated C/H chain from start_idx,
        or None if branched/heteroatom/unsaturated/cyclic."""
        length = 0
        prev = None
        cur = start_idx
        while True:
            atom = mol.GetAtomWithIdx(cur)
            if (atom.GetSymbol() != 'C' or atom.GetIsAromatic()
                    or atom.IsInRing() or atom.GetFormalCharge() != 0):
                return None
            nxt = [n.GetIdx() for n in atom.GetNeighbors()
                   if n.GetIdx() != prev and n.GetIdx() not in exclude]
            for n_idx in nxt:
                bond = mol.GetBondBetweenAtoms(cur, n_idx)
                if bond.GetBondTypeAsDouble() != 1.0:
                    return None
            length += 1
            if not nxt:
                return length
            if len(nxt) > 1:
                return None
            prev, cur = cur, nxt[0]

    def _chain_atoms(start_idx, prev_idx):
        out = []
        prev, cur = prev_idx, start_idx
        while cur is not None:
            out.append(cur)
            nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                   if n.GetIdx() != prev and n.GetIdx() not in ring_atom_set]
            prev, cur = cur, (nxt[0] if nxt else None)
        return out

    prefixes: List[str] = []
    covered: Set[int] = set()
    atom = mol.GetAtomWithIdx(ring_atom_idx)
    for nbr in atom.GetNeighbors():
        ni = nbr.GetIdx()
        if ni in ring_atom_set or ni in skip_atoms:
            continue
        bond = mol.GetBondBetweenAtoms(ring_atom_idx, ni)
        sym = nbr.GetSymbol()
        order = bond.GetBondTypeAsDouble()
        charge = nbr.GetFormalCharge()
        # oxo (exocyclic =O)
        if (sym == 'O' and order == 2.0 and charge == 0
                and nbr.GetTotalNumHs() == 0 and nbr.GetDegree() == 1):
            prefixes.append('oxo')
            covered.add(ni)
            continue
        # halogen
        if sym in _HAL and order == 1.0 and nbr.GetDegree() == 1 and charge == 0:
            prefixes.append(_HAL[sym])
            covered.add(ni)
            continue
        # hydroxy / simple n-alkoxy
        if sym == 'O' and order == 1.0 and charge == 0:
            o_nbrs = [n for n in nbr.GetNeighbors() if n.GetIdx() != ring_atom_idx]
            if nbr.GetTotalNumHs() == 1 and not o_nbrs:
                prefixes.append('hydroxy')
                covered.add(ni)
                continue
            if len(o_nbrs) == 1 and o_nbrs[0].GetSymbol() == 'C':
                chain = _linear_carbon_chain(o_nbrs[0].GetIdx(), {ni} | ring_atom_set)
                if chain is not None and chain in _ALKOXY:
                    prefixes.append(_ALKOXY[chain])
                    covered.add(ni)
                    covered.update(_chain_atoms(o_nbrs[0].GetIdx(), ni))
                    continue
            return None
        # amino (-NH2)
        if (sym == 'N' and order == 1.0 and charge == 0
                and nbr.GetTotalNumHs() == 2 and nbr.GetDegree() == 1):
            prefixes.append('amino')
            covered.add(ni)
            continue
        # nitro (-[N+](=O)[O-])
        if sym == 'N' and charge == 1 and nbr.GetDegree() == 3:
            n_os = [n for n in nbr.GetNeighbors() if n.GetIdx() != ring_atom_idx]
            if (len(n_os) == 2
                    and all(x.GetSymbol() == 'O' and x.GetDegree() == 1 for x in n_os)
                    and sorted(x.GetFormalCharge() for x in n_os) == [-1, 0]):
                prefixes.append('nitro')
                covered.add(ni)
                covered.update(x.GetIdx() for x in n_os)
                continue
            return None
        # cyano (-C#N) or carboxy (-COOH) or unbranched pure alkyl
        if sym == 'C' and order == 1.0 and charge == 0:
            c_nbrs = [x for x in nbr.GetNeighbors() if x.GetIdx() != ring_atom_idx]
            if (len(c_nbrs) == 1 and c_nbrs[0].GetSymbol() == 'N'
                    and c_nbrs[0].GetDegree() == 1
                    and mol.GetBondBetweenAtoms(ni, c_nbrs[0].GetIdx())
                          .GetBondTypeAsDouble() == 3.0):
                prefixes.append('cyano')
                covered.add(ni)
                covered.add(c_nbrs[0].GetIdx())
                continue
            # carboxy (P-65.1.7.2.1): exocyclic C (single bond, no H) bearing a
            # terminal =O AND a hydroxyl -OH (second O carries an H). The H on the
            # -OH is the ester-exclusion guard: -C(=O)OR esters have 0 H on that O
            # and stay out of scope (alkoxycarbonyl). Mirrors the carboxy branch in
            # name_ring_system_substituent. BlueBookV2:1804 (-COOH), :5154/:3262.
            if (len(c_nbrs) == 2 and nbr.GetTotalNumHs() == 0
                    and all(x.GetSymbol() == 'O' for x in c_nbrs)):
                has_carbonyl_O = any(
                    x.GetDegree() == 1 and x.GetTotalNumHs() == 0
                    and x.GetFormalCharge() == 0
                    and mol.GetBondBetweenAtoms(ni, x.GetIdx()).GetBondTypeAsDouble() == 2.0
                    for x in c_nbrs)
                has_hydroxyl_O = any(
                    x.GetTotalNumHs() >= 1 and x.GetFormalCharge() == 0
                    and mol.GetBondBetweenAtoms(ni, x.GetIdx()).GetBondTypeAsDouble() == 1.0
                    for x in c_nbrs)
                if has_carbonyl_O and has_hydroxyl_O:
                    prefixes.append('carboxy')
                    covered.add(ni)
                    covered.update(x.GetIdx() for x in c_nbrs)
                    continue
                return None
            chain = _linear_carbon_chain(ni, ring_atom_set)
            if chain is not None and chain in _ALKYL:
                prefixes.append(_ALKYL[chain])
                covered.update(_chain_atoms(ni, ring_atom_idx))
                continue
            return None
        # anything else exocyclic -> unsupported
        return None
    return prefixes, covered


def decorated_ring_substituent_name(mol, ring_atoms, attachment_atom: int,
                                    expected_atoms: Optional[Set[int]] = None
                                    ) -> Optional[str]:
    """PIN substituent name for a MONOCYCLIC ring substituent carrying its own
    substituent prefixes: ``2-nitrothiophen-3-yl``, ``2-oxocyclohexyl``,
    ``3-chloro-2-methylphenyl``.

    When ``expected_atoms`` is given, the name is returned only if the ring
    plus the detected decoration atoms account for EXACTLY that set — callers
    naming a specific fragment use this so the emitted name can never cover
    more or less than the fragment.

    This is the WS-A.2 demoted-ring emitter: when parent selection demotes a
    decorated ring to a substituent, its decoration must be carried (the
    Phase-171 / rt75_0582 class of regression: a parent flip that silently
    drops the demoted ring's groups).

    Numbering per P-14.4 (BlueBookV2.md:3221), applied in order:
      1. ring heteroatoms low (set, then element seniority) — criterion (a);
      2. free valence (attachment) low — criterion (c), which OUTRANKS the
         detachable prefixes (cf. '6-carboxynaphthalen-2-yl', :3262);
      3. detachable-prefix locant set low (first point of difference) — (f);
      4. first-cited (alphabetically) prefix low — (g).

    Deliberately GUARDED — returns None (caller keeps its legacy form, so the
    change is zero-regression by construction) when:
      * the ring is not a simple monocycle (fused/spiro/bridged atoms);
      * the ring is an N-H azole or otherwise needs indicated hydrogen;
      * the ring has mixed saturation (would need ene-locants in the stem);
      * any ring atom carries an exocyclic group outside the supported table;
      * the ring carries no supported decoration at all (bare rings keep
        their existing forms — this function only DECORATES);
      * the stem is not confidently known.
    """
    ring_list = list(ring_atoms)
    n = len(ring_list)
    ring_set = set(ring_list)
    if attachment_atom not in ring_set or not (3 <= n <= 8):
        return None

    # --- simple monocycle: every ring atom has exactly 2 ring neighbours ----
    ring_adj: Dict[int, List[int]] = {}
    for idx in ring_list:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0:
            return None
        nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors()
                if nbr.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None
        # fused/spiro guard: ring atom must not belong to any other ring
        if mol.GetRingInfo().NumAtomRings(idx) != 1:
            return None
        ring_adj[idx] = nbrs

    # --- classify the ring & pick the stem ----------------------------------
    het = [i for i in ring_list if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list)
    saturated = all(
        not mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
        and mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1.0
        for a in ring_list for b in ring_adj[a] if a < b)

    if het:
        # N-H ring atoms (pyrrole-type) need indicated hydrogen -> guard out
        if any(mol.GetAtomWithIdx(i).GetSymbol() == 'N'
               and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0
               for i in het):
            return None
        if not aromatic and not saturated:
            return None  # partially unsaturated heterocycle: ene-locants needed
        ring_name = identify_ring_system(mol, tuple(ring_list))
        _HET_STEMS = {
            # aromatic (no indicated-H needed)
            'pyridine': 'pyridin', 'furan': 'furan', 'thiophene': 'thiophen',
            'pyrimidine': 'pyrimidin', 'pyrazine': 'pyrazin',
            'pyridazine': 'pyridazin',
            # saturated
            'oxane': 'oxan', 'oxolane': 'oxolan', 'piperidine': 'piperidin',
            'pyrrolidine': 'pyrrolidin', 'morpholine': 'morpholin',
        }
        stem = _HET_STEMS.get(ring_name)
        if stem is None:
            return None
    else:
        if aromatic and n == 6:
            stem = 'phenyl'
        elif saturated:
            stem = 'cyclo' + _get_chain_prefix(n) + 'yl'
        else:
            return None  # mixed-saturation carbocycle: ene-locants needed

    # --- per-atom substituent detection (whole-ring guard) ------------------
    parent_nbrs = {nbr.GetIdx() for nbr in
                   mol.GetAtomWithIdx(attachment_atom).GetNeighbors()
                   if nbr.GetIdx() not in ring_set}
    atom_prefixes: Dict[int, List[str]] = {}
    covered_all: Set[int] = set()
    for idx in ring_list:
        skip = parent_nbrs if idx == attachment_atom else set()
        res = _ring_atom_simple_substituents(mol, idx, ring_set, skip_atoms=skip)
        if res is None:
            return None
        atom_prefixes[idx] = res[0]
        covered_all |= res[1]

    if not any(atom_prefixes.values()):
        return None  # nothing to decorate: keep legacy form

    if expected_atoms is not None and (ring_set | covered_all) != set(expected_atoms):
        return None  # name would not cover exactly the requested fragment

    # --- enumerate numberings per P-14.4 -------------------------------------
    def _walk(start, second):
        order = {1: start}
        prev, cur = start, second
        pos = 2
        while pos <= n:
            order[pos] = cur
            nxt = [x for x in ring_adj[cur] if x != prev]
            prev, cur = cur, nxt[0]
            pos += 1
        return order

    candidates = []
    if het:
        for start in ring_list:
            for second in ring_adj[start]:
                candidates.append(_walk(start, second))
        # (1) heteroatom locant set low
        def het_locs(o):
            return tuple(sorted(p for p, i in o.items()
                                if mol.GetAtomWithIdx(i).GetSymbol() != 'C'))
        best = min(het_locs(o) for o in candidates)
        candidates = [o for o in candidates if het_locs(o) == best]
        # (1b) element seniority at the heteroatom positions (P-22.2.3.1);
        # lower rank = more senior = lower locant. v22 E1 / DD4 (IN-01/02):
        # routed through the single ELEMENT_NUMBERING_SENIORITY source of truth
        # (order-consistent with the old inline F,Cl,Br,I,O,S,Se,Te,N,P list —
        # behaviour-preserving) so the P-15.4.1.2 order has exactly one definition.
        from .locants import element_seniority_rank as _element_seniority_rank
        def het_rank_seq(o):
            return tuple(
                _element_seniority_rank(mol.GetAtomWithIdx(o[p]).GetSymbol())
                for p in sorted(o)
                if mol.GetAtomWithIdx(o[p]).GetSymbol() != 'C')
        best = min(het_rank_seq(o) for o in candidates)
        candidates = [o for o in candidates if het_rank_seq(o) == best]
        # (2) free valence low
        def att_loc(o):
            return next(p for p, i in o.items() if i == attachment_atom)
        best = min(att_loc(o) for o in candidates)
        candidates = [o for o in candidates if att_loc(o) == best]
        attachment_locant = best
    else:
        for second in ring_adj[attachment_atom]:
            candidates.append(_walk(attachment_atom, second))
        attachment_locant = 1

    # (3) prefix locant set low (first point of difference)
    def prefix_locs(o):
        return tuple(sorted(p for p, i in o.items() for _ in atom_prefixes[i]))
    best = min(prefix_locs(o) for o in candidates)
    candidates = [o for o in candidates if prefix_locs(o) == best]
    # (4) first-cited (alphabetical) prefix low
    def alpha_keyed(o):
        pairs = sorted(
            (name, p) for p, i in o.items() for name in atom_prefixes[i])
        return tuple(p for _name, p in pairs)
    best = min(alpha_keyed(o) for o in candidates)
    order = next(o for o in candidates if alpha_keyed(o) == best)

    # --- assemble -------------------------------------------------------------
    groups: Dict[str, List[int]] = {}
    for p, i in order.items():
        for name in atom_prefixes[i]:
            groups.setdefault(name, []).append(p)
    from ..assembly.naming_utils import get_multiplier_prefix
    parts = []
    for name in sorted(groups):  # alphabetical citation order
        locs = sorted(groups[name])
        loc_str = ','.join(str(loc) for loc in locs)
        mult = get_multiplier_prefix(len(locs), name) if len(locs) > 1 else ''
        parts.append(f'{loc_str}-{mult}{name}')
    prefix_str = '-'.join(parts)

    if het:
        return f'{prefix_str}{stem}-{attachment_locant}-yl'
    return f'{prefix_str}{stem}'


def ring_atom_fg_prefixes(
    mol, ring_atom_idx: int, ring_atom_set: Set[int],
    return_atoms: bool = False,
):
    """Characteristic-group prefixes carried by a single ring atom when its
    ring is demoted to a substituent.

    When a ring is named as a substituent of a chain parent, the ring's own
    characteristic groups must still appear as prefixes inside the enclosing
    marks (e.g. ``4-(2-oxocyclohexyl)butanoic acid``, not the FG-dropped
    ``4-cyclohexylbutanoic acid``). This is the shared chemistry primitive that
    every ring-as-substituent emitter consults so the rule is applied once and
    identically.

    Scope (the previously *dropped* characteristic groups):
      * ``oxo``  (P-66.6.1) — the ring atom is a carbonyl carbon: an exocyclic
        double bond to an oxygen that bears no H and is otherwise terminal.
      * ``cyano`` (P-66.5.1) — the ring atom bears an exocyclic nitrile carbon
        (single bond to a C that is triple-bonded to a terminal N).
      * ``carboxy`` (P-65.1.7.2.1) — the ring atom bears an exocyclic
        carboxylic-acid carbon (an exocyclic C with =O carrying no H AND -OH).
        The S2 parent chokepoint (commit 835faffa) now demotes the ring-acid
        parent, so this branch is reachable. Esters ``-C(=O)OR`` are EXCLUDED
        (the second O carries no H) and stay alkoxycarbonyl / out of scope.

    Intentionally NOT handled here:
      * ``hydroxy`` / alkoxy / halogen / amino / alkyl — already emitted by the
        existing composer / benzene substituent branches; claiming them here
        would double-count.
      * ester ``-C(=O)O-`` carbonyls — excluded by the carboxy hydroxyl-O guard;
        they belong to the alkoxycarbonyl / ester pathway, not here.

    Args:
        mol: RDKit Mol.
        ring_atom_idx: the ring atom to inspect.
        ring_atom_set: atom indices of the whole ring system (to identify
            which neighbours are exocyclic).

    Returns:
        Sorted list of prefix strings (``[]`` when the atom carries none).
        With ``return_atoms=True``, returns ``(prefixes, claimed_atoms)``
        where ``claimed_atoms`` is the set of exocyclic atom indices the
        emitted prefixes account for (Wave2 T3a conservation accounting —
        callers use it to prove every branch atom is represented in the name).
    """
    prefixes: List[str] = []
    claimed: Set[int] = set()
    atom = mol.GetAtomWithIdx(ring_atom_idx)
    for nbr in atom.GetNeighbors():
        ni = nbr.GetIdx()
        if ni in ring_atom_set:
            continue
        bond = mol.GetBondBetweenAtoms(ring_atom_idx, ni)
        sym = nbr.GetSymbol()
        # oxo: exocyclic =O, terminal, no H (the ring atom is the carbonyl C)
        if (sym == 'O' and bond is not None
                and bond.GetBondTypeAsDouble() == 2.0
                and nbr.GetTotalNumHs() == 0
                and nbr.GetDegree() == 1):
            prefixes.append('oxo')
            claimed.add(ni)
            continue
        # cyano: exocyclic C, single bond, triple-bonded to a terminal N
        if (sym == 'C' and bond is not None
                and bond.GetBondTypeAsDouble() == 1.0):
            c_nbrs = [x for x in nbr.GetNeighbors() if x.GetIdx() != ring_atom_idx]
            if (len(c_nbrs) == 1 and c_nbrs[0].GetSymbol() == 'N'
                    and c_nbrs[0].GetDegree() == 1):
                cn_bond = mol.GetBondBetweenAtoms(ni, c_nbrs[0].GetIdx())
                if cn_bond is not None and cn_bond.GetBondTypeAsDouble() == 3.0:
                    prefixes.append('cyano')
                    claimed.add(ni)
                    claimed.add(c_nbrs[0].GetIdx())
                    continue
        # carboxy (P-65.1.7.2.1): the ring atom bears an exocyclic carboxylic-acid
        # carbon — an exocyclic C (single bond) that carries =O (terminal, no H) AND
        # -OH (the second O carries an H). The H on the second O is the ester-exclusion
        # guard: -C(=O)OR esters have GetTotalNumHs()==0 on that O and must stay
        # alkoxycarbonyl / out of scope. Lactone/anhydride topologies that slip past
        # here are caught by the None-return of decorated_ring_substituent_name (the
        # carry-all-or-None safety net). BlueBookV2:1804 (-COOH), :5154/:3262 (carboxy).
        if (sym == 'C' and bond is not None
                and bond.GetBondTypeAsDouble() == 1.0):
            c_nbrs = [x for x in nbr.GetNeighbors() if x.GetIdx() != ring_atom_idx]
            has_carbonyl_O = any(
                x.GetSymbol() == 'O' and x.GetDegree() == 1 and x.GetTotalNumHs() == 0
                and mol.GetBondBetweenAtoms(ni, x.GetIdx()).GetBondTypeAsDouble() == 2.0
                for x in c_nbrs)
            has_hydroxyl_O = any(
                x.GetSymbol() == 'O' and x.GetTotalNumHs() >= 1
                and mol.GetBondBetweenAtoms(ni, x.GetIdx()).GetBondTypeAsDouble() == 1.0
                for x in c_nbrs)
            if has_carbonyl_O and has_hydroxyl_O:
                prefixes.append('carboxy')
                claimed.add(ni)
                claimed.update(
                    x.GetIdx() for x in c_nbrs if x.GetSymbol() == 'O'
                )
                continue
    if return_atoms:
        return sorted(prefixes), claimed
    return sorted(prefixes)
