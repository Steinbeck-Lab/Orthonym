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
            elif heteroatoms == ['N', 'N', 'N']:
                # P-25.2.1 retained triazoles. 1,2,3-triazole has a MIDDLE N
                # bonded to two ring N's (the N1-N2-N3 run); 1,2,4-triazole has
                # no N with two N ring-neighbours (N1-N2-C3-N4-C5). The stem
                # itself encodes the heteroatom positions, so the free-valence
                # numbering cascade (which minimises the heteroatom locant set)
                # must and does reproduce {1,2,3} / {1,2,4}.
                _has_nnn = any(
                    mol.GetAtomWithIdx(i).GetSymbol() == 'N'
                    and sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                            if nb.GetIdx() in ring_set
                            and nb.GetSymbol() == 'N') == 2
                    for i in ring_atoms)
                return '1,2,3-triazole' if _has_nnn else '1,2,4-triazole'
            elif heteroatoms == ['N', 'N', 'N', 'N']:
                return 'tetrazole'
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
    # P-25.2.1 retained azole substituent stems. identify_ring_system
    # distinguishes 1,2,3- vs 1,2,4-triazole by N-adjacency; the free-valence
    # numbering cascade minimises the heteroatom locant set, reproducing the
    # {1,2,3}/{1,2,4}/{1,2,3,4} positions the stem name asserts.
    '1,2,3-triazole': '1,2,3-triazol',
    '1,2,4-triazole': '1,2,4-triazol',
    'tetrazole': 'tetrazol',
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

    # --- imidazole vs pyrazole (R-bug fix) -----------------------------------
    # identify_ring_system() reports BOTH 5-membered N,N arenes as 'imidazole'.
    # Genuine imidazole has its two ring N's NON-adjacent (1,3); pyrazole has
    # them adjacent (1,2). Retarget the adjacent-N case to the retained pyrazole
    # stem (P-25.2.1) so a bare pyrazolyl names correctly instead of emitting an
    # imidazol-*-yl name for a pyrazole (was: return None -> misid downstream).
    if ring_name == 'imidazole':
        n_idx = [i for i in het_atoms
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) != 2:
            return None
        if mol.GetBondBetweenAtoms(n_idx[0], n_idx[1]) is not None:
            stem = 'pyrazol'

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


_DECO_MULT: Dict[int, str] = {
    1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
}


def _decorated_heteroaryl_substituent_name(
    mol,
    frag_atoms,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
    allow_mancude: bool = False,
) -> Optional[str]:
    """BP-3 cluster R: PIN substituent name for a monocyclic heteroaryl ring that
    carries its OWN decorations, with the free valence on a ring atom — e.g.
    ``5-hydroxy-1,3-dimethylpyrazol-4-yl``.

    Extends the ``pin_heteroaryl_substituent_name`` free-valence numbering cascade
    (P-31.1.4.3.4: heteroatoms as a set → element seniority → indicated H → free
    valence) with a final P-59.2.3 lowest-decoration-locant term, then reads the
    decorations off the winning numbering. Decorations are identified with the
    same ``_identify_fused_substituent`` machinery the fused-ring path uses, so a
    branch it cannot name → ``None`` (fail closed; the caller keeps the whole
    name fail-closed rather than dropping a substituent).

    Aromatic monocycle only for this slice; anything else returns None.
    """
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)
    # A ring DECORATION (e.g. an N-phenyl on a triazole) is a separate ring the
    # caller lumps into ``ring_atoms`` (every ring atom of the fragment). This
    # producer names ONE monocyclic core; restrict to the ring bearing the free
    # valence so the decoration ring is instead cited as a recursive substituent
    # (its atoms land in ``exo`` and are covered by the decoration pass below).
    _ri_core = [r for r in mol.GetRingInfo().AtomRings()
                if attachment_atom in r]
    if len(_ri_core) == 1 and ring_set != set(_ri_core[0]):
        ring_list = list(_ri_core[0])
        ring_set = set(ring_list)
    n = len(ring_list)
    frag_set = set(frag_atoms)
    if attachment_atom not in ring_set or n < 3:
        return None

    ring_name = identify_ring_system(mol, tuple(ring_list))
    stem = _PIN_HETEROARYL_STEMS.get(ring_name)

    het_atoms = [i for i in ring_list if mol.GetAtomWithIdx(i).GetSymbol() != 'C']

    # v29 Phase 6: CARBOCYCLIC (benzene) rings have no heteroaryl stem, so a
    # DECORATED PHENYL substituent had no producer at all and fell straight
    # through to DROP-24 — even though it is among the commonest shapes in
    # drug-like space. Measured on the Phase 5 corpus, every one of these was
    # unnameable at EVERY ring attachment point:
    #     CN(C)c1ccccc1      -> 4-(dimethylamino)phenyl
    #     NS(=O)(=O)c1ccccc1 -> 4-sulfamoylphenyl
    #
    # P-29.6.1 "Retained prefixes that are preferred prefixes" (`:16270`) — the
    # free valence of a benzene substituent is position 1 and its locant is NOT
    # cited; the retained preferred prefix is 'phenyl', never 'benzen-1-yl'.
    # The verbatim BB example is `4-methylphenyl (preferred prefix)` (`:16298`).
    # Decorations then take the lowest locants. The numbering
    # cascade below already produces exactly that for a carbocycle: with no
    # heteroatoms and no indicated H the sort key degenerates to
    # (fv_loc, deco_locs), so the free valence wins position 1 and the
    # decorations are minimised against it. Only the CORE spelling differs.
    is_carbocyclic = (ring_name == 'benzene' and not het_atoms)

    # Aromatic simple monocycle: each ring atom has exactly two in-ring neighbours.
    if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list):
        return None
    ring_adj: Dict[int, List[int]] = {}
    for idx in ring_list:
        nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nb.GetIdx() in ring_set]
        if len(nbrs) != 2:
            return None  # fusion / spiro / not a simple monocycle
        ring_adj[idx] = nbrs

    # Pyrazole vs imidazole (R-bug): identify_ring_system reports BOTH N,N 5-rings
    # as 'imidazole'; adjacent ring N's = pyrazole (retained stem, P-25.2.1).
    if ring_name == 'imidazole':
        n_idx = [i for i in het_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) == 2 and mol.GetBondBetweenAtoms(n_idx[0], n_idx[1]) is not None:
            stem = 'pyrazol'
    if stem is None and not is_carbocyclic:
        return None

    # Indicated hydrogen: a ring NH (only when an actual H is present — an
    # N-substituted position carries no indicated H and no nH prefix).
    ih_candidates = [i for i in het_atoms
                     if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1]
    if len(ih_candidates) > 1:
        return None  # ambiguous indicated H; do not guess
    indicated_h_atom = ih_candidates[0] if ih_candidates else None

    # Collect + identify decorations (heavy exocyclic atoms in the fragment that
    # hang off a ring atom). The attachment atom's parent bond leaves the
    # fragment and is NOT a decoration.
    from .fused_rings import _identify_fused_substituent
    decorations = []  # (ring_atom_idx, substituent_name)
    accounted: Set[int] = set()
    has_parent_bond = False
    for ra in ring_list:
        for nb in mol.GetAtomWithIdx(ra).GetNeighbors():
            ni = nb.GetIdx()
            if ni in ring_set or nb.GetAtomicNum() <= 1:
                continue
            if ni not in frag_set:
                # Bond leaving the fragment — only legal at the attachment atom.
                if ra != attachment_atom:
                    return None
                has_parent_bond = True
                continue
            info = _identify_fused_substituent(mol, ni, ring_set)
            if info is None:
                # v30 tail: a decoration that carries its OWN ring through a
                # chain/hetero linker ((4-chlorophenoxy)methyl on a triazole) is
                # declined by _identify_fused_substituent, which does not thread
                # the best-effort ring-on-chain path. Fall back to the universal
                # substituent namer (allow_mancude), collecting the decoration
                # subgraph off this ring atom. Fail closed if it too declines.
                from ..assembly.substituent_enumerator import name_substituent
                from ..errors import is_refusal_sentinel
                _deco: Set[int] = set()
                _st = [ni]
                while _st:
                    _x = _st.pop()
                    if _x in _deco or _x in ring_set:
                        continue
                    _deco.add(_x)
                    for _n in mol.GetAtomWithIdx(_x).GetNeighbors():
                        if _n.GetIdx() not in ring_set and _n.GetIdx() in frag_set:
                            _st.append(_n.GetIdx())
                _nm2 = name_substituent(mol, sorted(_deco), ni,
                                        allow_mancude=allow_mancude)
                if (not _nm2 or is_refusal_sentinel(_nm2)
                        or _nm2 == 'substituent' or ' ' in _nm2):
                    return None
                decorations.append((ra, _nm2))
                accounted.update(
                    a for a in _deco
                    if mol.GetAtomWithIdx(a).GetAtomicNum() > 1)
                continue
            nm = info.get('name')
            if not nm or ' ' in nm:
                return None
            decorations.append((ra, nm))
            accounted.update(
                a for a in info.get('atoms', [])
                if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
            )
    if not has_parent_bond:
        return None
    if not decorations:
        return None  # bare ring -> pin_heteroaryl_substituent_name handles it
    exo = {a for a in frag_set
           if a not in ring_set and mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    if accounted != exo:
        return None  # a heavy atom unaccounted -> fail closed

    # Cyclic atom order.
    order = [ring_list[0], ring_adj[ring_list[0]][0]]
    while len(order) < n:
        prev, curr = order[-2], order[-1]
        nxt = [x for x in ring_adj[curr] if x != prev]
        if not nxt:
            return None
        order.append(nxt[0])
    if len(order) != n:
        return None

    from ..assembly.naming_utils import (
        alpha_sort_key as _alpha,
        enclose_if_compound as _enclose_if_compound,
    )
    deco_atoms = [ra for ra, _ in decorations]
    best_key = None
    best_pos = None
    for start in range(n):
        for direction in (1, -1):
            atom_to_pos = {order[(start + direction * p) % n]: p + 1
                           for p in range(n)}
            het_locs = tuple(sorted(atom_to_pos[i] for i in het_atoms))
            seniority_locs = tuple(
                atom_to_pos[i] for i in sorted(
                    het_atoms,
                    key=lambda a: (_HETEROATOM_SENIORITY.get(
                        mol.GetAtomWithIdx(a).GetSymbol(), 99), atom_to_pos[a]),
                )
            )
            ih_loc = (atom_to_pos[indicated_h_atom]
                      if indicated_h_atom is not None else 0)
            fv_loc = atom_to_pos[attachment_atom]
            deco_locs = tuple(sorted(atom_to_pos[ra] for ra in deco_atoms))
            # P-14.4(g): when every earlier criterion ties, the lower locant
            # goes to the substituent cited FIRST in alphanumerical order.
            # Without this term the tie was broken by whichever ring-traversal
            # direction happened to be enumerated first, which depends on RDKit
            # neighbour order — so the SAME molecule written two ways got two
            # names: `Brc1cccc(Cl)c1` -> 2-bromo-6-chlorophenyl (correct) but
            # `Clc1cccc(Br)c1` -> 6-bromo-2-chlorophenyl. Nondeterministic AND
            # non-PIN. Comparing (locant, alpha-key) pairs in locant order
            # resolves it: ((2,'bromo'),(6,'chloro')) < ((2,'chloro'),(6,'bromo')).
            deco_named = tuple(
                (atom_to_pos[ra], _alpha(nm))
                for ra, nm in sorted(decorations,
                                     key=lambda t: atom_to_pos[t[0]])
            )
            key = (het_locs, seniority_locs, ih_loc, fv_loc, deco_locs,
                   deco_named)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = atom_to_pos
    if best_pos is None:
        return None

    attach_locant = best_pos[attachment_atom]
    ih_locant = (best_pos[indicated_h_atom]
                 if indicated_h_atom is not None else None)

    # Group decorations by name; assemble alphabetised, multiplied prefixes.
    from collections import defaultdict as _dd
    groups = _dd(list)
    for ra, nm in decorations:
        groups[nm].append(best_pos[ra])
    prefix_parts = []  # (alpha_key, text)
    for nm, locs in groups.items():
        locs = sorted(locs)
        mult = _DECO_MULT.get(len(locs))
        if mult is None:
            return None
        # Enclosing marks. P-16.3.4 "Parentheses (round brackets) ... are used
        # to enclose multiplied components that are: ... (c) simple substituent
        # prefixes and functionalized parent hydrides beginning with a
        # multiplicative prefix" (`:7085`), and P-16.3.5(a) covers "compound or
        # complex (i.e. substituted) prefixes" with the verbatim example
        # `bis(dimethylamino) (preferred prefix)` (`:7104`). So
        # `4-dimethylaminophenyl` — which can read as di(methylamino) — must be
        # `4-(dimethylamino)phenyl`. `enclose_if_compound` is the shared
        # primitive for this — it unions needs_brackets with
        # is_complex_substituent (neither is complete alone), escalates
        # ( -> [ -> { for an already-bracketed inner name, leaves simple
        # prefixes bare, and is idempotent.
        text = f"{','.join(str(l) for l in locs)}-{mult}{_enclose_if_compound(nm)}"
        prefix_parts.append((_alpha(nm), text))
    prefix_parts.sort(key=lambda x: x[0])
    body = '-'.join(p[1] for p in prefix_parts)
    ih = f"{ih_locant}H-" if indicated_h_atom is not None else ""
    if is_carbocyclic:
        # P-29.3.5: 'phenyl', with the free valence at 1 and its locant elided.
        # The guard is not decorative — if the cascade ever failed to place the
        # free valence at 1 the elided locant would name a DIFFERENT molecule,
        # so fail closed rather than emit an unlocanted prefix.
        if attach_locant != 1 or indicated_h_atom is not None:
            return None
        core = "phenyl"
    else:
        core = f"{ih}{stem}-{attach_locant}-yl"
    # Hyphen between the last prefix and the stem only when the stem/indicated-H
    # tail is digit-initial (e.g. '-1H-pyrazol...'); elide before a letter-initial
    # stem ('dimethylpyrazol...').
    sep = '-' if core[0].isdigit() else ''
    return f"{body}{sep}{core}"


def _decorated_fused_substituent_name(
    mol,
    frag_atoms,
    ring_atoms: Tuple[int, ...],
    attachment_atom: int,
) -> Optional[str]:
    """v30 Piece 2: PIN substituent name for a FUSED ring system carrying its
    OWN decorations, free valence on a ring atom — e.g. ``6-methoxynaphthalen-2-yl``.

    The monocyclic sibling ``_decorated_heteroaryl_substituent_name`` declines
    the moment the ring system is fused (a ring atom with != 2 in-ring
    neighbours, ``:574``). The bare fused substituent (``naphthalen-2-yl``)
    already names, so the only gap was the DECORATED fused case — which fell
    through to a silent atom drop.

    Reuses the fused-ring PARENT numbering, not a per-atom locant: the naive
    ``_get_polycyclic_attachment_locant`` optimises each atom independently and
    returns 2 for BOTH the attach and the methoxy carbon of 6-methoxynaphthalene
    (inconsistent). Instead ``compute_fused_numbering`` gives ONE canonical
    atom->locant map and ``_ring_system_automorphisms`` gives the residual ring
    symmetry; among the symmetry-equivalent numberings we take the one giving
    the FREE VALENCE the lowest locant (P-31.1.4.3.4 free-valence priority),
    then the decorations the lowest set (P-59.2.3), and read every locant off
    that single numbering so they are mutually consistent.

    Fail closed (None) on anything not provably correct — an unnameable or
    unaccounted decoration, a system ``compute_fused_numbering`` declines, a
    ring with no retained fused stem. SELF-01 then backstops a mis-numbering
    (abstain, never a wrong molecule).
    """
    from rdkit import Chem
    from .fusion_numbering import compute_fused_numbering
    from .fused_rings import (
        _ring_system_automorphisms,
        _identify_fused_substituent,
        _fused_locant_to_output,
        _fused_locant_num,
    )
    from ..assembly.naming_utils import (
        alpha_sort_key as _alpha,
        enclose_if_compound as _enclose_if_compound,
    )

    ring_set = set(ring_atoms)
    frag_set = set(frag_atoms)
    if attachment_atom not in ring_set or len(ring_set) < 6:
        return None

    # Retained fused stem (naphthalene -> 'naphthalen'). Same derivation as
    # get_ring_substituent_name's bare-fused branch, so the stems agree.
    frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
    if not frag_smi:
        return None
    from ..data import get_retained_name
    retained = get_retained_name(frag_smi)
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
        _fh = FUSED_HETEROCYCLE_DATA.get(frag_smi)
        if _fh and _fh.get('name'):
            retained = _fh['name']
    except ImportError:
        pass
    if not retained:
        return None
    stem = retained[:-1] if retained.endswith('e') else retained

    # One canonical numbering + the ring system's residual symmetry.
    canonical = compute_fused_numbering(mol, ring_set)
    if not canonical:
        return None
    perms = _ring_system_automorphisms(mol, ring_set)

    # Collect + identify decorations (heavy exocyclic atoms hanging off a ring
    # atom). The attachment atom's parent bond leaves the fragment and is NOT a
    # decoration. Mirrors _decorated_heteroaryl_substituent_name's collection.
    decorations = []  # (ring_atom_idx, substituent_name)
    accounted: Set[int] = set()
    has_parent_bond = False
    for ra in ring_atoms:
        for nb in mol.GetAtomWithIdx(ra).GetNeighbors():
            ni = nb.GetIdx()
            if ni in ring_set or nb.GetAtomicNum() <= 1:
                continue
            if ni not in frag_set:
                if ra != attachment_atom:
                    return None
                has_parent_bond = True
                continue
            info = _identify_fused_substituent(mol, ni, ring_set)
            if info is None:
                return None
            nm = info.get('name')
            if not nm or ' ' in nm:
                return None
            decorations.append((ra, nm))
            accounted.update(
                a for a in info.get('atoms', [])
                if mol.GetAtomWithIdx(a).GetAtomicNum() > 1
            )
    if not has_parent_bond:
        return None
    if not decorations:
        return None  # bare fused ring -> get_ring_substituent_name handles it
    exo = {a for a in frag_set
           if a not in ring_set and mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    if accounted != exo:
        return None  # a heavy atom unaccounted -> fail closed

    # Enumerate symmetry-equivalent numberings; pick lowest FREE VALENCE first
    # (P-31.1.4.3.4), then lowest decoration set (P-59.2.3), then the alpha tie.
    deco_atoms = [ra for ra, _ in decorations]
    best_key = None
    best_pos = None
    for perm in perms:
        try:
            cand = {a: _fused_locant_to_output(canonical[perm[a]])
                    for a in ring_set}
        except KeyError:
            continue
        fv = cand.get(attachment_atom)
        if not isinstance(fv, int):
            continue  # free valence cannot sit on a fusion (letter) atom
        deco_locs = tuple(sorted(_fused_locant_num(cand[ra]) for ra in deco_atoms))
        deco_named = tuple(
            (_fused_locant_num(cand[ra]), _alpha(nm))
            for ra, nm in sorted(decorations,
                                 key=lambda t: _fused_locant_num(cand[t[0]]))
        )
        key = (fv, deco_locs, deco_named)
        if best_key is None or key < best_key:
            best_key = key
            best_pos = cand
    if best_pos is None:
        return None

    attach_locant = best_pos[attachment_atom]

    # Group decorations by name; assemble alphabetised, multiplied, enclosed
    # prefixes (shared primitive, same as the monocyclic path).
    from collections import defaultdict as _dd
    groups = _dd(list)
    for ra, nm in decorations:
        groups[nm].append(_fused_locant_num(best_pos[ra])[0])
    prefix_parts = []  # (alpha_key, text)
    for nm, locs in groups.items():
        locs = sorted(locs)
        mult = _DECO_MULT.get(len(locs))
        if mult is None:
            return None
        text = f"{','.join(str(l) for l in locs)}-{mult}{_enclose_if_compound(nm)}"
        prefix_parts.append((_alpha(nm), text))
    prefix_parts.sort(key=lambda x: x[0])
    body = '-'.join(p[1] for p in prefix_parts)
    # The prefix body ends in a letter (or a closing bracket) and the retained
    # fused stem is letter-initial, so they abut with no hyphen
    # ('...methoxynaphthalen...') — the same rule the monocyclic sibling applies
    # via its ``sep`` term. The free-valence locant then follows with a hyphen.
    return f"{body}{stem}-{attach_locant}-yl"


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
            l1, l2 = numbering[a1], numbering[a2]
            # v27 P1: the consecutive-locant `-n-ene` infix can only express a
            # double bond between adjacently-numbered atoms. A bridgehead/bridge
            # ene (non-consecutive locants, e.g. 8=15) needs the `-n(m)-ene`
            # form, which this narrow namer does not build -- it used to cite the
            # lower locant alone (`-8-ene`), an OPSIN-unparseable / ambiguous name
            # that never round-tripped (SELF-01 then suppressed it to unknown).
            # Fail closed so the universal von-Baeyer cage namer (which cites the
            # `n(m)` form correctly, under the complete tier) takes over.
            if abs(l1 - l2) != 1:
                return None
            ene.append(min(l1, l2))
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


#: v30 P3-T1c lead 1: the von Baeyer parent hydrides whose RETAINED name is the
#: PIN, keyed by the descriptor string the cage analyzer emits.
#:
#: **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"**
#: (``BlueBookV2/BlueBookV2.md:9879``): *"The retained names adamantane and cubane
#: are used in general nomenclature and as preferred IUPAC names. The name
#: quinuclidine is retained for general nomenclature only (see Table 2.6)."*
#: Table 2.6 (``:9885``) prints *"adamantane (PIN) tricyclo[3.3.1.1^3,7]decane"*
#: and *"cubane (PIN) pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane"* — so the
#: retained name is the PIN and the von Baeyer descriptor is the ALTERNATIVE,
#: which is the direction this table restores.
#:
#: ⚠ Only these two. ``tricyclo.TRICYCLO_RETAINED_NAMES`` also carries
#: ``twistane``, which Table 2.6 does NOT retain, and ``quinuclidine``, which
#: Table 2.6 retains for general nomenclature ONLY. Neither may be used as a PIN
#: stem, so this table is keyed off P-23.7 rather than reusing that dict.
#:
#: The key is the DESCRIPTOR, not a SMILES: by the time it is consulted the
#: descriptor has already passed ``audit_von_baeyer_descriptor``, so it is proven
#: to denote this cage — a canonical-SMILES compare proves nothing of the kind.
#:
#: Locant safety is PROVEN, not assumed: OPSIN 2.9.0 resolves
#: ``tricyclo[3.3.1.1^3,7]decan-N-ol`` and ``adamantan-N-ol`` to the same
#: InChIKey for every N in 1..10, and the cubane pair for every N in 1..5 —
#: 15/15. So the retained stem can take the analyzer's own von Baeyer locants
#: unchanged.
_VB_RETAINED_PIN_STEM = {
    'tricyclo[3.3.1.1^3,7]': ('decan', 'adamantan'),
    'pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]': ('octan', 'cuban'),
}


def _retained_pin_cage_stem(cage, stem: str) -> Optional[str]:
    """The P-23.7 retained PIN stem for ``cage``, or None.

    Applies only to an all-carbon, fully saturated cage whose audited descriptor
    is in ``_VB_RETAINED_PIN_STEM``: a heteroatom or a ring multiple bond makes
    it a different parent hydride, for which Table 2.6 asserts nothing.
    """
    entry = _VB_RETAINED_PIN_STEM.get(cage.descriptor)
    if entry is None:
        return None
    expected_stem, retained = entry
    if stem != expected_stem:
        return None
    if cage.hetero_prefix:
        return None
    unsat = cage.unsaturation or {}
    if unsat.get('double_bonds') or unsat.get('triple_bonds'):
        return None
    return retained


def _lowest_locant_cage_numbering(sub, cage, cage_atoms, attach_sub,
                                  deco_carriers_sub=()):
    """Re-map ``cage.atom_to_locant`` through the cage's own AUTOMORPHISMS to give
    the free valence (P-29.3.2) and then the substituents the lowest locants.

    Returns a ``{sub_idx: locant}`` dict, never None — the analyzer's own
    numbering is the fallback, so this can only improve the locant set.

    Why an automorphism and not a re-derivation: a graph automorphism preserves
    adjacency, so the relabelled numbering asserts the SAME bond set in locant
    space and the descriptor still describes the cage. That is re-proven, not
    assumed — every candidate is put back through
    ``audit_von_baeyer_descriptor`` and a candidate that fails is discarded.

    ⚠ Deliberately applied ONLY on the retained-PIN path (adamantane / cubane).
    Applying it to every von Baeyer substituent would relabel names that already
    ship. That generalisation is P-29.3.3, which
    ``_universal_cage_substituent_name``'s own docstring already defers.
    """
    from rdkit import Chem
    from .vonbaeyer_universal import audit_von_baeyer_descriptor
    base = {i: cage.atom_to_locant[i] for i in cage_atoms}
    best_key = (base[attach_sub],
                tuple(sorted(base[a] for a in deco_carriers_sub
                             if a in base)))
    best = base
    try:
        autos = sub.GetSubstructMatches(sub, uniquify=False, maxMatches=256)
    except Exception:  # noqa: BLE001 - a symmetry search must never break naming
        return best
    kek = Chem.RWMol(sub)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
        kekmol = kek.GetMol()
    except Exception:  # noqa: BLE001
        kekmol = sub
    for sigma in autos:
        try:
            cand = {i: base[sigma[i]] for i in cage_atoms}
        except (KeyError, IndexError):
            continue
        if sorted(cand.values()) != sorted(base.values()):
            continue
        key = (cand[attach_sub],
               tuple(sorted(cand[a] for a in deco_carriers_sub if a in cand)))
        if key >= best_key:
            continue
        if not audit_von_baeyer_descriptor(
                kekmol, set(cage_atoms), cand, cage.descriptor):
            continue
        best_key, best = key, cand
    return best


def _universal_cage_substituent_name(
    sub, attach_sub, allow_mancude: bool = False
) -> Optional[str]:
    """v27 P1 (BB P-29.2 / P-29.3.3-5): name ANY von-Baeyer cage as a
    ``...-<loc>-yl`` substituent by routing the detached ring system through the
    SAME audited ``analyze_cage_universal`` engine that ``name_general_ring``
    uses for the PARENT, then citing the free valence's P-23 locant.

    Restores parent<->substituent symmetry: the narrow ``_vonbaeyer_substituent_name``
    only handles a 2-bridgehead ``is_bicyclo_system`` carbocycle, so tricyclo+/
    adamantane cages (and, under ``allow_mancude``, aromatic/mancude fused
    systems emitted as kekulized von-Baeyer polyenes) previously failed closed.

    ``sub`` is the detached ring-only submol (from ``_extract_ring_submol``);
    ``attach_sub`` is the free-valence atom index within it. Returns None on any
    refusal (spiro / >40 atoms / kekulize failure / mancude-with-flag-off /
    numbering that does not cover exactly the ring atoms) — fail-closed, never a
    wrong or coverage-incomplete cage. The free-valence locant is the primary
    P-23 numbering ``analyze_cage_universal`` assigns (a valid numbering ->
    SELF-01-safe); P-29.3.3 free-valence-lowest refinement is Task 5.
    """
    from .vonbaeyer_universal import analyze_cage_universal
    from .polycyclic import _build_parent_with_unsaturation
    try:
        cage_atoms = [a.GetIdx() for a in sub.GetAtoms() if a.IsInRing()]
        if not cage_atoms or attach_sub not in cage_atoms:
            return None
        cage = analyze_cage_universal(
            sub, cage_atoms=set(cage_atoms), allow_mancude=allow_mancude)
        if cage is None:
            return None
        # Coverage floor: the audited numbering must cover EXACTLY the ring
        # atoms (never drop/add one) — a cage `-yl` that omits a ring atom would
        # be a different constitution. analyze_cage_universal already runs its
        # own skeleton edge-audit; this is the substituent-side belt.
        if set(cage.atom_to_locant) != set(cage_atoms):
            return None
        loc = cage.atom_to_locant.get(attach_sub)
        if loc is None:
            return None
        parent_block = _build_parent_with_unsaturation(
            cage.total_atoms, cage.unsaturation)
        if not parent_block:
            return None
        # P-29.2 suffix elision: elide the terminal 'e' of the parent stem
        # before '-<loc>-yl' (octane -> octan-3-yl; pentadec-1(15)-ene ->
        # pentadec-1(15)-en-8-yl). _build_parent_with_unsaturation always
        # returns an 'e'-terminal stem (…ane / …ene / …yne / …diene).
        stem = parent_block[:-1] if parent_block.endswith('e') else parent_block
        # v30 P3-T1c lead 1: P-23.7 retained PIN stem in place of the von Baeyer
        # descriptor. Until now this function emitted `tricyclo[3.3.1.1^3,7]
        # decan-N-yl` for a cage whose PIN stem is `adamantan-`, so the ONE thing
        # this project claims over -- a retained name where
        # a retained name exists -- was missing on the substituent side while the
        # parent side already had it (`tricyclo.get_retained_tricyclo_name`).
        _retained = _retained_pin_cage_stem(cage, stem)
        if _retained is not None:
            _numbering = _lowest_locant_cage_numbering(
                sub, cage, cage_atoms, attach_sub)
            _loc = _numbering.get(attach_sub, loc)
            return f'{_retained}-{_loc}-yl'
        return f'{cage.hetero_prefix}{cage.descriptor}{stem}-{loc}-yl'
    except Exception:
        return None


def _universal_spiro_substituent_name(
    sub, attach_sub, allow_mancude: bool = False
) -> Optional[str]:
    """v27 P3 (BB P-24.2 + P-29.3): name ANY spiro ring system as a
    ``...-<loc>-yl`` substituent by routing the detached ring system through the
    audited ``analyze_spiro_universal`` engine, then citing the free valence's
    spiro locant.

    Generalizes the narrow ``_spiro_substituent_name`` (carbocyclic monospiro
    only) to HETERO spiro (``2-oxaspiro[4.5]decan-8-yl``), POLYSPIRO
    (``dispiro[...]-yl``) and ring-unsaturated spiro. The free valence gets the
    lowest locant AFTER the heteroatoms (P-24.2.4 fixes the heteroatom locants
    first; ``get_spiro_numbering`` orders het < free-valence). Returns None on
    any refusal — fail-closed, never a mis-numbered or coverage-incomplete `-yl`.
    """
    from .vonbaeyer_universal import analyze_spiro_universal
    from .polycyclic import _build_parent_with_unsaturation
    try:
        ring_atoms = [a.GetIdx() for a in sub.GetAtoms() if a.IsInRing()]
        if not ring_atoms or attach_sub not in ring_atoms:
            return None
        sp = analyze_spiro_universal(
            sub, cage_atoms=None, allow_mancude=allow_mancude,
            free_valence_atoms={attach_sub})
        if sp is None:
            return None
        # coverage floor: numbering must cover EXACTLY the ring atoms
        if set(sp.atom_to_locant) != set(ring_atoms):
            return None
        loc = sp.atom_to_locant.get(attach_sub)
        if loc is None:
            return None
        parent_block = _build_parent_with_unsaturation(
            sp.total_atoms, sp.unsaturation)
        if not parent_block:
            return None
        stem = parent_block[:-1] if parent_block.endswith('e') else parent_block
        return f'{sp.hetero_prefix}{sp.descriptor}{stem}-{loc}-yl'
    except Exception:
        return None


def _polycyclic_substituent_name(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: Optional[int],
    allow_mancude: bool = False,
) -> Optional[str]:
    """Name a MULTI-RING substituent (von-Baeyer, spiro, or partially hydro fused
    carbocycle) by routing the detached ring system through the existing parent
    namers with free-valence numbering. Returns the ``...-<loc>-yl`` name, or None
    (fail-closed) for any polycyclic the routed namers cannot number — never a
    monocycle size-guess (the old ``cyclo{N}yl`` corruption).

    v27 P1: when ``allow_mancude`` is set (complete/best-effort engine tier only)
    the universal cage namer is appended AFTER the narrow PIN namers, so their
    existing emissions are untouched and the new capability only ADDS coverage
    (tricyclo+/adamantane, and mancude fused-aromatic cages as polyenes). Default
    False keeps the PIN path byte-identical."""
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
    if allow_mancude:
        # v27 P1/P3: the universal cage + spiro engines (tricyclo+/adamantane +
        # mancude cages; hetero/poly/unsaturated spiro), complete-tier only.
        # Fail-closed (None) on any refusal.
        for _uni in (_universal_cage_substituent_name,
                     _universal_spiro_substituent_name):
            try:
                name = _uni(sub, attach_sub, allow_mancude=allow_mancude)
            except Exception:
                name = None
            if name:
                return name
    return None


def _pah_core_numbering(mol, ring_atoms, attach, deco_carriers):
    """v28 Task 2b helper: full ``{orig_idx: int_locant}`` numbering + bare
    ``...-<fv>-yl`` tail for a retained fused CARBOCYCLIC aromatic core
    (naphthalene / anthracene / phenanthrene / pyrene ...). Reuses the fixed-PAH
    numbering machinery (``get_polycyclic_iupac_locants``) which already runs the
    automorphism minimization; the free valence takes the lowest locant (routed
    through the ``pcg`` tier), then the decorations. Returns ``(pos, tail)`` or
    ``None`` (not a cataloged PAH, or a needed locant is a fusion-atom string)."""
    from rdkit import Chem
    frag_smi = Chem.MolFragmentToSmiles(mol, list(ring_atoms), canonical=True)
    if not frag_smi:
        return None
    from ..data import get_retained_name
    ring_name = get_retained_name(frag_smi)
    if not ring_name:
        return None
    from .polycyclics import POLYCYCLIC_DATA, get_polycyclic_iupac_locants
    if ring_name not in POLYCYCLIC_DATA:
        return None  # fused-heterocycle / non-cataloged fused core -> abstain
    sub_atoms = {attach} | set(deco_carriers)
    try:
        pos = get_polycyclic_iupac_locants(
            mol, ring_name, substituent_atoms=sub_atoms, pcg_atoms={attach})
    except Exception:
        return None
    if not pos:
        return None
    # Every free-valence / decoration-carrier locant must be a plain int
    # peripheral locant (fusion carbons carry no substituent -> never here).
    needed = [attach] + list(deco_carriers)
    if any(a not in pos or not isinstance(pos[a], int) for a in needed):
        return None
    stem = ring_name[:-1] if ring_name.endswith('e') else ring_name
    tail = f'{stem}-{pos[attach]}-yl'
    if ' ' in tail:
        return None
    return pos, tail


def _cage_core_numbering(mol, ring_atoms, attach, deco_carriers=()):
    """v28 Task 2b helper: full ``{orig_idx: int_locant}`` numbering + bare
    ``...-<fv>-yl`` tail for a von-Baeyer / bridged CAGE core (adamantane /
    tricyclo+ ...). Detaches the ring-only submol (property-tagged so the
    submol->orig index map is preserved) and routes it through the SAME
    audited ``analyze_cage_universal`` engine ``_universal_cage_substituent_name``
    uses, then reads BOTH the free-valence locant and every ring-atom locant off
    the single P-23 numbering (mutually consistent -> SELF-01-safe). Returns
    ``(pos, tail)`` or ``None`` on any refusal (spiro / >MAX / kekulize /
    numbering that does not cover exactly the ring atoms, or -- mirroring
    ``_pah_core_numbering`` / ``_fused_heterocycle_core_numbering`` -- any of
    ``[attach] + deco_carriers`` lacking a plain-int locant). Von-Baeyer
    numbering maps every skeletal ring atom, so this guard is currently
    latent-safe; added for defensive symmetry with the sibling numberers."""
    from rdkit import Chem
    from .vonbaeyer_universal import analyze_cage_universal
    from .polycyclic import _build_parent_with_unsaturation
    keep = set(ring_atoms)
    if attach not in keep:
        return None
    try:
        rw = Chem.RWMol(mol)
        for a in rw.GetAtoms():
            a.SetIntProp('_o28', a.GetIdx())
        for idx in sorted((a.GetIdx() for a in mol.GetAtoms()
                           if a.GetIdx() not in keep), reverse=True):
            rw.RemoveAtom(idx)
        sub = rw.GetMol()
        Chem.SanitizeMol(sub)
    except Exception:
        return None
    sub_to_orig: Dict[int, int] = {}
    attach_sub = None
    for a in sub.GetAtoms():
        try:
            o = a.GetIntProp('_o28')
        except KeyError:
            return None
        sub_to_orig[a.GetIdx()] = o
        if o == attach:
            attach_sub = a.GetIdx()
    if attach_sub is None:
        return None
    cage_atoms = [a.GetIdx() for a in sub.GetAtoms() if a.IsInRing()]
    if attach_sub not in cage_atoms:
        return None
    try:
        cage = analyze_cage_universal(
            sub, cage_atoms=set(cage_atoms), allow_mancude=True)
    except Exception:
        cage = None
    if cage is None:
        return None
    if set(cage.atom_to_locant) != set(cage_atoms):
        return None  # coverage floor: numbering must cover exactly the ring
    loc = cage.atom_to_locant.get(attach_sub)
    if not isinstance(loc, int):
        return None
    parent_block = _build_parent_with_unsaturation(
        cage.total_atoms, cage.unsaturation)
    if not parent_block:
        return None
    stem = parent_block[:-1] if parent_block.endswith('e') else parent_block
    # v30 P3-T1c lead 1: the same P-23.7 retained-PIN substitution as
    # ``_universal_cage_substituent_name``, applied here so a DECORATED cage core
    # (the composer's route) gets it too. Both the tail and ``pos`` below are read
    # off the SAME numbering, so the decoration locants stay consistent with the
    # free-valence locant by construction. Here the decoration carriers are known,
    # so the lowest-locant re-map minimises the free valence FIRST (P-29.3.2) and
    # then the substituent set -- which is what turns `7-hydroxyadamantan-3-yl`
    # into the PIN `3-hydroxyadamantan-1-yl`.
    _retained = _retained_pin_cage_stem(cage, stem)
    _numbering = dict(cage.atom_to_locant)
    if _retained is not None:
        _deco_sub = [s for s, o in sub_to_orig.items() if o in set(deco_carriers)]
        _numbering = _lowest_locant_cage_numbering(
            sub, cage, cage_atoms, attach_sub, _deco_sub)
        loc = _numbering.get(attach_sub, loc)
        tail = f'{_retained}-{loc}-yl'
    else:
        tail = f'{cage.hetero_prefix}{cage.descriptor}{stem}-{loc}-yl'
    if ' ' in tail:
        return None
    pos: Dict[int, int] = {}
    for s_idx, lc in _numbering.items():
        if s_idx in sub_to_orig and isinstance(lc, int):
            pos[sub_to_orig[s_idx]] = lc
    # Coverage guard (M2, defensive symmetry with the PAH / fused-heterocycle
    # numberers): every free-valence / decoration-carrier atom the composer
    # will look up must carry a plain-int locant, else fail closed.
    needed = [attach] + list(deco_carriers)
    if any(a not in pos for a in needed):
        return None
    return pos, tail


def _fused_heterocycle_core_numbering(mol, ring_atoms, attach, deco_carriers):
    """v28 Task 2c helper: full ``{orig_idx: int_locant}`` numbering + bare
    ``...-<fv>-yl`` tail for a retained fused-HETEROCYCLE core (1H-indole /
    quinoline / 1H-benzimidazole / 1-benzothiophene / purine ...).

    Reuses the FIXED IUPAC numbering catalogued in ``data.fused_heterocycles``
    (the ``iupac_locants`` map — canonical-SMILES atom index -> peripheral
    locant — and the retained ``name`` that already embeds the indicated
    hydrogen). ONE canonical numbering is read for BOTH the free valence AND
    every decoration carrier, so they are mutually consistent -> any single
    valid numbering is SELF-01-safe. The indicated H is taken VERBATIM from the
    catalog name (a fused-heterocycle pitfall — never guessed); the production
    OPSIN round-trip gate arbitrates any residual indicated-H uncertainty.

    Returns ``(pos, tail)`` or ``None`` (clean abstain) when the core is not a
    cataloged fused heterocycle, or when the free valence / a decoration carrier
    has no plain-integer peripheral locant (e.g. it lands on a fusion atom).
    """
    from rdkit import Chem
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    ring_list = list(ring_atoms)
    frag_smi = Chem.MolFragmentToSmiles(mol, ring_list, canonical=True)
    if not frag_smi:
        return None
    entry = FUSED_HETEROCYCLE_DATA.get(frag_smi)
    if entry is None:
        # MolFragmentToSmiles can differ subtly from the whole-mol canonical
        # SMILES the catalog is keyed by; re-canonicalise once and retry.
        rc = Chem.MolFromSmiles(frag_smi)
        if rc is not None:
            frag_smi = Chem.MolToSmiles(rc)
            entry = FUSED_HETEROCYCLE_DATA.get(frag_smi)
    if entry is None:
        return None
    iupac_locants = entry.get('iupac_locants') or {}
    name = entry.get('name')
    if not iupac_locants or not name:
        return None
    # ``iupac_locants`` is keyed by the ENTRY reference SMILES' atom indices, so
    # rebuild the reference from the (matched) catalog key and translate onto
    # THIS ring system by substructure match (P-29.2: minimise the free-valence
    # locant over automorphic matches; asymmetric cores have a single match).
    ref = Chem.MolFromSmiles(frag_smi)
    if ref is None:
        return None
    ring_atom_set = set(ring_list)
    needed = [attach] + list(deco_carriers)
    best_key = None
    best_pos: Optional[Dict[int, int]] = None
    for match in mol.GetSubstructMatches(ref, uniquify=False):
        if set(match) != ring_atom_set:
            continue
        pos: Dict[int, int] = {}
        for ref_idx, mol_idx in enumerate(match):
            loc = iupac_locants.get(ref_idx)
            if isinstance(loc, int):
                pos[mol_idx] = loc
        # every free-valence / decoration-carrier atom needs a plain-int
        # peripheral locant (fusion atoms carry no substituent -> never here).
        if any(a not in pos for a in needed):
            continue
        key = (pos[attach], tuple(sorted(pos[a] for a in deco_carriers)))
        if best_key is None or key < best_key:
            best_key = key
            best_pos = pos
    if best_pos is None:
        return None
    fv = best_pos[attach]
    # bare tail: the retained name embeds the indicated H (e.g. '1H-indole');
    # drop a trailing 'e' for the substituent stem, then '-<fv>-yl'.
    stem = name[:-1] if name.endswith('e') else name
    tail = f'{stem}-{fv}-yl'
    if ' ' in tail:
        return None
    return best_pos, tail


def polycyclic_core_numbering(
    mol, ring_atoms: Tuple[int, ...], attachment_atom: int,
    deco_carriers, allow_mancude: bool = False,
) -> Optional[Tuple[Dict[int, int], str]]:
    """v28 Task 2b: return ``(pos_map, bare_tail)`` for a POLYCYCLIC ring core so
    the recursive substituent composer can place decoration locants on a
    fused / bridged / cage core (the biggest drug-like breadth lever).

    ``pos_map`` is ``{orig_atom_idx: int_locant}`` and ``bare_tail`` the
    ``...-<fv>-yl`` core token — BOTH derived from ONE numbering, so the free
    valence and every decoration locant are mutually consistent. Any single
    VALID numbering is SELF-01-safe: a wrong-locant name is suppressed by the
    production OPSIN round-trip gate, never shipped, so the composer ATTEMPTS a
    covered candidate rather than fail-closing on numbering uncertainty.

    Covered classes: retained fused CARBOCYCLIC aromatics (naphthalene,
    anthracene, phenanthrene, pyrene ...), von-Baeyer CAGES (adamantane /
    tricyclo+ ...) and — v28 Task 2c — retained fused HETEROCYCLES (indole,
    quinoline, benzimidazole, benzothiophene, purine ...) via the
    ``data.fused_heterocycles`` catalog numbering. Returns ``None`` (clean
    abstain) for every other polycyclic core class — spiro, partial-hydro fused,
    non-cataloged fusions — which stay deferred (the composer then fails closed,
    never a wrong locant).

    Precondition (M3, v28 Composer #1 final review): complete/best-effort
    tier only. The sole production caller (the recursive decoration composer
    in ``substituent_enumerator.py``) is itself gated on ``allow_mancude`` and
    never reaches this function with ``allow_mancude=False``; a hypothetical
    future caller passing ``allow_mancude=False`` gets a clean ``None`` from
    every branch below rather than a PAH numbering it did not ask for.
    """
    if attachment_atom not in set(ring_atoms):
        return None
    if allow_mancude:
        res = _pah_core_numbering(mol, ring_atoms, attachment_atom, deco_carriers)
        if res is not None:
            return res
        # v28 Task 2c: retained fused-HETEROCYCLE core (indole / quinoline /
        # benzimidazole / benzothiophene / purine ...) — tried BEFORE the
        # general von-Baeyer cage numberer, which would otherwise name these
        # mancude fused aromatics as (valid but non-preferred) aza-bicyclo
        # polyenes. The retained name is the preferred emission. Gated on
        # allow_mancude so the PIN-default path stays byte-identical.
        res = _fused_heterocycle_core_numbering(
            mol, ring_atoms, attachment_atom, deco_carriers)
        if res is not None:
            return res
        # von-Baeyer / bridged CAGE fallback (adamantane / tricyclo+ ..., and
        # any non-cataloged mancude fused system as a polyene).
        res = _cage_core_numbering(mol, ring_atoms, attachment_atom, deco_carriers)
        if res is not None:
            return res
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
    allow_mancude: bool = False,
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

    # Purine ring system (adenine/hypoxanthine/purine skeleton), free valence on
    # a ring atom: fixed-numbering PIN substituent (6-amino-9H-purin-9-yl). The
    # generic decorated-fused path declines it (its retained-stem lookup keys on
    # the bare-ring SMILES `c1ncc2ncnc2n1`, no [nH], which misses the catalog).
    # This producer derives purine's fixed numbering + graph indicated-H directly
    # and fails closed, so it is safe on any tier (SELF-01 backstops). Pass the
    # RING ATOMS ONLY (not frag_set, which also includes exocyclic decorations
    # like the C6-amino nitrogen) -- name_purine_substituent requires an exact
    # match on the 9-atom purine core; the exocyclic amino is picked up inside
    # the producer as a ring decoration, and the parent side is masked via
    # _external_subtree from attach_idx.
    from .purine import name_purine_substituent
    _pur = name_purine_substituent(mol, frag_ring_atoms, attach_idx)
    if _pur is not None:
        return _pur

    if frag_ring_atoms == frag_set and ring_info.NumAtomRings(attach_idx) > 0:
        try:
            name = get_ring_substituent_name(mol, tuple(frag_atoms), attach_idx,
                                             allow_mancude=allow_mancude)
        except Exception:
            name = None
    elif (frag_ring_atoms and frag_ring_atoms < frag_set
          and ring_info.NumAtomRings(attach_idx) > 0):
        # BP-3 cluster R: a ring that carries its OWN decorations, rooted at a
        # ring atom (proper subset: ring atoms + decoration atoms). Neither the
        # bare-ring branch nor the ring-on-chain branch matched. Name it by
        # recursing the free-valence numbering cascade over the decorated ring.
        # Fail closed (None -> enumerator fallback -> DROP-24) on anything not
        # provably correct, so this only ADDS successful emissions.
        try:
            name = _decorated_heteroaryl_substituent_name(
                mol, frag_atoms, tuple(frag_ring_atoms), attach_idx,
                allow_mancude=allow_mancude,
            )
        except Exception:
            name = None
        if name is None and allow_mancude:
            # v30 Piece 2: the monocyclic producer declines a FUSED decorated
            # ring-substituent (e.g. 6-methoxynaphthalen-2-yl). Reuse the
            # fused-ring parent numbering to place the decorations + free
            # valence. BEST-EFFORT ONLY (allow_mancude): fused-substituent
            # numbering is a NEW capability that has not been proven byte-
            # identical against the PIN gold set, so — exactly as Piece 1 —
            # gate it behind the best-effort tier to keep PIN byte-identical by
            # construction (the v30 axis is best-effort breadth, invariant 16).
            # Fail closed (None -> enumerator fallback -> DROP-24), so this only
            # ADDS successful best-effort emissions; SELF-01 backstops a
            # mis-numbering (abstain, never a wrong molecule).
            try:
                name = _decorated_fused_substituent_name(
                    mol, frag_atoms, tuple(frag_ring_atoms), attach_idx
                )
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
            mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
            allow_mancude=allow_mancude,
        )
    if not name and allow_enumerator_fallback:
        # Recursion guard: the public ``name_substituent`` cascade routes
        # ring-bearing fragments back here (Tier 1.95). When called from there,
        # the caller passes allow_enumerator_fallback=False so a decline returns
        # None instead of re-entering the cascade (no infinite loop).
        # v28 Composer1 Task 3: thread allow_mancude so a decorated/fused core
        # this function's own narrow producers decline (BP-3 et al.) still
        # reaches the recursive decoration composer
        # (``_recursive_fragment_substituent_name``) through the full cascade,
        # instead of silently dropping back to allow_mancude=False here. Every
        # PRE-EXISTING call site passes allow_mancude=False (or omits it), so
        # this is a no-op for them -> byte-identical.
        from ..assembly.substituent_enumerator import name_substituent
        name = name_substituent(
            mol, frag_atoms, attach_idx, allow_mancude=allow_mancude)
        # Fail-closed backstop: if this fragment IS a ring assembly but the
        # P-28.3 builder in get_ring_substituent_name declined it above (an
        # assembly it cannot number — e.g. >5 rings, replacement class), the
        # generic cascade returns the yl-LESS parent hydride ('1,1'-biphenyl',
        # '4-chloro-1,1'-biphenyl'). That token ends in 'yl' and carries no
        # space, so the check below cannot catch it, yet it is OPSIN-unparseable
        # as a substituent. Reject it (honest abstention) unless it already is a
        # proper bracketed free-valence form. Only a fragment the detector
        # claims as an assembly is affected; every other fallback name stands.
        if (name and ']' not in name
                and _fragment_is_ring_assembly(mol, frag_atoms)):
            return None
    if name and name != 'substituent' and ' ' not in name:
        return name
    return None


def _fold_nonring_decorations(mol, frag_set, frag_ring_atoms, attach_idx):
    """v30 sub-lever A / Composer #2: fold every non-carrier decoration subgraph
    of a chain-rooted ring branch into the ring-yl atom set.

    The CARRIER is the connected component of ``frag_set - frag_ring_atoms`` that
    contains the free-valence ``attach_idx``; every OTHER non-ring component hangs
    off a ring atom and is a RING DECORATION (methoxy ``-O-CH3``, isopropyl, ...),
    which the degree-1-only folder leaves behind so it lands in the carrier and
    breaks the all-carbon carrier-path guard. Fold each such component into
    ``frag_ring_atoms`` so the decorated ring-yl is named as a unit by the
    ``name_ring_system_substituent`` recursion.

    Returns the widened ``frag_ring_atoms`` set, or ``None`` (fail closed) when a
    NON-carrier component leaves the fragment (a second attachment to the parent —
    not a simple substituent) or does not touch a ring atom (unexpected topology).
    """
    ring_set = set(frag_ring_atoms)
    non_ring = set(frag_set) - ring_set
    seen: Set[int] = set()
    widened = set(ring_set)
    for start in non_ring:
        if start in seen:
            continue
        comp: Set[int] = set()
        stack = [start]
        while stack:
            a = stack.pop()
            if a in comp:
                continue
            comp.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                if ni in non_ring and ni not in comp:
                    stack.append(ni)
        seen |= comp
        if attach_idx in comp:
            continue  # the carrier — never part of the ring-yl
        touches_ring = False
        for a in comp:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                if ni in ring_set:
                    touches_ring = True
                elif ni not in frag_set:
                    return None  # a second parent bond -> not a simple substituent
        if not touches_ring:
            return None  # a non-carrier component not anchored to the ring
        widened |= comp
    return widened


def _compound_ring_on_chain_substituent(
    mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
    allow_mancude: bool = False,
) -> Optional[str]:
    """Build '(ring-yl)alkyl' for an unbranched carrier rooted at the
    attachment with exactly ONE ring system hanging off it. Returns None
    (caller falls back) whenever any guard fails — never guesses.

    v28 Composer1 Task 3: when ``allow_mancude`` is True (complete/
    best-effort engine tier only) the carrier may ALSO admit exactly ONE
    simple, neutral, non-aromatic, divalent S/O/N atom bridging the carbon
    carrier directly to the ring (the RING-ON-CHAIN heteroatom-carrier case,
    e.g. ``-CH2-S-Ar`` -> ``'(arylsulfanyl)methyl'``). The ring-yl is named by
    recursing through ``name_ring_system_substituent`` (which — with
    ``allow_mancude`` threaded — reaches the general recursive decorated-core
    composer for a fused/decorated ring the narrow producers decline).
    Default False keeps every existing caller byte-identical.
    """
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
    if allow_mancude:
        # v30 sub-lever A / Composer #2: also fold MULTI-ATOM ring decorations
        # (methoxy, isopropyl, ...) into the ring-yl, so '(4-methoxyphenyl)methyl'
        # and the decorated-aryl/-cycloalkyl-on-a-simple-carrier class names via
        # the name_ring_system_substituent recursion below. Best-effort only; the
        # PIN default keeps the degree-1-only folding above, byte-identical.
        _widened = _fold_nonring_decorations(
            mol, frag_set, frag_ring_atoms, attach_idx)
        if _widened is None:
            return None
        frag_ring_atoms = _widened
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
    # v28 Composer1 Task 3: admit exactly ONE simple, neutral, non-aromatic,
    # divalent S/O/N carrier atom under allow_mancude (the PIN default keeps
    # the ORIGINAL all-carbon-only rejection byte-identical). Anything richer
    # (sulfoxide/sulfone O, charged/H-bearing/aromatic heteroatom, a SECOND
    # heteroatom) is a different, richer shape and stays fail-closed here —
    # not yet built, never guessed.
    hetero_carrier_atom = None
    for a in carrier:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != 'C':
            if (not allow_mancude or atom.GetSymbol() not in ('S', 'O', 'N')
                    or atom.GetFormalCharge() != 0
                    or atom.GetNumRadicalElectrons() != 0
                    or atom.GetIsAromatic()
                    or atom.GetDegree() != 2
                    # O/S: a neutral degree-2 bridging atom always has 0
                    # implicit H (divalent ether/thioether). N differs: a
                    # neutral degree-2 bridging N is the ordinary -NH-
                    # secondary amine and ALWAYS carries exactly 1 implicit
                    # H -- requiring ==0 here made the 'amino' connective
                    # branch below permanently unreachable dead code. Admit
                    # <=1 for N: the 0-H degree-2 N cases (imine =N-,
                    # charged) are already excluded above by the bond-order
                    # check below and the charge guard, so <=1 admits
                    # exactly the -NH- carrier and nothing richer.
                    or (atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 1)
                    or (atom.GetSymbol() != 'N' and atom.GetTotalNumHs() != 0)
                    or hetero_carrier_atom is not None):
                return None
            hetero_carrier_atom = a
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
        mol, sorted(frag_ring_atoms), ring_side_atoms[0],
        allow_mancude=allow_mancude,
    )
    if not ring_name:
        return None
    if hetero_carrier_atom is not None:
        # v28 Composer1 Task 3: the heteroatom-carrier shape is handled by a
        # SEPARATE assembly path (the '{ring-yl}{connective}' compound prefix
        # replaces what would otherwise be a bare ring-yl decoration on the
        # carbon carrier) -- it never reaches the α-halogen citation branch or
        # the plain-ring-yl branch below.
        if decorations:
            # Combined α-halogen decoration + heteroatom carrier: not this
            # narrow class (no OPSIN-verified precedent) -> fail closed.
            return None
        if (ring_attach_positions[0] != len(path)
                or path[-1] != hetero_carrier_atom):
            # Heteroatom not directly ring-bonded (mid-chain ether/thioether)
            # -> a different, richer shape -> fail closed.
            return None
        carbon_path = path[:-1]
        if not carbon_path:
            # Bare heteroatom directly on the parent (no methylene carrier) —
            # a different substituent shape owned by other tiers.
            return None
        try:
            from ..assembly.naming_utils import get_alkyl_name
            hetero_alkyl = get_alkyl_name(len(carbon_path))
        except Exception:
            return None
        if not hetero_alkyl:
            return None
        connective = {'S': 'sulfanyl', 'O': 'oxy', 'N': 'amino'}[
            mol.GetAtomWithIdx(hetero_carrier_atom).GetSymbol()
        ]
        from ..assembly.naming_utils import (
            is_complex_substituent, apply_enclosing_marks, _is_fully_enclosed,
        )
        if connective == 'oxy':
            # P-63.2.1: the retained contraction 'phenoxy' for a BARE phenyl
            # ring-yl; every other ring-yl (retained or systematic) uses the
            # general uncontracted '{ring-yl}oxy' form (P-63.2.2.2), enclosed
            # as a unit only if the ring-yl itself left un-self-enclosed marks
            # inside the concatenation (mirrors
            # ``_name_ether_substituted_chain``'s O-branch, the established
            # convention for this exact shape elsewhere in the codebase).
            if ring_name == 'phenyl':
                group = 'phenoxy'
            else:
                _inner = (f'({ring_name})' if is_complex_substituent(ring_name)
                          else ring_name)
                group = f'{_inner}oxy'
                if ('(' in group or '[' in group) and not _is_fully_enclosed(group):
                    group = apply_enclosing_marks(group, -1)
        else:
            # P-16.3.3/P-29.5.2: an '{ring-yl}sulfanyl'/'{ring-yl}amino'
            # compound prefix is ALWAYS enclosed as a unit before
            # concatenating the carbon-carrier stem (mirrors
            # ``_name_ether_substituted_chain``'s S-branch: '(benzylsulfanyl)',
            # '[(methoxymethyl)sulfanyl]').
            _inner = (apply_enclosing_marks(ring_name, -1)
                      if is_complex_substituent(ring_name) else ring_name)
            group = apply_enclosing_marks(f'{_inner}{connective}', -1)
        if not group or ' ' in group:
            return None
        if len(carbon_path) == 1:
            return f'{group}methyl'
        return f'{len(carbon_path)}-{group}{hetero_alkyl}'
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
    # P-16.5.1.1: enclose the ring-yl in marks only when it is itself complex
    # (carries locants/parens, e.g. '(naphthalen-2-yl)methyl'); a simple ring-yl
    # is concatenated bare ('cyclohexylmethyl', 'phenylmethyl' is retained
    # 'benzyl' elsewhere). The old unconditional parens produced the non-PIN
    # '(cyclohexyl)methyl'.
    from ..assembly.naming_utils import (
        is_complex_substituent, apply_enclosing_marks)
    # P-16.3.3 enclosing-mark nesting: escalate the outer mark to brackets when a
    # DECORATED ring-yl already carries an inner enclosure ('4-(propan-2-yl)phenyl'
    # -> '[4-(propan-2-yl)phenyl]methyl'). apply_enclosing_marks is byte-identical
    # to the old f'({ring_name})' for every mark-free ring-yl (naphthalen-2-yl,
    # 4-methoxyphenyl, 4-chlorophenyl), so PIN-reachable cases are unchanged.
    inner = (apply_enclosing_marks(ring_name, -1)
             if is_complex_substituent(ring_name) else ring_name)
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


def _contained_rings(mol, ring_atoms: Tuple[int, ...]):
    """The SSSR rings wholly inside ``ring_atoms``.

    Mirrors the single-ring guard in :func:`identify_ring_system`; factored out
    so the Task AA5 heteromonocycle fallback applies the identical test rather
    than a second, drifting copy of it.
    """
    ring_set = set(ring_atoms)
    return [r for r in mol.GetRingInfo().AtomRings() if set(r) <= ring_set]


def _bare_stem_was_withdrawn_as_non_pin(frag_smi: str) -> bool:
    """True iff this ring fragment's retained bare stem was DENIED as non-PIN.

    Task AA5. Distinguishes "the retained table never had a name for this ring"
    (leave it fail-closed) from "the retained table had one and the adjudicated
    PIN list withdrew it" (the substituent path just lost its only answer and
    must fall back to the systematic parent name).

    Reads the demotion the deny performs: ``pin_policy``'s contract is that a
    denied name leaves ``ALL_RETAINED_NAMES`` and lands in the general-only
    companion dict, so the presence of a PIN-denied name there IS the signal.
    """
    if not frag_smi:
        return False
    try:
        from ..data import GENERAL_RETAINED_NAMES
        from ..data.pin_policy import is_pin_denied
    except ImportError:
        return False
    demoted = GENERAL_RETAINED_NAMES.get(frag_smi)
    return bool(demoted) and is_pin_denied(demoted)


def _heteromonocycle_parent_name(mol, ring_atoms: Tuple[int, ...]) -> Optional[str]:
    """Systematic parent name for a heteromonocyclic ring, or None (fail-closed).

    Task AA5. Delegates to the Hantzsch-Widman / retained-stem namer the PARENT
    path already uses, so the substituent path cannot disagree with the parent
    path about what a ring is called. Returns None for an all-carbon ring (the
    carbocyclic branches above own those), for anything the namer declines, and
    for any name carrying a locant-bearing substituent or a space -- only a bare
    parent hydride name is safe to turn into a '-yl' stem here.
    """
    if not any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_atoms):
        return None
    try:
        from .heterocycles import name_heterocycle
        name = name_heterocycle(mol, tuple(ring_atoms))
    except Exception:  # noqa: BLE001 - a declining namer must never propagate
        return None
    if not name or not isinstance(name, str):
        return None
    # A parent hydride only: no whitespace, no substituent prefixes, no suffix.
    if ' ' in name or name.endswith('yl'):
        return None
    return name


def _ring_assembly_substituent_prefix(
    mol, ring_atoms: Tuple[int, ...], attachment_point: Optional[int]
) -> Optional[str]:
    """P-28.3 substituent prefix for a RING ASSEMBLY fragment, or ``None``.

    A ring assembly is a set of identical (or single-heteroatom-replacement)
    ring systems joined directly by single bonds — biphenyl, terphenyl,
    2,2'-bipyridine. As a substituent it takes the PRIMED free-valence form
    ``[1,1'-biphenyl]-4-yl``, NOT the yl-less parent hydride ``1,1'-biphenyl``
    that the generic ``name_substituent`` cascade returns for such a fragment
    (an OPSIN-UNPARSEABLE substituent token — the elevated-blocker F1 class).

    Delegates to the SAME P-28.3 builder the whole-molecule composer already
    trusts on the PIN path (``composer.py`` ->
    ``4-([1,1'-biphenyl]-4-yl)butanoic acid``): restrict the whole molecule's
    ring systems to those lying wholly inside this fragment, require the free
    valence to sit in one of them, and let ``detect_ring_assembly`` /
    ``name_ring_assembly_prefix`` build the token. Fail closed (``None``) on
    anything that is not a numberable identical-ring assembly — a fused
    polycycle (one system), a mixed pair, or an assembly the builder cannot
    number — so this only ADDS correct emissions and never overrides the
    fused/retained/von-Baeyer producers below.
    """
    if attachment_point is None:
        return None
    from ..perception.rings import get_ring_systems
    from .ring_assemblies import detect_ring_assembly, name_ring_assembly_prefix

    ring_set = set(ring_atoms)
    frag_systems = [s for s in get_ring_systems(mol) if s <= ring_set]
    if len(frag_systems) < 2:
        return None  # a single (possibly fused) ring system — not an assembly
    if sum(len(s) for s in frag_systems) != len(ring_set):
        return None  # some fragment ring atom is not in a captured system
    if not any(attachment_point in s for s in frag_systems):
        return None  # free valence must lie in one assembly ring system
    info = detect_ring_assembly(mol, frag_systems)
    if info is None:
        return None  # non-identical / branched / atom-bridged — not P-28.3
    try:
        return name_ring_assembly_prefix(mol, info, attachment_point)
    except Exception:  # noqa: BLE001 — a builder error is a decline, not a crash
        return None


def _fragment_is_ring_assembly(mol, frag_atoms) -> bool:
    """True iff ``frag_atoms`` is exactly a ring assembly (biphenyl-like).

    Detection only, independent of whether the P-28.3 prefix builder can NUMBER
    it — used as the fail-closed backstop so the generic cascade's yl-less
    parent hydride can never leak for an assembly the builder declined.
    """
    from ..perception.rings import get_ring_systems
    from .ring_assemblies import detect_ring_assembly

    ring_info = mol.GetRingInfo()
    frag_set = set(frag_atoms)
    if any(ring_info.NumAtomRings(a) == 0 for a in frag_set):
        return False  # a chain-rooted / decorated fragment is not a bare assembly
    frag_systems = [s for s in get_ring_systems(mol) if s <= frag_set]
    if len(frag_systems) < 2 or sum(len(s) for s in frag_systems) != len(frag_set):
        return False
    return detect_ring_assembly(mol, frag_systems) is not None


def get_ring_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_point: Optional[int] = None,
    allow_mancude: bool = False,
) -> Optional[str]:
    """
    Get the substituent name for a ring when it becomes a substituent on a chain.

    This is the main entry point for ring-as-substituent naming.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_point: Optional ring atom index where the ring attaches to chain.
                         Used for position-specific names (e.g., 2-pyridyl vs 4-pyridyl).
        allow_mancude: v27 P1 opt-in (complete/best-effort engine tier only).
                       When True, an unretained multi-ring cage the narrow PIN
                       namers decline (tricyclo+/adamantane, and mancude
                       fused-aromatic systems) is named via the universal
                       von-Baeyer cage engine as a ``...-<loc>-yl`` polyene.
                       Default False keeps every existing caller byte-identical.

    Returns:
        Substituent name string (e.g., 'phenyl', 'cyclohexyl', '2-pyridyl'), or
        None when the ring system cannot be named by a provable rule (Phase 4
        SUBST-01: fail-closed — never a monocycle size-guess for a polycyclic).
    """
    # P-29.2 free-valence gate. Every stem below ('cyclohexyl', 'phenyl', a
    # retained fused name + '-yl') spells ONE free valence, chosen from the ring
    # system's identity alone -- the attachment BOND is never read. So a ring
    # joined to its parent by a double bond was named '-yl' and the double bond
    # silently became single: 'OCC=C1CCCCC1' was named 2-cyclohexylethan-1-ol.
    # Same shared primitive as the two general chokepoints; a single bond (and a
    # two-point attachment, which is a fusion rather than a prefix) defers and
    # leaves every existing name byte-identical.
    if attachment_point is not None:
        from ..assembly.substituent_enumerator import carbon_free_valence_prefix

        _fv = carbon_free_valence_prefix(mol, ring_atoms, attachment_point)
        if _fv.prefix is not None:
            return _fv.prefix
        if _fv.must_fail_closed:
            return None

    # Identify the ring system
    ring_name = identify_ring_system(mol, ring_atoms)

    if ring_name is None:
        # P-28.3: a RING ASSEMBLY (identical rings joined by single bonds —
        # biphenyl, terphenyl, bipyridine) takes the primed free-valence
        # substituent form '[1,1'-biphenyl]-4-yl'. Try it BEFORE the retained /
        # polycyclic producers: those decline an assembly and the generic
        # enumerator fallback then ships the yl-less parent hydride
        # '1,1'-biphenyl' (OPSIN-unparseable as a substituent). Fail closed
        # (falls through) on anything not a numberable identical-ring assembly.
        _asm = _ring_assembly_substituent_prefix(mol, ring_atoms, attachment_point)
        if _asm is not None:
            return _asm
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
        # 5,6,7,8-tetrahydronaphthalen-1-yl, spiro[4.5]decan-2-yl). v27 P1:
        # under the complete tier (allow_mancude) this also reaches the
        # universal cage engine (tricyclo+/adamantane, mancude polyenes).
        poly = _polycyclic_substituent_name(
            mol, ring_atoms, attachment_point, allow_mancude=allow_mancude)
        if poly:
            return poly

        # Task AA5 (P-22.2.1): last resort for a HETEROMONOCYCLE the retained
        # table does not name.
        #
        # ``identify_ring_system`` describes a ring by size + heteroatom set and
        # has no branch for a two-heteroatom saturated five-ring, so
        # 1,3-thiazolidine, 1,2-thiazolidine, 1,2-oxazolidine, their Se/Te
        # analogues, selenophene and 1,2-selenazole all arrive here with
        # ``ring_name is None`` even though they are plain monocycles. Until
        # Task AA5 the retained lookup above rescued them by returning a BARE
        # stem ('thiazolidine'), so the substituent path had no route of its own
        # to the systematic name -- it depended entirely on the retained table.
        # Withdrawing those bare stems as non-PINs (they are: P-22.2.1 Table 2.3
        # prints '1,3-thiazolidine (PIN)' at BlueBookV2.md:8182 and
        # '1,2-thiazolidine (PIN)' at :8184) therefore turned ten correct names
        # into abstentions -- '(thiazolidin-4-yl)methanol' became 'unknown'.
        # That is the the contributor guide invariant-9 trap: removing a wrong output
        # unmasked a worse one. The parent path never had this gap because it
        # reaches the Hantzsch-Widman namer directly.
        #
        # Placement is deliberate: this runs AFTER every existing producer has
        # declined, so it can never change a name any current path produces.
        #
        # ⚠ SCOPE IS DELIBERATELY NARROW, and the narrowing is measured, not
        # cautious by taste. The obvious generalisation -- "any heteromonocycle
        # the retained table misses" -- newly names 1660 distinct substituent
        # stems across pubchem_2000 + chebi_5000 + opsin_selftest_500, and it is
        # WRONG on a large share of them: for macrocycles such as
        # '1,10-dioxa-4-azacycloheptadeca-2,7,12-triene' the independently
        # computed attachment locant lands on a ring OXYGEN, and OPSIN either
        # refuses the name outright or resolves it to a different structure.
        # Turning an abstention into a wrong name is worse than the abstention.
        #
        # So this fires ONLY for a ring whose bare retained stem THIS project has
        # adjudicated non-PIN and withdrawn -- i.e. exactly the rings the deny
        # list took the substituent path's only answer away from. That is a
        # closed, auditable, data-driven class (read off the deny list itself,
        # not a molecule list), and it self-maintains: a future deny row of the
        # same shape gets the same rescue automatically.
        if attachment_point is not None and len(_contained_rings(mol, ring_atoms)) == 1:
            if _bare_stem_was_withdrawn_as_non_pin(frag_smi):
                het_name = _heteromonocycle_parent_name(mol, ring_atoms)
                if het_name:
                    stem = het_name[:-1] if het_name.endswith('e') else het_name
                    locant = _get_polycyclic_attachment_locant(
                        mol, ring_atoms, attachment_point, ring_name=het_name
                    )
                    if locant is not None:
                        return f'{stem}-{locant}-yl'

        # v30 P3-T1b: the AUDITED systematic terminal ring namer, complete /
        # best-effort tier only. This is the "downgrade, don't refuse" end of the
        # cascade: every retained / fused / von-Baeyer / spiro / hydro / HW
        # producer above has declined, so the choice here is between an uglier
        # systematic name and an abstention.
        #
        # It is what makes the GENERAL version of the narrow rescue above safe.
        # That rescue is deliberately restricted to withdrawn-stem rings because
        # the obvious generalisation was measured WRONG on a large share of 1660
        # newly-named stems -- "the independently computed attachment locant
        # lands on a ring OXYGEN, and OPSIN either refuses the name outright or
        # resolves it to a different structure". ``terminal_ring`` closes exactly
        # that hole: its reconstruction audit re-reads the emitted string and
        # requires the free-valence locant to land on THIS attachment atom and
        # every locant's element to match the graph, so a locant that drifted
        # onto a ring oxygen is refused rather than shipped.
        #
        # Gated on ``allow_mancude`` so the PIN default keeps the fail-closed
        # refusal below, byte-identical.
        if allow_mancude and attachment_point is not None:
            from .terminal_ring import terminal_ring_name
            _t = terminal_ring_name(mol, sorted(ring_atoms), attachment_point)
            if _t is not None:
                return _t.name

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
                        # v30 P3-T1b: this refusal exists because the SATURATED
                        # dictionary stem would silently drop the ring double bond
                        # ('oxanyl' for a dihydropyranyl fragment = a different
                        # molecule). The systematic generator cites every ring
                        # multiple bond explicitly and proves it by reconstruction
                        # audit, so under the complete / best-effort tier it can
                        # answer here instead of abstaining. The PIN default keeps
                        # the refusal, byte-identical.
                        if allow_mancude and attachment_point is not None:
                            from .terminal_ring import terminal_ring_name
                            _t6 = terminal_ring_name(
                                mol, sorted(ring_atoms), attachment_point)
                            if _t6 is not None:
                                return _t6.name
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
        best_het_locants = None
        best_attach_locant = None

        for next_atom in neighbors_of_start:
            path = _traverse_ring(start_atom, next_atom, ring_adj, len(ring_atoms))
            if len(path) != len(ring_atoms):
                continue
            # Compute heteroatom locant set (excluding position 1 which is always a het)
            het_locs = tuple(sorted(
                path.index(hi) + 1 for hi in het_indices if hi != start_atom
            ))
            if attachment_atom not in path:
                continue
            attach_locant = path.index(attachment_atom) + 1  # 1-indexed
            # Lowest heteroatom locants fix the ring numbering; among numberings
            # TIED on that (a symmetric ring has two equivalent directions), the
            # free valence takes the LOWEST locant it can (P-31.1.4.3.4 free-
            # valence lowest-locant rule). Previously the first tied direction
            # won, yielding a non-lowest locant for symmetric heteroarenes
            # (1,3,4-oxadiazol-5-yl instead of -2-yl). Because all tied
            # numberings describe the SAME atom equivalences, choosing the lower
            # attachment locant can never change the constitution — it only
            # lowers a locant, never mis-places the free valence.
            if (best_het_locants is None
                    or het_locs < best_het_locants
                    or (het_locs == best_het_locants
                        and attach_locant < best_attach_locant)):
                best_het_locants = het_locs
                best_attach_locant = attach_locant

        if best_attach_locant is not None:
            return best_attach_locant

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


def _collect_acyclic_branch(mol, start_idx: int, ring_atom_set: Set[int]
                            ) -> Optional[Set[int]]:
    """All atom indices of the exocyclic branch rooted at ``start_idx`` (BFS),
    excluding the parent ring. Returns None if the branch re-enters ANY ring
    (a nested ring substituent is out of the acyclic-branch scope — the caller
    must then fail closed rather than emit a partial name). Used to hand a
    branched/substituted acyclic-carbon substituent (e.g. ``1-chloroethyl``,
    with an optional CIP stereodescriptor) to the universal substituent namer.
    """
    seen: Set[int] = set()
    stack = [start_idx]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        if atom.IsInRing():
            return None
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j in ring_atom_set or j in seen:
                continue
            stack.append(j)
    return seen


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
    distribution): oxo, cyano, carboxy, halogen, hydroxy, methoxy/simple
    n-alkoxy, amino (-NH2), nitro, unbranched pure alkyl, and (v28 Cluster D,
    P-65.6.3) simple alkyl esters -> alkoxycarbonyl prefix. Deliberately NOT
    supported (guard out): amides/acyl, aryl/branched-alkyl esters, sulfonyl,
    nested ring substituents, branched alkyl, anything charged or exotic.
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
            # v30 tail #13 (P-63.2.2.2): the ether-O's ligand is ANOTHER RING
            # -- a glycosidic / inter-ring link (this ring atom bears
            # -O-(anomeric C of a second ring)). Name that inner ring system
            # RECURSIVELY as a substituent and cite '({inner-yl})oxy'. This is the
            # nested-ring-oxy case reaches by unbounded recursion; here
            # it routes back through name_substituent -> the ring chokepoint ->
            # this function, so a di/tri-saccharide nests naturally (a
            # trisaccharide is just one more level). ADDITIVE: only reached after
            # the hydroxy + linear-alkoxy cases decline, so every simple ring keeps
            # its byte-identical legacy form. Depth-bounded by the shrinking atom
            # set (the inner subgraph excludes this ring), like .
            if (len(o_nbrs) == 1 and o_nbrs[0].GetSymbol() == 'C'
                    and o_nbrs[0].IsInRing()
                    and o_nbrs[0].GetIdx() not in ring_atom_set):
                inner_attach = o_nbrs[0].GetIdx()
                inner_frag: Set[int] = set()
                _seen_r = set(ring_atom_set) | {ni}
                _stack_r = [inner_attach]
                while _stack_r:
                    _a = _stack_r.pop()
                    if _a in _seen_r:
                        continue
                    _seen_r.add(_a)
                    inner_frag.add(_a)
                    for _nn in mol.GetAtomWithIdx(_a).GetNeighbors():
                        if _nn.GetIdx() not in _seen_r:
                            _stack_r.append(_nn.GetIdx())
                if inner_frag:
                    from ..assembly.substituent_enumerator import (
                        name_substituent, alkoxy_prefix_from_substituent)
                    inner_name = name_substituent(
                        mol, sorted(inner_frag), inner_attach)
                    if (inner_name and inner_name != 'substituent'
                            and ' ' not in inner_name
                            and inner_name.endswith('yl')):
                        _oxy = alkoxy_prefix_from_substituent(inner_name)
                        if _oxy:
                            prefixes.append(_oxy)
                            covered.add(ni)
                            covered.update(inner_frag)
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
                # v28 Cluster D (P-65.6.3): an ALKYL-ESTER decoration -C(=O)-O-R
                # (the second O has NO H and is bonded onward to a carbon) is
                # expressed as the alkoxycarbonyl prefix (methoxycarbonyl, ...).
                # Reuse the canonical producer; principal_chain=None makes it use
                # SMARTS-based alkyl-side discrimination and skip the chain-
                # orientation guard. Fail closed (return None) for aryl/branched/
                # oversized R (the producer returns None) so no partial name leaks.
                ester_o = next(
                    (x for x in c_nbrs
                     if x.GetDegree() == 2 and x.GetTotalNumHs() == 0
                     and x.GetFormalCharge() == 0
                     and mol.GetBondBetweenAtoms(ni, x.GetIdx()).GetBondTypeAsDouble() == 1.0
                     and any(nn.GetSymbol() == 'C' and nn.GetIdx() != ni
                             for nn in x.GetNeighbors())),
                    None)
                if has_carbonyl_O and ester_o is not None:
                    carbonyl_o = next(x for x in c_nbrs if x.GetIdx() != ester_o.GetIdx())
                    alkyl_c = next(nn for nn in ester_o.GetNeighbors()
                                   if nn.GetIdx() != ni)
                    from ..assembly.substituent_prefix_forms import (
                        get_alkoxycarbonyl_prefix,
                    )
                    prefix = get_alkoxycarbonyl_prefix(
                        mol,
                        (ni, carbonyl_o.GetIdx(), ester_o.GetIdx(), alkyl_c.GetIdx()),
                        None,
                    )
                    if prefix:
                        # Cover the whole ester unit: carbonyl C, both O, and the
                        # entire R alkyl fragment (so the exact-coverage check that
                        # gates this substituent name accounts for every atom).
                        prefixes.append(prefix)
                        covered.add(ni)
                        covered.update(x.GetIdx() for x in c_nbrs)
                        _seen = {ni, ester_o.GetIdx()}
                        _stack = [alkyl_c.GetIdx()]
                        while _stack:
                            _a = _stack.pop()
                            if _a in _seen:
                                continue
                            _seen.add(_a)
                            covered.add(_a)
                            for _nn in mol.GetAtomWithIdx(_a).GetNeighbors():
                                if _nn.GetIdx() not in _seen:
                                    _stack.append(_nn.GetIdx())
                        continue
                return None
            chain = _linear_carbon_chain(ni, ring_atom_set)
            if chain is not None and chain in _ALKYL:
                prefixes.append(_ALKYL[chain])
                covered.update(_chain_atoms(ni, ring_atom_idx))
                continue
            # Branched / substituted acyclic-carbon substituent (e.g.
            # (1S)-1-chloroethyl for P-45.6): the v1 table deliberately excluded
            # branched alkyl and has no stereodescriptor support. Name the whole
            # exocyclic branch with the universal acyclic substituent namer (it
            # renders '(1S)-1-chloroethyl'); the assembly brackets it as a
            # complex prefix. Fail closed on a ring-bearing or unnameable branch
            # so no partially-described name can leak.
            frag = _collect_acyclic_branch(mol, ni, ring_atom_set)
            if frag is not None:
                from ..assembly.substituent_naming import name_substituent_fragment
                cname = name_substituent_fragment(
                    mol, sorted(frag), ni, list(ring_atom_set))
                if cname and ' ' not in cname and cname.endswith('yl'):
                    prefixes.append(cname)
                    covered.update(frag)
                    continue
            return None
        # pnictogen substituent (-As/-Sb/-Bi bearing organyl/H) -> arsanyl /
        # stibanyl / bismuthanyl prefix (P-67.1.5.1 / P-68.3). Collect the
        # pnictogen subgraph (the As/Sb/Bi + everything hanging off it, NOT
        # re-entering the ring) and reuse the general pnictogen-yl namer. This
        # is what lets a benzene ring bearing -Sb(C6H5)2 name AS a substituent
        # -> '4-(diphenylstibanyl)phenyl' (the P-69.5.2 organomercurial ligand).
        # Fail closed (return None) if the pnictogen namer declines.
        if sym in ('As', 'Sb', 'Bi') and order == 1.0 and charge == 0:
            pnic_frag: Set[int] = set()
            _seen_p = set(ring_atom_set) | {ring_atom_idx}
            _stack_p = [ni]
            while _stack_p:
                _a = _stack_p.pop()
                if _a in _seen_p:
                    continue
                _seen_p.add(_a)
                pnic_frag.add(_a)
                for _nn in mol.GetAtomWithIdx(_a).GetNeighbors():
                    if _nn.GetIdx() not in _seen_p:
                        _stack_p.append(_nn.GetIdx())
            from ..rules.mononuclear_hydrides import name_arsanyl_substituent
            pnic_name = name_arsanyl_substituent(mol, list(pnic_frag), ni)
            if pnic_name:
                prefixes.append(pnic_name)
                covered.update(pnic_frag)
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
    # Only the ONE bond that leaves the fragment is the free valence to the
    # parent; every other exocyclic neighbour of the attachment atom is a
    # genuine decoration and must be named.
    #
    # This used to skip ALL exocyclic neighbours of the attachment atom. On a
    # mono-substituted attachment carbon there is exactly one, so "skip them
    # all" and "skip the parent bond" coincide and the defect is invisible --
    # which is why it survived. On a GEM-DISUBSTITUTED attachment carbon the
    # ring's own substituent was discarded together with the parent bond,
    # `atom_prefixes` came back empty, and the "nothing to decorate" guard
    # below returned None: `N-(1-methylcyclohexyl)acetamide` was unreachable
    # while `N-(4,4-dimethylcyclohexyl)acetamide` (gem AWAY from the
    # attachment) always worked.
    #
    # The parent bond is identified STRUCTURALLY -- the exocyclic neighbour
    # lying outside the fragment being named -- never by a count or an index
    # ().
    # The `> 1` filter is DEFENSIVE and is a proven EQUIVALENT MUTANT today:
    # every production caller passes a heavy-atom `expected_atoms`, and the
    # production molecules carry implicit hydrogens, so there is no H neighbour
    # to filter. Measured under `Chem.AddHs` the function declines for an
    # unrelated reason with or without the filter. Kept because the signature
    # admits an explicit-H mol, in which case an H would otherwise be counted
    # as a candidate parent bond below.
    _exo_nbrs = {nbr.GetIdx() for nbr in
                 mol.GetAtomWithIdx(attachment_atom).GetNeighbors()
                 if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1}
    if expected_atoms is not None:
        parent_nbrs = _exo_nbrs - set(expected_atoms)
    elif len(_exo_nbrs) > 1:
        # No fragment scope was supplied, so which of these bonds leaves the
        # fragment is genuinely undecidable here. Treating them all as the
        # parent bond would DROP a decoration and name a different molecule,
        # so fail closed and let the caller keep its legacy form.
        return None
    else:
        parent_nbrs = _exo_nbrs
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

    def _is_complex_prefix(nm: str) -> bool:
        # simple v1-table prefixes are bare lowercase words; a branched/stereo
        # substituent name carries a locant, hyphen, or enclosing mark.
        # v28 Cluster D (P-16.3.3): a COMPOUND substituent prefix (formed by
        # substitution, e.g. the alkoxycarbonyl family methoxycarbonyl/
        # ethoxycarbonyl/phenoxycarbonyl) is enclosed even though it is a bare
        # lowercase word -> '2-(methoxycarbonyl)cyclohexyl', not '2-methoxy...'.
        if nm.endswith('oxycarbonyl'):
            return True
        # P-16.3.3: a SUBSTITUTED pnictogen-yl (diphenylstibanyl / dimethylarsanyl
        # / ...bismuthanyl) is a compound prefix requiring enclosing marks, even
        # though it is a bare lowercase word; the bare parent-hydride-yl
        # ('arsanyl'/'stibanyl'/'bismuthanyl') stays simple (cf. naming_utils
        # .is_complex_substituent). Narrowly scoped so simple alkyl/aryl (methyl,
        # phenyl) keep the byte-identical legacy path.
        for _pn in ('arsanyl', 'stibanyl', 'bismuthanyl'):
            if nm.endswith(_pn) and nm != _pn:
                return True
        # The hyphen in the character class made this a copy of the compound
        # predicate; the P-16.3.4 carve-out is the shared primitive, so
        # 'tert-butyl' is simple (BB 16286 cites it bare) while
        # 'tert-butylsulfanyl' stays compound.
        from ..assembly.naming_utils import italicized_prefix_is_bare
        if italicized_prefix_is_bare(nm):
            return False
        # v31 lever B (P-16.3.3): a compound FG-prefix + alkyl ('hydroxymethyl',
        # 'cyanomethyl', 'carboxymethyl', 'aminoethyl') is a SUBSTITUTED (compound)
        # substituent and takes enclosing marks even though it carries no
        # digit/hyphen/mark — the character-class heuristic below misses it, so a
        # ring decoration 'hydroxymethyl' was cited bare ('4-hydroxymethylphenyl').
        # BB PINs: '2-(hydroxymethyl)benzene-1,4-diol' (:6802), '(cyanomethyl)'
        # (:33087). Same set naming_utils.needs_brackets / is_complex_substituent
        # recognise; scoped to this class to keep every other decoration
        # byte-identical.
        from ..assembly.naming_utils import (
            _COMPOUND_FG_PREFIXES as _CFG, _ALKYL_ROOTS_FULL as _ARF,
        )
        _nml = nm.lower()
        for _fg in _CFG:
            if _nml.startswith(_fg) and _nml[len(_fg):] in _ARF:
                return True
        return any(c in nm for c in '-()[]0123456789')

    if not any(_is_complex_prefix(nm) for nm in groups):
        # byte-identical legacy path — simple-prefix rings are unchanged.
        parts = []
        for name in sorted(groups):  # alphabetical citation order
            locs = sorted(groups[name])
            loc_str = ','.join(str(loc) for loc in locs)
            mult = get_multiplier_prefix(len(locs), name) if len(locs) > 1 else ''
            parts.append(f'{loc_str}-{mult}{name}')
        prefix_str = '-'.join(parts)
    else:
        # P-16.5.4: a complex (branched/stereo) prefix is enclosed; ordering is
        # by the alphanumerical key of its stereo-stripped base name (P-14.5.2:
        # '(1S)-1-chloroethyl' alphabetizes at 'c', descriptor + locant ignored).
        from ..assembly.naming_utils import (
            alpha_sort_key as _alpha_sort_key, apply_enclosing_marks,
        )
        from .stereochemistry import strip_stereo as _strip_stereo

        def _sort_key(nm: str):
            return (_alpha_sort_key(_strip_stereo(nm)), nm)

        parts = []
        for name in sorted(groups, key=_sort_key):
            locs = sorted(groups[name])
            loc_str = ','.join(str(loc) for loc in locs)
            mult = get_multiplier_prefix(len(locs), name) if len(locs) > 1 else ''
            rendered = apply_enclosing_marks(name, -1) if _is_complex_prefix(name) else name
            parts.append(f'{loc_str}-{mult}{rendered}')
        prefix_str = '-'.join(parts)

    # W4-S2 (P-93.5.1.1.2 / BB 48117): emit the ring substituent's OWN internal
    # stereodescriptors, keyed to the SAME numbering `order` the name uses. The
    # pseudoasymmetric C-1/C-4 centres of a 1,4-disubstituted cyclohexane (and any
    # ring-member stereocentre) were previously DROPPED entirely: the multi-centre
    # branch of `_add_substituent_stereo` fail-closes because it cannot thread the
    # substituent's own numbering. Here that numbering IS known (order = locant ->
    # atom), so the descriptor is correct-by-construction and the locants match the
    # emitted name exactly. Additive: fires ONLY when a ring atom carries a CIP
    # label, so every non-stereo ring is byte-identical to the legacy output. The
    # lowercase r/s pseudoasymmetric casing is inherited VERBATIM from RDKit (D-15;
    # P-92.1.4.2) — the BB PIN is `bis[(1r,4r)-4-methylcyclohexyl]phosphane` (BB
    # 48117), and `collect_stereodescriptors` is CALLED read-only (not modified).
    # OPSIN's generation grammar rejects the r/s cyclohexane layer, but the
    # constitutional form parses, so the DEF-9 stereo carve-out in the validity
    # gate (namer.py) ships the full PIN — cf. `(1s,4s)-cyclohexane-1,4-diol`.
    _stereo_prefix = ''
    if any(mol.GetAtomWithIdx(i).HasProp('_CIPCode') for i in ring_list):
        from .stereochemistry import (
            collect_stereodescriptors as _collect_sd,
            format_stereodescriptor_string as _fmt_sd,
        )
        _ring_a2l = {atom_idx: loc for loc, atom_idx in order.items()}
        _stereo_prefix = _fmt_sd(
            _collect_sd(mol, _ring_a2l, include_near_parent_ez=False)
        )

    if het:
        return f'{_stereo_prefix}{prefix_str}{stem}-{attachment_locant}-yl'
    return f'{_stereo_prefix}{prefix_str}{stem}'


# --------------------------------------------------------------------------- #
# P-62.2.1.1.1 — the substituted-'anilino' PREFERRED PREFIX (v29 P4-a)        #
# --------------------------------------------------------------------------- #
_PHENYL_STEM = 'phenyl'
_ANILINO_STEM = 'anilino'


def anilino_preferred_prefix(ring_prefix: Optional[str],
                            n_substituent: Optional[str] = None,
                            *, enclose: bool = True) -> Optional[str]:
    """The Blue Book PREFERRED PREFIX for a (fully substitutable) C6H5-NH- group.

    Governing rule, verbatim, under the heading chain ``## P-62.2 AMINES`` /
    ``### P-62.2.1 Primary amines`` / ``### P-62.2.1.1 Retained names`` /
    ``**P-62.2.1.1.1**`` (BlueBookV2.md:26139):

        "Aniline, for C6H5-NH2, is the only name for a primary amine retained as a
         preferred IUPAC name for which full substitution is permitted on the ring
         and the nitrogen atom. ... The prefix name 'anilino' is retained as the
         preferred prefix for C6H5-NH- with full substitution allowed. The name
         'phenylamino' may be used in general nomenclature."

    Corroborated by P-62.2.3 (heading BB:26298, "The substituent prefix name
    'anilino' is a preferred IUPAC prefix and substitution is allowed"), the
    retained-prefix tables (BB:17800 / BB:17860 / BB:24565) and the P-34.2.1.3
    worked explanation (BB:24605, "'anilino' is chosen as retained prefix preferred
    to 'phenylamino'").

    The Blue Book's own two-column pairs — PREFERRED PREFIX | general nomenclature:

        BB:26151   anilino                  | phenylamino
        BB:26153   4-chloroanilino          | (4-chlorophenyl)amino
        BB:26166   4-methylanilino          | (4-methylphenyl)amino  (not p-toluidino)

    so ``(<X>phenyl)amino`` -> ``<X>anilino``, LOCANTS UNCHANGED. The locants
    coincide by construction: ``decorated_ring_substituent_name`` numbers a
    carbocyclic ring with the free valence at locant 1 (the ``else:
    attachment_locant = 1`` branch above), and aniline's C-1 is the N-bearing
    carbon — the same atom. That is why all three Blue Book pairs carry identical
    locants.

    This is a HEAD-MORPHEME substitution on an already-CONSTRUCTED ring-substituent
    name, not a lookup, because the class is OPEN: the set of substituents a ring
    may carry is unbounded, so any finite table of anilino spellings would be wrong
    on its complement by construction.

    Enclosure, from BB:26306 vs BB:26308 — a prefix carrying its own locant(s) takes
    enclosing marks, one carrying none does not:

        BB:26306   3-anilinobenzoic acid (PIN)        | 3-(phenylamino)benzoic acid
        BB:26308   3-(N-methylanilino)phenol (PIN)    | 3-[methyl(phenyl)amino]phenol

    ★ SCOPE. This function only SPELLS a prefix. It must be called only where an
    anilino-family prefix is already the chosen construction; it must never be used
    to promote one over a senior name-selection criterion. Two Blue Book boundary
    rows pin that limit: BB:26419 (P-62.2.5.1 — multiplicative nomenclature beats
    BOTH substitutive forms) and BB:26404 (P-45.2.1 — maximum prefix count declines
    an anilino prefix outright, and marks the anilino-bearing alternative 'not').

    Args:
        ring_prefix: the ring substituent name, ``'phenyl'`` or ``'<X>phenyl'``
            (e.g. ``'4-chlorophenyl'``, ``'2,3-dimethylphenyl'``) — normally the
            return value of ``decorated_ring_substituent_name``. Anything else
            fails closed.
        n_substituent: the OTHER substituent on the nitrogen, as a substituent
            prefix name (``'methyl'``, ``'phenyl'``). An anilino nitrogen carries
            at most one (parent + ring + this), so a single name, not a list.
        enclose: apply the P-16.5.1.1 enclosing marks when the prefix carries its
            own locant(s). Callers that wrap the result themselves pass False.

    Returns:
        The preferred prefix, or ``None`` when the class boundary is not met — in
        which case the caller keeps its own construction, so routing a site through
        this helper can never silently drop an atom.
    """
    if not ring_prefix or not ring_prefix.endswith(_PHENYL_STEM):
        return None
    ring_decoration = ring_prefix[:-len(_PHENYL_STEM)]
    # A decoration must be a real prefix sequence, not a stray separator: the
    # only unadorned form is the bare stem itself.
    if ring_decoration.endswith('-') or ring_decoration.endswith(','):
        return None
    if ring_decoration and n_substituent:
        # Both an N locant and ring locants would have to be merged into ONE
        # alphanumerical prefix sequence (P-14.5.2). The derivation found no Blue
        # Book worked example fixing that merge, so decline rather than invent an
        # order. Measured 2026-07-28: production already abstains on this shape
        # (e.g. Oc1cccc(N(C)c2ccc(C)cc2)c1 -> 'unknown organic compound'), so
        # declining is byte-identical.
        return None
    if n_substituent:
        from ..assembly.naming_utils import _wrap_n_substituent
        core = f'N-{_wrap_n_substituent(n_substituent)}{_ANILINO_STEM}'
    else:
        core = f'{ring_decoration}{_ANILINO_STEM}'
    # BB:26306 bare vs BB:26308 enclosed: marks iff the prefix has its own locant.
    carries_own_locant = bool(ring_decoration) or bool(n_substituent)
    if enclose and carries_own_locant:
        from ..assembly.naming_utils import apply_enclosing_marks
        return apply_enclosing_marks(core, -1)
    return core


def anilino_prefix_from_aniline_name(aniline_name: Optional[str],
                                     *, enclose: bool = True) -> Optional[str]:
    """``'<X>aniline'`` -> the P-62.2.1.1.1 preferred prefix ``'<X>anilino'``.

    Third entry point for ``anilino_preferred_prefix``, for callers that hold an
    ASSEMBLED aniline parent name rather than a ring name plus branch names.

    The head-morpheme substitution leaves the prefix sequence untouched, and that is
    a derived fact, not an assumption: the Blue Book prints the parent and prefix
    forms with IDENTICAL decoration —

        BB:26147   4-chloroaniline (PIN)
        BB:26153   4-chloroanilino (preferred prefix)  | (4-chlorophenyl)amino
        BB:26162   4-methylaniline (PIN)
        BB:26166   4-methylanilino (preferred prefix)  | (4-methylphenyl)amino

    — and P-14.5.2 orders detachable prefixes among THEMSELVES; the head morpheme
    does not participate. So whatever order is correct for the aniline parent is
    correct for the anilino prefix, and this function inherits the ordering that
    ``rules/benzene.py``'s aniline joiner already computes (which merges N- and
    ring-locant prefixes into one alphanumerical sequence per P-14.5.2). That is
    what lets this path serve the ring-AND-nitrogen-substituted case that
    ``anilino_preferred_prefix`` declines: here the merge is not invented, it is
    inherited from a producer that already ships the parent form.

    Why it exists: ``assembly/substituent_naming.py``'s ``parent_to_prefix`` reached
    the P-31.1.3 heterocyclic '-ine' -> '-inyl' rule with '4-methyl-N-methylaniline'
    and produced '4-methyl-N-methylanilinyl'. Aniline is not a heterocycle and
    'anilinyl' is not a Blue Book morpheme; the preferred prefix is the retained
    'anilino' (BB:26139).

    Returns None (fail closed) for anything that is not an aniline-family name.
    """
    if not aniline_name or not aniline_name.endswith('aniline'):
        return None
    decoration = aniline_name[:-len('aniline')]
    if decoration.endswith(',') or (decoration and not decoration.endswith('-')
                                    and not decoration[-1].isalnum()
                                    and decoration[-1] not in ')]}'):
        return None
    # The joiner emits '<rendered>-<rendered>-...' immediately followed by the head
    # morpheme, so a well-formed decoration never ends in a separator.
    if decoration.endswith('-'):
        return None
    core = f'{decoration}{_ANILINO_STEM}'
    if enclose and decoration:
        from ..assembly.naming_utils import apply_enclosing_marks
        return apply_enclosing_marks(core, -1)
    return core


def anilino_prefix_from_n_branch(mol, n_idx: int, branch_atoms,
                                 *, enclose: bool = True) -> Optional[str]:
    """``anilino_preferred_prefix`` derived from the GRAPH rather than from a name.

    Entry point for the emission sites that had no unsubstituted-ring check at all
    and returned the bare literal ``"anilino"`` as soon as a 6-membered isolated
    all-carbon aromatic ring was found INSIDE the substituent —
    ``assembly/composer.py`` (two sites) and
    ``assembly/substituent_enumerator.py``. Requiring the ring atoms to be inside
    the branch says nothing about the ring's own substituents, which are also
    inside the branch and were never examined, so every ring substituent was
    silently dropped and the emitted name described a DIFFERENT molecule.

    The atom-drop is closed by construction here: the ring name is built by
    ``decorated_ring_substituent_name`` with ``expected_atoms`` set to the WHOLE
    branch minus the nitrogen, and that function returns None unless the ring plus
    its detected decoration accounts for EXACTLY that set.

    Args:
        mol: the molecule.
        n_idx: the amine nitrogen.
        branch_atoms: the complete N-substituent branch (the nitrogen itself may be
            included or not — it is normalised in).
        enclose: as for ``anilino_preferred_prefix``.

    Returns:
        The preferred prefix, or ``None`` — fail closed — when the branch is not
        exactly a nitrogen plus one isolated benzene ring plus that ring's own
        nameable decoration.
    """
    branch_set = set(branch_atoms) | {n_idx}
    ring_info = mol.GetRingInfo()
    n_neighbours = {nbr.GetIdx() for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors()}

    candidate = None
    for ring in ring_info.AtomRings():
        if len(ring) != 6 or not set(ring) <= branch_set:
            continue
        if not (set(ring) & n_neighbours):
            continue  # a ring elsewhere in the branch is not the anilino ring
        # Isolated benzene only: a benzo sub-ring of a fused system is aromatic and
        # all-carbon but is NOT a C6H5- group (the naphthalene/quinoline trap the
        # legacy 'is_fused' checks were guarding).
        if any(ring_info.NumAtomRings(i) != 1 for i in ring):
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic()
                   and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                   and mol.GetAtomWithIdx(i).GetFormalCharge() == 0
                   for i in ring):
            continue
        if candidate is not None:
            return None  # two candidate rings on one N: ambiguous, fail closed
        candidate = ring
    if candidate is None:
        return None

    attachment = sorted(set(candidate) & n_neighbours)
    if len(attachment) != 1:
        return None  # the N bridges two atoms of the same ring: not an anilino
    rest = branch_set - {n_idx} - set(candidate)
    if not rest:
        # Unsubstituted C6H5-NH-: the bare retained prefix (BB:26151), which is
        # byte-identical to what these sites emitted before.
        return anilino_preferred_prefix(_PHENYL_STEM, enclose=enclose)
    decorated = decorated_ring_substituent_name(
        mol, candidate, attachment[0], expected_atoms=branch_set - {n_idx},
    )
    return anilino_preferred_prefix(decorated, enclose=enclose)


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
