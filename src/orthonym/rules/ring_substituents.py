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
        else:
            # Saturated 6-membered
            if not heteroatoms:
                return 'cyclohexane'
            elif heteroatoms == ['O']:
                return 'oxane'
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
}

# Element seniority for assigning low locants to heteroatoms (IUPAC Table 28:
# F > Cl > Br > I > O > S > Se > Te > N > P > ...). Lower value = senior.
_HETEROATOM_SENIORITY: Dict[str, int] = {
    'F': 0, 'Cl': 1, 'Br': 2, 'I': 3,
    'O': 4, 'S': 5, 'Se': 6, 'Te': 7,
    'N': 8, 'P': 9, 'As': 10, 'Sb': 11, 'Bi': 12,
    'Si': 13, 'Ge': 14, 'Sn': 15, 'Pb': 16, 'B': 17,
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


def name_ring_system_substituent(
    mol,
    frag_atoms,
    attach_idx: int,
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
    if not name:
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
    carrier = frag_set - frag_ring_atoms
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
    # No stray decorations: every fragment atom is carrier or ring.
    if carrier | frag_ring_atoms != frag_set:
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
    if len(path) == 1:
        return f'({ring_name})methyl'
    return f'{loc}-({ring_name}){alkyl}'


def get_ring_substituent_name(
    mol,
    ring_atoms: Tuple[int, ...],
    attachment_point: Optional[int] = None
) -> str:
    """
    Get the substituent name for a ring when it becomes a substituent on a chain.

    This is the main entry point for ring-as-substituent naming.

    Args:
        mol: RDKit Mol object
        ring_atoms: Tuple of atom indices in the ring
        attachment_point: Optional ring atom index where the ring attaches to chain.
                         Used for position-specific names (e.g., 2-pyridyl vs 4-pyridyl).

    Returns:
        Substituent name string (e.g., 'phenyl', 'cyclohexyl', '2-pyridyl')
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
                return f'{stem}-yl'

        # Unknown ring - generate generic cycloXyl name
        ring_size = len(ring_atoms)
        prefix = _get_chain_prefix(ring_size)
        return f'cyclo{prefix}yl'

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
        # cyano (-C#N) or unbranched pure alkyl
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
        # (1b) element seniority at the heteroatom positions: F,Cl,Br,I,O,S,
        # Se,Te,N,P (P-22.2.3.1); lower rank = more senior = lower locant.
        _SENIORITY = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P']
        def het_rank_seq(o):
            return tuple(
                _SENIORITY.index(mol.GetAtomWithIdx(o[p]).GetSymbol())
                if mol.GetAtomWithIdx(o[p]).GetSymbol() in _SENIORITY else 99
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


def ring_atom_fg_prefixes(mol, ring_atom_idx: int, ring_atom_set: Set[int]) -> List[str]:
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
    """
    prefixes: List[str] = []
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
                continue
    return sorted(prefixes)
