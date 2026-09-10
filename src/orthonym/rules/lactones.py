"""
Monocyclic lactone naming using IUPAC heterocyclic replacement nomenclature.

Lactones are cyclic esters. Monocyclic lactones are named as heterocyclic
ketones: the ring oxygen gives the heterocyclic parent name, and the
carbonyl is expressed as a -one suffix at position 2.

Naming algorithm:
1. Detect lactone: ring contains -C(=O)-O- where both C and ester O are
   in the same ring, and the exocyclic O is a double-bonded carbonyl.
2. Determine ring size.
3. For ring size 3-10: Get the Hantzsch-Widman heterocyclic parent name:
   - 3: oxirane, 4: oxetane, 5: oxolane, 6: oxane, 7: oxepane, etc.
4. For ring size 11+: Use replacement nomenclature with chain prefix:
   - 11: oxacycloundecan, 13: oxacyclotridecan, 15: oxacyclopentadecan
5. Apply vowel elision and append '-2-one'.

Reference: IUPAC 2013 Blue Book, (Lactones), (Replacement)

Examples:
    O=C1CCO1 (beta-propiolactone) -> oxetan-2-one
    O=C1CCCO1 (gamma-butyrolactone) -> oxolan-2-one
    O=C1CCCCO1 (delta-valerolactone) -> oxan-2-one
    O=C1CCCCCO1 (epsilon-caprolactone) -> oxepan-2-one
    O=C1CCCCCCCCCO1 (10-membered lactone) -> oxacycloundecan-2-one
"""

from collections import deque
from typing import Dict, List, Optional

from rdkit import Chem

from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)
from ..rules.heterocycles import build_hw_name

# ---------------------------------------------------------------------------
# Lactone detection
# ---------------------------------------------------------------------------

def is_monocyclic_lactone(mol) -> Optional[Dict]:
    """
    Detect whether a molecule is (or contains) a monocyclic lactone.

    A monocyclic lactone has:
    - A ring containing an ester motif: -C(=O)-O- where both the carbonyl
      carbon and the ester oxygen are in the same ring.
    - The carbonyl oxygen is exocyclic (double-bonded to the carbonyl C).
    - The ring is not fused with another ring (monocyclic only).

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        Dict with detection info if lactone found:
            ring_atoms: tuple of atom indices in the lactone ring
            carbonyl_idx: atom index of the carbonyl carbon (C=O)
            ester_O_idx: atom index of the ring (ester) oxygen
            carbonyl_O_idx: atom index of the exocyclic carbonyl oxygen
            ring_size: number of atoms in the ring
        None if not a monocyclic lactone.
    """
    if mol is None:
        return None

    # SMARTS: carbonyl(-like) carbon with a double-bonded chalcogen and a
    # single-bonded ester O. The double-bond partner may be O (ordinary
    # lactone) or S/Se/Te (thiono / seleno / telluro lactone,;
    # the ester O stays O in every case (the ring oxygen).
    # [CX3](=[O,S,#34,#52])[OX2]
    # match[0] = carbonyl carbon
    # match[1] = carbonyl chalcogen (=O/=S/=Se/=Te, exocyclic)
    # match[2] = ester oxygen (-O-, must be in ring)
    pattern = Chem.MolFromSmarts("[CX3](=[O,S,#34,#52])[OX2]")
    matches = mol.GetSubstructMatches(pattern)

    if not matches:
        return None

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        return None

    for match in matches:
        carbonyl_c = match[0]
        carbonyl_o = match[1]
        ester_o = match[2]

        # Both carbonyl C and ester O must be in the SAME ring
        for ring in atom_rings:
            ring_set = set(ring)
            if carbonyl_c in ring_set and ester_o in ring_set:
                # Exocyclic carbonyl O must NOT be in the ring
                if carbonyl_o in ring_set:
                    continue

                # Check monocyclic: no ring atom should appear in another ring
                is_monocyclic = True
                for other_ring in atom_rings:
                    if set(other_ring) == ring_set:
                        continue
                    if ring_set & set(other_ring):
                        is_monocyclic = False
                        break

                if not is_monocyclic:
                    continue

                # (a) cyclic carbonate: carbonyl bonded to TWO
                # ring oxygens -> 1,3-dioxan-2-one class. Detect a SECOND
                # ring O bonded to the carbonyl C.
                extra_o = None
                for nbr in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
                    i = nbr.GetIdx()
                    if i in ring_set and i != ester_o and nbr.GetAtomicNum() == 8:
                        extra_o = i
                # Any OTHER ring heteroatom -> not this namer's class
                # (fail closed). Plain lactones have an all-carbon ring
                # besides the ester O (and the optional carbonate extra O).
                for i in ring_set:
                    if i in (ester_o, extra_o):
                        continue
                    if mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
                        return None

                # a phase (A): unsaturated / dione monocyclic lactones.
                # name_lactone_ring/name_monocyclic_lactone only builds a
                # SATURATED single-oxo stem (oxacyclo...an-2-one) -- an
                # in-ring C=C or a second in-ring carbonyl would otherwise
                # be silently dropped, naming a different (more saturated)
                # molecule than the input.
                #
                # HW-range rings (<=10): the PIN for an unsaturated ring
                # ketone here uses a DIFFERENT scheme entirely -- indicated
                # hydrogen + hydro prefixes on the mancude aromatic parent
                # (furan-2(3H)-one, 3,4-dihydro-2H-pyran-2-one) -- which
                # this module does not build. DECLINE so dispatch routes to
                # the general heterocyclic-ketone engine
                # (rules.heterocycles.name_heterocycle + the substituted-
                # heterocycle assembler), which already has that machinery.
                #
                # Ring sizes >10 have no HW mancude parent at all; 's
                # replacement-nomenclature PIN there is direct ene-locant
                # citation on the SAME fixed O=1/C=2 numbering this module
                # already uses for the saturated case
                # (oxacyclotridec-10-en-2-one) -- built directly below via
                # ``ring_double_bonds`` instead of declining. (Measured: the
                # general engine's >10 replacement-nomenclature path has no
                # suffix-locant-aware numbering and names a constitutionally
                # different molecule for this size range.)
                ring_double_bonds = []
                for b in mol.GetBonds():
                    a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    if (a1 in ring_set and a2 in ring_set
                            and b.GetBondTypeAsDouble() == 2.0):
                        ring_double_bonds.append((a1, a2))
                if ring_double_bonds and len(ring) <= 10:
                    return None  # HW range -> general engine's indicated-H scheme

                for i in ring_set:
                    if i == carbonyl_c:
                        continue
                    at = mol.GetAtomWithIdx(i)
                    if at.GetSymbol() != "C":
                        continue
                    for nbr in at.GetNeighbors():
                        j = nbr.GetIdx()
                        if j in ring_set:
                            continue
                        if nbr.GetSymbol() not in ("O", "S", "Se", "Te"):
                            continue
                        bond2 = mol.GetBondBetweenAtoms(i, j)
                        if bond2 is not None and bond2.GetBondType() == Chem.BondType.DOUBLE:
                            return None  # 2nd in-ring carbonyl -> dione class

                return {
                    "ring_atoms": ring,
                    "carbonyl_idx": carbonyl_c,
                    "ester_O_idx": ester_o,
                    "carbonyl_O_idx": carbonyl_o,
                    "ring_size": len(ring),
                    "extra_ring_O_idx": extra_o,
                    # Exocyclic double-bond partner symbol: 'O' (ordinary
                    # lactone) or 'S'/'Se'/'Te' (thiono/seleno/telluro lactone,
                    # -> -thione/-selone/-tellone suffix).
                    "chalcogen": mol.GetAtomWithIdx(carbonyl_o).GetSymbol(),
                    # a phase (A): in-ring C=C atom-idx pairs (empty for
                    # the saturated case). Only ever non-empty here for
                    # ring_size > 10 (the HW range declines above), so the
                    # macrocyclic ene-locant path is the sole consumer.
                    "ring_double_bonds": ring_double_bonds,
                }

    return None


# ---------------------------------------------------------------------------
# Lactone ring naming
# ---------------------------------------------------------------------------

# Ring sizes supported via Hantzsch-Widman naming
_HW_RING_SIZES = frozenset(range(3, 11))

# Maximum ring size for macrolide lactone naming
_MAX_MACROLIDE_SIZE = 50

# Exocyclic-chalcogen -> lactone suffix. The ordinary lactone
# (=O) keeps '-one'; the thiono/seleno/telluro lactones take -thione/-selone/
# -tellone. Vowel elision is suffix-driven: '-one' elides the parent's terminal
# 'e' (oxolan-2-one) but the consonant-initial -thione does NOT (oxolane-2-thione).
_LACTONE_CHALCOGEN_SUFFIX = {"O": "one", "S": "thione", "Se": "selone", "Te": "tellone"}
_ELISION_VOWELS = "aeiouy"


def _join_lactone_suffix(parent_name: str, locant: int, suffix: str) -> str:
    """Join an HW parent name with a '-{locant}-{suffix}' lactone/thiono suffix,
    eliding the parent's terminal 'e' only when ``suffix`` begins with an elision
    vowel ('one' -> oxolan-2-one; 'thione' -> oxolane-2-thione)."""
    if parent_name.endswith("e") and suffix[:1] in _ELISION_VOWELS:
        stem = parent_name[:-1]
    else:
        stem = parent_name
    return f"{stem}-{locant}-{suffix}"


def name_lactone_ring(ring_size: int, extra_o_locant: Optional[int] = None,
                      chalcogen: str = "O",
                      ene_locants: Optional[List[int]] = None) -> Optional[str]:
    """
    Get the IUPAC name for a monocyclic lactone of a given ring size.

    For ring sizes 3-10: uses Hantzsch-Widman heterocyclic parent naming.
    For ring sizes 11+: uses replacement nomenclature (oxacyclo{prefix}an-2-one).

    Args:
        ring_size: Number of atoms in the lactone ring (3-50 supported).
        extra_o_locant: locant of a SECOND ring oxygen bonded to the
            carbonyl carbon (cyclic carbonate, (a)). When set to
            3 (the only geometry this namer describes), builds the
            1,3-dioxa Hantzsch-Widman parent (1,3-dioxan-2-one). Any other
            value returns None (fail closed).
        ene_locants: a phase (A). Sorted list of the lower-numbered
            locant of each in-ring C=C, on the SAME fixed O=1/C=2 numbering
            this function already uses. Ring sizes 11+ ONLY (the HW range
            has no replacement-nomenclature ring stem to attach an ene
            locant to -- ``is_monocyclic_lactone`` never supplies this for
            ring_size<=10). ``None``/empty -> the existing saturated
            ``...an`` stem (byte-identical to the pre-Phase-6 behaviour).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one'), or None if ring size
        is not supported.

    Examples:
        >>> name_lactone_ring(4)
        'oxetan-2-one'
        >>> name_lactone_ring(5)
        'oxolan-2-one'
        >>> name_lactone_ring(6)
        'oxan-2-one'
        >>> name_lactone_ring(7)
        'oxepan-2-one'
        >>> name_lactone_ring(11)
        'oxacycloundecan-2-one'
        >>> name_lactone_ring(13)
        'oxacyclotridecan-2-one'
        >>> name_lactone_ring(6, extra_o_locant=3)
        '1,3-dioxan-2-one'
    """
    if ring_size < 3:
        return None

    suffix = _LACTONE_CHALCOGEN_SUFFIX.get(chalcogen)
    if suffix is None:
        return None

    # (a) cyclic carbonate: two ring oxygens bonded to the
    # carbonyl C. Only the 1,3 geometry (extra O at locant 3) is a valid
    # dioxanone/dioxolanone; anything else is out of this namer's scope.
    if extra_o_locant is not None:
        if extra_o_locant != 3 or ring_size not in _HW_RING_SIZES:
            return None
        parent_name = build_hw_name(
            heteroatoms=[(1, "O"), (3, "O")],
            ring_size=ring_size,
            is_saturated=True,
            is_aromatic=False,
        )
        if not parent_name:
            return None
        # build_hw_name already emits the '1,3-' locant prefix.
        return _join_lactone_suffix(parent_name, 2, suffix)

    # Hantzsch-Widman naming for ring sizes 3-10
    if ring_size in _HW_RING_SIZES:
        parent_name = build_hw_name(
            heteroatoms=[(1, "O")],
            ring_size=ring_size,
            is_saturated=True,
            is_aromatic=False,
        )

        if not parent_name:
            return None

        # Suffix-driven vowel elision ('-one' elides, '-thione' does not).
        return _join_lactone_suffix(parent_name, 2, suffix)

    # Macrolide naming for ring sizes 11+
    # Uses replacement nomenclature: oxacyclo{chain_prefix}an-2-one
    if ring_size > _MAX_MACROLIDE_SIZE:
        return None

    from ..data.chain_names import get_chain_prefix

    try:
        # The chain prefix corresponds to the total ring size
        # (since one O replaces one C, the ring name uses the total count)
        chain_prefix = get_chain_prefix(ring_size)
    except ValueError:
        return None

    if not ene_locants:
        # Build: oxacyclo + {prefix} + an-2-{suffix}
        # The chain prefix already provides the stem (e.g., "undec" for 11);
        # 'oxacyclo...an' ends in a consonant so no elision applies.
        return f"oxacyclo{chain_prefix}an-2-{suffix}"

    # a phase (A): unsaturated macrocyclic lactone. Replace the
    # saturated '...an' stem with the standard cycloalkENE construction
    # ('...an' dropped, '-{locants}-{mult}ene' takes its place -- e.g.
    # cyclotridecane -> cyclotridec-10-ene), then join the -one suffix
    # through the SAME elision rule as every other branch here ('ene' + a
    # vowel-initial suffix elides its terminal 'e': tridec-10-en-2-one).
    locs = sorted(ene_locants)
    loc_str = ",".join(str(l) for l in locs)
    mult = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}.get(len(locs))
    if mult is None:
        return None  # more double bonds than this namer enumerates -> fail closed
    ene_stem = f"oxacyclo{chain_prefix}-{loc_str}-{mult}ene"
    return _join_lactone_suffix(ene_stem, 2, suffix)


# ---------------------------------------------------------------------------
# Full lactone naming
# ---------------------------------------------------------------------------

def name_monocyclic_lactone(mol) -> Optional[str]:
    """
    Generate the IUPAC name for a monocyclic lactone, including substituents.

    Detects whether the molecule is a monocyclic lactone and returns
    its name using heterocyclic replacement nomenclature with -one suffix.
    Substituents on the ring are included as prefixes with locants.

    Args:
        mol: RDKit Mol object (or None).

    Returns:
        IUPAC name string (e.g., 'oxolan-2-one', '3-aminooxolan-2-one'),
        or None if the molecule is not a monocyclic lactone.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('O=C1CCCO1')
        >>> name_monocyclic_lactone(mol)
        'oxolan-2-one'
        >>> mol = Chem.MolFromSmiles('NC1CCOC1=O')
        >>> name_monocyclic_lactone(mol)
        '3-aminooxolan-2-one'
    """
    info = is_monocyclic_lactone(mol)
    if info is None:
        return None

    # Detect substituents on the lactone ring
    ring_atoms = info["ring_atoms"]
    ester_o_idx = info["ester_O_idx"]
    carbonyl_idx = info["carbonyl_idx"]
    carbonyl_o_idx = info["carbonyl_O_idx"]

    # Build IUPAC locant mapping for lactone ring
    # Numbering: O atom = 1, carbonyl C = 2, then continue around ring
    ring_list = list(ring_atoms)
    ring_set = set(ring_list)

    # Find oxygen position in ring and reorder so O is first
    try:
        o_pos = ring_list.index(ester_o_idx)
    except ValueError:
        o_pos = None

    if o_pos is None:
        ordered = None
        atom_to_locant = {}
    else:
        # Reorder ring starting from O, going toward carbonyl C
        ordered = ring_list[o_pos:] + ring_list[:o_pos]

        # Check direction: next atom should be carbonyl C
        if len(ordered) > 1 and ordered[1] != carbonyl_idx:
            # Reverse direction (keep O first)
            ordered = [ordered[0]] + ordered[1:][::-1]

        # Build atom-to-locant mapping (1-indexed)
        atom_to_locant = {atom_idx: i + 1 for i, atom_idx in enumerate(ordered)}

    # a phase (A): in-ring C=C locants (macrocyclic only -- see
    # name_lactone_ring's ene_locants docstring) on this SAME fixed
    # numbering. Each bond's cited locant is the LOWER of its two atoms'
    # locants; safe without a wraparound check because the only ring
    # heteroatom (O, locant 1) never carries a ring double bond in this
    # namer's scope (fail-closed all-carbon-besides-O check above), so a
    # double bond can never span the ring-closing (n, 1) pair.
    ene_locants = None
    _dbl = info.get("ring_double_bonds")
    if _dbl and atom_to_locant:
        ene_locants = sorted(
            min(atom_to_locant[a1], atom_to_locant[a2])
            for a1, a2 in _dbl
            if a1 in atom_to_locant and a2 in atom_to_locant
        )
        if len(ene_locants) != len(_dbl):
            return None  # a double-bond atom fell outside the numbering -> fail closed

    # (a) cyclic carbonate: the second ring O bonded to the
    # carbonyl C sits at locant 3 by the O=1, carbonyl C=2 numbering.
    extra_o_locant = 3 if info.get("extra_ring_O_idx") is not None else None
    parent_name = name_lactone_ring(
        info["ring_size"], extra_o_locant=extra_o_locant,
        chalcogen=info.get("chalcogen", "O"),
        ene_locants=ene_locants,
    )
    if parent_name is None:
        return None

    if ordered is None:
        return parent_name

    # Collect stereodescriptors using lactone ring locant mapping
    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string

    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, atom_to_locant)

    # Discover exocyclic substituents via universal pipeline (a phase).
    # Parent atoms = ring atoms; exclude = carbonyl O (=O of the lactone).
    # The _integrate_universal_prefixes helper adds exclude_atoms to the
    # effective parent set so they are never discovered as substituents.
    from ..assembly.composer import _integrate_universal_prefixes
    prefix_str = _integrate_universal_prefixes(
        mol, ring_set,
        parent_type="ring",
        oriented_ring=ordered,
        atom_to_locant=atom_to_locant,
        exclude_atoms={carbonyl_o_idx},
    )

    if not prefix_str:
        # No substituents but may have stereo
        if stereo_descriptors:
            stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
            return f"{stereo_prefix}{parent_name}"
        return parent_name

    name = f"{prefix_str}{parent_name}"

    # Prepend stereo prefix if descriptors exist
    if stereo_descriptors:
        stereo_prefix = format_stereodescriptor_string(stereo_descriptors)
        name = f"{stereo_prefix}{name}"

    return name


# ---------------------------------------------------------------------------
# Cyclic di/polyester (lactide) —
# ---------------------------------------------------------------------------

def name_cyclic_polyester(mol) -> Optional[str]:
    """: name a cyclic di-/polyester (lactide) as a Hantzsch-Widman
    heterocycle whose acyl carbons are expressed with a ``-dione``/``-trione``
    suffix::

        O=C1COC(=O)CO1 (glycolide) -> 1,4-dioxane-2,5-dione (PIN)

    The single-carbonyl lactone namer (:func:`name_monocyclic_lactone`) fails
    closed on this class because a SECOND ring oxygen is itself an ester O
    bearing its own ring carbonyl. Here every ring O is an ester oxygen (bonded
    to exactly ONE ring carbonyl C) and there are >=2 such carbonyls, so the
    ring is numbered with the O heteroatoms lowest (HW) and the carbonyls become
    a multiplied ``-one`` suffix.

    Fail-closed scope (accuracy-first, never a wrong name): one saturated
    monocyclic ring of only C and O (ring size 3-10); >=2 ring ester carbonyls;
    every ring O bonded to exactly one ring carbonyl C. This DECLINES:
      * the ordinary single lactone (1 carbonyl, handled upstream),
      * the cyclic carbonate (both ring O on one carbonyl -> len(o_nbrs)!=1),
      * the cyclic anhydride (a bridging ring O bonded to TWO carbonyls),
      * any ring bearing an unnameable substituent.
    """
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

    # Ring atoms restricted to C/O; skeleton fully saturated (C=O exocyclic).
    for i in ring:
        at = mol.GetAtomWithIdx(i)
        if at.GetSymbol() not in ("C", "O") or at.GetIsAromatic():
            return None
    for b in bonds_of(mol):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring_set and j in ring_set and b.GetBondType() != Chem.BondType.SINGLE:
            return None

    carbonyls: List[int] = []
    ester_os: List[int] = []
    carbonyl_o: Dict[int, int] = {}
    for i in ring:
        at = mol.GetAtomWithIdx(i)
        if at.GetSymbol() == "O":
            if at.GetTotalNumHs() != 0:
                return None
            nbrs = [nb.GetIdx() for nb in at.GetNeighbors()]
            if len(nbrs) != 2 or any(x not in ring_set for x in nbrs):
                return None
            ester_os.append(i)
        else:  # carbon
            oxo = None
            for nb in at.GetNeighbors():
                if nb.GetIdx() in ring_set:
                    continue
                bond = mol.GetBondBetweenAtoms(i, nb.GetIdx())
                if (nb.GetSymbol() == "O" and nb.GetDegree() == 1
                        and nb.GetTotalNumHs() == 0
                        and bond.GetBondType() == Chem.BondType.DOUBLE):
                    if oxo is not None:
                        return None  # two =O on one ring C -> not this class
                    oxo = nb.GetIdx()
            if oxo is not None:
                carbonyls.append(i)
                carbonyl_o[i] = oxo
    if len(carbonyls) < 2 or not ester_os:
        return None

    carbonyl_set = set(carbonyls)
    # Every ring O is an ester O: exactly ONE ring-carbonyl neighbour (excludes
    # ether O [0 carbonyls] and the anhydride bridge O [2 carbonyls]).
    for o in ester_os:
        cnt = sum(1 for nb in mol.GetAtomWithIdx(o).GetNeighbors()
                  if nb.GetIdx() in carbonyl_set)
        if cnt != 1:
            return None
    # Every ring carbonyl is an ester carbonyl: exactly one ring-O neighbour
    # (excludes carbonate/anhydride carbonyls bonded to two ring O).
    for c in carbonyls:
        o_nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(c).GetNeighbors()
                  if nb.GetIdx() in ring_set
                  and mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol() == "O"]
        if len(o_nbrs) != 1:
            return None

    # Exocyclic heavy substituents on each ring atom (excluding the carbonyl =O).
    def _exo_heavy(i):
        out = []
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j in ring_set:
                continue
            if i in carbonyl_o and j == carbonyl_o[i]:
                continue
            if nb.GetAtomicNum() > 1:
                out.append(j)
        return out

    sub_atoms = {i for i in ring if _exo_heavy(i)}

    # Numbering: heteroatom(O) locants lowest as a set, then carbonyl(-one)
    # locants lowest, then substituent locants lowest (RingInfo returns the
    # ring in cyclic-adjacency order; enumerate all rotations x both directions).
    best = None
    for start in range(n):
        for direction in (1, -1):
            order = [ring[(start + direction * k) % n] for k in range(n)]
            loc = {a: k + 1 for k, a in enumerate(order)}
            key = (
                tuple(sorted(loc[a] for a in ester_os)),
                tuple(sorted(loc[a] for a in carbonyls)),
                tuple(sorted(loc[a] for a in sub_atoms)),
            )
            if best is None or key < best[0]:
                best = (key, order, loc)
    if best is None:
        return None
    _, order, loc = best

    o_locs = sorted(loc[a] for a in ester_os)
    parent = build_hw_name(
        heteroatoms=[(l, "O") for l in o_locs],
        ring_size=n, is_saturated=True, is_aromatic=False,
    )
    if not parent:
        return None

    c_locs = sorted(loc[a] for a in carbonyls)
    mult = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}.get(len(c_locs))
    if mult is None:
        return None
    suffix = f"{mult}one"  # 'dione' / 'trione' (consonant-initial: no elision)
    loc_str = ",".join(str(x) for x in c_locs)
    stem = parent[:-1] if (parent.endswith("e") and suffix[0] in _ELISION_VOWELS) else parent
    core = f"{stem}-{loc_str}-{suffix}"

    # Substituents (fail-closed) — mirror rules.anhydrides._name_saturated_oxa_dione.
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    from ..assembly.substituent_enumerator import name_substituent
    subs = []  # (locant, name)
    for a in sub_atoms:
        for attach in _exo_heavy(a):
            frag, seen, stack = [], set(ring_set), [attach]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                frag.append(x)
                stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors()
                             if nb.GetIdx() not in seen)
            nm = name_substituent(mol, frag, attach)
            if not nm:
                return None  # unnameable substituent -> fail closed
            subs.append((loc[a], nm))

    from ..perception.stereo import assign_stereochemistry
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    assign_stereochemistry(mol)
    stereo_descriptors = collect_stereodescriptors(mol, loc)

    prefix = ""
    if subs:
        groups: Dict[str, List[int]] = {}
        for locant, nm in subs:
            groups.setdefault(nm, []).append(locant)
        parts = []
        for nm in sorted(groups, key=alpha_sort_key):
            locs = sorted(groups[nm])
            m = get_multiplier_prefix(len(locs), nm) if len(locs) > 1 else ""
            parts.append(f"{','.join(str(x) for x in locs)}-{m}{nm}")
        prefix = "-".join(parts)

    if prefix:
        sep = "-" if core[:1].isdigit() else ""
        name = f"{prefix}{sep}{core}"
    else:
        name = core
    if stereo_descriptors:
        name = f"{format_stereodescriptor_string(stereo_descriptors)}{name}"
    return name


def _detect_lactone_substituents(mol, ordered_ring, atom_to_locant, excluded):
    """
    Detect substituents on a lactone ring.

    Args:
        mol: RDKit Mol object
        ordered_ring: List of atom indices in IUPAC numbering order
        atom_to_locant: Dict mapping atom index to IUPAC locant
        excluded: Set of atom indices to exclude (ring + carbonyl O)

    Returns:
        List of (substituent_name, locant) tuples
    """
    substituents = []

    for ring_atom_idx in ordered_ring:
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        locant = atom_to_locant[ring_atom_idx]

        for neighbor in ring_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in excluded:
                continue

            sub_name = _identify_lactone_substituent(mol, nbr_idx, excluded)
            if sub_name:
                substituents.append((sub_name, locant))

    return substituents


def _identify_lactone_substituent(mol, start_idx, excluded):
    """
    Identify a substituent on a lactone ring by its starting atom.

    Returns:
        Substituent prefix name (e.g., 'amino', 'methyl', 'hydroxy') or None
    """
    atom = mol.GetAtomWithIdx(start_idx)
    symbol = atom.GetSymbol()

    # Halogens
    halogen_map = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    if symbol in halogen_map:
        return halogen_map[symbol]

    # Nitrogen: amino (-NH2), nitro, etc.
    if symbol == 'N':
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 2 and len(neighbors) == 0:
            return 'amino'
        if atom.GetFormalCharge() == 1:
            o_count = sum(1 for n in neighbors if n.GetSymbol() == 'O')
            if o_count == 2:
                return 'nitro'

    # Oxygen: hydroxy
    if symbol == 'O':
        h_count = atom.GetTotalNumHs()
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() not in excluded]
        if h_count == 1 and len(neighbors) == 0:
            return 'hydroxy'

    # Carbon: alkyl groups
    if symbol == 'C':
        # BFS for pure alkyl
        visited = {start_idx}
        queue = deque([start_idx])
        all_atoms = []
        carbon_count = 0
        is_pure_alkyl = True

        while queue:
            idx = queue.popleft()
            a = mol.GetAtomWithIdx(idx)
            all_atoms.append(idx)
            if a.GetSymbol() == 'C':
                carbon_count += 1
            elif a.GetSymbol() != 'H':
                is_pure_alkyl = False

            for nbr in a.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited and nbr_idx not in excluded:
                    visited.add(nbr_idx)
                    queue.append(nbr_idx)

        if is_pure_alkyl and carbon_count > 0:
            from ..assembly.naming_utils import get_alkyl_name
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass

    return None
