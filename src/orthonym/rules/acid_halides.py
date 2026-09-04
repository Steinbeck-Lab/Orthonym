"""
Acid halide naming using IUPAC functional class nomenclature.

Acid halides use two-word "functional class" naming:
    {acyl name} {halide word}

Examples:
    CC(=O)Cl     -> acetyl chloride (retained acyl name for C2)
    CCC(=O)Cl    -> propanoyl chloride
    O=C(Cl)c1ccccc1 -> benzoyl chloride (retained acyl name for benzene)
    ClC(=O)CCCC(=O)Cl -> pentanedioyl dichloride (diacid halide)

The acyl name derives from the corresponding acid:
    ethanoic acid  -> ethanoyl  (systematic)
    acetic acid    -> acetyl    (retained, preferred for C2)
    formic acid    -> formyl    (retained, preferred for C1)
    benzoic acid   -> benzoyl   (retained, preferred for benzene-attached)

Reference: IUPAC 2013 Blue Book, P-65.5.1 (Acyl halides)
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from ..data.chain_names import get_chain_prefix

# Retained acyl names (chain length -> retained acyl prefix)
RETAINED_ACYL_NAMES = {
    1: "formyl",     # from formic acid
    2: "acetyl",     # from acetic acid
}

# Halide word mapping
HALIDE_WORDS = {
    "acid_chloride": "chloride",
    "acid_bromide": "bromide",
    "acid_fluoride": "fluoride",
    "acid_iodide": "iodide",
    # Wave2 T6c: acyl pseudohalides (P-65.5.2.1) — the same two-word
    # functional-class grammar ('butanoyl azide', 'propanoyl cyanide',
    # 'acetyl isocyanate', all BB-verbatim PINs).
    "acyl_azide": "azide",
    "acyl_cyanide": "cyanide",
    "acyl_isocyanate": "isocyanate",
}

# Halogen prefix names (for substituent halogens that are NOT part of acid halide)
HALOGEN_PREFIX = {
    "Cl": "chloro",
    "Br": "bromo",
    "F": "fluoro",
}

# W3-P06: halide functional-class word keyed by element symbol, cited in
# ALPHABETICAL order bromide < chloride < fluoride < iodide (P-65.5.1 / P-65.5.3.2).
_HALIDE_WORD = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}


# P-65.2.1: the acid-anion word of a mono-ester of carbonic acid, keyed by the
# acyl-halide FG type. 'chloride' -> 'carbonochloridate', etc. (OPSIN-verified).
_CARBONO_HALIDATE_WORDS = {
    "acid_chloride": "carbonochloridate",
    "acid_bromide": "carbonobromidate",
    "acid_fluoride": "carbonofluoridate",
    "acid_iodide": "carbonoiodidate",
}


def name_carbonic_monoester_acyl_halide(
    mol, match, halide_word: str
) -> Optional[str]:
    """P-35.4.2/P-65.2.1: X-C(=O)-O-R  ->  '<R> carbono<halide>idate'.

    ``match`` is the acyl-halide SMARTS tuple (carbonyl C, carbonyl O, halide).
    Returns None (fail-closed) unless the carbonyl C is a genuine carbonic-acid
    mono-ester acyl halide: exactly one =O, one halide, one single-bonded ester
    O leading to a fully nameable R group, and NO carbon neighbour on the
    carbonyl. R is named with the universal substituent namer (benzyl, ethyl).
    """
    from ..assembly.substituent_naming import name_substituent_fragment

    carbonyl_c = match[0]
    c_atom = mol.GetAtomWithIdx(carbonyl_c)
    if c_atom.GetSymbol() != 'C':
        return None
    halide_key = None
    for k, w in HALIDE_WORDS.items():
        if w == halide_word:
            halide_key = k
            break
    carbono_word = _CARBONO_HALIDATE_WORDS.get(halide_key)
    if carbono_word is None:
        return None

    o_double = o_single = ester_o = halide = carbon_nbr = None
    n_double = n_single = 0
    for nb in c_atom.GetNeighbors():
        bt = mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()).GetBondTypeAsDouble()
        sym = nb.GetSymbol()
        if sym == 'O' and bt == 2.0:
            o_double = nb.GetIdx()
        elif sym == 'O' and bt == 1.0:
            # ester O must carry the alkyl R and no H (an -OH would be the acid)
            if nb.GetTotalNumHs() != 0:
                return None
            ester_o = nb.GetIdx()
        elif sym in ('Cl', 'Br', 'F', 'I'):
            halide = nb.GetIdx()
        elif sym == 'C':
            carbon_nbr = nb.GetIdx()
        else:
            return None
    # Exactly the carbonic mono-ester acyl-halide skeleton, no C chain.
    if o_double is None or ester_o is None or halide is None:
        return None
    if carbon_nbr is not None:
        return None
    # R = the alkyl hanging off the ester O (exclude the carbonyl C side).
    r_root = next(
        (nb.GetIdx() for nb in mol.GetAtomWithIdx(ester_o).GetNeighbors()
         if nb.GetIdx() != carbonyl_c),
        None,
    )
    if r_root is None:
        return None
    # Collect R's atoms (everything reachable from r_root without crossing the
    # ester O), then name it as a substituent word.
    from collections import deque as _dq
    seen = {ester_o}
    r_atoms = []
    q = _dq([r_root])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        r_atoms.append(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() not in seen:
                q.append(nb.GetIdx())
    r_name = name_substituent_fragment(mol, r_atoms, r_root, [ester_o])
    if not r_name or ' ' in r_name:
        return None
    return f"{r_name} {carbono_word}"


def name_acid_halide(features, scope_out: Optional[dict] = None) -> Optional[str]:
    """
    Name an acid halide compound using functional class nomenclature.

    Produces two-word names: "{acyl name} {halide word}"

    Args:
        features: MolecularFeatures with principal_group set to an acid halide type.
        scope_out: Optional dict the caller owns. This function records
            ``scope_out['parent_scope'] = 'chain' | 'ring'`` for the branch it
            actually took, or leaves it absent when the parent is neither (or is
            not provably ``features.principal_chain``). Only the branch that
            picks the parent can know this --
            recovered downstream from ``features.principal_chain`` or
            ``features.chain_is_parent``, both of which report "chain" on
            ring-parented names. Consumed by ``handlers.acid_halide`` to tell
            ``_inject_stereo_if_missing`` which numbering a front-of-name
            stereodescriptor block is read in (P-91.3, BB:44639 "NAMING OF
            STEREOISOMERS", deciding sentence :44643).

    Returns:
        IUPAC name string, or None if not an acid halide.
    """
    def _scope(kind: str) -> None:
        if scope_out is not None:
            scope_out['parent_scope'] = kind

    mol = features.mol
    pg = features.principal_group

    if pg not in HALIDE_WORDS:
        return None

    halide_word = HALIDE_WORDS[pg]

    # Get acid halide matches
    acid_halide_matches = features.functional_groups.get(pg, [])
    if not acid_halide_matches:
        return None

    num_halide_groups = len(acid_halide_matches)

    # W3-P06 Task 1/2 (P-65.5.1): aggregate acyl-halide matches across ALL halide
    # FG types. name_acid_halide's ``pg`` is a SINGLE senior halide type, so a
    # MIXED diacyl halide (Br-CO-...-CO-Cl) or a mixed ring dicarbonyl would
    # otherwise see only one end and mis-name the other as an oxo+halo
    # substituent. ``combined`` = [(match, halide_symbol), ...] over every type.
    _combined = []
    for _ft, _sym in (("acid_chloride", "Cl"), ("acid_bromide", "Br"),
                      ("acid_fluoride", "F"), ("acid_iodide", "I")):
        for _m in features.functional_groups.get(_ft, []):
            _combined.append((_m, _sym))

    # P-35.4.2 / P-65.2.1 (BB 18114, W2E-P1FC Task 7): the acyl halide of a
    # MONO-ester of carbonic acid, X-C(=O)-O-R, is the functional-class name
    # '<R> carbono<halide>idate' (benzyl carbonochloridate, ethyl
    # carbonochloridate). The carbonyl C bears exactly: one =O, one halide, and
    # one ester-O to a nameable alkyl R; NO carbon neighbour (so it is NOT a
    # plain R-CO-X acyl halide). Fail-closed on any other decoration.
    if num_halide_groups == 1:
        _cc = name_carbonic_monoester_acyl_halide(mol, acid_halide_matches[0],
                                                  halide_word)
        if _cc is not None:
            return _cc

    # W3-P06 Task 1 (P-65.5.1): TWO OR MORE acyl halides on the SAME benzene ring
    # -> 'benzene-{locants}-{mult}carbonyl {mult}{halide}' (benzene-1,2-dicarbonyl
    # dichloride). Uses the combined cross-type list; same-halide only (mixed ring
    # halides fail closed -> fall through). Checked before the single-ring path,
    # which returns only 'benzoyl {halide}' (dropping the 2nd acyl).
    _bpc = _name_benzene_polycarbonyl_halide(mol, _combined)
    if _bpc:
        _scope('ring')
        return _bpc

    # Collect all atoms consumed by acid halide groups (C=O, halogen)
    consumed_atoms = set()
    for match in acid_halide_matches:
        consumed_atoms.update(match)

    # Check for ring-attached acid halide (e.g., benzoyl chloride, cyclohexanecarbonyl chloride)
    is_ring_attached = False
    ring_name = None
    for match in acid_halide_matches:
        # match = (C, O, X) from SMARTS [CX3](=O)[X]
        carbonyl_c = match[0]
        atom = mol.GetAtomWithIdx(carbonyl_c)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in consumed_atoms and neighbor.IsInRing():
                is_ring_attached = True
                # Check if it's benzene
                ring_name = _detect_ring_parent(mol, neighbor.GetIdx())
                break

    if is_ring_attached:
        _scope('ring')
        return _name_ring_attached_acid_halide(mol, features, acid_halide_matches,
                                                halide_word, ring_name, consumed_atoms)

    # Acyclic acid halide: determine chain length
    # Find the principal chain (the carbon chain ending at the carbonyl C)
    chain = features.principal_chain
    # Wave2 T6c: a PSEUDOhalide (azide/cyanide/isocyanate) contributes its own
    # skeletal atoms — the cyanide CARBON sits inside features.principal_chain
    # and inflated the acyl count (CCC(=O)C#N mis-sized to 'butanoyl
    # cyanide'). The acyl chain = chain atoms not consumed by the halide
    # group, plus the carbonyl carbon itself (which IS consumed).
    _carbonyl_cs = {m[0] for m in acid_halide_matches}
    chain_length = (
        sum(1 for a in chain if a not in consumed_atoms or a in _carbonyl_cs)
        if chain else 0
    )

    # The acyl parent is `chain` -- and therefore `features.atom_to_locant`, which
    # namer.py:4732 builds from `features.principal_chain` -- only on this branch,
    # where chain_length was counted FROM `chain`. The _count_acyl_chain fallback
    # below sizes the acyl group without `chain`, so the chain map is then not
    # provably the acyl numbering and the scope stays undeclared (fail closed).
    _chain_sized_from_principal_chain = bool(chain) and chain_length > 0

    if chain_length == 0:
        # Fallback: count carbons connected to carbonyl
        chain_length = _count_acyl_chain(mol, acid_halide_matches[0][0], consumed_atoms)

    # For diacid halides (acyl halide at BOTH chain ends) — same OR mixed halides,
    # via the combined cross-type list (W3-P06 Task 2). Handles the pre-existing
    # same-halide case (pentanedioyl dichloride) AND mixed (butanedioyl bromide
    # chloride, halide words alphabetical). A single-carbon 2-halide (Cl-CO-Cl,
    # carbonyl dichloride) is EXCLUDED by the distinct-chain-ends guard inside.
    _diacyl = _name_diacyl_halide_combined(mol, _combined, chain, chain_length)
    if _diacyl:
        if _chain_sized_from_principal_chain:
            _scope('chain')
        return _diacyl

    # Single acid halide: build acyl name.
    #
    # `_build_acyl_name`'s `unsaturation` parameter (and the
    # `get_enoate_name` grammar behind it) existed but had NO caller, so every
    # acyl chain was spelled saturated and the C=C was SILENTLY DROPPED --
    # `C[C@@H]1C[C@H]1/C=C/C(Cl)=O` was named `3-[...]propanoyl chloride`, which
    # OPSIN reads back as the saturated `CCC(=O)Cl` skeleton. Locants run from the
    # carbonyl carbon = 1 (P-65.5.1). Geometry is left EMPTY here on purpose: the
    # E/Z block is a stereodescriptor and belongs to the stereo layer, which
    # cites it at the front of the complete name (P-91.3, BB:44639 "NAMING OF
    # STEREOISOMERS", :44643); emitting it here too would double-cite it.
    _acyl_atoms = [a for a in chain
                   if a not in consumed_atoms or a in _carbonyl_cs] if chain else []
    if len(_acyl_atoms) > 1 and _acyl_atoms[0] not in _carbonyl_cs:
        if _acyl_atoms[-1] in _carbonyl_cs:
            _acyl_atoms = list(reversed(_acyl_atoms))
        else:
            _acyl_atoms = []          # carbonyl not at either end -> fail closed
    _unsaturation = []
    if _acyl_atoms and _acyl_atoms[0] in _carbonyl_cs:
        for _i in range(len(_acyl_atoms) - 1):
            _b = mol.GetBondBetweenAtoms(int(_acyl_atoms[_i]), int(_acyl_atoms[_i + 1]))
            if _b is not None and _b.GetBondType() == Chem.BondType.DOUBLE:
                _unsaturation.append((_i + 1, ''))
    acyl_name = _build_acyl_name(chain_length, _unsaturation or None)

    # Discover substituents on the acyl chain via universal pipeline (Phase 86).
    # Parent atoms = chain; exclude = all atoms consumed by acid halide groups
    # (carbonyl C, carbonyl O, halogen). This replaces the old
    # _get_chain_substituent_prefix() which only handled halogen substituents.
    from ..assembly.composer import _integrate_universal_prefixes
    sub_prefix = _integrate_universal_prefixes(
        mol, set(chain) if chain else set(),
        parent_type="chain",
        principal_chain=chain,
        exclude_atoms=consumed_atoms,
    )

    if _chain_sized_from_principal_chain:
        _scope('chain')
    if sub_prefix:
        return f"{sub_prefix}{acyl_name} {halide_word}"
    return f"{acyl_name} {halide_word}"


def _detect_ring_parent(mol, ring_atom_idx: int) -> Optional[str]:
    """Detect the ring parent name for a ring-attached acid halide."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if ring_atom_idx in ring:
            ring_set = set(ring)
            # Check if all ring atoms are aromatic carbons (benzene)
            all_aromatic_c = all(
                mol.GetAtomWithIdx(idx).GetIsAromatic() and
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                for idx in ring
            )
            if all_aromatic_c and len(ring) == 6:
                return "benzo"
            # Cycloalkane
            all_carbon = all(
                mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring
            )
            if all_carbon:
                prefix = get_chain_prefix(len(ring))
                return f"cyclo{prefix}"
    return None


def _is_benzene_ring(mol, ring) -> bool:
    """True iff ``ring`` is a benzene ring (6 aromatic carbons)."""
    return len(ring) == 6 and all(
        mol.GetAtomWithIdx(i).GetIsAromatic() and
        mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        for i in ring
    )


def _ring_cycle(mol, ring_set) -> List[int]:
    """Return the ring atoms in connected cyclic order."""
    start = next(iter(ring_set))
    order = [start]
    prev, cur = None, start
    while True:
        nxt = next((n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if n.GetIdx() in ring_set and n.GetIdx() != prev
                    and n.GetIdx() not in order), None)
        if nxt is None:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _lowest_locants(cycle: List[int], attach: List[int]) -> List[int]:
    """Lowest sorted locant set for ``attach`` atoms over all rotations/directions
    of the ring ``cycle`` (first-point-of-difference)."""
    n = len(cycle)
    attach_set = set(attach)
    best = None
    for direction in (cycle, cycle[::-1]):
        for i in range(n):
            rot = direction[i:] + direction[:i]
            locs = sorted(rot.index(a) + 1 for a in attach_set)
            if best is None or locs < best:
                best = locs
    return best


def _name_benzene_polycarbonyl_halide(mol, combined) -> Optional[str]:
    """P-65.5.1: >=2 acyl halides on ONE benzene ring ->
    'benzene-{locants}-{mult}carbonyl {mult}{halide}' (benzene-1,2-dicarbonyl
    dichloride). Same-halide only — mixed ring halides fail closed (None)."""
    if len(combined) < 2:
        return None
    ring_info = mol.GetRingInfo()
    ring_carbonyls = defaultdict(list)   # frozenset(ring) -> [(attach_atom, sym)]
    for match, sym in combined:
        carbonyl_c = match[0]
        atom = mol.GetAtomWithIdx(carbonyl_c)
        attach = next((n.GetIdx() for n in atom.GetNeighbors()
                       if n.IsInRing() and n.GetIdx() not in match), None)
        if attach is None:
            continue
        for ring in ring_info.AtomRings():
            if attach in ring and _is_benzene_ring(mol, ring):
                ring_carbonyls[frozenset(ring)].append((attach, sym))
                break
    for ring_fs, items in ring_carbonyls.items():
        if len(items) < 2:
            continue
        if len({s for _, s in items}) != 1:
            return None                  # mixed ring halides -> fail closed
        halide_word = _HALIDE_WORD[items[0][1]]
        locs = _lowest_locants(_ring_cycle(mol, set(ring_fs)),
                               [a for a, _ in items])
        k = len(items)
        loc_str = ",".join(str(l) for l in locs)
        mult_c = get_multiplier_prefix(k, "carbonyl")
        mult_h = get_multiplier_prefix(k, halide_word)
        return f"benzene-{loc_str}-{mult_c}carbonyl {mult_h}{halide_word}"
    return None


def _name_ring_attached_acid_halide(mol, features, matches, halide_word,
                                     ring_name, consumed_atoms) -> str:
    """Name a ring-attached acid halide (benzoyl chloride, cyclohexanecarbonyl chloride)."""
    if ring_name == "benzo":
        return f"benzoyl {halide_word}"

    if ring_name and ring_name.startswith("cyclo"):
        # cyclohexanecarbonyl chloride format
        ring_prefix = ring_name  # e.g., "cyclohex"
        return f"{ring_prefix}anecarbonyl {halide_word}"

    # Fallback for unknown ring types
    return f"carbonyl {halide_word}"


def _build_acyl_name(chain_length: int, unsaturation=None) -> str:
    """Build the acyl name from chain length.

    Args:
        chain_length: Number of carbons including the carbonyl carbon.
        unsaturation: Optional list of ``(double_bond_locant, 'E'|'Z'|'')`` tuples
            (locants from the carbonyl carbon = 1). None → saturated form,
            byte-identical to the prior bare-int behavior.

    Returns:
        Acyl name string (e.g., 'acetyl', 'propanoyl', 'butanoyl',
        '(9Z)-octadec-9-enoyl').
    """
    if not unsaturation:
        # Check retained names first
        if chain_length in RETAINED_ACYL_NAMES:
            return RETAINED_ACYL_NAMES[chain_length]
        # Systematic: {chain_prefix}anoyl
        prefix = get_chain_prefix(chain_length)
        return f"{prefix}anoyl"

    # Unsaturated: derive the acyl (-oyl) form from the systematic -oate stem
    # (reuse the chain en-locant grammar; do not hand-roll it). '...oate' -> '...oyl'.
    from ..data.chain_names import get_enoate_name
    oate = get_enoate_name(chain_length, unsaturation)
    return oate[:-4] + "oyl" if oate.endswith("oate") else oate


def _name_diacid_halide(chain_length: int, halide_word: str, num_groups: int) -> str:
    """Name a diacid halide (e.g., pentanedioyl dichloride).

    Args:
        chain_length: Total chain length.
        halide_word: "chloride", "bromide", or "fluoride".
        num_groups: Number of acid halide groups.

    Returns:
        IUPAC name string.
    """
    prefix = get_chain_prefix(chain_length)
    multiplier = get_multiplier_prefix(num_groups, halide_word)
    # pentane -> pentanedioyl (diacid form)
    return f"{prefix}anedioyl {multiplier}{halide_word}"


def _name_diacyl_halide_combined(mol, combined, chain, chain_length) -> Optional[str]:
    """P-65.5.1: diacyl halide with an acyl halide at BOTH distinct chain ends
    (same or mixed halides) -> '{chain}dioyl <halide word(s)>'.

    Halide words are cited in ALPHABETICAL order (bromide < chloride < fluoride <
    iodide); identical halides collapse with a numeric multiplier
    ('pentanedioyl dichloride'); different halides -> two words
    ('butanedioyl bromide chloride'). Returns None (fall through to single-acyl
    naming) unless there are EXACTLY two carbonyl carbons and they are the two
    DISTINCT ends of the chain — so a single-carbon 2-halide (carbonyl dichloride,
    Cl-CO-Cl) is never captured (its two Cl share one carbon = one chain 'end')."""
    if len(combined) != 2 or not chain or len(chain) < 2:
        return None
    if chain[0] == chain[-1]:
        return None
    carbonyls = {m[0] for m, _ in combined}
    if carbonyls != {chain[0], chain[-1]}:
        return None
    words = sorted(_HALIDE_WORD[sym] for _, sym in combined)
    prefix = get_chain_prefix(chain_length)
    if words[0] == words[1]:
        halide_part = f"{get_multiplier_prefix(2, words[0])}{words[0]}"
    else:
        halide_part = f"{words[0]} {words[1]}"
    return f"{prefix}anedioyl {halide_part}"


def _is_diacid_halide(mol, matches, chain) -> bool:
    """Check if acid halide groups are at both ends of the chain (diacid halide).

    Args:
        mol: RDKit Mol object.
        matches: List of acid halide SMARTS match tuples.
        chain: Principal chain atom indices.

    Returns:
        True if diacid halide pattern detected.
    """
    if len(matches) < 2:
        return False

    if not chain or len(chain) < 2:
        return False

    # Get carbonyl carbons from matches
    carbonyl_carbons = {match[0] for match in matches}
    chain_ends = {chain[0], chain[-1]}

    # Diacid if carbonyl carbons are at both chain ends
    return len(carbonyl_carbons & chain_ends) >= 2


def _count_acyl_chain(mol, carbonyl_c: int, consumed: set) -> int:
    """Count carbons in the acyl chain by BFS from carbonyl carbon.

    Args:
        mol: RDKit Mol object.
        carbonyl_c: Atom index of carbonyl carbon.
        consumed: Set of atom indices consumed by acid halide group.

    Returns:
        Chain length (number of carbons including carbonyl).
    """
    visited = {carbonyl_c}
    queue = deque([carbonyl_c])
    carbon_count = 1

    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx not in visited and neighbor.GetSymbol() == 'C' and not neighbor.IsInRing():
                # Skip the carbonyl oxygen and halogen (already in consumed)
                if nidx in consumed and nidx != carbonyl_c:
                    continue
                visited.add(nidx)
                queue.append(nidx)
                carbon_count += 1

    return carbon_count


def _get_chain_substituent_prefix(mol, features, chain, consumed_atoms) -> str:
    """Build substituent prefix for non-acid-halide groups on the acyl chain.

    For example, 3-chlorobutanoyl chloride has a substituent Cl at position 3
    that is NOT part of the acid halide functional group.

    Args:
        mol: RDKit Mol object.
        features: MolecularFeatures.
        chain: Principal chain atom indices.
        consumed_atoms: Atom indices consumed by acid halide groups.

    Returns:
        Prefix string (e.g., "3-chloro") or empty string.
    """
    if not chain:
        return ""

    # Build atom-to-locant mapping for the chain
    atom_to_locant = {}
    for i, atom_idx in enumerate(chain):
        atom_to_locant[atom_idx] = i + 1

    # Find substituent halogens (and other groups) NOT consumed by acid halide
    substituent_parts = []
    chain_set = set(chain)

    for atom_idx in chain:
        atom = mol.GetAtomWithIdx(atom_idx)
        locant = atom_to_locant[atom_idx]

        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx in chain_set:
                continue
            if nidx in consumed_atoms:
                continue

            sym = neighbor.GetSymbol()
            if sym in HALOGEN_PREFIX:
                substituent_parts.append((locant, HALOGEN_PREFIX[sym]))
            # Could add more substituent types here as needed

    if not substituent_parts:
        return ""

    # Group by prefix name for multipliers
    prefix_groups = {}
    for locant, prefix_name in sorted(substituent_parts):
        if prefix_name not in prefix_groups:
            prefix_groups[prefix_name] = []
        prefix_groups[prefix_name].append(locant)

    # Build prefix string (alphabetical order)
    parts = []
    for prefix_name in sorted(prefix_groups.keys()):
        locants = prefix_groups[prefix_name]
        locant_str = ",".join(str(l) for l in sorted(locants))
        if len(locants) > 1:
            multiplier = get_multiplier_prefix(len(locants), prefix_name)
            parts.append(f"{locant_str}-{multiplier}{prefix_name}")
        else:
            parts.append(f"{locant_str}-{prefix_name}")

    return "-".join(parts)


def get_acid_halide_consumed_atoms(functional_groups: Dict) -> set:
    """Get the set of atom indices consumed by acid halide groups.

    These atoms should be filtered from halogen FG detection to prevent
    double-counting (e.g., Cl appearing as both "oyl chloride" and "chloro").

    Args:
        functional_groups: Dict from detect_functional_groups().

    Returns:
        Set of atom indices consumed by acid halide groups.
    """
    consumed = set()
    for fg_name in ("acid_chloride", "acid_bromide", "acid_fluoride"):
        for match in functional_groups.get(fg_name, []):
            # match = (C, O, X) from SMARTS [CX3](=O)[X]
            # The halogen X is at index 2 (third element)
            if len(match) >= 3:
                consumed.add(match[2])  # halogen atom
    return consumed
