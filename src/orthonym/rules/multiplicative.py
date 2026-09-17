"""Multiplicative nomenclature for symmetric bridge-linked molecules (IUPAC.

Detects molecules with two or more identical parent structures connected by a
polyvalent linking group (bridge) and produces multiplicative names like
"4,4'-methylenedianiline" or "4,4'-oxydibenzoic acid".

Three conditions for multiplicative naming (IUPAC:
    1. Two or more identical parent structures (verified by canonical SMILES)
    2. Connected by a polyvalent linking group (the bridge atom(s))
    3. Parent has a principal characteristic group (suffix-forming FG)

Uses the recursion depth guard from fragment_naming.py to prevent infinite
loops when naming fragment halves.
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import RWMol

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
from ..data.chain_names import get_chain_prefix
from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)
from ..perception.molcache import atoms_of, canon_smiles

# ---------------------------------------------------------------------------
# Saturation/modification prefixes that must not be preceded by "di"
# ---------------------------------------------------------------------------
# When a parent name starts with one of these prefixes, multiplicative naming
# would produce unparseable concatenation (e.g., "ditetrahydropyran").
# In such cases, fall through to substitutive naming instead.

SATURATION_PREFIXES = [
    'tetrahydro', 'dihydro', 'hexahydro', 'perhydro', 'octahydro',
    'decahydro', 'dodecahydro',
]


# ---------------------------------------------------------------------------
# Bridge definitions: atom pattern -> IUPAC bridge name
# ---------------------------------------------------------------------------

# Single-atom bridge names
# IUPAC: the divalent -S- multiplicative group is the preselected
# prefix "sulfanediyl"; the legacy "thio" is deprecated (Blue Book.
# The two-identical-ring monosulfide PIN is the *multiplicative* name, e.g.
# c1ccccc1Sc1ccccc1 -> "1,1'-sulfanediyldibenzene" (Blue Book, line 27826:
# "1,1'-sulfanediyldibenzene (PIN) (not 1,1'-thiodibenzene)"), parallel to the
# "1,1'-oxydibenzene" PIN (line 23921); the -O-O- / -S-S- analogues are likewise
# multiplicative PINs (see the two-atom bridge table below for their cites).
_SINGLE_ATOM_BRIDGES: Dict[str, str] = {
    "O": "oxy",
    "S": "sulfanediyl",
    "NH": "azanediyl",
    "CH2": "methylene",
}

# Two-atom bridge patterns
_TWO_ATOM_BRIDGES = [
    # (element1, element2, h_count1, h_count2, bridge_name)
    ("C", "C", 2, 2, "ethane-1,2-diyl"),  # CH2-CH2 PIN; 'ethylene' is general-nomenclature only)
    # -FINAL I9: `vinylene` is a verbatim NON-PIN. The Blue Book's own
    # prefix table marks it so: `| ethene-1,2-diyl* (not<br>vinylene) | -CH=CH- |
    # |` (`the Blue Book`), and the stilbene entry gives the PIN
    # outright: `stilbene (ring substitution only) 1,1'-(ethene-1,2-diyl)dibenzene
    # (PIN)` (`:16615`). The bridge carries locants, so the generic enclosure leg
    # in `_assemble_multiplicative_name` parenthesises it, reproducing that PIN
    # exactly.
    ("C", "C", 1, 1, "ethene-1,2-diyl"),   # CH=CH
    # a phase.B audit-driven adds (internal notes-B.md ranks 1-2):
    ("O", "O", 0, 0, "peroxy"),       # O-O; IUPAC; OPSIN multiRadicalSubstituents.xml line 53; internal notes-B.md #1
    ("S", "S", 0, 0, "disulfanediyl"),# S-S; IUPAC; OPSIN multiRadicalSubstituents.xml; internal notes-B.md #2
]

# Multi-atom bridge names: (element, H_count, ring_neighbor_count) -> bridge_name
# These handle star-topology bridges where 3+ identical parents radiate from
# a single central atom (IUPAC.
_MULTI_BRIDGE_NAMES: Dict[Tuple[str, int, int], str] = {
    ("N", 0, 3): "nitrilo",           # N connecting 3 rings (trivalent)
    ("C", 1, 3): "methylidyne",       # CH connecting 3 rings (trivalent)
    ("C", 0, 4): "methanetetrayl",    # C connecting 4 rings (tetravalent)
    # a phase.B audit-driven add (internal notes-B.md rank 3):
    ("P", 0, 3): "phosphinidyne",     # P connecting 3 rings (trivalent); IUPAC; OPSIN multiRadicalSubstituents.xml line 47; internal notes-B.md #3
}


# a phase.B: _RETAINED_PARENT_NAMES (4-entry hardcoded dict) DELETED.
# The registry-query layer below (_resolve_parent_name) replaces it. The
# global ALL_RETAINED_NAMES registry in data/retained_names.py is the
# single source of truth (a phase reuse pattern). See
# internal notes-B.md for the registry-coverage verification that confirms
# all 4 previously-hardcoded SMILES are present in the registry when
# accessed via canon_smiles(...).


def _resolve_parent_name(canon_smiles: str) -> Optional[Tuple[str, int]]:
    """a phase.B: registry-query layer for retained parent + locant.

    Replaces the hardcoded _RETAINED_PARENT_NAMES dict (deleted) with a
    query against the global ALL_RETAINED_NAMES registry via
    get_retained_name. Falls back to name_fragment_recursively wrapped
    with _extract_pg_locant_from_fragment for the principal-group locant.

    No SMILES strings hardcoded inline -- single source of truth is the
    global registry per a phase reuse pattern.

    Args:
        canon_smiles: Canonical SMILES of the parent fragment.

    Returns:
        (parent_name, fg_locant) where fg_locant is the IUPAC position of the
        principal group on the fragment, or None if no parent name resolvable.

    Source: 154-internal notes; data/retained_names.py:get_retained_name:465.
    """
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..data.retained_names import get_retained_name

    # 1. Global registry first (single source of truth).
    retained = get_retained_name(canon_smiles)
    if retained is not None:
        locant = _extract_pg_locant_from_fragment(canon_smiles)
        return (retained, locant if locant is not None else 1)

    # 2. Algorithmic fallback.
    name = name_fragment_recursively(canon_smiles)
    if name is None:
        return None
    locant = _extract_pg_locant_from_fragment(canon_smiles)
    return (name, locant if locant is not None else 1)


def _extract_pg_locant_from_fragment(canon_smiles: str) -> Optional[int]:
    """Find the principal-group atom in the fragment and return its IUPAC locant.

    For benzene-derived parents (aniline, phenol, benzoic acid), the principal
    group is at locant 1. For other ring parents, the scope returns 1 as
    the safe default (the caller's `_get_bridge_locant` cascade computes
    the bridge attachment locant relative to that anchor).

    Args:
        canon_smiles: Canonical SMILES of the parent fragment.

    Returns:
        1-indexed IUPAC locant of the principal group, or None if extraction
        fails.

    Source: 154-internal notes; internal notes-B.md
    """
    mol = Chem.MolFromSmiles(canon_smiles)
    if mol is None:
        return None
    # For all 4 hardcoded benzene-derived parents (aniline / phenol / benzoic
    # acid x2), the principal group sits at locant 1 by IUPAC convention.
    # This is the safe default for the multiplicative scope (which only
    # consumes benzene-derived retained parents per internal notes-B.md).
    return 1


def _is_pure_single_bond_assembly(mol) -> bool:
    """a phase.B: detect single-bond-joined identical rings (ring_assemblies territory).

    True iff every inter-ring-system connection in the molecule is a single
    bond directly between two ring atoms with NO bridge atom. In that
    case the multiplicative path returns None and the cascade falls
    through to detect_ring_assembly (a phase's path).

    The contract is symmetric: this guard is the multiplicative-side
    enforcer; ring_assemblies._find_inter_system_bonds (lines 77-124) is
    the ring-assembly-side enforcer (only counts ring-to-ring single
    bonds; bridge atoms are not in rings, so atom-bridged cases never
    produce inter-system bonds there).

    Cross-handler regression test:
    tests/integration/test_assembly_vs_multiplicative_dispatch.py.

    Source: 154-internal notes; ring_assemblies.py:_find_inter_system_bonds:77.
    Source: internal notes
    """
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return False

    # Build ring_atoms set.
    atom_rings = ring_info.AtomRings()
    ring_atoms = set()
    for r in atom_rings:
        ring_atoms.update(r)

    # Use perception.rings.get_ring_systems to cluster fused rings.
    from ..perception.rings import get_ring_systems
    ring_systems = get_ring_systems(mol)
    # ring_systems is a list of sets of atom indices, one per fused ring system.
    if len(ring_systems) < 2:
        # Only one ring system (e.g., naphthalene) -- multiplicative does not apply.
        return False

    atom_to_system: Dict[int, int] = {}
    for sys_idx, atoms in enumerate(ring_systems):
        for a in atoms:
            atom_to_system[a] = sys_idx

    # Look for inter-ring-system bonds (single bond, both atoms in distinct systems).
    has_single_bond_inter_system = False
    for bond in bonds_of(mol):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in ring_atoms and a2 in ring_atoms:
            sys1 = atom_to_system.get(a1)
            sys2 = atom_to_system.get(a2)
            if sys1 is not None and sys2 is not None and sys1 != sys2:
                if bond.GetBondTypeAsDouble() == 1.0:
                    has_single_bond_inter_system = True

    # Detect bridge atoms (non-ring atoms touching >= 2 distinct ring systems).
    has_bridge_atom = False
    for atom in atoms_of(mol):
        if atom.GetIdx() in ring_atoms:
            continue
        touched_systems = set()
        for nbr in atom.GetNeighbors():
            nbr_sys = atom_to_system.get(nbr.GetIdx())
            if nbr_sys is not None:
                touched_systems.add(nbr_sys)
        if len(touched_systems) >= 2:
            has_bridge_atom = True
            break

    # Pure single-bond assembly = inter-system single bond present AND no bridge atom.
    return has_single_bond_inter_system and not has_bridge_atom


# ---------------------------------------------------------------------------
# Central-arene + acyclic identical arms (a)+(d) /
# ---------------------------------------------------------------------------
# A benzene ring bearing >=2 identical acyclic carboxylic-acid arms is a
# multiplicative case where the ARM is the parent functional compound (acetic
# acid, (d)) and the benzene is the central di-/poly-valent group:
# OC(=O)Cc1cc(CC(=O)O)cc(CC(=O)O)c1
# -> 2,2',2''-(benzene-1,3,5-triyl)triacetic acid (PIN, the Blue Book)
# OC(=O)Cc1ccc(CC(=O)O)cc1
# -> 2,2'-(1,4-phenylene)diacetic acid (PIN, cf. the Blue Book)
# Without this path, parent selection takes ONE acid arm and the symmetric
# others are DROPPED (a constitutional undercount, defect V-8).
#
# Scope (fail closed otherwise -- never a wrong name):
# * exactly one ring system, a bare benzene (6 aromatic carbons);
# * every exocyclic ring substituent is an UNBRANCHED aliphatic carboxylic
# acid -(CH2)k-COOH with k>=1 (the >=1 CH2 spacer is what distinguishes the
# multiplicative acetic-acid arm from a ring-attached -COOH / -CONH2, which
# stays the carbo* added-carbon suffix: benzene-1,3,5-tricarboxamide,
# benzene-1,2-dicarboxylic acid);
# * all arms identical; >=2 of them; no other substituents anywhere.

# yl-multiplier for a poly-valent benzene central group (n>=3; n==2 is phenylene)
_BENZENE_YL_MULTIPLIER = {3: "triyl", 4: "tetrayl", 5: "pentayl", 6: "hexayl"}


def _lowest_locant_set(ring_cyclic: List[int], attach_set: set) -> List[int]:
    """Lowest locant set for the attachment ring atoms, over the 12 dihedral
    numberings of a 6-ring. Order-independent (deterministic) result."""
    n = len(ring_cyclic)
    best: Optional[List[int]] = None
    for start in range(n):
        for direction in (1, -1):
            locant_of = {
                ring_cyclic[(start + direction * i) % n]: i + 1 for i in range(n)
            }
            locs = sorted(locant_of[a] for a in attach_set)
            if best is None or locs < best:
                best = locs
    return best or []


def _central_arene_substituent_name(
    ring_cyclic: List[int], attach_set: set
) -> Optional[str]:
    """Name the central benzene group: divalent -> "1,x-phenylene"
    preferred prefix), tri-/poly-valent -> "benzene-<locants>-triyl" etc."""
    locs = _lowest_locant_set(ring_cyclic, attach_set)
    if not locs:
        return None
    loc_str = ",".join(str(loc) for loc in locs)
    n = len(attach_set)
    if n == 2:
        return f"{loc_str}-phenylene"
    yl = _BENZENE_YL_MULTIPLIER.get(n)
    if yl is None:
        return None
    return f"benzene-{loc_str}-{yl}"


def _try_central_arene_acyclic_arms(mol) -> Optional[str]:
    """Multiplicative naming for a bare benzene bearing >=2 identical acyclic
    functional-parent arms (defect V-8; Wave2 extends the acid-only scope
    to alcohol arms). Returns the PIN or ``None`` (fail closed).

    2,2',2''-(benzene-1,3,5-triyl)triacetic acid (d));
    (benzene-1,3,5-triyl)trimethanol / (benzene-1,4-diyl)dimethanol
    ; mononuclear methanol units carry no locants, cf. the BB
    "[oxydi(pyridazine-4,3,5-triyl)]tetramethanol" witness).
    """
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 1:
        return None
    from ..perception.rings import get_ring_systems

    systems = get_ring_systems(mol)
    if len(systems) != 1 or len(systems[0]) != 6:
        return None
    ring_atoms = set(systems[0])
    for a in ring_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != "C" or not atom.GetIsAromatic():
            return None
    ring_cyclic = list(ring_info.AtomRings()[0])
    if set(ring_cyclic) != ring_atoms:
        return None

    # exocyclic attachments: ring atom -> its single exocyclic heavy neighbour
    attachments = {}
    for ra in ring_atoms:
        exo = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(ra).GetNeighbors()
            if n.GetIdx() not in ring_atoms and n.GetAtomicNum() > 1
        ]
        if not exo:
            continue
        if len(exo) > 1:
            return None
        attachments[ra] = exo[0]
    if len(attachments) < 2:
        return None

    parents = []
    all_arm_atoms = set()
    for ra, ac in attachments.items():
        if mol.GetAtomWithIdx(ac).GetSymbol() != "C":
            return None  # arm must attach to the ring via a carbon
        # Wave2 T5a: route through the shared strict linear-arm namer
        # (acid + alcohol arms, saturation/charge/heteroatom-coverage
        # hardened); the ring attachment atom acts as the bridge boundary.
        res = _acyclic_arm_parent_and_locant(mol, ac, ra)
        if res is None:
            return None
        parent_name, attach_locant = res
        parents.append((parent_name, attach_locant))
        # collect the arm atoms for the whole-molecule coverage check
        stack = [ac]
        while stack:
            a = stack.pop()
            if a in all_arm_atoms or a in ring_atoms:
                continue
            all_arm_atoms.add(a)
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nbr.GetIdx()
                if nbr.GetAtomicNum() > 1 and ni not in all_arm_atoms \
                        and ni not in ring_atoms:
                    stack.append(ni)

    if len(set(parents)) != 1:
        return None  # arms not identical -> multiplicative requires identical units
    parent_name, attach_locant = parents[0]

    # the whole molecule must be exactly ring + arms (no stray heavy atoms)
    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    if (ring_atoms | all_arm_atoms) != heavy:
        return None

    n = len(attachments)
    multiplier = SIMPLE_MULTIPLIERS.get(n)
    if multiplier is None:
        return None
    central = _central_arene_substituent_name(ring_cyclic, set(attachments.keys()))
    if central is None:
        return None

    # Mononuclear units (methanol) omit the meaningless '1' locants
    #; locant-bearing unit names take parentheses.
    if attach_locant is None:
        locant_prefix = ""
    else:
        locant_prefix = ",".join(
            f"{attach_locant}{chr(39) * i}" for i in range(n)
        ) + "-"
    if any(ch.isdigit() for ch in parent_name) or parent_name.startswith("dec"):
        unit_token = f"({parent_name})"
    else:
        unit_token = parent_name
    return f"{locant_prefix}({central}){multiplier}{unit_token}"


def _acyclic_arm_parent_and_locant(mol, attach_carbon: int, bridge_idx: int,
                                   allow_amine: bool = False):
    """Name a linear ACYCLIC arm (a suffix PCG at one terminus, the bridge
    attachment at the other) and return ``(parent_name, attach_locant)``.

    Scope (fail-closed -> None otherwise): an unbranched carbon chain whose two
    ends are exactly {PCG carbon, bridge-attach carbon}; the only functional
    atoms are the single PCG (carboxylic acid -OH/=O, or alcohol -OH; with
    ``allow_amine`` also a terminal -NH2). The PCG carbon is numbered 1
    (lowest locant), so the bridge attaches at C(chain length). Examples:
    HO-CH2-CH2-* -> ("ethan-1-ol", 2); HOOC-CH2-* -> ("acetic acid", 2);
    H2N-CH2-CH2-* -> ("ethan-1-amine", 2) [allow_amine only]. Tight by
    design (Risk R1: MULTIPLICATIVE@900 fires before GENERAL, so this
    must never claim an ordinary ester/ether).

    ``allow_amine`` is opt-in for the CHALCOGEN bridges ONLY (BB
    verbatim: "2,2'-oxydi(ethan-1-amine)" (PIN)); the N bridges must never
    pass it — with amine arms the substitutive polyamine parent is the PIN
     max-suffix rule; BB: diethylenetriamine names as a
    diamine, "not 2,2'-azanediyldi(ethan-1-amine)").
    """
    if mol.GetAtomWithIdx(attach_carbon).GetSymbol() != 'C':
        return None
    # Collect arm atoms (BFS from the attach carbon, not crossing the bridge).
    arm = set()
    stack = [attach_carbon]
    seen = {bridge_idx}
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        atom_a = mol.GetAtomWithIdx(a)
        if atom_a.IsInRing():
            return None  # arms must be acyclic
        arm.add(a)
        for nbr in atom_a.GetNeighbors():
            if nbr.GetIdx() not in seen:
                stack.append(nbr.GetIdx())

    arm_carbons = [a for a in arm if mol.GetAtomWithIdx(a).GetSymbol() == 'C']
    if not arm_carbons:
        return None
    # Wave2 hardening: the '{stem}anoic acid'/'{stem}an-1-ol' arm name
    # asserts a SATURATED, uncharged chain — any C=C/C#C or formal charge in
    # the arm would be silently dropped by the name. Fail closed instead.
    for a in arm:
        if mol.GetAtomWithIdx(a).GetFormalCharge() != 0:
            return None
    carbon_set = set(arm_carbons)
    for c in arm_carbons:
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if nbr.GetIdx() not in carbon_set:
                continue
            bond = mol.GetBondBetweenAtoms(c, nbr.GetIdx())
            if bond.GetBondTypeAsDouble() != 1.0:
                return None  # unsaturated arm -> out of scope
    deg = {
        c: sum(1 for nbr in mol.GetAtomWithIdx(c).GetNeighbors()
               if nbr.GetIdx() in carbon_set)
        for c in arm_carbons
    }
    if any(d > 2 for d in deg.values()):
        return None  # branched -> out of scope

    # Identify the single PCG: carboxylic acid (=O + -OH on one C) or alcohol.
    pcg_carbon = None
    pcg = None
    pcg_oxygens = set()
    for c in arm_carbons:
        has_carbonyl_o = None
        has_hydroxyl_o = None
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if nbr.GetSymbol() != 'O' or nbr.GetIdx() not in arm:
                continue
            bond = mol.GetBondBetweenAtoms(c, nbr.GetIdx())
            if bond.GetBondTypeAsDouble() == 2.0:
                has_carbonyl_o = nbr.GetIdx()
            elif nbr.GetTotalNumHs() >= 1 and nbr.GetDegree() == 1:
                has_hydroxyl_o = nbr.GetIdx()
        if has_carbonyl_o is not None and has_hydroxyl_o is not None:
            pcg_carbon, pcg = c, 'acid'
            pcg_oxygens = {has_carbonyl_o, has_hydroxyl_o}
            break
    if pcg_carbon is None:
        for c in arm_carbons:
            if any(mol.GetBondBetweenAtoms(c, nbr.GetIdx()).GetBondTypeAsDouble() == 2.0
                   for nbr in mol.GetAtomWithIdx(c).GetNeighbors()
                   if nbr.GetSymbol() == 'O' and nbr.GetIdx() in arm):
                continue  # carbonyl present -> not a plain alcohol carbon
            for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
                if (nbr.GetSymbol() == 'O' and nbr.GetIdx() in arm
                        and nbr.GetTotalNumHs() >= 1 and nbr.GetDegree() == 1):
                    pcg_carbon, pcg = c, 'ol'
                    pcg_oxygens = {nbr.GetIdx()}
                    break
            if pcg_carbon is not None:
                break
    if pcg_carbon is None and allow_amine:
        # Terminal primary amine -NH2 (BB "2,2'-oxydi(ethan-1-amine)" (PIN)).
        for c in arm_carbons:
            for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
                if (nbr.GetSymbol() == 'N' and nbr.GetIdx() in arm
                        and nbr.GetTotalNumHs() == 2 and nbr.GetDegree() == 1
                        and mol.GetBondBetweenAtoms(
                            c, nbr.GetIdx()).GetBondTypeAsDouble() == 1.0):
                    pcg_carbon, pcg = c, 'amine'
                    pcg_oxygens = {nbr.GetIdx()}
                    break
            if pcg_carbon is not None:
                break
    if pcg_carbon is None:
        return None

    # The arm may carry ONLY its chain carbons + the PCG heteroatoms (no stray
    # heteroatoms / extra FGs that the name would silently drop).
    arm_heteroatoms = {a for a in arm if mol.GetAtomWithIdx(a).GetSymbol() not in ('C', 'H')}
    if arm_heteroatoms != pcg_oxygens:
        return None

    n = len(arm_carbons)
    ends = {c for c, d in deg.items() if d == 1}
    if n == 1:
        if pcg_carbon != attach_carbon:
            return None
        if pcg == 'acid':
            # A 1-carbon acid arm is the COOH carbon itself attached to the
            # bridge — carboxy-on-parent territory (benzoic/carbonic class),
            # never a 'methanoic acid' multiplicative unit. Fail closed.
            return None
        # Mononuclear unit: locant omitted; BB
        # '[oxydi(pyridazine-4,3,5-triyl)]tetramethanol' carries no unit
        # locants). attach_locant=None signals the elision to the assembler.
        return ("methanol" if pcg == 'ol' else "methanamine"), None
    else:
        # PCG at one terminus (C1), bridge attachment at the other (C n).
        if ends != {pcg_carbon, attach_carbon}:
            return None
        attach_locant = n

    stem = get_chain_prefix(n)
    if pcg == 'acid':
        parent_name = "acetic acid" if n == 2 else f"{stem}anoic acid"
    elif pcg == 'ol':
        parent_name = f"{stem}an-1-ol"
    else:  # 'amine'
        parent_name = f"{stem}an-1-amine"
    return parent_name, attach_locant


def _try_acyclic_heteroatom_bridge(mol) -> Optional[str]:
    """Multiplicative naming for two identical ACYCLIC functional parents joined
    by a divalent heteroatom bridge: O -> oxy, S -> sulfanediyl.

    OCCSCCO -> 2,2'-sulfanediyldi(ethan-1-ol);
    OC(=O)COCC(=O)O -> 2,2'-oxydiacetic acid;
    OC(=O)CNCC(=O)O -> 2,2'-azanediyldiacetic acid (Wave2 T5a,
    azanediyl; BB sibling "3,3'-azanediyldipropanenitrile" (PIN));
    OCCNCCO -> 2,2'-azanediyldi(ethan-1-ol).

    Fail-closed (Risk R1): the bridge is a single non-ring O/S/NH with exactly
    two acyclic-carbon neighbours, the two arms are identical, and each arm is
    a linear chain with a single terminal PCG (handled by
    _acyclic_arm_parent_and_locant). Returns None for everything else (e.g.
    diethyl ether has no PCG; 2-ethoxyethanol's arms differ) so the GENERAL
    pipeline keeps ownership and no ester is ever stolen. The arm namer's
    acid/ol-only scope keeps the NH bridge away from polyamine parents where
    substitutive names are the PINs (BB: diethylenetriamine is a
    diamine parent, "not 2,2'-azanediyldi(ethan-1-amine)").
    """
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue
        bridge_type = _classify_single_atom_bridge(atom)
        # Divalent bridges that are multiplicative PINs: chalcogens (oxy /
        # sulfanediyl) + NH (azanediyl, Wave2 T5a).
        if bridge_type not in ('O', 'S', 'NH'):
            continue
        if atom.GetFormalCharge() != 0:
            continue
        bridge_name = _SINGLE_ATOM_BRIDGES.get(bridge_type)
        if bridge_name is None:
            continue
        heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs) != 2:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'C' for n in heavy_nbrs):
            continue

        nbr_indices = [n.GetIdx() for n in heavy_nbrs]
        fragments = _split_at_bridge(mol, idx, nbr_indices)
        if fragments is None:
            continue
        smi_a, smi_b, _conn_a, _conn_b = fragments
        if canon_smiles(smi_a) != canon_smiles(smi_b):
            continue

        # Amine arms only for the chalcogen bridges (BB "2,2'-oxydi(ethan-1-
        # amine)" (PIN)); NEVER for the NH bridge (polyamine-parent trap).
        allow_amine = bridge_type in ('O', 'S')
        arm = _acyclic_arm_parent_and_locant(
            mol, nbr_indices[0], idx, allow_amine=allow_amine
        )
        if arm is None:
            continue
        # the other arm must resolve identically (same parent + locant)
        arm_b = _acyclic_arm_parent_and_locant(
            mol, nbr_indices[1], idx, allow_amine=allow_amine
        )
        if arm_b != arm:
            continue
        parent_name, attach_locant = arm
        return _assemble_multiplicative_name(
            attach_locant, bridge_name, parent_name, unit_count=2
        )

    return None


def _try_acyclic_nitrilo_bridge(mol) -> Optional[str]:
    """Trivalent N joining THREE identical acyclic functional parents
    (Wave2 T5a; star topology, acyclic sibling of the ring-path
    nitrilo in _try_multi_atom_bridges):

    N(CH2COOH)3 -> 2,2',2''-nitrilotriacetic acid (BB;
    N(CCO)3 -> 2,2',2''-nitrilotri(ethan-1-ol) (BB, PIN).

    Fail-closed: bare uncharged non-ring N with exactly three acyclic-carbon
    neighbours; all three arms resolve to the SAME (parent, locant) via the
    strict linear-arm namer. Amine-arm cases never arrive here (the arm
    namer is acid/ol-only), so polyamine parents max-suffix rule)
    are never stolen.
    """
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue
        # P sibling (Wave2 completion, (d)):
        # P(CH2COOH)3 -> 2,2',2''-phosphanetriyltriacetic acid (BB verbatim).
        if atom.GetSymbol() not in ('N', 'P') or atom.GetTotalNumHs() != 0:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs) != 3:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'C' for n in heavy_nbrs):
            continue

        arms = [
            _acyclic_arm_parent_and_locant(mol, n.GetIdx(), idx)
            for n in heavy_nbrs
        ]
        if arms[0] is None or any(a != arms[0] for a in arms[1:]):
            continue
        parent_name, attach_locant = arms[0]
        bridge = "nitrilo" if atom.GetSymbol() == 'N' else "phosphanetriyl"
        return _assemble_multiplicative_name(
            attach_locant, bridge, parent_name, unit_count=3
        )

    return None


def _try_diamine_dinitrilo_bridge(mol) -> Optional[str]:
    """Two trivalent-N hubs joined by an unbranched alkanediyl chain, each hub
    bearing two identical acyclic functional-parent arms (EDTA family).

    The composite central group is ``{alkane}-1,m-diyldinitrilo``; the four
    identical arms are the multiplied parent:

        EDTA OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O ->
            2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetic acid
        PDTA OC(=O)CN(CC(=O)O)CCCN(CC(=O)O)CC(=O)O ->
            2,2',2'',2'''-(propane-1,3-diyldinitrilo)tetraacetic acid

    This is the four-fold multiplicative name, and is the PIN by DERIVATION:
     (the Blue Book) makes multiplicative nomenclature senior to
    substitutive for repeated senior parents; (the Blue Book)
    multiplies the MORE numerous parent (four acetic-acid units, not two
    glycine units), and its own worked example (the Blue Book) is EDTA.
    Note: the BB gives this name as its worked example but attaches NO explicit
    (PIN) tag in any of its four occurrences (:7669/:21586/:29809/:29970), and
    always pairs it with the glycine-based ``N,N'-(ethane-1,2-diyl)bis[N-
    (carboxymethyl)glycine]``. So the PIN status here is DERIVED from the two
    rules above, not read from a verbatim (PIN)-tagged string.

    Fail-closed (never a wrong name): exactly two bare uncharged non-ring N
    atoms, each with three acyclic-carbon neighbours; the two N are linked by a
    single unbranched all-CH2 chain (m>=1) attached at its two TERMINI; the two
    remaining arms on each N all resolve to the SAME (parent, locant) via the
    strict linear-arm namer (acid/ol only -- amine arms decline, the
    polyamine-parent trap of; and the whole molecule is exactly
    {2 N + chain + 4 arms}. Anything else -> None.
    """
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    # Exactly two trivalent-N hubs (bare, uncharged, non-ring, 3 carbon arms).
    hubs = []
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'N' or atom.GetTotalNumHs() != 0:
            continue
        if atom.GetIdx() in ring_atoms or atom.GetFormalCharge() != 0:
            continue
        heavy_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs) != 3:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'C'
               for n in heavy_nbrs):
            continue
        hubs.append(atom.GetIdx())
    if len(hubs) != 2:
        return None
    n1, n2 = hubs

    # The connecting chain is the shortest path between the two hubs; the arms
    # are dead-ends, so this path IS the alkanediyl bridge.
    path = list(Chem.GetShortestPath(mol, n1, n2))
    if len(path) < 3 or path[0] != n1 or path[-1] != n2:
        return None
    chain = path[1:-1]  # intermediate carbons only
    m = len(chain)
    if m < 1:
        return None

    # Every chain atom is a plain -CH2- link: carbon, non-ring, uncharged, and
    # carrying NO heavy substituent off the chain (only its path neighbours).
    path_set = set(path)
    for i, c in enumerate(chain):
        catom = mol.GetAtomWithIdx(c)
        if catom.GetSymbol() != 'C' or catom.IsInRing():
            return None
        if catom.GetFormalCharge() != 0 or catom.GetNumRadicalElectrons() != 0:
            return None
        heavy = [nb.GetIdx() for nb in catom.GetNeighbors()
                 if nb.GetAtomicNum() > 1]
        # Each chain carbon touches only its two path neighbours (the N hubs at
        # the ends, adjacent chain carbons in the middle) -- no branch.
        if any(h not in path_set for h in heavy):
            return None
        if len(heavy) != 2:
            return None

    # The two arms on each hub (its neighbours not on the connecting path).
    arm_carbons = []
    for hub in (n1, n2):
        exo = [nb.GetIdx() for nb in mol.GetAtomWithIdx(hub).GetNeighbors()
               if nb.GetAtomicNum() > 1 and nb.GetIdx() not in path_set]
        if len(exo) != 2:
            return None
        arm_carbons.extend((hub, ac) for ac in exo)
    if len(arm_carbons) != 4:
        return None

    # All four arms must resolve to the SAME (parent, locant). Amine arms
    # never allowed (polyamine-parent trap, -- allow_amine=False.
    resolved = [
        _acyclic_arm_parent_and_locant(mol, ac, hub, allow_amine=False)
        for hub, ac in arm_carbons
    ]
    if resolved[0] is None or any(r != resolved[0] for r in resolved[1:]):
        return None
    parent_name, attach_locant = resolved[0]

    # Whole-molecule coverage: {2 N + chain + 4 arms} == every heavy atom, so
    # no atom is silently dropped by the name.
    covered = {n1, n2} | set(chain)
    for hub, ac in arm_carbons:
        stack = [ac]
        while stack:
            a = stack.pop()
            if a in covered or a == hub:
                continue
            covered.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                if nb.GetAtomicNum() > 1 and ni not in covered and ni != hub:
                    stack.append(ni)
    heavy_all = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    if covered != heavy_all:
        return None

    # Compose the composite central group name. The two hubs sit on the two
    # termini of the m-carbon chain, so the alkanediyl locants are the lowest
    # set {1, m}: ethane-1,2-diyl, propane-1,3-diyl, butane-1,4-diyl.
    stem = get_chain_prefix(m)
    if stem is None:
        return None
    if m == 1:
        # >N-CH2-N< is 'methylenedinitrilo' (the Blue Book), not an
        # 'ane-1,1-diyl' form -- out of this handler's scope; fail closed.
        return None
    bridge_name = f"{stem}ane-1,{m}-diyldinitrilo"

    return _assemble_multiplicative_name(
        attach_locant, bridge_name, parent_name, unit_count=4
    )


def _group14_hydride_unit_name(canon_smiles: str) -> Optional[str]:
    """Name a FREE homonuclear Group-14 catenated hydride unit:
    [SiH3][SiH3] -> 'disilane'. Fail-closed: one element from Si/Ge/Sn/Pb,
    an unbranched fully-H-saturated standard-valence chain of 2-5 atoms."""
    _STEMS = {'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane',
              'Pb': 'plumbane'}
    _MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta'}
    frag = Chem.MolFromSmiles(canon_smiles)
    if frag is None:
        return None
    syms = {a.GetSymbol() for a in frag.GetAtoms()}
    if len(syms) != 1:
        return None
    sym = next(iter(syms))
    if sym not in _STEMS:
        return None
    n = frag.GetNumAtoms()
    if n not in _MULT:
        return None
    degrees = sorted(a.GetDegree() for a in frag.GetAtoms())
    if degrees != [1] * 2 + [2] * (n - 2):
        return None  # branched or disconnected
    # Bracket atoms keep their H count when the bridge bond is cut, so the
    # attachment atom arrives H-deficient by exactly one (valence 3). Require
    # exactly ONE such cut point, on a chain TERMINAL (= unit locant 1).
    cut_terminals = 0
    for a in frag.GetAtoms():
        if a.GetFormalCharge() != 0:
            return None
        tv, rad = a.GetTotalValence(), a.GetNumRadicalElectrons()
        if tv == 4 and rad == 0:
            continue
        if tv + rad == 4 and a.GetDegree() <= 1:
            cut_terminals += 1
            continue
        return None
    if cut_terminals != 1:
        return None
    return f"{_MULT[n]}{_STEMS[sym]}"


_G14_HYDRIDE_STEMS = {'Si': 'silane', 'Ge': 'germane',
                      'Sn': 'stannane', 'Pb': 'plumbane'}
_G14_CHAIN_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                   7: 'hepta', 8: 'octa', 9: 'nona', 10: 'deca'}

# a phase (a)): a MONONUCLEAR Group-14 hydride cited as a
# multiplicative parent unit ('silane', 'trimethylsilane',...) takes bis/tris,
# NOT di/tri. (a) Note lists Si/Ge/Sn/Pb explicitly: 'disilane' and
# 'trisilane' already name the CATENATED polynuclear hydrides (H3Si-SiH3,
# H3Si-SiH2-SiH3), so 'di(silane)'/'tri(silane)' would be ambiguous and the
# complex prefix is mandatory -> bis(silane)/tris(silane) (the Blue Book,
# (benzene-1,3,5-triyl)tris(silane) (PIN)) and bis(trimethylsilane)
# (the Blue Book). The methyl-substituted parents substitution on
# the silane parent hydride) are enumerated to the count the unit namer emits.
_METHYL_UNIT_PREFIXES = ('', 'methyl', 'dimethyl', 'trimethyl')
_G14_MONONUCLEAR_HYDRIDE_PARENTS = frozenset(
    mp + stem
    for stem in _G14_HYDRIDE_STEMS.values()
    for mp in _METHYL_UNIT_PREFIXES
)

# v50 A4 / (a) Note, the Blue Book): a MONONUCLEAR Group-15
# (pnictogen) hydride cited as a multiplicative parent unit ('phosphane',
# 'arsane',...) likewise takes bis/tris, NOT di/tri — 'diphosphane'/'triphosphane'
# already name the CATENATED hydrides (H2P-PH2, H2P-PH-PH2), so the complex prefix
# is mandatory: (butane-1,2,4-triyl)tris(phosphane) (PIN, the Blue Book),
# (1,2-phenylene)bis(arsane) (PIN, the Blue Book).
_PNICTOGEN_HYDRIDE_STEMS = {'P': 'phosphane', 'As': 'arsane',
                            'Sb': 'stibane', 'Bi': 'bismuthane'}
_PNICTOGEN_MONONUCLEAR_HYDRIDE_PARENTS = frozenset(
    _PNICTOGEN_HYDRIDE_STEMS.values()
)


def name_free_homonuclear_group14_hydride(mol) -> Optional[str]:
    """Free-molecule homonuclear Group-14 catenated parent hydride /
    : n identical Si/Ge/Sn/Pb atoms in an unbranched, fully-H-saturated,
    standard-valence chain -> di/tri/...+stem, with the ene/yne
    modification for a 2-atom unsaturated chain::

        [SiH3][SiH3] -> disilane
        [SiH3][SiH2][SiH3] -> trisilane
        [GeH3][GeH3] -> digermane
        [GeH2]=[GeH2] -> digermene, BB 38147)
        [SiH2]=[SiH2] -> disilene

    Fail-closed (returns None) for any carbon / halide / stray heteroatom, a
    branch, a ring, a charge / radical, a non-standard valence, a single hub
    (bare mononuclear hydride, handled elsewhere), or unsaturation on a chain
    longer than 2 atoms (locant machinery not built here — never a wrong name).
    Pure: no mol mutation.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    heavy = [a for a in atoms_of(mol) if a.GetSymbol() != 'H']
    if len(heavy) < 2:
        return None
    syms = {a.GetSymbol() for a in heavy}
    if len(syms) != 1:
        return None
    stem = _G14_HYDRIDE_STEMS.get(next(iter(syms)))
    if stem is None:
        return None
    n = len(heavy)
    mult = _G14_CHAIN_MULT.get(n)
    if mult is None:
        return None

    from .lambda_convention import nonstandard_bonding_number
    idxs = {a.GetIdx() for a in heavy}
    deg = {}
    for a in heavy:
        if (a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0
                or a.IsInRing()):
            return None
        if nonstandard_bonding_number(mol, a.GetIdx()) is not None:
            return None
        d = sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in idxs)
        if d > 2:
            return None  # branch
        deg[a.GetIdx()] = d

    ends = [i for i, d in deg.items() if d == 1]
    if len(ends) != 2:
        return None  # ring (no degree-1 atom) or malformed

    # Walk the chain from one end to order the atoms 1..n.
    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxts = [nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors()
                if nb.GetIdx() in idxs and nb.GetIdx() != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return None
        prev, cur = cur, nxts[0]
        order.append(cur)
    if len(order) != n:
        return None  # defensive: ring / disconnected

    # unsaturation along the chain.
    n_double = n_triple = 0
    for j in range(n - 1):
        bt = mol.GetBondBetweenAtoms(order[j], order[j + 1]).GetBondType()
        if bt == Chem.BondType.DOUBLE:
            n_double += 1
        elif bt == Chem.BondType.TRIPLE:
            n_triple += 1
        elif bt != Chem.BondType.SINGLE:
            return None
    total_unsat = n_double + n_triple
    if total_unsat == 0:
        return f"{mult}{stem}"
    if n != 2 or total_unsat != 1:
        return None  # only the 2-atom -ene/-yne is built (no locant needed)
    base = stem[:-3]  # 'sil' / 'germ' / 'stann' / 'plumb'
    return f"{mult}{base}{'ene' if n_double else 'yne'}"


def _try_group14_hydride_bridge(mol) -> Optional[str]:
    """CH2 joining two identical unbranched Group-14 catenated hydrides at
    their TERMINAL atoms (b), BB verbatim):
    [SiH3][SiH2]C[SiH2][SiH3] -> 1,1'-methylenebis(disilane).

    Fail-closed: acyclic CH2 bridge, both neighbours Si/Ge/Sn/Pb, identical
    fragments that _group14_hydride_unit_name certifies, attachment terminal
    (locant 1). Assembled directly — 'disilane' begins with a multiplying
    prefix, so mandates bis + enclosing marks."""
    _G14 = {'Si', 'Ge', 'Sn', 'Pb'}
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'C' or atom.GetTotalNumHs() != 2:
            continue
        if atom.IsInRing() or atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetSymbol() not in _G14 for n in heavy):
            continue
        # Terminal attachment: each bridged atom has exactly one further
        # chain neighbour (locant 1 on the unit).
        if any(len([m for m in n.GetNeighbors() if m.GetAtomicNum() > 1
                    and m.GetIdx() != atom.GetIdx()]) != 1 for n in heavy):
            continue
        fragments = _split_at_bridge(mol, atom.GetIdx(),
                                     [n.GetIdx() for n in heavy])
        if fragments is None:
            continue
        smi_a, smi_b, _ca, _cb = fragments
        canon_a = canon_smiles(smi_a)
        if canon_a != canon_smiles(smi_b):
            continue
        unit = _group14_hydride_unit_name(canon_a)
        if unit is None:
            continue
        return f"1,1'-methylenebis({unit})"
    return None


def _walk_homogeneous_chalcogen_arm(mol, start: int, bridge: int):
    """From ``start`` (a chalcogen bonded to a central CH2 ``bridge``), walk an
    unbranched HOMOGENEOUS chalcogen chain away from ``bridge``. Returns
    ``(element, length, atom_idxs)`` or None. The bridge attaches at a TERMINAL
    chalcogen (unit locant 1); every arm atom is the same chalcogen element,
    unbranched, and bears only H besides the chain (a BARE polychalcogen arm — an
    organyl on the arm declines, so only the bare-trisulfane methylenebis case
    fires). Pure."""
    element = mol.GetAtomWithIdx(start).GetSymbol()
    idxs = []
    prev, cur = bridge, start
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if atom.GetSymbol() != element or atom.GetFormalCharge() != 0:
            return None
        idxs.append(cur)
        chal_nbrs = []
        for n in atom.GetNeighbors():
            if n.GetAtomicNum() <= 1 or n.GetIdx() == prev:
                continue
            if n.GetSymbol() != element:
                return None                       # organyl / heteroatom on arm -> decline
            bond = mol.GetBondBetweenAtoms(cur, n.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            chal_nbrs.append(n.GetIdx())
        if len(chal_nbrs) == 0:
            break                                 # far terminus (bears only H)
        if len(chal_nbrs) > 1:
            return None                           # branch
        prev, cur = cur, chal_nbrs[0]
    return (element, len(idxs), idxs)


def _try_methylene_bis_polychalcogen(mol) -> Optional[str]:
    """ (BB 39379): a central -CH2- linking two IDENTICAL homogeneous
    chalcogen-chain parent hydrides (n>=3) at their terminal chalcogen ->
    ``1,1'-methylenebis(trisulfane)`` (HS-S-S-CH2-S-S-SH). Sibling of
    _try_group14_hydride_bridge; the divalent central 'methylene' is.

    Fail-closed: a single neutral non-ring CH2 whose two heavy neighbours are both
    chalcogens, each beginning an identical BARE homogeneous chalcogen chain of
    n>=3 (the >=3 gate keeps sulfides/disulfides out, mirroring polychalcogen), the
    whole molecule being exactly the CH2 + both arms. Pure: no mol mutation."""
    from .polychalcogen import _CHALCOGEN_STEMS, _MULTIPLIER
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'C' or atom.GetTotalNumHs() != 2 or atom.IsInRing():
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetSymbol() not in _CHALCOGEN_STEMS
                                  for n in heavy):
            continue
        arm_a = _walk_homogeneous_chalcogen_arm(mol, heavy[0].GetIdx(),
                                                atom.GetIdx())
        arm_b = _walk_homogeneous_chalcogen_arm(mol, heavy[1].GetIdx(),
                                                atom.GetIdx())
        if arm_a is None or arm_b is None:
            continue
        (e1, n1, s1), (e2, n2, s2) = arm_a, arm_b
        if e1 != e2 or n1 != n2 or n1 < 3:
            continue
        covered = {atom.GetIdx()} | set(s1) | set(s2)
        if any(a.GetIdx() not in covered
               for a in atoms_of(mol) if a.GetAtomicNum() > 1):
            continue
        stem = _CHALCOGEN_STEMS.get(e1)
        mult = _MULTIPLIER.get(n1)
        if stem is None or mult is None:
            continue
        return f"1,1'-methylenebis({mult}{stem})"
    return None


def _try_azanediyl_methylene_phosphonic(mol) -> Optional[str]:
    """ Note: OP(=O)(O)CNCP(=O)(O)O ->
    [azanediylbis(methylene)]bis(phosphonic acid) (BB verbatim — the
    composite central group takes no separate prefixes and the phosphonic
    acid unit carries no locants).

    Exact-shape fail-closed recognizer: central NH with two CH2 arms, each
    bonded to a clean neutral P(=O)(OH)2, and nothing else in the molecule."""
    if mol.GetNumAtoms() != 11:
        return None
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'N' or atom.GetTotalNumHs() != 1:
            continue
        if atom.IsInRing() or atom.GetFormalCharge() != 0:
            continue
        arms = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(arms) != 2:
            continue
        ok = True
        for c in arms:
            if c.GetSymbol() != 'C' or c.GetTotalNumHs() != 2:
                ok = False
                break
            p_nbrs = [n for n in c.GetNeighbors() if n.GetAtomicNum() > 1
                      and n.GetIdx() != atom.GetIdx()]
            if len(p_nbrs) != 1 or p_nbrs[0].GetSymbol() != 'P':
                ok = False
                break
            p = p_nbrs[0]
            if p.GetFormalCharge() != 0:
                ok = False
                break
            o_nbrs = [n for n in p.GetNeighbors() if n.GetIdx() != c.GetIdx()]
            if len(o_nbrs) != 3 or any(o.GetSymbol() != 'O' for o in o_nbrs):
                ok = False
                break
            oxo = [o for o in o_nbrs if o.GetTotalNumHs() == 0]
            oh = [o for o in o_nbrs if o.GetTotalNumHs() == 1]
            if len(oxo) != 1 or len(oh) != 2:
                ok = False
                break
        if ok:
            return "[azanediylbis(methylene)]bis(phosphonic acid)"
    return None


def _try_triyl_two_atom_bridge(mol, ring_atoms: set) -> Optional[str]:
    """Two-carbon central unit carrying THREE identical ring parents
     /:

    (HOOC-C6H4-)2CH-CH2-C6H4-COOH -> 4,4',4''-(ethane-1,1,2-triyl)tribenzoic
    acid; (H2N-C6H4-)2C=CH-C6H4-NH2 -> 4,4',4''-(ethene-1,1,2-triyl)trianiline.

    Fail-closed: both bridge carbons non-ring/neutral, ring-attachment
    pattern (2,1), no other heavy substituents, all three fragments
    identical, all three attachment locants equal."""
    for bond in bonds_of(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != 'C' or a2.GetSymbol() != 'C':
            continue
        idx1, idx2 = a1.GetIdx(), a2.GetIdx()
        if idx1 in ring_atoms or idx2 in ring_atoms:
            continue
        if a1.GetFormalCharge() != 0 or a2.GetFormalCharge() != 0:
            continue
        border = bond.GetBondTypeAsDouble()
        if border == 1.0:
            bridge_name = "ethane-1,1,2-triyl"
        elif border == 2.0:
            bridge_name = "ethene-1,1,2-triyl"
        else:
            continue
        ring_nbrs_1 = [n for n in a1.GetNeighbors()
                       if n.GetIdx() in ring_atoms]
        ring_nbrs_2 = [n for n in a2.GetNeighbors()
                       if n.GetIdx() in ring_atoms]
        heavy_1 = [n for n in a1.GetNeighbors() if n.GetAtomicNum() > 1]
        heavy_2 = [n for n in a2.GetNeighbors() if n.GetAtomicNum() > 1]
        # C1 carries two rings, C2 one ring; nothing else heavy on either.
        if not (len(ring_nbrs_1) == 2 and len(heavy_1) == 3
                and len(ring_nbrs_2) == 1 and len(heavy_2) == 2):
            # try the swapped orientation
            if (len(ring_nbrs_2) == 2 and len(heavy_2) == 3
                    and len(ring_nbrs_1) == 1 and len(heavy_1) == 2):
                a1, a2 = a2, a1
                idx1, idx2 = idx2, idx1
                ring_nbrs_1, ring_nbrs_2 = ring_nbrs_2, ring_nbrs_1
            else:
                continue

        # Split: remove both bridge carbons -> exactly 3 identical fragments.
        emol = RWMol(Chem.RWMol(mol))
        for ridx in sorted((idx1, idx2), reverse=True):
            emol.RemoveAtom(ridx)
        try:
            Chem.SanitizeMol(emol)
        except Exception:
            continue
        result_mol = emol.GetMol()
        frag_mols = Chem.GetMolFrags(result_mol, asMols=True,
                                     sanitizeFrags=True)
        if len(frag_mols) != 3:
            continue
        canon_set = {canon_smiles(Chem.MolToSmiles(f)) for f in frag_mols}
        if len(canon_set) != 1:
            continue
        parent_name = _name_parent(canon_set.pop())
        if parent_name is None:
            continue

        conns = ([(idx1, n.GetIdx()) for n in ring_nbrs_1]
                 + [(idx2, n.GetIdx()) for n in ring_nbrs_2])
        locants = [_get_bridge_locant(mol, b, c, ring_atoms)
                   for b, c in conns]
        if any(l != locants[0] for l in locants[1:]):
            continue
        result = _assemble_multiplicative_name(
            locants[0], bridge_name, parent_name, unit_count=3
        )
        if result is not None:
            return result
        continue
    return None


def _try_methylenebis_oxy_bridge(mol, ring_atoms: set) -> Optional[str]:
    """Composite O-CH2-O central bridge joining two identical ring parents
    (Wave-2 completion C,: Oc1ccc(OCOc2ccc(O)cc2)cc1 ->
    4,4'-[methylenebis(oxy)]diphenol (BB verbatim).

    Fail-closed: central non-ring neutral CH2 with exactly two non-ring
    ether-O neighbours, each attached to exactly one ring atom; identical
    fragments; unit resolution via the shared _resolve_unit_and_assemble."""
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'C' or atom.GetTotalNumHs() != 2:
            continue
        idx = atom.GetIdx()
        if idx in ring_atoms or atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'O'
               or n.GetTotalNumHs() != 0 or n.GetFormalCharge() != 0
               for n in heavy):
            continue
        ring_conns = []
        ok = True
        for o in heavy:
            rn = [n for n in o.GetNeighbors() if n.GetAtomicNum() > 1
                  and n.GetIdx() != idx]
            if len(rn) != 1 or rn[0].GetIdx() not in ring_atoms:
                ok = False
                break
            ring_conns.append(rn[0].GetIdx())
        if not ok:
            continue

        fragments = _split_at_bridge(
            mol, idx, ring_conns, extra_remove=[o.GetIdx() for o in heavy])
        if fragments is None:
            continue
        smi_a, smi_b, _ca, _cb = fragments
        canon_a = canon_smiles(smi_a)
        if canon_a != canon_smiles(smi_b):
            continue
        result = _resolve_unit_and_assemble(
            mol, [o.GetIdx() for o in heavy], ring_conns, [idx],
            "methylenebis(oxy)", ring_atoms, canon_a,
        )
        if result is not None:
            return result
        continue
    return None


def _try_oxybis_azanylylidenemethanylylidene_bridge(
    mol, ring_atoms: set,
) -> Optional[str]:
    """Composite =CH-N= -O- =N-CH= central bridge joining two identical ring
    parents (Wave-2 completion C, yl/ylidene composite):
    PhCH=N-O-N=CHPh -> 1,1'-[oxybis(azanylylidenemethanylylidene)]dibenzene.

    Fail-closed: central non-ring ether O with two =N neighbours, each N
    double-bonded to a CH that attaches to exactly one ring atom; any
    ASSIGNED C=N stereo declines (the composite bridge name cannot carry
    stereo descriptors); identical fragments."""
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'O' or atom.GetTotalNumHs() != 0:
            continue
        idx = atom.GetIdx()
        if idx in ring_atoms or atom.GetFormalCharge() != 0:
            continue
        n_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(n_nbrs) != 2 or any(n.GetSymbol() != 'N' or n.GetIdx()
                                   in ring_atoms for n in n_nbrs):
            continue
        bridge_idxs = [idx]
        ring_conns = []
        ok = True
        for n in n_nbrs:
            if n.GetFormalCharge() != 0 or n.GetTotalNumHs() != 0:
                ok = False
                break
            c_nbrs = [m for m in n.GetNeighbors() if m.GetAtomicNum() > 1
                      and m.GetIdx() != idx]
            bond = None if not c_nbrs else mol.GetBondBetweenAtoms(
                n.GetIdx(), c_nbrs[0].GetIdx())
            if (len(c_nbrs) != 1 or c_nbrs[0].GetSymbol() != 'C'
                    or bond.GetBondType() != Chem.BondType.DOUBLE
                    or bond.GetStereo() != Chem.BondStereo.STEREONONE):
                ok = False
                break
            c = c_nbrs[0]
            if (c.GetIdx() in ring_atoms or c.GetTotalNumHs() != 1
                    or c.GetFormalCharge() != 0):
                ok = False
                break
            rn = [m for m in c.GetNeighbors() if m.GetAtomicNum() > 1
                  and m.GetIdx() != n.GetIdx()]
            if len(rn) != 1 or rn[0].GetIdx() not in ring_atoms:
                ok = False
                break
            bridge_idxs.extend([n.GetIdx(), c.GetIdx()])
            ring_conns.append(rn[0].GetIdx())
        if not ok:
            continue

        fragments = _split_at_bridge(
            mol, idx, ring_conns,
            extra_remove=[b for b in bridge_idxs if b != idx])
        if fragments is None:
            continue
        smi_a, smi_b, _ca, _cb = fragments
        canon_a = canon_smiles(smi_a)
        if canon_a != canon_smiles(smi_b):
            continue
        result = _resolve_unit_and_assemble(
            mol, bridge_idxs, ring_conns, [],
            "oxybis(azanylylidenemethanylylidene)", ring_atoms, canon_a,
        )
        if result is not None:
            return result
        continue
    return None


_CHALCOGEN_BIS_METHYLENE = {
    # (BB 5242): '-CH2-O-CH2- oxybis(methylene) (preferred
    # prefix)'; S-analog family (BB 16256).
    'O': 'oxybis(methylene)',
    'S': 'sulfanediylbis(methylene)',
}


def _try_chalcogenbis_methylene_bridge(mol, ring_atoms: set) -> Optional[str]:
    """Composite CH2-X-CH2 central bridge (X = O/S) joining two identical
    ring parents (w2f p1, /:
    Oc1ccc(COCc2ccc(O)cc2)cc1 -> 4,4'-[oxybis(methylene)]diphenol.
    Modeled 1:1 on _try_methylenebis_oxy_bridge (:862) with the O/CH2 roles
    swapped.

    Fail-closed: central non-ring neutral 0-H chalcogen with exactly two
    non-ring neutral CH2 carbon neighbours, each CH2 bonded to exactly one
    ring atom; identical fragments; unit resolution AND the (3)
    locant-identity requirement via the shared _resolve_unit_and_assemble
    (its per-connection locant equality check is what rejects a 3-OH/4-OH
    pair whose free fragments are canon-identical phenols)."""
    for atom in atoms_of(mol):
        bridge_name = _CHALCOGEN_BIS_METHYLENE.get(atom.GetSymbol())
        if bridge_name is None or atom.GetTotalNumHs() != 0:
            continue
        idx = atom.GetIdx()
        if idx in ring_atoms or atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'C'
               or n.GetTotalNumHs() != 2 or n.GetFormalCharge() != 0
               for n in heavy):
            continue
        ring_conns = []
        ok = True
        for ch2 in heavy:
            rn = [n for n in ch2.GetNeighbors() if n.GetAtomicNum() > 1
                  and n.GetIdx() != idx]
            if len(rn) != 1 or rn[0].GetIdx() not in ring_atoms:
                ok = False
                break
            ring_conns.append(rn[0].GetIdx())
        if not ok:
            continue

        fragments = _split_at_bridge(
            mol, idx, ring_conns, extra_remove=[c.GetIdx() for c in heavy])
        if fragments is None:
            continue
        smi_a, smi_b, _ca, _cb = fragments
        canon_a = canon_smiles(smi_a)
        if canon_a != canon_smiles(smi_b):
            continue
        result = _resolve_unit_and_assemble(
            mol, [c.GetIdx() for c in heavy], ring_conns, [idx],
            bridge_name, ring_atoms, canon_a,
        )
        if result is not None:
            return result
        continue
    return None


def _try_silanediyl_methylene_bridge(mol, ring_atoms: set) -> Optional[str]:
    """Composite CH2-SiH2-CH2 central bridge joining two identical ring
    parents: HOOC-C6H4-CH2-SiH2-CH2-C6H4-COOH ->
    4,4'-[silanediylbis(methylene)]dibenzoic acid.

    Fail-closed: central non-ring neutral SiH2 with exactly two non-ring CH2
    neighbours, each attached to exactly one ring atom; identical fragments;
    equal attachment locants."""
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'Si' or atom.GetTotalNumHs() != 2:
            continue
        idx = atom.GetIdx()
        if idx in ring_atoms or atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2:
            continue
        if any(n.GetIdx() in ring_atoms or n.GetSymbol() != 'C'
               or n.GetTotalNumHs() != 2 or n.GetFormalCharge() != 0
               for n in heavy):
            continue
        ring_conns = []
        ok = True
        for c in heavy:
            rn = [n for n in c.GetNeighbors() if n.GetAtomicNum() > 1
                  and n.GetIdx() != idx]
            if len(rn) != 1 or rn[0].GetIdx() not in ring_atoms:
                ok = False
                break
            ring_conns.append(rn[0].GetIdx())
        if not ok:
            continue

        fragments = _split_at_bridge(
            mol, idx, ring_conns, extra_remove=[c.GetIdx() for c in heavy])
        if fragments is None:
            continue
        smi_a, smi_b, _ca, _cb = fragments
        canon_a = canon_smiles(smi_a)
        if canon_a != canon_smiles(smi_b):
            continue
        result = _resolve_unit_and_assemble(
            mol, [c.GetIdx() for c in heavy], ring_conns, [idx],
            "silanediylbis(methylene)", ring_atoms, canon_a,
        )
        if result is not None:
            return result
        continue
    return None


def _try_ethylenedioxy_bridge(mol) -> Optional[str]:
    """Composite central bridge -O-CH2-CH2-O- joining two identical acyclic
    functional parents (Wave2 T5a;:

    HOOC-CH2-O-CH2CH2-O-CH2-COOH ->
        2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid (BB verbatim PIN);
    HO-CH2CH2-O-CH2CH2-O-CH2CH2-OH (triethylene glycol) ->
        2,2'-[ethane-1,2-diylbis(oxy)]di(ethan-1-ol).

    Fail-closed: the central unit is exactly an acyclic, uncharged
    CH2-CH2 flanked by two ether oxygens; each O's outer neighbour starts a
    strict linear acid/ol arm; both arms resolve identically. Longer
    oxa-chains (tetraethylene glycol: the arm would contain a second ether O)
    decline via the arm namer's heteroatom coverage check.
    """
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    for bond in bonds_of(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != 'C' or a2.GetSymbol() != 'C':
            continue
        if a1.GetTotalNumHs() != 2 or a2.GetTotalNumHs() != 2:
            continue
        if a1.GetIdx() in ring_atoms or a2.GetIdx() in ring_atoms:
            continue
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        if a1.GetFormalCharge() != 0 or a2.GetFormalCharge() != 0:
            continue

        outer1 = [n for n in a1.GetNeighbors()
                  if n.GetIdx() != a2.GetIdx() and n.GetAtomicNum() > 1]
        outer2 = [n for n in a2.GetNeighbors()
                  if n.GetIdx() != a1.GetIdx() and n.GetAtomicNum() > 1]
        if len(outer1) != 1 or len(outer2) != 1:
            continue
        o1, o2 = outer1[0], outer2[0]
        if not all(
            o.GetSymbol() == 'O' and o.GetTotalNumHs() == 0
            and o.GetFormalCharge() == 0 and o.GetIdx() not in ring_atoms
            for o in (o1, o2)
        ):
            continue

        arm_start1 = [n for n in o1.GetNeighbors()
                      if n.GetIdx() != a1.GetIdx() and n.GetAtomicNum() > 1]
        arm_start2 = [n for n in o2.GetNeighbors()
                      if n.GetIdx() != a2.GetIdx() and n.GetAtomicNum() > 1]
        if len(arm_start1) != 1 or len(arm_start2) != 1:
            continue
        c1, c2 = arm_start1[0], arm_start2[0]
        if any(c.GetIdx() in ring_atoms or c.GetSymbol() != 'C' for c in (c1, c2)):
            continue

        arm_a = _acyclic_arm_parent_and_locant(mol, c1.GetIdx(), o1.GetIdx())
        if arm_a is None:
            continue
        arm_b = _acyclic_arm_parent_and_locant(mol, c2.GetIdx(), o2.GetIdx())
        if arm_b != arm_a:
            continue

        parent_name, attach_locant = arm_a
        return _assemble_multiplicative_name(
            attach_locant, "ethane-1,2-diylbis(oxy)", parent_name, unit_count=2
        )

    return None


def _group14_mononuclear_hydride_unit(mol, x_idx: int, attach_idx: int):
    """Name a MONONUCLEAR Group-14 hydride substituent unit attached to a
    central multiplicative group at ``attach_idx`` (b),
    , (a)).

    The unit is a single Si/Ge/Sn/Pb atom bearing, besides the one central
    attachment bond, only H and/or TERMINAL methyl groups::

        -SiH3 -> ("silane", {Si})
        -Si(CH3)3 -> ("trimethylsilane", {Si, 3 x CH3})

    Returns ``(unit_name, unit_atom_idxs)`` or ``None`` (fail closed) for a
    catenated hydride (Si-Si), any non-methyl substituent, a charge / radical,
    a non-standard valence, or a ring atom. Tight by design: only the
    methyl-decorated mononuclear parents the BB cites as PIN units are built.
    """
    x = mol.GetAtomWithIdx(x_idx)
    stem = _G14_HYDRIDE_STEMS.get(x.GetSymbol())
    if stem is None:
        return None
    if (x.GetFormalCharge() != 0 or x.GetNumRadicalElectrons() != 0
            or x.IsInRing() or x.GetTotalValence() != 4):
        return None
    bond = mol.GetBondBetweenAtoms(x_idx, attach_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    methyls: List[int] = []
    saw_attach = False
    for nb in x.GetNeighbors():
        if nb.GetIdx() == attach_idx:
            saw_attach = True
            continue
        # every OTHER heavy neighbour must be a terminal methyl carbon
        if (nb.GetSymbol() != 'C' or nb.GetTotalNumHs() != 3
                or nb.GetDegree() != 1 or nb.GetFormalCharge() != 0
                or nb.GetNumRadicalElectrons() != 0 or nb.IsInRing()):
            return None
        mbond = mol.GetBondBetweenAtoms(x_idx, nb.GetIdx())
        if mbond.GetBondType() != Chem.BondType.SINGLE:
            return None
        methyls.append(nb.GetIdx())
    if not saw_attach:
        return None
    m = len(methyls)
    # attach(1) + m methyls + H(3-m) must fill Group-14 valence 4 (no stray bonds)
    if m > 3 or x.GetTotalNumHs() != 3 - m:
        return None
    return _METHYL_UNIT_PREFIXES[m] + stem, {x_idx, *methyls}


def _lowest_chain_attach_locants(chain_order: List[int], attach_atoms: set) -> List[int]:
    """Lowest attachment-locant set for the free valences of a
    linear-chain central group, over the two numbering directions. Returns the
    sorted list of locants (deterministic, order-independent)."""
    best: Optional[List[int]] = None
    for seq in (chain_order, list(reversed(chain_order))):
        locant_of = {a: i + 1 for i, a in enumerate(seq)}
        locs = sorted(locant_of[a] for a in attach_atoms)
        if best is None or locs < best:
            best = locs
    return best or []


def _try_group14_hydride_substituted_chain(mol) -> Optional[str]:
    """A saturated carbon CHAIN central group bearing >=2 IDENTICAL mononuclear
    Group-14 hydride units on DISTINCT carbons /;
    the Blue Book verbatim)::

        CC(C[Si](C)(C)C)[Si](C)(C)C -> (propane-1,2-diyl)bis(trimethylsilane)
        [SiH3]CC[SiH3] -> (ethane-1,2-diyl)bis(silane)

    A carbon-only chain cannot be a multiplied parent (b)), so it is
    the central group and the silanes are the identical parents. Fail closed
    (return None) unless: one acyclic fragment; every Group-14 atom is a clean
    mononuclear methyl/H unit (``_group14_mononuclear_hydride_unit``); >=2
    units, all identically named, on distinct carbons; the non-unit heavy atoms
    are all carbon and form a single unbranched chain; chain + units cover every
    heavy atom. Asymmetric traps (1-chloro-...bis(silane) at the Blue Book) are
    excluded because chlorosilane is not a methyl/H unit and the units differ.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() != 0:
        return None

    _G14 = set(_G14_HYDRIDE_STEMS)
    unit_atoms_all: set = set()
    units: List[Tuple[str, int]] = []  # (unit_name, attach_carbon_idx)
    for atom in atoms_of(mol):
        if atom.GetSymbol() not in _G14:
            continue
        heavy = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
        # the central attachment is the single non-(terminal-methyl) neighbour
        attach = [nb for nb in heavy
                  if not (nb.GetSymbol() == 'C' and nb.GetTotalNumHs() == 3
                          and nb.GetDegree() == 1)]
        if len(attach) != 1 or attach[0].GetSymbol() != 'C':
            return None
        res = _group14_mononuclear_hydride_unit(mol, atom.GetIdx(),
                                                attach[0].GetIdx())
        if res is None:
            return None
        unit_name, uatoms = res
        units.append((unit_name, attach[0].GetIdx()))
        unit_atoms_all |= uatoms

    if len(units) < 2 or len({u for u, _ in units}) != 1:
        return None
    unit_name = units[0][0]
    attach_carbons = [c for _, c in units]
    if len(set(attach_carbons)) != len(attach_carbons):
        return None  # two units on one carbon (methanediyl-type) -> out of scope

    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    central = heavy - unit_atoms_all
    if not central or any(mol.GetAtomWithIdx(i).GetSymbol() != 'C'
                          for i in central):
        return None  # central skeleton must be all-carbon
    if set(attach_carbons) - central:
        return None  # every attach carbon must lie on the central skeleton

    # the central carbons must form a single unbranched chain (a simple path)
    ends: List[int] = []
    adj: Dict[int, List[int]] = {}
    for i in central:
        nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                if nb.GetIdx() in central]
        if len(nbrs) > 2:
            return None  # branched central skeleton -> out of tight scope
        adj[i] = nbrs
        if len(nbrs) <= 1:
            ends.append(i)
    if len(central) < 2 or len(ends) != 2:
        return None

    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxt = [j for j in adj[cur] if j != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        order.append(cur)
    if len(order) != len(central):
        return None  # disconnected central skeleton

    locs = _lowest_chain_attach_locants(order, set(attach_carbons))
    if len(locs) != len(units):
        return None
    stem = get_chain_prefix(len(order))
    yl = SIMPLE_MULTIPLIERS.get(len(units))
    multiplier = _select_multiplier(unit_name, len(units))
    if yl is None or multiplier is None:
        return None
    central_name = f"{stem}ane-{','.join(str(x) for x in locs)}-{yl}yl"
    return f"({central_name}){multiplier}({unit_name})"


def _try_central_arene_group14_arms(mol) -> Optional[str]:
    """A bare benzene central group bearing >=2 IDENTICAL mononuclear Group-14
    hydride units; the Blue Book verbatim)::

        [SiH3]c1cc([SiH3])cc([SiH3])c1 -> (benzene-1,3,5-triyl)tris(silane)

    The single benzene ring is the central multiplicative group (one ring, so
    it cannot be multiplied); the silanes are the identical parents. Fail
    closed unless: exactly one 6-membered all-carbon aromatic ring; >=2 ring
    carbons each bearing a clean mononuclear Group-14 hydride unit
    (``_group14_mononuclear_hydride_unit``); all units identically named; ring +
    units cover every heavy atom (no other ring substituent).
    """
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 1:
        return None
    from ..perception.rings import get_ring_systems
    systems = get_ring_systems(mol)
    if len(systems) != 1 or len(systems[0]) != 6:
        return None
    ring_atoms = set(systems[0])
    for a in ring_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
            return None
    ring_cyclic = list(ring_info.AtomRings()[0])
    if set(ring_cyclic) != ring_atoms:
        return None

    attach_ring_atoms: set = set()
    unit_atoms_all: set = set()
    unit_names: set = set()
    for ra in ring_atoms:
        exo = [nb.GetIdx() for nb in mol.GetAtomWithIdx(ra).GetNeighbors()
               if nb.GetIdx() not in ring_atoms and nb.GetAtomicNum() > 1]
        if not exo:
            continue
        if len(exo) > 1:
            return None
        res = _group14_mononuclear_hydride_unit(mol, exo[0], ra)
        if res is None:
            return None
        unit_name, uatoms = res
        attach_ring_atoms.add(ra)
        unit_atoms_all |= uatoms
        unit_names.add(unit_name)

    if len(attach_ring_atoms) < 2 or len(unit_names) != 1:
        return None
    unit_name = next(iter(unit_names))

    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    if (ring_atoms | unit_atoms_all) != heavy:
        return None

    central = _central_arene_substituent_name(ring_cyclic, attach_ring_atoms)
    if central is None:
        return None
    multiplier = _select_multiplier(unit_name, len(attach_ring_atoms))
    if multiplier is None:
        return None
    return f"({central}){multiplier}({unit_name})"


def _pnictogen_mononuclear_hydride_unit(mol, x_idx: int, attach_idx: int):
    """Name a BARE mononuclear Group-15 (pnictogen) hydride substituent unit
    attached to a central multiplicative group at ``attach_idx``
    ; the Blue Book verbatim):

        -PH2 -> ("phosphane", {P})
        -AsH2 -> ("arsane", {As})

    Returns ``(unit_name, unit_atom_idxs)`` or ``None`` (fail closed) for a
    catenated hydride (P-P), any organyl substituent (dimethylphosphane is out
    of this bare scope), a charge / radical, a non-standard valence, a
    non-single attachment bond, or a ring atom. Tight by design so the handler
    only fires for the whole-molecule {central skeleton + bare pnictogen units}
    shape the BB cites as the multiplicative PIN.
    """
    x = mol.GetAtomWithIdx(x_idx)
    stem = _PNICTOGEN_HYDRIDE_STEMS.get(x.GetSymbol())
    if stem is None:
        return None
    if (x.GetFormalCharge() != 0 or x.GetNumRadicalElectrons() != 0
            or x.IsInRing() or x.GetTotalValence() != 3):
        return None  # standard trivalent pnictogen only (no lambda-5)
    bond = mol.GetBondBetweenAtoms(x_idx, attach_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    heavy = [nb.GetIdx() for nb in x.GetNeighbors() if nb.GetAtomicNum() > 1]
    if heavy != [attach_idx]:
        return None  # bare -XH2: the ONLY heavy neighbour is the attachment
    if x.GetTotalNumHs() != 2:
        return None  # attach(1) + H(2) fills the trivalent valence
    return stem, {x_idx}


def _try_pnictogen_hydride_substituted_chain(mol) -> Optional[str]:
    """A saturated carbon CHAIN central group bearing >=2 IDENTICAL bare
    mononuclear pnictogen hydride units on DISTINCT carbons
     / (b); the Blue Book verbatim)::

        PCCC(P)CP -> (butane-1,2,4-triyl)tris(phosphane)

    A carbon-only chain cannot be a multiplied parent (b)), so it is
    the central group and the phosphanes are the identical parents. Pnictogen
    sibling of ``_try_group14_hydride_substituted_chain``; fail closed unless:
    one acyclic fragment; every P/As/Sb/Bi is a clean bare mononuclear unit
    (``_pnictogen_mononuclear_hydride_unit``); >=2 units, all identically named,
    on distinct carbons; the non-unit heavy atoms are all carbon forming a
    single unbranched chain; chain + units cover every heavy atom.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() != 0:
        return None

    _PN = set(_PNICTOGEN_HYDRIDE_STEMS)
    unit_atoms_all: set = set()
    units: List[Tuple[str, int]] = []  # (unit_name, attach_carbon_idx)
    for atom in atoms_of(mol):
        if atom.GetSymbol() not in _PN:
            continue
        heavy = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
        if len(heavy) != 1 or heavy[0].GetSymbol() != 'C':
            return None
        res = _pnictogen_mononuclear_hydride_unit(mol, atom.GetIdx(),
                                                  heavy[0].GetIdx())
        if res is None:
            return None
        unit_name, uatoms = res
        units.append((unit_name, heavy[0].GetIdx()))
        unit_atoms_all |= uatoms

    if len(units) < 2 or len({u for u, _ in units}) != 1:
        return None
    unit_name = units[0][0]
    attach_carbons = [c for _, c in units]
    if len(set(attach_carbons)) != len(attach_carbons):
        return None  # two units on one carbon -> out of scope

    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    central = heavy - unit_atoms_all
    if not central or any(mol.GetAtomWithIdx(i).GetSymbol() != 'C'
                          for i in central):
        return None  # central skeleton must be all-carbon
    if set(attach_carbons) - central:
        return None

    # the central carbons must form a single unbranched chain (a simple path)
    ends: List[int] = []
    adj: Dict[int, List[int]] = {}
    for i in central:
        nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                if nb.GetIdx() in central]
        if len(nbrs) > 2:
            return None  # branched central skeleton -> out of tight scope
        adj[i] = nbrs
        if len(nbrs) <= 1:
            ends.append(i)
    if len(central) < 2 or len(ends) != 2:
        return None

    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxt = [j for j in adj[cur] if j != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        order.append(cur)
    if len(order) != len(central):
        return None  # disconnected central skeleton

    locs = _lowest_chain_attach_locants(order, set(attach_carbons))
    if len(locs) != len(units):
        return None
    stem = get_chain_prefix(len(order))
    yl = SIMPLE_MULTIPLIERS.get(len(units))
    multiplier = _select_multiplier(unit_name, len(units))
    if stem is None or yl is None or multiplier is None:
        return None
    central_name = f"{stem}ane-{','.join(str(x) for x in locs)}-{yl}yl"
    return f"({central_name}){multiplier}({unit_name})"


def _try_central_arene_pnictogen_arms(mol) -> Optional[str]:
    """A bare benzene central group bearing >=2 IDENTICAL bare mononuclear
    pnictogen hydride units; the Blue Book verbatim)::

        [AsH2]c1ccccc1[AsH2] -> (1,2-phenylene)bis(arsane)

    The single benzene ring is the central multiplicative group; the pnictogen
    hydrides are the identical parents. Pnictogen sibling of
    ``_try_central_arene_group14_arms``; fail closed unless: exactly one
    6-membered all-carbon aromatic ring; >=2 ring carbons each bearing a clean
    bare mononuclear pnictogen unit; all units identically named; ring + units
    cover every heavy atom.
    """
    if mol is None:
        return None
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 1:
        return None
    from ..perception.rings import get_ring_systems
    systems = get_ring_systems(mol)
    if len(systems) != 1 or len(systems[0]) != 6:
        return None
    ring_atoms = set(systems[0])
    for a in ring_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
            return None
    ring_cyclic = list(ring_info.AtomRings()[0])
    if set(ring_cyclic) != ring_atoms:
        return None

    attach_ring_atoms: set = set()
    unit_atoms_all: set = set()
    unit_names: set = set()
    for ra in ring_atoms:
        exo = [nb.GetIdx() for nb in mol.GetAtomWithIdx(ra).GetNeighbors()
               if nb.GetIdx() not in ring_atoms and nb.GetAtomicNum() > 1]
        if not exo:
            continue
        if len(exo) > 1:
            return None
        res = _pnictogen_mononuclear_hydride_unit(mol, exo[0], ra)
        if res is None:
            return None
        unit_name, uatoms = res
        attach_ring_atoms.add(ra)
        unit_atoms_all |= uatoms
        unit_names.add(unit_name)

    if len(attach_ring_atoms) < 2 or len(unit_names) != 1:
        return None
    unit_name = next(iter(unit_names))

    heavy = {a.GetIdx() for a in atoms_of(mol) if a.GetAtomicNum() > 1}
    if (ring_atoms | unit_atoms_all) != heavy:
        return None

    central = _central_arene_substituent_name(ring_cyclic, attach_ring_atoms)
    if central is None:
        return None
    multiplier = _select_multiplier(unit_name, len(attach_ring_atoms))
    if multiplier is None:
        return None
    return f"({central}){multiplier}({unit_name})"


def name_multiplicative(mol) -> Optional[str]:
    """Detect and name multiplicative nomenclature cases.

    Args:
        mol: RDKit Mol object.

    Returns:
        Multiplicative IUPAC name if applicable, None otherwise.
    """
    if mol is None:
        return None

    # Central-arene + acyclic identical-acid arms (d); defect V-8).
    # Runs before the >=2-ring-system bridge logic because here the benzene is a
    # SINGLE ring acting as the central multiplicative group, not a parent.
    result = _try_central_arene_acyclic_arms(mol)
    if result is not None:
        return result

    # Acyclic identical functional parents bridged by a divalent heteroatom
    #: OCCSCCO -> 2,2'-sulfanediyldi(ethan-1-ol). Runs before the
    # >=2-ring gate below (which only guards the ring-system bridge logic).
    # Tightly fail-closed so it never preempts an ester (Risk R1).
    result = _try_acyclic_heteroatom_bridge(mol)
    if result is not None:
        return result

    # Trivalent-N star over acyclic arms (Wave2 T5a):
    # NTA -> 2,2',2''-nitrilotriacetic acid.
    result = _try_acyclic_nitrilo_bridge(mol)
    if result is not None:
        return result

    # Two trivalent-N hubs joined by an alkanediyl chain, each with two
    # identical acyclic arms (EDTA family; /:
    # EDTA -> 2,2',2'',2'''-(ethane-1,2-diyldinitrilo)tetraacetic acid.
    result = _try_diamine_dinitrilo_bridge(mol)
    if result is not None:
        return result

    # Composite -O-CH2CH2-O- central bridge (Wave2 T5a):
    # 2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid (BB verbatim).
    result = _try_ethylenedioxy_bridge(mol)
    if result is not None:
        return result

    # CH2 joining two Group-14 catenated hydrides (Wave2 completion,
    # (b)): 1,1'-methylenebis(disilane).
    result = _try_group14_hydride_bridge(mol)
    if result is not None:
        return result

    # CH2 joining two identical homogeneous chalcogen-chain parent hydrides
    # (W3-P14,: 1,1'-methylenebis(trisulfane).
    result = _try_methylene_bis_polychalcogen(mol)
    if result is not None:
        return result

    # Composite CH2-NH-CH2 over phosphonic acid units (Wave2 completion,
    #: [azanediylbis(methylene)]bis(phosphonic acid).
    result = _try_azanediyl_methylene_phosphonic(mol)
    if result is not None:
        return result

    # a phase: an ACYCLIC carbon chain central group bearing >=2 identical
    # mononuclear Group-14 hydride units /: (propane-1,2-
    # diyl)bis(trimethylsilane), (ethane-1,2-diyl)bis(silane). Runs before the
    # >=2-ring gate (the central chain is acyclic).
    result = _try_group14_hydride_substituted_chain(mol)
    if result is not None:
        return result

    # a phase: a SINGLE benzene ring central group bearing >=2 identical
    # mononuclear Group-14 hydride units: (benzene-1,3,5-triyl)tris-
    # (silane). Runs before the >=2-ring gate (benzene is a single ring acting
    # as the central multiplicative group, not a parent).
    result = _try_central_arene_group14_arms(mol)
    if result is not None:
        return result

    # v50 A4: pnictogen siblings of the two Group-14 handlers above
    #: a carbon-chain central group bearing >=2 bare phosphane/
    # arsane units -> (butane-1,2,4-triyl)tris(phosphane); a benzene central
    # group bearing >=2 bare units -> (1,2-phenylene)bis(arsane). Both run
    # before the >=2-ring gate (the central group is acyclic / a single ring).
    result = _try_pnictogen_hydride_substituted_chain(mol)
    if result is not None:
        return result

    result = _try_central_arene_pnictogen_arms(mol)
    if result is not None:
        return result

    # Quick reject: need at least 2 ring systems
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() < 2:
        return None

    # a phase.B: topology guard for ring-assembly mutual exclusion.
    # If every inter-fragment connection is a single bond between two ring
    # atoms with NO bridge atom, return None and let the cascade fall
    # through to detect_ring_assembly (a phase's path). Cross-handler
    # regression test: tests/integration/test_assembly_vs_multiplicative_dispatch.py.
    #
    # Source: 154-internal notes; ring_assemblies.py:_find_inter_system_bonds:77.
    if _is_pure_single_bond_assembly(mol):
        return None  # ring_assemblies.py owns this case

    # Collect ring atom indices
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)

    # --- Try single-atom bridges first ---
    result = _try_single_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try two-atom bridges ---
    result = _try_two_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Try multi-atom bridges (3+ units, star topology) ---
    result = _try_multi_atom_bridges(mol, ring_atoms)
    if result is not None:
        return result

    # --- Two-carbon triyl central unit over 3 identical ring parents
    # (Wave2 completion, / ---
    result = _try_triyl_two_atom_bridge(mol, ring_atoms)
    if result is not None:
        return result

    # --- Composite CH2-SiH2-CH2 bridge (Wave2 completion, ---
    result = _try_silanediyl_methylene_bridge(mol, ring_atoms)
    if result is not None:
        return result

    # --- Composite O-CH2-O bridge (Wave2 completion C, ---
    result = _try_methylenebis_oxy_bridge(mol, ring_atoms)
    if result is not None:
        return result

    # --- Composite =CH-N=O=N-CH= bridge (Wave2 completion C, ---
    result = _try_oxybis_azanylylidenemethanylylidene_bridge(mol, ring_atoms)
    if result is not None:
        return result

    # --- Composite CH2-O-CH2 / CH2-S-CH2 bridge (w2f p1, ---
    result = _try_chalcogenbis_methylene_bridge(mol, ring_atoms)
    if result is not None:
        return result

    return None


def _bridge_units_config_mismatch(mol, bridge_idx: int, conn_atoms: List[int]) -> bool:
    """: multiplicative names are configuration-dependent. When the ring
    double bonds directly borne by the two (or more) bridge-attached carbons carry
    DIFFERENT specified E/Z configurations, the units are NOT identical and the
    multiplicative name must decline (the substitutive path names the
    mixed-configuration isomer).

    The fragment split drops ring-double-bond stereo, so the identity test on the
    split fragments cannot see it; this compares the specified bond stereo on the
    whole molecule BEFORE the split. Conservative / fail-closed: it can only make
    the namer DECLINE more often, never emit a new wrong name. Unspecified stereo
    on all units (the common no-stereo case) is treated as matching.
    """
    stereos = []
    for c in conn_atoms:
        atom = mol.GetAtomWithIdx(c)
        db_stereo = None
        for b in atom.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            # only ring double bonds (both ends in a ring, bond in a ring)
            if not b.IsInRing():
                continue
            db_stereo = b.GetStereo()
            break
        stereos.append(db_stereo)
    # Ignore units with no ring double bond on the bridge carbon.
    present = [s for s in stereos if s is not None]
    if len(present) < 2:
        return False
    # Decline if any two specified/unspecified states disagree.
    return any(s != present[0] for s in present[1:])


def _try_single_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find single-atom bridges between identical ring systems."""
    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]
        ring_nbrs = [n for n in heavy_neighbors if n.GetIdx() in ring_atoms]
        sub_nbrs = [n for n in heavy_neighbors if n.GetIdx() not in ring_atoms]

        # Single-atom bridge: exactly 2 ring neighbours. The plain bridges
        # (O/S/NH/CH2) have no other heavy neighbour; the Wave-2 carbonyl /
        # halomethylene bridges carry one bridge-owned substituent (=O / halo).
        if len(ring_nbrs) != 2:
            continue

        extra_remove: List[int] = []
        if not sub_nbrs:
            bridge_type = _classify_single_atom_bridge(atom)
            if bridge_type is None:
                continue
            bridge_name = _SINGLE_ATOM_BRIDGES.get(bridge_type)
            if bridge_name is None:
                continue
        else:
            ext = _classify_single_atom_bridge_ext(mol, atom, ring_nbrs, sub_nbrs)
            if ext is None:
                continue
            bridge_type, bridge_name, extra_remove = ext

        # Split the molecule at the bridge (removing bridge-owned substituents)
        nbr_indices = [n.GetIdx() for n in ring_nbrs]
        fragments = _split_at_bridge(mol, idx, nbr_indices, extra_remove)
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b, conn_atom_a, conn_atom_b = fragments

        # Check if fragments are identical
        canon_a = canon_smiles(frag_smiles_a)
        canon_b = canon_smiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        #: decline when the bridge-attached ring double bonds carry
        # DIFFERENT specified configurations (the units are then not identical;
        # the substitutive path names the mixed-config isomer). Fail-closed.
        if _bridge_units_config_mismatch(mol, idx, nbr_indices):
            continue

        # C4 /: substitutive-PIN decline for the NITROGEN
        # (NH -> 'imino') single-atom bridge only. When the amine N links two
        # IDENTICAL SIMPLE CARBOCYCLES (plain benzene rings, no principal
        # characteristic group), the PIN is the substitutive aniline form
        # ('N-phenylaniline'), NOT the multiplicative '1,1'-iminodibenzene'.
        # This mirrors the guard already applied at the 3-unit sibling
        # _try_multi_atom_bridges (triphenylamine). Scoped strictly to N: the
        # O/S/CH2 bridges (diphenyl ether/sulfide/methylene) keep their
        # multiplicative PINs. Fragments bearing a senior PCG (e.g. the
        # 4,4'-iminodibenzoic acid CO2H rings) make the guard False, so their
        # imino multiplicative name is retained.
        if bridge_type == 'NH' and (
                _all_fragments_are_simple_carbocycles(mol, idx)
                or _all_fragments_are_prefix_only_carbocycles(mol, idx)):
            continue

        # Carbonyl-bridge PIN guard (mirrors the NH guard above): when the
        # C=O links two SIMPLE carbocycles the ketone IS the principal
        # characteristic group, so the PIN is the ketone name
        # (diphenylmethanone / retained benzophenone), NOT the multiplicative
        # '1,1'-carbonyldibenzene'. Units bearing a senior PCG (the
        # 4,4'-carbonyldibenzoic acid rings) keep the multiplicative name —
        # there the bridge C=O is correctly a mere bridge.
        #
        # NOTE (v50 A4, BLOCKED): also makes a =S/=O over two BARE
        # HETEROARENES the thione/ketone parent — 'di(1H-imidazol-1-yl)-
        # methanethione' (the Blue Book), not '1,1'-carbonothioylbis(1H-imidazole)'.
        # Broadening this guard to `_canon_unit_is_bare_ring(canon_a)` correctly
        # DECLINES, but the substitutive builder cannot place an N-attached
        # azolyl on the methanethione parent (it perceives an N-C(=S)-N
        # thiourea) and ABSTAINS — trading a valid non-PIN name for silence.
        # Blocked until that producer gap is closed; the multiplicative name is
        # retained (RIGHT_MOL_NONPIN, round-trips correct). C-attached bare
        # rings (diphenyl / di-C-pyridyl / di-C-imidazolyl) already reach the
        # substitutive ketone/thione namer, so no assembly-layer change helps.
        if (bridge_type in ('carbonyl', 'carbonothioyl')
                and _all_fragments_are_simple_carbocycles(
                    mol, idx, extra_remove=extra_remove)):
            continue

        # Pnictogen-oxoacid / -ester-bridge PIN guard (mirrors the NH / carbonyl
        # guards above;, the Blue Bookff). A mononuclear P/As/Sb bridge
        # bearing a =E chalcogen AND a free -OH/-SH is an arsinic/phosphinic/
        # stibinic ACID class 7c); the same centre bearing an -O-R ester
        # residual is an arsinate/phosphinate/stibinate ESTER (class 9). Both
        # classes are SENIOR to a plain ring and to every ring PCG that is JUNIOR
        # to them (hydroxy class 17, amine class 19, plus detachable halogen/
        # alkyl prefixes), so the acid/ester — not the ring — is the parent and
        # the two rings are `bis(aryl)` substituent prefixes: the PIN is the
        # substitutive di-organyl pnictogen acid/ester
        # (c1ccccc1[As](=O)(O)c1ccccc1 -> 'diphenylarsinic acid', BB L36052;
        # Cc1ccc(cc1)P(=O)(O)c1ccc(C)cc1 -> 'bis(4-methylphenyl)phosphinic acid';
        # c1ccccc1P(=O)(OC)c1ccccc1 -> 'methyl diphenylphosphinate'), NOT the
        # multiplicative "1,1'-(hydroxy/methoxy…oryl)di…". The guard declines the
        # multiplicative name only when NO ring fragment carries a group SENIOR to
        # the bridge acid; a fragment bearing a carboxylic/sulfonic acid (class 7a,
        # senior to a class-7c P/As acid — the 4,4'-(hydroxyarsoryl)dibenzoic acid
        # rings) keeps the multiplicative name, because there the ring acid is the
        # senior parent and the bridge is correctly a mere bridge.
        if (_bridge_is_pnictogen_oxoacid_or_ester(mol, atom)
                and _no_fragment_group_senior_to_pnictogen_oxoacid(
                    mol, idx, extra_remove=extra_remove)):
            continue

        # Resolve the unit name + attachment locant and assemble (shared
        # tail: PG-anchored path, attachment-anchored rename,
        # heterocyclic unit locant — see _resolve_unit_and_assemble).
        result = _resolve_unit_and_assemble(
            mol, [idx], nbr_indices, extra_remove, bridge_name, ring_atoms,
            canon_a,
        )
        if result is not None:
            return result
        # Decline / saturation-prefix parent: let the loop & caller fall through
        continue

    return None


def _unit_name_is_substituted(parent_name: str) -> bool:
    """A multiplied unit name beginning with a locant carries substituent
    prefixes ("2-chlorobenzoic acid", "4-bromobenzene") — the
    bis class. Mid-name suffix locants ("ethan-1-ol") are NOT substituted."""
    import re
    return re.match(r"^(?:\d|N[-,'0-9])", parent_name) is not None


def _ring_unit_direction_safe(frag_canon: str, attach_locant) -> bool:
    """True iff a substituted ring unit's prefix locants cannot disagree with
    the bridge attachment locant: a para (locant 4) attachment on a benzene
    unit is self-mirror, so the fragment namer's direction choice for the
    substituents is consistent with attachment@4 (the BB
    witnesses 4,4'-oxybis(2-chloro/2-bromobenzoic acid)). Any other shape
    fails closed."""
    if attach_locant != 4:
        return False
    frag = Chem.MolFromSmiles(frag_canon)
    if frag is None:
        return False
    ring_info = frag.GetRingInfo()
    if ring_info.NumRings() != 1:
        return False
    ring = ring_info.AtomRings()[0]
    if len(ring) != 6:
        return False
    return all(
        frag.GetAtomWithIdx(a).GetIsAromatic()
        and frag.GetAtomWithIdx(a).GetSymbol() == "C"
        for a in ring
    )


def _fragment_has_anchor_pg(frag) -> bool:
    """Mirror of ``_find_principal_group_atom_in_ring``'s PG semantics on a
    FREE unit fragment: a ring atom bearing a non-ring N/O/S first atom or a
    C=O carbon. Units WITH such an anchor keep the established PG-anchored
    locant path (aniline / phenol / benzoic acid); units WITHOUT one
    (halobenzene, toluene) have no PG for the heuristic to anchor on and must
    be renamed relative to the attachment or declined."""
    ring_atoms = set()
    for r in frag.GetRingInfo().AtomRings():
        ring_atoms.update(r)
    for idx in ring_atoms:
        atom = frag.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring_atoms:
                continue
            sym = nbr.GetSymbol()
            if sym in ("N", "O", "S"):
                return True
            if sym == "C":
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() == idx:
                        continue
                    bond = frag.GetBondBetweenAtoms(nbr.GetIdx(), nbr2.GetIdx())
                    if (bond and bond.GetBondTypeAsDouble() == 2.0
                            and nbr2.GetSymbol() == "O"):
                        return True
    return False


# Substituents admitted on an attachment-anchored benzene unit
#: 1,1'-oxybis(4-bromobenzene)). Anything else fails closed.
_ANCHORED_UNIT_PREFIXES = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo',
                           'I': 'iodo'}


def _attachment_anchored_benzene_unit(
    mol, bridge_idx: int, conn_idx: int,
) -> Optional[Tuple[str, int]]:
    """: rename a PG-free SUBSTITUTED benzene unit with locants
    anchored at the bridge attachment (locant 1): the 4-Br ring of
    Brc1ccc(O...)cc1 -> ('4-bromobenzene', 1), giving
    1,1'-oxybis(4-bromobenzene). The free-fragment namer elides/mis-anchors
    these locants ('bromobenzene'), which produced the structure-dropping
    '1,1'-oxydibromobenzene' class — this is the root fix.

    Scope (fail-closed): a single 6-membered aromatic all-carbon ring whose
    substituents are terminal halogens or methyl; the attachment carbon
    itself is otherwise bare. Returns (unit_name, 1) or None."""
    ring_info = mol.GetRingInfo()
    target = None
    for r in ring_info.AtomRings():
        if conn_idx in r:
            if target is not None:
                return None  # fused system — out of scope
            target = r
    if target is None or len(target) != 6:
        return None
    ring_set = set(target)
    for a in target:
        at = mol.GetAtomWithIdx(a)
        if not at.GetIsAromatic() or at.GetSymbol() != 'C':
            return None
    # Walk the ring cycle starting at the attachment carbon.
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(conn_idx).GetNeighbors()
            if n.GetIdx() in ring_set]
    if len(nbrs) != 2:
        return None
    order = [conn_idx]
    prev, cur = conn_idx, nbrs[0]
    while cur != conn_idx:
        order.append(cur)
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() in ring_set and n.GetIdx() != prev]
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
    if len(order) != 6:
        return None
    subs = {}
    for pos, a in enumerate(order):
        ext = [n for n in mol.GetAtomWithIdx(a).GetNeighbors()
               if n.GetIdx() not in ring_set and n.GetAtomicNum() > 1]
        if pos == 0:
            if any(n.GetIdx() != bridge_idx for n in ext):
                return None  # substituent on the attachment carbon
            continue
        if not ext:
            continue
        if len(ext) != 1 or ext[0].GetIdx() == bridge_idx:
            return None
        n = ext[0]
        sym = n.GetSymbol()
        if (sym in _ANCHORED_UNIT_PREFIXES and n.GetDegree() == 1
                and n.GetFormalCharge() == 0):
            subs[pos] = _ANCHORED_UNIT_PREFIXES[sym]
        elif (sym == 'C' and n.GetFormalCharge() == 0
                and n.GetTotalNumHs() == 3 and n.GetDegree() == 1):
            subs[pos] = 'methyl'
        else:
            return None
    if not subs:
        return None
    # Direction choice: attachment stays 1; lowest substituent locant set,
    # then alphabetically-first prefix at the lower locant.
    fwd = sorted((p + 1, subs[p]) for p in subs)
    rev = sorted((7 - p, subs[p]) for p in subs)
    fkey = ([l for l, _ in fwd], [s for _, s in fwd])
    rkey = ([l for l, _ in rev], [s for _, s in rev])
    placed = fwd if fkey <= rkey else rev
    by_name: Dict[str, List[int]] = {}
    for loc, sname in placed:
        by_name.setdefault(sname, []).append(loc)
    parts = []
    for sname in sorted(by_name):
        locs = sorted(by_name[sname])
        mult = SIMPLE_MULTIPLIERS.get(len(locs), '') if len(locs) > 1 else ''
        parts.append(f"{','.join(str(l) for l in locs)}-{mult}{sname}")
    return ('-'.join(parts) + 'benzene', 1)


def _fragment_attachment_locant(
    mol, remove_set: set, conn_idx: int,
) -> Optional[int]:
    """Attachment locant under the UNIT's own IUPAC numbering:
    4,4'-oxybis(1,3-thiazole) — the whole-molecule cascade cannot number the
    unit, so split the bridge out and query the ring-numbering cascade on the
    FREE fragment). Returns None (fail-closed) when the cascade offers no
    authoritative locant."""
    affected = set()
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a1 in remove_set) != (a2 in remove_set):
            affected.add(a2 if a1 in remove_set else a1)
    emol = RWMol(Chem.RWMol(mol))
    for ridx in sorted(remove_set, reverse=True):
        emol.RemoveAtom(ridx)
    # Aromatic ring-N H repair (Wave-2 completion C) — see _split_at_bridge.
    for orig in affected:
        _s = orig - sum(1 for r in remove_set if r < orig)
        _a = emol.GetAtomWithIdx(_s)
        if _a.GetSymbol() == 'N' and _a.GetIsAromatic():
            _a.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None
    result_mol = emol.GetMol()
    shifted = conn_idx - sum(1 for r in remove_set if r < conn_idx)
    frag_maps: List = []
    frags = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True,
                             fragsMolAtomMapping=frag_maps)
    for fmol, amap in zip(frags, frag_maps):
        amap = list(amap)
        if shifted not in amap:
            continue
        frag_atom = amap.index(shifted)
        try:
            from ..namer import _build_ring_info_for_parent_selection, compute_features
            feats = compute_features(fmol)
            rinfo = _build_ring_info_for_parent_selection(feats)
        except Exception:
            return None
        iupac = (rinfo or {}).get('iupac_locants')
        if not iupac or frag_atom not in iupac:
            return None
        loc = iupac[frag_atom]
        if isinstance(loc, tuple):
            loc = loc[0]
        return loc if isinstance(loc, int) else None
    return None


def _resolve_unit_and_assemble(
    mol, bridge_idxs: List[int], ring_conns: List[int],
    extra_remove: List[int], bridge_name: str, ring_atoms: set,
    canon_unit: str,
) -> Optional[str]:
    """Shared tail for the 2-unit bridge paths: resolve the unit name and
    attachment locant, then assemble. Three unit classes:

      * PG-anchored units (aniline / phenol / benzoic acid, optionally
        substituted) — the established locant path, byte-identical.
      * PG-free SUBSTITUTED units (halobenzene / toluene) — attachment-
        anchored rename (P-15.3.2.4.1) or fail-closed; the free-fragment
        name silently dropped the substituent position before.
      * Pure-ring units whose PARENT name is locant-led (1,3-thiazole) —
        the leading digits are intrinsic ring locants, not substituent
        prefixes; the attachment locant comes from the unit's own
        numbering via the fragment cascade (P-16.5.1.1).

    Returns the assembled name or None (decline — caller continues)."""
    parent_name = _name_parent(canon_unit)
    if parent_name is None:
        return None

    frag = Chem.MolFromSmiles(canon_unit)
    if frag is None:
        return None
    frag_ring_atoms = set()
    for r in frag.GetRingInfo().AtomRings():
        frag_ring_atoms.update(r)
    has_sub = any(a.GetIdx() not in frag_ring_atoms for a in frag.GetAtoms())

    def _bridge_for(conn):
        for n in mol.GetAtomWithIdx(conn).GetNeighbors():
            if n.GetIdx() in bridge_idxs:
                return n.GetIdx()
        return bridge_idxs[0]

    if has_sub and not _fragment_has_anchor_pg(frag):
        anchored = [_attachment_anchored_benzene_unit(mol, _bridge_for(c), c)
                    for c in ring_conns]
        if anchored[0] is None or any(a != anchored[0] for a in anchored[1:]):
            return None
        unit_name, unit_locant = anchored[0]
        return _assemble_multiplicative_name(unit_locant, bridge_name,
                                             unit_name)

    if not has_sub and _unit_name_is_substituted(parent_name):
        remove_set = set(bridge_idxs) | set(extra_remove or [])
        locs = [_fragment_attachment_locant(mol, remove_set, c)
                for c in ring_conns]
        if locs[0] is None or any(l != locs[0] for l in locs[1:]):
            return None
        return _assemble_multiplicative_name(locs[0], bridge_name,
                                             parent_name)

    # Wave-2 completion C: when EVERY attachment is a ring NITROGEN, the
    # PG-distance heuristic mis-anchors (returns 2 for pyridin-2(1H)-one);
    # use the unit's own fragment-cascade numbering or fail closed.
    if ring_conns and all(mol.GetAtomWithIdx(c).GetAtomicNum() == 7
                          for c in ring_conns):
        remove_set = set(bridge_idxs) | set(extra_remove or [])
        _nlocs = [_fragment_attachment_locant(mol, remove_set, c)
                  for c in ring_conns]
        if _nlocs[0] is None or any(l != _nlocs[0] for l in _nlocs[1:]):
            return None
        return _assemble_multiplicative_name(_nlocs[0], bridge_name,
                                             parent_name)

    locants = [_get_bridge_locant(mol, _bridge_for(c), c, ring_atoms)
               for c in ring_conns]
    if any(l != locants[0] for l in locants[1:]):
        return None
    if _unit_name_is_substituted(parent_name) and not \
            _ring_unit_direction_safe(canon_unit, locants[0]):
        return None
    return _assemble_multiplicative_name(locants[0], bridge_name, parent_name)


def _try_two_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find two-atom bridges (e.g., CH2-CH2 ethylene) between identical ring systems."""
    for bond in bonds_of(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        idx1 = a1.GetIdx()
        idx2 = a2.GetIdx()

        # Both bridge atoms must NOT be in rings
        if idx1 in ring_atoms or idx2 in ring_atoms:
            continue

        # Each bridge atom must have exactly one ring neighbor (besides its bridge partner)
        ring_nbrs_1 = [n for n in a1.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx2]
        ring_nbrs_2 = [n for n in a2.GetNeighbors()
                       if n.GetIdx() in ring_atoms and n.GetIdx() != idx1]

        # Each atom in the bridge must connect to exactly one ring atom
        if len(ring_nbrs_1) != 1 or len(ring_nbrs_2) != 1:
            continue

        # Each bridge atom must have exactly 2 heavy neighbors total
        # (one ring atom + one bridge partner)
        heavy_nbrs_1 = [n for n in a1.GetNeighbors() if n.GetAtomicNum() > 1]
        heavy_nbrs_2 = [n for n in a2.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy_nbrs_1) != 2 or len(heavy_nbrs_2) != 2:
            continue

        # Classify the two-atom bridge
        bridge_name = _classify_two_atom_bridge(a1, a2)
        if bridge_name is None:
            continue

        # Split at both bridge atoms
        ring_conn_1 = ring_nbrs_1[0].GetIdx()
        ring_conn_2 = ring_nbrs_2[0].GetIdx()

        fragments = _split_at_two_atom_bridge(
            mol, idx1, idx2, ring_conn_1, ring_conn_2
        )
        if fragments is None:
            continue

        frag_smiles_a, frag_smiles_b = fragments

        # Check if fragments are identical
        canon_a = canon_smiles(frag_smiles_a)
        canon_b = canon_smiles(frag_smiles_b)
        if canon_a != canon_b:
            continue

        # Resolve the unit name + attachment locant and assemble (shared
        # tail — see _resolve_unit_and_assemble).
        result = _resolve_unit_and_assemble(
            mol, [idx1, idx2], [ring_conn_1, ring_conn_2], [], bridge_name,
            ring_atoms, canon_a,
        )
        if result is not None:
            return result
        # Decline / saturation-prefix parent: let the loop & caller fall through
        continue

    return None


def _classify_single_atom_bridge(atom) -> Optional[str]:
    """Classify a single bridge atom into a known type.

    Returns:
        Bridge type key (e.g., 'CH2', 'O', 'NH', 'S') or None.
    """
    sym = atom.GetSymbol()
    total_h = atom.GetTotalNumHs()

    if sym == "C" and total_h == 2:
        return "CH2"
    elif sym == "O" and total_h == 0:
        return "O"
    elif sym == "N" and total_h == 1:
        return "NH"
    elif sym == "S" and total_h == 0:
        return "S"
    return None


_HALO_METHYLENE = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def _residual_subtree_atoms(mol, resid_atom, central_idx: int) -> List[int]:
    """Atom indices of an acid-derived residual ligand on a bridge acyl centre:
    the residual atom itself plus every atom reachable from it WITHOUT crossing
    back through the central P/As/Sb (``central_idx``). For -OH/-SH this is just
    the one atom; for an -O-R ester it is the O plus its whole R subtree, so the
    bridge split strips the ester carbons too."""
    seen: List[int] = []
    stack = [resid_atom.GetIdx()]
    visited = {central_idx}
    while stack:
        i = stack.pop()
        if i in visited:
            continue
        visited.add(i)
        seen.append(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() not in visited:
                stack.append(nb.GetIdx())
    return seen


def _classify_single_atom_bridge_ext(
    mol, atom, ring_nbr_idxs, sub_nbrs
) -> Optional[Tuple[str, str, List[int]]]:
    """Wave-2 completion /.1.2): classify a bridge atom that
    carries a bridge-owned substituent — a carbonyl =O or a halomethylene
    halogen — into (bridge_type_key, bridge_display_name, extra_atoms_to_remove).

    ``sub_nbrs`` are the bridge atom's NON-ring heavy neighbours. Returns None
    (fail-closed) for anything but the two clean cases:
      * C with exactly one terminal =O, 0 H, 2 ring neighbours -> 'carbonyl'
      * C with exactly one terminal single-bonded halogen, 1 H -> '{halo}methylene'
    """
    sym = atom.GetSymbol()
    # a phase: acyl-prefix bridge -E(=Y)(residual)- linking two
    # identical ring parents (each a benzoic acid). The acyl base name comes from the
    # SHARED table (acyl_prefix_for), no longer a hardcoded string, so
    # the family generalises from phosphoryl to arsoryl/stiboryl/thio- siblings while
    # staying byte-identical on the -P(=O)(OH)- case: 'hydroxy' + 'phosphoryl' =
    # '(hydroxyphosphoryl)'. The non-ring neighbours are exactly one =chalcogen (=O
    # -> 'oxo', =S -> 'thio') and one residual single-bonded group concatenated as a
    # front prefix (-OH -> 'hydroxy', -SH -> 'sulfanyl';. skeletal
    # (C/H on E) is 0 -> the -oryl cell.
    if (sym in ("P", "As", "Sb") and atom.GetFormalCharge() == 0
            and atom.GetTotalNumHs() == 0 and len(sub_nbrs) == 2):
        from .functional_replacement import acyl_prefix_for
        from .phosphorus import _acyl_residual_prefix
        _CHALCO_TOKEN = {"O": "oxo", "S": "thio"}
        dbl = None
        chalco_token = resid_prefix = None
        # residual atoms to strip when splitting: the =E' plus the residual
        # ligand AND, for an -O-R ester (a phase 'methoxyphosphoryl'), its
        # whole R subtree.
        resid_extra: List[int] = []
        for s in sub_nbrs:
            if s.GetFormalCharge() != 0:
                return None
            b = mol.GetBondBetweenAtoms(atom.GetIdx(), s.GetIdx())
            if (b.GetBondType() == Chem.BondType.DOUBLE and s.GetDegree() == 1
                    and s.GetTotalNumHs() == 0 and s.GetSymbol() in _CHALCO_TOKEN
                    and chalco_token is None):
                dbl = s
                chalco_token = _CHALCO_TOKEN[s.GetSymbol()]
                continue
            if (b.GetBondType() == Chem.BondType.SINGLE
                    and resid_prefix is None):
                # -OH -> hydroxy, -SH -> sulfanyl, -O-alkyl -> <alkyl>oxy
                #. Fail-closed inside the residual namer.
                rp = _acyl_residual_prefix(mol, s, atom.GetIdx())
                if rp is None:
                    return None
                resid_prefix = rp
                resid_extra = _residual_subtree_atoms(mol, s, atom.GetIdx())
                continue
            return None
        if dbl is None or resid_prefix is None:
            return None
        acyl = acyl_prefix_for(sym, chalco_token, 0)
        if acyl is None:
            return None
        name = f"{resid_prefix}{acyl}"
        return (name, name, [dbl.GetIdx()] + resid_extra)
    if sym != "C" or len(sub_nbrs) != 1:
        return None
    sub = sub_nbrs[0]
    if sub.GetDegree() != 1:
        return None
    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), sub.GetIdx())
    total_h = atom.GetTotalNumHs()
    # Carbonyl bridge: C(=O) linking two rings.
    if (sub.GetSymbol() == "O" and total_h == 0
            and bond.GetBondType() == Chem.BondType.DOUBLE):
        return ("carbonyl", "carbonyl", [sub.GetIdx()])
    # Carbonothioyl bridge: C(=S) linking two rings (Wave-2 completion C,
    #: 1,1'-carbonothioyldi(pyridin-2(1H)-one) BB verbatim).
    if (sub.GetSymbol() == "S" and total_h == 0
            and bond.GetBondType() == Chem.BondType.DOUBLE):
        return ("carbonothioyl", "carbonothioyl", [sub.GetIdx()])
    # Substituted-methylene bridge: C(H)(X) linking two rings.
    if (sub.GetSymbol() in _HALO_METHYLENE and total_h == 1
            and bond.GetBondType() == Chem.BondType.SINGLE):
        name = f"{_HALO_METHYLENE[sub.GetSymbol()]}methylene"
        return ("halomethylene", name, [sub.GetIdx()])
    return None


def _bridge_is_pnictogen_oxoacid(mol, atom) -> bool:
    """True when the single-atom bridge is a mononuclear P/As/Sb OXOACID centre:
    a neutral, H-free pnictogen bearing a terminal =E chalcogen (=O / =S) AND at
    least one free terminal -OH / -SH residual.

    Such a centre is an arsinic / phosphinic / stibinic ACID — a characteristic
    group SENIOR to a ring. When it bridges two units carrying no senior
    PCG of their own, the acid (not the ring) is the parent, so the substitutive
    di-organyl pnictogen acid is the PIN, not the multiplicative name. This is
    the recogniser behind the guard in:func:`_try_single_atom_bridges`.

    Fail-closed (returns False) for the ESTER residual (-O-R, degree 2), for a
    charged / hydridic centre, and for any non-pnictogen atom: those keep their
    established paths.
    """
    if atom.GetSymbol() not in ("P", "As", "Sb"):
        return False
    if atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0:
        return False
    has_dbl_chalco = False
    has_free_acid_xh = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
        if (bond.GetBondType() == Chem.BondType.DOUBLE
                and nbr.GetSymbol() in ("O", "S")
                and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 0
                and nbr.GetFormalCharge() == 0):
            has_dbl_chalco = True
        elif (bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetSymbol() in ("O", "S")
                and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 1
                and nbr.GetFormalCharge() == 0):
            has_free_acid_xh = True
    return has_dbl_chalco and has_free_acid_xh


def _bridge_is_pnictogen_oxoacid_or_ester(mol, atom) -> bool:
    """Like:func:`_bridge_is_pnictogen_oxoacid` but ALSO true for the ESTER
    residual. The single-atom bridge is a mononuclear P/As/Sb centre — neutral,
    H-free — bearing a terminal =E chalcogen (=O / =S) AND either

      * a free terminal -OH / -SH (a phosphinic/arsinic/stibinic ACID,
        class 7c), or
      * an ester -O-R / -S-R residual (an -O-/-S-bonded chalcogen of degree 2
        whose other neighbour is a carbon: a phosphinate/arsinate/stibinate
        ESTER, class 9).

    Both classes are senior to a ring bearing only class-≥17 groups, so the
    substitutive acid/ester is the parent, not the multiplicative bridge name
    , the Blue Bookff). This is the recogniser behind the broadened
    guard in:func:`_try_single_atom_bridges` — the acid-only sibling
    :func:`_bridge_is_pnictogen_oxoacid` stays for callers that must exclude the
    ester. Fail-closed (False) for a charged / hydridic centre and any
    non-pnictogen atom.
    """
    if atom.GetSymbol() not in ("P", "As", "Sb"):
        return False
    if atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0:
        return False
    has_dbl_chalco = False
    has_acid_or_ester = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
        if (bond.GetBondType() == Chem.BondType.DOUBLE
                and nbr.GetSymbol() in ("O", "S")
                and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 0
                and nbr.GetFormalCharge() == 0):
            has_dbl_chalco = True
        elif (bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetSymbol() in ("O", "S")
                and nbr.GetFormalCharge() == 0):
            if nbr.GetDegree() == 1 and nbr.GetTotalNumHs() == 1:
                has_acid_or_ester = True            # free -OH / -SH (acid)
            elif nbr.GetDegree() == 2 and nbr.GetTotalNumHs() == 0:
                others = [x for x in nbr.GetNeighbors()
                          if x.GetIdx() != atom.GetIdx()]
                if len(others) == 1 and others[0].GetSymbol() == "C":
                    has_acid_or_ester = True         # -O-R / -S-R (ester)
    return has_dbl_chalco and has_acid_or_ester


def _fragment_carries_senior_acid(frag) -> bool:
    """True when a split fragment carries an acid group SENIOR to a P/As/Sb
    oxoacid (class 7c) in the order (the Blue Bookff): a carboxylic
    acid or its chalcogen analogue (class 7a, ``-C(=E)-E'H``) or an S/Se/Te
    oxoacid (class 7a, sulfonic/sulfinic/... ``-[S/Se/Te](=O)-OH``).

    Such a fragment is the senior parent, so the multiplicative (ring-parent)
    name is correct and the pnictogen-acid/ester guard must NOT decline it — e.g.
    ``4,4'-(hydroxyarsoryl)dibenzoic acid`` keeps its multiplicative PIN because
    the ring carboxylic acid (7a) outranks the bridge arsinic acid (7c).
    """
    _CHALCO = {8, 16, 34, 52}  # O, S, Se, Te
    for atom in frag.GetAtoms():
        num = atom.GetAtomicNum()
        # carboxylic acid & chalcogen analogues: a C bearing =E and a terminal -E'H
        # S/Se/Te oxoacid: a centre bearing =O and a terminal -OH
        if num == 6 or num in (16, 34, 52):
            has_dbl_e = False
            has_single_eh = False
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() not in _CHALCO:
                    continue
                bond = frag.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    has_dbl_e = True
                elif (bond.GetBondType() == Chem.BondType.SINGLE
                      and nbr.GetDegree() == 1 and nbr.GetTotalNumHs() >= 1):
                    has_single_eh = True
            if has_dbl_e and has_single_eh:
                return True
    return False


def _no_fragment_group_senior_to_pnictogen_oxoacid(
    mol, bridge_idx: int, extra_remove: Optional[List[int]] = None,
) -> bool:
    """True when NO fragment left after removing the P/As/Sb oxoacid (or ester)
    bridge carries a characteristic group SENIOR to that bridge's acid/ester in
    the class order (the Blue Bookff).

    The pnictogen-acid/ester guard declines the multiplicative name (→ acid/ester
    parent) exactly when this is True. A plain substituted arene, a phenol
    (hydroxy, class 17) or an aniline (amine, class 19) fragment carries nothing
    senior → decline; a benzoic-acid / sulfonic-acid fragment (class 7a) does →
    keep multiplicative. Fail-closed (False, i.e. keep multiplicative) if the
    split cannot be sanitised.
    """
    emol = RWMol(Chem.RWMol(mol))
    for ridx in sorted({bridge_idx} | set(extra_remove or []), reverse=True):
        emol.RemoveAtom(ridx)
    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return False
    frag_mols = Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True)
    for frag in frag_mols:
        if _fragment_carries_senior_acid(frag):
            return False
    return True


def _ethene_diyl_bridge_name(mol, atom1, atom2) -> str:
    """`ethene-1,2-diyl`, carrying its configuration when the input defines one.

    -FINAL I9. Without this, (E)- and (Z)-stilbene emitted the SAME name:
    two different compounds, one string. The Blue Book puts the descriptor
    INSIDE the multiplicative bracket, so there is a channel for it --
    `1,1'-[(1*E*)-1-(4-chlorophenyl)ethene-1,2-diyl]dibenzene (PIN, see `
    (`the Blue Book`). The bridge's own locant 1 is the atom cited first,
    hence `(1E)`/`(1Z)`.

    `_CIPCode` is the repo's single source of truth for E/Z
    (`perception/stereo.get_double_bond_stereo`); an UNDEFINED geometry yields the
    bare prefix, which is correct -- the molecule then has no configuration to
    express, and inventing one would be worse than omitting it.
    """
    bond = mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
    if bond is not None and bond.HasProp('_CIPCode'):
        code = bond.GetProp('_CIPCode')
        if code in ('E', 'Z'):
            return f"(1{code})-ethene-1,2-diyl"
    return "ethene-1,2-diyl"


def _classify_two_atom_bridge(atom1, atom2) -> Optional[str]:
    """Classify a two-atom bridge into a known type."""
    sym1 = atom1.GetSymbol()
    sym2 = atom2.GetSymbol()
    h1 = atom1.GetTotalNumHs()
    h2 = atom2.GetTotalNumHs()

    for s1, s2, expected_h1, expected_h2, name in _TWO_ATOM_BRIDGES:
        matched = (
            (sym1 == s1 and sym2 == s2 and h1 == expected_h1 and h2 == expected_h2)
            # Check reverse order
            or (sym1 == s2 and sym2 == s1 and h1 == expected_h2 and h2 == expected_h1)
        )
        if not matched:
            continue
        if name == "ethene-1,2-diyl":
            return _ethene_diyl_bridge_name(atom1.GetOwningMol(), atom1, atom2)
        return name
    return None


def _classify_multi_bridge(atom, ring_nbr_count: int) -> Optional[str]:
    """Classify a multi-valent bridge atom by element + H count + ring neighbor count.

    Args:
        atom: RDKit atom (the bridge candidate).
        ring_nbr_count: Number of ring-atom neighbors.

    Returns:
        Bridge name (e.g., 'nitrilo', 'methylidyne', 'methanetetrayl') or None.
    """
    sym = atom.GetSymbol()
    h = atom.GetTotalNumHs()
    return _MULTI_BRIDGE_NAMES.get((sym, h, ring_nbr_count))


def _all_fragments_are_simple_carbocycles(
    mol, bridge_idx: int, extra_remove: Optional[List[int]] = None,
) -> bool:
    """Check whether removing the bridge atom yields only simple carbocyclic
    fragments (rings with no principal characteristic group / no heteroatoms
    in the parent ring).

    a phase cleanup helper: the substitutive-PIN guard in
    `_try_multi_atom_bridges` invokes this function to distinguish:
      - Plain benzene fragments (e.g., (Ph)3P -> triphenylphosphane PIN)
      - Phenol-like fragments with -OH (e.g., (HOPh)3P -> 4,4',4''-
        phosphinidynetriphenol multiplicative PIN per

    A fragment is "simple carbocyclic" if every atom is a ring carbon
    OR an explicit hydrogen — no heteroatoms (O/N/S/etc.) AND no
    extra-ring substituent atoms. Plain benzene qualifies;
    phenol `Oc1ccccc1` does NOT (has the hydroxyl O).

    Args:
        mol: the original RDKit Mol.
        bridge_idx: atom index of the bridge to remove.

    Returns:
        True if every fragment after bridge removal is a simple
        carbocycle with no principal characteristic group; False
        otherwise (in which case multiplicative may be preferred).
    """
    emol = RWMol(Chem.RWMol(mol))
    for ridx in sorted({bridge_idx} | set(extra_remove or []), reverse=True):
        emol.RemoveAtom(ridx)
    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return False
    frag_mols = Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True)
    for frag in frag_mols:
        for atom in frag.GetAtoms():
            # Heteroatom in or out of ring -> NOT simple carbocyclic
            if atom.GetAtomicNum() != 6 and atom.GetAtomicNum() != 1:
                return False
            # Extra-ring carbon (e.g., methyl substituent) -> still
            # carbocyclic but has a substituent; out of "simple" scope.
            if not atom.IsInRing() and atom.GetAtomicNum() == 6:
                return False
    return True


def _all_fragments_are_prefix_only_carbocycles(
    mol, bridge_idx: int, extra_remove: Optional[List[int]] = None,
) -> bool:
    """True when every fragment left after removing the bridge is a CARBOCYCLIC
    ring system whose only substituents are non-PCG prefixes — every atom is C,
    H, or a halogen (F/Cl/Br/I). Such a ring carries NO principal characteristic
    group (halogen and alkyl are always detachable prefixes), so an amine
    bridging two of them is itself the senior characteristic group and the PIN
    is the substitutive aniline /, NOT the azanediyl
    multiplicative name (e.g. Clc1ccccc1Nc1ccccc1Cl ->
    2-chloro-N-(2-chlorophenyl)aniline, not 1,1'-azanediylbis(2-chlorobenzene)).

    Broader than `_all_fragments_are_simple_carbocycles` (which rejects a mere
    chloro/methyl decoration); used ONLY by the NH-bridge guard so a fragment
    bearing a real O/N/S PCG (phenol, benzoic acid) still keeps its
    multiplicative name.
    """
    emol = RWMol(Chem.RWMol(mol))
    for ridx in sorted({bridge_idx} | set(extra_remove or []), reverse=True):
        emol.RemoveAtom(ridx)
    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return False
    _HALO = {9, 17, 35, 53}
    frag_mols = Chem.GetMolFrags(emol.GetMol(), asMols=True, sanitizeFrags=True)
    for frag in frag_mols:
        if not any(a.IsInRing() for a in frag.GetAtoms()):
            return False  # only ring fragments (the diaryl/dicyclo amine class)
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() not in (1, 6) and atom.GetAtomicNum() not in _HALO:
                return False
    return True


def _try_multi_atom_bridges(mol, ring_atoms: set) -> Optional[str]:
    """Try to find star-topology bridges connecting 3+ identical ring systems.

    A multi-atom bridge is a single non-ring atom connected to 3 or more ring
    atoms, where removing the bridge produces 3+ identical fragments.

    Examples:
        N connecting 3 phenol rings -> nitrilotriphenol
        CH connecting 3 phenol rings -> methylidynetriphenol
        C connecting 4 phenol rings -> methanetetrayltetraphenol

    a phase cleanup substitutive-PIN guard:
        For mononuclear parent hydrides (NH3, PH3, AsH3, SiH4, GeH4, SnH4,
        PbH4, BH3, plus the multivalent CH4 case) substituted with 3+
        IDENTICAL SIMPLE-RING groups, IUPAC / /
         mandate the SUBSTITUTIVE form (e.g., `triphenylphosphane`,
        `triphenylamine`, `triphenylmethane`) as PIN — not the
        multiplicative form (`1,1',1''-phosphinidynetribenzene` etc.).
        See `cleanup-deferred-items.md` -11 for the full bug
        provenance and IUPAC rule citations.
    """
    # Mononuclear-parent-hydride elements where the SUBSTITUTIVE form is
    # PIN over the multiplicative form ONLY when the substituent rings
    # are SIMPLE (no principal characteristic group). Per IUPAC,
    # multiplicative is preferred when the parent ring has a principal
    # characteristic group (e.g., 4,4',4''-nitrilotriphenol uses
    # `triphenol` parent with a `nitrilo` bridge). When the parent ring
    # has NO principal characteristic group (e.g., plain benzene), the
    # substitutive form on the mononuclear parent hydride is PIN per
    # / /: `triphenylamine`,
    # `triphenylphosphane`, `triphenylmethane`.
    _SUBSTITUTIVE_PIN_CENTERS = frozenset({
        'B', 'C', 'N', 'P', 'As', 'Sb', 'Bi',
        'Si', 'Ge', 'Sn', 'Pb', 'S', 'Se', 'Te',
    })

    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if idx in ring_atoms:
            continue  # Bridge atoms are NOT in rings

        neighbors = atom.GetNeighbors()
        heavy_neighbors = [n for n in neighbors if n.GetAtomicNum() > 1]

        # Count ring neighbors
        ring_neighbors = [n for n in heavy_neighbors if n.GetIdx() in ring_atoms]
        ring_nbr_count = len(ring_neighbors)

        # Must have 3+ ring neighbors for multi-bridge
        if ring_nbr_count < 3:
            continue

        # All heavy neighbors must be ring atoms (no non-ring non-H substituents)
        if len(heavy_neighbors) != ring_nbr_count:
            continue

        # a phase cleanup substitutive-PIN guard: when the bridge atom
        # is a mononuclear-parent-hydride element AND the substituent
        # rings are simple carbocycles with no principal characteristic
        # group, defer to substitutive nomenclature. This honors IUPAC
        # / / PIN preference for
        # triphenylamine / triphenylphosphane / triphenylmethane while
        # leaving the multiplicative path active for cases where the
        # parent ring has a principal characteristic group (e.g.,
        # 4,4',4''-nitrilotriphenol per.
        if atom.GetSymbol() in _SUBSTITUTIVE_PIN_CENTERS:
            if _all_fragments_are_simple_carbocycles(mol, idx):
                continue

        # Classify the bridge
        bridge_name = _classify_multi_bridge(atom, ring_nbr_count)
        if bridge_name is None:
            continue

        # Split molecule by removing the bridge atom
        emol = RWMol(Chem.RWMol(mol))
        emol.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(emol)
        except Exception:
            continue

        result_mol = emol.GetMol()
        frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
        unit_count = len(frag_mols)

        if unit_count < 3:
            continue

        # Check all fragments are identical
        canon_smiles_list = [Chem.MolToSmiles(f) for f in frag_mols]
        canon_set = set(canon_smiles(s) for s in canon_smiles_list)
        if len(canon_set) != 1:
            continue  # Non-identical fragments

        canon_parent = canon_set.pop()

        # Name the parent structure
        parent_name = _name_parent(canon_parent)
        if parent_name is None:
            continue

        # Get bridge locant using the first ring neighbor
        first_ring_nbr_idx = ring_neighbors[0].GetIdx()
        locant = _get_bridge_locant(mol, idx, first_ring_nbr_idx, ring_atoms)

        # Assemble multiplicative name with unit_count
        result = _assemble_multiplicative_name(
            locant, bridge_name, parent_name, unit_count=unit_count
        )
        if result is not None:
            return result
        continue

    return None


def _split_at_bridge(
    mol, bridge_idx: int, nbr_indices: List[int],
    extra_remove: Optional[List[int]] = None,
) -> Optional[Tuple[str, str, int, int]]:
    """Split molecule by removing a single bridge atom (plus any ``extra_remove``
    atoms belonging to the bridge, e.g. the carbonyl =O or a halomethylene Cl).

    Returns:
        (frag_smiles_a, frag_smiles_b, conn_atom_in_a, conn_atom_in_b)
        or None if split doesn't produce exactly 2 fragments.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Record neighbors before removal (atom indices will shift!)
    nbr_a, nbr_b = nbr_indices[0], nbr_indices[1]

    # Remove the bridge atom + any bridge-owned substituent atoms. Remove in
    # descending index order so earlier removals don't shift later indices.
    remove_set = {bridge_idx} | set(extra_remove or [])
    for ridx in sorted(remove_set, reverse=True):
        emol.RemoveAtom(ridx)

    # Wave-2 completion C: a connection atom that is an AROMATIC ring N
    # genuinely gains an H in the free unit (pyridin-2(1H)-one). Without the
    # repair, sanitize fails and every ring-N-attached unit silently declined.
    def _shift_pre(orig):
        return orig - sum(1 for r in remove_set if r < orig)
    for conn in nbr_indices:
        _a = emol.GetAtomWithIdx(_shift_pre(conn))
        if _a.GetSymbol() == 'N' and _a.GetIsAromatic():
            _a.SetNumExplicitHs(1)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_indices = Chem.GetMolFrags(result_mol)

    if len(frag_indices) != 2:
        return None

    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)
    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    # Adjust neighbor indices for atom removal: each removed atom with a lower
    # index shifts a surviving atom's index down by one.
    def _shift(orig):
        return orig - sum(1 for r in remove_set if r < orig)
    adj_a = _shift(nbr_a)
    adj_b = _shift(nbr_b)

    return (smi_a, smi_b, adj_a, adj_b)


def _split_at_two_atom_bridge(
    mol, bridge_idx1: int, bridge_idx2: int,
    ring_conn1: int, ring_conn2: int
) -> Optional[Tuple[str, str]]:
    """Split molecule by removing two bridge atoms.

    Returns:
        (frag_smiles_a, frag_smiles_b) or None.
    """
    emol = RWMol(Chem.RWMol(mol))

    # Remove in reverse index order to avoid index shifting issues
    idx_high = max(bridge_idx1, bridge_idx2)
    idx_low = min(bridge_idx1, bridge_idx2)

    emol.RemoveAtom(idx_high)
    emol.RemoveAtom(idx_low)

    try:
        Chem.SanitizeMol(emol)
    except Exception:
        return None

    result_mol = emol.GetMol()
    frag_mols = Chem.GetMolFrags(result_mol, asMols=True, sanitizeFrags=True)

    if len(frag_mols) != 2:
        return None

    smi_a = Chem.MolToSmiles(frag_mols[0])
    smi_b = Chem.MolToSmiles(frag_mols[1])

    return (smi_a, smi_b)


def _name_parent(canon_smiles: str) -> Optional[str]:
    """Backwards-compat wrapper around _resolve_parent_name .

    Returns just the name string (drops the principal-group locant) so
    existing callers in this module see no behavior change. New callers
    should prefer _resolve_parent_name directly to get both the parent
    name and the principal-group locant.

    Source: 154-internal notes (replaces hardcoded _RETAINED_PARENT_NAMES dict).
    """
    result = _resolve_parent_name(canon_smiles)
    return result[0] if result else None


def _get_bridge_locant(
    mol, bridge_idx: int, ring_conn_idx: int, ring_atoms: set
) -> int:
    """a phase.B: query a phase cascade for IUPAC ring locants.

    Replaces the legacy "ring shortest-path from principal group"
    heuristic (which defaulted to 4 / para on failure -- the
    architectural debt this edit clears) with a query against the
    existing handler's IUPAC locant map (a phase cascade --
    heterocycle / fused / VB / spiro / ring-assembly locants).

    Falls back to the legacy shortest-path heuristic
    (_shortest_path_heuristic_locant) only when the cascade returns None
    for the target ring -- preserves current behavior on cascade misses.
    The legacy hard-default-4 branches at lines 501 and 511 are GONE per
    internal notes-B.md

    Args:
        mol: RDKit Mol object.
        bridge_idx: index of the bridge atom.
        ring_conn_idx: index of the ring atom connected to the bridge.
        ring_atoms: set of all ring atom indices.

    Returns:
        Locant number (1-indexed).

    Source: 154-internal notes; rules/parent_selection.py:_build_ring_pos:77;
            namer.py:_build_ring_info_for_parent_selection:372;
            rules/locants.py:compare_locant_sets:96.
    """
    from .parent_selection import _build_ring_pos

    # 1. Find the target ring set containing ring_conn_idx
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if ring_conn_idx in ring:
            target_ring = set(ring)
            break
    if target_ring is None:
        # No ring contains the connection atom -- fall back to legacy
        # heuristic. Defensive guard: should never hit because callers
        # assert ring_conn_idx is in ring_atoms.
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # 2. Build ring_info dict via a phase cascade (fused-hetero, PAH,
    # benzene, heterocycle, spiro, mixed-spiro/fused, VB, ring-assembly).
    handler_ring_info = None
    try:
        from ..namer import _build_ring_info_for_parent_selection, compute_features

        features = compute_features(mol)
        handler_ring_info = _build_ring_info_for_parent_selection(features)
    except Exception:
        # Cascade unavailable for this molecule shape -- fall back gracefully.
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # 3. a phase cascade locants (authoritative IUPAC numbering when available)
    #
    # IMPORTANT: only trust the cascade when handler_ring_info contains an
    # `iupac_locants` dict covering EVERY atom of target_ring. When the
    # cascade has no IUPAC numbering for this ring shape (e.g. functional-
    # group-anchored cyclohexane-1,3-dione where the locants come from the
    # PG positions, not from the ring-system handler), `_build_ring_pos`
    # falls back to atom-sorted positional integers -- which are NOT the
    # IUPAC locants the caller needs. Detect this case via the
    # `iupac_locants` presence + complete-coverage check and fall through
    # to the heuristic, which IS PG-anchored. This preserves the v17
    # heuristic correctness on functional-group-anchored rings while
    # adopting the cascade's authoritative numbering on Hantzsch-Widman /
    # fused / PAH / spiro / VB / ring-assembly handler-controlled rings.
    iupac_locants = (
        handler_ring_info.get("iupac_locants") if handler_ring_info else None
    )
    cascade_covers_ring = (
        iupac_locants
        and all(a in iupac_locants for a in target_ring)
    )
    if not cascade_covers_ring:
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    try:
        ring_pos = _build_ring_pos(target_ring, ring_info=handler_ring_info)
    except Exception:
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )
    locant = ring_pos.get(ring_conn_idx) if ring_pos else None

    if locant is None:
        # Cascade returned no locant for this atom -- fall back to heuristic
        # (preserves current behavior on cascade misses; documented as
        # "remaining heuristic share" in internal notes per Q-B5).
        return _shortest_path_heuristic_locant(
            mol, bridge_idx, ring_conn_idx, ring_atoms
        )

    # Tuple-coercion: tuple locants like (4, 'a') reduce to int (a phase)
    if isinstance(locant, tuple):
        return locant[0]
    return locant


def _shortest_path_heuristic_locant(
    mol, bridge_idx: int, ring_conn_idx: int, ring_atoms: set
) -> int:
    """Legacy ring-shortest-path-from-PG heuristic (fallback path).

    Preserved as the fallback when the a phase cascade returns None for
    the target ring. The legacy "default to 4" branches at the previous
    lines 501 and 511 are GONE -- this function returns the heuristic
    distance + 1 even on edge cases; if the heuristic itself fails (no
    PG atom found), it returns 1 (top-of-ring) instead of silently
    emitting 4 / para.

    Source: 154-internal notes (fallback path); legacy
            multiplicative.py:475-526 pre-Plan-02.
    """
    ring_info = mol.GetRingInfo()
    target_ring = None
    for ring in ring_info.AtomRings():
        if ring_conn_idx in ring:
            target_ring = ring
            break
    if target_ring is None:
        return 1

    pg_atom_idx = _find_principal_group_atom_in_ring(
        mol, target_ring, ring_atoms, bridge_idx
    )
    if pg_atom_idx is None:
        # NOT 4 anymore -- legacy default removed per. Returning 1
        # makes cascade misses observably wrong rather than silently
        # right-ish (since 1 is rarely the correct bridge locant).
        return 1

    ring_list = list(target_ring)
    try:
        pg_pos = ring_list.index(pg_atom_idx)
        conn_pos = ring_list.index(ring_conn_idx)
    except ValueError:
        return 1

    ring_size = len(ring_list)
    dist = abs(conn_pos - pg_pos)
    dist = min(dist, ring_size - dist)
    return dist + 1  # 1-indexed (PG is at position 1)


def _find_principal_group_atom_in_ring(
    mol, ring: tuple, ring_atoms: set, bridge_idx: int
) -> Optional[int]:
    """Find the ring atom bearing the principal characteristic group.

    The principal group atom is a ring atom that has a non-ring, non-bridge
    neighbor that is part of a functional group (N, O attached to C=O, etc.).

    Returns:
        Atom index of the ring atom bearing the principal group, or None.
    """
    ring_set = set(ring)

    for idx in ring:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            # Skip ring atoms and the bridge atom
            if nbr_idx in ring_set or nbr_idx == bridge_idx:
                continue
            if nbr_idx in ring_atoms:
                continue  # Part of another ring

            # This is a non-ring, non-bridge substituent
            # Check if it's a functional group atom
            sym = nbr.GetSymbol()
            if sym in ("N", "O", "S"):
                return idx
            # Check for C=O type groups (carboxylic acid, aldehyde, etc.)
            if sym == "C":
                for nbr2 in nbr.GetNeighbors():
                    if nbr2.GetIdx() != idx:
                        bond = mol.GetBondBetweenAtoms(nbr_idx, nbr2.GetIdx())
                        if (bond and bond.GetBondTypeAsDouble() == 2.0
                                and nbr2.GetSymbol() == "O"):
                            return idx

    return None


def _build_primed_locant_str(locant: int, unit_count: int) -> str:
    """Build a primed locant string for multiplicative naming.

    For each unit i in [0, unit_count), appends i primes to the locant.
    Example: locant=4, unit_count=3 -> "4,4',4''"
    Example: locant=4, unit_count=4 -> "4,4',4'',4'''"

    Args:
        locant: The IUPAC locant number.
        unit_count: Number of identical parent units.

    Returns:
        Comma-separated primed locant string.
    """
    prime = "'"
    parts = []
    for i in range(unit_count):
        parts.append(f"{locant}{prime * i}")
    return ",".join(parts)


def _select_multiplier(parent_name: str, unit_count: int) -> Optional[str]:
    r"""a phase.B: select between SIMPLE_MULTIPLIERS and COMPLEX_MULTIPLIERS.

    Per IUPAC Blue Book:
      - (simple): di / tri / tetra default for clean parent names.
      - (group): bis / tris / tetrakis fire when the parent name
        contains a comma-separated locant pattern that would make di+name
        parse ambiguously (e.g., "1,3-thiazole" -> "bis(1,3-thiazole)"
        instead of the unparseable "di-1,3-thiazole").
      - (ring-assembly): bi / ter / quater for identical rings
        joined by single bonds -- a phase's territory
        (rules.ring_assemblies.ASSEMBLY_MULTIPLIERS); NOT touched here.

    Heuristic (R-154- refinement): trigger COMPLEX_MULTIPLIERS only
    when parent_name contains a comma-separated locant pattern (\d+,\d).
    Bare digits without commas (e.g., "but-2-ene") use SIMPLE_MULTIPLIERS;
    those don't create di+name parse ambiguity.

    Args:
        parent_name: name of the parent fragment.
        unit_count: number of identical parent units (2, 3, 4,...).

    Returns:
        The multiplier prefix string (e.g., "di", "bis"), or None if
        unit_count is unsupported by both tables.

    Source: 154-internal notes; internal notes-B.md;
            IUPAC Blue Book,;
            OPSIN multipliers.xml type="basic" / type="group".
    """
    import re

    from ..assembly.naming_utils import COMPLEX_MULTIPLIERS

    # / trigger (Wave2 T5a): the unit name BEGINS
    # with a locant — a SUBSTITUTED unit ("2-chlorobenzoic acid",
    # "4-bromobenzene") or a locant-led ring name ("1,3-thiazole",
    # "1-azacyclododecane"). 'di' would sit ambiguously against the leading
    # locant, and mandates bis/tris for substituted identical
    # units: 4,4'-oxybis(2-chlorobenzoic acid) (PIN, BB verbatim),
    # 1,1'-oxybis(4-bromobenzene) (PIN,.
    # Mid-name SUFFIX locants keep di/tri + parentheses per:
    # 2,2'-oxydi(ethan-1-ol) (PIN), 4,4'-oxydi(cyclohexane-1-carboxylic
    # acid) (PIN), N1,N1'-methylenedi(benzene-1,4-diamine) (PIN,
    # — mid-name comma locants do NOT trigger bis).
    if re.match(r"^(?:\d|N[-,'0-9])", parent_name):
        return COMPLEX_MULTIPLIERS.get(unit_count)

    # (a) (a phase): a mononuclear Group-14 hydride parent
    # ('silane'/'trimethylsilane'/...) takes bis/tris to avoid the catenated
    # 'disilane'/'trisilane' ambiguity — the count still derives from the
    # occurrence count, not the unit (2 -> bis, 3 -> tris).
    if parent_name in _G14_MONONUCLEAR_HYDRIDE_PARENTS:
        return COMPLEX_MULTIPLIERS.get(unit_count)

    # v50 A4 (a) Note): the Group-15 pnictogen mononuclear hydride
    # parents ('phosphane'/'arsane'/...) take bis/tris to avoid the catenated
    # 'diphosphane'/'triphosphane' ambiguity.
    if parent_name in _PNICTOGEN_MONONUCLEAR_HYDRIDE_PARENTS:
        return COMPLEX_MULTIPLIERS.get(unit_count)

    # default for clean parent names.
    return SIMPLE_MULTIPLIERS.get(unit_count)


# (e): a multiplied FUNCTIONALIZED parent hydride carrying a
# characteristic-group suffix that takes a locant must be enclosed in
# parentheses after the numerical multiplier, with the suffix locant made
# explicit: di(cyclohexane-1-carboxylic acid), di(benzene-1-sulfonic acid).
# (Retained/added-name acids like 'benzoic acid'/'acetic acid' do NOT match
# these systematic suffix stems and keep their existing bare form.)
_P1634_PARENS_SUFFIXES = (
    'carboxylic acid', 'sulfonic acid', 'sulfinic acid',
    'phosphonic acid', 'phosphinic acid', 'carbaldehyde', 'carbonitrile',
)


def _needs_p1634_parens(parent_name: str) -> bool:
    """True iff the parent is a functionalized parent hydride whose systematic
    characteristic-group suffix takes a ring locant (e))."""
    return any(parent_name.endswith(s) for s in _P1634_PARENS_SUFFIXES)


def _bridge_token(bridge_name: str) -> str:
    """Enclose a multiplicative bridge per / /.

    A composite bridge that already contains parentheses (e.g.
    'ethane-1,2-diylbis(oxy)') moves up the nesting order to SQUARE BRACKETS.
    A bridge carrying bare locants/digits ('ethane-1,2-diyl') or a compound
    substituent prefix (a substituted methylene like 'chloromethylene', or a
    substituted acyl like 'hydroxyphosphoryl') is parenthesised.
    A simple locant-free bridge ('oxy', 'methylene', 'sulfanediyl',
    'disulfanediyl', 'nitrilo', 'peroxy') stays BARE -- BB:
    "4,4'-oxydi(cyclohexane-1-carboxylic acid)", "4,4'-oxydi(benzene-1-sulfonic
    acid)" (both cite the bridge bare before the multiplier)."""
    _substituted_methylene = (
        bridge_name != "methylene" and bridge_name.endswith("methylene")
    )
    from .functional_replacement import ACYL_PREFIX_TABLE
    _acyl_bases = set(ACYL_PREFIX_TABLE.values())
    _substituted_acyl = (
        bridge_name not in _acyl_bases
        and any(bridge_name.endswith(base) for base in _acyl_bases)
    )
    if "(" in bridge_name:
        return f"[{bridge_name}]"
    if (any(ch.isdigit() for ch in bridge_name) or _substituted_methylene
            or _substituted_acyl):
        return f"({bridge_name})"
    return bridge_name


def _insert_ring_pg_locant(parent_name: str, locant: int = 1) -> Optional[str]:
    """Insert the explicit ring PG locant before the characteristic-group
    suffix: 'cyclohexanecarboxylic acid' -> 'cyclohexane-1-carboxylic acid';
    'benzenesulfonic acid' -> 'benzene-1-sulfonic acid'. Returns None if the
    suffix cannot be located."""
    for suffix in _P1634_PARENS_SUFFIXES:
        if parent_name.endswith(suffix):
            stem = parent_name[: -len(suffix)]
            # Already carries a locant right before the suffix -> leave as-is.
            if stem.endswith('-'):
                return parent_name
            return f"{stem}-{locant}-{suffix}"
    return None


# (c): skeletal replacement ('a') prefixes. A multiplied unit whose
# name BEGINS with one of these + 'cyclo...' (a single-heteroatom replacement
# monocycle, e.g. 'azacyclododecane') must take 'bis'/'tris' with enclosing
# marks and carry its heteroatom locant '1-', because 'di<a>...' would read as a
# replacement-atom count: the Blue Book "bis(azacyclododecane)... whereas the name
# diazacyclododecane describes a cyclododecane ring with two nitrogen atoms".
_A_REPLACEMENT_PREFIXES = (
    "oxa", "thia", "selena", "tellura", "aza", "phospha", "arsa", "stiba",
    "bisma", "sila", "germa", "stanna", "plumba", "bora", "alumina", "galla",
    "inda", "thalla",
)


def _normalize_replacement_monocycle_unit(parent_name: str) -> str:
    """(c): give a single-'a'-prefix replacement monocycle unit its
    heteroatom locant ('azacyclododecane' -> '1-azacyclododecane').

    A unit name that starts with exactly one skeletal replacement prefix
    immediately followed by 'cyclo' and carries NO locant of its own denotes a
    single heteroatom, which IUPAC numbers as position 1 (lowest locant).
    Prefixing '1-' makes the leading-digit test in ``_select_multiplier`` /
    ``_assemble_multiplicative_name`` route it to 'bis'/'tris' + parentheses.
    Any other name (already locanted, multiplied 'di<a>...', non-cyclo) is
    returned unchanged.
    """
    import re

    if re.match(r"^\d", parent_name):
        return parent_name  # already carries a leading locant
    for pref in _A_REPLACEMENT_PREFIXES:
        if parent_name.startswith(pref + "cyclo"):
            return f"1-{parent_name}"
    return parent_name


def _assemble_multiplicative_name(
    locant: Optional[int], bridge_name: str, parent_name: str, unit_count: int = 2
) -> Optional[str]:
    """Assemble the final multiplicative name.

    Format: [locants]-[bridge][multiplier][parent]
    Examples:
        4,4'-methylenedianiline (2 units;
        4,4',4''-nitrilotriphenol (3 units;
        4,4',4'',4'''-methanetetrayltetraphenol (4 units;
        4,4'-bis(1,3-thiazole)... group multiplier)

    For acid names with spaces (like "benzoic acid"), the multiplier prefix
    goes before the base name: 4,4'-oxydibenzoic acid

    If the parent name starts with a saturation/modification prefix
    (e.g., "tetrahydro", "dihydro"), returns None to signal that
    multiplicative naming would produce an unparseable concatenation
    (e.g., "ditetrahydropyran") and the caller should fall through
    to substitutive naming.

    Args:
        locant: IUPAC locant of the bridge attachment point.
        bridge_name: Name of the bridge group (e.g., "methylene", "oxy").
        parent_name: IUPAC name of one parent unit (e.g., "aniline", "benzoic acid").
        unit_count: Number of identical parent units (default 2 for backward compat).

    Returns:
        Complete multiplicative name, or None if the parent name has a
        saturation prefix that would produce unparseable multiplier+prefix
        output OR the multiplier table has no entry for unit_count.
    """
    # (c): a single-'a'-prefix replacement monocycle unit
    # ('azacyclododecane') gets its heteroatom locant so the leading-digit
    # tests below route it to 'bis'/'tris' + parentheses ('1-azacyclododecane').
    parent_name = _normalize_replacement_monocycle_unit(parent_name)

    # Check for saturation/modification prefixes in the parent name
    # that would produce unparseable "di+prefix" concatenation
    parent_lower = parent_name.lower()
    if any(parent_lower.startswith(p) for p in SATURATION_PREFIXES):
        return None

    # a phase.B: / three-way split
    # belongs to ring_assemblies.py and is NOT touched here).
    multiplier = _select_multiplier(parent_name, unit_count)
    if not multiplier:
        return None

    # Build primed locant string: e.g., "4,4'" for 2 units, "4,4',4''" for 3.
    # locant=None signals a MONONUCLEAR parent unit (methanol): the locant is
    # meaningless and omitted per (BB witness:
    # "[oxydi(pyridazine-4,3,5-triyl)]tetramethanol" — no unit locants).
    locant_prefix = (
        "" if locant is None
        else _build_primed_locant_str(locant, unit_count) + "-"
    )

    # (e): a multiplied functionalized parent hydride whose systematic
    # characteristic-group suffix takes a ring locant is enclosed in parentheses
    # after the multiplier, with the suffix locant made explicit, and the bridge
    # is parenthesised too:
    # 1,1'-(disulfanediyl)di(cyclohexane-1-carboxylic acid)
    # (This precedes the generic space/digit assembly below; it fires only when
    # the parent carries no substituent-prefix locant of its own, i.e. the PG
    # sits at ring position 1 — the multiplied attachment carbon.)
    if (_needs_p1634_parens(parent_name)
            and not _unit_name_is_substituted(parent_name)
            and "(" not in bridge_name):
        unit_with_locant = _insert_ring_pg_locant(parent_name, 1)
        if unit_with_locant is not None:
            #: a simple unsubstituted bridge (oxy, methylene) is
            # cited BARE before the multiplier ("4,4'-oxydi(...)"); it is
            # enclosed only if it carries locants/substitution. Use the same
            # bridge-enclosure test as the generic assembly below.
            _bridge = _bridge_token(bridge_name)
            return (f"{locant_prefix}{_bridge}"
                    f"{multiplier}({unit_with_locant})")

    # Enclosure /: a BRIDGE name that itself contains
    # parentheses (composite bridge "ethane-1,2-diylbis(oxy)") moves up the
    # nesting order to SQUARE BRACKETS: "2,2'-[ethane-1,2-diylbis(oxy)]di..."
    # (BB verbatim). A bridge carrying bare locants
    # ("ethane-1,2-diyl") is parenthesised -> "4,4'-(ethane-1,2-diyl)di...".
    # Locant-free bridges ("oxy", "sulfanediyl", "azanediyl", "nitrilo",
    # "methylene", "peroxy", "disulfanediyl") stay bare.
    # A SUBSTITUTED simple bridge (a compound substituent name like
    # "chloromethylene" = chloro + methylene) is also parenthesised even
    # though it carries no locant/digit — "4,4'-(chloromethylene)
    # diphenol" (Wave-2 completion).
    # a phase: a composite substituted-ACYL bridge
    # ('hydroxyphosphoryl', 'hydroxyarsoryl', 'sulfanylphosphonoyl',...) is a
    # compound prefix -> parenthesised (like the substituted-methylene case); a
    # BARE acyl base ('phosphoryl'/'arsoryl'/...) stays unenclosed. Generalised
    # from the old phosphoryl-only special case over the shared acyl
    # table, so every element/chalcogen sibling is enclosed consistently. All of
    # this enclosure logic now lives in the shared _bridge_token helper, so the
    # (e) branch above and this generic assembly stay in lock-step.
    bridge_token = _bridge_token(bridge_name)

    # Enclosure /: a parent name carrying locants (e.g.
    # "ethan-1-ol", "2-chlorobenzoic acid") is enclosed in parentheses so the
    # parent boundary is unambiguous -> "...di(ethan-1-ol)" /
    # "...bis(2-chlorobenzoic acid)". Names beginning with 'dec' are also
    # enclosed per (d) ("di(decanoic acid)" vs "didecanoic acid").
    # Locant-free parents ("aniline", "acetic acid", "benzoic acid") are
    # unaffected (no parentheses). Whether the multiplier is di/tri
    # or bis/tris substituted units) is decided by
    # _select_multiplier on the leading-locant test.
    if any(ch.isdigit() for ch in parent_name) or parent_lower.startswith("dec"):
        return f"{locant_prefix}{bridge_token}{multiplier}({parent_name})"

    # Handle names with spaces (e.g., "benzoic acid" -> "tribenzoic acid")
    if " " in parent_name:
        parts = parent_name.split(" ", 1)
        base = parts[0]  # "benzoic"
        suffix = parts[1]  # "acid"
        return f"{locant_prefix}{bridge_token}{multiplier}{base} {suffix}"
    else:
        # Simple name: "aniline" -> "dianiline"
        return f"{locant_prefix}{bridge_token}{multiplier}{parent_name}"
