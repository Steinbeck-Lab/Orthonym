"""Centralized substituent fragment naming module.

Provides name_substituent_fragment() -- the single entry point for naming
any substituent (linear, branched, functionalized, or ring-containing) from
its atom indices within a parent molecule.

Architecture:
  1. Fast path: linear alkyl substituents use get_alkyl_name() directly.
  2. Retained names: isopropyl, tert-butyl, sec-butyl, isobutyl, phenyl.
  3. Recursive path: extract fragment SMILES, name via name_fragment_recursively(),
     convert parent name to prefix form via parent_to_prefix().

The function returns RAW prefix names WITHOUT enclosing marks (parentheses/brackets).
The caller (format_substituent_prefix in naming_utils.py) handles wrapping based on
is_complex_substituent() and multiplier logic.

References:
    IUPAC 2013 P-31.1.3 (substituent prefix naming)
    IUPAC 2013 P-14.5.2 (compound substituent enclosing marks)
"""

import re
import logging
from collections import deque
from typing import List, Optional, Set

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from .naming_utils import get_alkyl_name
from .fragment_naming import name_fragment_recursively

logger = logging.getLogger(__name__)


# ============================================================================
# Linear Alkyl Detection (Fast Path Guard)
# ============================================================================


def _is_linear_alkyl(mol, sub_atoms: List[int]) -> bool:
    """Check if a substituent is a straight-chain pure saturated alkyl group.

    Returns True if ALL atoms in sub_atoms are carbon, no carbon has
    more than 2 carbon neighbors within sub_atoms (i.e., no branching),
    AND all bonds between fragment atoms are single bonds.

    Chains with C=C or C#C bonds are NOT linear alkyl — they are
    alkenyl/alkynyl substituents that need the unsaturated naming path
    to produce correct prefix forms (ethenyl, prop-2-en-1-yl, ethynyl).

    This is the fast-path guard: if True, use get_alkyl_name(carbon_count)
    directly, avoiding unnecessary recursion.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if the substituent is a linear (unbranched) saturated pure-carbon chain.
    """
    if not sub_atoms:
        return False

    sub_set = set(sub_atoms)

    # Reject if any atom is in a ring (cyclic substituents are not linear alkyl)
    ring_info = mol.GetRingInfo()
    for idx in sub_atoms:
        if ring_info.NumAtomRings(idx) > 0:
            return False

    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Any non-carbon atom means it's not pure alkyl
        if atom.GetSymbol() != 'C':
            return False

        # Count carbon neighbors within the substituent
        c_nbrs_in_sub = sum(
            1 for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
        )

        # A linear chain carbon has at most 2 C neighbors within the fragment
        # (or 1 at the terminal). More than 2 means branching.
        if c_nbrs_in_sub > 2:
            return False

        # Reject if any intra-fragment bond is unsaturated (C=C or C#C).
        # These must go through the unsaturated naming path to produce
        # correct alkenyl/alkynyl names.
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    return False

    return True


# ============================================================================
# Unsaturated Linear Chain Naming (IUPAC P-31.1.3)
# ============================================================================


def _name_unsaturated_chain(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Name an unsaturated linear chain substituent per IUPAC P-31.1.3.

    Constructs the correct alkenyl/alkynyl prefix name by:
    1. Verifying the fragment is an unbranched all-carbon chain with unsaturation
    2. Tracing the chain from the attachment point
    3. Trying both numbering directions
    4. Choosing the direction that gives lowest locant to the free-valence
       (attachment point) first, then lowest locants to unsaturation

    Examples:
        CH2=CH-  (attached at CH=)  -> ethenyl
        CH2=CH-CH2- (attached at CH2) -> prop-2-en-1-yl
        CH3-CH=CH- (attached at CH=)  -> prop-1-en-1-yl
        CH2=C(CH3)- (attached at C=) -> prop-1-en-2-yl
        HC#C- (attached at C)         -> ethynyl

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: Index of the first atom of the substituent (bonded to parent).
        parent_set: Set of atom indices in the parent chain/ring.

    Returns:
        Prefix-form name (e.g., "ethenyl", "prop-2-en-1-yl") or None if this
        is not an unsaturated linear chain.
    """
    if len(sub_atoms) < 2:
        return None

    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    # Verify: all carbon, no rings, no branching, has unsaturation
    has_unsat = False
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        if ring_info.NumAtomRings(idx) > 0:
            return None
        c_nbrs = sum(
            1 for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
        )
        if c_nbrs > 2:
            return None  # branched
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    has_unsat = True

    if not has_unsat:
        return None

    carbon_count = len(sub_atoms)

    # Trace the full chain from one terminal to the other.
    # Find a terminal atom (exactly 1 neighbor in fragment).
    terminal = None
    for idx in sub_atoms:
        frag_nbrs = [
            nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in sub_set
        ]
        if len(frag_nbrs) == 1:
            terminal = idx
            break

    if terminal is None:
        return None  # no terminal found (cycle?), shouldn't happen

    ordered = [terminal]
    visited = {terminal}
    current = terminal
    while len(ordered) < carbon_count:
        atom = mol.GetAtomWithIdx(current)
        next_atom = None
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in sub_set and ni not in visited:
                next_atom = ni
                break
        if next_atom is None:
            break
        ordered.append(next_atom)
        visited.add(next_atom)
        current = next_atom

    if len(ordered) != carbon_count:
        return None

    # Try both numbering directions and pick the best per IUPAC rules.
    fwd = ordered
    rev = list(reversed(ordered))

    best_name = None
    best_key = None

    for chain in [fwd, rev]:
        try:
            a_locant = chain.index(attach_idx) + 1
        except ValueError:
            continue

        # Collect double/triple bond locants (lower-numbered atom in each bond)
        double_locs = []
        triple_locs = []
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond is None:
                continue
            bt = bond.GetBondTypeAsDouble()
            if bt == 2.0:
                double_locs.append(i + 1)
            elif bt == 3.0:
                triple_locs.append(i + 1)

        all_unsat = sorted(double_locs + triple_locs)
        # IUPAC P-31.1.3.4: lowest to free valence first, then to unsaturation
        key = (a_locant, all_unsat)

        if best_key is None or key < best_key:
            best_key = key
            # Build the name for this direction
            best_name = _build_alkenyl_name(
                carbon_count, a_locant, double_locs, triple_locs
            )

    return best_name


def _build_alkenyl_name(
    carbon_count: int,
    attach_locant: int,
    double_locants: List[int],
    triple_locants: List[int],
) -> str:
    """Build an alkenyl/alkynyl prefix name from chain data.

    Args:
        carbon_count: Number of carbons in the substituent chain.
        attach_locant: 1-indexed position of the free valence (attachment point).
        double_locants: Sorted list of double bond locants.
        triple_locants: Sorted list of triple bond locants.

    Returns:
        Prefix-form name, e.g., "ethenyl", "prop-2-en-1-yl", "ethynyl".
    """
    from ..data.chain_names import get_chain_prefix

    stem = get_chain_prefix(carbon_count)
    double_locants = sorted(double_locants)
    triple_locants = sorted(triple_locants)

    MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta"}

    # For 2-carbon chains: no locants needed for unsaturation
    if carbon_count == 2:
        if double_locants:
            return f"{stem}enyl"
        if triple_locants:
            return f"{stem}ynyl"
        return f"{stem}yl"

    # For longer chains: build with locants using structured assembly.
    # Each segment is (locant_str, multiplier, bond_suffix).
    # The 'a' euphonic connector is added when multiple bonds have multiplied
    # locants (diene, diyne) per IUPAC P-31.1.3.4.
    num_double = len(double_locants)
    num_triple = len(triple_locants)
    needs_a = (num_double > 1) or (num_triple > 1 and num_double == 0)

    segments = []
    if double_locants:
        loc_str = ",".join(str(l) for l in double_locants)
        mult = MULT.get(num_double, str(num_double)) if num_double > 1 else ""
        segments.append((loc_str, mult, "en"))

    if triple_locants:
        loc_str = ",".join(str(l) for l in triple_locants)
        mult = MULT.get(num_triple, str(num_triple)) if num_triple > 1 else ""
        segments.append((loc_str, mult, "yn"))

    # Assemble infix: "a" (if needed) then "-locants-[mult]bond" per segment
    infix = ""
    if needs_a:
        infix = "a"
    for loc_str, mult, bond in segments:
        infix += f"-{loc_str}-{mult}{bond}"

    if not needs_a and infix:
        # infix already starts with "-" from the first segment
        pass
    elif not infix:
        infix = ""

    name = f"{stem}{infix}-{attach_locant}-yl"

    return name


# ============================================================================
# Fragment SMILES Extraction
# ============================================================================


def _extract_fragment_smiles(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: Set[int],
) -> Optional[str]:
    """Extract a valid SMILES for a substituent fragment.

    Uses RDKit's MolFragmentToSmiles to extract the substituent as a
    standalone molecule. The attachment point atom gets its valence
    satisfied implicitly (MolFragmentToSmiles caps the cut bond with H).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: Index of the first atom of the substituent (bonded to parent).
        parent_chain: Set of atom indices in the parent chain.

    Returns:
        SMILES string of the fragment, or None if extraction fails.
    """
    if not sub_atoms:
        return None

    try:
        frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms)
        if not frag_smi:
            return None

        # Validate: the fragment SMILES must parse back to a valid molecule
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            return None

        return frag_smi
    except Exception as e:
        logger.debug(
            "DROP-15 substituent_skip: reason=extract_exception atom_count=%d error=%s",
            len(sub_atoms), e,
        )
        return None


# ============================================================================
# Parent-to-Prefix Conversion (IUPAC P-31.1.3)
# ============================================================================


def parent_to_prefix(parent_name: str, chain_length: int, attach_locant: int = 1) -> str:
    """Convert a parent compound name to substituent prefix form.

    Per IUPAC P-31.1.3, the parent compound name is transformed into
    a substituent prefix by:
    1. Removing the suffix (e.g., -oic acid, -ol, -one)
    2. Converting the suffix to its prefix form (e.g., -ol -> hydroxy)
    3. Adding the prefix at the correct locant
    4. Appending -yl at the free-valence position

    Per IUPAC P-46.2, the point of free valency receives the lowest
    possible locant consistent with any fixed numbering of the parent
    hydride. For chain-derived substituents with no fixed numbering,
    the chain is oriented so the attachment point is at locant 1.

    Args:
        parent_name: Parent compound IUPAC name (e.g., "propan-2-ol").
        chain_length: Number of carbons in the substituent chain.
        attach_locant: Locant of the free-valence carbon (default 1).

    Returns:
        Prefix-form name (e.g., "2-hydroxypropyl"). Returns raw name
        WITHOUT enclosing marks.
    """
    if not parent_name:
        return ""

    name = parent_name.strip()

    # ---- Carboxylic acids: -oic acid / -anoic acid ----
    # e.g., "butanoic acid" -> "3-carboxypropyl"
    m_oic = re.match(r'^(.+?)(?:an)?oic acid$', name)
    if m_oic:
        stem = m_oic.group(1)
        # Carboxy goes on the terminal carbon (chain_length for original chain)
        # The stem is shortened by one carbon (the COOH carbon becomes "carboxy")
        carboxy_locant = chain_length
        # Build: {locant}-carboxy{shortened_stem}yl
        # For butanoic acid (4C): carboxy at C4, stem = prop (3C), result = 3-carboxypropyl
        # Actually: butanoic acid -> carboxy replaces the acid, chain becomes 3C (propyl)
        # The parent chain of the substituent WITHOUT the carboxylic acid is chain_length - 1
        shortened_length = chain_length - 1
        if shortened_length >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened_length)
            # carboxy locant is the shortened chain length (farthest from attachment)
            return f"{shortened_length}-carboxy{short_stem}yl"
        else:
            return "carboxy"

    # ---- Locanted alcohol: -N-ol ----
    # Matches saturated (-an-N-ol), unsaturated (-en-N-ol, -yn-N-ol),
    # and bare (-N-ol) patterns.
    # e.g., "propan-2-ol" -> "2-hydroxypropyl"
    # e.g., "3-methylbut-2-en-1-ol" -> "1-hydroxy-3-methylbut-2-en-1-yl"
    m_ol = re.search(r'-(\d+)-ol$', name)
    if m_ol:
        locant = m_ol.group(1)
        # Get the stem (everything before "-N-ol")
        stem = name[:m_ol.start()]
        # Convert saturated suffix to yl: -an -> -yl (propan -> propyl)
        # Keep unsaturation: -en stays as -en, -yn stays as -yn
        if stem.endswith('an'):
            stem = stem[:-2]  # propan -> prop
        # Insert hyphen between hydroxy and stem when stem starts with a
        # digit (e.g., "3-methylbut-2-en") to avoid "hydroxy3-methylbut"
        sep = "-" if stem and stem[0].isdigit() else ""
        return f"{locant}-hydroxy{sep}{stem}yl"

    # ---- Unlocanted alcohol: ends in -ol (e.g., "ethanol", "methanol") ----
    if name.endswith('ol') and not name.endswith('diol'):
        # Strip -ol, check for -an prefix
        base = name[:-2]  # remove "ol"
        if base.endswith('an'):
            stem = base[:-2]  # remove "an"
        elif base.endswith('a'):
            # e.g., "methan" case (methanol -> methan -> meth)
            stem = base[:-1]
        else:
            stem = base
        # Alcohol: add hydroxy prefix
        return f"hydroxy{stem}yl"

    # ---- Locanted ketone: -an-N-one ----
    # e.g., "butan-2-one" -> "2-oxobutyl"
    m_one = re.search(r'an?-(\d+)-one$', name)
    if m_one:
        locant = m_one.group(1)
        stem_end = m_one.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-oxo{stem}yl"

    # ---- Unlocanted ketone: ends in -one ----
    if name.endswith('one') and not name.endswith('none'):
        base = name[:-3]  # remove "one"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"oxo{stem}yl"

    # ---- Locanted amine: -an-N-amine ----
    # e.g., "propan-1-amine" -> "1-aminopropyl"
    m_amine = re.search(r'an?-(\d+)-amine$', name)
    if m_amine:
        locant = m_amine.group(1)
        stem_end = m_amine.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-amino{stem}yl"

    # ---- Unlocanted amine: ends in -amine / -anamine ----
    if name.endswith('amine'):
        base = name[:-5]  # remove "amine"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"amino{stem}yl"

    # ---- Aldehyde: ends in -al or -anal ----
    # e.g., "propanal" -> "1-oxopropyl"
    if name.endswith('al') and not name.endswith('nal') or name.endswith('anal'):
        if name.endswith('anal'):
            stem = name[:-4]  # remove "anal"
        elif name.endswith('al'):
            base = name[:-2]  # remove "al"
            if base.endswith('an'):
                stem = base[:-2]
            elif base.endswith('a'):
                stem = base[:-1]
            else:
                stem = base
        # Aldehyde: oxo at C-1
        return f"1-oxo{stem}yl"

    # ---- Ester: -oate suffix ---- (IUPAC P-65.6.3)
    # e.g., "propanoate" -> carboxy prefix form
    # When ester is not the principal group, the acid portion uses "carboxy"
    m_oate = re.search(r'(?:an)?oate$', name)
    if m_oate:
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-carboxy{short_stem}yl"
        return "carboxy"

    # ---- Amide: -carboxamide (most specific first) ---- (IUPAC P-66.1.1.4)
    # e.g., "benzcarboxamide" -> "carbamoyl" prefix
    if name.endswith('carboxamide'):
        stem = name[:-11]  # remove "carboxamide"
        if stem:
            return f"carbamoyl{stem}yl"
        return "carbamoyl"

    # ---- Amide: general -amide suffix ---- (IUPAC P-66.1.1.4)
    # e.g., "propanamide" -> "2-carbamoylethyl", "acetamide" -> "carbamoylmethyl"
    m_amide = re.search(r'(?:an)?amide$', name)
    if m_amide:
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-carbamoyl{short_stem}yl"
        return "carbamoyl"

    # ---- Nitrile: -carbonitrile (most specific first) ---- (IUPAC P-66.1.4.1)
    # e.g., "benzonitrile" -> "cyanophenyl" (cyano + stem + yl)
    if name.endswith('carbonitrile'):
        stem = name[:-12]  # remove "carbonitrile"
        if stem:
            return f"cyano{stem}yl"
        return "cyano"

    # ---- Nitrile: general -nitrile suffix ---- (IUPAC P-66.1.4.1)
    # e.g., "propanenitrile" -> "2-cyanoethyl", "acetonitrile" -> "cyanomethyl"
    if name.endswith('nitrile') and not name.endswith('carbonitrile'):
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-cyano{short_stem}yl"
        return "cyano"

    # ---- Simple acid names: convert to acyl prefix ---- (IUPAC P-65.1.7)
    # Only handle simple retained acid names that appear as substituents.
    # e.g., "formic acid" -> "formyl", "acetic acid" -> "acetyl"
    _ACID_TO_ACYL_PREFIX = {
        'formic acid': 'formyl',
        'acetic acid': 'acetyl',
        'propionic acid': 'propionyl',
        'butyric acid': 'butyryl',
        'benzoic acid': 'benzoyl',
    }
    if name in _ACID_TO_ACYL_PREFIX:
        return _ACID_TO_ACYL_PREFIX[name]

    # ---- Cyclic names: cyclo...ane -> cyclo...yl ---- (IUPAC P-31.1.3)
    # e.g., "cyclohexane" -> "cyclohexyl", "cyclopentane" -> "cyclopentyl"
    if 'cyclo' in name and name.endswith('ane'):
        stem = name[:-3]  # remove "ane"
        return f"{stem}yl"

    # ---- Heterocyclic -ane ending ---- (IUPAC P-31.1.3)
    # Heterocyclic ring names (oxirane, thiirane, oxetane, thietane, oxolane,
    # oxane, thiane, etc.) replace -e with -yl, NOT strip -ane and add -yl.
    # e.g., "oxirane" -> "oxiranyl" (not "oxiryl"), "oxane" -> "oxanyl" (not "oxyl")
    _HETERO_ANE_RINGS = {
        'oxirane', 'thiirane', 'oxetane', 'thietane', 'oxolane',
        'oxane', 'thiane', 'thiolane',
        'dioxane', 'dioxolane', 'dithiane', 'dithiolane', 'trioxane',
        'borolane', 'boroxane', 'silolane',
    }
    base_name = name.split('-')[-1] if '-' in name else name
    if base_name in _HETERO_ANE_RINGS:
        return name[:-1] + "yl"  # replace -e with -yl

    # ---- Alkane: -ane or -e ending ----
    # e.g., "propane" -> "propyl", "2-methylpropane" -> "2-methylpropyl"
    if name.endswith('ane'):
        stem = name[:-3]  # remove "ane"
        return f"{stem}yl"

    # ---- Heterocyclic -ine ending ---- (IUPAC P-31.1.3)
    # e.g., "pyridine" -> "pyridinyl", "piperidine" -> "piperidinyl"
    # Note: -ine must come BEFORE the generic -e fallback
    if name.endswith('ine'):
        return name[:-1] + "yl"  # pyridine -> pyridinyl

    # ---- Fallback: strip terminal -e if present, add -yl ----
    if name.endswith('e'):
        return name[:-1] + "yl"

    # If name already ends in -yl, return as-is
    if name.endswith('yl'):
        return name

    return name + "yl"


# ============================================================================
# Retained Substituent Names
# ============================================================================


def _check_retained_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
) -> Optional[str]:
    """Check for common retained substituent names.

    Detects isopropyl, tert-butyl, sec-butyl, isobutyl, neopentyl,
    and phenyl by analyzing the branching pattern at the attachment point.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.
        attach_idx: First atom of the substituent (bonded to parent chain).

    Returns:
        Retained name string, or None if no retained name applies.
    """
    frag_set = set(sub_atoms)

    # --- Phenyl detection ---
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(frag_set) and len(ring) == 6:
            if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                   mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                # Has a benzene ring
                # Phase 125 fix: count ALL non-ring heavy atoms, not
                # just carbons.  Heteroatom substituents (Cl, OH, NH2,
                # F, Br, NO2) on the ring were invisible to the old
                # carbon-only check, causing "phenyl" to be returned
                # for substituted rings like 4-chlorophenyl.
                non_ring_heavy = sum(
                    1 for i in sub_atoms
                    if i not in ring_set
                    and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                )
                if non_ring_heavy == 0:
                    return "phenyl"
                elif non_ring_heavy == 1:
                    # Benzyl only if the attachment point is a non-ring
                    # carbon (CH2 bridging parent to ring). If the
                    # attachment point is a ring carbon, this is a
                    # substituted phenyl (e.g., 4-methylphenyl or
                    # 4-chlorophenyl), not benzyl.
                    non_ring_atoms = [
                        i for i in sub_atoms
                        if i not in ring_set
                        and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                    ]
                    if (len(non_ring_atoms) == 1
                            and mol.GetAtomWithIdx(non_ring_atoms[0]).GetSymbol() == 'C'
                            and attach_idx not in ring_set):
                        return "benzyl"

    # --- Cycloalkyl detection (IUPAC P-31.1.3.4) ---
    # Saturated carbocyclic rings used as substituents: cyclopropyl, cyclobutyl,
    # cyclopentyl, cyclohexyl, cycloheptyl, cyclooctyl.
    # Must be all-carbon, all-single-bond, no extra non-ring heavy atoms.
    _CYCLO_RETAINED = {
        3: 'cyclopropyl', 4: 'cyclobutyl', 5: 'cyclopentyl',
        6: 'cyclohexyl', 7: 'cycloheptyl', 8: 'cyclooctyl',
    }
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if not ring_set.issubset(frag_set):
            continue
        ring_size = len(ring)
        if ring_size not in _CYCLO_RETAINED:
            continue
        # All ring atoms must be carbon
        if not all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
            continue
        # All bonds in ring must be single
        all_single = True
        for i in range(ring_size):
            bond = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % ring_size])
            if bond and bond.GetBondTypeAsDouble() != 1.0:
                all_single = False
                break
        if not all_single:
            continue
        # No non-ring heavy atoms (unsubstituted ring only)
        non_ring_heavy = sum(
            1 for i in sub_atoms
            if i not in ring_set and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        )
        if non_ring_heavy == 0 and attach_idx in ring_set:
            return _CYCLO_RETAINED[ring_size]

    # --- Alkyl branching detection ---
    # Retained alkyl names (isopropyl, tert-butyl, etc.) only apply to
    # saturated fragments.  If any bond within the fragment is double or
    # triple, this is an unsaturated substituent (alkenyl/alkynyl) that
    # must be named systematically.
    carbon_atoms = [i for i in sub_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
    carbon_count = len(carbon_atoms)

    if carbon_count == 0:
        return None

    for idx in sub_atoms:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() in frag_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    return None  # Unsaturated — use systematic naming

    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != 'C':
        return None

    # Count carbon neighbors of the attachment atom within the fragment
    c_neighbors_in_frag = [
        nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
        if nbr.GetIdx() in frag_set and nbr.GetSymbol() == 'C'
    ]

    if carbon_count == 3:
        # isopropyl: CH(CH3)2 -- 2 branches at attachment point
        if len(c_neighbors_in_frag) == 2:
            return "isopropyl"

    elif carbon_count == 4:
        if len(c_neighbors_in_frag) == 3:
            # tert-butyl: C(CH3)3 -- 3 branches at attachment
            return "tert-butyl"
        elif len(c_neighbors_in_frag) == 2:
            # Could be sec-butyl: CH(CH3)(CH2CH3)
            # Check branch sizes
            branch_sizes = []
            for nb_idx in c_neighbors_in_frag:
                sub_visited = {attach_idx}
                sub_queue = deque([nb_idx])
                sub_count = 0
                while sub_queue:
                    a = sub_queue.popleft()
                    if a in sub_visited or a not in frag_set:
                        continue
                    sub_visited.add(a)
                    if mol.GetAtomWithIdx(a).GetSymbol() == 'C':
                        sub_count += 1
                    for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                        if nn.GetIdx() not in sub_visited and nn.GetIdx() in frag_set:
                            sub_queue.append(nn.GetIdx())
                branch_sizes.append(sub_count)

            branch_sizes.sort()
            if branch_sizes == [1, 2]:
                return "sec-butyl"
        elif len(c_neighbors_in_frag) == 1:
            # Check for isobutyl: CH2CH(CH3)2
            next_c = c_neighbors_in_frag[0]
            next_atom = mol.GetAtomWithIdx(next_c)
            next_c_nbrs = [
                nbr.GetIdx() for nbr in next_atom.GetNeighbors()
                if nbr.GetIdx() in frag_set and nbr.GetIdx() != attach_idx
                and nbr.GetSymbol() == 'C'
            ]
            if len(next_c_nbrs) == 2:
                return "isobutyl"

    elif carbon_count == 5:
        # neopentyl: CH2C(CH3)3
        if len(c_neighbors_in_frag) == 1:
            next_c = c_neighbors_in_frag[0]
            next_atom = mol.GetAtomWithIdx(next_c)
            next_c_nbrs = [
                nbr.GetIdx() for nbr in next_atom.GetNeighbors()
                if nbr.GetIdx() in frag_set and nbr.GetIdx() != attach_idx
                and nbr.GetSymbol() == 'C'
            ]
            if len(next_c_nbrs) == 3:
                return "neopentyl"

    return None


# ============================================================================
# Substituent CIP Stereo Descriptor
# ============================================================================


def _add_substituent_stereo(mol, sub_atoms, name):
    """Add CIP stereodescriptors to a substituent name if stereocenters exist.

    When a substituent contains one or more stereocenters with defined CIP
    labels (R/S), the descriptor is prepended: e.g. "sec-butyl" becomes
    "(R)-sec-butyl", "1-methylpropyl" becomes "(1R)-1-methylpropyl".

    For a single stereocenter the format is "(R)-name" or "(S)-name".
    For multiple stereocenters the format uses locants: "(1R,2S)-name".

    This is a systemic fix: any substituent on any parent (chain, ring,
    heterocycle) that has a stereocenter gets the descriptor.

    Args:
        mol: RDKit Mol object (CIP labels must already be assigned).
        sub_atoms: Atom indices of the substituent fragment.
        name: The substituent name without stereo (e.g., "sec-butyl").

    Returns:
        Name with stereo prefix if stereocenters found, otherwise unchanged.
    """
    if not sub_atoms or not name:
        return name

    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    # Collect CIP-labeled atoms within the substituent
    stereo_atoms = []
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPCode'):
            cip = atom.GetProp('_CIPCode')
            # Enforce uppercase for OPSIN compatibility
            if cip in ('r', 's'):
                cip = cip.upper()
            stereo_atoms.append((idx, cip))

    if not stereo_atoms:
        return name

    if len(stereo_atoms) == 1:
        # Single stereocenter: just (R) or (S), no locant needed
        _, cip = stereo_atoms[0]
        return f"({cip})-{name}"

    # Multiple stereocenters: need locants within the substituent.
    # Use a simplified numbering: sort by atom index and assign 1-based.
    sorted_atoms = sorted(sub_atoms)
    idx_to_internal = {idx: pos + 1 for pos, idx in enumerate(sorted_atoms)}

    descs = []
    for atom_idx, cip in stereo_atoms:
        locant = idx_to_internal.get(atom_idx, 0)
        descs.append((locant, cip))
    descs.sort(key=lambda x: x[0])

    desc_str = ",".join(f"{loc}{cip}" for loc, cip in descs)
    return f"({desc_str})-{name}"


# ============================================================================
# Main Entry Point
# ============================================================================


def name_substituent_fragment(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: list,
) -> Optional[str]:
    """Name a substituent fragment, routing to the appropriate naming path.

    This is the centralized entry point for all substituent naming.
    It detects the complexity of the substituent and routes accordingly:
    1. Retained names: isopropyl, tert-butyl, sec-butyl, isobutyl, phenyl.
    2. Linear alkyl (fast path): get_alkyl_name() directly.
    3. Recursive naming: extract SMILES, name recursively, convert to prefix.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: First atom of the substituent (bonded to parent chain).
        parent_chain: List of atom indices in the parent chain.

    Returns:
        Raw prefix name (e.g., "methyl", "isopropyl", "2-methylpropyl")
        WITHOUT enclosing marks. Returns None if naming fails.
    """
    if not sub_atoms:
        return None

    parent_set = set(parent_chain) if parent_chain else set()

    # Step 1: Check retained substituent names FIRST (isopropyl, tert-butyl,
    # sec-butyl, isobutyl, neopentyl, phenyl, benzyl). These depend on
    # the attachment point context and must be checked before the linear
    # alkyl fast path, which cannot distinguish e.g. propyl from isopropyl.
    retained = _check_retained_substituent(mol, sub_atoms, attach_idx)
    if retained:
        return _add_substituent_stereo(mol, sub_atoms, retained)

    # Step 2: Fast path -- linear saturated alkyl (no branching, no unsaturation)
    if _is_linear_alkyl(mol, sub_atoms):
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                return None
        return None

    # Step 2b: Unsaturated linear chain (all C, no branching, has C=C or C#C).
    # Must be handled before general recursion because the recursive path
    # doesn't know the attachment point, producing wrong locants
    # (e.g., "prop-1-enyl" instead of "prop-2-en-1-yl" for allyl).
    unsat_name = _name_unsaturated_chain(mol, sub_atoms, attach_idx, parent_set)
    if unsat_name is not None:
        return _add_substituent_stereo(mol, sub_atoms, unsat_name)

    # Step 3: Extract fragment SMILES and name recursively
    frag_smiles = _extract_fragment_smiles(mol, sub_atoms, attach_idx, parent_set)
    if frag_smiles is None:
        # Fallback: try simple carbon count for pure-carbon substituents
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        logger.warning(
            "DROP-11 substituent_skip: reason=smiles_extraction_failure atom_count=%d",
            len(sub_atoms),
        )
        return None

    # Step 4: Recursive naming via name_fragment_recursively()
    parent_name = name_fragment_recursively(frag_smiles)
    if parent_name is None:
        # Recursion depth limit or naming failure: fallback to carbon count
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        logger.warning(
            "DROP-12 substituent_skip: reason=recursion_depth_fallback frag_smiles=%s",
            frag_smiles[:60],
        )
        return None

    # Step 5: Convert parent name to prefix form
    carbon_count = sum(
        1 for i in sub_atoms
        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
    )
    prefix_name = parent_to_prefix(parent_name, chain_length=carbon_count)

    return prefix_name


# ============================================================================
# Convenience: check if substituent needs recursive naming
# ============================================================================


def needs_recursive_naming(mol, sub_atoms: List[int]) -> bool:
    """Check whether a substituent requires recursive naming.

    Returns True if the substituent is branched or contains heteroatoms
    (i.e., not a simple linear alkyl chain).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if recursive naming may be needed, False for simple linear alkyls.
    """
    return not _is_linear_alkyl(mol, sub_atoms)
