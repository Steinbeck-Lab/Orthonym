"""
Phosphorus compound naming rules per IUPAC 2013.

Handles:
- Phosphines (R3P): substitutive naming with "phosphane" (PIN, not "phosphine")
- Phosphine oxides (R3P=O): functional class naming
- Phosphonic acids (RP(O)(OH)2): suffix -phosphonic acid
- Phosphinic acids (R2P(O)OH): suffix -phosphinic acid
- Phosphate esters: functional class naming (methyl phosphate)
- Phosphanyl prefix for P as substituent (e.g., diphenylphosphanyl)
"""

from collections import Counter, deque
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name
from ..perception.molcache import atoms_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from .lambda_convention import LAMBDA, nonstandard_bonding_number


def _characterize_substituent(mol, start_idx: int, exclude: set) -> Optional[Tuple[str, str]]:
    """
    Characterize a substituent attached to phosphorus.

    Distinguishes aryl rings (phenyl, naphthyl) from alkyl chains.

    Args:
        mol: RDKit Mol object
        start_idx: Index of the atom directly bonded to P
        exclude: Set of atom indices to exclude (typically {phosphorus_idx})

    Returns:
        Tuple of (type, name) where type is "aryl" or "alkyl" and name is the
        substituent name string (e.g., "phenyl", "naphthyl", "methyl", "ethyl").
        Returns None if substituent cannot be characterized.
    """
    start_atom = mol.GetAtomWithIdx(start_idx)

    # Check if start atom is aromatic
    if start_atom.GetIsAromatic() and start_atom.GetSymbol() == 'C':
        # Collect all aromatic carbon atoms reachable from start_idx
        aromatic_atoms = set()
        queue = deque([start_idx])
        while queue:
            idx = queue.popleft()
            if idx in aromatic_atoms or idx in exclude:
                continue
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
                aromatic_atoms.add(idx)
                for neighbor in atom.GetNeighbors():
                    nbr_idx = neighbor.GetIdx()
                    if nbr_idx not in aromatic_atoms and nbr_idx not in exclude:
                        if neighbor.GetIsAromatic():
                            queue.append(nbr_idx)

        atom_count = len(aromatic_atoms)

        if atom_count == 6:
            # Verify it's a proper 6-membered aromatic ring of all carbons
            # Find the ring containing start_idx
            ri = mol.GetRingInfo()
            for ring in ri.AtomRings():
                if start_idx in ring and len(ring) == 6:
                    all_aromatic_c = all(
                        mol.GetAtomWithIdx(a).GetIsAromatic()
                        and mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                        for a in ring
                    )
                    if all_aromatic_c:
                        return ("aryl", "phenyl")
            # Fallback: if we found 6 aromatic carbons but no matching ring
            return ("aryl", "phenyl")

        elif atom_count == 10:
            # Naphthalene system (two fused 6-membered aromatic rings)
            return ("aryl", "naphthyl")

        # Unrecognized aromatic system
        return None

    # Not aromatic: the only alkyl name reachable from here is the unbranched
    # ``get_alkyl_name(n)`` stem, so PROVE the fragment is the one shape that
    # stem can spell before counting anything. A count cannot distinguish a
    # branched alkyl from its straight-chain isomer, a ring from a chain, an
    # alkene from an alkane, or a chain carrying a heteroatom from a bare one --
    # and the old walker followed C-C bonds ONLY, so every such fragment was
    # renamed as the straight chain of the same carbon count and any hanging
    # heteroatom was silently DROPPED:
    # isopropyl -> 'propyl', cyclohexyl -> 'hexyl', benzyl -> 'heptyl',
    # allyl -> 'propyl', -CH2CH2CH2OH -> 'propyl' (the -OH vanishes).
    # Each produced a well-formed name for a DIFFERENT MOLECULE. Fail closed
    # instead; the callers all treat None as "cannot characterize".
    subtree = _prove_unbranched_terminal_alkyl(mol, start_idx, exclude)
    if subtree is None:
        return None

    count = len(subtree)
    if count == 0 or count > 10:
        return None

    return ("alkyl", get_alkyl_name(count))


def _prove_unbranched_terminal_alkyl(mol, start_idx: int, exclude: set) -> Optional[List[int]]:
    """Return the fragment's carbon indices iff ``get_alkyl_name(len(...))`` is
    an HONEST name for it, else None.

    ``get_alkyl_name(n)`` can only ever spell an unbranched, saturated, acyclic,
    all-carbon chain attached at one of its two ends. This walks the WHOLE
    substituent subtree -- every element, not just the carbon skeleton -- so a
    fragment carrying a heteroatom is refused rather than counted past. That
    whole-subtree walk is the part a carbon-only counter structurally cannot do:
    it never visits the atom it drops.

    This is a structure proof, not a count. Anything it cannot prove is refused.
    """
    subtree: List[int] = []
    seen = set(exclude)
    stack = [start_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        # Any non-carbon anywhere in the substituent means the alkyl stem would
        # claim atoms it cannot spell (the dropped-heteroatom class).
        if atom.GetSymbol() != 'C':
            return None
        if atom.IsInRing():                    # a ring is not a chain
            return None
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
        subtree.append(idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in exclude or nbr.GetSymbol() == 'H':
                continue
            stack.append(nbr_idx)

    if not subtree:
        return None

    sub_set = set(subtree)

    def _in_sub_carbons(idx: int) -> int:
        return sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                   if n.GetIdx() in sub_set)

    # The free valence must sit at a chain TERMINUS -- an internal attachment is
    # 'propan-2-yl', which the unlocanted stem cannot express.
    if _in_sub_carbons(start_idx) > 1:
        return None

    for idx in subtree:
        if _in_sub_carbons(idx) > 2:           # branch point
            return None
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in sub_set and end in sub_set:
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None                # alkenyl / alkynyl

    return subtree


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """
    Count carbon atoms in an alkyl group via BFS.

    Backward-compatible wrapper around _characterize_substituent.
    """
    result = _characterize_substituent(mol, start_idx, exclude)
    if result is None:
        return 0
    sub_type, name = result
    if sub_type == "aryl":
        # For aryl groups, return the carbon count of the ring system
        # phenyl = 6, naphthyl = 10
        aryl_counts = {"phenyl": 6, "naphthyl": 10}
        return aryl_counts.get(name, 0)
    # For alkyl, reconstruct count via BFS (keep original logic for exact count)
    visited = set()
    queue = deque([start_idx])
    count = 0
    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            count += 1
            for neighbor in atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx not in visited and nbr_idx not in exclude:
                    if neighbor.GetSymbol() == 'C':
                        queue.append(nbr_idx)
    return count


def _build_substituent_string(names: List[str]) -> str:
    """
    Build a substituent prefix string from a list of substituent names.

    Handles multipliers (di-, tri-) for identical groups, and alphabetical ordering.

    Per (May 2021 errata): For mononuclear parent hydrides with
    2+ DIFFERENT substituent groups, the first alphabetically-sorted unique
    substituent gets NO enclosing marks, and subsequent unique substituents
    are wrapped in parentheses.

    Examples:
        ["methyl", "methyl", "methyl"] -> "trimethyl"
        ["butyl", "ethyl", "methyl", "propyl"] -> "butyl(ethyl)(methyl)(propyl)"
        ["ethyl", "methyl", "methyl"] -> "ethyl(dimethyl)"
        ["ethyl", "phenyl", "phenyl"] -> "ethyl(diphenyl)"
        ["phenyl", "phenyl"] -> "diphenyl"
        ["methyl", "phenyl", "phenyl"] -> "methyl(diphenyl)"

    Note: IUPAC alphabetical ordering ignores multiplicative prefixes (di, tri).

    : two corrections that only became REACHABLE when the organyl guard
    started admitting compound prefixes (``benzyl``, ``cyclohexylmethyl``,
    ``(4-bromophenyl)methyl``):

    * the order is the alphanumerical one (letters only, sec-/tert-
      excluded), not raw string order -- raw order sorts ``(4-bromophenyl)methyl``
      on its leading ``(``;
    * the enclosing marks escalate (-> [ -> { for a name that already carries
      brackets, and the FIRST cited group is enclosed too when it is
      itself a compound prefix, since 's "first group bare"
      only removes the marks that separate the groups from each other.
    """
    from ..assembly.naming_utils import (
        apply_enclosing_marks,
        enclose_if_compound,
        multiplied_component,
        prefix_citation_sort_key,
    )

    counts = Counter(names)
    # -CLOSEOUT Item A: the arity BOUND stays local (fail closed beyond
    # it); the multiplier WORD comes from the shared primitive, which knows
    # (a). This table could only say di/tri/tetra, so a SUBSTITUTED
    # organyl on a phosphane could never take bis/tris.
    _SUPPORTED_COUNTS = frozenset((1, 2, 3, 4))

    # -FIX Item 2: this held a THIRD private copy of the citation key --
    # `_alnum_key = re.sub(r"[^a-z]", "", alpha_sort_key(name))` -- which is
    # exactly the shared key's letters-only TIER 1 and nothing else. With no
    # locant tier it could not tell `2-methylbutyl` from `3-methylbutyl`, and
    # `sorted` being stable, the tie resolved to Counter insertion order =
    # RDKit neighbour order = how the SMILES was written. One molecule, two
    # names: `CCC(C)C[As](C)CCC(C)C` gave `methyl(2-methylbutyl)(3-methylbutyl)-
    # arsane` and its own re-spelling gave `methyl(3-methylbutyl)(2-methylbutyl)-
    # arsane`. The shared key keeps tier 1 byte-identical and adds the
    # `****` locant tier (`the Blue Book`) plus a total-order
    # backstop, so it can only change an order that was previously UNDEFINED.
    sorted_unique = sorted(counts.keys(), key=prefix_citation_sort_key)

    parts = []
    for i, name in enumerate(sorted_unique):
        count = counts[name]
        if count not in _SUPPORTED_COUNTS:
            return None
        #: with the widened organyl class these three branches began
        # receiving locanted and italicized prefixes, exposing three defects that
        # BB 7272 settles verbatim -- "the second and further
        # substituents are each enclosed with parentheses even for simple
        # substituents. When the simple substituent groups are accompanied by
        # multiplicative prefixes such as 'di' and 'tri', the multiplicative
        # prefixes are NOT included in the parentheses."
        marked = enclose_if_compound(name)
        if len(sorted_unique) >= 2 and i > 0:
            # Was `apply_enclosing_marks(f"{multiplier}{name}")`, which put the
            # multiplier INSIDE the marks: 'tert-butyl(dimethyl)phosphane'. BB
            # 16286 writes 'tert-butyldi(methyl)phosphane' (PIN).
            if marked == name:
                marked = apply_enclosing_marks(name, -1)
            parts.append(multiplied_component(count, name, marked))
        elif count > 1:
            # BB 25721 '1,4-di(propan-2-yl)cyclohexane' (PIN) keeps the marks
            # with the SIMPLE multiplier outside; (d) keeps the
            # italicized hyphen. Both now come from `multiplied_component`.
            parts.append(multiplied_component(count, name, marked))
        else:
            # First (or only) group: bare unless it is itself compound/complex.
            parts.append(marked)

    return "".join(parts)


def name_phosphine(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Name a phosphine using substitutive nomenclature.

    IUPAC 2013 PIN: "phosphane" (not "phosphine")
    - methylphosphane, dimethylphosphane, trimethylphosphane
    - triphenylphosphane, diphenylmethylphosphane

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Substitutive name like "trimethylphosphane", or None if not a simple phosphine
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon/aromatic-carbon neighbors
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']

    if len(neighbors) == 0:
        return "phosphane"  # Parent hydride PH3

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None  # Unrecognized substituent
        _, name = result
        sub_names.append(name)

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphane"


def name_phosphine_oxide(mol, phosphine_oxide_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a phosphine oxide using substitutive nomenclature.

    IUPAC: trimethylphosphane oxide (substitutive, preferred)

    Args:
        mol: RDKit Mol object
        phosphine_oxide_atoms: Atom indices from SMARTS match

    Returns:
        Name like "trimethylphosphane oxide", or None if not simple
    """
    # Find the phosphorus atom
    phosphorus_idx = None
    for idx in phosphine_oxide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'P':
            phosphorus_idx = idx
            break

    if phosphorus_idx is None:
        return None

    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors (exclude oxygen)
    neighbors = [n for n in phosphorus.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 3:
        return None  # Must have exactly 3 carbon substituents

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None
        _, name = result
        sub_names.append(name)

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    return f"{prefix}phosphane oxide"


# organo-oxoacid stems for the pnictogens, keyed by central element:
# symbol -> (-onic stem, -inic stem)
# BB L36051-36054 gives all six as PRESELECTED names. Bismuth has no oxoacid
# analogue in the Blue Book, so the family stops at Sb. This table is what makes
# the two namers below element-generic instead of phosphorus-only -- an arsonic
# acid is a oxoacid, never a organometallic.
_PNICTOGEN_OXOACID_STEMS = {
    'P':  ('phosphonic', 'phosphinic'),
    'As': ('arsonic', 'arsinic'),
    'Sb': ('stibonic', 'stibinic'),
}

# a phase: the TRIVALENT -ous analogues (no =O; HE(OH)2 / H2E(OH) parents).
# symbol -> (-onous full-word stem, -inous full-word stem)
# BB gives all six as PRESELECTED preferred names (BB L35499-35507 phosphorus;
# L35413-35443 arsenic/antimony). The whole word is stored (not just the base)
# because the -ous namers append " acid" directly, exactly as the -onic/-inic
# helpers do with the -ic words above.
_PNICTOGEN_OUS_STEMS = {
    'P':  ('phosphonous', 'phosphinous'),
    'As': ('arsonous', 'arsinous'),
    'Sb': ('stibonous', 'stibinous'),
}


def _find_central_atom(mol, atoms: Tuple[int, ...], symbol: str) -> Optional[int]:
    """Index of the first ``symbol`` atom in a SMARTS match, else None."""
    for idx in atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == symbol:
            return idx
    return None


def _organyl_or_junior_prefix(mol, c_idx: int, central_idx: int) -> Optional[str]:
    """Name the organyl rooted at ``c_idx`` (parent-side neighbour ``central_idx``)
    as a detachable prefix for a pnictogen-oxoacid parent.

    Primary path: the narrow, stereo-safe:func:`organyl_prefix_name`, which
    admits only C/H/halogen prefixes and keeps every existing pnictogen-acid name
    BYTE-IDENTICAL (``4-methylphenyl``, ``benzyl``, ``diphenyl``…). When the
    substituent additionally carries a group JUNIOR to the pnictogen oxoacid, that
    narrow guard fail-closes. These namers are reached only after seniority
    selection made the P/As/Sb acid the PRINCIPAL group class 7c),
    so a ring hydroxy (class 17) or amine (class 19) on the substituent is a mere
    detachable prefix; the fallback names the FULL fragment with the general
    substituent enumerator (``4-hydroxyphenyl``, ``4-aminophenyl``) so
    ``bis(4-hydroxyphenyl)phosphinic acid`` / ``bis(4-aminophenyl)phosphinic acid``
    can be built.

    Fail-closed (``None``) off a non-carbon attachment, a charged/radical atom in
    the fragment (an ion/radical is a class SENIOR to the acid,, a
    multi-word / sentinel token, or an unexpressed stereo element (a
    stereo-stripped prefix names a different compound and cannot see it —
    it compares the InChIKey skeleton block). Pure: no mol mutation.
    """
    from .substituent_purity import organyl_prefix_name  # lazy: avoid import cycle

    name = organyl_prefix_name(mol, c_idx, central_idx)
    if name is not None:
        return name

    # Fallback: name the full substituent fragment, crossing junior O/N groups.
    if mol.GetAtomWithIdx(c_idx).GetSymbol() != 'C':
        return None                         # an organyl is carbon-attached
    frag: set = set()
    stack = [c_idx]
    while stack:
        a = stack.pop()
        if a in frag:
            continue
        frag.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j == central_idx or j in frag:
                continue
            stack.append(j)
    for a in frag:                          # ion / radical -> senior class, defer
        at = mol.GetAtomWithIdx(a)
        if at.GetFormalCharge() != 0 or at.GetNumRadicalElectrons() != 0:
            return None

    from ..assembly.substituent_enumerator import name_substituent
    from ..errors import is_refusal_sentinel
    try:
        token = name_substituent(mol, frozenset(frag), c_idx, allow_mancude=False)
    except Exception:  # noqa: BLE001 - a producer bug must degrade, not crash
        return None
    if (not token or not isinstance(token, str)
            or is_refusal_sentinel(token) or ' ' in token):
        return None

    # Stereo-expression obligation (mirror organyl_prefix_name): an identity, not
    # an inequality -- a partial- or over-attributed stereo name is a different
    # compound.
    from .stereochemistry import (
        count_defined_stereo_in_fragment,
        count_expressed_stereo_descriptors,
    )
    defined = count_defined_stereo_in_fragment(mol, frozenset(frag))
    if defined and count_expressed_stereo_descriptors(token) != defined:
        return None
    return token


def _name_pnictogen_onic_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R-E(=O)(OH)2 -> ``{R-yl}{stem} acid`` for E in {P, As, Sb}.

    The single shared implementation behind:func:`name_phosphonic_acid` and its
    arsenic/antimony analogues. Fail-closed (``None``) unless the one organyl can
    be named as a detachable prefix, so an unprovable substituent defers to the
    generic path rather than risking a wrong name.

    : the organyl goes through:func:`organyl_prefix_name`, which routes to
    the shared substituent chokepoint, so a ring-bearing / branched / unsaturated
    organyl is NAMED instead of refused -- ``benzylphosphonic acid`` with the
     retained preferred prefix (BB ``2-benzylpyridine`` PIN), and
    ``[(4-bromophenyl)methyl]phosphonic acid`` for the substituted benzyl that
     forbids spelling as a benzyl (BB ``2-[(4-bromophenyl)methyl]-
    pyridine`` PIN). A compound or complex prefix takes enclosing marks.
    """
    from ..assembly.naming_utils import enclose_if_compound

    stem = _PNICTOGEN_OXOACID_STEMS.get(symbol, (None, None))[0]
    if stem is None:
        return None

    central_idx = _find_central_atom(mol, match_atoms, symbol)
    if central_idx is None:
        return None

    central = mol.GetAtomWithIdx(central_idx)
    carbons = [n for n in central.GetNeighbors() if n.GetSymbol() == 'C']
    if len(carbons) != 1:               # an -onic acid has exactly one C-E bond
        return None

    prefix = _organyl_or_junior_prefix(mol, carbons[0].GetIdx(), central_idx)
    if prefix is None:
        return None                     # unprovable organyl -> defer (fail-closed)
    return f"{enclose_if_compound(prefix)}{stem} acid"


def _name_pnictogen_inic_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R2E(=O)OH -> ``{R-yl}{R-yl}{stem} acid`` for E in {P, As, Sb}.

    Shared implementation behind:func:`name_phosphinic_acid` and its
    arsenic/antimony analogues. Mixed substituents get the enclosing
    marks from:func:`_build_substituent_string`, which yields the BB L36066
    verbatim PIN ``methyl(phenyl)arsinic acid`` for C6H5-As(CH3)(O)OH.

    ACCURACY FIX (Phase B): each substituent goes through the shared organyl
    guard instead of calling ``_characterize_substituent`` raw.
    ``_characterize_substituent`` counts the carbons of a non-aromatic subtree
    as a LINEAR chain, so a cyclic substituent was silently renamed --
    benzyl(methyl)phosphinic acid came back as 'heptyl(methyl)phosphinic acid'
    and cyclohexyl(methyl) as 'hexyl(methyl)', both a WRONG CONSTITUTION. Those
    strings were caught downstream by the OPSIN check, but that gate
    fails OPEN when no JRE is present, so the refusal has to happen here. Fixed
    for P, As and Sb together since all three share this code path.

    : that guard is now:func:`organyl_prefix_name`, which routes to the
    shared substituent chokepoint, so the cyclic substituent is NAMED rather than
    refused -- ``benzyl(methyl)phosphinic acid``.

     a review: substituents are named through
    :func:`_organyl_or_junior_prefix`, which falls back to the full-fragment
    enumerator for an aryl bearing a group JUNIOR to the acid, so
    ``bis(4-hydroxyphenyl)phosphinic acid`` / ``bis(4-aminophenyl)phosphinic
    acid`` build: hydroxy/amine are junior to a class-7c acid).
    """
    stem = _PNICTOGEN_OXOACID_STEMS.get(symbol, (None, None))[1]
    if stem is None:
        return None

    central_idx = _find_central_atom(mol, match_atoms, symbol)
    if central_idx is None:
        return None

    central = mol.GetAtomWithIdx(central_idx)
    neighbors = [n for n in central.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:             # an -inic acid has exactly two C-E bonds
        return None

    sub_names = []
    for neighbor in neighbors:
        name = _organyl_or_junior_prefix(mol, neighbor.GetIdx(), central_idx)
        if name is None:
            return None                 # unprovable organyl -> defer (fail-closed)
        sub_names.append(name)

    return f"{_build_substituent_string(sub_names)}{stem} acid"


def _name_pnictogen_onous_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R-E(OH)2 -> ``{R-yl}{stem}ous acid`` for E in {P, As, Sb}.

    The trivalent (NO =O) analogue of:func:`_name_pnictogen_onic_acid`, sharing
    its organyl-prefix chokepoint and fail-closed contract. BB L35467 verbatim
    ``phenylstibonous acid`` for C6H5-Sb(OH)2; BB L35472 verbatim
    ``(naphthalene-2,6-diyl)bis(phosphonous acid)`` shows the same
    substituent-prefix construction.
    """
    from ..assembly.naming_utils import enclose_if_compound

    stem = _PNICTOGEN_OUS_STEMS.get(symbol, (None, None))[0]
    if stem is None:
        return None

    central_idx = _find_central_atom(mol, match_atoms, symbol)
    if central_idx is None:
        return None

    central = mol.GetAtomWithIdx(central_idx)
    carbons = [n for n in central.GetNeighbors() if n.GetSymbol() == 'C']
    if len(carbons) != 1:               # an -onous acid has exactly one C-E bond
        return None

    prefix = _organyl_or_junior_prefix(mol, carbons[0].GetIdx(), central_idx)
    if prefix is None:
        return None                     # unprovable organyl -> defer (fail-closed)
    return f"{enclose_if_compound(prefix)}{stem} acid"


def _name_pnictogen_inous_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R2E(OH) -> ``{R-yl}{R-yl}{stem}ous acid`` for E in {P, As, Sb}.

    The trivalent (NO =O) analogue of:func:`_name_pnictogen_inic_acid`. Mixed
    substituents get the enclosing marks from
    :func:`_build_substituent_string`. BB L35465 verbatim ``diphenylarsinous
    acid`` for (C6H5)2As(OH).
    """
    stem = _PNICTOGEN_OUS_STEMS.get(symbol, (None, None))[1]
    if stem is None:
        return None

    central_idx = _find_central_atom(mol, match_atoms, symbol)
    if central_idx is None:
        return None

    central = mol.GetAtomWithIdx(central_idx)
    neighbors = [n for n in central.GetNeighbors() if n.GetSymbol() == 'C']
    if len(neighbors) != 2:             # an -inous acid has exactly two C-E bonds
        return None

    sub_names = []
    for neighbor in neighbors:
        name = _organyl_or_junior_prefix(mol, neighbor.GetIdx(), central_idx)
        if name is None:
            return None                 # unprovable organyl -> defer (fail-closed)
        sub_names.append(name)

    return f"{_build_substituent_string(sub_names)}{stem} acid"


def name_phosphonous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R-P(OH)2 -> ``phenylphosphonous acid`` substituent-prefix PIN)."""
    return _name_pnictogen_onous_acid(mol, match_atoms, 'P')


def name_phosphinous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R2P(OH) -> ``diphenylphosphinous acid`` substituent-prefix PIN)."""
    return _name_pnictogen_inous_acid(mol, match_atoms, 'P')


def name_arsonous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R-As(OH)2 -> ``phenylarsonous acid`` (BB L35413 preselected name)."""
    return _name_pnictogen_onous_acid(mol, match_atoms, 'As')


def name_arsinous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R2As(OH) -> ``diphenylarsinous acid`` (BB L35465 preselected PIN)."""
    return _name_pnictogen_inous_acid(mol, match_atoms, 'As')


def name_stibonous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R-Sb(OH)2 -> ``phenylstibonous acid`` (BB L35467 preselected PIN)."""
    return _name_pnictogen_onous_acid(mol, match_atoms, 'Sb')


def name_stibinous_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R2Sb(OH) -> ``diphenylstibinous acid`` (BB L35435 preselected name)."""
    return _name_pnictogen_inous_acid(mol, match_atoms, 'Sb')


def name_arsonic_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R-As(=O)(OH)2 -> ``methylarsonic acid`` (BB L36051 preselected name)."""
    return _name_pnictogen_onic_acid(mol, match_atoms, 'As')


def name_arsinic_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R2As(=O)OH -> ``dimethylarsinic acid`` (BB L36052 preselected name)."""
    return _name_pnictogen_inic_acid(mol, match_atoms, 'As')


def name_stibonic_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R-Sb(=O)(OH)2 -> ``methylstibonic acid`` (BB L36054 preselected name)."""
    return _name_pnictogen_onic_acid(mol, match_atoms, 'Sb')


def name_stibinic_acid(mol, match_atoms: Tuple[int, ...]) -> Optional[str]:
    """R2Sb(=O)OH -> ``dimethylstibinic acid`` (BB L36054 preselected name)."""
    return _name_pnictogen_inic_acid(mol, match_atoms, 'Sb')


def name_phosphonic_acid(mol, phosphonic_atoms: Tuple[int, ...]) -> Optional[str]:
    """Name an organyl phosphonic acid in substituent-prefix mode (IUPAC PIN).

    R-P(=O)(OH)2 -> ``methylphosphonic acid`` / ``ethylphosphonic acid`` /
    ``phenylphosphonic acid``. Phosphonic acid is a functional
    *parent* (HP(=O)(OH)2) whose central-atom H is substituted by the organyl
    group; the PIN is therefore ``{R-yl}phosphonic acid`` — NOT the
    parent-hydride-stem form ``{R-ane}phosphonic acid`` (``ethanephosphonic``),
    which the generic suffix assembler would otherwise emit (it correctly serves
    the genuine *suffix* acids like ``ethanesulfonic``).

    Mirrors:func:`name_phosphinic_acid`. Returns ``None`` (fail-closed) when the
    single organyl substituent is not a clean simple alkyl / aryl — the caller
    then defers to the generic path (no regression for complex parents).

    Thin wrapper over the element-generic:func:`_name_pnictogen_onic_acid`,
    which the arsenic/antimony analogues share.
    """
    return _name_pnictogen_onic_acid(mol, phosphonic_atoms, 'P')


def name_acyloxy_phosphonic_acid(mol) -> Optional[str]:
    """ (BB L36999): a mixed acyl/phosphoric anhydride named
    SUBSTITUTIVELY as an ``(acyloxy)phosphonic acid`` (acid is senior to
    anhydride, so the substitutive acid name is the PIN, NOT the functional-class
    ``acetic phosphoric monoanhydride``).

        CH3-CO-O-P(O)(OH)2 -> (acetyloxy)phosphonic acid

    Structural shape: exactly one NEUTRAL P of degree 4 bearing one ``P=O``, two
    ``-OH`` and one ``-O-acyl`` (the bridging O joins P to a carbonyl carbon). The
    fourth position of phosphonic acid ``HP(=O)(OH)2`` — normally the central-atom
    H (methylphosphonic) — is here substituted by the acyloxy group.

    Fail-closed (returns ``None``) off this shape: any C-P bond (genuine
    organophosphonic acid -> the suffix path), an ``-O-alkyl`` bridge (a phosphate
    ester, not an anhydride), fewer/more than two -OH, a charge, or an acyl group
    the acyloxy namer cannot spell. Pure: no mol mutation.
    """
    from rdkit import Chem as _Chem

    from .lipids import _acyloxy_for_site

    if mol is None or len(_Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in atoms_of(mol)) != 0:
        return None
    ps = [a for a in atoms_of(mol) if a.GetSymbol() == "P"]
    if len(ps) != 1:
        return None
    p = ps[0]
    if (p.GetFormalCharge() != 0 or p.GetDegree() != 4
            or p.GetTotalNumHs() != 0):
        return None
    dbl_o = 0
    oh_count = 0
    acyloxy_site = None
    for b in p.GetBonds():
        nb = b.GetOtherAtom(p)
        bt = b.GetBondType()
        if bt == _Chem.BondType.DOUBLE and nb.GetSymbol() == "O":
            dbl_o += 1
            continue
        if bt != _Chem.BondType.SINGLE or nb.GetSymbol() != "O" \
                or nb.GetFormalCharge() != 0:
            return None                              # any C-P / N-P / X-P -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != p.GetIdx()]
        if nb.GetTotalNumHs() == 1 and not others:
            oh_count += 1
        elif nb.GetDegree() == 2 and len(others) == 1 and others[0].GetSymbol() == "C":
            c = others[0]
            # the bridge carbon must be a CARBONYL carbon (acyl), not alkyl:
            is_carbonyl = any(
                bb.GetBondType() == _Chem.BondType.DOUBLE
                and bb.GetOtherAtom(c).GetSymbol() == "O"
                for bb in c.GetBonds())
            if not is_carbonyl:
                return None                          # -O-alkyl -> phosphate ester
            if acyloxy_site is not None:
                return None                          # >1 acyloxy -> out of scope
            acyloxy_site = (c.GetIdx(), nb.GetIdx())
        else:
            return None
    if dbl_o != 1 or oh_count != 2 or acyloxy_site is None:
        return None
    acyloxy = _acyloxy_for_site(mol, ("acyl", acyloxy_site[0], acyloxy_site[1]))
    if not acyloxy:
        return None
    return f"({acyloxy})phosphonic acid"


_PNICTOGEN_ACID_STEM = {'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane'}
_PNICTOGEN_ACID_MULT = {1: '', 2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta'}


def name_phosphane_carboxylic_acid(mol) -> Optional[str]:
    """ (BB 39117): a carboxylic acid ``-C(=O)OH`` on a Group-15
    P/As/Sb parent hydride is expressed as the added-carbon ``-carboxylic acid``
    suffix on the phosphane/arsane/stibane parent hydride — NOT a phosphanyl
    prefix on methanoic acid (the generic acid namer's non-PIN
    ``1-phosphanylmethanoic acid``)::

        H2P-COOH -> phosphanecarboxylic acid (PIN)

    Exactly analogous to the added-carbon ``-carboxylic acid`` on carbocycles
    (cyclohexanecarboxylic acid) and to polyazane's azane-1-carboxylic acid.

    Scope (fail-closed graph classifier, NOT SMARTS): exactly ONE bare carboxyl
    carbon (``=O`` + ``-OH`` + one pnictogen neighbour, degree 3) on a PURE
    homonuclear P/As/Sb chain (H-saturated, standard bonding number 3, neutral,
    acyclic). A ``P=O`` (phosphonic/phosphinic, retained acids), a
    lambda5 hydride, a ring, a second characteristic group, a stray heteroatom, a
    charge/radical, or an interior carboxyl attachment fails a guard and cascades
    onward. Pure: no mol mutation.
    """
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    if mol.GetRingInfo().NumRings() > 0:
        return None

    # Locate exactly ONE bare carboxyl carbon bonded to a pnictogen.
    carboxyl_c = hub = None
    n_carboxyls = 0
    for atom in atoms_of(mol):
        if atom.GetSymbol() != 'C':
            continue
        nbrs = atom.GetNeighbors()
        o_double = [n for n in nbrs if n.GetSymbol() == 'O' and n.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(
                        atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2.0]
        o_single = [n for n in nbrs if n.GetSymbol() == 'O' and n.GetDegree() == 1
                    and n.GetTotalNumHs() >= 1
                    and mol.GetBondBetweenAtoms(
                        atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1.0]
        p_nbrs = [n for n in nbrs if n.GetSymbol() in _PNICTOGEN_ACID_STEM]
        if (len(o_double) == 1 and len(o_single) == 1 and len(p_nbrs) == 1
                and atom.GetDegree() == 3):
            n_carboxyls += 1
            carboxyl_c = atom.GetIdx()
            hub = p_nbrs[0].GetIdx()
    if carboxyl_c is None or n_carboxyls != 1:
        return None

    element = mol.GetAtomWithIdx(hub).GetSymbol()
    stem = _PNICTOGEN_ACID_STEM[element]

    # The rest of the molecule must be a PURE homonuclear pnictogen chain of the
    # SAME element plus the single carboxyl (C + its two O's).
    pnic = [a for a in atoms_of(mol) if a.GetSymbol() in _PNICTOGEN_ACID_STEM]
    if any(a.GetSymbol() != element for a in pnic):
        return None
    pnic_idxs = {a.GetIdx() for a in pnic}
    carboxyl_atoms = {carboxyl_c}
    for n in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if n.GetSymbol() == 'O':
            carboxyl_atoms.add(n.GetIdx())
    for a in atoms_of(mol):
        if a.GetSymbol() == 'H':
            continue
        if a.GetIdx() in pnic_idxs or a.GetIdx() in carboxyl_atoms:
            continue
        return None                              # stray heteroatom / extra carbon

    # Every pnictogen: standard bonding number 3, only single bonds to another
    # pnictogen or the carboxyl carbon (a P=O phosphoryl / any hetero on P -> defer).
    for a in pnic:
        if nonstandard_bonding_number(mol, a.GetIdx()) is not None:
            return None
        for b in a.GetBonds():
            other = b.GetOtherAtom(a)
            if other.GetSymbol() == 'H':
                continue
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None
            if other.GetIdx() not in pnic_idxs and other.GetIdx() != carboxyl_c:
                return None

    deg = {a.GetIdx(): sum(1 for nb in a.GetNeighbors()
                           if nb.GetIdx() in pnic_idxs) for a in pnic}
    if any(d > 2 for d in deg.values()):
        return None                              # branched pnictogen chain

    n = len(pnic)
    if n == 1:
        return f"{stem}carboxylic acid"          # phosphanecarboxylic acid (PIN)
    # Multinuclear: the acid-bearing pnictogen (hub) must be a chain terminus so
    # the '-carboxylic acid' suffix takes locant 1, lowest locant).
    ends = [i for i, d in deg.items() if d <= 1]
    if len(ends) != 2 or hub not in ends:
        return None
    mult = _PNICTOGEN_ACID_MULT.get(n)
    if not mult:
        return None
    return f"{mult}{stem}-1-carboxylic acid"


def name_phosphinic_acid(mol, phosphinic_atoms: Tuple[int, ...]) -> Optional[str]:
    """
    Name a phosphinic acid with dialkyl/aryl prefix and phosphinic acid suffix.

    R2P(O)(OH) -> dialkylphosphinic acid, diphenylphosphinic acid

    Args:
        mol: RDKit Mol object
        phosphinic_atoms: Atom indices from SMARTS match

    Returns:
        Name like "dimethylphosphinic acid", or None if not simple

    Thin wrapper over the element-generic:func:`_name_pnictogen_inic_acid`,
    which the arsenic/antimony analogues share.
    """
    return _name_pnictogen_inic_acid(mol, phosphinic_atoms, 'P')


def _p_ester_owner_group(mol, o_idx: int, p_idx: int) -> Optional[str]:
    """Name the -O-R ester owner R as a substituent token (recursive namer).

    Returns the owner '-yl' name (e.g. 'ethyl', 'propan-2-yl',
    '2,3-dichloropropyl') and the set of owner atoms, or None if R is not a
    clean carbon-anchored group the substituent namer can spell.
    """
    from ..assembly.substituent_enumerator import name_substituent
    o_atom = mol.GetAtomWithIdx(o_idx)
    # The owner root is the O's single non-P heavy neighbour: a carbon (ordinary
    # ester, -O-CR) or a sulfur (sulfenyl ester -O-S-R, tail #16).
    r_neighbors = [n for n in o_atom.GetNeighbors()
                   if n.GetIdx() != p_idx and n.GetSymbol() in ('C', 'S')]
    if len(r_neighbors) != 1:
        return None
    root = r_neighbors[0]

    if root.GetSymbol() == 'S':
        # Sulfenyl ester -O-S-R -> the owner token is '{R}sulfanyl'
        # ('dodecylsulfanyl'). name_substituent rooted at S mis-fires (its S-attach
        # sulfanyl tier needs a CARBON parent side, but here the parent is the
        # ester O), so name the carbon arm R directly and append 'sulfanyl'
        # / the composed-chalcogen morphology). Fail-closed off a clean
        # single-carbon-arm sulfenyl.
        s_idx = root.GetIdx()
        s_arms = [n for n in root.GetNeighbors()
                  if n.GetIdx() != o_idx and n.GetAtomicNum() > 1]
        if len(s_arms) != 1 or s_arms[0].GetSymbol() != 'C':
            return None
        rc = s_arms[0].GetIdx()
        r_frag: set = set()
        stack = [rc]
        while stack:
            a = stack.pop()
            if a in r_frag or a in (o_idx, s_idx):
                continue
            r_frag.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j == p_idx:
                    return None
                if j not in (o_idx, s_idx) and j not in r_frag:
                    stack.append(j)
        try:
            r_token = name_substituent(mol, frozenset(r_frag), rc)
        except Exception:  # noqa: BLE001 - a producer bug must degrade, not crash
            return None
        if not r_token or r_token == 'substituent' or not isinstance(r_token, str):
            return None
        from ..assembly.naming_utils import apply_enclosing_marks, is_complex_substituent
        if is_complex_substituent(r_token):
            r_token = apply_enclosing_marks(r_token, -1)
        return f"{r_token}sulfanyl", frozenset(r_frag | {s_idx})

    c0 = root.GetIdx()
    # Collect R = everything reachable from c0 without crossing the ester O.
    frag: set = set()
    stack = [c0]
    while stack:
        a = stack.pop()
        if a in frag or a == o_idx:
            continue
        frag.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j == p_idx:                    # R loops back to P -> not a clean ester
                return None
            if j != o_idx and j not in frag:
                stack.append(j)
    try:
        token = name_substituent(mol, frozenset(frag), c0)
    except Exception:  # noqa: BLE001 - a producer bug must degrade, not crash
        return None
    if not token or token in ('substituent',) or not isinstance(token, str):
        return None
    return token, frozenset(frag)


def _assemble_p_owner_text(owner_tokens: List[str]) -> str:
    """Functional-class ester owner list -> multiplied, alphabetised word block.

    Identical simple owners collapse with di/tri (``dimethyl``); a substituted /
    compound owner takes bis/tris with enclosing marks (``bis(2-chloroethyl)``).
    """
    from collections import Counter

    from ..assembly.naming_utils import (
        COMPLEX_MULTIPLIERS,
        SIMPLE_MULTIPLIERS,
        alpha_sort_key,
        is_complex_substituent,
    )
    counts = Counter(owner_tokens)
    parts = []
    for token in sorted(counts, key=alpha_sort_key):
        k = counts[token]
        # A compound owner takes bis/tris with marks. The char scan catches
        # locanted/parenthesised owners; is_complex_substituent additionally
        # catches a char-free compound prefix such as 'dodecylsulfanyl' (#16
        # tris(dodecylsulfanyl) phosphite), which 'tridodecylsulfanyl' would
        # otherwise render ambiguously.
        # carve-out (shared primitive): an italicized-led simple owner
        # ('tert-butyl') is cited bare with a hyphenated simple multiplier,
        # 'tri-tert-butyl phosphite', as in '1,2-di-*tert*-butylbenzene (PIN)'
        # (the Blue Book); the raw hyphen test made it
        # 'tris(tert-butyl) phosphite' (j7, TRIAGE g7 C20).
        from ..assembly.naming_utils import (
            italicized_prefix_is_bare,
            multiplier_needs_hyphen,
        )
        is_complex = (not italicized_prefix_is_bare(token)
                      and (any(ch in token for ch in "()[]-, 0123456789")
                           or is_complex_substituent(token)))
        if k == 1:
            parts.append(f"({token})" if False else token)
        elif is_complex:
            parts.append(f"{COMPLEX_MULTIPLIERS[k]}({token})")
        else:
            _sep = "-" if multiplier_needs_hyphen(token) else ""
            parts.append(f"{SIMPLE_MULTIPLIERS[k]}{_sep}{token}")
    return " ".join(parts)


# P-oxo-acid stem by (has_P_double_O/S, number of C ligands). Phosphite = no =O.
_P_ACID_STEM = {
    (True, 0): "phosphate",     # (RO)nP(=O)... -> "... phosphate"
    (True, 1): "phosphonate",   # R-P(=O)(OR')(OR'') -> "... {R}phosphonate"
    (True, 2): "phosphinate",   # R2-P(=O)(OR') -> "... {R,R}phosphinate"
    (False, 0): "phosphite",    # (RO)3P -> "... phosphite"
}
_HYDROGEN_MULT = {0: "", 1: "hydrogen", 2: "dihydrogen"}

# -inate ester stem for E in {P, As, Sb} (R2E(=O)OR', class 9).
_PNICTOGEN_INATE_ESTER_STEM = {"P": "phosphinate", "As": "arsinate", "Sb": "stibinate"}


def _name_pnictogen_inate_ester(mol, match_atoms, symbol) -> Optional[str]:
    """R2E(=O)(O-R') -> ``{owner} {di-organyl}{stem}`` for E in {P, As, Sb}.

    The functional-class ester of a phosphinic/arsinic/stibinic acid class 9): ``methyl diphenylphosphinate``, ``methyl diphenylarsinate``.
    This is the EXACT bridge shape the multiplicative pnictogen guard declines —
    two C-E bonds, one terminal =O, one -O-R ester residual — so declining that
    multiplicative name hands the molecule here. Element-generic so P/As/Sb cannot
    drift (As/Sb had NO ester namer at all; the P-only ``name_phosphate_ester``
    never perceived a phosphinate ester either). Reuses the ester-owner and
    di-organyl machinery. Fail-closed (``None``) off the clean neutral single-E
    ``-inate`` ester, a thio residual (P-S / P=S out of scope), or a
    C-ligand / owner the substituent namer cannot spell, or a name that would not
    cover every atom.
    """
    stem = _PNICTOGEN_INATE_ESTER_STEM.get(symbol)
    if stem is None or mol is None:
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    e_idx = _find_central_atom(mol, match_atoms, symbol)
    if e_idx is None:
        return None
    e = mol.GetAtomWithIdx(e_idx)
    if e.GetFormalCharge() != 0 or e.IsInRing() or e.GetTotalNumHs() != 0:
        return None

    accounted = {e_idx}
    ester_oxygens: List[int] = []
    c_ligands: List[int] = []
    dbl_oxo = 0
    for b in e.GetBonds():
        nb = b.GetOtherAtom(e)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # E=C / E=S etc. -> defer
        if sym == 'C':
            c_ligands.append(nb.GetIdx())
            continue
        if sym != 'O' or nb.GetFormalCharge() != 0:
            return None                         # E-S / E-N / E-O-E bridge -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != e_idx]
        if len(others) == 1 and others[0].GetSymbol() == 'C':
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # free -OH (acid) or O bridge -> defer
    if dbl_oxo != 1 or len(c_ligands) != 2 or len(ester_oxygens) != 1:
        return None                             # not the clean -inate ester shape

    got = _p_ester_owner_group(mol, ester_oxygens[0], e_idx)
    if got is None:
        return None
    owner_token, owner_frag = got
    accounted |= owner_frag
    owner_text = _assemble_p_owner_text([owner_token])

    # The two C-E ligands become the '-inate' stem's di-organyl carbon prefix.
    from ..assembly.substituent_enumerator import name_substituent
    c_tokens: List[str] = []
    for c_idx in c_ligands:
        frag: set = set()
        stack = [c_idx]
        while stack:
            a = stack.pop()
            if a in frag:
                continue
            frag.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j == e_idx or j in frag:
                    continue
                stack.append(j)
        try:
            tok = name_substituent(mol, frozenset(frag), c_idx, allow_mancude=True)
        except Exception:  # noqa: BLE001 - a producer bug must degrade, not crash
            return None
        if not tok or tok == 'substituent' or not isinstance(tok, str):
            return None
        c_tokens.append(tok)
        accounted |= frag
    stem_prefix = _build_substituent_string(c_tokens)

    if accounted != set(range(mol.GetNumAtoms())):
        return None                             # not a whole-molecule name
    return f"{owner_text} {stem_prefix}{stem}"


def name_phosphate_ester(mol, phosphorus_idx: int) -> Optional[str]:
    """Functional-class name for an ester of a phosphorus oxo-acid.

    Covers phosphoric-acid esters (mono/di/tri: ``methyl dihydrogen phosphate``,
    ``dimethyl hydrogen phosphate``, ``trimethyl phosphate``), phosphonate esters
    (``dimethyl methylphosphonate`` — one P-C bond), phosphinate esters
    (``methyl dimethylphosphinate`` — two P-C bonds), and phosphite triesters
    (``triethyl phosphite`` — trivalent P, no P=O). The acidic H that remains on a
    partial ester IS cited (``hydrogen``/``dihydrogen``, /; the older
    code dropped it and emitted the anion name ``dimethyl phosphate`` for a
    NEUTRAL diester — a wrong (charged) structure 's skeleton block does
    not catch. Fail-closed (``None``) off the clean neutral single-P shape, or if
    any owner / C-ligand is not spellable or the name would not cover every atom.
    Thio (P=S, P-S) is out of scope here -> ``None`` (honest defer).
    """
    if mol is None or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    p = mol.GetAtomWithIdx(phosphorus_idx)
    if p.GetSymbol() != 'P' or p.GetFormalCharge() != 0 or p.IsInRing():
        return None

    accounted = {phosphorus_idx}
    ester_oxygens: List[int] = []
    oh_count = 0
    c_ligands: List[int] = []
    dbl_oxo = 0
    for b in p.GetBonds():
        nb = b.GetOtherAtom(p)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # P=C / P#N / P=S etc. -> defer
        if sym == 'C':
            c_ligands.append(nb.GetIdx())
            continue
        if sym != 'O' or nb.GetFormalCharge() != 0:
            return None                         # P-S / P-N / P-O-P bridge -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != phosphorus_idx]
        if not others and nb.GetTotalNumHs() >= 1:
            oh_count += 1
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() in ('C', 'S'):
            # C: an ordinary ester owner (-O-CR). S: a sulfenyl-ester owner
            # (-O-S-R, tail #16 tris(dodecylsulfanyl) phosphite); both are
            # named as a substituent token by _p_ester_owner_group. A P-O-P
            # bridge (others[0]=='P') or O-N still defers.
            ester_oxygens.append(nb.GetIdx())
            accounted.add(nb.GetIdx())
        else:
            return None                         # O bridging to non-C/S / P-O-P -> defer

    if not ester_oxygens:
        return None                             # a free acid, not an ester
    if dbl_oxo > 1:
        return None

    stem = _P_ACID_STEM.get((dbl_oxo == 1, len(c_ligands)))
    if stem is None:
        return None
    if len(c_ligands) > 0 and stem == "phosphite":
        return None

    # Name each ester owner via the recursive substituent namer.
    owner_tokens: List[str] = []
    for o_idx in ester_oxygens:
        got = _p_ester_owner_group(mol, o_idx, phosphorus_idx)
        if got is None:
            return None
        token, frag = got
        owner_tokens.append(token)
        accounted |= frag
    owner_text = _assemble_p_owner_text(owner_tokens)

    # Phosphon/phosphin-ate: the P-C ligand(s) become the stem's carbon prefix.
    stem_prefix = ""
    if c_ligands:
        from ..assembly.substituent_enumerator import name_substituent
        c_tokens: List[str] = []
        for c_idx in c_ligands:
            frag: set = set()
            stack = [c_idx]
            while stack:
                a = stack.pop()
                if a in frag:
                    continue
                frag.add(a)
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nb.GetIdx()
                    if j == phosphorus_idx:
                        continue
                    if j not in frag:
                        stack.append(j)
            try:
                # allow_mancude so a complex P-C ligand (a cyano/isocyano-bearing
                # carbon, tail #17) is spelled instead of falling to the
                # 'substituent' sentinel; a plain alkyl ligand is byte-identical.
                tok = name_substituent(mol, frozenset(frag), c_idx,
                                       allow_mancude=True)
            except Exception:  # noqa: BLE001
                return None
            if not tok or not isinstance(tok, str) or tok == 'substituent':
                return None
            c_tokens.append(tok)
            accounted |= frag
        stem_prefix = _assemble_p_owner_text(c_tokens)

    # Every heavy atom must be accounted for, or this is not a whole-molecule name.
    if accounted != set(range(mol.GetNumAtoms())):
        return None

    hyd = _HYDROGEN_MULT.get(oh_count)
    if hyd is None:
        return None                             # >2 free OH is not an ester shape
    pieces = [owner_text]
    if hyd:
        pieces.append(hyd)
    pieces.append(f"{stem_prefix}{stem}")
    return " ".join(pieces)


def name_phosphate_ester_anion(mol, phosphorus_idx: int) -> Optional[str]:
    """Functional-class name for the ANION of a P-oxoacid acid-ester.

    Sibling of:func:`name_phosphate_ester` for the deprotonated form. The
    protonation word is derived IN PLACE from the surviving free ``-OH`` count
    (``_HYDROGEN_MULT``): the ``[O-]`` carry the charge and are NOT counted as
    hydrogens, so a monoester dianion (0 OH) -> ``dodecyl phosphate``, a monoanion
    (1 OH) -> ``dodecyl hydrogen phosphate``, a diester monoanion (0 OH, 2 owners)
    -> ``diethyl phosphate``. Never neutralize-then-rename . Requires at
    least one terminal ``[O-]`` (else -> ``None``, the NEUTRAL producer owns that
    shape) and that the molecule's only charges are those ``[O-]``. Fail-closed
    (``None``) off the clean single-P ester-anion shape (thio P=S/P-S, ring P,
    P-N, P-O-P bridge, no ester owner, unspellable owner, incomplete coverage).
    """
    if mol is None:
        return None
    p = mol.GetAtomWithIdx(phosphorus_idx)
    if p.GetSymbol() != 'P' or p.GetFormalCharge() != 0 or p.IsInRing():
        return None

    accounted = {phosphorus_idx}
    ester_oxygens: List[int] = []
    oh_count = 0
    anion_count = 0
    c_ligands: List[int] = []
    dbl_oxo = 0
    anion_oxygens: Set[int] = set()
    for b in p.GetBonds():
        nb = b.GetOtherAtom(p)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and sym == 'O':
            dbl_oxo += 1
            accounted.add(nb.GetIdx())
            continue
        if bt != Chem.BondType.SINGLE:
            return None                         # P=C / P#N / P=S -> defer
        if sym == 'C':
            c_ligands.append(nb.GetIdx())
            continue
        if sym != 'O':
            return None                         # P-S / P-N -> defer
        others = [x for x in nb.GetNeighbors() if x.GetIdx() != phosphorus_idx]
        if not others and nb.GetFormalCharge() < 0:
            anion_count += 1                    # terminal [O-]
            anion_oxygens.add(nb.GetIdx())
            accounted.add(nb.GetIdx())
        elif not others and nb.GetTotalNumHs() >= 1 and nb.GetFormalCharge() == 0:
            oh_count += 1                       # terminal -OH
            accounted.add(nb.GetIdx())
        elif len(others) == 1 and others[0].GetSymbol() in ('C', 'S') \
                and nb.GetFormalCharge() == 0:
            ester_oxygens.append(nb.GetIdx())   # -O-C / -O-S ester owner
            accounted.add(nb.GetIdx())
        else:
            return None                         # P-O-P bridge / charged owner O -> defer

    if anion_count < 1:
        return None                             # neutral -> name_phosphate_ester owns it
    if not ester_oxygens:
        return None                             # bare inorganic anion, not an ester
    if dbl_oxo > 1:
        return None
    # 0-wrong: ONLY the counted terminal [O-] may carry charge. A net-sum
    # check would pass a charge-separated zwitterion (e.g. an O-phospho
    # amino-acid) whose remote +/- cancel; scan per-atom so any other charged
    # centre fails closed before a fragment reaches the substituent namer.
    for a in mol.GetAtoms():
        if a.GetFormalCharge() != 0 and a.GetIdx() not in anion_oxygens:
            return None

    stem = _P_ACID_STEM.get((dbl_oxo == 1, len(c_ligands)))
    if stem is None or (len(c_ligands) > 0 and stem == "phosphite"):
        return None

    owner_tokens: List[str] = []
    for o_idx in ester_oxygens:
        got = _p_ester_owner_group(mol, o_idx, phosphorus_idx)
        if got is None:
            return None
        token, frag = got
        owner_tokens.append(token)
        accounted |= frag
    owner_text = _assemble_p_owner_text(owner_tokens)

    stem_prefix = ""
    if c_ligands:
        from ..assembly.substituent_enumerator import name_substituent
        c_tokens: List[str] = []
        for c_idx in c_ligands:
            frag = set()
            stack = [c_idx]
            while stack:
                a = stack.pop()
                if a in frag:
                    continue
                frag.add(a)
                for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nb.GetIdx()
                    if j == phosphorus_idx:
                        continue
                    if j not in frag:
                        stack.append(j)
            try:
                tok = name_substituent(mol, frozenset(frag), c_idx, allow_mancude=True)
            except Exception:  # noqa: BLE001
                return None
            if not tok or not isinstance(tok, str) or tok == 'substituent':
                return None
            c_tokens.append(tok)
            accounted |= frag
        stem_prefix = _assemble_p_owner_text(c_tokens)

    if accounted != set(range(mol.GetNumAtoms())):
        return None
    hyd = _HYDROGEN_MULT.get(oh_count)
    if hyd is None:
        return None                             # >2 free OH is not an ester shape
    pieces = [owner_text]
    if hyd:
        pieces.append(hyd)
    pieces.append(f"{stem_prefix}{stem}")
    return " ".join(pieces)


def get_phosphanyl_prefix(mol, phosphorus_idx: int, exclude_atoms: set = None) -> Optional[str]:
    """
    Generate IUPAC phosphanyl prefix string.

    Used when phosphorus is a substituent on a parent chain/ring.
    Characterizes C/c neighbors of P (excluding parent attachment atoms)
    and builds the prefix.

    Examples:
        -PPh2 -> "diphenylphosphanyl"
        -PMe2 -> "dimethylphosphanyl"
        -PMePhPh -> not typical, but "methyldiphenylphosphanyl"
        -PMe -> "methylphosphanyl"

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom
        exclude_atoms: Optional set of atom indices to exclude from substituent
            counting (typically the parent ring/chain atoms that P is bonded to).

    Returns:
        Prefix string like "diphenylphosphanyl", or None if no valid substituents
    """
    if exclude_atoms is None:
        exclude_atoms = set()

    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Get carbon neighbors, excluding parent attachment atoms
    neighbors = [n for n in phosphorus.GetNeighbors()
                 if n.GetSymbol() == 'C' and n.GetIdx() not in exclude_atoms]

    if len(neighbors) == 0:
        return "phosphanyl"  # Bare -PH2

    # Characterize each substituent
    sub_names = []
    for neighbor in neighbors:
        result = _characterize_substituent(mol, neighbor.GetIdx(), {phosphorus_idx})
        if result is None:
            return None  # Unrecognized substituent
        _, name = result
        sub_names.append(name)

    if not sub_names:
        return None

    # Build substituent string with multipliers and alphabetical ordering
    prefix = _build_substituent_string(sub_names)
    # / lambda-convention: a P at a NON-STANDARD bonding number (a
    # 5-bonded organyl phosphorane -PR4, e.g. -P(C6H5)4) carries the lambda
    # descriptor between the substituent prefixes and 'phosphanyl':
    # 'tetraphenyl-λ5-phosphanyl'. Standard-valence (bonding number 3) P returns
    # None here and stays plain 'phosphanyl'. Routed through the shared LAMBDA
    # constant (resolved lambda-convention fix). The all-H -PH4 case returns
    # early above (no C neighbours) and gets its λ from name_phosphanyl_substituent.
    lam = nonstandard_bonding_number(mol, phosphorus_idx)
    if lam is not None:
        return f"{prefix}-{LAMBDA}{lam}-phosphanyl"
    return f"{prefix}phosphanyl"


def name_phosphanyl_substituent(mol, frag_atoms, attach_idx: int) -> Optional[str]:
    """: name a phosphorus-rooted substituent on an acyclic parent.

    ``-PH2`` -> ``phosphanyl``; ``-PR2`` -> ``dialkyl/diarylphosphanyl`` (via
    :func:`get_phosphanyl_prefix`). Wired into the chain/acid substituent path
    (``name_substituent`` Tier 1.93) so a trivalent-P substituent on a carbon
    chain is cited as the ``phosphanyl`` prefix rather than dropped.

    Fail-closed collision guard load-bearing case): fires ONLY when
    ``attach_idx`` is a NEUTRAL, radical-free phosphorus whose every in-fragment
    heavy neighbour is a single-bonded carbon (organyl) — so a phosphoryl /
    phosphonic ``P=O`` (bonding number 5, but with an O neighbour) is EXCLUDED
    and left to the oxoacid subsystem. Standard-valence (bonding number 3) only
    in this task; the λ5 hydride branch is added in Task 3.

    Non-standard valence: only the all-H λ-hydride (-PH4) is named here as
    ``lambda5-phosphanyl`` /; house ASCII ``lambda5`` with NO
    internal locant on a mononuclear prefix). Any other non-standard shape
    (organyl λ5, phosphoryl/phosphonic P=O) fails closed.

    Returns None (caller falls through, fail-closed) for any non-P attachment,
    a P bearing a heteroatom / multiple bond, or an unrecognised organyl ligand.
    """
    if attach_idx is None:
        return None
    frag_set = set(frag_atoms)
    if attach_idx not in frag_set:
        return None
    p = mol.GetAtomWithIdx(attach_idx)
    if (p.GetSymbol() != 'P' or p.GetFormalCharge() != 0
            or p.GetNumRadicalElectrons() != 0):
        return None
    # Every heavy neighbour is either the parent attachment (out of fragment) or
    # an in-fragment single-bonded carbon; any O/N/S neighbour or multiple bond
    # (i.e. a phosphoryl/phosphonic/phosphine-oxide P=O) declines here.
    for b in p.GetBonds():
        nb = b.GetOtherAtom(p)
        if nb.GetIdx() not in frag_set:
            continue  # bond to the parent structure
        if nb.GetSymbol() != 'C' or b.GetBondType() != Chem.BondType.SINGLE:
            return None
    exclude = set(range(mol.GetNumAtoms())) - frag_set
    base = get_phosphanyl_prefix(mol, attach_idx, exclude_atoms=exclude)
    if base is None:
        return None
    lam = nonstandard_bonding_number(mol, attach_idx)
    if lam is None:
        return base  # standard-valence phosphanyl (bonding number 3)
    # Non-standard bonding number: name ONLY the all-H λ-hydride (-PH4) ->
    # 'lambda5-phosphanyl' /. Organyl λ5 / any P=O declines.
    from ..perception.lambda_hydride import is_lambda_hydride_phosphorus
    if is_lambda_hydride_phosphorus(mol, attach_idx):
        return f"{LAMBDA}{lam}-{base}"
    return None


# --- a phase /: a mononuclear P/As/Sb ACYL
# group cited as a SUBSTITUENT prefix on a senior parent -----------------------
_ACYL_CHALCO_DOUBLE = {"O": "oxo", "S": "thio", "Se": "seleno", "Te": "telluro"}
# alkyl root -> -oxy residual ester residual on an acid position)
_ACYL_OXY = {"methyl": "methoxy", "ethyl": "ethoxy", "propyl": "propoxy",
             "butyl": "butoxy", "phenyl": "phenoxy"}


def _acyl_residual_prefix(mol, resid_atom, central_idx: int) -> Optional[str]:
    """Name an acid-derived residual ligand on a P/As/Sb acyl centre
    : ``-OH`` -> ``hydroxy``, ``-SH`` -> ``sulfanyl``,
    ``-O-alkyl`` -> ``<alkyl>oxy``. Fail-closed ``None`` for any other shape, which
    leaves the whole acyl group to a more capable path rather than mis-spelling it.
    """
    if resid_atom.GetFormalCharge() != 0:
        return None
    sym = resid_atom.GetSymbol()
    if sym == "O" and resid_atom.GetTotalNumHs() == 1 and resid_atom.GetDegree() == 1:
        return "hydroxy"
    if sym == "S" and resid_atom.GetTotalNumHs() == 1 and resid_atom.GetDegree() == 1:
        return "sulfanyl"
    if sym == "O" and resid_atom.GetTotalNumHs() == 0 and resid_atom.GetDegree() == 2:
        others = [n for n in resid_atom.GetNeighbors()
                  if n.GetIdx() != central_idx]
        if len(others) != 1 or others[0].GetSymbol() != "C":
            return None
        r = _characterize_substituent(mol, others[0].GetIdx(),
                                      {resid_atom.GetIdx()})
        if r is None:
            return None
        alkyl = r[1]
        return _ACYL_OXY.get(alkyl, f"{alkyl}oxy")
    return None


def name_acyl_prefix_substituent(mol, frag_atoms, attach_idx: int) -> Optional[str]:
    """: name a mononuclear P/As/Sb ACYL group carried as a
    SUBSTITUENT prefix on a senior parent, CONSUMING the shared
    ``acyl_prefix_for`` table (never re-spelling the acyl base).

    The central atom (``attach_idx``) is a neutral, radical-free P/As/Sb that bears
    EXACTLY ONE terminal =E' multiple bond -- ``=O`` -> ``oxo``, ``=S`` -> ``thio``,
    ``=NH`` -> ``imido``, ``|=N`` (triple) -> ``nitrido`` -- and attaches to the
    parent by a SINGLE bond leaving the fragment (the yl-bond). The acyl base
    (``-oryl`` / ``-onoyl`` / ``-inoyl``, or its thio/imido/nitrido sibling) is
    chosen by the SKELETAL count = the number of in-fragment P-C(organyl) / P-H
    ligands (0 -> ``-oryl``, 1 -> ``-onoyl``, 2 -> ``-inoyl``), exactly the
    ``R,R'-E(=E')OH`` acid skeleton of where R/R' = H or organyl.

    Those organyl ligands ARE the ``-inoyl`` base's own defining substituents, so
    they are cited WITHOUT a P-locant (BB L36080 ``methyl(phenyl)arsinoyl``, L39255
    ``dimethylphosphinothioyl``). Any remaining in-fragment heteroatom ligand
    (``-OH`` -> ``hydroxy``, ``-SH`` -> ``sulfanyl``, ``-OR`` -> ``<R>oxy``) is an
    acid-derived residual cited as a front prefix (BB L39242 ``hydroxyarsoryl``).

    Returns ``None`` (caller falls through, fail-closed) for a trivalent
    phosphanyl / arsanyl (no =E'), a multiplicative >P(=E)< bridge (>1 attachment,
    named by ``rules/multiplicative.py``), an untabled cell, or a residual it cannot
    name. The top-level OPSIN round-trip gate voids any mis-spelling, so a wrong
    skeletal / residual guess degrades to an abstention, never a wrong structure.
    """
    if attach_idx is None:
        return None
    frag_set = set(frag_atoms)
    if attach_idx not in frag_set:
        return None
    central = mol.GetAtomWithIdx(attach_idx)
    if (central.GetSymbol() not in ("P", "As", "Sb")
            or central.GetFormalCharge() != 0
            or central.GetNumRadicalElectrons() != 0):
        return None

    # Exactly one bond leaves the fragment -> the single-attachment yl-bond. The
    # multiplicative >P(=E)< bridge (2 attachments) is named elsewhere.
    out_nbrs = [n for n in central.GetNeighbors() if n.GetIdx() not in frag_set]
    if len(out_nbrs) != 1:
        return None

    token = None                  # oxo / thio / imido / nitrido
    skeletal_c: List[int] = []    # in-frag single-bonded organyl carbons
    residuals: List[str] = []     # residual prefix names (hydroxy / methoxy /...)
    for b in central.GetBonds():
        nb = b.GetOtherAtom(central)
        if nb.GetIdx() not in frag_set:
            continue              # the parent attachment (already accounted for)
        bt = b.GetBondType()
        sym = nb.GetSymbol()
        if bt == Chem.BondType.DOUBLE and nb.GetDegree() == 1:
            if sym in _ACYL_CHALCO_DOUBLE and token is None:
                token = _ACYL_CHALCO_DOUBLE[sym]
                continue
            if sym == "N" and nb.GetTotalNumHs() == 1 and token is None:
                token = "imido"           # =NH
                continue
            return None
        if (bt == Chem.BondType.TRIPLE and sym == "N"
                and nb.GetDegree() == 1 and token is None):
            token = "nitrido"             # |=N
            continue
        if bt != Chem.BondType.SINGLE:
            return None
        # single-bonded ligand: skeletal organyl (C) or acid-derived residual
        if sym == "C":
            skeletal_c.append(nb.GetIdx())
            continue
        resid = _acyl_residual_prefix(mol, nb, central.GetIdx())
        if resid is None:
            return None
        residuals.append(resid)
    if token is None:
        return None               # trivalent phosphanyl/arsanyl -> not an acyl group

    skeletal = len(skeletal_c) + central.GetTotalNumHs()
    # RETAINED prefixes own the skeletal-0 -E(=O)(OH)2 /
    # -P(=S)(OH)2 shapes: azono / phosphono / arsono / stibono / thiophosphono /
    # trithiophosphono are the PINs there, not the generic 'dihydroxy...oryl'
    # synonym this builder would form. Decline so the retained path keeps
    # ownership (never regress a retained prefix to a nonPIN). Any residual that is
    # NOT a plain -OH/-SH (e.g. an -OR ester) is outside that retained set, so the
    # acyl builder still owns it.
    if skeletal == 0 and residuals and all(
            r in ("hydroxy", "sulfanyl") for r in residuals):
        return None
    from .functional_replacement import acyl_prefix_for
    acyl = acyl_prefix_for(central.GetSymbol(), token, skeletal)
    if acyl is None:
        return None

    organyl_names: List[str] = []
    for c_idx in skeletal_c:
        r = _characterize_substituent(mol, c_idx, {attach_idx})
        if r is None:
            return None
        organyl_names.append(r[1])
    front_names = residuals + organyl_names
    if not front_names:
        return acyl
    prefix = _build_substituent_string(front_names)
    if prefix is None:
        return None
    return f"{prefix}{acyl}"


def _reachable_carbon_fragment(mol, start_c: int, block_o: int) -> List[int]:
    """Atoms reachable from ``start_c`` without crossing ``block_o`` (the bridging
    ester oxygen). The carbon-side subgraph of an ``-O-C…`` phosphoester branch."""
    frag: Set[int] = set()
    stack = [start_c]
    while stack:
        a = stack.pop()
        if a in frag or a == block_o:
            continue
        frag.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j != block_o and j not in frag:
                stack.append(j)
    return sorted(frag)


def name_phosphoanhydride_oxy_substituent(
        mol, o_idx: int, from_idx: int, _depth: int = 0) -> Optional[str]:
    """ method (1): name an ``-O-P(=O)(…)…`` phosphoanhydride subgraph as a
    recursive phosphoryl-oxy substituent prefix.

    ``o_idx`` is the ester/bridging oxygen bonded to the parent atom ``from_idx``;
    the oxygen's other neighbour must be a phosphorus. Returns a substituent string
    ENDING in ``oxy`` — ``'phosphonooxy'`` for a terminal ``-O-P(=O)(OH)2``, or the
    nested ``'[<branches>phosphoryl]oxy'`` for a P-O-P(-O-P…) anhydride bridge —
    or None (fail-closed) for any P outside the neutral mono-oxo phosphoryl class
    (a P-C phosphonate, a no-oxo phosphite, a charged/oxido/radical P, etc.), which
    carry their own nomenclature.

    This is the valid SYSTEMATIC form the Blue Book lists as alternative (1) under
     (`the Blue Book Blue Book`); the PIN (method 2) uses the
    ``…diphosphoxan-1-yl`` skeletal-replacement parent and is a future PIN-tier
    build. Emitted only on the best-effort path, where a valid systematic name is
    preferred over silence. 0-wrong is preserved by the top-level /OPSIN
    round-trip gate. Full record: internal notes.
    """
    if _depth > 12:
        return None
    o = mol.GetAtomWithIdx(o_idx)
    ps = [n for n in o.GetNeighbors()
          if n.GetIdx() != from_idx and n.GetSymbol() == 'P']
    if len(ps) != 1:
        return None
    p = ps[0]
    p_idx = p.GetIdx()
    if p.GetFormalCharge() != 0 or p.GetNumRadicalElectrons() != 0:
        return None
    oxo = 0
    branches: List[str] = []
    for nb in p.GetNeighbors():
        if nb.GetIdx() == o_idx:
            continue
        bond = mol.GetBondBetweenAtoms(p_idx, nb.GetIdx())
        if nb.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE:
            oxo += 1
            continue
        if (nb.GetSymbol() == 'O'
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nb.GetFormalCharge() == 0):
            o2 = nb
            heavy = [x for x in o2.GetNeighbors()
                     if x.GetIdx() != p_idx and x.GetAtomicNum() > 1]
            if not heavy:
                if o2.GetTotalNumHs() >= 1:
                    branches.append('hydroxy')          # -OH
                else:
                    return None                          # bare -O- / oxido: decline
            elif len(heavy) == 1 and heavy[0].GetSymbol() == 'P':
                sub = name_phosphoanhydride_oxy_substituent(
                    mol, o2.GetIdx(), p_idx, _depth + 1)   # recurse P-O-P
                if sub is None:
                    return None
                branches.append(sub)
            elif len(heavy) == 1 and heavy[0].GetSymbol() == 'C':
                from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
                from ..assembly.substituent_naming import name_substituent_fragment
                c = heavy[0]
                frag = _reachable_carbon_fragment(mol, c.GetIdx(), o2.GetIdx())
                yl = name_substituent_fragment(mol, frag, c.GetIdx(), [])
                if not yl:
                    return None
                alk = alkoxy_prefix_from_substituent(yl)
                if not alk:
                    return None
                branches.append(alk)
            else:
                return None                              # -O-N etc.: decline
        else:
            return None                                  # P-C / P-N / no-O ligand
    if oxo != 1:
        return None                                      # need exactly one P=O
    # Terminal -O-P(=O)(OH)2 -> contracted 'phosphonooxy'.
    if sorted(branches) == ['hydroxy', 'hydroxy']:
        return 'phosphonooxy'
    # General phosphoryl bridge: cite branches in alphanumerical order,
    # then 'phosphoryl', wrapped and attached via the linking O -> '...oxy'
    # method 1). Enclose each branch that is complex (carries a locant or
    # its own enclosure); a bare 'hydroxy' stays unenclosed.
    from ..assembly.naming_utils import (_is_fully_enclosed, alpha_sort_key,
                                         apply_enclosing_marks)
    ordered = sorted(branches, key=alpha_sort_key)

    # The marks ESCALATE past any inside the branch and inside the body
    #, the Blue Book: nesting '{[({})]}'). A raw '(...)' /
    # '[...]' put '[' directly around '[' for a nested P-O-P bridge
    # ('[hydroxy[hydroxy(phosphonooxy)phosphoryl]oxyphosphoryl]oxy', where the
    # inner composed prefix was not even enclosed as a unit).
    def _enc(tok: str) -> str:
        if tok == 'hydroxy' or _is_fully_enclosed(tok):
            return tok
        return apply_enclosing_marks(tok, -1)

    body = ''.join(_enc(t) for t in ordered) + 'phosphoryl'
    return apply_enclosing_marks(body, -1) + 'oxy'


_PHOSPHOXANE_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                     7: 'hepta', 8: 'octa', 9: 'nona', 10: 'deca'}


def _walk_phosphoanhydride_chain(mol, o_idx: int, from_idx: int):
    """From the bridging oxygen ``o_idx`` (bonded to parent ``from_idx``) walk the
    LINEAR P-O-P… phosphoanhydride chain. Return the ordered list of P atom indices
    ``[P1, P2, …]`` (P1 bonded to ``o_idx``), or None if it is not a single
    unbranched neutral chain of phosphorus atoms bridged by shared oxygens."""
    o = mol.GetAtomWithIdx(o_idx)
    ps = [n for n in o.GetNeighbors()
          if n.GetIdx() != from_idx and n.GetSymbol() == 'P']
    if len(ps) != 1:
        return None
    p_list = []
    prev_o = o_idx
    cur_p = ps[0].GetIdx()
    seen = set()
    while True:
        if cur_p in seen:
            return None                                  # cycle guard
        seen.add(cur_p)
        p = mol.GetAtomWithIdx(cur_p)
        if (p.GetSymbol() != 'P' or p.GetFormalCharge() != 0
                or p.GetNumRadicalElectrons() != 0):
            return None
        p_list.append(cur_p)
        next_p = next_o = None
        for nb in p.GetNeighbors():
            if nb.GetIdx() == prev_o or nb.GetSymbol() != 'O':
                continue
            b = mol.GetBondBetweenAtoms(cur_p, nb.GetIdx())
            if b.GetBondType() != Chem.BondType.SINGLE or nb.GetFormalCharge() != 0:
                continue
            nb_heavy = [x for x in nb.GetNeighbors() if x.GetAtomicNum() > 1]
            other_p = [x for x in nb_heavy
                       if x.GetIdx() != cur_p and x.GetSymbol() == 'P']
            if len(nb_heavy) == 2 and len(other_p) == 1:
                if next_p is not None:
                    return None                          # branched P-O-P: not linear
                next_p, next_o = other_p[0].GetIdx(), nb.GetIdx()
        if next_p is None:
            break
        prev_o, cur_p = next_o, next_p
    return p_list


def name_phosphoxane_oxy_substituent(
        mol, o_idx: int, from_idx: int) -> Optional[str]:
    """ method (2) — the PIN: name an ``-O-P(=O)(…)…`` phosphoanhydride
    bridge as the skeletal-replacement ``…phosphoxan-1-yl`` parent, returning
    ``'(<subs>[n]phosphoxan-1-yl)oxy'``.

    ``diphosphoxane``/``triphosphoxane``/… are preselected parent hydrides,
    `the Blue Book Blue Book,2971`): a chain of n phosphorus atoms bridged by
    (n-1) oxygens, numbered P at the odd locants 1,3,5,…,(2n-1). Each P carries its
    ``=O`` (``oxo``), its ``-OH`` (``hydroxy``), any ``-O-R`` ester (``{R}oxy``),
    and a ``λ⁵`` designator (all P are pentavalent). The attachment is at.

    This is the PREFERRED form (method 2, the PIN) over the recursive-phosphoryl
    method-1 (:func:`name_phosphoanhydride_oxy_substituent`). Returns None
    (fail-closed) for any P outside the neutral mono-oxo phosphoryl anhydride class
    (P-C phosphonate, no-oxo phosphite, charged/oxido P, branched P-O-P), so the
    caller can fall back to method-1. RT-verified through OPSIN 2.9.0 for di/tri/
    tetraphosphoxane. Record: internal notes.
    """
    p_list = _walk_phosphoanhydride_chain(mol, o_idx, from_idx)
    if not p_list or len(p_list) < 2:
        return None                                      # single P -> phosphonooxy/method-1
    backbone_o = set()
    # backbone O = o_idx (attach) + every O bridging two chain P's
    backbone_o.add(o_idx)
    p_set = set(p_list)
    for i in range(len(p_list) - 1):
        pa, pb = p_list[i], p_list[i + 1]
        for nb in mol.GetAtomWithIdx(pa).GetNeighbors():
            if (nb.GetSymbol() == 'O'
                    and mol.GetBondBetweenAtoms(pb, nb.GetIdx()) is not None):
                backbone_o.add(nb.GetIdx())

    hydroxy_loc, oxo_loc, lam_loc = [], [], []
    ester_subs = []                                      # (locant, "{R}oxy")
    for k, p_idx in enumerate(p_list):
        locant = 2 * k + 1                               # P at 1,3,5,…
        lam_loc.append(locant)
        p = mol.GetAtomWithIdx(p_idx)
        oxo_here = 0
        for nb in p.GetNeighbors():
            if nb.GetIdx() in backbone_o:
                continue
            b = mol.GetBondBetweenAtoms(p_idx, nb.GetIdx())
            if nb.GetSymbol() == 'O' and b.GetBondType() == Chem.BondType.DOUBLE:
                oxo_here += 1
                oxo_loc.append(locant)
                continue
            if (nb.GetSymbol() == 'O' and b.GetBondType() == Chem.BondType.SINGLE
                    and nb.GetFormalCharge() == 0):
                heavy = [x for x in nb.GetNeighbors()
                         if x.GetIdx() != p_idx and x.GetAtomicNum() > 1]
                if not heavy and nb.GetTotalNumHs() >= 1:
                    hydroxy_loc.append(locant)
                elif len(heavy) == 1 and heavy[0].GetSymbol() == 'C':
                    from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
                    from ..assembly.substituent_naming import name_substituent_fragment
                    frag = _reachable_carbon_fragment(mol, heavy[0].GetIdx(),
                                                      nb.GetIdx())
                    yl = name_substituent_fragment(mol, frag, heavy[0].GetIdx(), [])
                    if not yl:
                        return None
                    alk = alkoxy_prefix_from_substituent(yl)
                    if not alk:
                        return None
                    ester_subs.append((locant, alk))
                else:
                    return None                          # -O-P off-chain / -O-N / oxido
            else:
                return None                              # P-C phosphonate, etc.
        if oxo_here != 1:
            return None                                  # need exactly one P=O per P

    n = len(p_list)
    parent = f"{_PHOSPHOXANE_MULT.get(n, '')}phosphoxane"
    if not _PHOSPHOXANE_MULT.get(n):
        return None
    from ..assembly.naming_utils import get_multiplier_prefix

    def _locant_prefix(locants, word):
        if not locants:
            return None
        locs = ','.join(str(x) for x in sorted(locants))
        mult = get_multiplier_prefix(len(locants), word)
        return f"{locs}-{mult}{word}"

    # detachable prefixes in alphanumerical order: esters ({R}oxy) and
    # hydroxy interleave by name; oxo comes after hydroxy ('h' < 'o').
    from ..assembly.naming_utils import enclose_if_compound
    detach = []
    for loc, tok in ester_subs:
        #: a COMPOUND ester branch (its own locant/brackets, e.g. an
        # amino-acyl-decorated alkoxy) must be enclosed before the diphosphoxane
        # locant is prepended -- otherwise the leading digit of the branch
        # collides with the prepended locant ('3-4-[...]butoxy', OPSIN-unparseable).
        # A SIMPLE ester ('methoxy') stays bare.
        detach.append((tok, f"{loc}-{enclose_if_compound(tok)}"))
    hp = _locant_prefix(hydroxy_loc, 'hydroxy')
    if hp:
        detach.append(('hydroxy', hp))
    op = _locant_prefix(oxo_loc, 'oxo')
    from ..assembly.naming_utils import alpha_sort_key
    detach_sorted = [s for _, s in sorted(detach, key=lambda t: alpha_sort_key(t[0]))]
    pieces = detach_sorted[:]
    if op:
        pieces.append(op)                                # oxo last (after hydroxy)
    lam = ','.join(f"{x}{LAMBDA}5" for x in sorted(lam_loc))
    # Elide the terminal 'e' of the parent before the '-1-yl' ending
    # (diphosphoxane -> diphosphoxan-1-yl), per / standard '-yl' elision.
    stem = parent[:-1] if parent.endswith('e') else parent
    body = '-'.join(pieces) if pieces else ''
    core = f"{body}-{lam}-{stem}-1-yl" if body else f"{lam}-{stem}-1-yl"
    return f"({core})oxy"


def get_phosphorus_prefix(fg_name: str) -> Optional[str]:
    """
    Get prefix form for phosphorus functional groups.

    Returns:
        Prefix string, or None if group uses functional class naming
    """
    PHOSPHORUS_PREFIXES = {
        "phosphonic_acid": "phosphono",
        "phosphinic_acid": "phosphino",
        # Phosphines use phosphanyl prefix when P is a substituent
        "tertiary_phosphine": "phosphanyl",
        "secondary_phosphine": "phosphanyl",
        "primary_phosphine": "phosphanyl",
        # These use functional class naming, no simple prefix:
        "phosphine_oxide": None,
        "phosphate_triester": None,
        "phosphate_diester": None,
        "phosphate_monoester": None,
    }
    return PHOSPHORUS_PREFIXES.get(fg_name)


def carbon_free_phospho_prefix(mol, sub_atoms, attach_idx: int) -> Optional[str]:
    """A CARBON-FREE ``-P(=O)(OH)2`` / ``-P(=O)(O-)2`` substituent fragment ->
    its RETAINED detachable prefix.

    ``phosphono``, the neutral di-hydroxy group) and ``phosphonato``
     preselected prefix for the ``-P(O)(O-)2`` di-anion, BB:41213).

    Why this exists: the recursive substituent path (``name_substituent_fragment``
    Step 4) names a fragment by round-tripping it through the WHOLE-molecule
    namer, but the bare P-oxo fragment ``O=P(O)O`` has NO carbon, so
    ``name_compound`` classifies it ``inorganic compound (not supported)`` and
    the fragment was DROPPED (substituent_recursion_depth_exceeded). The neutral prefix is exactly the one the
    FG-on-parent-chain path already emits via
    ``get_phosphorus_prefix('phosphonic_acid')`` -> ``'phosphono'``; this restores
    it for the fragment shape the whole-molecule reject cannot see.

    Fail-closed (``None``) for anything that is not exactly this shape:
    attach atom is P; the fragment is carbon-free with a single P centre; that P
    carries exactly one ``=O`` and two single-bonded O; and the two single O are
    BOTH neutral hydroxyl (-> ``phosphono``) or BOTH oxido anions (->
    ``phosphonato``). A mixed/partial charge, an extra H on P (phosphinic), or a
    second P (anhydride) declines here and falls through to the existing guards.
    The top-level /OPSIN validity gate voids any non-RT composed name, so
    0-wrong holds regardless.
    """
    if attach_idx is None or attach_idx not in set(sub_atoms):
        return None
    sub_set = set(sub_atoms)
    p_atom = mol.GetAtomWithIdx(attach_idx)
    if p_atom.GetSymbol() != 'P':
        return None
    # Carbon-free, single-P fragment.
    for i in sub_set:
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if sym == 'C':
            return None
        if sym == 'P' and i != attach_idx:
            return None            # a second P (anhydride/chain) -> not this shape
    if p_atom.GetTotalNumHs() != 0:
        return None                # P-H present (phosphinic family) -> decline
    # The P's oxygen neighbours WITHIN the fragment: exactly one =O + two -O.
    from rdkit import Chem as _Chem
    dbl_o = []
    sgl_o = []
    other = []
    for nbr in p_atom.GetNeighbors():
        if nbr.GetIdx() not in sub_set:
            continue               # the free valence (bond to the parent chain)
        if nbr.GetSymbol() != 'O':
            return None            # any non-O substituent on P -> decline
        bt = mol.GetBondBetweenAtoms(attach_idx, nbr.GetIdx()).GetBondType()
        if bt == _Chem.BondType.DOUBLE:
            dbl_o.append(nbr)
        elif bt == _Chem.BondType.SINGLE:
            sgl_o.append(nbr)
        else:
            other.append(nbr)
    if other or len(dbl_o) != 1 or len(sgl_o) != 2:
        return None
    # The =O must be a plain oxo (no charge, no H).
    od = dbl_o[0]
    if od.GetFormalCharge() != 0 or od.GetTotalNumHs() != 0:
        return None
    # Classify the two single O: both neutral -OH, or both anionic -O(-).
    charges = [o.GetFormalCharge() for o in sgl_o]
    hcounts = [o.GetTotalNumHs() for o in sgl_o]
    # Each single O must be terminal (only bonded to P within the fragment).
    for o in sgl_o:
        heavy = [n for n in o.GetNeighbors() if n.GetIdx() != attach_idx]
        if heavy:
            return None            # an -O-R ester bridge, not a bare acid/anion
    if all(c == 0 and h >= 1 for c, h in zip(charges, hcounts)):
        return get_phosphorus_prefix("phosphonic_acid")          # 'phosphono'
    if all(c == -1 and h == 0 for c, h in zip(charges, hcounts)):
        # preselected anionic prefix (matches rules/ions.py usage).
        return "phosphonato"
    return None
