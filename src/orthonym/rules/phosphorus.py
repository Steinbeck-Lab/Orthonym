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

from typing import Optional, Tuple, List
from collections import deque, Counter
from rdkit import Chem

from ..assembly.naming_utils import get_alkyl_name
from .lambda_convention import nonstandard_bonding_number


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

    # Not aromatic: count carbons via BFS (same as old _count_alkyl_carbons)
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
                    # Only follow C-C bonds for simple alkyls
                    if neighbor.GetSymbol() == 'C':
                        queue.append(nbr_idx)

    if count == 0 or count > 10:
        return None

    return ("alkyl", get_alkyl_name(count))


def _count_alkyl_carbons(mol, start_idx: int, exclude: set) -> int:
    """
    Count carbon atoms in an alkyl group via BFS.

    Backward-compatible wrapper around _characterize_substituent().
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

    Per P-16.5.1.3 (May 2021 errata): For mononuclear parent hydrides with
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
    """
    counts = Counter(names)
    multiplier_map = {1: "", 2: "di", 3: "tri", 4: "tetra"}

    # Sort unique names alphabetically
    sorted_unique = sorted(counts.keys())

    parts = []
    for i, name in enumerate(sorted_unique):
        count = counts[name]
        multiplier = multiplier_map.get(count, "")
        if len(sorted_unique) >= 2 and i > 0:
            # P-16.5.1.3: second+ different substituents get parentheses
            if count > 1:
                parts.append(f"({multiplier}{name})")
            else:
                parts.append(f"({name})")
        else:
            # First substituent or only unique substituent: no enclosing marks
            parts.append(f"{multiplier}{name}")

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


# P-67 organo-oxoacid stems for the pnictogens, keyed by central element:
#   symbol -> (-onic stem, -inic stem)
# BB L36051-36054 gives all six as PRESELECTED names. Bismuth has no oxoacid
# analogue in the Blue Book, so the family stops at Sb. This table is what makes
# the two namers below element-generic instead of phosphorus-only -- an arsonic
# acid is a P-67 oxoacid, never a P-69 organometallic.
_PNICTOGEN_OXOACID_STEMS = {
    'P':  ('phosphonic', 'phosphinic'),
    'As': ('arsonic', 'arsinic'),
    'Sb': ('stibonic', 'stibinic'),
}


def _find_central_atom(mol, atoms: Tuple[int, ...], symbol: str) -> Optional[int]:
    """Index of the first ``symbol`` atom in a SMARTS match, else None."""
    for idx in atoms:
        if mol.GetAtomWithIdx(idx).GetSymbol() == symbol:
            return idx
    return None


def _name_pnictogen_onic_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R-E(=O)(OH)2 -> ``{R-yl}{stem} acid`` for E in {P, As, Sb}.

    The single shared implementation behind :func:`name_phosphonic_acid` and its
    arsenic/antimony analogues. Fail-closed (``None``) unless the one organyl is
    a clean simple alkyl/aryl, so a complex substituent defers to the generic
    path rather than risking a wrong name.
    """
    from .substituent_purity import pure_organyl_prefix_name  # lazy: avoid import cycle

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

    prefix = pure_organyl_prefix_name(mol, carbons[0].GetIdx(), central_idx)
    if prefix is None:
        return None                     # complex organyl -> defer (fail-closed)
    return f"{prefix}{stem} acid"


def _name_pnictogen_inic_acid(
    mol, match_atoms: Tuple[int, ...], symbol: str,
) -> Optional[str]:
    """R2E(=O)OH -> ``{R-yl}{R-yl}{stem} acid`` for E in {P, As, Sb}.

    Shared implementation behind :func:`name_phosphinic_acid` and its
    arsenic/antimony analogues. Mixed substituents get the P-16.5.1.3 enclosing
    marks from :func:`_build_substituent_string`, which yields the BB L36066
    verbatim PIN ``methyl(phenyl)arsinic acid`` for C6H5-As(CH3)(O)OH.

    ACCURACY FIX (Phase B): each substituent goes through
    ``pure_organyl_prefix_name``, the same purity gate the -onic path already
    used, instead of calling ``_characterize_substituent`` raw.
    ``_characterize_substituent`` counts the carbons of a non-aromatic subtree
    as a LINEAR chain, so a cyclic substituent was silently renamed --
    benzyl(methyl)phosphinic acid came back as 'heptyl(methyl)phosphinic acid'
    and cyclohexyl(methyl) as 'hexyl(methyl)', both a WRONG CONSTITUTION.  Those
    strings were caught downstream by the SELF-01 OPSIN check, but that gate
    fails OPEN when no JRE is present, so the refusal has to happen here.  Fixed
    for P, As and Sb together since all three share this code path.
    """
    from .substituent_purity import pure_organyl_prefix_name  # lazy: avoid import cycle

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
        name = pure_organyl_prefix_name(mol, neighbor.GetIdx(), central_idx)
        if name is None:
            return None                 # complex organyl -> defer (fail-closed)
        sub_names.append(name)

    return f"{_build_substituent_string(sub_names)}{stem} acid"


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
    ``phenylphosphonic acid`` (P-67.1.1.2). Phosphonic acid is a functional
    *parent* (HP(=O)(OH)2) whose central-atom H is substituted by the organyl
    group; the PIN is therefore ``{R-yl}phosphonic acid`` — NOT the
    parent-hydride-stem form ``{R-ane}phosphonic acid`` (``ethanephosphonic``),
    which the generic suffix assembler would otherwise emit (it correctly serves
    the genuine *suffix* acids like ``ethanesulfonic``).

    Mirrors :func:`name_phosphinic_acid`. Returns ``None`` (fail-closed) when the
    single organyl substituent is not a clean simple alkyl / aryl — the caller
    then defers to the generic path (no regression for complex parents).

    Thin wrapper over the element-generic :func:`_name_pnictogen_onic_acid`,
    which the arsenic/antimony analogues share.
    """
    return _name_pnictogen_onic_acid(mol, phosphonic_atoms, 'P')


def name_acyloxy_phosphonic_acid(mol) -> Optional[str]:
    """P-67.3.1 (BB L36999): a mixed acyl/phosphoric anhydride named
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
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    ps = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
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
    """P-68.3.2.3.1 (BB 39117): a carboxylic acid ``-C(=O)OH`` on a Group-15
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
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    if mol.GetRingInfo().NumRings() > 0:
        return None

    # Locate exactly ONE bare carboxyl carbon bonded to a pnictogen.
    carboxyl_c = hub = None
    n_carboxyls = 0
    for atom in mol.GetAtoms():
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
    pnic = [a for a in mol.GetAtoms() if a.GetSymbol() in _PNICTOGEN_ACID_STEM]
    if any(a.GetSymbol() != element for a in pnic):
        return None
    pnic_idxs = {a.GetIdx() for a in pnic}
    carboxyl_atoms = {carboxyl_c}
    for n in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
        if n.GetSymbol() == 'O':
            carboxyl_atoms.add(n.GetIdx())
    for a in mol.GetAtoms():
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
    # the '-carboxylic acid' suffix takes locant 1 (P-31.1.4, lowest locant).
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

    Thin wrapper over the element-generic :func:`_name_pnictogen_inic_acid`,
    which the arsenic/antimony analogues share.
    """
    return _name_pnictogen_inic_acid(mol, phosphinic_atoms, 'P')


def name_phosphate_ester(mol, phosphorus_idx: int) -> Optional[str]:
    """
    Name a phosphate ester using functional class nomenclature.

    Simple esters: "methyl phosphate", "dimethyl phosphate", "trimethyl phosphate"

    Args:
        mol: RDKit Mol object
        phosphorus_idx: Index of phosphorus atom

    Returns:
        Functional class name, or None if not a simple phosphate
    """
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)

    # Find O-C groups attached to P (not P=O, not P-OH)
    ester_oxygens = []
    for neighbor in phosphorus.GetNeighbors():
        if neighbor.GetSymbol() == 'O':
            # Check bond type - skip double-bonded oxygen (P=O)
            bond = mol.GetBondBetweenAtoms(phosphorus_idx, neighbor.GetIdx())
            if bond.GetBondTypeAsDouble() > 1.5:  # Double bond
                continue

            # Check if O is bonded to C (ester) vs H (acid)
            o_neighbors = [n for n in neighbor.GetNeighbors() if n.GetIdx() != phosphorus_idx]
            if o_neighbors and o_neighbors[0].GetSymbol() == 'C':
                ester_oxygens.append(neighbor.GetIdx())

    if not ester_oxygens:
        return None  # Not an ester

    # Get alkyl names for each ester group
    alkyl_names = []
    for o_idx in ester_oxygens:
        o_atom = mol.GetAtomWithIdx(o_idx)
        c_neighbors = [n for n in o_atom.GetNeighbors() if n.GetSymbol() == 'C']
        if not c_neighbors:
            continue
        c_neighbor = c_neighbors[0]
        carbon_count = _count_alkyl_carbons(mol, c_neighbor.GetIdx(), {o_idx, phosphorus_idx})
        if carbon_count > 0:
            alkyl_names.append(get_alkyl_name(carbon_count))

    if not alkyl_names:
        return None

    # Sort alphabetically
    alkyl_names.sort()

    # Format based on count and symmetry
    if len(alkyl_names) == 1:
        return f"{alkyl_names[0]} phosphate"
    elif len(alkyl_names) == 2:
        if alkyl_names[0] == alkyl_names[1]:
            return f"di{alkyl_names[0]} phosphate"
        else:
            return f"{alkyl_names[0]} {alkyl_names[1]} phosphate"
    elif len(alkyl_names) == 3:
        if len(set(alkyl_names)) == 1:
            return f"tri{alkyl_names[0]} phosphate"
        else:
            return f"{' '.join(alkyl_names)} phosphate"

    return None


def get_phosphanyl_prefix(mol, phosphorus_idx: int, exclude_atoms: set = None) -> Optional[str]:
    """
    Generate IUPAC P-68 phosphanyl prefix string.

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
    return f"{prefix}phosphanyl"


def name_phosphanyl_substituent(mol, frag_atoms, attach_idx: int) -> Optional[str]:
    """P-68.3: name a phosphorus-rooted substituent on an acyclic parent.

    ``-PH2`` -> ``phosphanyl``; ``-PR2`` -> ``dialkyl/diarylphosphanyl`` (via
    :func:`get_phosphanyl_prefix`). Wired into the chain/acid substituent path
    (``name_substituent`` Tier 1.93) so a trivalent-P substituent on a carbon
    chain is cited as the ``phosphanyl`` prefix rather than dropped.

    Fail-closed collision guard (P-45.3.1 load-bearing case): fires ONLY when
    ``attach_idx`` is a NEUTRAL, radical-free phosphorus whose every in-fragment
    heavy neighbour is a single-bonded carbon (organyl) — so a phosphoryl /
    phosphonic ``P=O`` (bonding number 5, but with an O neighbour) is EXCLUDED
    and left to the oxoacid subsystem. Standard-valence (bonding number 3) only
    in this task; the λ5 hydride branch is added in Task 3.

    Non-standard valence: only the all-H λ-hydride (-PH4) is named here as
    ``lambda5-phosphanyl`` (P-45.3.1 / P-14.1.3; house ASCII ``lambda5`` with NO
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
    # 'lambda5-phosphanyl' (P-45.3.1 / P-14.1.3). Organyl λ5 / any P=O declines.
    from ..perception.lambda_hydride import is_lambda_hydride_phosphorus
    if is_lambda_hydride_phosphorus(mol, attach_idx):
        return f"lambda{lam}-{base}"
    return None


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
