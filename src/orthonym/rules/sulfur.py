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


def name_sulfonyl_halide(features, style: str = "pin") -> Optional[str]:
    """P-67.1.4.4.1 / P-68.5.0 / P-65.3.1: the acid halide of a sulfonic /
    sulfinic acid, named by the two-word functional-class grammar
    '{parent-stem}sulfonyl {halide}' / '{parent-stem}sulfinyl {halide}'.

    BB-verbatim targets: 'ethanesulfonyl chloride' (@39650, PIN),
    'propane-1-sulfonyl chloride', '4-isocyanatobenzene-1-sulfonyl chloride
    (PIN)' (@26014).

    Implementation (root-cause reuse, not a band-aid): cap the S-bonded halogen
    with -OH to form the parent sulfonic/sulfinic acid, name that acid with the
    full engine — reusing ALL chain/ring numbering (yields 'propane-1-sulfonic
    acid' / 'ethanesulfonic acid' / 'benzenesulfonic acid', substituents and
    all) — then rewrite the '-onic/-inic acid' suffix to the acyl '-onyl/-inyl
    {halide}' functional-class word. Fail-closed (None) unless the capped
    molecule names cleanly as a sulfonic/sulfinic acid, so a wrong name is never
    emitted.
    """
    mol = features.mol
    pg = features.principal_group
    if pg not in ("sulfonyl_halide", "sulfinyl_halide"):
        return None
    matches = features.functional_groups.get(pg, [])
    # Poly-sulfonyl-halide (>1) deferred: the two-word grammar would need a
    # multiplied acyl word — fail closed until that class is built.
    if len(matches) != 1:
        return None
    match = matches[0]
    # Match tuples: sulfonyl (S, =O, =O, X); sulfinyl (S, =O, X) — S first, X last.
    halide_idx = match[-1]
    halide_sym = mol.GetAtomWithIdx(halide_idx).GetSymbol()
    from .acid_halides import _HALIDE_WORD
    halide_word = _HALIDE_WORD.get(halide_sym)
    if halide_word is None:
        return None

    # Cap the halide -> -OH, forming the parent oxoacid.
    rw = Chem.RWMol(mol)
    o_atom = rw.GetAtomWithIdx(halide_idx)
    o_atom.SetAtomicNum(8)
    o_atom.SetFormalCharge(0)
    o_atom.SetNumExplicitHs(0)
    o_atom.SetNoImplicit(False)
    try:
        capped = rw.GetMol()
        Chem.SanitizeMol(capped)
        capped_smiles = Chem.MolToSmiles(capped)
    except Exception:
        return None

    from ..namer import Orthonym
    acid_name = Orthonym(style=style, _disable_opsin_validity_gate=True).name(
        capped_smiles
    )
    if not acid_name or not isinstance(acid_name, str):
        return None

    if acid_name.endswith("sulfonic acid"):
        acyl = acid_name[: -len("sulfonic acid")] + "sulfonyl"
    elif acid_name.endswith("sulfinic acid"):
        acyl = acid_name[: -len("sulfinic acid")] + "sulfinyl"
    else:
        # Not a clean sulfonic/sulfinic acid (e.g. retained/complex parent the
        # rewrite cannot safely handle) -> fail closed.
        return None

    # The retained benzene stem carries no locant ('benzenesulfonic acid'), but
    # the PIN sulfonyl-halide form takes the '-1-' locant (BB
    # '...benzene-1-sulfonyl chloride'). Insert it for the bare-benzene stem.
    if acyl == "benzenesulfonyl":
        acyl = "benzene-1-sulfonyl"
    elif acyl == "benzenesulfinyl":
        acyl = "benzene-1-sulfinyl"

    return f"{acyl} {halide_word}"


_OXIDE_ACID_SUFFIX = {"sulfinyl": "sulfinic acid", "sulfonyl": "sulfonic acid"}


def _acid_stem_unsaturated_oxide_prefix(
    mol, sub_carbon: int, sulfur_idx: int, oxide_kind: str,
) -> Optional[str]:
    """P-63.6 acid-stem prefix ('prop-2-ene-1-sulfinyl') for an arm that
    ``_classify_oxide_side`` declines (unsaturated / branched / hetero).

    Root-cause reuse (mirrors ``name_sulfonyl_halide``): isolate the arm + the
    S(=O)x centre, cap S with a single ``-OH`` to form the parent sulfinic /
    sulfonic acid, name THAT with the full engine (reusing all chain / ring
    numbering + unsaturation locants), then rewrite the ``... sulfinic/sulfonic
    acid`` suffix to ``...sulfinyl/sulfonyl``. Fail closed (None) on any shape
    the acid namer does not return as a clean sulfinic/sulfonic acid, so a wrong
    name is never emitted. Additive: fires only when ``_classify_oxide_side``
    returned None, so all saturated-linear / benzene / cycloalkane outputs are
    byte-identical.
    """
    acid_suffix = _OXIDE_ACID_SUFFIX.get(oxide_kind)
    if acid_suffix is None:
        return None
    s_atom = mol.GetAtomWithIdx(sulfur_idx)
    # Terminal =O oxo on S define sulfinyl (1) vs sulfonyl (2).
    oxo = [
        n.GetIdx() for n in s_atom.GetNeighbors()
        if n.GetSymbol() == "O" and n.GetDegree() == 1
        and mol.GetBondBetweenAtoms(
            sulfur_idx, n.GetIdx()).GetBondTypeAsDouble() == 2.0
    ]
    if oxide_kind == "sulfinyl" and len(oxo) != 1:
        return None
    if oxide_kind == "sulfonyl" and len(oxo) != 2:
        return None
    # Arm side = component reachable from sub_carbon without crossing S.
    arm = set()
    q = deque([sub_carbon])
    while q:
        i = q.popleft()
        if i in arm or i == sulfur_idx:
            continue
        arm.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j not in arm and j != sulfur_idx:
                q.append(j)
    # The single parent-side neighbour of S (the one we detach). Historically this
    # was restricted to CARBON (the R-S(=O)x-R' sulfone/sulfoxide shape). Extended
    # to also accept a NITROGEN parent so an R-SO2-N< sulfonamide (whose other side
    # of S is the amide N, not a carbon) can have its R-sulfonyl stem built: cap S
    # with -OH after detaching the N and the arm names as the R-sulfonic acid
    # (`4-aminobenzene-1-sulfonic acid`), rewritten to `...sulfonyl` (task #28,
    # unblocks the substituted-arene `{Ar}sulfonamido` PIN, BB P-66.1.1.4.3 :33034).
    # Purely additive: sulfone/sulfoxide callers require two-carbon S (SMARTS), so
    # this branch only fires where the old carbon-only filter returned None. The
    # cap-name-rewrite is fail-closed (a non-clean `... sulfonic/sulfinic acid`
    # yields None), so a non-nameable R never emits. Restricted to C/N to leave
    # sulfonate-ester (parent = O) and other heteroatom shapes failing closed.
    parent_c = [
        n.GetIdx() for n in s_atom.GetNeighbors()
        if n.GetSymbol() in ("C", "N") and n.GetIdx() not in arm
    ]
    if len(parent_c) != 1:
        return None  # not the R-S(=O)x-R'/N< shape we cap
    rw = Chem.RWMol(mol)
    oh = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(sulfur_idx, oh, Chem.BondType.SINGLE)
    rw.RemoveBond(sulfur_idx, parent_c[0])
    try:
        frag = rw.GetMol()
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    # Isolate the S-bearing fragment (the parent side is now disconnected).
    try:
        pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    acid_smiles = None
    acid_piece = None
    for p in pieces:
        if any(a.GetSymbol() == "S" for a in p.GetAtoms()):
            acid_smiles = Chem.MolToSmiles(p)
            acid_piece = p
            break
    if acid_smiles is None:
        return None
    # Two soundness guards on the acid-stem -> acyl rewrite. That rewrite trusts a
    # gate-DISABLED acid sub-namer, which a fable review of 7c621b84 showed could emit a
    # wrong constitution for the N-parent (sulfonamido) case. 8afa533c added the guards
    # but scoped them to the N-parent branch to avoid a gold-risk on a path it had not
    # itself broken. v30 F6 confirmed the CARBON-parent sulfone/sulfoxide-prefix path
    # has the SAME false friends (`CS(=O)(=O)CCS(O)(=O)=O` -> `ethane-1,2-disulfonyl`),
    # so both guards now fire for either parent element (C or N). Failing closed here
    # returns the caller to its own fail-closed handling; the corrupted stem never ships.
    #
    # Guard 1 (structural): if R itself carries a COMPETING S-oxo-acid (-SO3H / -SO2H),
    # the capped fragment has >=2 acid groups and the acid namer spells it as a MULTIPLIED
    # acid ('ethane-1,2-disulfonic acid'). Stripping the 'sulfonic acid' suffix then
    # leaves the multiplier ('ethane-1,2-di') re-bound to a meaning it never had ->
    # 'ethane-1,2-disulfonyl' (carbon path) / 'ethane-1,2-disulfonamido' (N path, also
    # OPSIN-unparseable). Fail closed unless exactly ONE S bears an -OH/-O- (the one we
    # just capped) -- an RT-invisible defect, so a structural guard, not a round-trip.
    #
    # ⚠ Guard 1 and guard 2 are COMPLEMENTARY, not redundant (fable review of F6): guard 2
    # CANNOT catch the multiplied-acid case, because 'ethane-1,2-disulfonic acid' is the
    # CORRECT name of the capped fragment and re-anchors skeleton-EXACT -- the corruption
    # lives entirely in the suffix->acyl STRIP, which guard 2 never inspects. Do NOT delete
    # guard 1 as "redundant with guard 2".
    #
    # NOTE (fable review of F6): this counts ANY S bearing a single-bonded O, so it also
    # vetoes a sulfonate/sulfinate ESTER arm, a mesyloxy arm, and a charge-separated
    # sulfoxide arm -- broader than "competing acid". Today that is pure gain: the
    # gate-disabled acid sub-namer mis-names every one of those (silent atom drop / SO3H
    # migration), so the veto only blocks already-wrong prefixes. Revisit ONLY if that
    # sub-namer later learns to name such arms correctly (then guard 1 would cap a
    # then-legitimate prefix); it does not today.
    _acid_s = sum(
        1 for a in acid_piece.GetAtoms()
        if a.GetSymbol() == "S"
        and any(nb.GetSymbol() == "O"
                and acid_piece.GetBondBetweenAtoms(
                    a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 1.0
                for nb in a.GetNeighbors())
    )
    if _acid_s != 1:
        return None
    from ..namer import Orthonym
    acid_name = Orthonym(style="pin", _disable_opsin_validity_gate=True).name(
        acid_smiles
    )
    if not isinstance(acid_name, str) or not acid_name.endswith(acid_suffix):
        return None  # fail closed: not a clean sulfinic/sulfonic acid
    # Guard 2 (gate-INDEPENDENT re-anchor): the acid sub-namer runs with its OPSIN
    # validity gate DISABLED, so an acid-namer defect that still ends in 'sulfonic acid'
    # (SO3H migrating onto a ring -> 'methylcyclohexanesulfonic acid'; a dropped arm ->
    # 'ethanesulfonic acid') would become a wrong sulfinyl/sulfonyl prefix. RE-ANCHOR the
    # accepted acid name explicitly (gate-INDEPENDENT: does not rely on the global SELF-01
    # setting, which is off in tests and absent with no jar): OPSIN-parse it and require
    # the same constitutional skeleton as the capped fragment; fail closed on mismatch or
    # when OPSIN is unavailable.
    from ..namer import (
        _validity_gate_name_to_smiles, _self_consistency_skeleton,
        _registration_stereo_layer,
    )
    _reparsed = _validity_gate_name_to_smiles(acid_name)
    if _reparsed is None:
        return None
    _sk_name = _self_consistency_skeleton(_reparsed)
    _sk_frag = _self_consistency_skeleton(acid_smiles)
    if _sk_name is None or _sk_frag is None or _sk_name != _sk_frag:
        return None
    # v30 #36 (fable F6 finding 3): the skeleton block above is the InChIKey first
    # block, which EXCLUDES stereo (ADR-18-07), so a WRONG CIP descriptor (E/Z, R/S)
    # from the gate-disabled acid sub-namer would re-anchor skeleton-exact and ship a
    # wrong-stereo `...sulfinyl/sulfonyl` prefix. Also require the C6 RegistrationHash
    # stereo layer (stereo-bearing, tautomer-canonical) to match — gate-INDEPENDENT,
    # fail-closed on mismatch/unhashable. No live witness today (the sub-namer routes
    # stereo through the standard CIP path, so its descriptors are correct), so this is
    # defence-in-depth closing the gap invariant 2 requires: honest WITHOUT the gate,
    # for both the N-parent (8afa533c) and carbon-parent (F6) paths.
    _st_name = _registration_stereo_layer(_reparsed)
    _st_frag = _registration_stereo_layer(acid_smiles)
    if _st_name is None or _st_frag is None or _st_name != _st_frag:
        return None
    return acid_name[: -len(acid_suffix)] + oxide_kind


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
