"""
Stereochemistry rules - IUPAC stereodescriptor collection and formatting.

This module handles the mapping from RDKit stereochemistry (indexed by atom
indices) to IUPAC nomenclature (indexed by locants on the principal chain/ring).

Key functions:
- collect_stereodescriptors: Get R/S and E/Z descriptors with IUPAC locants
- format_stereodescriptor_string: Format as "(2R,3S)-" prefix
- get_double_bond_locant: Get lower locant for E/Z double bond

Ring junction stereochemistry functions:
- get_bridgehead_atoms: Find ring fusion stereocenter atoms
- collect_ring_junction_stereo: Collect junction stereo with 'a' suffix locants
- format_ring_junction_stereo: Format as "(4aR,8aS)-" or "(4ar,8ac)-"
- determine_simple_cis_trans: Return "cis" or "trans" for simple bicyclics

IMPORTANT: This module requires atom_to_locant mapping to be provided by the
caller (computed during chain/ring classification). It does NOT fall back to
atom indices as those are NOT valid IUPAC locants.
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from rdkit import Chem

from ..perception.molcache import (  # audit 2026-09-03 (S2): per-call atom/bond tuples
    atoms_of,
    bonds_of,
)
from ..perception.stereo import assign_stereochemistry, detect_axial_chirality

logger = logging.getLogger(__name__)


def _composite_locant_sort_key(
    item: Tuple[Union[int, str], str],
) -> Tuple[int, str]:
    """Sort key for descriptor lists that may mix int and '<int><letter>' locants.

    Mirrors _junction_locant_sort_key (line ~727, a phase) so that the
    universal collector (collect_stereodescriptors, line ~33) and the
    ring-junction collector (collect_ring_junction_stereo, line ~680) share
    one sort discipline.

    Per IUPAC, composite locants like '3a' sort BETWEEN integer 3
    and integer 4 (after 3 because '' < 'a' in tuple-lex comparison).

    a phase -- single source of truth for descriptor ordering. Per
    Pitfall 2, branches on isinstance(locant, int) FIRST so int input does
    NOT fall into the int('3'[:-1]) = int('') ValueError trap.

    Behaviour on existing int-only locants (regression invariant):
    byte-identical sort order to lambda x: x[0]. Verified by
    TestStereoBackstopRegressionInvariant in
    tests/unit/rules/test_handler_stereo_injection.py.
    """
    locant = item[0]
    if isinstance(locant, int):
        return (locant, '')
    # Primed locant from a MULTI-COMPONENT ring: atom_to_locant returns tuples
    # like (5, "'") / (3, "''") for the primed component (fused_rings /
    # multi-component numbering). Per a primed locant sorts AFTER its
    # unprimed twin at the same number ('' < "'" < "''"), so key on
    # (number, prime-suffix). Fixes a TypeError crash (a review review RISK 6):
    # int(locant) on a tuple raised instead of failing closed.
    if isinstance(locant, tuple):
        num = locant[0] if locant and isinstance(locant[0], int) else 0
        return (num, ''.join(str(x) for x in locant[1:]))
    # str path -- '3a' / '7a' / '12b'
    if isinstance(locant, str) and locant and locant[-1].isalpha():
        return (int(locant[:-1]), locant[-1])
    # Fail closed, never crash: an unparseable locant sorts last. The descriptor
    # ORDER only affects the emitted string (OPSIN parses the descriptor set
    # order-independently), so a defensive last-sort degrades spelling at worst,
    # never structure -- and the whole name is still RT/ gated.
    try:
        return (int(locant), '')
    except (TypeError, ValueError):
        return (float('inf'), str(locant))


def _is_true_exocyclic(mol, in_scope_idx: int, other_idx: int) -> bool:
    """Is this double bond genuinely EXOcyclic to the ring carrying the locant?

    The exocyclic licence lets a bond borrow the locant of its one in-scope atom
    (`cyclohexan-1-ylidene`-shaped cases: the ring atom is numbered, the other end
    hangs off the ring). It requires BOTH halves of "exocyclic":

    1. the in-scope atom is in a ring, and
    2. the other end is NOT in a ring.

    Only (1) used to be tested, so an ENDOcyclic double bond of some OTHER ring —
    one the scope's numbering does not cover — also qualified and borrowed a locant
    from whichever of its atoms happened to appear in the map. That is how
    `2,2,3-tri(cyclodec-1-en-1-yl)propanoate` acquired a parent-scope `(3E)`: the
    C=C lies inside a cyclodecene SUBSTITUENT, and propanoate has no double bond at
    its C3 for the descriptor to resolve to (OPSIN: `Could not find bond that:
    <stereoChemistry locant="3" …> was referring to`). Per (the Blue Book) that
    descriptor belongs on the substituent prefix, not at parent scope.

    Condition (1) is not required any more: an in-scope CHAIN atom's double
    bond to an acyclic atom outside the scope is the '-ylidene' double bond of
    that chain position, and (1)(a) (the Blue Book, "the
    double bond is considered as an integral part of the parent structure; the
    stereodescriptor is placed at the front of the substitutive name, preceded
    by the locant indicating its point of attachment to the parent structure";
    :48279 "Method (a) generates preferred IUPAC names") cites it with the
    chain atom's locant, as for a ring: '(4E)-4-(chloromethylidene)heptanoic
    acid'. Skipping it left the name stereo-incomplete: the round trip then
    refused every candidate of the ChEBI chloromethylidene lactams (it used to
    pass only through a misplaced-locant '(1Z)' block OPSIN happened to bind).
    Condition (2) still keeps an endocyclic bond of another ring out.
    """
    if mol.GetAtomWithIdx(other_idx).IsInRing():
        return False
    return True


def collect_stereodescriptors(
    mol,
    atom_to_locant: Dict[int, int],
    include_near_parent_ez: bool = False,
    *,
    skip_bonds: Iterable[int] = (),
) -> List[Tuple[int, str]]:
    """
    Collect all stereodescriptors from a molecule using IUPAC locants.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        atom_to_locant: Mapping from atom index to IUPAC locant number.
                       Only atoms in this mapping are considered (principal
                       chain/ring atoms).
        include_near_parent_ez: RETAINED FOR CALL-SITE COMPATIBILITY; NO LONGER
                       EMITS. It used to admit E/Z bonds one hop away from the
                       parent (neither bond atom in atom_to_locant, but one has
                       a neighbour that is), citing the neighbour's locant, on
                       the stated authority of "IUPAC ". That citation
                       is wrong — (the Blue Book) is "von Baeyer compounds".
                        (the Blue Book) instead requires a substituent's
                       descriptor to be cited "at the front of the corresponding
                       prefix", so a bond lying wholly inside a substituent has
                       no locant in the PARENT's numbering and is now skipped.
                       See the fail-closed branch below for the full derivation.
        skip_bonds: indices of E/Z bonds this scope must NOT cite because
                       another scope of the same name cites them. The one
                       caller is a '-ylidene' substituent whose attachment
                       double bond its host already cites with the host's
                       locant (1)(a), the Blue Book):
                       citing it in both scopes is one stereogenic unit cited
                       twice. Default empty (every other caller unchanged).

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.
        cip_code is 'R', 'S', 'r', 's' (lowercase for pseudoasymmetric),
        or 'E', 'Z' for double bonds.

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H](O)CC')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1} # chain oriented
        >>> collect_stereodescriptors(mol, atom_to_locant)
        [(3, 'R')]
    """
    # Ensure stereochemistry is assigned (idempotent guard)
    assign_stereochemistry(mol)

    descriptors: List[Tuple[int, str]] = []

    # Collect R/S stereocenters (atom-based)
    for atom in atoms_of(mol):
        if atom.HasProp('_CIPCode'):
            atom_idx = atom.GetIdx()
            # Only include atoms on principal chain/ring
            if atom_idx in atom_to_locant:
                locant = atom_to_locant[atom_idx]
                cip_code = atom.GetProp('_CIPCode')
                # Preserve CIP code as-is: R/S for normal stereocenters,
                # r/s for pseudoasymmetric centers per IUPAC.
                descriptors.append((locant, cip_code))

    # Collect E/Z double bonds (bond-based)
    #
    # ⚠ `_CIPCode` ON A BOND IS NOT EXCLUSIVELY AN E/Z CODE. RDKit also sets it on
    # STEREOATROPCW / STEREOATROPCCW bonds — an atropisomeric AXIS — where the value
    # is the helicity letter 'M' or 'P', not 'E'/'Z'. Testing only `HasProp` therefore
    # fed an axial stereogenic unit into the E/Z channel, and because the axis is ALSO
    # reported (correctly) by `detect_axial_chirality` below, one unit was emitted
    # twice: `collect_stereodescriptors` returned [(7,'M'),(7,'Sa')] and
    # `format_stereodescriptor_string` spelled it '(7M,7Sa)-' — a malformed
    # duplicate-locant block. Measured on the suite's own `_make_biaryl_atropisomer`.
    #
    # The filter is on the CIP CODE VALUE, deliberately, NOT on `bond.GetStereo`.
    # Gating on `GetStereo in (STEREOE, STEREOZ)` looks like the obvious test and is
    # WRONG: measured with RDKit here, an ordinary SMILES double bond reports
    # STEREOTRANS / STEREOCIS while carrying `_CIPCode` 'E'/'Z' —
    # 'C/C=C/C' -> stereo=STEREOTRANS, _CIPCode=E
    # 'C/C=C\\C' -> stereo=STEREOCIS, _CIPCode=Z
    # so that allow-list would have silently deleted essentially EVERY legitimate E/Z
    # descriptor the project emits. Keying on the value is exact: the E/Z channel
    # carries E/Z codes, whatever enum RDKit used to record the geometry, and any
    # future letter RDKit adds is excluded by construction rather than by enumeration.
    #
    # The axis itself is NOT lost — `detect_axial_chirality` remains its single source.
    _skip_bond_ids = frozenset(skip_bonds) if skip_bonds else frozenset()
    for bond in bonds_of(mol):
        if bond.HasProp('_CIPCode') and bond.GetProp('_CIPCode') in ('E', 'Z'):
            if bond.GetIdx() in _skip_bond_ids:
                continue  # cited by another scope of the name (see ``skip_bonds``)
            # Skip ring-constrained double bonds in small rings: double bonds
            # in rings of size 7 or fewer have geometry fixed by ring strain.
            # Macrocyclic rings (8+ members) CAN have meaningful E/Z geometry
            # per errata (Sep 2024).
            if bond.IsInRing():
                ri = mol.GetRingInfo()
                min_ring_size = float('inf')
                for ring in ri.BondRings():
                    if bond.GetIdx() in ring:
                        min_ring_size = min(min_ring_size, len(ring))
                if min_ring_size < 8:
                    continue

            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Determine locant for this E/Z bond.
            # Standard case: both atoms in atom_to_locant -> use lower locant.
            # Exocyclic / '-ylidene' case (1)(a)): one atom in
            # atom_to_locant (a ring or chain atom of this scope), the other an
            # acyclic atom outside it -> use the in-scope atom's locant. A bond
            # wholly inside a substituent (neither atom in scope) is skipped
            # below, and an endocyclic bond of another ring by _is_true_exocyclic.
            if begin_idx in atom_to_locant and end_idx in atom_to_locant:
                # Both in parent -- standard behaviour
                locant = min(atom_to_locant[begin_idx],
                             atom_to_locant[end_idx])
            elif begin_idx in atom_to_locant and end_idx not in atom_to_locant:
                # Only an exocyclic / '-ylidene' bond (the other end acyclic),
                # never an endocyclic bond of another ring (_is_true_exocyclic)
                if not _is_true_exocyclic(mol, begin_idx, end_idx):
                    continue
                locant = atom_to_locant[begin_idx]
            elif end_idx in atom_to_locant and begin_idx not in atom_to_locant:
                if not _is_true_exocyclic(mol, end_idx, begin_idx):
                    continue
                locant = atom_to_locant[end_idx]
            else:
                # Neither atom is in this scope's numbering, so this bond has NO locant
                # in this scope and cannot be cited here.
                #
                # A "one-hop" branch used to stand here: it walked to a neighbouring
                # in-scope atom and cited THAT atom's locant, on the stated authority of
                # "IUPAC ". That citation is wrong — (the Blue Book) is
                # "von Baeyer compounds", and says nothing about projecting a
                # substituent's double bond onto a parent locant.
                #
                # The rule that does govern is "NAMING OF STEREOISOMERS"
                # (the Blue Book): "They are placed at the front of the complete name when
                # related to the parent structure... WHEN THEY RELATE TO SUBSTITUENT
                # GROUPS, THEY ARE CITED AT THE FRONT OF THE CORRESPONDING PREFIX." A C=C
                # lying wholly inside a substituent is a substituent stereogenic unit, so
                # its descriptor belongs on that prefix — which the substituent naming
                # path already does, e.g. `[(E)-2-phenylethenyl]`. Citing it at parent
                # scope invents a locant the parent does not have a double bond at, which
                # OPSIN rejects with `Could not find bond that: <stereoChemistry …> was
                # referring to`, or — worse — silently binds it to an unrelated parent
                # double bond that IS at that locant.
                #
                # So fail closed. Measured effect (n=1000, pubchem_2000): see
                # internal notes
                continue

            cip_code = bond.GetProp('_CIPCode')  # 'E' or 'Z'
            descriptors.append((locant, cip_code))

    # Collect axial chirality (Ra/Sa) -- atropisomers and allenes (IUPAC
    axial_elements = detect_axial_chirality(mol)
    for element in axial_elements:
        cip = element.get('cip')
        if cip is None:
            continue  # Undetermined chirality -- skip
        locant_atom = element.get('locant_atom')
        if locant_atom is not None and locant_atom in atom_to_locant:
            locant = atom_to_locant[locant_atom]
            descriptors.append((locant, cip))
        # If locant_atom not in atom_to_locant, the axial chirality element
        # is not on the principal chain/ring -- skip (same filtering as R/S)

    # Sort by locant ascending.: use the composite-locant safe key so
    # mixed int / '<int><letter>' (e.g. '3a', '7a' from ComplexRingResult.
    # atom_to_locant per fused_rings.py:317) sort per IUPAC
    # ('3a' BETWEEN integer 3 and 4). Byte-identical to the pre-153
    # `lambda x: x[0]` for int-only inputs (locked by
    # TestStereoBackstopRegressionInvariant in tests/unit/rules/
    # test_handler_stereo_injection.py).
    descriptors.sort(key=_composite_locant_sort_key)

    # Filter out stereo descriptors with invalid locants (locant 0 or > parent size)
    #
    # ⚠ NOTE this check is SELF-REFERENTIAL and therefore much weaker than it looks:
    # `parent_size` is the max of the very map that produced every locant, so it can only
    # ever catch 0/negatives. It cannot tell that the map itself belongs to a different
    # scope than the name being decorated. The caller owns that (see
    # `handlers/general_acyclic.py`,.
    if descriptors and atom_to_locant:
        int_locants = [v for v in atom_to_locant.values() if isinstance(v, int)]
        parent_size = max(int_locants) if int_locants else 0
        if parent_size > 0:
            from .locant_validation import validate_stereo_locants
            descriptors = validate_stereo_locants(descriptors, parent_size)

    # "NAMING OF STEREOISOMERS" (the Blue Book): a stereodescriptor is "preceded by a
    # numerical or letter locant to describe THE POSITION OF THE STEREOGENIC UNIT". A
    # locant names one position in one numbering, so two descriptors sharing a locant in
    # a single block do not describe two positions — the block is unresolvable, and OPSIN
    # rejects it with `Could not find atom/bond that: <stereoChemistry …> was referring to`.
    #
    # Two DIFFERENT double bonds cannot share one parent locant, so an E/Z locant cited
    # twice means at least one of them was projected in from another scope, e.g.
    # `2,2,3-tri(cyclodec-1-en-1-yl)propanoate` -> `(2E,2E,3E)-`, where the bonds live in
    # the SUBSTITUENTS and puts their descriptors "at the front of the corresponding
    # prefix". Fail closed on the contested locant rather than arbitrarily keeping one,
    # which would assert a configuration for a position with more than one unit on it.
    #
    # SCOPED TO E/Z DELIBERATELY, because only E/Z can be projected in from another scope.
    # R/S cannot collide (one atom carries one code), and an axial element may share its
    # locant with a genuine E/Z bond at the same position (e.g. a cumulene numbered so the
    # axis and a double bond start at one locant) — that is two DIFFERENT kinds of unit,
    # not two E/Z bonds contesting one locant, and an unscoped rule deletes both (locked by
    # TestCollectStereodescriptorsAxial).
    #
    # This comment used to justify the scoping differently: it said an axial element
    # "legitimately shares its locant with the underlying bond's own CIP code —
    # `detect_axial_chirality` reports (7,'Sa') for the very bond the E/Z loop reports as
    # (7,'M')". That was describing the DEFECT, not a legitimate case. The E/Z loop had no
    # business reporting an atropisomeric axis as (7,'M') at all; it is now filtered at
    # source (see the E/Z collection loop above), so one axis yields exactly one
    # descriptor and the duplicate never forms.
    _ez = [d for d in descriptors if d[1] in ('E', 'Z')]
    if len(_ez) > 1:
        counts: Dict[Any, int] = {}
        for locant, _cip in _ez:
            counts[locant] = counts.get(locant, 0) + 1
        if any(c > 1 for c in counts.values()):
            descriptors = [
                d for d in descriptors
                if d[1] not in ('E', 'Z') or counts.get(d[0], 0) == 1
            ]

    return descriptors


def get_double_bond_locant(
    bond,
    atom_to_locant: Dict[int, int]
) -> Optional[int]:
    """
    Get the IUPAC locant for a double bond.

    Per IUPAC convention, the locant is the lower of the two atom locants.

    Args:
        bond: RDKit Bond object
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Lower locant of the two bond atoms, or None if either atom is
        not in the mapping (not on principal chain/ring).

    Example:
        >>> mol = Chem.MolFromSmiles('C/C=C/C')
        >>> bond = mol.GetBondWithIdx(1) # the C=C bond
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        >>> get_double_bond_locant(bond, atom_to_locant)
        2
    """
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    # Both atoms must be in mapping
    if begin_idx not in atom_to_locant or end_idx not in atom_to_locant:
        return None

    locant_a = atom_to_locant[begin_idx]
    locant_b = atom_to_locant[end_idx]

    return min(locant_a, locant_b)


def _render_locant_token(locant: Union[int, str, Tuple]) -> str:
    """Render a descriptor locant as its IUPAC token string.

    Scalar locants render byte-identically to ``str(locant)`` — an int ('2') or
    a composite ring-junction string ('7a'). A PRIMED tuple locant from a
    multi-component ring — ``(6, "'")`` / ``(3, "''")`` for the 2nd (primed)
    spiro/fused component — renders as its parts concatenated: ``6'`` / ``3''``.
    Mirrors ``_composite_locant_sort_key``, which already accepts the tuple.
    """
    if isinstance(locant, tuple):
        return "".join(str(x) for x in locant)
    return str(locant)


def format_stereodescriptor_string(
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format stereodescriptors as an IUPAC name prefix.

    STEREO-04: Multiple descriptors in single block with comma separation.

    Args:
        descriptors: List of (locant, cip_code) tuples, sorted by locant.

    Returns:
        Formatted string like "(2R)-", "(2R,3S)-", "(2E,3R,5Z)-"
        Returns empty string if no descriptors.

    Examples:
        >>> format_stereodescriptor_string()
        ''
        >>> format_stereodescriptor_string([(2, 'R')])
        '(2R)-'
        >>> format_stereodescriptor_string([(2, 'R'), (3, 'S')])
        '(2R,3S)-'
        >>> format_stereodescriptor_string([(2, 'E'), (3, 'R'), (5, 'Z')])
        '(2E,3R,5Z)-'
        >>> format_stereodescriptor_string([(2, 'r'), (3, 's')])
        '(2r,3s)-'
    """
    if not descriptors:
        return ""

    # Build comma-separated list of "locantCIP".
    # A PRIMED tuple locant from a MULTI-COMPONENT (spiro/fused) name — e.g.
    # (6, "'") / (3, "''") for the 2nd component — must render as its number
    # followed by the prime(s) ('6'', '3''') per, NOT the Python tuple
    # repr `(6, "'")`. This is the render-side counterpart to
    # `_composite_locant_sort_key`, which already accepts the (int, "'") tuple.
    # Scalar int / composite-str locants ('2', '7a') render byte-identically to
    # str(locant), so the common single-component path is unchanged.
    parts = [f"{_render_locant_token(locant)}{cip}" for locant, cip in descriptors]

    return f"({','.join(parts)})-"


# ---------------------------------------------------------------------------
# a phase: Handler-level stereo injection (/ predicate-first wiring)
# ---------------------------------------------------------------------------

#: detection regex set REUSED VERBATIM from namer.py:62-75. DO NOT BROADEN.
# Pattern A — prefix form (already-stereoed name; matched against name start)
_STEREO_PREFIX_RE = re.compile(
    # a phase / Pitfall 8: accept composite locants like '7a', '3a'
    # in addition to plain digits, so an already-stereoed name like
    # `(7aS)-2,3a-dichloro-...benzofuran-...` is correctly recognized as
    # already-stereoed and the injector does not double-emit the prefix.
    # Compatible with the existing `(2R)-` / `(R)-` / `(E)-` / `(2R,3S)-`
    # / `(2r,3s)-` matches (the locant prefix is optional via \d*).
    # -A2: also accept an optional PRIME `'`/`''` after the locant, so a
    # correct multi-component (spiro/fused) block like `(2S,1'R,3'R,11'R)-` is
    # recognized as already-stereoed and the injector does not double-emit.
    # The prime is optional (`'{0,2}`), so unprimed matches are byte-identical.
    r"\(\d*[a-z]?'{0,2}[RSrsEZez](,\d*[a-z]?'{0,2}[RSrsEZez])*\)-"
)
# Pattern B — embedded block (descriptor block anywhere in the name body)
_STEREO_EMBEDDED_RE = re.compile(
    # a phase: also accept composite locants like '7a', '3a' in
    # embedded blocks (e.g., 'something-(3aR,7aS)-else').
    # -A2: also accept an optional prime `'`/`''` after the locant (primed
    # spiro/fused component); optional, so unprimed matches are byte-identical.
    r"\(\d*[a-z]?'{0,2}[RSEZrsez](,\d*[a-z]?'{0,2}[RSEZrsez])*\)"
)
# Pattern C — carbohydrate / amino-acid traditional notation.: also match the
# Blue-Book GREEK anomeric symbols α/β, now emitted in place of the ASCII
# words (OPSIN parses both identically).
_CARBOHYDRATE_STEREO_RE = re.compile(r'(alpha|beta|alfa|α|β)-[DL]-', re.IGNORECASE)

# a phase.0 : Pattern D — a leading/embedded D-/L- configurational
# token (D-alanine, d-glyceraldehyde, L-valine) AND peptide acyl chains
# (L-valyl-… / D-glucosaminyl-…) are treated as stereo-already-present, alongside
# the existing alpha/beta-anomeric Pattern C. These names already encode their
# configuration via the D/L (or L-…yl-) descriptor, so the injector must NOT
# double-encode them with a (nR)/(nS) block. Anchored at name-start OR after a
# separator (whitespace / hyphen) — a bare [dDlL]- anywhere is too broad
# (RESEARCH Assumption A1). This is the load-bearing protection for the Plan-02
# backstop flip: peptides report complex_ring/unknown/direct (NOT heterocycle),
# so the allowlist exclusion alone is NOT sufficient — this predicate is.
# Case-insensitive: the traditional notation appears as upper-case (D-alanine,
# L-valine) AND lower-case (d-glyceraldehyde — which is exactly what Orthonym
# emits for its retained glyceraldehyde name). The leading anchor keeps it from
# matching a stray 'l'/'d' mid-token.
_DL_CONFIG_RE = re.compile(r'(^|[\s\-])([DL])-', re.I)
# Peptide acyl chain (L-valyl- / D-glucosaminyl-) — case-insensitive.
_PEPTIDE_ACYL_RE = re.compile(r'\b[DL]-[a-z]+yl-', re.I)

# W5-A4: detector for an L-SUPPRESSED peptide name. After L-omission
# a peptide such as 'alanylalanine' / 'valyltyrosylisoleucine' carries no D/L
# token, so Pattern D can no longer see it. This does an EXACT greedy
# decomposition against the amino-acid acyl/terminal tables: internal residues
# are acyl stems (optionally cited 'D-'), the final residue is a terminal AA name
# (optionally cited 'D-'). Exact matching (not a loose regex) keeps it from
# false-positively suppressing stereo injection on a non-peptide name.
_PEPTIDE_STEM_SETS = None


def _peptide_stem_sets():
    """Return (acyl_stems, terminal_names) frozensets, L/D-stripped, cached."""
    global _PEPTIDE_STEM_SETS
    if _PEPTIDE_STEM_SETS is None:
        try:
            from ..data.amino_acids import AMINO_ACID_ACYL_NAMES
        except Exception:
            _PEPTIDE_STEM_SETS = (frozenset(), frozenset())
            return _PEPTIDE_STEM_SETS

        def _strip_dl(s: str) -> str:
            for p in ("D-", "L-", "d-", "l-"):
                if s.startswith(p):
                    return s[len(p):]
            return s

        acyls = frozenset(_strip_dl(v) for v in AMINO_ACID_ACYL_NAMES.values())
        terms = frozenset(_strip_dl(k) for k in AMINO_ACID_ACYL_NAMES.keys())
        _PEPTIDE_STEM_SETS = (acyls, terms)
    return _PEPTIDE_STEM_SETS


def _looks_like_peptide(name: str) -> bool:
    """True iff *name* is an Orthonym peptide of >=2 residues /.3.4).

    Greedy longest-match decomposition: strip a cited 'D-' descriptor (leading, or
    '-D-' before a later residue), consume the whole remainder as a terminal AA
    name when it matches, else consume the longest acyl stem that leaves more to
    parse, and repeat. Returns False on any residue that is not a known stem/name,
    so non-peptides (and the exotic hyphenated AA-derivatives) fall through to the
    normal injection path (which the allowlist already gates safely)."""
    acyls, terms = _peptide_stem_sets()
    if not acyls:
        return False
    acyls_by_len = sorted(acyls, key=len, reverse=True)
    s = name.strip()
    residues = 0
    while s:
        # Strip a cited D- descriptor: leading for the first residue, '-D-' later.
        if residues == 0 and s.startswith("D-"):
            s = s[2:]
        elif residues > 0 and s.startswith("-D-"):
            s = s[3:]
        elif residues > 0 and s.startswith("-"):
            return False  # a hyphen that is not '-D-' is not this peptide grammar
        # Terminal residue: the remainder is exactly a terminal AA name.
        if s in terms:
            return residues + 1 >= 2
        # Internal residue: longest acyl stem that leaves more of the chain.
        stem = next(
            (a for a in acyls_by_len if s.startswith(a) and len(s) > len(a)), None
        )
        if stem is None:
            return False
        s = s[len(stem):]
        residues += 1
    return False

# OPSIN-validity stereo carve-out (a phase): a LEADING stereo / relative-configuration descriptor-block
# matcher for strip_stereo. Mirrors scripts/pin_strict_eval._STEREO_PREFIX (the
# validation precedent). Matches a leading (...)- block whose contents are PURELY
# stereo descriptors (digits, optional composite-locant letter, r/s/e/z/R/S/E/Z/*,
# RS/SR, commas/hyphens/+/space), OR a bare rel-/rac-/cis-/trans-/(±)- prefix. A
# substituent enclosing group like "(2-chloroethyl)-" does NOT match (its content has
# non-descriptor letters). Deliberately SEPARATE from _STEREO_PREFIX_RE (do NOT broaden
# that one — a phase).
_STRIP_STEREO_LEADING_RE = re.compile(
    r"^(\((?:[0-9]+[a-z]?[rsezRSEZ*]|[rsezRSEZ]|RS|SR|[,\-+ ])+\)|rel|rac|cis|trans|\(±\))-",
    re.IGNORECASE,
)
# W4-S2: the SAME pure-stereo descriptor block, but NESTED — i.e. sitting right
# after an enclosing-mark opening as a substituent's own configuration
# (``[(1r,4r)-4-methylcyclohexyl]benzene``, ``N-[(1s,4s)-4-methylcyclohexyl]…``).
# The leading matcher above only reaches a block at name-start, so a nested
# substituent-stereo block was invisible to the "does the constitutional form
# parse?" probe, and the gate wrongly suppressed a correct-by-construction
# PIN whose ONLY OPSIN-unparseable feature is the r/s cyclohexane stereo layer.
# This removes ONLY the pure-stereo block, preserving the enclosing mark (so the
# substituent constitution is left intact for the OPSIN parse test); it can never
# turn a constitutionally-wrong name into a parseable one, so shipping stays
# strictly gated on constitution. Same content class as the leading matcher.
_STRIP_STEREO_NESTED_RE = re.compile(
    r"([\[(])\((?:[0-9]+[a-z]?[rsezRSEZ*]|[rsezRSEZ]|RS|SR|[,\-+ ])+\)-",
    re.IGNORECASE,
)


def strip_stereo(name: str) -> str:
    """Return *name* with leading stereo / relative-config descriptor blocks removed
    (to a fixpoint) — the OPSIN-validity stereo carve-out "where does OPSIN fail" probe (internal notes).

    READ-ONLY: this is NOT a postprocessor on shipped names. The validity gate uses it
    only to test whether a name's CONSTITUTIONAL (stereo-stripped) form parses; the
    SHIPPED name keeps its stereo. Strips leading ``(2R)-`` / ``(1s,4s)-`` / ``(E)-`` /
    ``rel-`` / ``rac-`` / ``cis-`` / ``trans-`` / ``(±)-`` blocks; leaves a name with no
    leading descriptor (``hexane``) and substituent enclosing groups (``(2-chloroethyl)``)
    untouched.
    """
    return strip_stereo_blocks(name)[0]


def strip_stereo_blocks(name: str):
    """``(strip_stereo(name), blocks)``: the stripped name and every descriptor block
    it removed, in the order removed -- a parenthesised block WITH its parentheses
    (``'(7R,8S)'``, ``'(1r,4r)'``) or a relative-configuration word (``'rel'``,
    ``'rac'``, ``'cis'``, ``'trans'``, ``'(±)'``).

    READ-ONLY, like:func:`strip_stereo` (which is this function's first element).
    The validity gate's stereo-layer carve-out reads the removed blocks to check
    every descriptor it would ship against the input's CIP labels
    (``namer._stereo_descriptors_verified``).
    """
    if not name:
        return name, []
    blocks = []

    def _nested(m):
        # group(0) is '<mark>(<block>)-'; keep the enclosing mark.
        blocks.append(m.group(0)[1:-1])
        return m.group(1)

    s = name.strip()
    prev = None
    while prev != s:
        prev = s
        # The leading matcher is anchored at the start, so it removes at most one
        # block per pass (the same as re.sub with the '^' anchor).
        m = _STRIP_STEREO_LEADING_RE.match(s)
        if m:
            blocks.append(m.group(1))
            s = s[m.end():]
        s = s.strip()
        # W4-S2: also strip nested substituent-stereo blocks (``[(1r,4r)-…``),
        # keeping the enclosing mark. Applied in the same fixpoint so a name with
        # both a leading and a nested block converges (read-only gate probe only).
        s = _STRIP_STEREO_NESTED_RE.sub(_nested, s).strip()
    return s, blocks


def needs_stereo_injection(mol, name: str) -> bool:
    """Return True iff *mol* carries CIP stereo not represented in *name*.

    Pure read-only predicate  used both by the namer.py backstop
    (after a phase refactor) and by inject_stereo_from_locant_map.

    Per, the three name-side detection patterns are byte-identical to
    namer.py:_final_stereo_check lines 62-75. Per, do NOT broaden.

    Args:
        mol: RDKit Mol object (may be None).
        name: Generated IUPAC name string (may be empty / 'unknown').

    Returns:
        True iff (a) mol is not None and name is non-empty and != 'unknown'
        AND (b) name does NOT match any of the three stereo-detection patterns
        AND (c) mol has at least one atom or bond with _CIPCode set after
        idempotent assign_stereochemistry.
    """
    if mol is None or not name or name == 'unknown':
        return False

    # Pattern A — prefix form (already-stereoed name;)
    if _STEREO_PREFIX_RE.match(name):
        return False
    # Pattern B — embedded block
    if _STEREO_EMBEDDED_RE.search(name):
        return False
    # Pattern C — carbohydrate / amino acid traditional notation
    if _CARBOHYDRATE_STEREO_RE.search(name):
        return False
    # Pattern D (a phase.0 /) — a D/L configurational token or a
    # peptide acyl chain means the name already carries its configuration.
    # SHARED primitive : both the namer backstop (_final_stereo_check) and
    # inject_stereo_from_locant_map delegate detection here, so this suppresses
    # both injection seams uniformly. Load-bearing for the Plan-02 backstop flip
    # (peptides report complex_ring/unknown/direct, NOT heterocycle — RESEARCH
    # Pitfall 1 — so the inject-allowlist exclusion is not sufficient on its own).
    if _DL_CONFIG_RE.search(name) or _PEPTIDE_ACYL_RE.search(name):
        return False
    # Pattern E (W5-A4) — an L-SUPPRESSED peptide name. With the L
    # descriptor omitted (alanylalanine, glycylalanine, valyltyrosylisoleucine),
    # Patterns C/D no longer fire, so the descriptor-free peptide would wrongly
    # look injection-eligible. name_peptide already carries all cited (D-) config
    # and correctly omits L, so treat a recognised peptide name as stereo-complete
    # (restores the Phase-177 Pattern-D protection for the post-W5-A4 name form).
    if _looks_like_peptide(name):
        return False

    # Idempotent CIP assignment (read-only — assign_stereochemistry uses
    # _Orthonym_CIPAssigned marker so re-entry is cheap and side-effect free
    # beyond what RDKit's CIP labeler already does).
    assign_stereochemistry(mol)
    has_atom_stereo = any(a.HasProp('_CIPCode') for a in mol.GetAtoms())
    has_bond_stereo = any(b.HasProp('_CIPCode') for b in mol.GetBonds())
    return has_atom_stereo or has_bond_stereo


def count_defined_stereo_elements(mol) -> int:
    """Count the DEFINED CIP stereogenic units on *mol* (Phase S Task 1).

    Counts, after idempotent CIP assignment:
      * every atom with ``_CIPCode`` (R/S/r/s tetrahedral + pseudoasymmetric);
      * every bond with ``_CIPCode`` EXCEPT ring bonds whose smallest ring is
        <8 (ring-strain-fixed geometry — not a free stereogenic unit; the SAME
        exclusion ``collect_stereodescriptors`` applies, so what is *counted*
        as defined matches exactly what CAN be expressed,;
      * every detected axial element with a determined CIP label.

    Read-only. Used by ``general_engine_stereo_complete`` for the all-or-nothing
    completeness gate on general-engine emissions.
    """
    if mol is None:
        return 0
    assign_stereochemistry(mol)
    n = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    ri = mol.GetRingInfo()
    for b in mol.GetBonds():
        if not b.HasProp('_CIPCode'):
            continue
        if b.IsInRing():
            min_ring = min((len(r) for r in ri.BondRings()
                            if b.GetIdx() in r), default=99)
            if min_ring < 8:
                continue  # ring-strain-fixed; matches the collector's skip
        n += 1
    try:
        for el in detect_axial_chirality(mol):
            if el.get('cip') is not None:
                n += 1
    except Exception:
        pass
    return n


def count_defined_stereo_in_fragment(mol, frag_atoms) -> int:
    """Count the DEFINED CIP stereogenic units that lie INSIDE ``frag_atoms``.

    The fragment-scoped twin of:func:`count_defined_stereo_elements`, for the
    substituent-prefix producers: a prefix names only its own fragment, so only
    the stereo elements inside that fragment are its obligation to express.

    Same three membership rules as the whole-molecule counter, so the two agree
    on any fragment that happens to be the whole molecule:

      * every fragment atom carrying ``_CIPCode``;
      * every bond with ``_CIPCode`` whose BOTH ends are in the fragment, EXCEPT
        a ring bond whose smallest ring is <8 — ``### **** Omission of
        stereodescriptors`` (``the Blue Book``) recommends omitting the
        descriptor for "*three- through seven-membered unsaturated alicyclic
        compounds where any double bond has a fixed configuration*", so such a
        bond is not an expressible unit and must not be demanded of the name;
      * axial elements are NOT counted. ``count_expressed_stereo_descriptors``
        cannot count an ``Ra``/``Sa`` token either, so counting them here would
        make a correctly-axial name look INCOMPLETE. Both sides omit them, which
        keeps the identity honest instead of biased.

    A bond with exactly ONE end in the fragment is deliberately excluded: its
    geometry is expressed by whoever names the atom on the other side (the
    parent), not by this prefix.

    Read-only apart from the idempotent CIP assignment.
    """
    if mol is None or not frag_atoms:
        return 0
    assign_stereochemistry(mol)
    frag = set(frag_atoms)
    n = sum(1 for i in frag if mol.GetAtomWithIdx(i).HasProp('_CIPCode'))
    ri = mol.GetRingInfo()
    for b in mol.GetBonds():
        if not b.HasProp('_CIPCode'):
            continue
        if b.GetBeginAtomIdx() not in frag or b.GetEndAtomIdx() not in frag:
            continue
        if b.IsInRing():
            min_ring = min((len(r) for r in ri.BondRings()
                            if b.GetIdx() in r), default=99)
            if min_ring < 8:
                continue  # ring-strain-fixed; not an expressible unit
        n += 1
    return n


def count_expressed_stereo_descriptors(name: str) -> int:
    """Count the stereodescriptor TOKENS the *name* actually carries.

    Sums the comma-separated descriptors across every ``(...)`` stereo block
    (leading, embedded, or nested substituent blocks), using the same block
    grammar as ``_STEREO_EMBEDDED_RE`` (R/S/r/s/E/Z with optional composite
    locants like ``7a``). Axial ``Ra``/``Sa``/``M``/``P`` blocks fall outside
    that grammar and are NOT counted — which only ever UNDER-counts, so the
    completeness gate fails CLOSED (the safe direction) rather than over-claim.
    """
    if not name:
        return 0
    total = 0
    for m in _STEREO_EMBEDDED_RE.finditer(name):
        inner = m.group(0).strip('()-')
        total += len([t for t in inner.split(',') if t])
    return total


def general_engine_stereo_complete(mol, name: str) -> bool:
    """ Phase S Task 1 (accuracy keystone): all-or-nothing stereo
    completeness for GENERAL-ENGINE emissions.

    Returns True iff *name* expresses EXACTLY every defined CIP stereo element
    *mol* carries (descriptor count == defined count). This REPLACES the coarse
    name-side boolean (``not needs_stereo_injection``) at the general-engine
    emission sites, closing the verified hole where a PARTIAL-stereo name (some
    elements expressed, others dropped) matched Pattern A and shipped as if
    fully specified — invisible to the stereo-blind: a PIN
    must specify every stereogenic unit).

    The exact ``==`` (not ``>=``) ALSO fail-closes on OVER-expression (a
    spurious / double-counted descriptor, e.g. the nested-block double-apply
    bug) — a name that cites MORE stereo than the structure defines is an
    attribution error and must not ship as complete. Any mismatch -> False
    (fail-closed; best-effort then ships the flagged constitution-superset name,
    complete abstains). Axial ``Ra``/``Sa`` tokens are not countable, so a name
    expressing axial chirality fails closed here (safe; axial detection is out
    of Phase S scope).
    """
    if mol is None or not name:
        return False
    return (count_expressed_stereo_descriptors(name)
            == count_defined_stereo_elements(mol))


def _inject_parent_block_beside_prefix_blocks(
    name: str,
    mol,
    atom_to_locant: Optional[Dict[int, int]],
) -> str:
    """The parent's own block for a name whose only stereo blocks sit
    inside substituent prefixes; else *name* unchanged.

     (the Blue Book): a substituent's descriptors are cited "at
    the front of the corresponding prefix", the parent's at the front of the
    complete name. ``needs_stereo_injection`` treats ANY embedded block as
    "stereo already present", so once a composed prefix carried its own block
    ('5-{(1E,3E)-8-[...]-9-oxadeca-1,3-dien-1-yl}-...', since 5620d69f2) the
    parent's '(21E,23Z,...)' block was never written and the name denoted the
    stereo-incomplete molecule.

    Fail-closed: only a one-word name with no leading block and no traditional
    (D/L, alpha/beta, peptide) configuration; only the parent's descriptors
    (``collect_stereodescriptors`` reads the locant map's atoms only); and only
    when the new block exactly completes the count of defined stereo elements
    (so no descriptor is ever cited twice). Every caller's round trip still
    verifies the result."""
    if (mol is None or not name or ' ' in name
            or _STEREO_PREFIX_RE.match(name)
            or not _STEREO_EMBEDDED_RE.search(name)
            or _CARBOHYDRATE_STEREO_RE.search(name)
            or _DL_CONFIG_RE.search(name) or _PEPTIDE_ACYL_RE.search(name)
            or _looks_like_peptide(name)):
        return name
    if not atom_to_locant or not any(
        isinstance(v, int) and v > 0 for v in atom_to_locant.values()
    ):
        return name
    descriptors = collect_stereodescriptors(mol, atom_to_locant)
    if not descriptors:
        return name
    if (count_expressed_stereo_descriptors(name) + len(descriptors)
            != count_defined_stereo_elements(mol)):
        return name
    return f"{format_stereodescriptor_string(descriptors)}{name}"


def inject_stereo_from_locant_map(
    name: str,
    mol,
    atom_to_locant: Optional[Dict[int, int]],
    *,
    include_near_parent_ez: bool = True,
) -> str:
    """Prepend a stereo descriptor block to *name* using authoritative locants.

    Per IUPAC /, prepends a `(R/S/E/Z)-` block built from
    collect_stereodescriptors + format_stereodescriptor_string.

    Per, **no atom-index fallback**: when atom_to_locant is None, empty,
    or all-zero (degenerate), returns *name* unchanged and emits a single
    DEBUG log line. The backstop in namer.py will still log WARNING in this
    case so handler attribution is preserved.

    Args:
        name: The candidate IUPAC name from a handler.
        mol: RDKit Mol object with stereo info.
        atom_to_locant: Authoritative {atom_idx: 1-indexed locant} map from
            the handler's own perception (heterocycle / benzene / cycloalkane
            / cycloalkene). Must NOT be derived from raw atom indices .
        include_near_parent_ez: When True (default -- preserves benzene /
            heterocycle Tier-A behaviour), exocyclic E/Z bonds one hop from
            the parent are attributed to the lowest neighbouring locant per
            . When False (cycloalkane / cycloalkene caller post-
             fix), exocyclic E/Z bonds are NOT attributed to ring
            locants; only ring-atom R/S and ring-bond E/Z are emitted. This
            is the conservative gate per ("better a missing stereo
            block than a wrong one") for handlers where exocyclic E/Z can
            be mis-attributed via include_near_parent_ez=True.

    Returns:
        name unchanged (predicate False / no locant map / no descriptors)
        OR prefix + name where prefix is e.g. '(2R)-', '(2R,3S)-',
        '(2E,3R,5Z)-', '(2r,3s)-' per.

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> inject_stereo_from_locant_map('butan-2-ol', mol, {1: 2})
        '(2R)-butan-2-ol'
    """
    if not needs_stereo_injection(mol, name):
        return _inject_parent_block_beside_prefix_blocks(
            name, mol, atom_to_locant)

    #: hard precondition — no atom-index fallback.
    # fix (a phase-02, 2026-05-03): also reject all-zero / non-positive
    # locant maps. A locant of 0 or negative is IUPAC-malformed (locants are
    # 1-indexed); accepting it would emit '(0R)-name' or '(-1R)-name' garbage.
    # Per ("missing > wrong"), skip injection.
    if not atom_to_locant or not any(
        isinstance(v, int) and v > 0 for v in atom_to_locant.values()
    ):
        logger.debug(
            "inject_stereo: skipped, no locant map / all-zero locants (name=%r)",
            name[:50],
        )
        return name

    #: include_near_parent_ez defaults to True for compliance
    # (top-level only; is_top_level_naming guard at the call site enforces
    # this). fix (a phase-02, 2026-05-03): cycloalkane / cycloalkene
    # caller passes include_near_parent_ez=_ring_is_whole_molecule so chain-
    # side exocyclic E/Z is NOT attributed to ring locants.
    descriptors = collect_stereodescriptors(
        mol, atom_to_locant, include_near_parent_ez=include_near_parent_ez,
    )
    if not descriptors:
        return name

    prefix = format_stereodescriptor_string(descriptors)
    return f"{prefix}{name}"


def inject_stereo_reanchored_rt_gated(
    base_name: str,
    mol,
    builder_map: Optional[Dict[int, Any]],
    *,
    include_near_parent_ez: bool = True,
    input_smiles: Optional[str] = None,
) -> str:
    """Inject a stereo block on *base_name*, RT-gating the LOCANT numbering
    (the contributor guide a project rule — offer numberings, keep the one that round-trips).

    Candidate A uses ``builder_map`` (the handler's own numbering) exactly as
    ``inject_stereo_from_locant_map`` does. If A full-round-trips (name -> OPSIN
    -> InChI == input's InChI), A is returned — BYTE-IDENTICAL to the plain
    injector for every currently-passing name. Only when A does NOT full-round-
    trip is the numbering RE-ANCHORED to OPSIN's OWN locants for *base_name*
    (``opsin_atom_locant_map``, the authoritative numbering the name is read back
    with) and candidate B injected on that map; B is returned ONLY if it
    full-round-trips. Otherwise A is returned unchanged.

    This rescues the mixed-spiro-fused leak class (a spiro-of-fused-component
    parent whose ``combined_locants`` map is numbered inconsistently with the
    printed descriptor, so the stereo descriptor lands on the wrong locant and
    the full name is OPSIN-unparseable) WITHOUT disturbing a legitimate
    OPSIN-can't-parse-the-stereo-layer carve-out PIN: that PIN's re-anchored form
    also fails full-RT (OPSIN cannot parse the layer in ANY spelling), so A is
    kept. Fail-OPEN on any OPSIN unavailability -> returns A (current behaviour).
    """
    candidate_a = inject_stereo_from_locant_map(
        base_name, mol, builder_map,
        include_near_parent_ez=include_near_parent_ez,
    )
    # Nothing was injected (no stereo needed / map rejected) AND the flat name is
    # what we have: still RT-gate below, because a rejected map is exactly the
    # case a re-anchor can rescue.
    try:
        from ..validation.opsin_roundtrip import (
            opsin_atom_locant_map,
            opsin_roundtrip_check,
        )
    except Exception:
        return candidate_a
    if input_smiles is None:
        try:
            input_smiles = Chem.MolToSmiles(mol)
        except Exception:
            return candidate_a
    try:
        if opsin_roundtrip_check(input_smiles, candidate_a)["passed"]:
            return candidate_a  # already correct -> unchanged (byte-identical)
    except Exception:
        return candidate_a  # RT unavailable -> fail-OPEN to current behaviour
    # Candidate A does not full-round-trip: re-anchor to OPSIN's own numbering.
    try:
        rmap = opsin_atom_locant_map(base_name, mol)
    except Exception:
        rmap = None
    if rmap:
        candidate_b = inject_stereo_from_locant_map(
            base_name, mol, rmap,
            include_near_parent_ez=include_near_parent_ez,
        )
        if candidate_b != candidate_a:
            try:
                if opsin_roundtrip_check(input_smiles, candidate_b)["passed"]:
                    logger.debug(
                        "stereo re-anchor: %r -> %r (RT-gated)",
                        candidate_a[:60], candidate_b[:60])
                    return candidate_b
            except Exception:
                pass
    # Wave E — 0-wrong hardening. Re-anchor FAILED (no OPSIN locant map, or
    # candidate B still does not full-round-trip): the stereo descriptor cannot be
    # placed on a numbering OPSIN reads back correctly. Returning candidate A here
    # ships a name whose stereo layer OPSIN cannot verify — the OPSIN-validity stereo
    # carve-out (namer.py) then emits it whole because its CONSTITUTION parses,
    # i.e. a WRONG/unverifiable-stereo name reaches (a residual 0-wrong leak).
    # Instead return the stereo-STRIPPED FLAT name: constitution-correct,
    # OPSIN-parseable, stereo OMITTED per project policy
    # (feedback_stereo_omission_is_not_wrong_molecule). ``base_name`` is the
    # pre-injection constitution name by the caller's ``needs_stereo_injection``
    # contract; ``strip_stereo`` is applied belt-and-suspenders so any leading
    # descriptor that did survive on it is also dropped.
    return strip_stereo(base_name)


def _ring_atom_to_locant_from_oriented(oriented_ring: List[int]) -> Dict[int, int]:
    """Build atom-idx → 1-indexed locant map from oriented_ring.

    Single source of truth for the {idx: pos+1} formula previously inlined
    at composer.py:7184. Used by benzene / cycloalkane / cycloalkene
    handler wiring per.

    Args:
        oriented_ring: List of atom indices in IUPAC ring-position order
            (position 1 first).

    Returns:
        Dict mapping each atom idx to its 1-indexed locant. When duplicate
        atom indices appear, the LAST occurrence wins (matches dict semantics
        of the original one-liner at composer.py:7184). fix
        (a phase-02, 2026-05-03): also emits a WARNING log when duplicates
        are present so a buggy upstream orientator does not silently produce
        a wrong locant map.

    Example:
        >>> _ring_atom_to_locant_from_oriented([10, 11, 12, 13, 14, 15])
        {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6}
    """
    if oriented_ring and len(set(oriented_ring)) != len(oriented_ring):
        logger.warning(
            "_ring_atom_to_locant_from_oriented: duplicate atom indices in "
            "oriented_ring=%r -- last position wins (upstream orientator may have a bug)",
            oriented_ring,
        )
    return {idx: pos + 1 for pos, idx in enumerate(oriented_ring)}


def collect_ring_stereodescriptors(
    mol,
    ring_atom_to_locant: Dict[int, int]
) -> List[Tuple[int, str]]:
    """
    Collect stereodescriptors for ring compounds.

    Same as collect_stereodescriptors but specifically for ring naming,
    where only atoms in the ring are considered.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        ring_atom_to_locant: Mapping from ring atom indices to IUPAC locants.
                            Only atoms that are keys in this dict are included.

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.

    Note:
        For rings, double bond E/Z is less common (ring strain), but
        we still handle it for completeness.
    """
    # Reuse the main function - it already handles the filtering correctly
    return collect_stereodescriptors(mol, ring_atom_to_locant)


def determine_ring_cis_trans(
    mol,
    ring_atoms: List[int],
    sub1_idx: int,
    sub2_idx: int
) -> Optional[str]:
    """
    Determine if two substituents on a ring are cis or trans.

    For a ring with exactly 2 substituents at specified positions,
    determine their relative stereochemistry based on CIP labels.

    The rule for 1,2-disubstituted rings:
    - SAME CIP codes (R,R or S,S) -> CIS (substituents on same face)
    - DIFFERENT CIP codes (R,S or S,R) -> TRANS (substituents on opposite faces)

    This is because in a ring, atoms with the same absolute configuration
    at adjacent positions have their substituents on the same face.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices that form the ring
        sub1_idx: Atom index of first substituted ring carbon
        sub2_idx: Atom index of second substituted ring carbon

    Returns:
        'cis' or 'trans', or None if cannot be determined (missing CIP labels)

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C') # cis-1,2-dimethylcyclohexane
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> determine_ring_cis_trans(mol, ring_atoms, 1, 6)
        'cis'
    """
    # Both atoms must be in the ring
    if sub1_idx not in ring_atoms or sub2_idx not in ring_atoms:
        return None

    # Get atoms
    atom1 = mol.GetAtomWithIdx(sub1_idx)
    atom2 = mol.GetAtomWithIdx(sub2_idx)

    # Both must have CIP labels
    if not atom1.HasProp('_CIPCode') or not atom2.HasProp('_CIPCode'):
        return None

    cip1 = atom1.GetProp('_CIPCode')
    cip2 = atom2.GetProp('_CIPCode')

    # Only handle R/S (not r/s pseudoasymmetric for now)
    if cip1 not in ['R', 'S'] or cip2 not in ['R', 'S']:
        return None

    # Same CIP = cis, Different CIP = trans
    if cip1 == cip2:
        return 'cis'
    else:
        return 'trans'


def get_simple_ring_stereo(
    mol,
    ring_atoms: List[int],
    ring_substituents: Dict[int, str]
) -> Optional[str]:
    """
    Get cis/trans prefix for simple disubstituted rings.

    This function handles the common case of exactly 2 substituted positions
    on a ring, returning a cis- or trans- prefix for the name.

    Note: This is a simplification. More complex rings (3+ substituents)
    would need the full IUPAC r/c/t reference system, which is deferred.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        ring_substituents: Dict mapping ring atom locant -> substituent name.
                          Only keys (locants) are used to identify substituted positions.

    Returns:
        'cis-' or 'trans-' prefix string, or None if:
        - Not exactly 2 substituted positions
        - Cannot determine stereochemistry

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> # Assuming oriented_ring maps locant -> atom_idx
        >>> ring_substituents = {1: 'methyl', 2: 'methyl'}
        >>> get_simple_ring_stereo(mol, ring_atoms, ring_substituents)
        'cis-'
    """
    # Need exactly 2 substituted positions for simple cis/trans
    if len(ring_substituents) != 2:
        return None

    # Get the locants with substituents
    locants = list(ring_substituents.keys())

    # We need to find which atom indices correspond to these locants
    # This requires knowing the atom_to_locant mapping
    # Since we don't have it directly, we'll need the caller to provide
    # the actual atom indices

    # For now, return None and let the caller handle the mapping
    # This function needs the actual atom indices, not locants
    return None


def get_simple_ring_stereo_from_atoms(
    mol,
    ring_atoms: List[int],
    substituted_atom_indices: List[int]
) -> Optional[str]:
    """
    Get cis/trans prefix given the actual atom indices of substituted positions.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        substituted_atom_indices: List of exactly 2 atom indices that have substituents

    Returns:
        'cis-' or 'trans-' prefix string, or None if cannot determine

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> get_simple_ring_stereo_from_atoms(mol, ring_atoms, [1, 6])
        'cis-'
    """
    if len(substituted_atom_indices) != 2:
        return None

    result = determine_ring_cis_trans(
        mol,
        ring_atoms,
        substituted_atom_indices[0],
        substituted_atom_indices[1]
    )

    if result:
        return f'{result}-'
    return None


def format_ring_stereo_with_descriptors(
    ring_cis_trans: Optional[str],
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format ring stereo prefix with optional R/S descriptors.

    IUPAC format for ring stereo:
    - If only cis/trans: "cis-1,2-dimethylcyclohexane"
    - If cis/trans + R/S: "cis-(1R,2S)-1,2-dimethylcyclohexane"
    - The cis/trans goes BEFORE the stereodescriptors

    Args:
        ring_cis_trans: 'cis-' or 'trans-' prefix, or None
        descriptors: List of (locant, cip_code) tuples from collect_stereodescriptors

    Returns:
        Combined prefix string like "cis-", "trans-(1R,2S)-", etc.
        Empty string if no stereo information.

    Example:
        >>> format_ring_stereo_with_descriptors('cis-', [(1, 'R'), (2, 'S')])
        'cis-(1R,2S)-'
        >>> format_ring_stereo_with_descriptors('trans-', )
        'trans-'
        >>> format_ring_stereo_with_descriptors(None, [(1, 'R')])
        '(1R)-'
    """
    rs_string = format_stereodescriptor_string(descriptors)

    if ring_cis_trans:
        if rs_string:
            return f'{ring_cis_trans}{rs_string}'
        else:
            return ring_cis_trans
    else:
        return rs_string


# =============================================================================
# Ring Junction Stereochemistry Functions
# =============================================================================

def get_bridgehead_atoms(mol) -> List[int]:
    """
    Find atoms at ring fusion points (bridgehead atoms).

    Bridgehead atoms are atoms that are shared by multiple rings. For fused
    ring systems like decalin, these are the atoms at the junction where
    rings meet.

    Args:
        mol: RDKit Mol object

    Returns:
        List of atom indices that are bridgehead atoms (in 2+ rings and sp3)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1') # decalin
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> len(bridgeheads) # 2 bridgehead atoms
        2
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1') # naphthalene (aromatic)
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> len(bridgeheads) # 0 - sp2 atoms are not stereocenters
        0
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return []

    # Count how many rings each atom belongs to
    atom_ring_count: Dict[int, int] = {}
    for ring in atom_rings:
        for atom_idx in ring:
            atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1

    # Find atoms in 2+ rings that are sp3 (potential stereocenters)
    bridgehead_atoms = []
    for atom_idx, count in atom_ring_count.items():
        if count >= 2:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Check if sp3 (saturated) - these can be stereocenters
            # Hybridization SP3 indicates tetrahedral geometry
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP3:
                bridgehead_atoms.append(atom_idx)

    return sorted(bridgehead_atoms)


def collect_ring_junction_stereo(
    mol,
    bridgehead_atoms: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Tuple[str, str]]:
    """
    Collect stereodescriptors for ring junction (bridgehead) atoms.

    Ring junction atoms in fused systems use 'a' suffix locants in IUPAC naming.
    For example, in decahydronaphthalene, the junction atoms are 4a and 8a.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        bridgehead_atoms: List of atom indices at ring junctions
        atom_to_locant: Mapping from atom index to IUPAC locant (int or str)
                       The locant should already include 'a' suffix if needed

    Returns:
        List of (locant_str, cip_code) tuples, sorted by locant.
        Locant is string to handle 'a' suffix (e.g., '4a', '8a').

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1') # cis-decalin
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> atom_to_locant = {3: '4a', 8: '8a'} # junction atoms with 'a' suffix
        >>> stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)
        >>> # Returns [('4a', 'S'), ('8a', 'S')] or similar
    """
    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    descriptors: List[Tuple[str, str]] = []

    for atom_idx in bridgehead_atoms:
        if atom_idx not in atom_to_locant:
            continue

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.HasProp('_CIPCode'):
            locant = atom_to_locant[atom_idx]
            # Convert to string if int, to handle both int and 'a' suffix locants
            locant_str = str(locant)
            cip_code = atom.GetProp('_CIPCode')
            descriptors.append((locant_str, cip_code))

    # Sort by locant (handle '4a', '8a' style locants)
    def _junction_locant_sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        """Sort key for junction locants like '4a', '8a'."""
        locant_str = item[0]
        if locant_str and locant_str[-1].isalpha():
            return (int(locant_str[:-1]), locant_str[-1])
        return (int(locant_str), '')

    descriptors.sort(key=_junction_locant_sort_key)

    return descriptors


def format_ring_junction_stereo(
    descriptors: List[Tuple[str, str]],
    use_rct: bool = False
) -> str:
    """
    Format ring junction stereodescriptors as IUPAC name prefix.

    IUPAC 2013 provides two notations for ring junction stereo:
    1. R/S notation (PIN style): "(4aR,8aS)-"
    2. r/c/t notation (reference plane): "(4ar,8ac)-"

    The r/c/t notation uses:
    - r: reference stereocenter (first one)
    - c: cis to reference (same CIP code)
    - t: trans to reference (different CIP code)

    Args:
        descriptors: List of (locant_str, cip_code) tuples from collect_ring_junction_stereo
        use_rct: If True, use r/c/t notation; if False (default), use R/S

    Returns:
        Formatted string like "(4aR,8aS)-" or "(4ar,8ac)-"
        Returns empty string if no descriptors.

    Examples:
        >>> format_ring_junction_stereo([('4a', 'S'), ('8a', 'S')])
        '(4aS,8aS)-'
        >>> format_ring_junction_stereo([('4a', 'S'), ('8a', 'S')], use_rct=True)
        '(4ar,8ac)-'
        >>> format_ring_junction_stereo([('4a', 'R'), ('8a', 'S')], use_rct=True)
        '(4ar,8at)-'
    """
    if not descriptors:
        return ""

    if use_rct:
        # Convert to r/c/t notation
        # First stereocenter is reference (r)
        # Same CIP as reference = cis (c)
        # Different CIP = trans (t)
        rct_parts = []
        ref_cip = None

        for i, (locant, cip) in enumerate(descriptors):
            if i == 0:
                # First is reference
                ref_cip = cip
                rct_parts.append(f"{locant}r")
            else:
                # Compare to reference
                if cip == ref_cip:
                    rct_parts.append(f"{locant}c")  # cis
                else:
                    rct_parts.append(f"{locant}t")  # trans

        return f"({','.join(rct_parts)})-"
    else:
        # Standard R/S notation
        parts = [f"{locant}{cip}" for locant, cip in descriptors]
        return f"({','.join(parts)})-"


def determine_simple_cis_trans(
    mol,
    junction_atoms: List[int]
) -> Optional[str]:
    """
    Determine cis/trans for simple bicyclic systems (decalin type).

    For simple fused bicyclics with exactly 2 junction atoms:
    - Same chiral tag (both CCW or both CW) = cis (both H on same face)
    - Different chiral tags (one CCW, one CW) = trans (H on opposite faces)

    Note: CIP codes (R/S) are NOT reliable for cis/trans determination in
    symmetric fused systems like decalin, because the molecular symmetry
    can cause both junction atoms to have the same CIP code even in trans.
    Instead, we use the ChiralTag which reflects the actual tetrahedral
    configuration.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        junction_atoms: List of exactly 2 atom indices at ring junction

    Returns:
        'cis' or 'trans', or None if:
        - Not exactly 2 junction atoms
        - Missing stereochemistry on junction atoms

    Examples:
        >>> # cis-decalin: both [C@@H] = same ChiralTag = cis
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> determine_simple_cis_trans(mol, bridgeheads)
        'cis'
        >>> # trans-decalin: [C@@H]...[C@H] = different ChiralTag = trans
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> determine_simple_cis_trans(mol, bridgeheads)
        'trans'
    """
    if len(junction_atoms) != 2:
        return None

    atom1 = mol.GetAtomWithIdx(junction_atoms[0])
    atom2 = mol.GetAtomWithIdx(junction_atoms[1])

    # Get chiral tags
    tag1 = atom1.GetChiralTag()
    tag2 = atom2.GetChiralTag()

    # Both must have defined chirality
    unspecified = Chem.ChiralType.CHI_UNSPECIFIED
    if tag1 == unspecified or tag2 == unspecified:
        return None

    # Also reject CHI_OTHER which indicates unknown/invalid
    other = Chem.ChiralType.CHI_OTHER
    if tag1 == other or tag2 == other:
        return None

    # Compare chiral tags:
    # CHI_TETRAHEDRAL_CW and CHI_TETRAHEDRAL_CCW are the common stereo tags
    # Same tag = cis (both H on same face)
    # Different tag = trans (H on opposite faces)
    if tag1 == tag2:
        return 'cis'
    else:
        return 'trans'


def get_junction_locants_for_fused_system(
    mol,
    bridgehead_atoms: List[int],
    ring_size_1: int,
    ring_size_2: int
) -> Dict[int, str]:
    """
    Generate IUPAC 'a' suffix locants for junction atoms in fused systems.

    For ortho-fused bicyclics like decahydronaphthalene:
    - Ring 1 (6-membered): positions 1-4, then 4a
    - Ring 2 (6-membered): positions 4a-8, then 8a
    - Junction atoms get 'a' suffix locants (4a, 8a)

    This is a simplified implementation for common cases.

    Args:
        mol: RDKit Mol object
        bridgehead_atoms: List of atom indices at ring junctions
        ring_size_1: Size of first ring
        ring_size_2: Size of second ring

    Returns:
        Dict mapping atom index to locant string (e.g., {3: '4a', 8: '8a'})

    Note:
        This is a simplified mapping. Full IUPAC numbering requires
        consideration of ring system orientation and heteroatom positions.
    """
    # For 6,6-fused (decalin/naphthalene type):
    # Total unique positions = ring1 + ring2 - 2 shared atoms
    # Standard IUPAC: 4a and 8a for 6,6-fused
    if ring_size_1 == 6 and ring_size_2 == 6 and len(bridgehead_atoms) == 2:
        # Simple case: assign 4a and 8a
        return {
            bridgehead_atoms[0]: '4a',
            bridgehead_atoms[1]: '8a'
        }

    # For 5,6-fused (indane type): 3a and 7a
    if (ring_size_1 == 5 and ring_size_2 == 6) or (ring_size_1 == 6 and ring_size_2 == 5):
        if len(bridgehead_atoms) == 2:
            return {
                bridgehead_atoms[0]: '3a',
                bridgehead_atoms[1]: '7a'
            }

    # For 5,5-fused (pentalene type): 3a and 6a
    # (pentalene numbering 1,2,3,3a,4,5,6,6a — junctions 3a/6a, NOT 3a/5a).
    if ring_size_1 == 5 and ring_size_2 == 5 and len(bridgehead_atoms) == 2:
        return {
            bridgehead_atoms[0]: '3a',
            bridgehead_atoms[1]: '6a'
        }

    # For 5,7-fused (azulene type): 3a and 8a
    # (azulene numbering 1,2,3,3a,4,5,6,7,8,8a — junctions 3a/8a).
    if ((ring_size_1 == 5 and ring_size_2 == 7) or
            (ring_size_1 == 7 and ring_size_2 == 5)) and len(bridgehead_atoms) == 2:
        return {
            bridgehead_atoms[0]: '3a',
            bridgehead_atoms[1]: '8a'
        }

    # For 6,7-fused (heptalene type): 4a and 9a
    # (heptalene numbering 1,2,3,4,4a,5,6,7,8,9,9a — junctions 4a/9a).
    if ((ring_size_1 == 6 and ring_size_2 == 7) or
            (ring_size_1 == 7 and ring_size_2 == 6)) and len(bridgehead_atoms) == 2:
        return {
            bridgehead_atoms[0]: '4a',
            bridgehead_atoms[1]: '9a'
        }

    # Generic fallback: use sequential 'a' suffix locants
    result = {}
    # Calculate locant based on position in combined ring system
    # For generic case, use (ring_size_1 - 1)a pattern
    if len(bridgehead_atoms) >= 1:
        result[bridgehead_atoms[0]] = f'{ring_size_1 - 1}a'
    if len(bridgehead_atoms) >= 2:
        total_positions = ring_size_1 + ring_size_2 - 2
        result[bridgehead_atoms[1]] = f'{total_positions}a'

    return result
