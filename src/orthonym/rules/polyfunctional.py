"""
Polyfunctional compound naming coordinator.

Handles naming of compounds with multiple functional groups:
- Principal group (highest seniority) becomes the suffix
- Lower-seniority groups become prefixes with locants
- Ethers (no seniority) are always named as alkoxy prefixes

Based on IUPAC 2013 Blue Book P-41 to P-43.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from ..assembly.naming_utils import (
    BRANCH_HANDLED_FGS,
    format_suffix_with_locants,
    get_alkyl_name,
    get_multiplier_prefix,
    get_suffix_multiplier_prefix,
    prefix_citation_sort_key,
)
from ..assembly.substituent_prefix_forms import (
    get_alkoxy_prefix as _ASSEMBLY_get_alkoxy_prefix,
)
from ..assembly.substituent_prefix_forms import (
    get_alkoxycarbonyl_prefix as _ASSEMBLY_get_alkoxycarbonyl_prefix,
)

# Phase 160.1 CONTEXT: lift the 5 prefix-form generators + dispatcher to
# assembly/substituent_prefix_forms.py. The functions below remain as thin
# re-exports preserving the existing polyfunctional.py public API.
from ..assembly.substituent_prefix_forms import (
    get_substituent_prefix_form as _ASSEMBLY_get_substituent_prefix_form,
)
from ..assembly.substituent_prefix_forms import (
    get_sulfanyl_prefix as _ASSEMBLY_get_sulfanyl_prefix,
)
from ..assembly.substituent_prefix_forms import (
    get_sulfinyl_prefix as _ASSEMBLY_get_sulfinyl_prefix,
)
from ..assembly.substituent_prefix_forms import (
    get_sulfonyl_prefix as _ASSEMBLY_get_sulfonyl_prefix,
)
from ..errors import is_refusal_sentinel
from .ring_unsaturation import render_ring_unsaturation
from .seniority import (
    PREFIX_FORMS,
    get_prefix,
    get_principal_group,
)

# Multiplier stems for skeletal-unsaturation infixes (di/tri/tetra…), P-31.1.4.
_RING_UNSAT_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa'}


def _ring_stem_with_unsaturation(stem, ene_locants, yne_locants):
    """Assemble ``<stem>[a]-<locs>-[mult]en[-<locs>-[mult]yn]`` from DISPLAY
    locant strings (``render_ring_unsaturation`` already applies the P-31.1.4.2(1)
    compound ``lo(hi)`` form), leaving the trailing 'e' to the suffix formatter.

    ``cyclohex`` + ene=['2']          -> ``cyclohex-2-en``
    ``cyclohex`` + ene=['2','4']      -> ``cyclohexa-2,4-dien`` (euphonic 'a')
    Per P-31.1.4: the stem keeps its 'a' before a multiplied unsaturation infix
    but elides it before a single 'ene'.
    """
    has_mult = len(ene_locants) >= 2 or len(yne_locants) >= 2
    out = stem + ('a' if has_mult else '')
    if ene_locants:
        m = _RING_UNSAT_MULT.get(len(ene_locants), '') if len(ene_locants) > 1 else ''
        out += '-' + ','.join(ene_locants) + '-' + m + 'en'
    if yne_locants:
        m = _RING_UNSAT_MULT.get(len(yne_locants), '') if len(yne_locants) > 1 else ''
        out += '-' + ','.join(yne_locants) + '-' + m + 'yn'
    return out


# W2F-P2 (P-65.6.3.3.5 method (1)): the composite-FG split runs by DEFAULT for
# exactly this principal-group class (acid-principal partial (thio)esters whose
# carbonyl is a chain member). v1 scope; extensible. Every default split is
# per-candidate OPSIN-RT gated in split_composite_fg (fail-closed).
_SPLIT_DEFAULT_PGS = frozenset({"carboxylic_acid"})


# Functional groups that can be detected but have no seniority
# These are always named as prefixes (never suffix)
NO_SENIORITY_GROUPS = {
    "ether",
    "vinyl_ether",
    "aromatic_ether",
    "fluoro",
    "chloro",
    "bromo",
    "iodo",
    "nitro",
    "nitroso",
    "azido",
    # Wave2 T2a (P-61.8): the substitutive 'isocyanato'/'isothiocyanato'
    # prefixes are the PINs (BB VERBATIM 'isocyanatocyclohexane (PIN)
    # cyclohexyl isocyanate'), so with no senior group present the parent
    # hydride is named with the prefix (isocyanatoethane) — the functional-
    # class handlers now decline under style='pin' and keep the 'ethyl
    # isocyanate' form for general (--trivial) nomenclature.
    "isocyanate",
    "isothiocyanate",
    "alkene",  # Handled separately via unsaturation
    "alkyne",  # Handled separately via unsaturation
}

# Chain prefixes for ether alkoxy naming (match ALKYL_NAMES pattern)
ALKOXY_NAMES = {
    1: "methoxy",
    2: "ethoxy",
    3: "propoxy",
    4: "butoxy",
    5: "pentyloxy",
    6: "hexyloxy",
    7: "heptyloxy",
    8: "octyloxy",
    9: "nonyloxy",
    10: "decyloxy",
}


# PERC-06 /: Normalize FG subtypes to parent class for polyfunctional counting.
# IUPAC treats primary/secondary/tertiary alcohol as the same FG class.
PARENT_CLASS_MAP = {
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "phenol": "alcohol",
    "enol": "alcohol",
    "alcohol": "alcohol",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aromatic_amine": "amine",
}


def _drop_union_class_subtypes(
    non_principal: Dict[str, List[tuple]], principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """Remove non-principal FGs already absorbed into the principal multiplied suffix.

    DD5 RC-4 / SEN-03 parity (audit fix 2026-06-22): ``get_principal_group`` unions
    every present subtype of the principal group's equal-seniority *union* class
    (alcohols, ``_RC4_UNION_CLASSES``) into ``principal_group_atoms``, so a mixed
    primary+secondary OH chain is expressed as ONE multiplied suffix
    (``…-1,3-diol``). Such same-class subtypes therefore must NOT also leak into
    the non-principal prefix loop — doing so double-listed the secondary OH as both
    a ``hydroxy`` prefix and a ``-ol`` suffix locant, producing OPSIN-invalid names
    like ``2-amino-3-hydroxyoctadecane-1,3-diol`` (sphinganine) / geminal forms.

    This mirrors the identical skip in
    ``assembly/handlers/_handler_shared._generate_prefixes`` (the general-acyclic
    path); ``polyfunctional`` is the parallel multi-FG handler and needs the same
    guard. Scoped to ``_RC4_UNION_CLASSES`` (alcohols only) via the shared
    ``_SENIORITY_PARENT`` map, so amine/other subtypes are untouched.
    """
    if not principal_group:
        return non_principal
    from .seniority import _RC4_UNION_CLASSES, _SENIORITY_PARENT
    pclass = _SENIORITY_PARENT.get(principal_group, principal_group)
    if pclass not in _RC4_UNION_CLASSES:
        return non_principal
    return {
        k: v for k, v in non_principal.items()
        if _SENIORITY_PARENT.get(k, k) != pclass
    }


def detect_polyfunctional(mol, functional_groups: Dict[str, List[tuple]]) -> bool:
    """
    Determine if a molecule has multiple distinct functional groups.

    Returns True if the molecule has 2+ distinct functional groups that
    participate in naming (excludes unsaturation markers and halogens).

    A compound with multiple instances of the SAME group (e.g., diol)
    is NOT considered polyfunctional by this function - that's multiplicity.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()

    Returns:
        True if molecule is polyfunctional (multiple distinct FGs)
    """
    # Groups that count as "functional" for polyfunctional detection
    # Excludes: alkene/alkyne (unsaturation), halogens (always prefix anyway)
    exclude_groups = {
        "alkene", "alkyne",
        "fluoro", "chloro", "bromo", "iodo",
    }

    # Count distinct functional group types with seniority or special handling
    distinct_groups = set()

    for fg_name, matches in functional_groups.items():
        if not matches:
            continue
        if fg_name in exclude_groups:
            continue

        # Normalize to parent class before counting (PERC-06 /)
        parent = PARENT_CLASS_MAP.get(fg_name, fg_name)
        distinct_groups.add(parent)

    return len(distinct_groups) >= 2


def get_fg_prefix_form(
    fg_name: str,
    mol,
    atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """
    Get the prefix form for a functional group.

    For most groups, uses the standard prefix from seniority.py.
    For ethers, determines the alkoxy prefix based on substituent size.

    Phase 160.1 CONTEXT: chemistry rules lifted to
    assembly/substituent_prefix_forms.py. This shim consults the new
    dispatcher first; on None (out-of-14-row-set), falls back to the
    static PREFIX_FORMS lookup via get_prefix() — preserving the original
    polyfunctional caller's expectation that any FG in PREFIX_FORMS
    (e.g., hydroxyl, amino, carboxy) gets its static prefix form returned.

    Args:
        fg_name: Name of the functional group
        mol: RDKit Mol object
        atoms: Atom indices matching this FG instance
        principal_chain: Atom indices of the principal chain

    Returns:
        Prefix string (e.g., "hydroxy", "oxo", "methoxy") or None if no prefix
    """
    # Phase 160.1: consult the 14-row dispatcher first (P-65 / P-66 forms).
    result = _ASSEMBLY_get_substituent_prefix_form(fg_name, mol, atoms, principal_chain)
    if result is not None:
        return result

    # Fall through to the static PREFIX_FORMS table for FGs not in the
    # 14-row Phase 160.1 closed-set (e.g., carboxylic_acid → "carboxy",
    # primary_alcohol → "hydroxy", primary_amine → "amino").
    return get_prefix(fg_name)


def _get_alkoxy_prefix(
    mol,
    ether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_alkoxy_prefix.

    Phase 160.1 CONTEXT: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_alkoxy_prefix(mol, ether_atoms, principal_chain)


def _get_alkoxycarbonyl_prefix(
    mol,
    ester_atoms: tuple,
    principal_chain: List[int],
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_alkoxycarbonyl_prefix.

    Phase 160.1 CONTEXT: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_alkoxycarbonyl_prefix(mol, ester_atoms, principal_chain)


def _get_sulfinyl_prefix(
    mol,
    sulfoxide_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfinyl_prefix.

    Phase 160.1 CONTEXT: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfinyl_prefix(mol, sulfoxide_atoms, principal_chain)


def _get_sulfonyl_prefix(
    mol,
    sulfone_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfonyl_prefix.

    Phase 160.1 CONTEXT: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfonyl_prefix(mol, sulfone_atoms, principal_chain)


def _get_sulfanyl_prefix(
    mol,
    thioether_atoms: tuple,
    principal_chain: List[int]
) -> Optional[str]:
    """Thin re-export — see assembly/substituent_prefix_forms.get_sulfanyl_prefix.

    Phase 160.1 CONTEXT: chemistry rule lifted to
    assembly/substituent_prefix_forms.py. This shim preserves the
    polyfunctional.py public API for existing callers.
    """
    return _ASSEMBLY_get_sulfanyl_prefix(mol, thioether_atoms, principal_chain)


def _count_fragment_atoms(
    mol,
    start_atom: int,
    exclude: Set[int],
    carbons_only: bool = False
) -> int:
    """
    Count atoms in a fragment via BFS.

    Args:
        mol: RDKit Mol object
        start_atom: Starting atom index
        exclude: Atoms to not cross (boundary)
        carbons_only: If True, only count carbon atoms

    Returns:
        Number of atoms (or carbons if carbons_only=True)
    """
    from collections import deque

    visited = set()
    queue = deque([start_atom])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if carbons_only:
            if atom.GetSymbol() == 'C':
                count += 1
        else:
            count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def format_fg_prefix(prefix_form: str, locants: List[int], count: int) -> str:
    """
    Format functional group prefix with locants and multiplier.

    Per IUPAC P-16.3.3, compound substituent prefixes (e.g., methylsulfinyl,
    methylsulfonyl) are enclosed in parentheses when used with locants.
    Simple prefixes (hydroxy, oxo, amino) are not parenthesized.

    Args:
        prefix_form: Base prefix name (e.g., "hydroxy", "oxo", "methoxy")
        locants: List of locant positions
        count: Number of instances

    Returns:
        Formatted prefix string (e.g., "2-hydroxy", "3-oxo", "2-(methylsulfinyl)")
    """
    from ..assembly.naming_utils import (
        COMPLEX_MULTIPLIERS,
        _is_fully_enclosed,
        apply_enclosing_marks,
        is_substituted_substituent,
        needs_brackets,
    )

    def _enclose(_p: str) -> str:
        """Give ``_p`` its outermost enclosing marks — and only if it lacks them.

        . Callers legitimately hand over a PRE-ENCLOSED prefix
        (`:1693` passes ``f"({_oxa}imino)"`` verbatim), and such a prefix already
        satisfies `**P-16.5.1.1**` (BlueBookV2.md:7232), verbatim: "*Parentheses
        are used around compound (see P-29.1.2) and complex (see P-29.1.3)
        prefixes; after the multiplicative prefixes 'bis', 'tris', etc.*"  The next
        level out is taken only when one is required — `### **P-16.5.4** Multiple
        types of enclosing marks` (:7444), verbatim at :7446: "*When multiple types
        of enclosing marks are required, the nesting order is as follows:
        {[({[( )]})]}, etc.*"  Here no second type is required, so none is taken.

        Idempotence lives in this ONE place because THREE independent branches
        below reach the enclosure step — `compound`, `_compound_nl`, and
        `_is_derived(multiplier)` — and guarding them one at a time is what let the
        `count > 1` path keep double-enclosing after the first two were fixed.

        `_is_fully_enclosed` is stricter than the `startswith`/`endswith` pair used
        for `_partially_enclosed` below: it matches the OPENING mark to its own
        close, so `(a)-(b)` is correctly NOT fully enclosed and still escalates.
        """
        if _is_fully_enclosed(_p):
            return _p
        # Auto-detect nesting depth: no inner marks -> '(p)'; already contains
        # '()' -> '[p]'.  Byte-identical to the previous hard-coded '(p)' for
        # prefixes without inner enclosing marks.
        return apply_enclosing_marks(_p, -1)

    def _is_derived(multiplier: str) -> bool:
        # P-16.5.1.10: a term modified by a DERIVED multiplier (bis/tris/
        # tetrakis/...kis) is parenthesized, even for a simple base name
        # (bis(sulfanyl), P-35.1).
        return bool(multiplier) and (
            multiplier in COMPLEX_MULTIPLIERS.values()
            or multiplier.endswith("kis"))

    if not locants:
        # No locants (mononuclear parent, locant '1' omitted per P-14.3.4.2) —
        # a COMPOUND prefix must STILL be enclosed per P-16.5.1.1, else the name
        # is ambiguous ('methoxy(methylsulfanyl)methane', NOT the run-together
        # 'methoxymethylsulfanylmethane'). Mirror the locant path's compound
        # detection; simple prefixes (methoxy/methyl/chloro) stay bare -> the
        # single-simple-prefix case is byte-identical.
        _partially_enclosed = (
            (')' in prefix_form or ']' in prefix_form)
            and not (prefix_form.startswith(('(', '['))
                     and prefix_form.endswith((')', ']')))
        )
        # I13: `is_substituted_substituent` is consulted TOO, so the
        # enclosing marks are no longer held by the multiplier WORD.  Before this,
        # `needs_brackets('phosphonooxy')` was False and the marks survived only
        # because the multiplier happened to be derived; the moment the multiplier
        # became basic, `1,3-bis(phosphonooxy)propan-2-ol` collapsed to
        # `1,3-diphosphonooxypropan-2-ol`, and `format_fg_prefix('methoxymethyl',
        # [1,3], 2)` gave `1,3-dimethoxymethyl` -- a string that reads as
        # -CH(OMe)2, a DIFFERENT GROUP.  BB :36333 carries the marks even at count
        # 1: `(phosphonooxy)acetic acid (PIN)`.  Enclosure (P-16.3.4 / P-16.5.1.1)
        # and multiplier choice (P-16.3.2/P-16.3.5) are different questions and
        # must not be coupled in EITHER direction.
        #
        # ⚠: this disjunct asks the MULTIPLIER question, and
        # `is_substituted_substituent` deliberately looks THROUGH enclosing marks
        # ("A fully enclosed token is a single component: decide on its interior"),
        # so `is_substituted_substituent('(dimethylamino)')` is True.  It may
        # therefore say "needs marks" about a prefix that already HAS them; the
        # idempotence that makes that harmless is in `_enclose`.
        _compound_nl = (needs_brackets(prefix_form) or _partially_enclosed
                        or is_substituted_substituent(prefix_form))
        if count > 1:
            multiplier = get_multiplier_prefix(count, prefix_form)
            # P-16.5.1.10: a DERIVED multiplier (bis/tris/...kis) forces enclosure
            # even on a simple base name -> 'bis(sulfanyl)', not 'bissulfanyl'.
            #
            # Item 2: this was the FOURTH branch reaching the
            # enclosure step and the only one still hand-rolling it
            # (`f"{multiplier}({prefix_form})"` behind a
            # `not prefix_form.startswith(("(", "["))` guard). Two defects came
            # with that: the guard tuple omitted `{`, so a brace-enclosed prefix
            # got a REDUNDANT extra level (`bis({[(...)]methyl})` where the locant
            # path gives `bis{[(...)]methyl}`), and the parenthesis type was
            # hard-coded instead of following P-16.5.4. Routed through the shared
            # `_enclose` — which is idempotent, so the guard is no longer needed at
            # all — this branch now agrees with the locant path by construction.
            #
            # It is DEAD today (instrumented: 0 executions over all 1934 gold rows
            # and 300 `pubchem_2000` rows; every producer feeding `prefix_form`
            # calls `apply_enclosing_marks(..., 0)`, which yields `(` or an
            # escalated `[`, never `{`). Kept rather than deleted because the rule
            # it encodes is real and the no-locant path could reach it the moment a
            # mononuclear parent takes a derived multiplier; a dead branch that
            # bypasses the shared sink is the trap, not the branch itself.
            if _is_derived(multiplier):
                return f"{multiplier}{_enclose(prefix_form)}"
            if _compound_nl:
                return f"{multiplier}{_enclose(prefix_form)}"
            return f"{multiplier}{prefix_form}"
        if _compound_nl:
            return _enclose(prefix_form)
        return prefix_form

    # Format locants
    locant_str = ",".join(str(loc) for loc in sorted(locants))

    # Check if prefix is a compound substituent needing parentheses (P-16.5.1.1).
    # A prefix that already carries inner enclosing marks but is not itself
    # fully wrapped (e.g. '(methylcarbamoyl)amino', the N-substituted urea
    # prefix, W2F-P6) is compound and needs an ESCALATED outer enclosure
    # ('[(methylcarbamoyl)amino]') per the P-16.3.3 nesting order (( ) -> [ ]).
    _partially_enclosed = (
        (')' in prefix_form or ']' in prefix_form)
        and not (prefix_form.startswith(('(', '['))
                 and prefix_form.endswith((')', ']')))
    )
    # I13 (see the no-locant branch above for the full reasoning):
    # ask the substituted predicate as well, so the marks are not held by the
    # multiplier word.
    #
    # four gold targets broke here (gate 1641 -> 1637) because
    # this verdict was then used to ADD a level to a prefix that already carried
    # one — `methyl 4-(dimethylamino)-4-(ethylimino)butanoate` shipped as
    # `methyl 4-[(dimethylamino)]4-[(ethylimino)]butanoate`, OPSIN-clean in BOTH
    # forms, so only the exact-match PIN gate could see it.  The verdict is
    # unchanged; `_enclose` is now idempotent, which is where the fix belongs
    # because `_is_derived(multiplier)` below reaches the enclosure step without
    # consulting `compound` at all.
    compound = (needs_brackets(prefix_form) or _partially_enclosed
                or is_substituted_substituent(prefix_form))

    # Get multiplier if multiple instances
    if count > 1:
        multiplier = get_multiplier_prefix(count, prefix_form)
        if compound or _is_derived(multiplier):
            return f"{locant_str}-{multiplier}{_enclose(prefix_form)}"
        return f"{locant_str}-{multiplier}{prefix_form}"

    if compound:
        return f"{locant_str}-{_enclose(prefix_form)}"
    return f"{locant_str}-{prefix_form}"


def _name_amidine_chain_side(
    mol, n_idx: int, amidine_c: int, base: str
) -> Optional[str]:
    """Amino-/imino-side prefix for a chain-terminal amidine (AM-4, P-66.4.1.3.2).

    ``base`` is 'amino' (single-bonded N) or 'imino' (double-bonded =N). Returns
    the BARE prefix — 'amino'/'imino' when the N is unsubstituted, else
    '{alkyl}{base}' ('dimethylamino', 'ethylimino') — or None (fail closed) when
    any N-substituent branch is un-nameable.

    ⚠ THE OUTER ENCLOSURE IS NOT THIS FUNCTION'S JOB. All
    five call sites pass the result straight to `format_fg_prefix`, which owns the
    outer marks and picks their TYPE from the P-16.5.4 nesting order. This
    function used to return a pre-enclosed `'({alkyl}{base})'`, and because it
    hard-coded parentheses it produced `'((4-chlorophenyl)methylamino)'` — a `(`
    directly inside a `(` — for any N-substituent that already carried marks of
    its own. `### **P"16.5.4** Multiple!types!of!enclosing!marks`
    (BlueBookV2.md:7444), verbatim at :7446: "*When multiple types of enclosing
    marks are required, the nesting order is as follows: {[({[( )]})]}*". The
    idempotent `_enclose` then had no way to tell that malformed pre-enclosure
    from a legitimate one and shipped it verbatim.

    The INNER level is this function's job, and it was missing entirely: a
    COMPOUND N-substituent (P-16.5.1.1 / P-29.1.2) needs its own marks before
    `{base}` is glued on, or the boundary is lost. Blue Book, on point —
    `4-{[(4-chlorophenyl)methylidene]amino}aniline (PIN)` (:26529) under
    `## **P-62.3.1** Substitutive names for imines` (:26508): same aryl, same
    N-attached prefix, spelled brace / bracket / paren from the outside in.

    Losing that boundary was not merely cosmetic. For
    `COC(=O)CCC(=NCC)N(Cc1ccc(Cl)cc1)Cc1ccc(F)cc1` the run-together prefix parsed
    as a DIFFERENT molecule (a diarylmethyl); SELF-01 caught it and the compound
    abstained to `unknown organic compound`, so the malformation was costing
    coverage as well as conformance.
    """
    from collections import Counter as _C

    from ..assembly.composer import _name_r_group

    _names = []
    for _nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        _ni = _nb.GetIdx()
        if _ni == amidine_c or _nb.GetAtomicNum() <= 1:
            continue
        _rn = _name_r_group(mol, _ni, exclude_atoms={n_idx})
        if not _rn:
            return None
        _names.append(_rn)
    if not _names:
        return base
    from ..assembly.naming_utils import apply_enclosing_marks, needs_brackets

    _counts = _C(_names)
    _parts = []
    for _nm in sorted(_counts):
        _c = _counts[_nm]
        # P-16.5.1.1: a COMPOUND N-substituent takes its own enclosing marks,
        # at the next level out from any it already carries (`(4-chlorophenyl)methyl`
        # -> `[(4-chlorophenyl)methyl]`, auto-detected per P-16.5.4). A SIMPLE one
        # (methyl, ethyl, benzyl, phenyl) stays bare, so `dimethylamino` and
        # `ethylimino` are byte-identical to before.
        _enc = apply_enclosing_marks(_nm, -1) if needs_brackets(_nm) else _nm
        _parts.append(
            _enc if _c == 1 else f"{get_multiplier_prefix(_c, _enc)}{_enc}"
        )
    # BARE: `format_fg_prefix` owns the outer marks and their type. See the
    # docstring — returning them from here is what produced `(` inside `(`.
    return f"{''.join(_parts)}{base}"


# WS-A task 9 / P-66.6.1.2: FGs whose PREFIX_FORMS string ('oxo',
# 'hydroxy') decorates a SKELETAL atom — valid only when the FG center
# carbon is part of the parent ring. Carbon-including prefixes
# (carboxylic_acid->carboxy, nitrile->cyano, amide->carbamoyl) are correct
# exocyclic by design and MUST NOT be listed here.
_SKELETON_DECORATION_FGS = {
    'aldehyde',
    'ketone', 'thioketone', 'selenoketone', 'telluroketone',
    'alcohol', 'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol',
}


def _exocyclic_component(mol, start: int, ring_set: set) -> set:
    """Connected exocyclic component containing ``start`` (BFS that never
    enters the parent ring)."""
    visited = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nbr.GetIdx()
            if ni not in visited and ni not in ring_set:
                visited.add(ni)
                queue.append(ni)
    return visited


def _name_exocyclic_decoration(
    mol, center: int, ring_set: set, atom_to_locant: dict, fg_name: str
):
    """Name an exocyclic decoration-FG branch with its carbon-including
    substituent prefix (P-66.6.1.2; Phase-172/MBA-02 table: formyl /
    n-oxoalkyl / hydroxyalkyl).

    Returns (formatted_prefix_or_None, component_atoms). A None prefix means
    the branch could not be faithfully named — the caller must leave its
    atoms to the universal pipeline, never emit a skeleton-decoration
    prefix for it.
    """
    comp = _exocyclic_component(mol, center, ring_set)
    ring_attach_loc = None
    for a in comp:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ring_set and ni in atom_to_locant:
                ring_attach_loc = atom_to_locant[ni]
                break
        if ring_attach_loc is not None:
            break
    if ring_attach_loc is None:
        return None, comp
    # Guard: the branch must be exactly chain-C + oxygen(s) — anything else
    # (rings, N, S decorations) is beyond the table's vocabulary.
    for a in comp:
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym not in ('C', 'O', 'H'):
            return None, comp
        if mol.GetAtomWithIdx(a).IsInRing():
            return None, comp
    # W2F-P3 (P-59.2.1.8, fail-closed): the 'n-hydroxyalkyl' / 'n-oxoalkyl' /
    # 'formyl' table rows each express EXACTLY ONE oxygen functional group over an
    # otherwise-plain carbon chain. A branch bearing MORE than one O-FG (e.g. a
    # hydroxymethyl branch AND a ketone) would be named from the single FG's view
    # ('6-hydroxyhexyl'), silently DROPPING the others — structure loss, and it
    # collides with the universal-pipeline substituent name for the same atoms.
    # Decline so the caller defers the whole branch to the universal pipeline
    # (Step 6), which names the branched multi-FG substituent faithfully.
    if sum(1 for a in comp if mol.GetAtomWithIdx(a).GetSymbol() == 'O') != 1:
        return None, comp
    carbon_count = sum(
        1 for a in comp if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
    )
    if fg_name == 'aldehyde':
        table_key = 'aldehyde'
    elif fg_name.endswith('alcohol'):
        table_key = 'alcohol'
    else:
        return None, comp  # ketone family: no faithful table row yet
    from .benzene import _name_functionalized_chain_substituent
    sub_name = _name_functionalized_chain_substituent(carbon_count, table_key)
    if not sub_name:
        return None, comp
    return format_fg_prefix(sub_name, [ring_attach_loc], 1), comp


def _find_fg_center_atom(mol, match: tuple, fg_name: str) -> Optional[int]:
    """
    Find the functional group CENTER atom in a SMARTS match.

    Different FGs have different definitions of "center":
    - Alcohols/thiols: The carbon attached to -OH/-SH (typically index 1 in SMARTS)
    - Ketones: The carbonyl carbon (typically index 1 in SMARTS)
    - Aldehydes: The aldehyde carbon (typically index 0 in SMARTS)

    For most FGs, the center is the FIRST carbon in the match that is
    directly bonded to a heteroatom that is ALSO in the match.

    Args:
        mol: RDKit Mol object
        match: Tuple of atom indices from SMARTS match
        fg_name: Name of the functional group

    Returns:
        Atom index of the FG center, or None if not found
    """
    match_set = set(match)

    # For each carbon in match, check if it's bonded to a heteroatom in match
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue

        # Check if this carbon is bonded to a heteroatom that's also in the match
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in match_set and neighbor.GetSymbol() not in ('C', 'H'):
                # This carbon is directly bonded to a heteroatom in the FG match
                return atom_idx

    # Fallback: return first carbon in match
    for atom_idx in match:
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == 'C':
            return atom_idx

    return None


def get_non_principal_fg_locants(
    mol,
    fg_atoms: List[tuple],
    principal_chain: List[int],
    atom_to_locant: Dict[int, int],
    fg_name: str = ""
) -> List[int]:
    """
    Get locants for non-principal functional groups.

    For each FG match, find the functional group CENTER atom (the carbon
    bearing the heteroatom) on the principal chain and return its locant.

    Args:
        mol: RDKit Mol object
        fg_atoms: List of atom index tuples for each FG match
        principal_chain: Atom indices of the principal chain
        atom_to_locant: Mapping from atom index to locant
        fg_name: Name of the functional group (for specialized handling)

    Returns:
        Sorted list of locants for the functional group positions
    """
    locants = []

    for match in fg_atoms:
        # Find the FG center atom
        center_idx = _find_fg_center_atom(mol, match, fg_name)

        if center_idx is not None and center_idx in atom_to_locant:
            locants.append(atom_to_locant[center_idx])
        else:
            # Fallback: find any atom in match that's on chain
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    locants.append(atom_to_locant[atom_idx])
                    break
            else:
                # Check if any atom's neighbor is on the chain
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for neighbor in atom.GetNeighbors():
                        nbr_idx = neighbor.GetIdx()
                        if nbr_idx in atom_to_locant:
                            locants.append(atom_to_locant[nbr_idx])
                            break
                    else:
                        continue
                    break

    # Do NOT deduplicate locants: two groups at the same position need
    # repeated locants (e.g., 3,3-diamino). The count and locant list
    # must agree for OPSIN compatibility.
    return sorted(locants)


def get_non_principal_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that are not the principal group.

    Filters out the principal group and unsaturation markers.

    Args:
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of the principal group (or None)

    Returns:
        Dict of non-principal functional group names to their atom indices
    """
    exclude_groups = {"alkene", "alkyne"}  # Handled by unsaturation

    result = {}
    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if fg_name in exclude_groups:
            continue
        if matches:
            result[fg_name] = matches

    return result


def _name_ring_as_parent_polyfunctional(features: Any) -> Optional[str]:
    """Name a polyfunctional compound where the ring is the parent structure.

    Handles the case where ``principal_chain`` is None because the molecule
    uses a ring as its parent hydride. Builds the name from:
    1. Ring parent name (from oriented ring or ring detection)
    2. Principal group suffix with locants on the ring
    3. Non-principal group prefixes with ring locants
    4. Substituent prefixes via the universal pipeline

    IUPAC 2013 P-31 through P-43.

    Args:
        features: MolecularFeatures object (must have is_cyclic=True,
            principal_group set).

    Returns:
        Complete IUPAC name, or None if ring parent cannot be determined.
    """
    mol = features.mol
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})
    # audit fix 2026-06-22: drop same-union-class subtypes already in the
    # multiplied suffix (parity with the chain path; ring polyols too).
    non_principal = _drop_union_class_subtypes(non_principal, principal_group)

    if not principal_group:
        return None

    # --- Step 1: Identify ring parent ---
    ring_info = mol.GetRingInfo()
    all_rings = ring_info.AtomRings()
    if not all_rings:
        return None

    # Use oriented_ring if available, else find the largest ring
    oriented_ring = getattr(features, 'oriented_ring', None)
    if oriented_ring:
        ring_atoms = list(oriented_ring)
    else:
        # Pick the largest ring as parent
        ring_atoms = list(max(all_rings, key=len))

    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # --- Step 2: Build ring parent name ---
    ring_parent_name = _get_ring_parent_name(mol, ring_atoms)
    if not ring_parent_name:
        return None

    # --- Step 3: Build locant mapping for ring atoms ---
    # Use features.atom_to_locant if available, else build from ring order
    atom_to_locant = features.atom_to_locant
    if not atom_to_locant:
        atom_to_locant = {}
        for i, idx in enumerate(ring_atoms):
            atom_to_locant[idx] = i + 1

    # --- Step 3b: Skeletal ring unsaturation (P-31.1.4) ---
    # Render -ene/-yne onto the ring stem using the SAME numbering that drives
    # the suffix and prefixes, so the name is self-consistent (round-trip valid)
    # regardless of whether the numbering is the lowest-locant PIN choice. Only
    # all-carbon monocycles are handled; a heterocyclic (Hantzsch-Widman) or
    # non-consecutive-yne unsaturation fails closed to the deferred handlers.
    _ring_numbering = {a: atom_to_locant[a] for a in ring_set
                       if a in atom_to_locant}
    _ru = render_ring_unsaturation(mol, _ring_numbering)
    if _ru is None:
        return None  # non-consecutively-numbered triple bond (soundness refuse)
    if _ru.double_locants or _ru.triple_locants:
        _all_carbon_ring = all(
            mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in ring_atoms
        )
        if not _all_carbon_ring or not ring_parent_name.endswith('an'):
            return None  # HW/heterocyclic or non-standard stem -> defer
        ring_parent_name = _ring_stem_with_unsaturation(
            ring_parent_name[:-2],  # 'cyclohexan' -> 'cyclohex'
            list(_ru.double_locants), list(_ru.triple_locants),
        )

    # --- Step 4: Generate principal group suffix with locants ---
    from .seniority import get_suffix
    suffix = get_suffix(principal_group, is_ring=True)
    if not suffix:
        # Try chain suffix as fallback
        suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    suffix_locants = []
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            # Wave2 T2c (P-64.7.1): the suffix locant must come from the FG
            # CENTER atom, not the first match atom that happens to be in the
            # map. The ketone SMARTS match is (C_neighbor, C=O, O, C_neighbor)
            # — taking match[0] read the NEIGHBOR's locant, producing the
            # impossible '2-aminocyclohexan-2-one' for NC1CCCCC1=O (amino and
            # oxo cannot share one carbon; PIN is 2-aminocyclohexan-1-one).
            _center = _find_fg_center_atom(mol, match, principal_group)
            if _center is not None and _center in atom_to_locant:
                suffix_locants.append(atom_to_locant[_center])
                continue
            for atom_idx in match:
                if atom_idx in atom_to_locant:
                    suffix_locants.append(atom_to_locant[atom_idx])
                    break
                # For FGs attached TO the ring (e.g., -COOH on cyclohexane),
                # the center atom may not be in the ring; check neighbors
                atom = mol.GetAtomWithIdx(atom_idx)
                for nbr in atom.GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in atom_to_locant and nidx in ring_set:
                        suffix_locants.append(atom_to_locant[nidx])
                        break
    suffix_locants = sorted(set(suffix_locants))

    # Determine multiplier for multiple principal groups
    count = len(suffix_locants) if suffix_locants else 1
    multiplier = get_suffix_multiplier_prefix(count, suffix) if count > 1 else ""

    # --- Step 4b: Ester functional-class O-alkyl word (P-65.6.3.2) ---
    # An ester principal group on a RING parent is named
    # '<alkyl> <ring>-carboxylate': get_suffix('ester') yields only the
    # acid-side '-carboxylate', and the ring path previously DROPPED the O-alkyl
    # word, emitting the bare carboxylate ANION — a DIFFERENT molecule that
    # SELF-01 correctly vetoed, so the whole polyfunctional ring ester abstained
    # ('methyl 4-oxocyclohexane-1-carboxylate' -> '4-oxocyclohexane-1-carboxylate').
    # Extract the O-alkyl fragment, name it, consume its atoms, and prepend the
    # word at assembly. Fail closed when the alkyl cannot be named faithfully.
    _ester_alkyl_word = None
    _ester_consumed: set = set()
    if principal_group == 'ester':
        _em = getattr(features, 'ester_match', None)
        if not _em:
            return None
        from .esters import get_alkyl_fragment_name, parse_ester_fragments
        _acid_atoms, _alkyl_atoms = parse_ester_fragments(mol, _em)
        if not _alkyl_atoms:
            return None
        _ester_alkyl_word = get_alkyl_fragment_name(mol, _alkyl_atoms)
        if not _ester_alkyl_word:
            return None
        # Consume the whole O-alkyl fragment plus both ester oxygens so the
        # universal pipeline does not re-name them (the SMARTS match only
        # carries the first alkyl carbon; a larger alkyl would otherwise leak).
        _ester_consumed = set(_alkyl_atoms)
        for _i in _em:
            if _i not in ring_set:
                _ester_consumed.add(_i)

    # --- Step 5: Generate non-principal FG prefixes with ring locants ---
    all_prefixes = []
    # WS-A task 9 consumption plumbing: atoms named by an exocyclic
    # carbon-including prefix (formyl/n-oxoalkyl/...) must be consumed even
    # when outside the SMARTS match (the CH2 of -CH2CHO); atoms of a branch
    # we could NOT faithfully name must NOT be consumed (the universal
    # pipeline names them instead of them silently vanishing).
    _extra_consumed: set = set()
    _do_not_consume: set = set()
    for fg_name, matches in non_principal.items():
        if not matches:
            continue
        if fg_name in NO_SENIORITY_GROUPS and fg_name not in ('alkene', 'alkyne'):
            # Named as prefix if it has a prefix form
            prefix_form = get_fg_prefix_form(fg_name, mol, matches[0], None)
            if prefix_form:
                fg_locants = []
                for match in matches:
                    for atom_idx in match:
                        if atom_idx in atom_to_locant:
                            fg_locants.append(atom_to_locant[atom_idx])
                            break
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for nbr in atom.GetNeighbors():
                            nidx = nbr.GetIdx()
                            if nidx in atom_to_locant and nidx in ring_set:
                                fg_locants.append(atom_to_locant[nidx])
                                break
                fg_locants = sorted(set(fg_locants))
                fg_count = len(fg_locants) if fg_locants else len(matches)
                formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
                all_prefixes.append(formatted)
            continue

        # Seniority-bearing non-principal groups
        prefix_form = get_prefix(fg_name)
        if prefix_form:
            fg_locants = []
            _diverted = 0
            for match in matches:
                # WS-A task 9 / P-66.6.1.2: skeleton-decoration prefixes
                # ('oxo', 'hydroxy') are valid ONLY when the FG center carbon
                # is IN the ring skeleton. The old context-blind path rewrote
                # an exocyclic -CHO as a ring ketone '4-oxo...' — a
                # structurally DIFFERENT molecule — by re-anchoring the
                # locant on the ring neighbor. An exocyclic center must emit
                # the carbon-including substituent prefix instead (formyl /
                # n-oxoalkyl / hydroxymethyl, the Phase-172/MBA-02 table).
                if fg_name in _SKELETON_DECORATION_FGS:
                    center = _find_fg_center_atom(mol, match, fg_name)
                    if center is not None:
                        if center in atom_to_locant:
                            # Locant from the CENTER atom (also fixes ring
                            # ketones citing the flanking match atom).
                            fg_locants.append(atom_to_locant[center])
                            continue
                        if center not in ring_set:
                            _diverted += 1
                            exo_prefix, exo_atoms = _name_exocyclic_decoration(
                                mol, center, ring_set, atom_to_locant, fg_name
                            )
                            if exo_prefix is not None:
                                all_prefixes.append(exo_prefix)
                                _extra_consumed.update(exo_atoms)
                            else:
                                # Cannot produce a faithful name — leave the
                                # branch to the universal pipeline rather
                                # than emit a structurally wrong prefix.
                                _do_not_consume.update(exo_atoms)
                            continue
                for atom_idx in match:
                    if atom_idx in atom_to_locant:
                        fg_locants.append(atom_to_locant[atom_idx])
                        break
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nbr in atom.GetNeighbors():
                        nidx = nbr.GetIdx()
                        if nidx in atom_to_locant and nidx in ring_set:
                            fg_locants.append(atom_to_locant[nidx])
                            break
            fg_locants = sorted(set(fg_locants))
            _remaining = len(matches) - _diverted
            if fg_locants or _remaining > 0:
                fg_count = len(fg_locants) if fg_locants else _remaining
                formatted = format_fg_prefix(prefix_form, fg_locants, fg_count)
                all_prefixes.append(formatted)

    # --- Step 6: Discover substituents on the ring via universal pipeline ---
    # Collect atoms consumed by the principal group and non-principal FGs
    consumed_atoms = set()
    if features.principal_group_atoms:
        for match in features.principal_group_atoms:
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)
    # residual fix: secondary_amide/tertiary_amide have NO prefix
    # form (get_prefix() is None, seniority.py:916-917 -- "Named via acylamino
    # pathway in universal pipeline"), so Step 5 above never emits a prefix for
    # them. The blanket loop below used to mark their off-ring match atoms
    # (the amide N + carbonyl C=O; a 2-atom-short-of-the-whole-branch SMARTS,
    # `functional_groups.py`'s `"[CX3](=O)[NX3H1][#6]"`) as "consumed" anyway,
    # which orphaned the acyl R-group carbon -- its only path back to the ring
    # runs THROUGH the now-"parent" N/C=O -- so the universal substituent
    # walker below could never discover the whole -NH-C(=O)-R branch.
    # `_verify_completeness`'s hard assert on that orphaned atom was then
    # silently swallowed by `_integrate_universal_prefixes`'s bare
    # `except Exception: return ""`, dropping the WHOLE acylamino group from
    # the assembled name (an atom-drop SELF-01 correctly vetoed downstream,
    # so the molecule abstained rather than emitting wrong -- but a table
    # miss must degrade to an uglier name, not to silence). Leaving these two
    # FG names OUT of consumed_atoms lets the universal pipeline discover and
    # name the branch itself (`_name_amino_branch` -> `linear_acyl_amido_prefix`
    # -> 'acetamido' / 'N-methylacetamido'), mirroring the CHAIN-parent path's
    # existing `_POLY_GUARD_FG_TYPES` allowlist (only consume atoms of an FG
    # actually spelled as a whole-branch prefix).
    _ACYLAMINO_PATHWAY_FGS = ('secondary_amide', 'tertiary_amide')
    for fg_name, matches in non_principal.items():
        if fg_name in _ACYLAMINO_PATHWAY_FGS:
            continue
        for match in matches:
            # lead e: a non-principal 'ester' expressed as a
            # whole-branch acyloxy arm (-CH2-O-C(=O)R attached to the ring
            # through a carbon linker) has the SAME shape of bug as the
            # acylamino case above: get_prefix('ester') is None (no Step-5
            # prefix), yet the blanket loop below used to still mark the
            # match's off-ring atoms (acyl C, both O's, and the FIRST alkyl
            # carbon captured by the ester SMARTS) as "consumed". When that
            # first alkyl carbon IS the branch's only anchor back to the ring
            # (e.g. the CH2 of -CH2-O-C(=O)CH3), consuming it orphans the
            # whole arm from `_integrate_universal_prefixes`'s walk, which
            # then returns '' and the arm silently vanishes -> atom-incomplete
            # -> abstain (the composer already names it correctly via
            # `name_substituent_fragment` -> '(acetyloxy)methyl' when reached
            # directly). Defer ONLY when: (a) the match's alkyl-side anchor
            # (the ester SMARTS's last atom, always present and always a
            # carbon -- `[CX3](=O)[OX2][#6]` yields exactly 4 atoms) sits
            # OUTSIDE the ring (i.e. it is a true off-ring linker, not the
            # ring atom itself), and (b) it is not the ring's own principal
            # ester (guarded by `principal_group == 'ester'` above, a
            # disjoint code path) -- so deferring here can only ever apply to
            # a genuine non-principal branch. This is direction-agnostic: it
            # also defers the mirror shape Ring-C(=O)-O-R (a mono-alkyl ester
            # of a ring diacid, e.g. 'methyl ... carboxylic acid' ->
            # '2-(methoxycarbonyl)cyclohexane-1-carboxylic acid'), which is
            # correct extra breadth, not a leak -- see
            # test_mirror_shape_methoxycarbonyl_ring_diacid_monoester.
            if fg_name == 'ester' and match[-1] not in ring_set:
                continue
            for idx in match:
                if idx not in ring_set:
                    consumed_atoms.add(idx)
    # WS-A task 9: see Step-5 plumbing comment.
    consumed_atoms |= _extra_consumed
    consumed_atoms |= _ester_consumed
    consumed_atoms -= _do_not_consume

    try:
        from ..assembly.composer import _integrate_universal_prefixes
        sub_prefix_str = _integrate_universal_prefixes(
            mol, ring_set,
            parent_type="ring",
            oriented_ring=ring_atoms,
            atom_to_locant=atom_to_locant,
            exclude_atoms=consumed_atoms,
        )
        if sub_prefix_str:
            all_prefixes.append(sub_prefix_str)
    except Exception as exc:
        logger.debug("Ring-as-parent universal prefix failed: %s", exc)

    # Sort prefixes alphabetically (P-14.5.2), with P-14.5.4 lowest-locant
    # tie-break for identical-letter prefixes.
    # RENDERED prefix strings (locants already attached) -> parent_locants=True.
    all_prefixes.sort(key=lambda p: prefix_citation_sort_key(
        p, parent_locants=True))

    # --- Step 7: Assemble the complete name ---
    # Format: [prefixes]-[ring_parent]-[suffix_locants]-[multiplier][suffix]
    # e.g., "4-hydroxy-cyclohexan-1-one"
    formatted_suffix = format_suffix_with_locants(
        ring_parent_name, "", suffix, suffix_locants, multiplier
    )

    if all_prefixes:
        prefix_str = _join_prefixes(all_prefixes)
        if prefix_str and formatted_suffix and prefix_str[-1].isalpha() and formatted_suffix[0].isdigit():
            name = f"{prefix_str}-{formatted_suffix}"
        else:
            name = f"{prefix_str}{formatted_suffix}"
    else:
        name = formatted_suffix

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    # Prepend the ester O-alkyl functional-class word (Step 4b): the leading
    # 'methyl ' / 'ethyl ' / 'propan-2-yl ' token of '<alkyl> <ring>-carboxylate'.
    if _ester_alkyl_word:
        name = f"{_ester_alkyl_word} {name}"

    return name


def _get_ring_parent_name(mol, ring_atoms: list) -> Optional[str]:
    """Get the parent hydride name for a ring.

    Checks retained names first (benzene, pyridine, etc.), then falls
    back to systematic cyclo- naming.

    Args:
        mol: RDKit Mol object.
        ring_atoms: List of atom indices forming the ring.

    Returns:
        Ring parent stem (e.g., ``"cyclohexan"``, ``"benzene"``),
        or None if not determinable.
    """
    ring_set = set(ring_atoms)
    ring_size = len(ring_atoms)

    # Check if ring is all-carbon aromatic (benzene)
    all_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
    all_carbon = all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring_atoms)

    if ring_size == 6 and all_aromatic and all_carbon:
        return "benzene"

    # Check for heterocyclic retained names
    has_heteroatom = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in ring_atoms
    )
    if has_heteroatom:
        try:
            from ..rules.heterocycles import classify_heterocycle
            het_info = classify_heterocycle(mol, tuple(ring_atoms))
            if het_info and het_info.get('name'):
                return het_info['name']
        except Exception:
            pass

    # Systematic cyclo- naming for carbocyclic rings
    if all_carbon:
        from ..data.chain_names import get_chain_prefix
        stem = get_chain_prefix(ring_size)
        if stem:
            # Return stem without "ane" suffix -- format_suffix_with_locants
            # will add the appropriate suffix. For cycloalkanes with
            # functional groups, we need "cyclohexan" not "cyclohexane".
            return f"cyclo{stem}an"

    return None


def _acyl_is_simple_saturated_chain(mol, acyl_c: int, n_atom: Optional[int]) -> bool:
    """True iff the acyl carbon skeleton of an amide is an UNBRANCHED, fully
    SATURATED, ACYCLIC carbon chain — the only shape ``rules.amides.name_amide``
    names correctly.

    ``name_amide``'s parent name comes from ``get_amide_chain_length`` (the
    LONGEST carbon chain) mapped by count in ``get_amide_parent_name``; branches
    and C=C/C#C bonds on the acyl side are structurally invisible to it, so a
    branched acyl (isobutyryl -> 'propanamide') or an unsaturated acyl (acryloyl
    -> 'propanamide') would be named as a straight-chain saturated FALSE FRIEND
    (a different constitution). The polyfunctional amide-delegation must therefore
    only hand such an amide to ``name_amide`` when the acyl is a simple chain;
    otherwise it falls through to the chain machinery (on-chain) or fails closed
    (off-chain). This is a producer-honesty guard that holds gate-OFF, per the
    ``8afa533c`` lesson (honesty must not depend on the SELF-01 gate it backstops).
    """
    from collections import deque

    from rdkit import Chem  # module has no top-level Chem import (only a local _Chem)
    seen = {acyl_c}
    q = deque([acyl_c])
    while q:
        cur = q.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j == n_atom or j in seen or nb.GetSymbol() != 'C':
                continue
            seen.add(j)
            q.append(j)
    # acyclic
    if any(mol.GetAtomWithIdx(c).IsInRing() for c in seen):
        return False
    # saturated: every C-C bond within the acyl skeleton is single
    for c in seen:
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            if nb.GetIdx() in seen and mol.GetBondBetweenAtoms(
                    c, nb.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                return False
    # unbranched: the acyl carbons form a simple path (the carbonyl carbon has
    # exactly one acyl-chain neighbour; every other carbon has at most two)
    for c in seen:
        deg = sum(1 for nb in mol.GetAtomWithIdx(c).GetNeighbors()
                  if nb.GetIdx() in seen)
        if c == acyl_c and deg > 1:
            return False
        if c != acyl_c and deg > 2:
            return False
    return True


def _acyl_ring_is_simple_unsub_carbocycle(
        mol, acyl_c: int, n_atom: Optional[int]) -> bool:
    """True iff the amide's acyl carbon is attached to a SINGLE, unsubstituted,
    all-carbon MONOCYCLIC ring (benzene or a fully-saturated cycloalkane) — the
    only ring-attached shapes ``rules.amides.name_amide`` names correctly
    (-> ``benzamide`` / ``cyclohexanecarboxamide``).

    ``name_amide``'s ring-attached path keys only on the ipso ring's size +
    benzene-aromaticity, so it (a) DROPS any substituent on the acyl ring
    (3-chlorobenzoyl -> bare ``benzamide``), (b) gets the wrong stem for a FUSED
    acyl ring, and (c) MISNAMES a heteroaromatic acyl ring as a carbocycle
    (isonicotinoyl -> ``cyclohexanecarboxamide``, dropping the ring N). Those
    must NOT be delegated — a producer-honesty guard that holds gate-OFF (the
    ``8afa533c`` standard), exactly as ``_acyl_is_simple_saturated_chain`` guards
    the chain path.
    """
    from rdkit import Chem  # module has no top-level Chem import (only a local _Chem)
    ipso = None
    for nb in mol.GetAtomWithIdx(acyl_c).GetNeighbors():
        j = nb.GetIdx()
        if j == n_atom:
            continue
        if (nb.GetSymbol() in ('O', 'S', 'Se', 'Te')
                and mol.GetBondBetweenAtoms(acyl_c, j)
                       .GetBondTypeAsDouble() == 2.0):
            continue
        if nb.IsInRing():
            if ipso is not None:
                return False
            ipso = j
        else:
            # acyl carbon also bears an acyclic heavy neighbour -> not a plain
            # ring-attached acyl (name_amide's ring branch would not apply).
            if nb.GetAtomicNum() > 1:
                return False
    if ipso is None:
        return False
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings() if ipso in r]
    if len(rings) != 1:
        return False  # fused / spiro ipso
    ring = rings[0]
    if any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in ring):
        return False  # heteroaromatic / heterocyclic acyl ring
    for other in ri.AtomRings():
        if set(other) != ring and set(other) & ring:
            return False  # ring is fused to another
    aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    if not aromatic:
        for a in ring:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() in ring and mol.GetBondBetweenAtoms(
                        a, nb.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
                    return False  # unsaturated cycloalkene -> not '-ane'carboxamide
    # unsubstituted: no ring atom bears a heavy non-ring neighbour except the
    # ipso->acyl_c bond (a ring substituent would be silently dropped).
    for a in ring:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring or nb.GetAtomicNum() <= 1:
                continue
            if a == ipso and j == acyl_c:
                continue
            return False
    return True


def _name_ring_attached_anilide(features: Any) -> Optional[str]:
    """Lever C: a ring-attached secondary/tertiary amide
    whose acyl ring is a simple unsubstituted carbocycle (benzoyl -> benzamide,
    cyclohexanecarbonyl -> cyclohexanecarboxamide) and whose only junior FGs live
    inside the N-substituent(s). Such an amide has an EMPTY principal_chain (the
    parent is the ring-attached acyl), so name_polyfunctional's main body returns
    None before the chain amide-delegation runs; without this the benzamide
    candidate double-counts the junior phenol and is suppressed, so the benzene
    handler wins the P-41-violating phenol-as-parent 'N-benzoyl-4-aminophenol'.

    Delegate to rules.amides.name_amide (which names the ring-attached parent
    correctly) under the SAME fail-closed coverage guards as the chain path:
    single amide match, no defined stereo, every N-branch named, every junior FG
    contained in the N-substituent fragments, and the acyl ring a simple
    unsubstituted carbocycle. Returns None (fall through) otherwise.
    """
    mol = features.mol
    pg = features.principal_group
    if pg not in ('secondary_amide', 'tertiary_amide'):
        return None
    matches = (features.functional_groups or {}).get(pg, [])
    if len(matches) != 1:
        return None
    match = list(matches[0])
    from rdkit import Chem
    acyl_c = next(
        (i for i in match
         if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
         and any(nb.GetSymbol() in ('O', 'S', 'Se', 'Te')
                 and mol.GetBondBetweenAtoms(i, nb.GetIdx())
                        .GetBondTypeAsDouble() == 2.0
                 for nb in mol.GetAtomWithIdx(i).GetNeighbors())),
        None)
    if acyl_c is None:
        return None
    n_atom = next(
        (i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
    from .amides import (
        get_n_substituents,
        is_ring_attached_amide,
    )
    from .amides import (
        name_amide as _ramide,
    )
    if not is_ring_attached_amide(mol, tuple(match)):
        return None
    # RISK-4: name_amide names a C-amide (benzamide / carboxamide). An
    # N-O amide (hydroxamic / Weinreb) or N-N amide (hydrazide) has a DIFFERENT
    # suffix and would be MISNAMED. get_n_substituents collects only carbon
    # subs, so these decline today via the branch-count mismatch below — make
    # that intent EXPLICIT so a future get_n_substituents extension to O/N-subs
    # cannot silently arm the intercept on an N-heteroatom amide.
    if n_atom is not None and any(
            nb.GetAtomicNum() > 1 and nb.GetSymbol() != 'C'
            and nb.GetIdx() != acyl_c
            for nb in mol.GetAtomWithIdx(n_atom).GetNeighbors()):
        return None
    if not _acyl_ring_is_simple_unsub_carbocycle(mol, acyl_c, n_atom):
        return None
    # fail closed on any DEFINED stereo (the delegated path emits no descriptors)
    if (Chem.FindMolChiralCenters(mol, includeUnassigned=False)
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE
                   for b in mol.GetBonds())):
        return None
    n_subs = get_n_substituents(mol, tuple(match))
    claimed = set()
    for s in n_subs:
        claimed |= set(s.get('atoms') or [])
    if not claimed:
        return None
    expected = 0
    if n_atom is not None:
        expected = sum(
            1 for nb in mol.GetAtomWithIdx(n_atom).GetNeighbors()
            if nb.GetAtomicNum() > 1 and nb.GetIdx() != acyl_c)
    if len(n_subs) != expected:
        return None
    non_principal = getattr(features, 'non_principal_groups', {}) or {}
    if not all(set(fm).issubset(claimed)
               for fmatches in non_principal.values() for fm in fmatches):
        return None
    return _ramide(mol, tuple(match))


def name_polyfunctional(features: Any) -> Optional[str]:
    """
    Generate IUPAC name for a polyfunctional compound.

    This is the main coordinator function for multi-FG naming.
    If the compound is not polyfunctional, returns None.

    Args:
        features: MolecularFeatures object with extracted features

    Returns:
        Complete IUPAC name string, or None if not polyfunctional
    """
    # Check if this is a polyfunctional compound
    if not getattr(features, 'is_polyfunctional', False):
        return None

    mol = features.mol
    principal_chain = features.principal_chain
    atom_to_locant = features.atom_to_locant
    principal_group = features.principal_group
    non_principal = getattr(features, 'non_principal_groups', {})
    # audit fix 2026-06-22: drop same-union-class subtypes already in the
    # multiplied suffix (prevents the 'n-hydroxy…-1,n-diol' double-listing).
    non_principal = _drop_union_class_subtypes(non_principal, principal_group)

    if not principal_chain or not atom_to_locant:
        # --- Phase 86-02: Ring-as-parent polyfunctional path ---
        # When principal_chain is None but the molecule is cyclic, attempt
        # ring-as-parent naming. Conservative guards prevent intercepting
        # compounds better handled by specialized fallthrough handlers:
        # - Only monocyclic (1 ring) -- fused/polycyclic have specialized handlers
        # - Only fully saturated rings -- unsaturated rings need -ene/-yne handling
        # - No aromatic rings -- benzene/pyridine/etc. have retained name handlers
        if getattr(features, 'is_cyclic', False) and not getattr(features, 'chain_is_parent', False) and principal_group:
            ring_info = mol.GetRingInfo()
            n_rings = ring_info.NumRings()
            if n_rings == 1:
                ring_atoms = list(ring_info.AtomRings()[0])
                ring_set = set(ring_atoms)
                ring_size = len(ring_atoms)
                total_heavy = mol.GetNumHeavyAtoms()

                # Guard: ring must be a significant portion of the molecule.
                # If the ring is < 40% of heavy atoms, it's likely a
                # substituent on a chain parent, not the parent itself.
                # E.g., sphingolipid with cyclohexane ring but 50+ chain atoms.
                # W2F-P3 (P-44.1.1): EXCEPT when the principal characteristic
                # group is attached to the ring — then the ring necessarily
                # holds the PCG and MUST be the parent (the acyclic branch bears
                # only junior groups), regardless of relative size. Without this
                # exception a large FG-bearing acyclic substituent (item 15) drops
                # the ring below the size heuristic and the whole molecule fails
                # closed even though cyclohexane-1-carboxylic acid is the PIN parent.
                _pcg_on_ring = False
                for _match in (getattr(features, 'principal_group_atoms', None) or []):
                    for _a in _match:
                        if _a in ring_set:
                            _pcg_on_ring = True
                            break
                        for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                            if _nb.GetIdx() in ring_set:
                                _pcg_on_ring = True
                                break
                        if _pcg_on_ring:
                            break
                    if _pcg_on_ring:
                        break
                if (total_heavy > 0 and ring_size / total_heavy < 0.35
                        and not _pcg_on_ring):
                    pass  # Fall through to return None
                else:
                    # Check ring is non-aromatic. A NON-aromatic ring double/triple
                    # bond is now allowed through: _name_ring_as_parent_polyfunctional
                    # renders skeletal -ene/-yne (P-31.1.4) for an all-carbon
                    # monocycle and fails closed (returns None) for anything it
                    # cannot render faithfully. Aromatic monocycles still have
                    # dedicated retained-name handlers and are excluded here.
                    is_aromatic = any(
                        mol.GetAtomWithIdx(idx).GetIsAromatic()
                        for idx in ring_atoms
                    )

                    if not is_aromatic:
                        ring_name = _name_ring_as_parent_polyfunctional(
                            features,
                        )
                        if ring_name:
                            return ring_name
        return None

    # --- Wave2 T5b (P-66.1.1.3): OFF-CHAIN acyl secondary/tertiary amide ---
    # When the PCG amide's acyl carbon is NOT on the principal chain (the
    # chain is the N-side), the chain-suffix machinery below is structurally
    # WRONG: it double-expresses the amide (acetamido prefix + a phantom
    # '...anamide' suffix whose carbonyl C isn't in the chain) — production
    # SELF-01 suppressed these to unknown. The PIN keeps the amide as PCG
    # with the ACYL side as parent and the whole N-side as a located
    # N-substituent prefix carrying its own junior FGs:
    # CC(CO)NC(C)=O -> N-(1-hydroxypropan-2-yl)acetamide (amide senior to
    # alcohol, P-41). Delegate to rules.amides.name_amide (whose N-substituent
    # namer now handles simple internal substituents) under fail-closed
    # guards; decline to None (jar-independent) when they don't hold.
    if principal_group in ('secondary_amide', 'tertiary_amide'):
        _amide_matches = features.functional_groups.get(principal_group, [])
        if len(_amide_matches) == 1:
            _match = list(_amide_matches[0])
            _acyl_c = next(
                (i for i in _match
                 if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() == 'O'
                     and mol.GetBondBetweenAtoms(i, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(i).GetNeighbors())),
                None,
            )
            if _acyl_c is not None:
                # Fail closed on any DEFINED stereocentre/stereobond: the
                # delegated path emits no parent-level descriptors (the T3d
                # oxime-decline precedent).
                #
                # (E3 Task 5 follow-on): `Chem.FindMolChiralCenters`
                # has an undocumented DESTRUCTIVE side effect — its internal
                # legacy re-perception pass clears every bond's `_CIPCode`
                # property (the E/Z label `rdCIPLabeler.AssignCIPLabels` /
                # the `centres` bridge already set during perception),
                # while leaving `Bond.GetStereo()` itself untouched. Verified
                # in isolation: a bond reporting `(1,2,HasProp=1,'E')`
                # BEFORE the call reports `(1,2,HasProp=0,None)` straight
                # after, though STEREOE survives. Called on the SHARED
                # `mol` (not a copy), this permanently erases the E/Z
                # descriptors for every LATER consumer in this SAME naming
                # attempt — including this function's own final
                # `collect_stereodescriptors` call far below, which is why a
                # stereo-bearing on-chain amide (e.g. anandamide's
                # 5,8,11-trienoyl homologue) always emitted a name with
                # "0 R/S + N E/Z but name lacks descriptors" once this branch
                # ran, WITH OR WITHOUT the N-substituent fix. Call it on a
                # disposable COPY so the probe cannot clobber the shared
                # mol's already-assigned CIP labels; `_has_stereo` itself is
                # unaffected (chirality centres are copy-invariant).
                from rdkit import Chem as _Chem

                from .amides import (
                    get_n_substituents,
                )
                from .amides import (
                    name_amide as _rules_name_amide,
                )
                _has_stereo = bool(
                    _Chem.FindMolChiralCenters(
                        _Chem.Mol(mol), includeUnassigned=False)
                ) or any(
                    b.GetStereo() != _Chem.BondStereo.STEREONONE
                    for b in mol.GetBonds()
                )
                _n_subs = get_n_substituents(mol, tuple(_match))
                _n_atom = next(
                    (i for i in _match
                     if mol.GetAtomWithIdx(i).GetSymbol() == 'N'), None)
                # Coverage: every heavy N-neighbour besides the acyl C must
                # have produced a named substituent (get_n_substituents
                # silently omits unnameable ones), and every junior FG must
                # live wholly inside those N-substituent fragments.
                _claimed = set()
                for _s in _n_subs:
                    _claimed |= set(_s.get('atoms') or [])
                _expected_n_branches = 0
                if _n_atom is not None:
                    _expected_n_branches = sum(
                        1 for nb in mol.GetAtomWithIdx(_n_atom).GetNeighbors()
                        if nb.GetAtomicNum() > 1 and nb.GetIdx() != _acyl_c
                    )
                _fgs_contained = all(
                    set(_fm).issubset(_claimed)
                    for _fg_matches in (non_principal or {}).values()
                    for _fm in _fg_matches
                )
                _amide_guards_ok = (
                    not _has_stereo and _claimed
                    and len(_n_subs) == _expected_n_branches
                    and _fgs_contained
                )
                if _amide_guards_ok and _acyl_is_simple_saturated_chain(
                        mol, _acyl_c, _n_atom):
                    _nm = _rules_name_amide(mol, tuple(_match))
                    if _nm:
                        return _nm
                # A1: the count-based
                # rules.amides.name_amide above names ONLY a simple saturated
                # acyl. When the acyl is BRANCHED or UNSATURATED but ON the
                # principal chain (paracetamol's isobutyryl / acryloyl cousins:
                # CC(C)C(=O)Nc1ccc(O)cc1, C=CC(=O)Nc1ccc(O)cc1), route to the
                # acyl-AWARE composer namer _assemble_amide_name instead of
                # falling through. It names the acyl chain via the general chain
                # machinery (branch prefixes + prop-2-en/-yn unsaturation come
                # for free, exactly the unified acid/amide suffix-swap the
                # use) and the N-substituent via
                # get_n_substituents, whose fragment already contains the junior
                # ring-FG scoped by disjoint atom ownership (no double-count —
                # verified). SAME coverage guards as the simple path (single
                # amide; no defined stereo; every N-branch named; every junior
                # FG contained in the N-substituent fragments). Previously this
                # fell through to the chain machinery, which mis-rooted the
                # N-aryl and abstained. Prefix ORDER follows the existing uniform
                # amide convention; the P-14.5.2/P-16.3.3 N/C merge-and-order fix
                # is a separate conformance lever tracked by
                # test_amide_merged_prefix_p16_3_3::test_chain_amide_merges_too.
                if _amide_guards_ok and _acyl_c in set(principal_chain):
                    from ..assembly.composer import (
                        _assemble_amide_name as _composer_name_amide,
                    )
                    from ..assembly.composer import (
                        _generate_prefixes as _gen_prefixes_am,
                    )
                    # guard: _assemble_amide_name appends
                    # _generate_prefixes(features) at PARENT level. That set is
                    # honest for a LOCATED acyl-chain substituent (a '2-methyl'
                    # branch, locant 2) but RE-EXPRESSES a junior FG that lives
                    # inside an N-substituent yet cannot be placed on the acyl
                    # parent — it comes back UNLOCATED ('cyano', 'dihydroxy'),
                    # double-counting it: C=CC(=O)NCC#N ->
                    # 'N-(cyanomethyl)cyanoprop-2-enamide',
                    # C=CC(=O)N(CCO)CCO -> 'N,N-bis(2-hydroxyethyl)dihydroxyprop-2-enamide'
                    # (parseable WRONG molecules gate-OFF). The acyl carbon is C1,
                    # so EVERY legitimate acyl-chain substituent carries a locant;
                    # an unlocated prefix is the double-count signature. Delegate
                    # only when every generated prefix is located, else fall
                    # through (fail closed — restores the pre-change behaviour).
                    _has_unlocated_prefix = any(
                        (not getattr(_p, 'locants', None))
                        and not str(getattr(_p, 'text', ''))[:1].isdigit()
                        for _p in _gen_prefixes_am(features)
                    )
                    if not _has_unlocated_prefix:
                        _nm2 = _composer_name_amide(features, "pin")
                        if _nm2 and _nm2 != "amide":
                            return _nm2
                # (A ring-attached anilide — benzoyl / cyclohexanecarbonyl on an
                # N-aryl bearing a junior FG — never reaches here: its acyl carbon
                # is not on the principal chain and its principal_chain is empty,
                # so it returns None at the top guard and is named by the lever-C
                # intercept in handlers/tier_a_ring.py via _name_ring_attached_anilide.)
                # OFF-chain acyl (N-side is the chain): the chain-suffix
                # machinery below is structurally wrong for this shape (double-
                # expresses the amide) -> fail closed. ON-chain acyl (the
                # ordinary N-aryl / N-alkyl amide, e.g. paracetamol
                # CC(=O)Nc1ccc(O)cc1 whose junior phenol lives on the N-aryl):
                # name_amide is the correct producer and was tried above; if its
                # guards did not hold, fall through to the chain machinery (which
                # correctly handles the cases the delegation does not cover).
                # Previously the acyl-on-chain N-aryl amide fell straight through
                # and the chain path mis-rooted the N-aryl as a C1 substituent
                # ('1-(4-hydroxyanilino)ethanamide', OPSIN-unparseable -> abstain).
                if _acyl_c not in set(principal_chain):
                    return None

    # --- Ester is the most-senior group (P-41: esters outrank acyl halides,
    # amides, nitriles, aldehydes, ketones, alcohols) ---
    # When get_principal_group selected the ester, the ester IS the most senior
    # group present, so it STAYS the principal characteristic group (suffix
    # '-oate') and the junior groups become prefixes (oxo/halo/cyano/hydroxy).
    # Name it as "alkyl <acid-with-prefixes>oate" via the acid-analog builder.
    # (The legacy EL-02 fallback below DEMOTED the ester and promoted a LESS
    # senior group to PCG -- a different, wrong molecule; kept only as a
    # fail-closed fallback for cases the acid-analog path declines.)
    # P-35.4.2 / P-65.2.1 (BB 18114, W2E-P1FC Task 7): the acyl HALIDE of a
    # mono-ester of carbonic acid, X-C(=O)-O-R, is perceived here as ester +
    # acid_halide on the SAME carbonyl and would otherwise be mis-built as
    # 'ethyl 1-chloro-1-oxomethanoate' (OPSIN-invalid). It is the functional-
    # class name '<R> carbono<halide>idate' (benzyl carbonochloridate). Fires
    # ONLY on that exact skeleton (carbonyl C: one =O, one halide, one ester-O
    # to a nameable R, NO carbon neighbour); else falls through fail-closed.
    if principal_group == "ester":
        _ester_matches_all = features.functional_groups.get("ester", [])
        _halide_fgs = ("acid_chloride", "acid_bromide",
                       "acid_fluoride", "acid_iodide")
        if (len(_ester_matches_all) == 1
                and any(features.functional_groups.get(h)
                        for h in _halide_fgs)):
            from ..rules.acid_halides import (
                HALIDE_WORDS,
                name_carbonic_monoester_acyl_halide,
            )
            _em = _ester_matches_all[0]
            _acyl_c = _em[0]
            _c_at = mol.GetAtomWithIdx(_acyl_c)
            _hal = next((nb for nb in _c_at.GetNeighbors()
                         if nb.GetSymbol() in ('Cl', 'Br', 'F', 'I')), None)
            _o_dbl = next((nb.GetIdx() for nb in _c_at.GetNeighbors()
                           if nb.GetSymbol() == 'O'
                           and mol.GetBondBetweenAtoms(
                               _acyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                          None)
            if _hal is not None and _o_dbl is not None:
                _hal_pg = {'Cl': 'acid_chloride', 'Br': 'acid_bromide',
                           'F': 'acid_fluoride', 'I': 'acid_iodide'}[
                    _hal.GetSymbol()]
                _cc = name_carbonic_monoester_acyl_halide(
                    mol, (_acyl_c, _o_dbl, _hal.GetIdx()), HALIDE_WORDS[_hal_pg]
                )
                if _cc is not None:
                    return _cc

    if principal_group == "ester":
        _ester_matches_all = features.functional_groups.get("ester", [])
        if len(_ester_matches_all) == 1:
            from ..rules.esters import name_polyfunctional_ester_via_acid
            _ester_pin = name_polyfunctional_ester_via_acid(
                mol, _ester_matches_all[0]
            )
            if _ester_pin:
                return _ester_pin
        elif len(_ester_matches_all) == 2:
            # (diacylglycerol-shape fix, PHASE6-DIESTER-OL-
            # SENIORITY): P-65.6.3.3.4.2 + P-44.3 -- a partially-esterified
            # polyol carrying TWO different noncyclic esters plus a free
            # hydroxyl. The senior acid (longer principal chain) stays the
            # functional-class parent; the other ester becomes an acyloxy
            # prefix and the free -OH a hydroxy prefix, both on the 'yl'
            # word -- NOT the EL-02 demotion below, which wrongly promotes
            # the junior hydroxy class to principal (inverting P-41: ester
            # class 9 outranks hydroxy class 17). Fail-closed: declines
            # (None) for anything outside its narrow scope (a tied acid
            # length, a ring acid, an off-chain decoration, ...), falling
            # through to EL-02 unchanged.
            from ..rules.esters import name_polyfunctional_diester_free_hydroxy
            _diester_pin = name_polyfunctional_diester_free_hydroxy(
                mol, _ester_matches_all, principal_chain,
            )
            if _diester_pin:
                return _diester_pin

    # --- EL-02 (fallback): Ester demotion in polyfunctional context ---
    # Legacy path, reached only when the acid-analog naming above declined
    # (e.g. >1 ester, or a junior group on the removed alkyl side).
    ester_acyloxy_prefixes = []
    _esters_demoted = False
    if principal_group == "ester":
        from ..rules.esters import name_ester_as_prefix
        ester_matches = features.functional_groups.get("ester", [])
        for match in ester_matches:
            acyloxy = name_ester_as_prefix(mol, match)
            if acyloxy:
                # Find the alkyl carbon (last in match) on the principal chain
                alkyl_c = match[-1] if len(match) >= 4 else match[3] if len(match) > 3 else None
                locant = atom_to_locant.get(alkyl_c) if alkyl_c is not None else None
                ester_acyloxy_prefixes.append((acyloxy, locant))

        if ester_acyloxy_prefixes:
            _esters_demoted = True

        # Re-select principal group excluding esters
        filtered_fgs = {k: v for k, v in features.functional_groups.items() if k != "ester"}
        new_principal, new_atoms = get_principal_group(mol, filtered_fgs)
        if new_principal:
            principal_group = new_principal
            features.principal_group_atoms = new_atoms
            # Update non_principal to exclude the new principal group
            non_principal = {k: v for k, v in filtered_fgs.items()
                            if k != new_principal and k not in ("alkene", "alkyne")}

    # Collect all prefixes (FG prefixes + alkyl substituents)
    all_prefixes = []
    # AM-4: N atoms of chain-terminal amidines whose 'amino'/'imino' prefixes
    # are emitted below — their substituent branches must be skipped by the
    # alkyl-substituent walk (else double-counted as (ethylamino)/(dimethylamino)).
    _amidine_excluded_n: Set[int] = set()

    # W2E-P1FG Task 10 (P-66.1.1.4.5.1): pre-pass to suppress the primary_amide
    # FG whose H2N-CO-CO- unit is consumed by an '(oxamoylimino)' prefix on a
    # chain imine. The amide FG is a SEPARATE perception match that would
    # otherwise emit a spurious 'carbamoyl'; drop those matches whose amide C
    # is part of a recognized oxamoyl branch. Done before the FG loop so it is
    # order-independent w.r.t. the imine block below.
    if non_principal.get('imine') and non_principal.get('primary_amide'):
        from ..assembly.substituent_naming import oxamoyl_branch_name as _oxa_fn
        _oxamoyl_amide_c: Set[int] = set()
        for _im in non_principal['imine']:
            _imc, _imn = _im[0], _im[1]
            for _nb in mol.GetAtomWithIdx(_imn).GetNeighbors():
                if _nb.GetIdx() == _imc or _nb.GetAtomicNum() <= 1:
                    continue
                if _oxa_fn(mol, _imn, _nb.GetIdx()) is not None:
                    # both acyl carbons of the oxamoyl unit are consumed
                    _oxamoyl_amide_c.add(_nb.GetIdx())
                    for _c2 in mol.GetAtomWithIdx(_nb.GetIdx()).GetNeighbors():
                        if (_c2.GetAtomicNum() == 6
                                and _c2.GetIdx() != _imn):
                            _oxamoyl_amide_c.add(_c2.GetIdx())
        if _oxamoyl_amide_c:
            non_principal = dict(non_principal)
            non_principal['primary_amide'] = [
                _m for _m in non_principal['primary_amide']
                if not any(a in _oxamoyl_amide_c for a in _m)]

    # Add ester acyloxy prefixes if esters were demoted
    if ester_acyloxy_prefixes:
        # Group identical acyloxy prefixes for multipliers

        # (B): enclose the acyloxy prefix via the shared escalation
        # primitive ( -> [ -> { instead of a raw f"({acyloxy_name})", which
        # produced a double `((...)...)` when the name already held an inner
        # mark (e.g. an '(9Z)-octadec-9-enoyloxy' glycerolipid arm).
        from ..assembly.naming_utils import apply_enclosing_marks as _enc
        acyloxy_groups: Dict[str, List] = defaultdict(list)
        for acyloxy_name, locant in ester_acyloxy_prefixes:
            acyloxy_groups[acyloxy_name].append(locant)

        for acyloxy_name, locants in sorted(acyloxy_groups.items()):
            valid_locants = sorted(loc for loc in locants if loc is not None)
            count = len(locants)
            enc = _enc(acyloxy_name, -1)
            if count > 1 and valid_locants:
                locant_str = ",".join(str(loc) for loc in valid_locants)
                multiplier = get_multiplier_prefix(count, acyloxy_name)
                all_prefixes.append(f"{locant_str}-{multiplier}{enc}")
            elif valid_locants:
                all_prefixes.append(f"{valid_locants[0]}-{enc}")
            else:
                if count > 1:
                    multiplier = get_multiplier_prefix(count, acyloxy_name)
                    all_prefixes.append(f"{multiplier}{enc}")
                else:
                    all_prefixes.append(enc)

    # --- Generate FG prefixes from non-principal groups ---
    chain_set = set(principal_chain)
    ring_fg_groups = defaultdict(list)  # FGs on ring atoms, for ring substituent naming

    # P-14.3.4.2(a) / P-14.3.4.4: on a MONONUCLEAR (single-carbon 'methane')
    # CHAIN parent every substitutable position is trivially locant '1', so both
    # prefix and suffix locants are omitted (BB verbatim 'chloromethanol', line
    # 5110 — a methane parent with a chloro prefix AND an -ol suffix, neither
    # carrying a '1'). Route the decision through the shared should_omit_locant_one
    # chokepoint. Gated to a chain parent so a ring parent whose principal_chain
    # happens to be one atom does not trigger it.
    from ..assembly.naming_utils import should_omit_locant_one as _should_omit_l1
    _mononuclear_chain_parent = (
        len(principal_chain) == 1
        and not (getattr(features, 'is_cyclic', False)
                 and not getattr(features, 'chain_is_parent', True))
    )
    _omit_mononuclear_locants = (
        _mononuclear_chain_parent
        and _should_omit_l1(context="prefix", chain_length=len(principal_chain))
    )

    # ------------------------------------------------------------------------- #
    # P-14.3.4.3 (BB:2939) -- the THIRD handler that reaches this licence class.  #
    # ------------------------------------------------------------------------- #
    # Measured 2026-07-30: `OC(=O)C(N)C(=O)O` and `OC(=O)C(O)C(=O)O` are named by
    # THIS handler (POOL source `polyfunctional`), not by `general_acyclic`, because
    # amino/hydroxy are detected functional groups and chloro is not -- so
    # `chloropropanedioic acid` (BB:2951, verbatim (PIN)) got the licence via
    # `_l3_prefix_locant_omitted` while `2-aminopropanedioic acid` and
    # `2-hydroxypropanedioic acid` recorded ZERO calls to it. Same parent compound,
    # same one-prefix-one-locant scope, different handler.
    #
    # The PIN really is the systematic form for both: BB:4973 `HOOC-CH2-COOH
    # propanedioic acid (PIN) malonic acid (not 2-carboxyacetic acid)` and BB:2951
    # `chloropropanedioic acid (PIN) chloromalonic acid` establish that a
    # substituted malonic acid is named on the systematic parent. Their trivial
    # names -- tartronic acid (2-hydroxy-) and aminomalonic acid (2-amino-) --
    # appear ZERO times anywhere in BlueBookV2.md, so no retained name pre-empts
    # them (checked because P-63/P-103 do retain some acid and amino-acid names).
    #
    # ★ "monosubstituted" is established STRUCTURALLY, before any prefix is
    # rendered, and that is what makes the decision safe in a handler whose
    # `all_prefixes` list is assembled from eight different branches:
    # `l3_locant_omitted_for_parent_atoms` proves via `_one_substituent_removed`
    # that the atoms outside the parent compound form exactly ONE connected
    # component joined by exactly ONE bond. A second FG prefix, ring prefix, alkyl
    # prefix or N-substituent prefix would each be a further such component, so the
    # proof fails for every scope that could hold two prefixes. The
    # `len(all_prefixes) == 1` re-check at the application site below is belt and
    # braces against a branch double-counting one component.
    _l3_parent_atoms = set()
    for _seq in (principal_chain or (),
                 getattr(features, 'principal_ring', None) or ()):
        for _idx in _seq:
            _l3_parent_atoms.add(int(_idx))
    for _match in (getattr(features, 'principal_group_atoms', None) or ()):
        for _idx in _match:
            _l3_parent_atoms.add(int(_idx))
    from ..assembly.handlers._handler_shared import (
        locant_scope_is_a_name_component as _l3_scope_is_nested,
    )
    _l3_scope_ok = bool(
        _l3_parent_atoms
        and not _omit_mononuclear_locants        # that rule already omits, P-14.3.4.2(a)
        and not features.double_bonds and not features.triple_bonds
        and not features.stereocenters
        and not getattr(features, 'double_bond_stereo', None)
        # A ring parent's stem cites its own locants in this handler; restrict to a
        # chain parent, deny-by-default (mirrors the `_acetic_context` guard).
        and not (getattr(features, 'is_cyclic', False)
                 and not getattr(features, 'chain_is_parent', False))
        # The third scope check: this licence empties the whole name of locants.
        and not _l3_scope_is_nested()
    )
    #: (prefix_form, count) of the single FG prefix the licence has cleared, so the
    #: application below can RE-RENDER it from its parts. Never a string edit of the
    #: already-rendered prefix.
    _l3_render_from = None

    # Collect substituent branch atoms for FG-on-branch filtering (BUG-B).
    # FGs located entirely on a substituent branch are already named by the
    # substituent naming path (e.g., hydroxymethyl), so skip them here.
    _branch_atoms = set()
    if features.substituents:
        for _pos, sub_list in features.substituents.items():
            for sub_atoms in sub_list:
                _branch_atoms.update(sub_atoms)

    for fg_name, matches in non_principal.items():
        if not matches:
            continue

        # When chain is parent, separate FGs on ring from FGs on chain
        if getattr(features, 'chain_is_parent', False):
            ring_atom_set = set()
            for rg in getattr(features, 'ring_substituents_as_groups', []):
                ring_atom_set.update(rg)

            chain_matches = []
            for match in matches:
                # Check if this FG's center atom is on the chain
                center = _find_fg_center_atom(mol, match, fg_name)
                if center is not None and center in chain_set and center not in ring_atom_set:
                    chain_matches.append(match)
                elif any(a in chain_set and a not in ring_atom_set for a in match):
                    chain_matches.append(match)
                else:
                    # FG is on a ring - track for ring substituent naming
                    ring_fg_groups[fg_name].append(match)
            matches = chain_matches
            if not matches:
                continue

        # BUG-B guard: Skip FG matches located entirely on a small substituent
        # branch (<=3 carbons) when the substituent naming path demonstrably
        # handles them (producing e.g. "hydroxymethyl", "chloromethyl").
        # Uses shared BRANCH_HANDLED_FGS from naming_utils (unified in Phase 113).
        if _branch_atoms and fg_name in BRANCH_HANDLED_FGS:
            filtered_matches = []
            for match in matches:
                if not all(a in _branch_atoms for a in match):
                    filtered_matches.append(match)
                    continue
                # Check if this FG is on a small branch with carbons
                # (i.e., a branch where _name_heteroatom_substituent handles the FG)
                on_small_branch = False
                for _pos, sub_list in features.substituents.items():
                    for sub_atoms in sub_list:
                        sub_set = set(sub_atoms)
                        if all(a in sub_set for a in match):
                            # This branch contains the entire FG match
                            c_count = sum(1 for a in sub_atoms
                                          if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
                            if 1 <= c_count <= 3:
                                on_small_branch = True
                                break
                    if on_small_branch:
                        break
                if not on_small_branch:
                    filtered_matches.append(match)
            original_count = len(non_principal[fg_name])
            matches = filtered_matches
            if not matches:
                if original_count > 0:
                    logger.debug(
                        "substituent_all_candidates_filtered polyfunc_bugb: fg_name=%s filtered=%d "
                        "(substituent naming handles these on small branches)",
                        fg_name, original_count,
                    )
                continue

        # HYG-04 site#2 double-count guard: a secondary/tertiary amine whose
        # nitrogen sits in a substituent branch is named IN FULL by the
        # substituent-naming path as a single "(...amino)" prefix
        # (methylamino / dimethylamino / ...). Emitting a bare "amino" FG-prefix
        # for the same nitrogen here double-counts it, producing geminal-diamine
        # artifacts like "2-amino-2-(dimethylamino)ethan-1-ol". Drop such matches
        # so the substituent walk owns the naming. Primary -NH2 is unaffected:
        # its branch is carbon-free, the substituent walk skips it, and this
        # FG-prefix remains its sole, correct name. Mirrors the BUG-B guard above
        # but keyed on the nitrogen (the amine match also spans the chain carbon,
        # so an "entirely on branch" test would miss it).
        # AM-2 ROOT-2 (plan P1AM Task 11, P-66.1.1.3 / P-41): an amine match
        # whose nitrogen IS the nitrogen of a perceived amide FG double-
        # expresses that N (suffix 'amide' + '(dimethylamino)' prefix = a
        # DIFFERENT molecule, e.g. '2-amino-1-(dimethylamino)ethanamide' for
        # NCC(=O)N(C)C). The amide handler owns that N; drop such amine matches
        # unconditionally (P-41: the amide class outranks the amine).
        _N_SUBSTITUTED_AMINE_FGS = {'secondary_amine', 'tertiary_amine'}
        if fg_name in ({'primary_amine', 'aromatic_amine'}
                       | _N_SUBSTITUTED_AMINE_FGS):
            _amide_n_atoms = set()
            for _ak in ('primary_amide', 'secondary_amide', 'tertiary_amide',
                        'thioamide', 'selenoamide', 'telluroamide'):
                for _am in features.functional_groups.get(_ak, ()):
                    _amide_n_atoms.update(
                        a for a in _am
                        if mol.GetAtomWithIdx(a).GetSymbol() == 'N')
            if _amide_n_atoms:
                matches = [
                    _m for _m in matches
                    if not any(a in _amide_n_atoms for a in _m
                               if mol.GetAtomWithIdx(a).GetSymbol() == 'N')]
                if not matches:
                    continue

        if fg_name in _N_SUBSTITUTED_AMINE_FGS and features.substituents:
            _amine_kept = []
            for _match in matches:
                _n_owned_by_branch = False
                for _n in (a for a in _match
                           if mol.GetAtomWithIdx(a).GetSymbol() == 'N'):
                    for _sub_list in features.substituents.values():
                        if any(_n in _sub_atoms for _sub_atoms in _sub_list):
                            _n_owned_by_branch = True
                            break
                    if _n_owned_by_branch:
                        break
                if not _n_owned_by_branch:
                    _amine_kept.append(_match)
            if not _amine_kept:
                logger.debug(
                    "suppress_duplicate_bare_amino_prefix polyfunc: skip bare 'amino' FG-prefix for %s — "
                    "N-substituted amine named in full by the substituent walk",
                    fg_name,
                )
                continue
            matches = _amine_kept

        # Non-principal acyl halide on a CHAIN parent (P-65.5.4): the
        # acyl-halide carbon is a chain member, expressed as 'oxo' (=O) +
        # 'halo' (X), NOT the 'carbonochloridoyl' prefix.  Blue Book worked
        # examples: "4-chloro-4-oxobutanoic acid" (PIN, line 5108),
        # "3-chloro-3-oxopropanoic acid" (PIN, line 31531).  The
        # 'carbonochloridoyl' prefix is the PIN only on a RING parent (the
        # carbon cannot join the ring, e.g. "2-carbonochloridoylbenzoic acid",
        # line 31533) — handled elsewhere; here we fall through to it only when
        # the acyl-halide carbon is NOT a member of the principal chain.
        _ACYL_HALIDE_HALO = {
            'acid_fluoride': 'fluoro', 'acid_chloride': 'chloro',
            'acid_bromide': 'bromo', 'acid_iodide': 'iodo',
        }
        if fg_name in _ACYL_HALIDE_HALO and chain_set:
            _ah_locs = []
            for _m in matches:
                _c = _m[0]  # acyl-halide carbonyl carbon (SMARTS index 0)
                if _c in chain_set:
                    _loc = atom_to_locant.get(_c)
                    if _loc is not None:
                        _ah_locs.append(_loc)
            if _ah_locs and len(_ah_locs) == len(matches):
                _ah_locs.sort()
                _n = len(_ah_locs)
                all_prefixes.append(format_fg_prefix('oxo', _ah_locs, _n))
                all_prefixes.append(
                    format_fg_prefix(_ACYL_HALIDE_HALO[fg_name], _ah_locs, _n)
                )
                continue

        # W3-P03-5 (P-65.1.6.1, BB 30384): a non-principal PRIMARY amide whose
        # carbonyl C terminates an ACYCLIC chain parent (a senior group present,
        # e.g. COOH) is expressed as 'oxo' (=O) + 'amino' (-NH2) at that carbon's
        # locant -- '4-amino-4-oxobutanoic acid' (PIN) -- NOT the 'carbamoyl'
        # prefix (the non-PIN general form '3-carbamoylpropanoic acid', and the
        # PIN only for a RING-attached amide, L30377 '2-carbamoylbenzoic acid').
        # Mirrors the acid-halide oxo+halo split above. Fires only when EVERY
        # primary-amide C is a chain member; else falls through to the carbamoyl
        # PREFIX_FORMS path (ring / off-chain case). SMARTS [CX3](=O)[NX3H2]:
        # match[0]=C, match[2]=N.
        if fg_name == 'primary_amide' and chain_set:
            _pa_locs = []
            _pa_ns = []
            for _m in matches:
                _pac = _m[0]
                if _pac in chain_set:
                    _loc = atom_to_locant.get(_pac)
                    if _loc is not None:
                        _pa_locs.append(_loc)
                        _pa_ns.append(_m[2])
            if _pa_locs and len(_pa_locs) == len(matches):
                _pa_locs.sort()
                _n = len(_pa_locs)
                all_prefixes.append(format_fg_prefix('oxo', _pa_locs, _n))
                all_prefixes.append(format_fg_prefix('amino', _pa_locs, _n))
                _amidine_excluded_n.update(_pa_ns)
                continue

        # W3-P03-2 (P-65.1.4.2, BB 30197): a non-principal PEROXY acid -CO-OOH
        # whose carbonyl C terminates an ACYCLIC chain parent (a senior group
        # present, e.g. COOH) is expressed as 'oxo' (=O) + 'hydroperoxy' (-OOH)
        # at that carbon's locant -- '6-hydroperoxy-6-oxohexanoic acid' (PIN,
        # L30193: "the combination of the simple prefixes hydroperoxy and oxo ...
        # is used in preferred IUPAC names rather than ... carbonoperoxoyl-").
        # Mirrors the acid-halide oxo+halo and amide oxo+amino splits above.
        # Fires only when every peroxy-acid C is a chain member; else falls
        # through (PREFIX_FORMS['peroxy_acid'] is None -> the RING/off-chain
        # 'carbonoperoxoyl' acyl prefix is not built -> fail closed). SMARTS
        # ...(=[OX1])[OX2][OX2H1]: match[0]=C, [1]==O, [2]=O, [3]=OH.
        if fg_name == 'peroxy_acid' and chain_set:
            _px_locs = []
            _px_os = []
            for _m in matches:
                _pxc = _m[0]
                if _pxc in chain_set:
                    _loc = atom_to_locant.get(_pxc)
                    if _loc is not None:
                        _px_locs.append(_loc)
                        _px_os.extend([_m[1], _m[2], _m[3]])
            if _px_locs and len(_px_locs) == len(matches):
                _px_locs.sort()
                _n = len(_px_locs)
                all_prefixes.append(format_fg_prefix('oxo', _px_locs, _n))
                all_prefixes.append(
                    format_fg_prefix('hydroperoxy', _px_locs, _n))
                _amidine_excluded_n.update(_px_os)
                continue

        # W3-P03-6 (P-65.1.6.2, BB 30396): a non-principal SECONDARY amide
        # -CO-NH-R whose carbonyl C terminates an ACYCLIC chain parent (a senior
        # group present, e.g. COOH) is expressed as 'oxo' (=O) + the amino-side
        # prefix at that carbon's locant. The N-phenyl anilide's amino side is the
        # retained 'anilino' -> '5-anilino-5-oxopentanoic acid' (PIN, L30392);
        # a general N-alkyl gives '({alkyl}amino)'. NOT the 'N-...carbamoyl'
        # prefix (which mis-places the in-chain carbon; the old generic path even
        # walked the phenyl ring open to 'hexyl'). Mirrors the primary-amide
        # oxo+amino split above. The N-ATTACHED orientation (chain-N-CO-R,
        # acetamido) is UNAFFECTED: its amide C is off-chain so the guard declines.
        # Fires only when every secondary-amide C is a chain member with exactly
        # one nameable N-substituent; else falls through (fail closed).
        # SMARTS [CX3](=O)[NX3H1][#6]: match[0]=C, match[2]=N.
        if fg_name == 'secondary_amide' and chain_set:
            from ..assembly.composer import _name_r_group
            _sa_ok = True
            _sa_units = {}  # amide C -> (amino-side prefix, N-side branch atoms)
            for _m in matches:
                _sac, _san = _m[0], _m[2]
                if _sac not in chain_set:
                    _sa_ok = False
                    break
                _r_atoms = [
                    nb.GetIdx()
                    for nb in mol.GetAtomWithIdx(_san).GetNeighbors()
                    if nb.GetIdx() != _sac and nb.GetAtomicNum() > 1]
                if len(_r_atoms) != 1:
                    _sa_ok = False
                    break
                _r_name = _name_r_group(mol, _r_atoms[0], exclude_atoms={_san})
                if not _r_name:
                    _sa_ok = False
                    break
                # P-62.2.1.1.1 (BB:26139): 'anilino' is the retained PREFERRED
                # PREFIX for C6H5-NH- with FULL SUBSTITUTION ALLOWED, so a
                # substituted ring substitutes ONTO 'anilino' (BB:26153
                # '4-chloroanilino (preferred prefix) | (4-chlorophenyl)amino').
                # The old else-branch put the general-nomenclature form in a PIN
                # ('5-(4-chlorophenylamino)-5-oxopentanoic acid' shipped) — and not
                # even the well-formed general spelling, which BB:26153 writes with
                # the inner enclosure. Bare 'anilino' stays bare (BB:26306).
                # Non-phenyl rings keep the legacy '({R}amino)' (P-16.3.3).
                from .ring_substituents import anilino_preferred_prefix
                _amino_prefix = anilino_preferred_prefix(_r_name)
                if _amino_prefix is None:
                    _amino_prefix = f'({_r_name}amino)'
                # Collect the whole N-side branch (N + R subtree), never crossing
                # the amide C, to exclude it from the alkyl substituent walk.
                _seen = {_sac}
                _stack = [_san]
                _branch = []
                while _stack:
                    _a = _stack.pop()
                    if _a in _seen:
                        continue
                    _seen.add(_a)
                    _branch.append(_a)
                    for _nb in mol.GetAtomWithIdx(_a).GetNeighbors():
                        if _nb.GetIdx() not in _seen:
                            _stack.append(_nb.GetIdx())
                _sa_units[_sac] = (_amino_prefix, _branch)
            if _sa_ok and _sa_units:
                _sa_prefixes = []
                for _sac, (_amino_prefix, _branch) in _sa_units.items():
                    _loc = atom_to_locant.get(_sac)
                    if _loc is None:
                        _sa_ok = False
                        break
                    _sa_prefixes.append(format_fg_prefix('oxo', [_loc], 1))
                    _sa_prefixes.append(
                        format_fg_prefix(_amino_prefix, [_loc], 1))
                if _sa_ok:
                    all_prefixes.extend(_sa_prefixes)
                    for _sac, (_amino_prefix, _branch) in _sa_units.items():
                        _amidine_excluded_n.update(_branch)
                    continue

        # AM-6 (P-66.1.1.4.3): a sulfonamide bonded to the chain via its N
        # (R-SO2-NH-chain) is expressed by _check_for_acylamino as the
        # '(...sulfonamido)' prefix, NOT the 'sulfamoyl' FG prefix (which is the
        # S-attached orientation, chain-SO2-NH2). Skip the sulfamoyl prefix when
        # NO match has its sulfonyl S bonded to a chain atom. S-on-chain matches
        # (e.g. 3-sulfamoylpropanoic acid) keep the existing path byte-identical.
        _SULFONAMIDE_N_ATTACH_FGS = {
            'primary_sulfonamide', 'secondary_sulfonamide',
            'tertiary_sulfonamide',
        }
        if fg_name in _SULFONAMIDE_N_ATTACH_FGS and chain_set:
            _s_on_chain = False
            for _m in matches:
                for _a in _m:
                    if mol.GetAtomWithIdx(_a).GetSymbol() != 'S':
                        continue
                    if any(nb.GetIdx() in chain_set
                           for nb in mol.GetAtomWithIdx(_a).GetNeighbors()):
                        _s_on_chain = True
                    break
                if _s_on_chain:
                    break
            if not _s_on_chain:
                continue

        # P-66.1.4.4 (BB 33196, W2E-P1FG Task 9): a thioamide carbon on the
        # principal chain, junior to a senior suffix (acid/ester), is expressed
        # by 'amino' (the single-bonded N; '({alkyl}amino)' if substituted) +
        # 'sulfanylidene' (the =S) at that C's locant — BB verbatim
        # '3-amino-3-sulfanylidenepropanoic acid'. Mirrors AM-4. Off-chain /
        # ring thioamide keeps the existing 'carbamothioyl' PREFIX_FORMS path
        # (seniority.py). Fires only when EVERY thioamide C is a chain member
        # with the amino side nameable; else falls through to carbamothioyl.
        if fg_name == 'thioamide' and chain_set:
            _ta_ok = True
            _ta_units = {}  # thioamide C -> (S idx, N idx)
            for _m in matches:
                _tac, _tas, _tan = _m[0], _m[1], _m[2]
                if _tac not in chain_set:
                    _ta_ok = False
                    break
                _ta_units[_tac] = (_tas, _tan)
            if _ta_ok and _ta_units:
                _ta_prefixes = []
                for _tac, (_tas, _tan) in _ta_units.items():
                    _loc = atom_to_locant.get(_tac)
                    _amino = _name_amidine_chain_side(mol, _tan, _tac, 'amino')
                    if _loc is None or _amino is None:
                        _ta_ok = False
                        break
                    _ta_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                    _ta_prefixes.append(
                        format_fg_prefix('sulfanylidene', [_loc], 1))
                if _ta_ok:
                    all_prefixes.extend(_ta_prefixes)
                    for _tac, (_tas, _tan) in _ta_units.items():
                        _amidine_excluded_n.add(_tan)
                    continue

        # P-66.1.1.4.5.1 (BB 33071/55479, W2E-P1FG Task 10): a chain imine C
        # whose =N bears the exact H2N-CO-CO- branch is cited as the preferred
        # composite prefix '(oxamoylimino)' at the imine C's locant. A =N
        # carrying ANY OTHER heavy substituent the recognizer cannot name is
        # SKIPPED WITHOUT emitting bare 'imino' (fail closed — the dropped
        # branch then fails coverage -> unknown). An unsubstituted =NH keeps
        # the existing bare-'imino' generic path.
        if fg_name == 'imine' and chain_set:
            from ..assembly.substituent_naming import oxamoyl_branch_name
            _im_prefixes = []
            _im_ok = True
            _im_handled = False
            for _m in matches:
                _imc, _imn = _m[0], _m[1]
                _imn_a = mol.GetAtomWithIdx(_imn)
                _heavy = [nb.GetIdx() for nb in _imn_a.GetNeighbors()
                          if nb.GetIdx() != _imc and nb.GetAtomicNum() > 1]
                if not _heavy:
                    continue  # unsubstituted =NH -> generic 'imino' path owns it
                if _imc not in chain_set or len(_heavy) != 1:
                    _im_ok = False
                    break
                _oxa = oxamoyl_branch_name(mol, _imn, _heavy[0])
                _loc = atom_to_locant.get(_imc)
                if _oxa is None or _loc is None:
                    _im_ok = False   # substituted =N we cannot name -> fail closed
                    break
                _im_prefixes.append(
                    format_fg_prefix(f"({_oxa}imino)", [_loc], 1))
                _amidine_excluded_n.add(_imn)
                _im_handled = True
            if not _im_ok:
                return None  # never a bare 'imino' dropping the branch
            if _im_handled:
                all_prefixes.extend(_im_prefixes)
                continue

        # AM-4 (P-66.4.1.3.2, BB 34338): an amidine carbon that TERMINATES a
        # chain parent is expressed with -NH2 -> 'amino' and =NH -> 'imino'
        # (each N-substituent decorating the prefix), NOT 'carbamimidoyl' (the
        # PIN only for ring / off-chain amidine carbons, BB 34332). Fires only
        # when every amidine carbon is a chain member with exactly one =N and
        # one single-bonded -N, both nameable; else falls through to the
        # carbamimidoyl prefix path below.
        if fg_name == 'amidine' and chain_set:
            _seen_amc = {}
            _am_ok = True
            for _m in matches:
                _amc = _dbl = _sgl = None
                for _a in _m:
                    _at = mol.GetAtomWithIdx(_a)
                    if _at.GetSymbol() != 'C':
                        continue
                    _dN = [
                        nb.GetIdx() for nb in _at.GetNeighbors()
                        if nb.GetSymbol() == 'N'
                        and mol.GetBondBetweenAtoms(
                            _a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                    ]
                    _sN = [
                        nb.GetIdx() for nb in _at.GetNeighbors()
                        if nb.GetSymbol() == 'N'
                        and mol.GetBondBetweenAtoms(
                            _a, nb.GetIdx()).GetBondTypeAsDouble() == 1.0
                    ]
                    if len(_dN) == 1 and len(_sN) == 1:
                        _amc, _dbl, _sgl = _a, _dN[0], _sN[0]
                        break
                if _amc is None or _amc not in chain_set:
                    _am_ok = False
                    break
                _seen_amc[_amc] = (_dbl, _sgl)
            if _am_ok and _seen_amc:
                _am_prefixes = []
                _am_hydrazinyl_tails = {}
                for _amc, (_dbl, _sgl) in _seen_amc.items():
                    _loc = atom_to_locant.get(_amc)
                    # P-66.4.2.3.1 (BB 34490, plan P1AM Task 6): when the
                    # single-bonded N's sole heavy branch is a terminal NH2,
                    # the -NH-NH2 unit is ONE 'hydrazinyl' prefix (PIN
                    # '3-hydrazinyl-3-iminopropanoic acid'), never the
                    # fragmenting '(aminoamino)'.
                    _sgl_branches = [
                        nb.GetIdx()
                        for nb in mol.GetAtomWithIdx(_sgl).GetNeighbors()
                        if nb.GetIdx() != _amc and nb.GetAtomicNum() > 1]
                    _hydrazinyl_tail = None
                    if (len(_sgl_branches) == 1
                            and mol.GetAtomWithIdx(_sgl_branches[0]).GetSymbol() == 'N'
                            and mol.GetAtomWithIdx(_sgl_branches[0]).GetDegree() == 1):
                        _amino = 'hydrazinyl'
                        _hydrazinyl_tail = _sgl_branches[0]
                    else:
                        _amino = _name_amidine_chain_side(mol, _sgl, _amc, 'amino')
                    _imino = _name_amidine_chain_side(mol, _dbl, _amc, 'imino')
                    if _loc is None or _amino is None or _imino is None:
                        _am_ok = False
                        break
                    _am_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                    _am_prefixes.append(format_fg_prefix(_imino, [_loc], 1))
                    if _hydrazinyl_tail is not None:
                        _am_hydrazinyl_tails[_amc] = _hydrazinyl_tail
                if _am_ok:
                    all_prefixes.extend(_am_prefixes)
                    for _amc, (_dbl, _sgl) in _seen_amc.items():
                        _amidine_excluded_n.add(_dbl)
                        _amidine_excluded_n.add(_sgl)
                        if _amc in _am_hydrazinyl_tails:
                            _amidine_excluded_n.add(_am_hydrazinyl_tails[_amc])
                    continue

        # P-66.4.2.3.2 (BB 34498, plan P1AM Task 6): an amidrazone carbon
        # -C(=N-NH2)-NH2 that TERMINATES a chain parent is expressed with
        # -NH2 -> 'amino' and =N-NH2 -> 'hydrazinylidene'. Fires only when
        # every match's amidrazone C is a chain member, the =N-NH2 hydrazono
        # arm is UNSUBSTITUTED (terminal NH2), and the amino side is
        # nameable; else falls through (off-chain matches reach the PF-2
        # filter + 'carbamohydrazonoyl' path below unchanged).
        if fg_name == 'hydrazonamide' and chain_set:
            _hz_ok = True
            _hz_units = {}
            for _m in matches:
                _hc = _m[0]  # SMARTS [CX3](=[NX2][NX2,NX3])[NX3]: match[0]=C
                if _hc not in chain_set:
                    _hz_ok = False
                    break
                _at = mol.GetAtomWithIdx(_hc)
                _dbl = next((nb.GetIdx() for nb in _at.GetNeighbors()
                             if nb.GetSymbol() == 'N'
                             and mol.GetBondBetweenAtoms(
                                 _hc, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                            None)
                _sgl = next((nb.GetIdx() for nb in _at.GetNeighbors()
                             if nb.GetSymbol() == 'N'
                             and mol.GetBondBetweenAtoms(
                                 _hc, nb.GetIdx()).GetBondTypeAsDouble() == 1.0),
                            None)
                if _dbl is None or _sgl is None:
                    _hz_ok = False
                    break
                # =N-NH2 arm must be exactly one terminal NH2 (unsubstituted)
                _dbl_branches = [nb.GetIdx()
                                 for nb in mol.GetAtomWithIdx(_dbl).GetNeighbors()
                                 if nb.GetIdx() != _hc and nb.GetAtomicNum() > 1]
                if (len(_dbl_branches) != 1
                        or mol.GetAtomWithIdx(_dbl_branches[0]).GetSymbol() != 'N'
                        or mol.GetAtomWithIdx(_dbl_branches[0]).GetDegree() != 1):
                    _hz_ok = False
                    break
                _hz_units[_hc] = (_dbl, _dbl_branches[0], _sgl)
            if _hz_ok and _hz_units:
                _hz_prefixes = []
                for _hc, (_dbl, _nh2, _sgl) in _hz_units.items():
                    _loc = atom_to_locant.get(_hc)
                    _amino = _name_amidine_chain_side(mol, _sgl, _hc, 'amino')
                    if _loc is None or _amino is None:
                        _hz_ok = False
                        break
                    _hz_prefixes.append(format_fg_prefix(_amino, [_loc], 1))
                    _hz_prefixes.append(
                        format_fg_prefix('hydrazinylidene', [_loc], 1))
                if _hz_ok:
                    all_prefixes.extend(_hz_prefixes)
                    for _hc, (_dbl, _nh2, _sgl) in _hz_units.items():
                        _amidine_excluded_n.update({_dbl, _nh2, _sgl})
                    continue

        # P-66.4.2.3.1 (BB 34490, plan P1AM Task 6/7): the imidohydrazide
        # tautomer -C(=NH)-NH-NH2 that TERMINATES a chain parent is the same
        # chain-end split as the amidine hydrazinyl case: =NH -> 'imino' and
        # -NH-NH2 -> 'hydrazinyl' ('3-hydrazinyl-3-iminopropanoic acid' for
        # N=C(NN)CC(=O)O). Perception (Task 7) reclassified this molecule from
        # amidine+hydrazine_fg to imidohydrazide, so the amidine block above no
        # longer catches it — handle the same shape here. SMARTS
        # [CX3](=[NX2;D1])[NX3][NX3H2]: match[0]=C, [1]==NH, [2]=NH, [3]=NH2.
        # Fires only when every C is a chain member and the -NH-NH2 arm is a
        # terminal NH2; else falls through (fail closed).
        if fg_name == 'imidohydrazide' and chain_set:
            _ih_ok = True
            _ih_units = {}
            for _m in matches:
                if len(_m) < 4:
                    _ih_ok = False
                    break
                _ihc, _ih_dbl, _ih_sgl, _ih_nh2 = _m[0], _m[1], _m[2], _m[3]
                if _ihc not in chain_set:
                    _ih_ok = False
                    break
                # -NH-NH2 arm must terminate in a degree-1 NH2
                if mol.GetAtomWithIdx(_ih_nh2).GetDegree() != 1:
                    _ih_ok = False
                    break
                _ih_units[_ihc] = (_ih_dbl, _ih_sgl, _ih_nh2)
            if _ih_ok and _ih_units:
                _ih_prefixes = []
                for _ihc, (_ih_dbl, _ih_sgl, _ih_nh2) in _ih_units.items():
                    _loc = atom_to_locant.get(_ihc)
                    _imino = _name_amidine_chain_side(mol, _ih_dbl, _ihc, 'imino')
                    if _loc is None or _imino is None:
                        _ih_ok = False
                        break
                    _ih_prefixes.append(format_fg_prefix('hydrazinyl', [_loc], 1))
                    _ih_prefixes.append(format_fg_prefix(_imino, [_loc], 1))
                if _ih_ok:
                    all_prefixes.extend(_ih_prefixes)
                    for _ihc, (_ih_dbl, _ih_sgl, _ih_nh2) in _ih_units.items():
                        _amidine_excluded_n.update({_ih_dbl, _ih_sgl, _ih_nh2})
                    continue

        # W3-P02-4 (P-65.1.3.2.2, BB 30089): a hydrazonic acid C(=N-NH2)(OH)
        # DEMOTED by a senior group (COOH) whose geminal carbon sits on the
        # parent chain splits into 'hydroxy' (the geminal C-OH) + 'hydrazinylidene'
        # (the =N-NH2) prefixes on that carbon's locant — PIN
        # '5-hydrazinylidene-5-hydroxypentanoic acid'. SMARTS match tuple
        # (C, imino-N, amino-N, hydroxyl-O). Fires only when every match's C is a
        # chain member; else falls through (fail closed). The imino-N/amino-N/O
        # are added to the substituent-exclusion set so the alkyl walk does not
        # re-name the =N-NH2 or -OH branch.
        if fg_name == 'hydrazonic_acid' and chain_set:
            _hzc_ok = True
            _hzc_units = {}
            for _m in matches:
                if len(_m) < 4:
                    _hzc_ok = False
                    break
                _c, _nim, _nam, _o = _m[0], _m[1], _m[2], _m[3]
                if _c not in chain_set:
                    _hzc_ok = False
                    break
                _hzc_units[_c] = (_nim, _nam, _o)
            if _hzc_ok and _hzc_units:
                _hzc_prefixes = []
                for _c, (_nim, _nam, _o) in _hzc_units.items():
                    _loc = atom_to_locant.get(_c)
                    if _loc is None:
                        _hzc_ok = False
                        break
                    _hzc_prefixes.append(format_fg_prefix('hydroxy', [_loc], 1))
                    _hzc_prefixes.append(
                        format_fg_prefix('hydrazinylidene', [_loc], 1))
                if _hzc_ok:
                    all_prefixes.extend(_hzc_prefixes)
                    for _c, (_nim, _nam, _o) in _hzc_units.items():
                        _amidine_excluded_n.update({_nim, _nam, _o})
                    continue

        # W3-P02-6 (P-65.1.3.3.2, BB 30140): a hydroximic acid C(=N-OH)(OH)
        # DEMOTED by a senior group (COOH) whose geminal carbon sits on the
        # parent chain splits into 'hydroxy' (the geminal C-OH) + '(hydroxyimino)'
        # (the =N-OH) prefixes on that carbon's locant — PIN
        # '5-hydroxy-5-(hydroxyimino)pentanoic acid'. 'hydroxyimino' is a compound
        # prefix (hydroxy + imino) and is parenthesised per P-16.5.1.1; scoped here
        # to the hydroximic_acid class (the general demoted-oxime path P-66.6 is a
        # separate class, out of scope). SMARTS match tuple
        # (C, imino-N, O-on-N, hydroxyl-O). Fires only when every match's C is a
        # chain member; else falls through (fail closed).
        if fg_name == 'hydroximic_acid' and chain_set:
            _hxc_ok = True
            _hxc_units = {}
            for _m in matches:
                if len(_m) < 4:
                    _hxc_ok = False
                    break
                _c, _nim, _on, _oc = _m[0], _m[1], _m[2], _m[3]
                if _c not in chain_set:
                    _hxc_ok = False
                    break
                _hxc_units[_c] = (_nim, _on, _oc)
            if _hxc_ok and _hxc_units:
                _hxc_prefixes = []
                for _c, (_nim, _on, _oc) in _hxc_units.items():
                    _loc = atom_to_locant.get(_c)
                    if _loc is None:
                        _hxc_ok = False
                        break
                    _hxc_prefixes.append(format_fg_prefix('hydroxy', [_loc], 1))
                    _hxc_prefixes.append(
                        format_fg_prefix('(hydroxyimino)', [_loc], 1))
                if _hxc_ok:
                    all_prefixes.extend(_hxc_prefixes)
                    for _c, (_nim, _on, _oc) in _hxc_units.items():
                        _amidine_excluded_n.update({_nim, _on, _oc})
                    continue

        # PF-2 (P-66.4.2.3.5): an amidrazone attached to the chain via its AMINO
        # nitrogen (chain-N(H)-C(=N-NH2)-R) is expressed by _check_for_acylamino
        # as the '(...hydrazonamido)' prefix, NOT the C-attached
        # 'carbamohydrazonoyl' FG prefix. SMARTS [CX3](=[NX2][NX2,NX3])[NX3] ->
        # match[0] is the amidrazone carbon; keep a match only when THAT carbon
        # is bonded to a chain atom (the genuine C-attached direction). If none
        # survive (all N-attached) -> skip the FG prefix.
        if fg_name == 'hydrazonamide' and chain_set:
            _c_attached = [
                _m for _m in matches
                if _m and any(nb.GetIdx() in chain_set
                              for nb in mol.GetAtomWithIdx(_m[0]).GetNeighbors())
            ]
            if not _c_attached:
                continue
            matches = _c_attached

        # P-63.3.1 (BB 27858-27860): a peroxide / disulfide bridge R-XX-R' whose
        # far side is a walked substituent branch is named by method (1) --
        # substitutively, ONE MONOVALENT prefix ('tert-butylperoxy',
        # 'tert-butyldisulfanyl'), which the substituent walk builds whole from
        # the shared cascade. PREFIX_FORMS['disulfide'] holds 'disulfanediyl',
        # which is method (3): the MULTIPLICATIVE divalent bridge -SS-, licensed
        # only when both ends are parent structures. Emitting it here for a
        # single attachment both double-counted the S-S -- the walk already named
        # it, giving '2-disulfanediyl-2-(tert-butyldisulfanyl)ethan-1-ol' -- and
        # cited a divalent prefix for a monovalent group. Hand the class to the
        # walk. Nothing is dropped by doing so: a branch the walk cannot name
        # now fails the whole molecule closed there (see the FAILCLOSED-POLY-SUB
        # refusal in _generate_alkyl_prefixes_for_polyfunctional).
        if fg_name in _CHALCOGEN_BRIDGE_FGS and features.substituents:
            _bridge_kept = [
                _m for _m in matches
                if not _bridge_far_side_is_walked_branch(
                    mol, _m, chain_set, features)
            ]
            if not _bridge_kept:
                continue
            matches = _bridge_kept

        # Get prefix form for this FG
        prefix_form = get_fg_prefix_form(
            fg_name, mol, matches[0], principal_chain
        )
        if not prefix_form:
            # Phase 169 tier-3 fallback (POLY-01): when a composite loser FG
            # has no clean strict-IUPAC prefix (the substituent_no_prefix_form case), decompose it into
            # its ordered sub-group prefix components instead of dropping it. Gated
            # behind the default-OFF flag (Stage A byte-identical); the split only
            # fires for table-listed composites (ester/thioester/iminoester) and is
            # OPSIN-RT gated (FAIL-CLOSED,). Lazy import (Pattern-S3) avoids a
            # cycle. iminoester -> imino + alkoxy (P-65.1.3.1.2(2), the chain-
            # end simple-prefix form '4-imino-4-methoxybutanoic acid').
            components = None
            _anchored = []
            _split_default = (
                principal_group in _SPLIT_DEFAULT_PGS
                and fg_name in ("ester", "thioester", "iminoester")
            )
            if getattr(features, "_enable_group_splitting", False) or _split_default:
                from ..assembly.group_splitting import (
                    _get_default_oracle,
                    split_composite_fg,
                )
                # W2F-P2 (P-65.6.3.3.5): split ONLY the match whose carbonyl C
                # is IN the principal chain (in-chain ester C -> oxo + R-oxy /
                # R-sulfanyl); acyloxy-orientation matches (carbonyl off-chain)
                # keep today's continue (composer acyloxy path owns them). The
                # exactly-one guard keeps multi-(thio)ester acids fail-closed
                # (v1) and the locant computation single-valued (fixes the
                # brief item-6 '9,9' defect: locants from anchored ONLY).
                _anchored = [m for m in matches if m and m[0] in chain_set]
                if len(_anchored) == 1:
                    _oracle = (
                        getattr(features, "_split_oracle", None)
                        or _get_default_oracle()
                    )
                    components = split_composite_fg(
                        fg_name, mol, _anchored[0], principal_chain,
                        oracle=_oracle,
                    )
            if not components:
                # By-design: FGs using functional class naming (ester→alkoxycarbonyl,
                # secondary_amide→acylamino, thioether, etc.) are handled by
                # specialized naming paths, not as simple prefixes. Unchanged drop
                # path for the OFF / non-splittable / RT-rejected cases.
                logger.debug(
                    "substituent_no_prefix_form substituent_skip: reason=no_fg_prefix_form fg_name=%s",
                    fg_name,
                )
                continue
            # POLY-02 native: each split component re-enters the existing prefix
            # pipeline (format_fg_prefix + alpha_sort_key) sharing the central-carbon
            # locant (comp.locants is None -> the caller's locants apply to both).
            split_locants = get_non_principal_fg_locants(
                mol, _anchored, principal_chain, atom_to_locant, fg_name
            )
            for comp in components:
                _comp_locants = comp.locants or split_locants
                all_prefixes.append(
                    format_fg_prefix(
                        comp.prefix_form,
                        [] if _omit_mononuclear_locants else _comp_locants,
                        comp.count)
                )
            continue

        # Get locants for this FG
        locants = get_non_principal_fg_locants(
            mol, matches, principal_chain, atom_to_locant, fg_name
        )

        # W2F-P3 (defect a, fail-closed, P-14.3.2): get_non_principal_fg_locants
        # returns exactly one locant per on-chain / chain-adjacent match (no short-chain
        # empty-return special case — 'nitromethane' locant omission happens later via
        # should_omit_locant). After the BUG-B branch filter has already skipped
        # branch-OWNED matches (BRANCH_HANDLED_FGS now includes 'nitro'), any shortfall
        # here means an unlocatable off-chain FG instance that no branch namer owns ->
        # emitting a bare un-locanted prefix ('dinitro') or silently dropping instances
        # is structure loss. Refuse the whole handler (namer falls through / fails
        # closed) rather than emit a wrong name.
        if len(locants) < len(matches):
            return None
        count = len(locants)

        # Format the prefix (mononuclear parent -> omit the trivially-'1' locant)
        formatted = format_fg_prefix(
            prefix_form,
            [] if _omit_mononuclear_locants else locants,
            count)
        all_prefixes.append(formatted)

        # P-14.3.4.3: with the scope preconditions already established above, ask
        # the licence about THIS prefix's single locant. The rule itself is not
        # re-derived here -- it is `assembly.locant_omission`, the one place the
        # P-14.3.4 licences live, and it measures the orbits of the parent
        # compound's substitutable hydrogens (propanedioic acid: C2 only, because
        # both acid O-H sit on a chalcogen and are excluded by BB:3007 -> ONE kind
        # -> fires; pentanedioic acid: C2/C4 in one orbit and C3 in another -> two
        # kinds -> denied).
        if _l3_scope_ok and count == 1 and len(locants) == 1:
            from ..assembly.locant_omission import (
                l3_locant_omitted_for_parent_atoms as _l3_licence,
            )
            if _l3_licence(
                mol, _l3_parent_atoms,
                prefix_locants=list(locants),
                suffix_locants=[],
                parent_cites_locants=False,
                stereo_text="",
            ):
                _l3_render_from = (prefix_form, count)

    # --- Generate ring substituent prefixes (when chain is parent) ---
    if getattr(features, 'chain_is_parent', False):
        from ..assembly.composer import _generate_ring_substituent_prefixes
        ring_prefixes = _generate_ring_substituent_prefixes(features)
        for ring_prefix in ring_prefixes:
            all_prefixes.append(ring_prefix.text)

    # --- Generate alkyl substituent prefixes ---
    if features.substituents:
        alkyl_prefixes = _generate_alkyl_prefixes_for_polyfunctional(
            features, skip_acyloxy=_esters_demoted,
            exclude_branch_atoms=_amidine_excluded_n,
        )
        if alkyl_prefixes is None:
            # P-29.2 fail-closed (see the free-valence gate in that function):
            # a branch whose double/triple attachment has no correct prefix.
            return None
        all_prefixes.extend(alkyl_prefixes)

    # --- HYG-04 site#2: N-substituent prefixes for a PRINCIPAL amine ---
    # When the amine is the principal characteristic group (suffix -amine), its
    # N-substituents are cited as N-/N,N- prefixes (e.g. N,N-dimethylethan-1-amine),
    # mirroring composer._assemble_amine_name. Without this, a tertiary amine
    # (e.g. diphenhydramine core) leaks its N-alkyls into a "(dimethylamino)"
    # substituent that double-counts the principal nitrogen. The principal-amine
    # branch itself is excluded from the substituent walk (see
    # _generate_alkyl_prefixes_for_polyfunctional). These prefixes join the same
    # alphanumerical sort below, so they interleave correctly with C-substituents.
    _AMINE_PRINCIPAL_FGS = {
        'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine',
    }
    _principal_amine_n_atoms = set()
    if principal_group in _AMINE_PRINCIPAL_FGS:
        for _pmatch in getattr(features, 'principal_group_atoms', None) or []:
            for _pa in _pmatch:
                if mol.GetAtomWithIdx(_pa).GetSymbol() == 'N':
                    _principal_amine_n_atoms.add(_pa)
    # Scope: exactly ONE principal amine nitrogen. Di-/polyamines need primed
    # N-locants (N,N,N',N'-tetramethyl...) keyed to each amine position — a
    # distinct feature; applying single-N logic there would mis-assign locants
    # (e.g. "N,N-tetramethyl"). Multi-N principal amines fall back to the prior
    # path (the substituent walk names each "(dimethylamino)"), avoiding a drop.
    if len(_principal_amine_n_atoms) == 1:
        from collections import Counter as _NCounter

        from ..assembly.composer import _name_r_group, _wrap_n_substituent

        _amine_n_atom = next(iter(_principal_amine_n_atoms))
        _chain_set_amine = set(principal_chain)
        _n_sub_names: List[str] = []
        for _pnb in mol.GetAtomWithIdx(_amine_n_atom).GetNeighbors():
            _pni = _pnb.GetIdx()
            # Skip the parent-chain attachment carbon and hydrogens;
            # the remaining heavy neighbours are the N-substituents.
            if _pni in _chain_set_amine or _pnb.GetAtomicNum() <= 1:
                continue
            _rn = _name_r_group(mol, _pni, exclude_atoms={_amine_n_atom})
            if not _rn:
                # Fail closed. This block is the ONLY producer of the principal
                # amine's N-substituent prefixes -- the branch is deliberately
                # excluded from the substituent walk a few lines above -- so
                # skipping an un-nameable one does not degrade the name, it
                # DELETES atoms from it. Measured on
                # 'CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)'
                # 'OC(=O)CCCCCCCCCCCCCCCCC': the phosphate-bearing arm returned
                # None here and the emitted name was
                # '1-((11Z,14Z)-icosa-11,14-dienoyloxy)-N-methyl-2-'
                # '(stearoyloxy)propanamine' -- a molecule with no phosphate at
                # all. Before T2 (56833b46) the same arm was FABRICATED as
                # 'N-tritetracontyl', a C43 chain; removing the fabrication made
                # the refusal honest and exposed this caller (CLAUDE.md #9).
                return None
            _n_sub_names.append(_rn)
        if _n_sub_names:
            _n_counts = _NCounter(_n_sub_names)
            for _rn in sorted(_n_counts.keys()):
                _rc = _n_counts[_rn]
                if _rc == 1:
                    all_prefixes.append(f"N-{_wrap_n_substituent(_rn)}")
                else:
                    _rmult = get_multiplier_prefix(_rc, _rn)
                    all_prefixes.append(
                        f"N,N-{_rmult}{_wrap_n_substituent(_rn)}"
                    )

    # ---: N-substituent prefix for a PRINCIPAL AMIDE
    # whose acyl carbon carries stereo the earlier delegation cannot render ---
    # The dedicated amide delegate a few hundred lines above this
    # (`rules.amides.name_amide` / `composer._assemble_amide_name`) already
    # returns the correct name for a stereo-free on-chain-acyl secondary/
    # tertiary amide; it fails CLOSED (declines, not wrong) whenever the
    # molecule carries a defined stereocentre or stereo double bond, because
    # it cannot render the parent chain's own descriptors (the T3d
    # oxime-decline precedent cited there). When that guard declines and
    # control reaches this generic chain-building section, the amide's own
    # N-branch has ALREADY been excluded from the substituent walk
    # (`_generate_alkyl_prefixes_for_polyfunctional`'s mirror of the amine
    # exclusion above) — so it must be rendered HERE or the atoms are lost.
    # Before this block the exclusion did not exist and the amide N (bonded
    # directly to the chain's acyl carbon, chain position 1) was walked as an
    # ordinary heteroatom substituent and mis-named a plain "(R)amino" PREFIX
    # ON C1: 'CCCCCCCC/C=C\\C/C=C\\C/C=C\\CCCC(=O)NCCO' (anandamide's
    # 5,8,11-trienoyl homologue) shipped '1-[(2-hydroxyethyl)amino]icosa-
    # 5,8,11-trienamide' — a name that double-expresses the amide carbon
    # (chain C1 as BOTH the amide carbonyl AND an amino-bearing substituent
    # carbon) and is OPSIN-unparseable. Mirrors the principal-amine N,N-
    # prefix block above, same `_name_r_group`/`_wrap_n_substituent`
    # primitives; fails closed (returns None, never emits) on an un-nameable
    # R or a shape with more than one amide match, so a previously-abstaining
    # molecule can only become a NAMED, atom-complete, RT-correct name — never
    # a new atom drop. The earlier successful delegation always `return`s
    # before this point, so this block never double-fires for the cases it
    # already names correctly (verified: 'CCCCC(=O)NCCO' stays
    # 'N-(2-hydroxyethyl)pentanamide', unaffected).
    _AMIDE_PRINCIPAL_FGS = {'secondary_amide', 'tertiary_amide'}
    if principal_group in _AMIDE_PRINCIPAL_FGS:
        _amide_matches_ns = features.functional_groups.get(
            principal_group, []) or []
        if len(_amide_matches_ns) == 1:
            _amide_match_ns = _amide_matches_ns[0]
            _amide_n_atom_ns = next(
                (a for a in _amide_match_ns
                 if mol.GetAtomWithIdx(a).GetSymbol() == 'N'), None)
            _amide_acyl_c_ns = next(
                (a for a in _amide_match_ns
                 if mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                 and any(
                     nb.GetSymbol() == 'O'
                     and mol.GetBondBetweenAtoms(a, nb.GetIdx())
                            .GetBondTypeAsDouble() == 2.0
                     for nb in mol.GetAtomWithIdx(a).GetNeighbors())),
                None,
            )
            if (_amide_n_atom_ns is not None
                    and _amide_acyl_c_ns is not None
                    and _amide_acyl_c_ns in set(principal_chain)):
                from collections import Counter as _NCounterAmide

                from ..assembly.composer import _name_r_group, _wrap_n_substituent
                from ..assembly.naming_utils import enclose_if_compound

                _an_sub_names: List[str] = []
                for _anb in mol.GetAtomWithIdx(_amide_n_atom_ns).GetNeighbors():
                    _ani = _anb.GetIdx()
                    # Skip the acyl carbon itself and hydrogens; the remaining
                    # heavy neighbours are the amide's N-substituent(s).
                    if _ani == _amide_acyl_c_ns or _anb.GetAtomicNum() <= 1:
                        continue
                    _arn = _name_r_group(
                        mol, _ani, exclude_atoms={_amide_n_atom_ns})
                    if not _arn:
                        # Fail closed -- see docstring above: this is now the
                        # ONLY producer of this amide's N-substituent prefix
                        # (the branch was excluded from the walk), so skipping
                        # an un-nameable one would DELETE atoms, not degrade.
                        return None
                    # P-16.5.1.1: a COMPOUND R group (e.g. "2-hydroxyethyl")
                    # takes its own enclosing marks before the "N-" locant is
                    # prefixed -- `_name_r_group` returns the BARE compound
                    # token (verified: "2-hydroxyethyl", no parens), and
                    # `_wrap_n_substituent` deliberately only ESCALATES an
                    # enclosure that is already there (its own docstring: "do
                    # not need brackets" for a name with none). Without this,
                    # the emitted prefix read "N-2-hydroxyethyl..." instead of
                    # "N-(2-hydroxyethyl)...".
                    _arn = enclose_if_compound(_arn)
                    _an_sub_names.append(_arn)
                if _an_sub_names:
                    _an_counts = _NCounterAmide(_an_sub_names)
                    for _arn in sorted(_an_counts.keys()):
                        _arc = _an_counts[_arn]
                        if _arc == 1:
                            all_prefixes.append(
                                f"N-{_wrap_n_substituent(_arn)}")
                        else:
                            _armult = get_multiplier_prefix(_arc, _arn)
                            all_prefixes.append(
                                f"N,N-{_armult}{_wrap_n_substituent(_arn)}"
                            )

    # --- Merge duplicate bare prefix names ---
    # When two FG detection paths (e.g., primary_alcohol and secondary_alcohol)
    # both produce the same bare prefix form ("hydroxy"), merge into one entry
    # with the correct multiplier ("dihydroxy"). Only bare (no-locant) prefixes
    # are merged; locanted prefixes represent distinct chain positions.
    all_prefixes = _merge_bare_duplicate_prefixes(all_prefixes)

    # Sort all prefixes alphabetically (P-14.5.2), with P-14.5.4 lowest-locant
    # tie-break for identical-letter prefixes.
    # RENDERED prefix strings (locants already attached) -> parent_locants=True.
    all_prefixes.sort(key=lambda p: prefix_citation_sort_key(
        p, parent_locants=True))

    # Cluster D (P-41 / P-63.6): a polyfunctional parent with NO principal
    # characteristic group (every group is prefix-only, e.g. sulfoxide +
    # thioether -> '1-(methanesulfinyl)-2-(methylsulfanyl)ethane') takes a
    # SUBSTITUTIVE name with no suffix. All prefixes are already built by the FG
    # loop above via the correct producers (get_sulfinyl_prefix ->
    # 'methanesulfinyl', get_sulfanyl_prefix -> 'methylsulfanyl') and carry their
    # chain locants. First cut: SATURATED acyclic (chain) parents only; fail
    # closed (return None -> deferred handlers / SELF-01) on unsaturation, rings,
    # or any unresolved prefix so no wrong name can ship.
    if principal_group is None:
        from ..data.chain_names import get_chain_prefix as _gcp
        if (features.double_bonds or features.triple_bonds
                or not all_prefixes
                or (getattr(features, 'is_cyclic', False)
                    and not getattr(features, 'chain_is_parent', False))):
            return None
        _stem = _gcp(len(principal_chain))
        if not _stem:
            return None
        _parent = f"{_stem}ane"
        _pfx = _join_prefixes(all_prefixes)
        if _pfx and _pfx[-1].isalpha() and _parent[0].isdigit():
            _name = f"{_pfx}-{_parent}"
        else:
            _name = f"{_pfx}{_parent}"
        if features.stereocenters or getattr(features, 'double_bond_stereo', None):
            from .stereochemistry import (
                collect_stereodescriptors,
                format_stereodescriptor_string,
            )
            _desc = collect_stereodescriptors(mol, atom_to_locant)
            if _desc:
                _name = f"{format_stereodescriptor_string(_desc)}{_name}"
        return _name

    # --- Build the name ---
    from ..data.chain_names import get_chain_prefix

    chain_length = len(principal_chain)
    stem = get_chain_prefix(chain_length)

    # Get suffix for principal group
    from ..rules.locants import get_functional_group_locants
    from .seniority import get_suffix

    suffix = get_suffix(principal_group, is_ring=False)
    if not suffix:
        return None

    # Get locants for principal group.
    # count ONLY the principal-group instances that actually sit on the
    # PARENT CHAIN. An instance wholly off the chain (all its atoms outside
    # principal_chain) belongs to a substituent and is spelled by that
    # substituent's own name -- counting it in the suffix multiplicity
    # double-counts it (N-(carboxymethyl)aspartic acid
    # `C([C@@H](C(=O)O)NCC(=O)O)C(=O)O` was named `...butanetrioic acid`, three
    # -oic on a two-acid butane parent whose third COOH is the carboxymethyl
    # substituent, then suppressed as OPSIN-unparseable). Defensive: fall back to
    # the full list if the filter would leave nothing on the chain.
    _chain_set = set(principal_chain)
    _pga_all = features.principal_group_atoms or []
    _pga_on_chain = [m for m in _pga_all if any(a in _chain_set for a in m)]
    _pga_for_suffix = _pga_on_chain if _pga_on_chain else _pga_all
    suffix_locants = []
    if _pga_for_suffix:
        suffix_locants = get_functional_group_locants(
            principal_chain,
            _pga_for_suffix,
            atom_to_locant,
            mol=mol
        )
        # Deduplicate locants (overlapping SMARTS can produce duplicates)
        suffix_locants = sorted(set(suffix_locants))

    # For terminal groups (acid, aldehyde), locant is implicit; and on a
    # mononuclear (single-carbon) parent the suffix locant is trivially '1'
    # (P-14.3.4.2(a)) — pass chain_length so Rule 1 fires ('methanamine', not
    # 'methan-1-amine').
    from ..assembly.naming_utils import should_omit_locant_one
    if should_omit_locant_one(context="suffix", fg_type=principal_group,
                              chain_length=chain_length):
        suffix_locants = []

    # Determine multiplier for multiple principal groups
    # Validate: suffix count cannot exceed parent chain/ring capacity
    count = len(_pga_for_suffix)  # on-chain instances only (see above)
    max_capacity = chain_length
    if count > max_capacity:
        count = max_capacity
    # If we have unique locants, use those as the count (more reliable)
    if suffix_locants:
        count = len(suffix_locants)
    multiplier = get_suffix_multiplier_prefix(count, suffix) if count > 1 else ""

    # Build unsaturation infix
    from ..assembly.composer import _build_unsaturation_infix
    from ..rules.locants import get_bond_locants

    double_locants = get_bond_locants(
        principal_chain, features.double_bonds, atom_to_locant
    )
    triple_locants = get_bond_locants(
        principal_chain, features.triple_bonds, atom_to_locant
    )
    unsaturation = _build_unsaturation_infix(double_locants, triple_locants)

    # Detect suffix-prefix locant collisions (safety net for ring parents)
    if suffix_locants and all_prefixes:
        import re as _re

        from .locant_validation import detect_locant_collisions

        # Extract prefix locants from formatted prefix strings
        prefix_locant_groups = []
        for ptext in all_prefixes:
            match = _re.match(r'^([\d,]+)-', ptext)
            if match:
                try:
                    locs = [int(x) for x in match.group(1).split(',')]
                    prefix_locant_groups.append(locs)
                except ValueError:
                    pass

        is_ring = getattr(features, 'is_cyclic', False) and not getattr(features, 'chain_is_parent', False)
        if prefix_locant_groups:
            collisions = detect_locant_collisions(
                suffix_locants,
                prefix_locant_groups,
                parent_type="ring" if is_ring else "chain",
            )
            if collisions:
                collision_set = set(loc for _, loc in collisions)
                # Remove colliding prefix locants from affected prefix strings
                cleaned = []
                for ptext in all_prefixes:
                    match = _re.match(r'^([\d,]+)-(.+)$', ptext)
                    if match:
                        locs = [int(x) for x in match.group(1).split(',') if int(x) not in collision_set]
                        name_part = match.group(2)
                        if locs:
                            cleaned.append(f"{','.join(str(l) for l in locs)}-{name_part}")
                        else:
                            cleaned.append(name_part)
                    else:
                        cleaned.append(ptext)
                all_prefixes = cleaned

    # P-65.1.1.1 + P-14.3.4.6: a SUBSTITUTED 2-carbon monocarboxylic acid whose
    # heteroatom-FG substituent (cyano / sulfanyl / hydroxy / aryloxy) routes
    # through this polyfunctional path uses the retained parent 'acetic acid'
    # with the alpha-carbon locants OMITTED — 'cyanoacetic acid' (BB 30999),
    # 'sulfanylacetic acid' (BB 4967), '(4-chlorophenoxy)acetic acid'. Same rule
    # as the general_acyclic path (_handler_shared). Guarded to a saturated
    # 2-carbon CHAIN monoacid with substituents and no stereocentre (a C2 stereo
    # locant cannot be omitted safely -> keep the systematic 'ethanoic' form).
    _acetic_context = (
        principal_group == 'carboxylic_acid'
        and chain_length == 2 and count == 1
        and not double_locants and not triple_locants
        and all_prefixes
        and not (getattr(features, 'is_cyclic', False)
                 and not getattr(features, 'chain_is_parent', False))
        and not features.stereocenters
        and not getattr(features, 'double_bond_stereo', None)
    )
    # P-14.3.4.3 APPLICATION. Everything the scope needs is final here:
    # `all_prefixes` is complete, and `suffix_locants` is known (P-14.3.3 restores
    # every locant in a scope where the SUFFIX still cites one, so a suffix locant
    # vetoes -- the multiplied but locant-LESS `-dioic acid` does not, which is
    # exactly the `chloropropanedioic acid` shape). The prefix is RE-RENDERED from
    # the parts the FG loop used, so the locant-free spelling comes from
    # `format_fg_prefix` like every other prefix in this handler and not from
    # editing a rendered string.
    #
    # ⚠ Both conditions are MUTATION-SURVIVING and DELIBERATELY KEPT (measured
    # 2026-07-30: removing either changes no name), for two DIFFERENT reasons:
    #
    # * `not suffix_locants` is over-determined today by the orbit test inside the
    #   licence -- a parent compound with only ONE kind of substitutable hydrogen and
    #   a suffix that still cites a locant does not occur, because the one-orbit chain
    #   parents are the diacids, whose terminal suffix locants P-14.3.4.1 has already
    #   withdrawn. `OC(=O)CC(O)C` (suffix locant 3, one prefix) is refused by the
    #   orbit test on butanoic acid instead. It is the direct statement of P-14.3.3
    #   for the suffix half of the scope, and what would stop the licence the moment
    #   either the orbit test or P-14.3.4.1 changed.
    #
    # * `len(all_prefixes) == 1` covers a case the structural monosubstitution proof
    #   canNOT: ONE substituent component matched by TWO different FG types would
    #   append two prefixes while `_one_substituent_removed` still measures one
    #   component. No witness for that shape was found, so it survives mutation, but
    #   it is not redundant with the structural proof -- do not remove it on the
    #   strength of the mutation result.
    if (_l3_render_from is not None and len(all_prefixes) == 1
            and not suffix_locants
            and not double_locants and not triple_locants):
        all_prefixes = [format_fg_prefix(_l3_render_from[0], [], _l3_render_from[1])]

    if _acetic_context:
        from ..assembly.composition_primitives import retained_acetic_from_prefixes
        name = retained_acetic_from_prefixes(all_prefixes)
    else:
        # Assemble the name
        name = format_suffix_with_locants(
            stem, unsaturation, suffix, suffix_locants, multiplier
        )

        # Add prefixes with proper hyphenation at boundary
        if all_prefixes:
            prefix_str = _join_prefixes(all_prefixes)
            # Ensure hyphen between prefix ending with letter and name starting with digit
            if prefix_str and name and prefix_str[-1].isalpha() and name[0].isdigit():
                name = f"{prefix_str}-{name}"
            else:
                name = f"{prefix_str}{name}"

    # Add stereodescriptors if present
    if features.stereocenters or getattr(features, 'double_bond_stereo', None):
        from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
        descriptors = collect_stereodescriptors(mol, atom_to_locant)
        if descriptors:
            stereo_prefix = format_stereodescriptor_string(descriptors)
            name = f"{stereo_prefix}{name}"

    # ASSEMBLY_AUDIT: detect FGs present in molecule but missing from final name.
    # Guarded by logger level check so there is no performance impact in production.
    if logger.isEnabledFor(logging.DEBUG):
        detected_fgs = set()
        for fg_name_audit, fg_matches in non_principal.items():
            if fg_name_audit in ('alkene', 'alkyne'):
                continue
            if fg_matches:
                detected_fgs.add(fg_name_audit)
        missing_fgs = set()
        for fg_audit in detected_fgs:
            prefix = PREFIX_FORMS.get(fg_audit)
            if prefix is None:
                continue  # functional-class-only, no prefix form expected
            if prefix and prefix in name:
                continue
            missing_fgs.add(fg_audit)
        if missing_fgs:
            logger.debug(
                "ASSEMBLY_AUDIT: missing_fg=%s in name=%s smiles=%s",
                missing_fgs, name, getattr(features, 'canonical_smiles', '?'),
            )

    # M2 Task 3 fail-closed splice guard: a prefix built from a substituent
    # sub-call can carry the cascade's refusal sentinel (bare 'substituent',
    # 'unknown ...', '(not supported)') into `all_prefixes` and from there into
    # this assembled `name`. Never ship that -- void the candidate so the
    # caller abstains or falls through to another producer, instead of
    # emitting a name with the placeholder woven in.
    if is_refusal_sentinel(name):
        return None

    return name


# P-63.3.1 (BB 27858): the divalent-chalcogen BRIDGE functional groups, whose
# SMARTS span a carbon on each side of the two chalcogens ('peroxide'
# [#6][OX2][OX2][#6], 'disulfide' [#6][SX2][SX2][#6]). Their branch-side PIN is
# the monovalent method-(1) prefix, so the substituent walk owns them, not the
# FG-prefix loop.
_CHALCOGEN_BRIDGE_FGS = frozenset({'peroxide', 'disulfide'})


def _bridge_far_side_is_walked_branch(mol, match, chain_set, features) -> bool:
    """True when a P-63.3.1 bridge ``match`` (C, X, X, C) has exactly one carbon
    on the parent chain and its ENTIRE far side -- both chalcogens and the far
    carbon -- inside ONE walked substituent branch.

    That is exactly the condition under which the substituent walk sees the whole
    bridge and can name it with a single monovalent prefix. When it does not hold
    (both carbons on the chain, an in-chain bridge, a bridge split across
    branches) this returns False and the FG-prefix loop keeps the match, so no
    caller can mistake "the walk owns it" for "nobody owns it".
    """
    if not match or len(match) < 4 or not features.substituents:
        return False
    carbons = [a for a in match
               if mol.GetAtomWithIdx(a).GetSymbol() == 'C']
    chalcogens = [a for a in match
                  if mol.GetAtomWithIdx(a).GetSymbol() in ('O', 'S', 'Se', 'Te')]
    if len(carbons) != 2 or len(chalcogens) != 2:
        return False
    in_chain = [a for a in carbons if a in chain_set]
    if len(in_chain) != 1:
        return False  # in-chain or off-chain bridge -> not a branch prefix
    far_side = set(chalcogens) | (set(carbons) - set(in_chain))
    for _sub_list in features.substituents.values():
        for _sub_atoms in _sub_list:
            if far_side <= set(_sub_atoms):
                return True
    return False


def _is_linear_terminal_pure_c_branch(mol, sub_atoms, chain_set) -> bool:
    """True iff `sub_atoms` is an all-carbon acyclic branch attached at a terminus
    (attach has <=1 in-branch carbon neighbour, every branch carbon <=2), so
    get_alkyl_name(carbon_count) is the correct P-29.2 name. Otherwise the branch is
    branched / secondarily attached and must go to the Tier-4 enumerator."""
    sub_set = set(sub_atoms)
    attach = None
    for si in sub_set:
        for nb in mol.GetAtomWithIdx(si).GetNeighbors():
            if nb.GetIdx() in chain_set:
                attach = si
                break
        if attach is not None:
            break
    if attach is None:
        return False
    for i in sub_set:
        deg = sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                  if nb.GetIdx() in sub_set)
        limit = 1 if i == attach else 2
        if deg > limit:
            return False
    return True


def _enumerate_pure_c_branch_name(mol, sub_atoms, features, position):
    """Name a non-linear pure-C branch via the Tier-4 enumerator (verified:
    propan-2-yl / butan-2-yl). Mirrors the het-arm enumerator block; fail-closed
    (None) for ring branches or if the enumerator declines."""
    ring_info = mol.GetRingInfo()
    if any(ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms):
        return None
    from ..assembly.naming_utils import needs_brackets
    from ..assembly.substituent_enumerator import (
        SubstituentInfo,
        classify_and_name_fragment,
    )
    chain = features.principal_chain
    chain_set = set(chain) if chain else set()
    frag_info = SubstituentInfo(
        frag_mol=None,
        locant=position,
        attach_mol_idx=(chain[position - 1] if chain and position <= len(chain)
                        else sub_atoms[0]),
        frag_atoms=frozenset(sub_atoms),
    )
    name = classify_and_name_fragment(mol, frag_info, chain_set, features)
    if name and needs_brackets(name):
        name = f"({name})"
    return name


def _generate_alkyl_prefixes_for_polyfunctional(
    features: Any, skip_acyloxy: bool = False,
    exclude_branch_atoms: Optional[Set[int]] = None,
) -> Optional[List[str]]:
    """
    Generate alkyl substituent prefixes for polyfunctional compounds.

    Similar to _generate_alkyl_prefixes in composer.py but returns
    just the formatted strings for combination with FG prefixes.

    Args:
        features: MolecularFeatures object
        skip_acyloxy: If True, skip acyloxy detection (esters already handled
                      by ester demotion in name_polyfunctional)
        exclude_branch_atoms: Atom indices whose containing substituent branch
                      must be skipped (owned by a dedicated FG-prefix path, e.g.
                      AM-4 chain-terminal amidine N atoms named amino/imino).
    """
    from collections import defaultdict

    mol = features.mol
    substituent_groups: Dict[str, List[int]] = defaultdict(list)

    # Collect ring atoms that should be skipped (handled by ring substituent prefixes)
    ring_atoms_to_skip: set = set()
    if getattr(features, 'chain_is_parent', False):
        ring_groups = getattr(features, 'ring_substituents_as_groups', [])
        for ring_atoms in ring_groups:
            ring_atoms_to_skip.update(ring_atoms)

    # HYG-04 site#2: when a PRINCIPAL amine carries N-substituents (e.g. -N(CH3)2),
    # those atoms ARE the principal group; they are expressed as N,N- prefixes on
    # the amine suffix (in name_polyfunctional), NOT walked as a "(dimethylamino)"
    # substituent. Skipping the principal-amine's own branch here prevents naming
    # the same nitrogen twice (the "1-(dimethylamino)...-1-amine" double-count).
    _AMINE_PRINCIPALS = {
        'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine',
    }
    # The principal amine's NITROGEN atom(s). A substituent branch that contains
    # the principal nitrogen IS the principal group (its alkyls extend past the
    # FG-match atoms, so an "issubset" test misses multi-carbon N-substituents).
    _principal_amine_n: set = set()
    if getattr(features, 'principal_group', None) in _AMINE_PRINCIPALS:
        for _pm in getattr(features, 'principal_group_atoms', []) or []:
            for _pa in _pm:
                if mol.GetAtomWithIdx(_pa).GetSymbol() == 'N':
                    _principal_amine_n.add(_pa)
    # Scope to a single principal amine nitrogen (matches the N,N-prefix emission
    # in name_polyfunctional). Di-/polyamines fall back to the prior path.
    _exclude_principal_amine = len(_principal_amine_n) == 1

    # (E3 Task 5): mirror of the amine exclusion above for a
    # PRINCIPAL AMIDE (secondary_amide / tertiary_amide) whose acyl carbon sits
    # ON the principal chain but whose N-substituent(s) are named via the
    # "N-substituent prefix for a PRINCIPAL AMIDE" block later in
    # name_polyfunctional (reached only when the earlier stereo-free amide
    # delegation -- rules.amides.name_amide / composer._assemble_amide_name --
    # already declined, e.g. a stereo-bearing acyl it cannot render parent-
    # level descriptors for). Before this guard the amide N (bonded directly
    # to the chain's acyl carbon) was walked here like any other heteroatom
    # substituent and mis-named a plain "(R)amino" PREFIX ON C1 --
    # '...icosa-5,8,11-trien-1-yl [(2-hydroxyethyl)amino]...' shaped names that
    # double-express the amide carbon and fail OPSIN. Scoped to a SINGLE amide
    # match (di-amide chains fall back to the prior, unaffected, path).
    _AMIDE_PRINCIPALS = {'secondary_amide', 'tertiary_amide'}
    _principal_amide_n: set = set()
    if getattr(features, 'principal_group', None) in _AMIDE_PRINCIPALS:
        _amide_matches_excl = features.functional_groups.get(
            features.principal_group, []) or []
        if len(_amide_matches_excl) == 1:
            _n_idx_excl = next(
                (a for a in _amide_matches_excl[0]
                 if mol.GetAtomWithIdx(a).GetSymbol() == 'N'), None)
            if _n_idx_excl is not None:
                _principal_amide_n.add(_n_idx_excl)
    _exclude_principal_amide = len(_principal_amide_n) == 1

    _excl_branch = exclude_branch_atoms or set()
    for position, sub_list in features.substituents.items():
        for sub_atoms in sub_list:
            # HYG-04 site#2: don't name the principal amine's own branch as a
            # substituent — its N-substituents are emitted as N,N- prefixes.
            if _exclude_principal_amine and (_principal_amine_n & set(sub_atoms)):
                continue
            # (E3 Task 5): same exclusion for a principal amide's
            # own N-branch — see comment above.
            if _exclude_principal_amide and (_principal_amide_n & set(sub_atoms)):
                continue
            # AM-4: skip branches owned by the amidine amino/imino FG-prefix path.
            if _excl_branch & set(sub_atoms):
                continue
            # Skip substituents whose ring atoms are handled by
            # _generate_ring_substituent_prefixes (ring + its own substituents).
            # A substituent is a "ring substituent" if its attachment atom
            # (the atom bonded to the principal chain) is IN the ring.
            # Substituents that contain rings deeper in the branch (e.g.,
            # benzoylamino where N attaches to chain) are NOT skipped.
            if ring_atoms_to_skip:
                _chain_set_local = set(features.principal_chain) if features.principal_chain else set()
                _attach_in_ring = False
                for _si in sub_atoms:
                    if _si in ring_atoms_to_skip:
                        _sa = mol.GetAtomWithIdx(_si)
                        for _nb in _sa.GetNeighbors():
                            if _nb.GetIdx() in _chain_set_local:
                                _attach_in_ring = True
                                break
                    if _attach_in_ring:
                        break
                if _attach_in_ring:
                    continue

            # Skip substituent branches entirely covered by a single non-principal
            # FG match ONLY for FG types whose prefix form includes the carbon
            # (carbamoyl, carboxy, carbonochloridoyl, etc.).  These are the FG itself
            # (e.g., -C(=O)NH2 for amide) and are emitted as FG prefixes by the
            # FG prefix loop above.  Processing them here would produce incorrect
            # compound substituent names like "(aminomethyl)".
            # IUPAC P-66.1(c): non-principal amide = carbamoyl prefix.
            # NOTE: Only applies to specific terminal-C FG types. Other FGs
            # (amine, ketone, secondary_amide, etc.) must NOT trigger this guard.
            _POLY_GUARD_FG_TYPES = {
                'primary_amide', 'carboxylic_acid',
                'acid_chloride', 'acid_bromide', 'acid_fluoride',
                # follow-on (F3 ureido): a urea substituent
                # -NH-C(=O)-NH2 is emitted as the 'carbamoylamino' FG prefix by
                # the FG loop above (its carbon is in the prefix). Without this
                # guard the SAME branch is ALSO named by _check_for_acylamino
                # below, which counts the ureido C(=O) as a 1-carbon acyl ->
                # the spurious 'methanoylamino', doubling the prefix on one
                # carbon ('5-carbamoylamino-5-(methanoylamino)' = a different,
                # OPSIN-unparseable molecule). The urea match [NX3][CX3](=O)[NX3]
                # covers the whole branch, so the subset test skips it here.
                'urea',
                # Wave2 T2b (claimed-atom mask): FG prefixes that fully name
                # their branch — isocyanato/isothiocyanato (P-35.2.1),
                # isocyano (P-66.5.3), guanidino (P-66.4.1.2.2). Without the
                # skip the SAME branch is re-walked by the generic namers,
                # which mis-read the heterocumulene/guanidine atoms and emit
                # a phantom co-substituent on the same locant
                # ('8-formamido-8-isocyanatooctanoic acid',
                #  '8-isothiocyanato-8-(methylamino)octanoic acid',
                #  '4-guanidino-4-(methylamino)butanoic acid') — a different,
                # often unparseable molecule. Subset test: each SMARTS spans
                # the whole branch (isocyanate/isothiocyanate/isocyanide
                # include the chain-anchor C; guanidine IS the branch).
                'isocyanate', 'isothiocyanate', 'isocyanide', 'guanidine',
                # Wave-2 completion: thiourea -NH-C(=S)-NH2 -> carbamothioylamino
                # FG prefix (whole branch), same double-count guard as urea.
                'thiourea',
            }
            _sub_set = set(sub_atoms)
            _skip_fg_branch = False
            _non_principal = getattr(features, 'non_principal_groups', {})
            for _fg_nm, _fg_ms in _non_principal.items():
                if _fg_nm not in _POLY_GUARD_FG_TYPES:
                    continue
                if not _fg_ms:
                    continue
                for _fg_m in _fg_ms:
                    _fg_set = set(_fg_m)
                    if not _fg_set:
                        continue
                    if _sub_set.issubset(_fg_set):
                        _skip_fg_branch = True
                        break
                    # W2F-P6 (P-66.1.6.1.1.3): when the urea/thiourea DISTAL N is
                    # substituted (-NH-CO-NHR), the R atoms lie OUTSIDE the
                    # 4-atom [NX3][CX3](=O)[NX3] match, so branch-subset-of-match
                    # fails. The whole branch is still named by the
                    # '(R-carbamoyl)amino' FG prefix emitted by the FG loop; also
                    # skip when the urea match is fully INSIDE this branch and a
                    # urea atom is directly chain-attached (the extra branch
                    # atoms are then necessarily the distal substituents).
                    # Without this the branch is re-walked by
                    # _check_for_acylamino -> a spurious 'methanoylamino'
                    # co-prefix on the same locant (a wrong, OPSIN-unparseable
                    # molecule).
                    if (_fg_nm in ('urea', 'thiourea')
                            and _fg_set.issubset(_sub_set)):
                        _cs_guard = (set(features.principal_chain)
                                     if features.principal_chain else set())
                        if any(_nbg.GetIdx() in _cs_guard
                               for _ua in _fg_set
                               for _nbg in mol.GetAtomWithIdx(_ua).GetNeighbors()):
                            _skip_fg_branch = True
                            break
                if _skip_fg_branch:
                    break
            if _skip_fg_branch:
                continue

            # Count only carbon atoms
            carbon_count = sum(
                1 for idx in sub_atoms
                if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
            )

            # W2F-P7 (P-68.3): a CARBON-FREE phosphanyl substituent (-PH2) on a
            # polyfunctional parent (e.g. the acid OC(=O)CCP) is dropped by the
            # carbon_count==0 guard below and is NOT recoverable via the FG-prefix
            # loop ('phosphanyl' is a substituent prefix, not a functional group)
            # -> the whole -PH2 vanished ('propanoic acid', SELF-01-suppressed to
            # unknown). Name it via the shared phosphorus namer (the SAME one wired
            # into name_substituent Tier 1.93); its all-H/organyl-only guard
            # EXCLUDES a phosphoryl/phosphonic P=O (named by the oxoacid subsystem).
            # Fail-closed (falls through to the carbon_count==0 drop) on decline.
            if carbon_count == 0:
                _p_attach = next(
                    (si for si in sub_atoms
                     if mol.GetAtomWithIdx(si).GetSymbol() == 'P'),
                    None,
                )
                if _p_attach is not None:
                    from ..rules.phosphorus import name_phosphanyl_substituent
                    _ph = name_phosphanyl_substituent(
                        mol, list(sub_atoms), _p_attach)
                    if _ph:
                        substituent_groups[_ph].append(position)
                        continue

            # Skip non-alkyl substituents
            if carbon_count == 0:
                continue

            # Check for heteroatoms
            has_heteroatom = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in sub_atoms
            )

            if has_heteroatom:
                from ..assembly.composer import (
                    _check_for_acylamino,
                    _check_for_acyloxy,
                    _name_heteroatom_substituent,
                )
                # NOTE: Do NOT check alkoxy here - ethers are already handled
                # by the FG prefix system in name_polyfunctional
                het_name = _check_for_acylamino(mol, sub_atoms, features.principal_chain)
                if not het_name and not skip_acyloxy:
                    het_name = _check_for_acyloxy(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    het_name = _name_heteroatom_substituent(mol, sub_atoms, features.principal_chain)
                if not het_name:
                    # Enumerator fallback for simple non-ring, non-ether branches.
                    # Ethers (O-attached) are handled by the FG prefix system.
                    ring_info = mol.GetRingInfo()
                    sub_has_ring = any(ring_info.NumAtomRings(idx) > 0 for idx in sub_atoms)
                    # Find attachment atom
                    _attach = None
                    _chain_set_tmp = set(features.principal_chain) if features.principal_chain else set()
                    for _si in sub_atoms:
                        for _nb in mol.GetAtomWithIdx(_si).GetNeighbors():
                            if _nb.GetIdx() in _chain_set_tmp:
                                _attach = _si
                                break
                        if _attach is not None:
                            break
                    _attach_sym = mol.GetAtomWithIdx(_attach).GetSymbol() if _attach is not None else ''
                    # Skip S-attached branches if S is part of a named FG
                    # (sulfoxide, sulfone, thioether) -- already named by FG prefix system
                    _S_FG_NAMES = {'sulfoxide', 'sulfone', 'thioether'}
                    _skip_s_branch = False
                    if _attach is not None and _attach_sym == 'S':
                        _all_fgs = features.functional_groups if hasattr(features, 'functional_groups') else {}
                        for _fg_n in _S_FG_NAMES:
                            for _fg_match in _all_fgs.get(_fg_n, []):
                                if _attach in _fg_match:
                                    _skip_s_branch = True
                                    break
                            if _skip_s_branch:
                                break
                    # OWNERSHIP, NOT ELEMENT. This veto used to read
                    # `_attach_sym != 'O'`, justified as "Ethers (O-attached) are
                    # handled by the FG prefix system". That holds only for a MONO
                    # ether oxygen: the 'ether' SMARTS [OX2]([#6])[#6] demands a
                    # CARBON on both sides, and get_alkoxy_prefix then names that
                    # whole branch ('tert-butoxy'). When the attachment O's
                    # in-branch neighbour is a second chalcogen the branch is a
                    # P-63.3.1 peroxide (or P-63.3.2 mixed bridge): no ether match
                    # exists, PREFIX_FORMS['peroxide'] is None, so the FG loop
                    # emits nothing -- and this veto made the substituent walk
                    # decline too. BOTH owners declined and the branch VANISHED:
                    # 'CC(C)(C)OOCCO' (9 heavy atoms) was named 'ethan-1-ol'
                    # (3 heavy atoms), and 'CC(C)OOCCO' -- a different molecule --
                    # got the same name. Veto only the oxygen the FG prefix system
                    # really owns; the dichalcogen bridge goes to the walk, whose
                    # shared cascade names the class whole ('tert-butylperoxy',
                    # '(propan-2-yl)peroxy', 'tert-butyldisulfanyl').
                    _o_owned_by_fg_prefix = False
                    if _attach is not None and _attach_sym == 'O':
                        from ..assembly.substituent_enumerator import (
                            is_dichalcogen_bridge_attach,
                        )
                        _o_owned_by_fg_prefix = not is_dichalcogen_bridge_attach(
                            mol, _attach, set(sub_atoms))
                    if (not sub_has_ring and not _o_owned_by_fg_prefix
                            and not _skip_s_branch):
                        from ..assembly.naming_utils import (
                            apply_enclosing_marks,
                            needs_brackets,
                        )
                        from ..assembly.substituent_enumerator import (
                            SubstituentInfo,
                            classify_and_name_fragment,
                        )
                        chain_set = set(features.principal_chain) if features.principal_chain else set()
                        frag_info = SubstituentInfo(
                            frag_mol=None,
                            locant=position,
                            attach_mol_idx=features.principal_chain[position - 1] if features.principal_chain and position <= len(features.principal_chain) else sub_atoms[0],
                            frag_atoms=frozenset(sub_atoms),
                        )
                        het_name = classify_and_name_fragment(mol, frag_info, chain_set, features)
                        if het_name and needs_brackets(het_name):
                            # P-16.5.4 nesting ORDER (BB 7444; escalation P-16.5.4.1.5, BB 7509) under the P-16.5.1.1 marks requirement (BB 7232): the marks go ()->[]->{} outward, so
                            # a prefix that ALREADY carries parentheses must be
                            # wrapped in SQUARE brackets. The literal f"({...})"
                            # here produced the doubled-paren
                            # '2-((propan-2-yl)peroxy)ethan-1-ol' where the PIN is
                            # '2-[(propan-2-yl)peroxy]ethan-1-ol' (BB 27876 spells
                            # the analogue '1-[(propan-2-yl)diselanyl]propane
                            # (PIN)'). apply_enclosing_marks is the shared primitive
                            # that picks the right level; depth=-1 = "one level
                            # outside whatever is already there".
                            het_name = apply_enclosing_marks(het_name, depth=-1)
                if het_name:
                    substituent_groups[het_name].append(position)
                    continue
                # An UNNAMED branch is either an ownership handoff or structure
                # loss, and the two must not be conflated.
                #
                # Handoff: the branch is named by the FG-prefix loop in
                # name_polyfunctional, which runs BEFORE this walk -- a mono ether
                # oxygen ('tert-butoxy') or an S in a sulfoxide/sulfone/thioether
                # match. The walk is *supposed* to decline; the atoms are already
                # spoken for, so skip as before. Failing closed here instead
                # abstained on every ether-bearing polyfunctional molecule and let
                # a weaker downstream handler name them, which is how
                # 'CC(C)(C)OCCO' turned from the BB-correct '2-tert-butoxyethan-
                # 1-ol' into '2-(tert-butyloxy)ethan-1-ol' (BB 27679: "not
                # tert-butyloxy").
                #
                # Loss: nobody named it. The old code `continue`d here too and went
                # on to name the REST of the molecule, so the atoms silently
                # vanished -- 'CC(C)(C)OOCCO' (9 heavy atoms) shipped as
                # 'ethan-1-ol' (3 heavy), and 'CC(C)OOCCO', a DIFFERENT molecule,
                # shipped the same name. A name describing fewer atoms than were
                # drawn is the worst failure this system has and is worse than no
                # name, so refuse the whole handler: the caller returns None and the
                # namer falls through to another handler or abstains.
                if _o_owned_by_fg_prefix or _skip_s_branch:
                    continue
                logger.debug(
                    "FAILCLOSED-POLY-SUB: unnameable heteroatom branch "
                    "atoms=%s attach=%s attach_sym=%s ring=%s smiles=%s",
                    sorted(sub_atoms), _attach, _attach_sym, sub_has_ring,
                    getattr(features, 'canonical_smiles', '?'),
                )
                return None

            # W2F-P3 (defect b-pure-C, P-29.2): get_alkyl_name(carbon_count) is
            # correct ONLY for a linear terminal branch. A branched / secondary pure-C
            # branch (propan-2-yl, butan-2-yl) would be mis-named as the straight chain
            # of its carbon COUNT ('propyl' for propan-2-yl = a different constitution).
            # Keep get_alkyl_name for linear-terminal (zero change); route non-linear to
            # the Tier-4 enumerator (verified: propan-2-yl). Fail-closed if it declines.
            chain_set = set(features.principal_chain) if features.principal_chain else set()

            # P-29.2 free-valence gate. _is_linear_terminal_pure_c_branch reads
            # the branch's SHAPE (degrees, terminality) but never the order of
            # the bond joining it to the chain, so an exocyclic '=CH2' passed
            # as "linear terminal" and became 'methyl': itaconamic acid
            # 'OC(=O)C(=C)CC(N)=O' was named 4-amino-2-methyl-4-oxobutanoic
            # acid, a different molecule. Shared primitive; a single bond
            # defers and leaves both branches below untouched.
            _fv_attach = next(
                (si for si in sub_atoms
                 if any(nb.GetIdx() in chain_set
                        for nb in mol.GetAtomWithIdx(si).GetNeighbors())),
                None)
            if _fv_attach is not None:
                from ..assembly.substituent_enumerator import carbon_free_valence_prefix
                _fv = carbon_free_valence_prefix(mol, sub_atoms, _fv_attach)
                if _fv.prefix is not None:
                    substituent_groups[_fv.prefix].append(position)
                    continue
                if _fv.must_fail_closed:
                    # Fail closed: this producer declines the whole molecule
                    # rather than cite a single-valence prefix for a double
                    # attachment or drop the branch.
                    return None

            if _is_linear_terminal_pure_c_branch(mol, sub_atoms, chain_set):
                try:
                    alkyl_name = get_alkyl_name(carbon_count)
                    substituent_groups[alkyl_name].append(position)
                except ValueError:
                    continue
            else:
                branch_name = _enumerate_pure_c_branch_name(mol, sub_atoms, features, position)
                if branch_name:
                    substituent_groups[branch_name].append(position)
                continue

    # Build formatted prefixes using format_substituent_prefix for proper
    # complex substituent handling (e.g., bis(acetyloxy) not bisacetyloxy)
    from ..assembly.naming_utils import format_substituent_prefix as _fmt_sub
    prefixes = []
    for name, locants in substituent_groups.items():
        count = len(locants)
        sorted_locs = sorted(locants)
        formatted = _fmt_sub(name, sorted_locs, count)
        prefixes.append(formatted)

    return prefixes


def _merge_bare_duplicate_prefixes(prefixes: List[str]) -> List[str]:
    """Merge duplicate prefix strings (bare or locanted) into single entries.

    Merges prefixes that share the same base name, combining locants and
    adjusting multipliers. Handles both bare ("hydroxy" + "hydroxy" ->
    "dihydroxy") and locanted ("4-hydroxy" + "5-hydroxy" -> "4,5-dihydroxy").

    Prefixes with parentheses or N- are never merged (compound substituents).

    Args:
        prefixes: Formatted prefix strings.

    Returns:
        De-duplicated prefix list.
    """
    import re as _re

    if len(prefixes) <= 1:
        return prefixes

    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    def _parse_prefix(ptext: str):
        """Parse prefix into (locants, base_name) or None if unparseable."""
        # Skip compound/N-substituted prefixes
        if ptext.startswith('(') or ptext.startswith('N-') or ptext.startswith('N,'):
            return None

        locants = []
        rest = ptext

        # Extract leading locants: "4,5-dihydroxy" -> locants=[4,5], rest="dihydroxy"
        m = _re.match(r'^([\d,]+)-(.+)$', ptext)
        if m:
            loc_str = m.group(1)
            rest = m.group(2)
            for loc in loc_str.split(','):
                try:
                    locants.append(int(loc))
                except ValueError:
                    return None  # string locants like '3a' -- skip merge

        # Strip multiplier prefix to get base name
        base = rest
        for _cnt, mult in sorted(SIMPLE_MULTIPLIERS.items(),
                                 key=lambda x: len(x[1]), reverse=True):
            if rest.startswith(mult):
                cand = rest[len(mult):]
                if cand and cand[0].islower():
                    base = cand
                    break

        return (locants, base)

    # Group prefixes by base name
    groups: Dict[str, List[Tuple[int, List[int]]]] = {}  # base -> [(index, locants)]
    ungrouped: List[int] = []  # indices of prefixes that can't be parsed/merged

    for i, ptext in enumerate(prefixes):
        parsed = _parse_prefix(ptext)
        if parsed is None:
            ungrouped.append(i)
            continue
        locants, base = parsed
        if base not in groups:
            groups[base] = []
        groups[base].append((i, locants))

    # Check if any group needs merging
    needs_merge = any(len(entries) > 1 for entries in groups.values())
    if not needs_merge:
        return prefixes

    # Build merged result
    merged_indices: set = set()
    extra: List[str] = []

    for base, entries in groups.items():
        if len(entries) <= 1:
            continue

        # Separate bare (no locants) from locanted entries
        bare_entries = [(idx, locs) for idx, locs in entries if not locs]
        locanted_entries = [(idx, locs) for idx, locs in entries if locs]

        # Only merge within the same category (all-bare or all-locanted).
        # Mixed groups (bare + locanted) are NOT merged because bare
        # prefixes have no position info to combine with locants.
        merge_sets = []
        if len(bare_entries) > 1:
            merge_sets.append(bare_entries)
        if len(locanted_entries) > 1:
            merge_sets.append(locanted_entries)

        for merge_group in merge_sets:
            all_locants: List[int] = []
            for idx, locs in merge_group:
                all_locants.extend(locs)
                merged_indices.add(idx)
            all_locants = sorted(set(all_locants))
            total_count = len(all_locants) if all_locants else len(merge_group)

            # Rebuild using format_fg_prefix (peer function in this module)
            rebuilt = format_fg_prefix(base, all_locants, total_count)
            extra.append(rebuilt)

    result = [prefixes[i] for i in range(len(prefixes)) if i not in merged_indices]
    result.extend(extra)
    return result


# SUB-05/ (4b): bare detachable oxo-acid prefixes that, when glued
# letter->letter to a preceding substituent, must be hyphen-separated. EXACT
# membership only (so 'phosphonooxy'/'phenylsulfanyl' fused tokens are never
# split). Conservative oxo-acid set; deliberately excludes the common
# nitro/nitroso to avoid false splits.
_DETACHABLE_LETTER_PREFIXES = frozenset({
    "phosphono", "sulfo", "sulfino", "arsono", "borono",
})


def _join_prefixes(prefix_texts: List[str]) -> str:
    """
    Join multiple prefix strings with proper IUPAC hyphenation.

    When concatenating prefixes like "3-ethyl" and "2-hydroxy", the result
    should be "3-ethyl-2-hydroxy" (with hyphen between letter and digit).
    """
    if not prefix_texts:
        return ""

    if len(prefix_texts) == 1:
        return prefix_texts[0]

    result = prefix_texts[0]
    for i in range(1, len(prefix_texts)):
        current = prefix_texts[i]
        if result and current:
            last_char = result[-1]
            first_char = current[0]
            # An italic-N locant prefix ("N-..." / "N,N-...") is a locant-bearing
            # term like a numeric locant — it must be hyphen-separated from a
            # preceding substituent (e.g. "2-methoxy" + "N,N-dimethyl" ->
            # "2-methoxy-N,N-dimethyl"). HYG-04 site#2.
            next_is_n_locant = (
                first_char == 'N' and len(current) > 1 and current[1] in (',', '-')
            )
            # SUB-05/ (4b): a bare detachable oxo-acid prefix (phosphono,
            # sulfo, ...) glued letter->letter to a preceding substituent must
            # be separated ("3-oxo"+"phosphono" -> "3-oxo-phosphono", NOT
            # "3-oxophosphono"). GATED to an exact detachable-prefix token set
            # so legitimately-fused single tokens are NEVER split:
            # 'phosphonooxy'/'phenylsulfanyl' (.split('-')[-1] != a gate member)
            # and '2,3-dimethyl' (digit-initial) stay intact.
            current_tail = current.split('-')[-1]
            is_detachable = (current in _DETACHABLE_LETTER_PREFIXES
                             or current_tail in _DETACHABLE_LETTER_PREFIXES)
            # `)`, `]` and `}` are ONE rule at three depths.
            # `### **P-16.5.4** Multiple types of enclosing marks`
            # (BlueBookV2.md:7444) makes them the same device taken in the nesting
            # order `{[({[( )]})]}` (:7446), so a term ending in ANY of them
            # separates from a following locant identically.
            # Testing only `)` dropped the separator after an escalated prefix:
            # `_join_prefixes(['2-[(methylcarbamoyl)amino]', '4-methyl'])` gave
            # `2-[(methylcarbamoyl)amino]4-methyl`, reachable on the escalated
            # W2F-P6 prefix documented at :407 and independent of any multiplier.
            # (The `)`-then-LETTER rule below is a separate, pre-existing choice
            # and is deliberately left as it is.)
            closes_enclosure = last_char in ')]}'
            # Insert hyphen between letter/enclosing-mark and digit
            # e.g., "amino" + "4-methyl" → "amino-4-methyl"
            # e.g., "(ethanoyl)amino" + "4-methyl" → "(ethanoyl)amino-4-methyl"
            if first_char.isdigit() and (last_char.isalpha() or closes_enclosure):
                result += "-"
            # Also between ')' and letter for clarity
            elif last_char == ')' and first_char.isalpha():
                result += "-"
            # Letter/enclosing-mark followed by an italic-N locant prefix
            elif next_is_n_locant and (last_char.isalpha() or closes_enclosure):
                result += "-"
            # Letter followed by a bare detachable oxo-acid prefix (4b)
            elif is_detachable and last_char.isalpha() and first_char.isalpha():
                result += "-"
        result += current

    return result
