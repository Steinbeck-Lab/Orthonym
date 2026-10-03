"""-05 (a phase) nitrite-ester handler — functional-class naming.

A nitrite ester R-O-N=O is named in the two-word functional-class form
``<R> nitrite`` /; e.g. ``ethyl nitrite``), NOT as a C-nitroso
compound. This handler is the genuinely-missing emitter co-shipped with the
 ``nitroso`` ``[#6]`` guard: without it, guarding ``nitroso`` would leave
``CCON=O`` nameless (there is no nitrite/nitrous seniority entry; the
``nitric acid -> nitrate`` mapping was deferred — resolvers.py:299).

Modeled on handlers/imidate.py (two-word functional-class, predicate-pure +
pool.add + stereo injection) and registered in inner_dispatch at the specialty
tier (priority 2960, after chalcogen_ester).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


#:, Table 4.1 (the Blue Book): esters (class 9) rank above acid halides
#: (10), amides (11), hydrazides, imides, nitriles, aldehydes, ketones, alcohols and
#: phenols, hydroperoxides, amines and imines. The ``SENIORITY_ORDER`` entries from
#: the amides on are the groups an ester of nitric or nitrous acid outranks (the
#: acid halides are left out: their prefix forms are not built here).
_FIRST_CLASS_JUNIOR_TO_ESTERS = "primary_amide"


def principal_group_is_junior_to_esters(pg: Optional[str]) -> bool:
    """True when no principal group, or one of a class junior to the esters:
    an ester of a mononuclear noncarbon oxoacid then names the molecule in the
    functional-class form, '2-hydroxyethyl nitrate', '3-oxobutyl nitrate'
    , the Blue Book; the book's own junior-group row is
    '3-oxobutyl bromate (PIN)',:35974). Another ester, an acid, an anhydride or a
    group this table does not rank -> False (fail closed)."""
    if pg is None:
        return True
    from ...rules.seniority import SENIORITY_ORDER
    if "ester" in pg or pg not in SENIORITY_ORDER:
        return False
    return SENIORITY_ORDER.index(pg) >= SENIORITY_ORDER.index(_FIRST_CLASS_JUNIOR_TO_ESTERS)


def _is_nitrite_ester(features: Any) -> bool:
    """Predicate (pure): the nitrite ester R-O-N=O is the senior/sole
    characteristic group. Defers (False) when a higher-seniority PG is present,
    so the handler never claims a polyfunctional molecule where nitrite loses.
    NO mol/features mutation; NO module state."""
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('nitrite'):
        return False
    pg = getattr(features, 'principal_group', None)
    return pg == 'nitrite' or principal_group_is_junior_to_esters(pg)


def name_nitrite_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Emit ``<R> nitrite`` for a nitrite ester R-O-N=O.

    Algorithm: from the ``nitrite`` SMARTS match (R_carbon, O, N, O), name the
    R (alkyl/aryl) fragment via the universal substituent pipeline and join as
    ``"<R> nitrite"``. Returns None (defer) if the R cannot be named.
    Style is ignored (single PIN per compound, internal notes).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ..substituent_enumerator import name_substituent

    matches = (getattr(features, 'functional_groups', None) or {}).get('nitrite', [])
    if not matches:
        return None
    if getattr(features, 'principal_group', None) not in (None, 'nitrite'):
        # A junior principal group goes into the alcohol component's prefixes
        # ('2-hydroxyethyl nitrite'). Inside another name the fragment becomes a
        # prefix through parent_to_prefix, which reads a substitutive parent name,
        # so there the substitutive name is kept (as for nitrate_ester).
        from ..fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            return None
    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None

    # nitrite SMARTS = [#6][OX2][NX2]=[OX1] -> (R_carbon, ester_O, N, =O)
    if len(matches) >= 2:
        # Identical nitrite anions on one polyol: functional class multiplicative
        # ('ethane-1,2-diyl dinitrite'), not '2-(nitrosooxy)ethyl nitrite':
        # (the Blue Book), "When anions are identical
        # functional class multiplicative nomenclature is used"; (:35918).
        from ...rules.esters import name_polyol_identical_inorganic_ester
        poly = name_polyol_identical_inorganic_ester(
            mol, [m[0] for m in matches], {m[1] for m in matches},
            {a for m in matches for a in m[1:4]}, "nitrite")
        if poly:
            return _emit_name(features, poly)
    match = matches[0]
    if len(match) < 4:
        return None
    r_carbon, ester_o = match[0], match[1]

    # An acyl, thioacyl or imidoyl carbon on the oxygen makes a mixed anhydride of
    # nitrous acid, not an ester (as for nitric acid in handlers/nitrate_ester.py):
    # (the Blue Book), "Mixed anhydrides with carbonic acid, cyanic
    # acid, and inorganic acids are named as anhydrides" ('benzoic phosphinous
    # anhydride (PIN)',:32280). 'acetyl nitrite' is 'acetic nitrous anhydride'.
    for bond in mol.GetAtomWithIdx(r_carbon).GetBonds():
        other = bond.GetOtherAtom(mol.GetAtomWithIdx(r_carbon))
        if (bond.GetBondTypeAsDouble() == 2.0
                and other.GetSymbol() in ("O", "S", "Se", "Te", "N")):
            return None

    # R fragment = subgraph anchored at R_carbon, NOT crossing the ester oxygen.
    r_atoms = _collect_subgraph(mol, r_carbon, exclude={ester_o})
    if not r_atoms:
        return None
    try:
        r_word = name_substituent(mol, set(r_atoms), r_carbon)
    except Exception:
        r_word = None
    from ...errors import is_refusal_sentinel
    if not r_word or is_refusal_sentinel(r_word):
        return None             # 'substituent nitrate' is no name: decline

    name = f"{r_word} nitrite"
    # A second, different ester on the same alcohol component makes this method (2)
    # of ('2-(nitrooxy)ethyl nitrite'): correct, not the PIN.
    from ...rules.esters import record_polyol_mixed_anion_non_pin
    record_polyol_mixed_anion_non_pin(mol, r_carbon, ester_o, name)
    pool = get_current_pool()
    cand = pool.add(name, "nitrite_ester", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrite_ester",
            iupac_section_cite="P-67", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


def _emit_name(features: Any, name: str) -> Optional[NamingResult]:
    """Pool the name and return it as the handler result (stereo injected)."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    cand = get_current_pool().add(name, "nitrite_ester", features)
    if cand is None:
        return None
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrite_ester",
            iupac_section_cite="P-67", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


def _collect_subgraph(mol: Any, anchor_idx: int, exclude: "set[int]") -> "tuple[int, ...]":
    """BFS subgraph from anchor, excluding given atom indices (pure, read-only)."""
    visited: "set[int]" = set()
    stack = [anchor_idx]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in exclude:
            continue
        visited.add(idx)
        for bond in mol.GetAtomWithIdx(idx).GetBonds():
            other_idx = bond.GetOtherAtomIdx(idx)
            if other_idx not in visited and other_idx not in exclude:
                stack.append(other_idx)
    return tuple(sorted(visited))


__all__ = ["name_nitrite_ester", "_is_nitrite_ester"]
