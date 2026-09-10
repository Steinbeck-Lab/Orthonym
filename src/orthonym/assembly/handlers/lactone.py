"""a phase lactone handler — direct-return shim with coverage gate.

Verbatim lift of composer.py:826-848 (inline branch) wrapping
``rules.lactones.is_monocyclic_lactone`` + ``rules.lactones.name_monocyclic_lactone``.
Per internal notes, the rule bodies stay in rules.lactones unchanged.

The handler preserves the inline branch's coverage guard
(``ring_size > 8 or total_heavy <= ring_size + 8``) — without this guard,
substituted lactones in larger molecules would produce incomplete names.

IUPAC cite: (lactones / cyclic esters as heterocyclic ketones).

References:
- composer.py:826-848 (inline dispatch branch; REMOVED at this commit).
- rules.lactones.{is_monocyclic_lactone, name_monocyclic_lactone}.
- internal notes-DECOMP.md row 'lactone' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _has_separate_senior_group(mol: Any, lactone_info: dict) -> bool:
    """True when a suffix-capable characteristic group SENIOR to the lactone
    (a pseudoketone) is present as a SEPARATE group from the lactone itself.

     (the Blue Book): "A lactone, as a pseudoketone, ranks lower in the
    seniority of classes than an acid or an ester, but higher than an alcohol,
    amine, or imine." Table 4.1 (the Blue Book): Acids (7) > Ketones/pseudoketones
    (16). So when a molecule carries BOTH a lactone and a group senior to the
    ketone tier, that senior group owns the principal-characteristic-group
    suffix and the lactone C=O degrades to an ``oxo`` prefix.

    A senior-group match confined ENTIRELY to the lactone's own C(=O)-O ring
    motif IS the lactone (its ring ester re-detected), not a separate group, so
    it never triggers the guard — plain gamma-butyrolactone keeps oxolan-2-one.
    ``_PREFIX_ONLY_PRINCIPAL`` classes (e.g. isocyanide) can never be a suffix,
    so they never displace the lactone.
    """
    from ...perception.functional_groups import detect_functional_groups
    from ...rules.seniority import SENIORITY_ORDER, _PREFIX_ONLY_PRINCIPAL

    ketone_rank = SENIORITY_ORDER.index("ketone")
    senior_classes = {
        g for g in SENIORITY_ORDER[:ketone_rank]
        if g not in _PREFIX_ONLY_PRINCIPAL
    }

    # Atoms of the lactone's own ring motif (ring atoms + the exocyclic
    # carbonyl chalcogen + the optional carbonate second ring O).
    lactone_atoms = set(lactone_info["ring_atoms"])
    lactone_atoms.add(lactone_info["carbonyl_O_idx"])
    extra_o = lactone_info.get("extra_ring_O_idx")
    if extra_o is not None:
        lactone_atoms.add(extra_o)

    fgs = detect_functional_groups(mol)
    for fg_name, matches in fgs.items():
        if fg_name not in senior_classes:
            continue
        for match in matches:
            if not set(match).issubset(lactone_atoms):
                return True
    return False


def _is_lactone(features: Any) -> bool:
    """Predicate: features.mol contains a monocyclic lactone (SMARTS check)
    AND no senior acid/ester owns the principal-group suffix.

    Lazy import per PATTERNS § Lazy Import. The SMARTS substruct match is
    cheap; duplicate call between predicate + handler is acceptable for
    Plan-02 byte-identical preservation. Plan-04 performance benchmark
    can identify if memoization is needed.

    Pure read-only per internal notes / -26: reads features.mol via
    Chem.MolFromSmarts + GetSubstructMatches; no mutation.

     seniority guard: decline the lactone-as-parent when a
    SEPARATE suffix-capable group senior to the ketone/pseudoketone tier is
    present, so dispatch falls through to the general suffix assembler (which
    builds e.g. ``5-oxooxolane-2-carboxylic acid`` with the ring C=O as
    ``oxo``). See ``_has_separate_senior_group``.
    """
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    from ...rules.lactones import is_monocyclic_lactone
    lactone_info = is_monocyclic_lactone(mol)
    if lactone_info is None:
        return False
    if _has_separate_senior_group(mol, lactone_info):
        return False
    return True


def name_lactone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return lactone handler with coverage guard.

    Verbatim semantics of composer.py:827-848. Returns None if the
    coverage guard rejects (large substituted lactone where the bare
    name would be incomplete).
    """
    from ...rules.lactones import is_monocyclic_lactone, name_monocyclic_lactone
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    lactone_info = is_monocyclic_lactone(features.mol)
    if not lactone_info:
        return None

    total_heavy = features.mol.GetNumHeavyAtoms()
    ring_size = lactone_info.get('ring_size', 0)

    # Coverage guard: bare lactone name only when molecule is not much
    # larger than the ring. Macrocycles (ring_size > 8) bypass this guard
    # — the ring IS the parent. Per composer.py:828-834.
    if not (ring_size > 8 or total_heavy <= ring_size + 8):
        return None

    lactone_name = name_monocyclic_lactone(features.mol)
    if not lactone_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "lactone", _ha, lactone_name[:60],
        )

    pool = get_current_pool()
    pool.add(lactone_name, "lactone", features)
    # composer.py:848 inline: _inject_stereo_if_missing(features, pool.best.name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="lactone", iupac_section_cite="P-65.7", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_lactone", "_is_lactone"]
