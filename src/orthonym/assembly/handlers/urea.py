"""a phase urea handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:876-885 (inline branch) +
composer.py:2672-2759 (_try_name_urea body). Per internal notes, body
stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (ureas; retained name with N-substitution).

References:
- composer.py:876-885 (inline dispatch branch; REMOVED at this commit).
- composer.py:2672-2759 (_try_name_urea body).
- internal notes-DECOMP.md row 'urea' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


# (the Blue Book-:18200): 11 Amides "in the order of the
# corresponding acids", 12 Hydrazides, 13 Imides, 14 Nitriles,... Urea is an amide of
# carbonic acid (7b), so the carboxylic and other 7a amides (carboxamides, thio-
# amides, sulfonamides, amidines, hydrazonamides, N-hydroxy amides) and the esters
# stay senior to it, while every hydrazide, the imides and every class from the
# nitriles on rank below it.
_PRINCIPALS_JUNIOR_TO_UREA_ABOVE_NITRILE = frozenset({
    "sulfonohydrazide", "sulfinohydrazonohydrazide", "hydrazide", "thiohydrazide",
    "imidohydrazide", "imide",
})


def urea_is_senior_to(principal) -> bool:
    """True when urea outranks the perceived principal group (see above)."""
    if principal in _PRINCIPALS_JUNIOR_TO_UREA_ABOVE_NITRILE:
        return True
    from ...rules.seniority import SENIORITY_ORDER
    try:
        return SENIORITY_ORDER.index(principal) >= SENIORITY_ORDER.index("nitrile")
    except ValueError:
        return False


def _urea_principal_rank(features) -> Optional[str]:
    """'senior' when urea is the parent class against a perceived principal group
    (``urea_is_senior_to``) and its unit is a plain urea, else None.

    A plain urea: neither nitrogen bears a hydrazine nitrogen (a semicarbazide is
    named on hydrazinecarboxamide,, the Blue Book) or an acyl
    carbon (an N-acyl urea is a condensed urea,:33505, or a carboxamide:
    'N-carbamoyl-4-cyanobenzamide', cf. 'N-carbamoyl-2-phenylacetamide (PIN)':33364);
    a nitroso nitrogen is a prefix ('N-methyl-N-nitrosourea (PIN)',:25965). (The
    hydrazine test is defensive and mutation-surviving today: every probed semicarbazide
    with a junior principal group is declined or named on hydrazinecarboxamide before
    this is read.)"""
    fg = getattr(features, 'functional_groups', None) or {}
    matches = fg.get('urea') or ()
    principal = getattr(features, 'principal_group', None)
    mol = getattr(features, 'mol', None)
    if not matches or principal is None or mol is None:
        return None
    if not urea_is_senior_to(principal):
        return None
    from rdkit import Chem
    for match in matches:
        if len(match) < 4:
            return None
        c_idx = match[1]
        for n_idx in (match[0], match[3]):
            for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
                if nb.GetIdx() == c_idx or nb.GetAtomicNum() == 1:
                    continue
                sym = nb.GetSymbol()
                if sym == 'N':
                    nitroso = (nb.GetDegree() == 2 and any(
                        b.GetBondType() == Chem.BondType.DOUBLE
                        and b.GetOtherAtom(nb).GetSymbol() == 'O'
                        for b in nb.GetBonds()))
                    if not nitroso:
                        return None
                elif sym == 'C' and any(
                        b.GetBondType() == Chem.BondType.DOUBLE
                        and b.GetOtherAtom(nb).GetSymbol() in ('O', 'S', 'Se', 'Te', 'N')
                        for b in nb.GetBonds()):
                    return None
    return 'senior'


def urea_parent_expected(features) -> bool:
    """True when the PIN of this molecule is a urea name while another group was
    perceived as principal (``_urea_principal_rank``); used to label any other
    handler's name below the PIN."""
    try:
        return _urea_principal_rank(features) == 'senior'
    except Exception:  # noqa: BLE001 -- no answer: no label change
        return False


def _is_urea(features: Any) -> bool:
    """Urea FG present AND no principal group senior to urea.

    Mirrors composer.py:876 (principal_group None), widened by PIN class program
    Task 13: (the Blue Book) "Derivatives of urea formed by
    substitution on the nitrogen atom(s) are named as substitution products in
    accordance with the seniority order of urea that is ranked as an amide of
    carbonic acid" -- "N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea (PIN)"
    (:33336). The handler takes a plain urea whose perceived principal group ranks at
    or below the nitrile; a hydrazide or imide principal is junior to urea too,
    but the urea builder has no correct prefix for that group (it spells the
    hydrazide 'aminoamino'), so such a molecule keeps its handler and the name is
    labelled below the PIN (``urea_parent_expected``, inner_dispatch)."""
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('urea'):
        return False
    principal = getattr(features, 'principal_group', None)
    if principal is None:
        return True
    if principal in _PRINCIPALS_JUNIOR_TO_UREA_ABOVE_NITRILE:
        return False
    return urea_parent_expected(features)


def name_urea(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B urea handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _try_name_semicarbazone,
        _try_name_urea,
    )

    # (W2E-P1FG Task 12): a semicarbazone (R2C=N-NH-CO-NH2) is a urea
    # FG with principal_group None; name it substitutively BEFORE the plain
    # urea path (which would drop the ylidene). Fail-closed -> falls through.
    urea_name = _try_name_semicarbazone(features)
    # `_try_name_urea` builds a COMPLETE name: the retained parent plus every
    # N-substituent it walked itself. Enrichment can therefore only spell an
    # atom a second time -- it turned `N-(1-methylcyclohexyl)urea` into
    # `1-methylN-(1-methylcyclohexyl)urea`, numbering the gem-methyl against a
    # parent (urea) that has no atom 1 at all. Only the semicarbazone branch,
    # which names a different parent, is enriched.
    _complete_name = False
    if not urea_name:
        urea_name = _try_name_urea(features)
        _complete_name = bool(urea_name)
    if not urea_name:
        return None

    if not _complete_name:
        urea_name = _enrich_handler_name(features, urea_name, "urea")

    pool = get_current_pool()
    cand = pool.add(urea_name, "urea", features)
    if cand is None:
        return None

    # The retained urea parent has NO numbered skeleton -- its only locants
    # are the italic letters N / N'. Declare that scope so a
    # numeric front-of-name stereo block, which could not resolve against it,
    # is never prepended. See _inject_stereo_if_missing for the measurement.
    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="urea", iupac_section_cite="P-66.4.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_urea", "_is_urea"]
