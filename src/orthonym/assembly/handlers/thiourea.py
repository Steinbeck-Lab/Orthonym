"""Chalcogen-urea handler — retained parent thiourea/selenourea/tellurourea.

Residue R3. Mirrors ``handlers/urea.py`` exactly; the body lives in
``composer._try_name_thiourea`` for the same reason urea's does.

IUPAC cite: "Chalcogen analogues of urea and isourea"
(``the Blue Book Blue Book``), subsection (``:33439``):
*"Chalcogen analogues of urea are named by functional replacement nomenclature
using the prefixes 'thio', 'seleno', and 'telluro'. Preferred IUPAC names use
the letter locants N, and N'."* Worked ``(PIN)`` examples: ``thiourea (PIN)``
(``:33444``) and ``N-(butan-2-yl)selenourea (PIN)`` (``:33451``).

Why the parent is the thiourea and not the ring the molecule also carries:
 (``:33320``) ranks urea *"as an amide of carbonic acid"*; puts amides at class 11 (``:18184``) against carbon rings at class 40
(``:18216``); (``:18875``) selects on that order; and the
"ring outranks chain" licence is gated on *"Within the same heteroatom class"*
, ``:24096``), which a class-40 ring does not share with an amide.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _thiourea_principal_rank(features: Any) -> Optional[str]:
    """'senior' when a plain chalcogen-urea is the parent class against a
    perceived principal group (``urea_is_senior_to``), else None.

    Mirrors ``handlers.urea._urea_principal_rank``: thiourea ranks as an
    amide of carbonic acid (P-41 class 11, P-66.1.6.1.3), junior to urea
    (=O > =S), acids, esters, acid halides, and 7a amides, and senior to
    hydrazides, cyclic imides, nitriles, aldehydes, ketones, alcohols,
    thiols, amines, and imines.
    """
    from rdkit import Chem

    from .urea import acyclic_imide_present, urea_is_senior_to

    fg = getattr(features, 'functional_groups', None) or {}
    matches = fg.get('thiourea') or ()
    principal = getattr(features, 'principal_group', None)
    mol = getattr(features, 'mol', None)
    if not matches or principal is None or mol is None:
        return None
    if fg.get('urea') or fg.get('nitrooxy') or fg.get('nitrite'):
        return None
    if not urea_is_senior_to(principal):
        return None
    if principal == "imide" and acyclic_imide_present(mol, fg.get('imide')):
        return None
    for match in matches:
        if len(match) < 4:
            return None
        c_idxs = [i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        n_idxs = [i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(c_idxs) != 1 or len(n_idxs) != 2:
            return None
        c_idx = c_idxs[0]
        for n_idx in n_idxs:
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


def thiourea_parent_expected(features: Any) -> bool:
    """True when the PIN of this molecule is a chalcogen-urea name while
    another group was perceived as principal (``_thiourea_principal_rank``);
    used to label any other handler's name below the PIN.
    """
    try:
        return _thiourea_principal_rank(features) == 'senior'
    except Exception:  # noqa: BLE001 -- no answer: no label change
        return False


def _is_thiourea(features: Any) -> bool:
    """Chalcogen-urea FG present AND no principal group senior to thiourea.

    Mirrors ``handlers.urea._is_urea``: handles a plain chalcogen-urea when
    no senior characteristic group (or urea / nitrooxy / nitrite) is present.
    """
    from .urea import _PRINCIPALS_JUNIOR_TO_UREA_ABOVE_NITRILE

    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('thiourea') or fg.get('urea') or fg.get('nitrooxy') or fg.get('nitrite'):
        return False
    principal = getattr(features, 'principal_group', None)
    if principal is None:
        return True
    if principal in _PRINCIPALS_JUNIOR_TO_UREA_ABOVE_NITRILE:
        return False
    return thiourea_parent_expected(features)


def name_thiourea(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B chalcogen-urea retained-name handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _inject_stereo_if_missing,
        _try_name_thiourea,
    )

    name = _try_name_thiourea(features)
    if not name:
        return None

    # `_try_name_thiourea` builds a COMPLETE name (retained parent + every
    # N-substituent; it REFUSES rather than skipping an un-nameable one), so
    # enrichment can only spell an atom twice -- it re-discovered the handler's
    # own core and produced `1-(carbamothioylamino)N-cyclohexylthiourea`, the
    # single thiourea unit spelled TWICE. Not enriched.

    pool = get_current_pool()
    cand = pool.add(name, "thiourea", features)
    if cand is None:
        return None

    # The retained chalcogen-urea parent has NO numbered skeleton -- its only
    # locants are the italic letters N / N', the Blue Book;:33446
    # "Numerical locants are no longer used for thiourea in the IUPAC preferred
    # name"). Declaring that scope stops a numeric front-of-name stereo block
    # being prepended: `(1S,3R,5S)-N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]-N'-
    # (prop-2-en-1-yl)thiourea` cites 1/3/5 against a parent that has no atom
    # 1, 3 or 5, and is OPSIN-unparseable. Stereo inside an N-substituent is
    # still cited there by the substituent namer
    # (`N-[(S)-1-cyclohexylethyl]-N'-(prop-2-en-1-yl)thiourea`).
    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name,
            class_id="thiourea",
            iupac_section_cite="P-66.1.6.1.3",
            fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_thiourea", "_is_thiourea"]
