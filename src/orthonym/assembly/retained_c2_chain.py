"""The retained name of a substituted two-carbon chain parent, for the general engine's chain parent.

 "Retained names as preferred IUPAC names" (the Blue Book): "Only the following five
carboxylic acids retained names and are also preferred IUPAC names. All can be functionalized, but only
acetic acid, benzoic acid, and oxamic acid can be substituted"; its row ':29725' reads 'CH3-COOH acetic
acid (PIN) ethanoic acid'. (:34750-:34754) 'CH3-CN acetonitrile (PIN) ethanenitrile', "with
unlimited substitution"; (:32691-:32693) 'acetamide (PIN)'; (:34964) acetaldehyde
"with substitution allowed". A substituted two-carbon acid, nitrile or aldehyde has ONE substitutable
position, so every substituent locant is omitted,:3031: "All locants are omitted for parent
compounds when all substitutable hydrogen atoms have the same locant") and every substituent after the
first is enclosed,:7304, 'bromo(chloro)acetic acid (PIN)':7312); the amide has two
substitutable positions (N and C-2), keeps its locants, and only its parent word changes ('N-carbamoyl-2-
phenylacetamide (PIN)',:33364). "In some cases, such as acetic acid, alkyl substituents are not allowed as
the resulting compound requires a different parent structure" (:7304): a substituent of a two-carbon chain
that is the longest chain through its characteristic carbon is never an alkyl group.

The general engine's chain parent (``general_engine._assemble``) wrote ``2-X-ethanoic acid`` for every
such parent. This module holds the one decision, on the structure side the engine already has; the
assembly is ``composition_primitives.retained_acetic_from_prefixes``, the writer the other producers use.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rdkit import Chem

from .composition_primitives import (
    RETAINED_C2_ACYL_BY_SUFFIX,
    RETAINED_C2_PARENT_BY_SUFFIX,
    _join_prefix_to_name,
    _join_prefixes,
    retained_acetic_from_prefixes,
)

#: the retained word of the two-carbon parent, by the suffix the systematic name would carry
RETAINED_C2_WORD = {**RETAINED_C2_PARENT_BY_SUFFIX, **RETAINED_C2_ACYL_BY_SUFFIX, "al": "acetaldehyde"}

_ALPHA_PREFIX = re.compile(r"^2(?:,2)*-")
_N_PREFIX = re.compile(r"^N(?:,N)*(?:,2)*-|^2(?:,2)*-")


def is_alkyl_attachment(atom) -> bool:
    """True when ``atom``, the first atom of a substituent, is an acyclic carbon of a hydrocarbon group:
    no double or triple bond to a heteroatom, so not the carbon of a cyano, carboxy, formyl or acyl group,
    and not an atom of a ring. An alkyl, alkenyl or alkynyl substituent of a two-carbon chain extends the
    chain: (the Blue Book) "In some cases, such as acetic acid, alkyl substituents are not
    allowed as the resulting compound requires a different parent structure"."""
    if atom.GetSymbol() != 'C' or atom.IsInRing():
        return False
    return not any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                   and b.GetOtherAtom(atom).GetSymbol() != 'C' for b in atom.GetBonds())


def retained_c2_name(*, hydride: str, chain_length: int, suffix_core: Optional[str],
                     pg_locants: List[int], prefix_parts: List[str], charged: bool,
                     parent_stereo: bool, alkyl_prefix: bool = False) -> Optional[Tuple[str, str]]:
    """``(name, retained word)`` of a saturated two-carbon chain parent with one suffix group and the
    substituent prefixes ``prefix_parts`` (each 'locants-prefix', in alphanumerical order), or
    ``None`` when the systematic name stays: no substituent, a charge, a descriptor of the parent
    (a bare '(R)-' is written by the other producers; here the systematic spelling cites it with its
    locant), a second suffix group, or an aldehyde with more than one prefix (their enclosing
    follows another rule,,:29852)."""
    word = RETAINED_C2_WORD.get(suffix_core or "")
    if (word is None or hydride != "ethane" or chain_length != 2 or pg_locants != [1]
            or not prefix_parts or charged or parent_stereo or alkyl_prefix):
        return None
    if suffix_core == "amide":
        if not all(_N_PREFIX.match(p) for p in prefix_parts):
            return None
        return _join_prefix_to_name(_join_prefixes(prefix_parts), word), word
    if not all(_ALPHA_PREFIX.match(p) for p in prefix_parts):
        return None
    if suffix_core == "al" and len(prefix_parts) != 1:
        return None
    return retained_acetic_from_prefixes(
        prefix_parts, parent=word, enclose_subsequent=len(prefix_parts) > 1), word
