"""Spelling check: a name of a structure whose PIN is a linear phane name (registered
with:func:`orthonym.validation.pin_spelling.register`).

 (2) (the Blue Book) "linear phanes consist of four or more rings or ring systems,
two of which must be terminal, and together with acyclic atoms or chains must consist of at least
seven nodes (components)."; (:23901) phane names are PINs for such compounds "even
though the compounds could also be named by substitutive or multiplicative nomenclature";
 (:24088) "Phane names are preferred IUPAC names rather than ring assembly names when
seven or more rings or ring systems are present."

The structure side is:mod:`.linear_phane` (RDKit only); the name side is a lexical test for a
linear phane parent hydride. OPSIN 2.9.0 reads no phane name, so no phane name is ever
pin_verified; the check withdraws the label from the substitutive, multiplicative or ring
assembly name the engine built instead. It also withdraws it where another parent cites more
of the principal characteristic group but the qualifying chain stays whole in one of its
substituents: the PIN may cite that substituent as a linear phane prefix (:19325
'trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane (PIN)';,
:16148), which no printed row decides for such a parent, so the name is not certified (fail
closed). No naming code, no OPSIN.
"""
from __future__ import annotations

from typing import Optional

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .linear_phane import is_linear_phane_name, linear_phane_pin_expected


@register("P-52.2.5.1")
def linear_phane_pin_check(mol: Chem.Mol, name: str) -> Optional[SpellingFailure]:
    """ (2) with and: a name of a structure whose PIN is a
    linear phane name must be that phane name; any other name (substitutive, multiplicative,
    ring assembly) is not the PIN. A name of a structure whose PIN may cite a linear phane
    prefix is not certified unless it cites one."""
    if is_linear_phane_name(name):
        return None
    verdict = linear_phane_pin_expected(mol)
    if verdict is None:
        return None
    chain = (f"{verdict.ring_systems} rings or ring systems on an unbranched chain of "
             f"{verdict.nodes} nodes with ring systems at both ends")
    if verdict.as_prefix:
        return SpellingFailure(
            "P-52.2.5.1",
            f"the PIN may cite a linear phane substituent prefix (P-29.3.6): {chain}, outside "
            f"the parent that cites the most of the principal characteristic group")
    return SpellingFailure("P-52.2.5.1", f"the PIN is a linear phane name: {chain}")
