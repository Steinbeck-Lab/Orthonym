"""Spelling checks: a name of a structure whose PIN is a linear or a cyclic phane name
(registered with:func:`orthonym.validation.pin_spelling.register`).

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
closed).

The cyclic half: (1) (:23828) "cyclophanes are cyclic phane structures containing one or more rings
or ring systems, at least one ring or ring system of which must be a mancude system attached to adjacent
atoms or chains at nonadjacent ring positions" and (:23831) "When the conditions given in
 for cyclic phane systems are not fulfilled, names of fused ring systems, bridged fused systems, or
von Baeyer systems are preferred IUPAC names": when the conditions are met, a fused, bridged fused, von Baeyer
or spiro name is not the PIN. The structure test is:mod:`.cyclophane`; it judges only where the book prints a
phane PIN, and abstains elsewhere. The same holds where a parent hydride with a senior atom "The senior parent
structure, whether cyclic or acyclic, has the senior atom...",:18917;,:18919) outranks the
phane: ``trimethyl[...heptaphan-7^4-yl]silane (PIN) (Si is senior to O)`` (:19325). The third check,:49684)
withdraws the label from a cyclophane name of more than seven nodes that cites no descriptor of an amplificant ring double
bond, as the book's PIN does (:49696). No naming code, no OPSIN.
"""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .cyclophane import is_phane_name, large_cyclophane_ring_double_bonds, p52_cyclophane_systems
from .lexer import normalise
from .linear_phane import is_linear_phane_name, linear_phane_pin_expected

#: a phane group with two attachments, the central group of a multiplicative name ('...heptaphane-1^4,7^4-diyl]di...')
_PHANE_DIYL = re.compile(r"phane-[\d^,a-z]+-diyl")


@register("P-52.2.5.1")
def linear_phane_pin_check(mol: Chem.Mol, name: str) -> Optional[SpellingFailure]:
    """ (2) with and: a name of a structure whose PIN is a
    linear phane name must be that phane name; any other name (substitutive, multiplicative,
    ring assembly) is not the PIN. A name of a structure whose PIN may cite a linear phane
    prefix is not certified unless it cites one."""
    verdict = linear_phane_pin_expected(mol)
    if verdict is None:
        return None
    if verdict.criterion == "P-51.3.1":
        # (:23178): two identical parents hold every principal group, so the PIN is the multiplicative name with
        # the phane as its central group ('3,3'-[...phane-1^4,7^4-diyl]dipropanoic acid'); a name that cites the phane in a
        # prefix, or a substitutive name, is not the PIN
        if _PHANE_DIYL.search(name):
            return None
        return SpellingFailure(
            "P-52.2.5.1", f"the PIN is a multiplicative name with the linear phane of {verdict.ring_systems} rings or "
                          "ring systems as its central group (P-51.3.1)")
    if is_linear_phane_name(name):
        return None
    chain = (f"{verdict.ring_systems} rings or ring systems on an unbranched chain of "
             f"{verdict.nodes} nodes with ring systems at both ends")
    if verdict.as_prefix:
        outside = ("the parent that has the senior atom (P-44.1.2)" if verdict.criterion == "P-44.1.2"
                   else "the parent that cites the most of the principal characteristic group")
        return SpellingFailure(
            "P-52.2.5.1",
            f"the PIN may cite a linear phane substituent prefix (P-29.3.6): {chain}, outside {outside}")
    return SpellingFailure("P-52.2.5.1", f"the PIN is a linear phane name: {chain}")


@register("P-52.2.5.1")
def cyclic_phane_pin_check(mol: Chem.Mol, name: str) -> Optional[SpellingFailure]:
    """ (1) with: a name of a structure whose PIN is a cyclic phane name must be a phane
    name; any other name (von Baeyer, spiro, fused, bridged fused) is not the PIN. A structure the test does not
    judge is not decided here."""
    if is_phane_name(name):
        return None
    if not p52_cyclophane_systems(mol):
        return None
    return SpellingFailure("P-52.2.5.1", "the PIN is a cyclic phane name: the structure meets P-52.2.5.1 (1)")


#: the stereodescriptor block at the front of a name, and a descriptor of an atom of an amplificant in it ('1^2Z')
_FRONT_BLOCK = re.compile(r"^\(([^()]*)\)-")
_COMPOSITE_DESCRIPTOR = re.compile(r"(?:^|,)\d+\^\d+[a-z]?[EZ](?=,|$)")


@register("P-93.5.5.3")
def cyclophane_descriptor_check(mol: Chem.Mol, name: str) -> Optional[SpellingFailure]:
    """ "Specification of double bonds" (the Blue Book): "The omission of stereodescriptors 'E' and 'Z' to
    denote the configuration of double bonds in cyclophanes is related to the total number of nodes in the system; for up
    to seven nodes, no descriptor is necessary. When more than seven nodes are present, all descriptors must be cited."
    The example (:49696) cites '1^2Z' for the double bond of a hydro amplificant, and gives the name without it as the
    other, not preferred form: '(1^2Z,2E)-1^1,1^4,1^5,1^6-tetrahydro-1(2,6)-pyridina-7(1,3)-benzenacyclododecaphan-2-ene (PIN)
    (2E)-...'. A phane name of a cyclophane of more than seven nodes that has such a double bond and cites no descriptor of
    an amplificant atom is not the PIN. No naming code, no OPSIN; a structure the cyclophane test does not judge is not
    decided here."""
    if not is_phane_name(name):
        return None
    bonds = large_cyclophane_ring_double_bonds(mol)
    if not bonds:
        return None
    front = _FRONT_BLOCK.match(normalise(name))
    if front is not None and _COMPOSITE_DESCRIPTOR.search(front.group(1)):
        return None
    return SpellingFailure(
        "P-93.5.5.3", f"a cyclophane of more than seven nodes cites all descriptors: {len(bonds)} double bond(s) "
                      "of an amplificant ring, none cited")
