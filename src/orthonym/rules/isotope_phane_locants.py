"""The locants of an isotope descriptor that cite an atom of a phane amplificant are composite locants.

 Numbering, Numbering of phane parent hydrides, (the Blue Book): "locants for
amplificant atoms are constructed by citing the actual locants of the amplificant as superscripts to the locant
of the superatom that represents the amplificant in the simplified skeleton" (``1^4``: atom 4 of the amplificant
at superatom 1). (:15115): "The seniority of a composite locant is determined first on the basis of its
primary locants,... and, if these locants are identical, on the basis of the complete composite locant itself".
 Letter and/or numeral locants (:44208): "Specific positions of nuclides must be indicated in the isotopic
descriptor by appropriate locants".

The placement search of ``rules.isotopes`` writes every locant as an integer (1... the number of heavy atoms) and lets
OPSIN say which one reads back as the labelled atom. OPSIN reads ``14`` as ``1^4`` in a phane name, so the integer
spelling reads back, but only where no node of the simplified skeleton has that number: in a skeleton of 12 or more
nodes ``12`` and ``13`` are nodes, the search never reaches ``1^2`` and ``1^3``, and the lowest composite locant
 is skipped. So the search is offered the composite locants of the amplificants of the skeleton name
(:func:`composite_tokens`, the superatom locants of its amplification blocks, ``1,7(1),3,5(1,4)-tetrabenzena``,
,, with the amplificant locants 1... n), before the integers, in the order of: the
strategies that cite one locant, or one locant per distinct position, take them (``_find_best_placement``,
``_decorate_distinct_multi_locant``), and every candidate is read back like any other.

The strategy of mixed nuclides and the others still write integers, so the spelling is also settled on the name the
passes return (``rules.isotope_passes.decorate_passes``): an integer locant of two or more digits in a descriptor of
a name with amplification blocks is split into a superatom locant of the name and the amplificant locant, and the
composite spelling replaces it where OPSIN reads the name back as the labelled molecule. A locant that no split reads
back as stays as it is: it is the number of a node of the simplified skeleton. A name made that way in a skeleton of
11 or more nodes (or of a node count the name does not give) is recorded below the PIN: the integer search could not
reach a composite locant whose integer is a node, so the set it found is not proven the lowest.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from rdkit import Chem

from ..data.locant_text import render_locants

#: ``1,7(1)`` and ``3,5(1,4)`` of ``1,7(1),3,5(1,4)-tetrabenzenaheptaphane``: superatom locants and the attachment locants
#: of their amplificant in parentheses,.
_AMPLIFICATION_RE = re.compile(r"(?<![\w^'])(\d+(?:,\d+)*)\(\d+[a-z]?(?:,\d+[a-z]?)*\)")

_NUCLIDE = r"\d+[A-Z][a-z]?\d*"
_GROUP_RE = re.compile(rf"(?:(?P<locs>[^,()\s-]+(?:,[^,()\s-]+)*)-)?(?P<nuc>{_NUCLIDE})")
_PAREN_RE = re.compile(r"\(([^()]*)\)")
_COMPOSITE_RE = re.compile(r"(\d+)\^(\d+)([a-z]?)'*$")

#: The smallest number of nodes of a skeleton in which the integer of a composite locant can be a node: ``1^1`` is ``11``.
_FIRST_COLLISION = 11


class CompositeLocant(int):
    """A composite locant of an atom of a phane amplificant (``1^4``) as the placement search carries it: an integer
    that orders as (:15115) orders composite locants (the primary locant first, then the superscript, a
    letter after its number) and is written as the composite locant. It is above every integer locant of a molecule."""

    def __new__(cls, primary: int, number: int, letter: str = ""):
        self = super().__new__(cls, primary * 1_000_000 + number * 1_000 + (ord(letter) - 96 if letter else 0))
        self.text = f"{primary}^{number}{letter}"
        self.parts = (primary, number, letter)
        return self

    def __str__(self) -> str:
        return self.text

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return format(self.text, spec)


def composite_sort_key(token: str) -> Optional[tuple]:
    """The key of a composite locant text (``1^4``, ``2^4a``) among the keys of ``isotopes._locant_sort_key``
    (``(number, has_letter, letter, primes)``): its primary locant, then the superscript; None for a
    token that is no composite locant."""
    m = _COMPOSITE_RE.match(token) if "^" in token else None
    if m is None:
        return None
    return (int(m.group(1)), 0, "", token.count("'"), int(m.group(2)), 1 if m.group(3) else 0, m.group(3))


def _amplificant_shape(original: Chem.Mol) -> Tuple[int, bool]:
    """``(atoms of the largest ring system of a phane skeleton, whether a ring system is fused)``: no amplificant
    locant is higher, and a fusion atom has a lettered locant. The ring systems are the nodes of
    ``validation.spelling.cyclophane`` (rings of up to eight atoms that share an atom); the macrocycle is none."""
    try:
        from ..validation.spelling.cyclophane import small_ring_systems
        rings = [set(r) for r in original.GetRingInfo().AtomRings() if len(r) <= 8]
        fused = any(sum(1 for r in rings if atom in r) >= 2
                    for atom in {a for r in rings for a in r})
        return max([len(system) for system in small_ring_systems(original)] or [0]), fused
    except Exception:  # noqa: BLE001 -- a structure RDKit cannot ring-perceive has no amplificants to offer
        return 0, False


def composite_tokens(skel: str, original: Chem.Mol) -> List[CompositeLocant]:
    """The composite locants of the atoms of the amplificants of the phane skeleton name ``skel``, in the
    order of: for each superatom locant of its amplification blocks the amplificant locants 1... the atoms
    of the largest ring system of ``original``, with the lettered locants of a fusion atom when a ring system is
    fused. Empty for a name that is no phane name."""
    if "phan" not in skel:
        return []
    superatoms = superatom_locants(skel)
    if not superatoms:
        return []
    reach, fused = _amplificant_shape(original)
    letters = ("", "a") if fused else ("",)
    return sorted((CompositeLocant(p, q, x) for p in superatoms for q in range(1, reach + 1) for x in letters))


def skeleton_node_count(name: str) -> Optional[int]:
    """The number of nodes of the simplified skeleton of the phane name ``name``: ``tridecaphane`` 13), the
    largest when the name has several; None when no skeleton word is found."""
    from .phane import _phane_node_multiplier
    spans = []
    for n in range(3, 201):
        word = _phane_node_multiplier(n) + "phan"
        start = name.find(word)
        while start >= 0:
            spans.append((start, start + len(word), n))
            start = name.find(word, start + 1)
    kept = [n for s, e, n in spans if not any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for s2, e2, _ in spans)]
    return max(kept) if kept else None


def superatom_locants(name: str) -> List[int]:
    """The superatom locants the amplification blocks of ``name`` cite, ascending (empty for a name with none)."""
    return sorted({int(n) for m in _AMPLIFICATION_RE.finditer(name) for n in m.group(1).split(",")})


def _descriptor_locant_lists(name: str):
    """``(start, end, locants)`` of the locant list of every nuclide group of every isotope descriptor in ``name``."""
    from .isotope_induced import _is_nuclide
    for paren in _PAREN_RE.finditer(name):
        body, base = paren.group(1), paren.start(1)
        pos, groups = 0, []
        while pos < len(body):
            group = _GROUP_RE.match(body, pos)
            if group is None or not _is_nuclide(group.group("nuc")):
                groups = []
                break
            groups.append(group)
            pos = group.end()
            if pos < len(body):
                if body[pos] != ",":
                    groups = []
                    break
                pos += 1
        for group in groups:
            if group.group("locs"):
                yield base + group.start("locs"), base + group.end("locs"), group.group("locs").split(",")


def _seniority(locant: str) -> Optional[tuple]:
    """: the primary locant first, then the superscript; None for a locant of another kind."""
    if locant.isdigit():
        return (int(locant), 0, "")
    m = _COMPOSITE_RE.match(locant)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def _splits(locant: str, superatoms: List[int]) -> List[str]:
    """The composite spellings of the integer ``locant``: a superatom locant of the name and the rest as its
    superscript, lowest primary locant first."""
    out = []
    for primary in superatoms:
        head = str(primary)
        rest = locant[len(head):]
        if locant.startswith(head) and rest and not rest.startswith("0"):
            out.append(f"{primary}^{rest}")
    return out


def _sorted_in_seniority(name: str) -> str:
    """``name`` with the locants of each descriptor group in the order of (the primary locants first)."""
    for start, end, locants in reversed(list(_descriptor_locant_lists(name))):
        keys = [_seniority(x) for x in locants]
        if len(locants) > 1 and "^" in name[start:end] and None not in keys:
            ordered = [x for _k, x in sorted(zip(keys, locants), key=lambda kv: kv[0])]
            name = name[:start] + render_locants(ordered) + name[end:]
    return name


def composite_locants(name: str, reads_back: Callable[[str], bool]) -> str:
    """``name`` with every integer locant of its isotope descriptors that cites an atom of an amplificant spelt as
    a composite locant; ``reads_back(candidate)`` says whether OPSIN reads ``candidate`` as the labelled
    molecule. ``name`` itself when it has no amplification block, no such locant, or no spelling reads back."""
    if "phan" not in name:
        return name
    superatoms = superatom_locants(name)
    if not superatoms:
        return name
    tokens: List[Tuple[int, int, str]] = []
    for start, end, locants in _descriptor_locant_lists(name):
        at = start
        for locant in locants:
            if locant.isdigit() and len(locant) > 1:
                tokens.append((at, at + len(locant), locant))
            at += len(locant) + 1
    changed = False
    for start, end, locant in reversed(tokens):      # right to left: the offsets of the tokens still to come stand
        for candidate in _splits(locant, superatoms):
            trial = name[:start] + candidate + name[end:]
            if reads_back(trial):
                name, changed = trial, True
                break
    if not changed:
        return name
    ordered = _sorted_in_seniority(name)
    return ordered if ordered == name or reads_back(ordered) else name


def decorated_with_composite_locants(name: Optional[str], original: Chem.Mol) -> Optional[str]:
    """The name the decorator found, its descriptor locants spelt as composite locants where they cite an atom of a
    phane amplificant (:func:`composite_locants`), checked by the read-back of ``rules.isotopes``. A non-PIN record
    of the name (``metrics.provenance``) goes to the spelling that replaces it."""
    if not name or "phan" not in name:
        return name
    from . import isotopes as _iso
    shift = _iso._BOND_SHIFT_PARSE_REFUSED.get()    # a probe the read-back refuses must not mark the name it replaces
    radical = _iso._RADICAL_READING.set(any(a.GetNumRadicalElectrons() > 0 for a in original.GetAtoms()))
    try:
        respelt = composite_locants(name, lambda candidate: _iso._isotope_round_trips(candidate, original))
    finally:
        _iso._RADICAL_READING.reset(radical)
        _iso._BOND_SHIFT_PARSE_REFUSED.set(shift)
    if respelt != name:
        from ..metrics.provenance import (
            record_derived_non_pin_fragment, record_derived_uncertified_pin_name, record_non_pin_fragment)
        record_derived_non_pin_fragment(name, respelt)
        record_derived_uncertified_pin_name(name, respelt)
        nodes = skeleton_node_count(respelt)
        if nodes is None or nodes >= _FIRST_COLLISION:
            # written as integers, so a composite locant whose integer is a node of the skeleton was out of reach
            record_non_pin_fragment(respelt)
    return respelt
