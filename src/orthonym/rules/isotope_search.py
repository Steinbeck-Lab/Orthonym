"""What the isotope placement search asks OPSIN, and what it does not need to.

The search of ``rules.isotopes`` builds a candidate name for every pair of an insertion offset and a
locant and reads each back (OPSIN, then the comparison with the input). The offsets are every letter of
the skeleton name and the locants every number up to the size of the molecule, so the pairs grow with
both: a CD3 of a five-component mixture was 40,000 candidates, 56 s, and not one of them was read. The
refused ones tell where the rest are refused, and this module is that knowledge, drawn from OPSIN 2.9.0
itself (``jvm_bridge.refusal_kind`` reads its message):

* An isotope descriptor is one grammar symbol (``isotopeSpecification``, ``regexTokens.xml``), whatever
  nuclide and locants it cites. A name OPSIN cannot tokenise with a descriptor and a locant at an offset
  ("<name> is unparsable...") cannot be tokenised with another locant there: the offset is dead for every
  locant.
* The locants of a descriptor are looked up in the one fragment of the part of the name that follows it
  (``StructureBuildingMethods.applyIsotopeSpecifications``, ``Fragment.getAtomByLocantOrThrow``: "Could not
  find the atom with locant N."); a letter locant with a numeral ('O12', ``Fragment.getAtomByLocant``) needs
  the atom of that number. Almost every fragment numbers its atoms 1..n, so the locants above the first one
  OPSIN does not find are not found either, and the offset is capped there. That is NOT a rule of OPSIN: its
  data has fragments with gaps in their integer locants, measured in 2.9.0: ``(34-2H1)gorgostane`` is read
  and ``(29-2H1)gorgostane`` is not (the side chain of gorgostane has 33 and 34 and no 29-32); rotenane and
  aurone read ``(21-2H1)`` after 8 and 9 (the "2^1" of a substituent position); arginine reads ``(99-13C)``
  (the guanidino carbon) and not ``(6-13C)``. A cap is taken only when the first:data:`GAP_WINDOW`
  locants above the first absent one, and the limit of the search, are refused for a missing locant too:
  that sees the gaps of gorgostane, rotenane and aurone (the widest measured, 13) and not the far locant
  of arginine. Orthonym names no gorgostane, rotenane or aurone parent, and a search reaches arginine's 99
  only for a molecule of 99 heavy atoms with its guanidino carbon labelled; a candidate pruned by mistake
  is a name lost, never a wrong one, since what is asked is read back as ever. A natural-product parent
  that Orthonym comes to name is to be added to the test that pins this
  (``tests/unit/rules/test_isotope_search.py``).

Neither rule changes what a candidate is read as: it only leaves out candidates a refusal of the same
kind has already shown OPSIN cannot read. A refusal of any other kind, a refusal that names another locant
than the one asked, a name whose twin spelling is read instead (``validation.twin_spellings``), an OPSIN that
is not the pinned 2.9.0 release the rules were drawn from (``ORTHONYM_OPSIN=main`` floats with its
branch, ``ORTHONYM_OPSIN_JAR`` is anything), and anything OPSIN cannot say leave every candidate in the
search.
"""
from __future__ import annotations

import collections
import functools
import os
import re
from typing import Callable, Dict, Optional, Tuple

from rdkit import Chem

from .isotope_compare import RADICAL_READING as _RADICAL_READING

#: A letter locant that cites an atom by the numeral of the atom it is attached to ('N7', 'O12')
_NUMBERED_LETTER_LOCANT_RE = re.compile(r"[A-Z][a-z]?(\d+)")

#: How many locants above the first one a part lacks are asked before the part is taken to have no more
#: (the widest gap measured in OPSIN 2.9.0: aurone's 8 -> 21, 13; rotenane's 9 -> 21, 12; gorgostane's
#: 29 -> 33, 4).
GAP_WINDOW = 16


@functools.lru_cache(maxsize=8)
def _pinned_release(opsin_choice: str, opsin_jar: str) -> bool:
    """Is the OPSIN this process uses the pinned 2.9.0 release jar (``jars.active_opsin``)? The inputs are
    the variables that choose it, so the answer is kept for as long as they stand."""
    try:
        from .. import jars
        found = jars.active_opsin()
    except Exception:  # no jar: no read-back either
        return False
    return found["label"] == "2.9.0" and found["path"] is not None


def pinned_opsin() -> bool:
    """The rules above hold for the OPSIN in use: the pinned 2.9.0 release jar."""
    return _pinned_release(os.environ.get("ORTHONYM_OPSIN", "").strip(), os.environ.get("ORTHONYM_OPSIN_JAR", ""))


def direct_refusal_kind(candidate: str, token=None) -> Optional[str]:
    """How OPSIN refused ``candidate`` (``jvm_bridge.refusal_kind``: 'grammar', 'locant', 'other'), when
    its refusal of the name as written is the whole story and the rules of this module hold for it.

    None when OPSIN reads it; when a twin spelling of it is read instead (``validation.twin_spellings``: the
    read-back asks the twin when OPSIN refuses the name); when OPSIN cannot say or is not the pinned
    release (:func:`pinned_opsin`); and, for a refusal of a locant, when it is not the locant ``token``
    (a descriptor that cites several locants is refused for the first one the part lacks, which need not be
    the one the search asked about)."""
    from ..jvm_bridge import opsin_refusal
    from ..validation.twin_spellings import twin_of
    if not pinned_opsin():
        return None
    refusal = opsin_refusal(candidate, _RADICAL_READING.get())
    if refusal is None or twin_of(candidate) is not None:
        return None
    kind, locant = refusal
    if kind == "locant" and token is not None and locant != str(token):
        return None
    return kind


def _refused_for_the_locant(name: str, token) -> bool:
    from . import isotopes as _iso
    return not _iso._opsin_parse(name) and direct_refusal_kind(name, token) == "locant"


def locants_absent_above(skel: str, off: int, descriptor_of: Callable, first_absent: int, limit: int,
                         suffix: str = "") -> bool:
    """Is every one of the:data:`GAP_WINDOW` locants above ``first_absent`` (the first one OPSIN did not
    find at ``off``), and the ``limit``, refused for a locant OPSIN does not find too? False when OPSIN
    reads one of them or refuses it for another reason: the part has a gap in its numbering, and no cap is
    known."""
    for probe in sorted({first_absent + step for step in range(1, GAP_WINDOW + 1)} | {limit}):
        if first_absent < probe <= limit:
            token = f"{probe}{suffix}"
            if not _refused_for_the_locant(skel[:off] + descriptor_of(token if suffix else probe) + skel[off:], token):
                return False
    return True


def integer_locant_reach(skel: str, off: int, descriptor_of: Callable, limit: int,
                         suffix: str = "") -> Tuple[bool, Optional[int]]:
    """Where OPSIN reads a descriptor that cites the locants ``1``, ``2``,... (each followed by ``suffix``:
    ``"'"`` for the primed locants of a ring assembly) at offset ``off`` of ``skel``, asked in order up to
    ``limit``: ``(dead, cap)``. ``dead``: no descriptor is read at the offset; ``cap``: the highest locant of
    the series the part has (0 when it has none), None when nothing is known. ``descriptor_of(token)``
    spells the descriptor of one locant token."""
    from . import isotopes as _iso
    for number in range(1, limit + 1):
        token = f"{number}{suffix}"
        candidate = skel[:off] + descriptor_of(token) + skel[off:]
        if _iso._opsin_parse(candidate):
            continue
        kind = direct_refusal_kind(candidate, token)
        if kind is None:
            return False, None
        if kind == "grammar":
            return (True, 0) if number == 1 and not suffix else (False, None)
        if kind == "locant":
            known = locants_absent_above(skel, off, descriptor_of, number, limit, suffix)
            return False, (number - 1 if known else None)
    return False, limit


class LocantReach:
    """The offsets and locants of one placement search that OPSIN has shown it cannot read.

    ``descriptor_of(locant)`` spells the descriptor of the search for one locant, ``limit`` is the highest
    integer locant the search tries. The search asks:meth:`beyond` before it builds a candidate and
    tells:meth:`learn` about every one that was not read back.

    A refusal of the grammar at an offset makes the offset dead for every locant. The refusal of an integer
    locant for want of the locant (the first absent one) refuses that locant again in the next spelling of
    the descriptor. The locants above it are refused once the:data:`GAP_WINDOW` locants above it have been
    asked by the search itself and were all refused for a missing locant, and so is the limit: the integer
    locants of a fragment are 1..n with the exceptions this module states, and a locant that is read, or refused
    for another reason, above the first absent one leaves the offset uncapped. So a search that ends
    within the window asks nothing it would not have asked. A locant with a numeral ('O12') comes before the
    integers in the search and is as many as the atoms of the molecule, so the first one asked at an offset
    reads the integers there (:func:`integer_locant_reach`). Nothing is learnt, and nothing is beyond reach,
    where the rules of this module do not hold (:func:`pinned_opsin`)."""

    def __init__(self, skel: str, limit: int, descriptor_of: Callable):
        self.skel = skel
        self.limit = limit
        self.descriptor_of = descriptor_of
        self.dead: set = set()
        self.caps: Dict[int, int] = {}
        self.absent: Dict[int, int] = {}      # offset -> the first integer locant OPSIN did not find
        self.refused: Dict[int, set] = {}     # offset -> the locants above that one refused for a missing locant
        self.gapped: set = set()              # offsets where a locant above the first absent one is not refused so
        self._probed: set = set()

    @staticmethod
    def number_of(locant) -> Optional[int]:
        """The numeral a locant cites: the integer itself, the digits of 'N7'; None for no locant or a
        bare letter."""
        if isinstance(locant, int):
            return locant
        numbered = _NUMBERED_LETTER_LOCANT_RE.fullmatch(locant) if isinstance(locant, str) else None
        return int(numbered.group(1)) if numbered else None

    def _cap_at(self, off: int, cap: int) -> None:
        self.caps[off] = min(self.caps.get(off, cap), cap)

    def _window_refused(self, off: int, first_absent: int) -> bool:
        """The window above ``first_absent`` has been asked and refused for a missing locant, and so is the
        limit (asked here)."""
        window = range(first_absent + 1, min(first_absent + GAP_WINDOW, self.limit) + 1)
        if not self.refused.get(off, set()) >= set(window):
            return False
        return _refused_for_the_locant(self.skel[:off] + self.descriptor_of(self.limit) + self.skel[off:], self.limit)

    def beyond(self, off: int, locant) -> bool:
        """Has OPSIN been shown unable to read a descriptor that cites ``locant`` at ``off``?"""
        number = self.number_of(locant)
        if number is None:
            return False
        if off in self.dead:
            return True
        if not isinstance(locant, int) and off not in self._probed:
            self._probed.add(off)
            dead, cap = integer_locant_reach(self.skel, off, self.descriptor_of, self.limit)
            if dead:
                self.dead.add(off)
                return True
            if cap is not None:
                self._cap_at(off, cap)
        if off in self.caps:
            return number > self.caps[off]
        first_absent = self.absent.get(off)
        if first_absent is None or off in self.gapped:
            return False
        if number == first_absent:
            return True                       # the refusal that was learnt, in the next spelling
        if number > first_absent + GAP_WINDOW:
            if self._window_refused(off, first_absent):
                self._cap_at(off, first_absent - 1)
                return True
            self.gapped.add(off)
        return False

    def learn(self, off: int, locant, candidate: str, spelling: int = 0) -> None:
        """Learn from ``candidate`` (the first spelling of the descriptor of ``locant`` at ``off``), which
        was not read back: how OPSIN refused it (:func:`direct_refusal_kind`). A refusal of another kind
        (a valency, a bond assignment) says the locant exists, and learns nothing, unless it is a locant
        above the first absent one (a gap)."""
        from . import isotopes as _iso
        number = self.number_of(locant)
        if number is None or spelling != 0:
            return
        kind = None if _iso._opsin_parse(candidate) else direct_refusal_kind(candidate, locant)
        if kind == "grammar":
            self.dead.add(off)
            return
        first_absent = self.absent.get(off)
        if kind == "locant" and isinstance(locant, int) and locant >= 1:
            if first_absent is None or locant < first_absent:
                self.absent[off] = locant
                self.refused[off] = set()
            elif locant > first_absent:
                self.refused.setdefault(off, set()).add(locant)
        elif first_absent is not None and isinstance(locant, int) and locant > first_absent:
            self.gapped.add(off)


#: The fixed-H InChI of a molecule by its canonical isomeric CXSMILES. One placement search reads the
#: same few structures over and over (the search of each labelled group of a molecule parses the same
#: candidate names, so OPSIN answers the same structures, and each is judged against the same input);
#: InChI of a 200-atom molecule takes 0.4-1 s, which was the whole cost of a search on one.
_FIXED_H_INCHI: "collections.OrderedDict[str, str]" = collections.OrderedDict()
_FIXED_H_INCHI_MAX = 256


def fixed_h_inchi(mol: Chem.Mol) -> str:
    """``MolToInchi(mol, options="/FixedH")``, remembered by the canonical isomeric CXSMILES of ``mol``: what
    the canonical SMILES leaves out and InChI reads (the radical electrons, ``|^1:0|``) is in the key
    ('[CH2]CC(=O)O' with and without a radical electron are one SMILES and two InChIs). Raises what InChI
    raises."""
    key = Chem.MolToCXSmiles(mol)
    hit = _FIXED_H_INCHI.get(key)
    if hit is not None:
        _FIXED_H_INCHI.move_to_end(key)
        return hit
    from rdkit.Chem import inchi as _inchi
    value = _inchi.MolToInchi(mol, options="/FixedH") or ""
    _FIXED_H_INCHI[key] = value
    if len(_FIXED_H_INCHI) > _FIXED_H_INCHI_MAX:
        _FIXED_H_INCHI.popitem(last=False)
    return value
