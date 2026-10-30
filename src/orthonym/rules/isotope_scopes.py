"""Isotope labels in several scopes of one name (IUPAC 2013 recommendations,.

The decorator of ``rules.isotopes`` places one descriptor per attachment atom, each found
alone against a copy of the molecule that carries only that atom's labels, and it asks for
integer locants only. A molecule whose labels sit in several parts of the name then fails in
three ways: labels on symmetry-equivalent atoms all choose the same locant, a label inside a
multiplied substituent that holds only part of the label set has no scope of its own, and a
label on a ring of an assembly or on a fusion atom needs a primed or lettered locant. The
Blue Book name of such a compound is one descriptor per labelled part.

 NAMES,: "The name of an isotopically substituted compound is formed by adding
or inserting the nuclide symbol(s) enclosed in parentheses, preceded by any necessary
locant(s), letters, and/or numerals, before the part of the compound that is isotopically
substituted... When two or more nuclide symbols appear at the same place in the name they are
cited first in alphabetical order and then by their mass number, if necessary (see ";
its preferred names carry several independently scoped descriptors in one name
('1-(13C)methyl(2-13C)benzene (PIN)') and one descriptor per unit of a multiplied group
('1,2-di[(13C)methyl]benzene (PIN)'). Different isotopic modifications on
otherwise identical substituents: "When two substituent groups are isotopically modified in
different ways so that they cannot be combined together using multiplicative terms such as
'di-', 'bis-', etc., they are cited separately.": "In a name consisting of two or
more words, the isotopic descriptor is placed before the appropriate word or part of the word
that includes the nuclide(s)". ORDER OF NUCLIDE SYMBOLS, and: the
symbols of one place are alphabetical, then by increasing mass number. Omission of
locants,: "if isotopic modification requires a locant to specify its position,
then all locants must be specified and none are omitted";: "Locants are omitted
in compounds or substituent groups in which all positions are completely isotopically
substituted or modified in the same way". Multiple types of enclosing marks,
: "the parentheses used to indicate... specific isotopic substitution... are
taken into consideration" in the nesting order.

The composer binds each label to a slot of the name from OPSIN's own reading, not from a
producer. OPSIN writes the atoms of one name in an order that does not depend on the isotope
descriptors cited in it, so the heavy atoms of the parse of the bare skeleton are a frame
every probe of that skeleton shares, and one isomorphism of that frame onto the input names
each frame atom as an atom of the input. A probe, one label cited at one slot with one locant
token, is parsed, and the atom it labels (one per copy, inside a multiplied group) is read in
the frame: that binds the slot and the token to atoms of the input. The labelled atoms of the
input are then covered by probes, innermost slot first and lowest locants first, and every
cover is spelled with one descriptor per slot and read back whole through
``rules.isotopes._isotope_round_trips`` (canonical SMILES and the full InChIKey with its
isotope layer). Nothing but a name that OPSIN itself reads, and reads as the input, is
returned.

A unit that holds every position of its nuclide is cited without locants,
'(2H5)phenyl', '(2H9)tert-butyl'); its piece labels all the positions of the unit at once. When the
skeleton as the engine names it has no cover, the same search runs on its other spellings: the
multiplied groups cited once per copy; a group none of whose copies is labelled is
multiplied again in the name) and 'tert-butyl' as '2-methylpropan-2-yl'.

The composer is reached only when every strategy of the decorator has declined, and only at the
best-effort tier (its names are below the PIN, which every other tier declines), so no name the
decorator builds is changed. Its names are labelled below the PIN: the descriptors are
placed by what OPSIN reads, and a spelling the Blue Book prints for a part is not always the
one built.

Bounded and deterministic: the OPSIN parses one call may spend are counted (``PARSE_BUDGET``),
the covers it reads back are counted (``COVER_CAP``), and every order is fixed. The bounds are
per call of the composer, which the decorator makes once for each labelled molecule or fragment
it names. The clock (``time.perf_counter``) is read only to fill ``LAST_STATS``, which no decision
reads.
"""
from __future__ import annotations

import re
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..data.chain_names import multiplying_prefix
from ..perception.substructure import ordered_matches
from . import isotopes as _iso

#: OPSIN parses the probes of one composer call may spend. Beyond it the probing stops and the
#: composer works with what it has read (a name is still gated by its own read back).
PARSE_BUDGET = 8_000
#: Covers of the labelled atoms read back for one spelling of the skeleton.
COVER_CAP = 32
#: Nodes of the cover search; it fails closed beyond.
_NODE_CAP = 20_000
#: Highest discrepancy (a cover's distance from the all-preferred cover) the search visits.
_DISCREPANCY_CAP = 6
#: Highest integer locant probed at a slot.
_TOKEN_LIMIT = 64
#: Highest numeral of a numbered heteroatom locant (N1, O4...) probed at a slot.
_NUMBERED_LETTER_CAP = 24
#: Highest number of atoms of one nuclide cited without locants at a slot (a completely labelled unit).
_COMPLETE_LIMIT = 32

#: What the last call spent, and what the first calls of the process spent (measurement only;
#: no decision reads them).
LAST_STATS: Dict[str, object] = {}
CALLS: List[Dict[str, object]] = []

Key = Tuple[int, str]                       # (mass number, element symbol) of a nuclide
Record = Tuple[Tuple[Key, int], ...]        # the nuclides, each with its atom count, of one position

#: A ring-assembly locant set before its stem ('-1,1':4'-biphenyl'): the slot of a descriptor
#: that labels the assembly. ``_iso._INTERIOR_LOCANT_STEM_RE`` admits neither primes nor
#: colons.
_ASSEMBLY_STEM_RE = re.compile(r"-(?=[\d,':]+-[a-z])")
#: The same locant set right after an opening mark, where it opens a substituent prefix:
#: '(9,9'-spirobi[fluoren]-4-yl)'. A descriptor that labels the primed component goes before the
#: set, '[(3',4'-2H2)-9,9'-spirobi[fluoren]-4-yl]', as for a parent, the Blue Book: "before the
#: part of the compound that is isotopically substituted"); the unprimed
#: component's own slot inside 'spirobi[...]' reads only unprimed locants.
_PRIMED_STEM_AFTER_MARK_RE = re.compile(r"(?<=[(\[{])(?=\d+[a-z]?'*(?:,\d+[a-z]?'*)*'(?:,\d+[a-z]?'*)*-[a-z])")


# ---------------------------------------------------------------------------
# The labelled atoms of a molecule
# ---------------------------------------------------------------------------


def _heavy_view(mol: Chem.Mol) -> Optional[Tuple[Chem.Mol, List[Tuple[Key, int]]]]:
    """``(label-free heavy-atom molecule, [(nuclide, heavy index of the position)])`` of ``mol``.

    The heavy index of an atom is its place among the atoms of ``mol`` that are not hydrogen,
    in the order of ``mol``. The position of a labelled hydrogen is its heavy neighbour (the atom
    whose locant a descriptor cites), that of a labelled heavy atom the atom itself. None when
    a label is on a hydrogen that is not bonded to exactly one atom, or the heavy atoms do not
    keep their order when the hydrogens go."""
    heavy: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            heavy[atom.GetIdx()] = len(heavy)
    labels: List[Tuple[Key, int]] = []
    for atom in mol.GetAtoms():
        iso = atom.GetIsotope()
        if not iso:
            continue
        if atom.GetAtomicNum() == 1:
            nbrs = atom.GetNeighbors()
            if len(nbrs) != 1 or nbrs[0].GetIdx() not in heavy:
                return None
            labels.append(((iso, "H"), heavy[nbrs[0].GetIdx()]))
        else:
            labels.append(((iso, atom.GetSymbol()), heavy[atom.GetIdx()]))
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        atom.SetIsotope(0)
    try:
        work = Chem.RemoveHs(work)
    except Exception:  # noqa: BLE001 - a molecule RDKit cannot sanitise is not composed
        return None
    if work.GetNumAtoms() != len(heavy):
        return None
    return work, labels


class _Model:
    """The labelled atoms of the input.

    ``need[p]`` is the record of the labelled position ``p`` (a heavy index of the input): each
    nuclide that sits there with its number of atoms (a CD3 is one position with three
    deuterium atoms)."""

    def __init__(self, original: Chem.Mol):
        view = _heavy_view(original)
        if view is None:
            raise ValueError("labels not on single-bonded atoms")
        self.heavy, labels = view
        self.original = original
        heavy_atoms = [a.GetIdx() for a in original.GetAtoms() if a.GetAtomicNum() != 1]
        self.heavy_atoms = heavy_atoms
        per_pos: Dict[int, Dict[Key, int]] = {}
        for key, pos in labels:
            per_pos.setdefault(pos, {})[key] = per_pos.get(pos, {}).get(key, 0) + 1
        self.need: Dict[int, Record] = {p: tuple(sorted(c.items())) for p, c in per_pos.items()}
        self.maxpos: Dict[Tuple[int, Key], int] = {}
        for pos, counts in per_pos.items():
            for key in counts:
                self.maxpos[(pos, key)] = _iso._max_atoms_at_position(
                    _carrier(original, key, heavy_atoms[pos]))
        self.keys: List[Key] = sorted({key for c in per_pos.values() for key in c},
                                      key=lambda k: (k[1], k[0]))
        self.base_key = Chem.MolToSmiles(self.heavy)
        #: the symmetry class of each heavy atom (equal ranks are atoms the graph cannot tell apart)
        self.cls = list(Chem.CanonicalRankAtoms(self.heavy, breakTies=False))
        ring_info = original.GetRingInfo()
        self.fusion = any(ring_info.NumAtomRings(heavy_atoms[p]) >= 2 for p in per_pos)

    def denotable(self, element: str) -> set:
        """Heavy indices of the atoms an ``element`` letter can denote (``rules.isotope_letters``)."""
        from .isotope_letters import denotable_atoms
        ours = set(denotable_atoms(self.original, element))
        return {p for p, a in enumerate(self.heavy_atoms) if a in ours}


def _bare_letter_ok(model: _Model, element: str, positions) -> bool:
    """Whether ``element``'s letter may stand bare for the atoms of one unit, ``positions`` (heavy
    indices of the input): they are one symmetry class (``rules.isotope_letters``, the decorator's
    predicate)."""
    from .isotope_letters import bare_letter_allowed
    return bare_letter_allowed(model.original, element, [model.heavy_atoms[p] for p in positions])


def _carrier(mol: Chem.Mol, key: Key, idx: int) -> Chem.Atom:
    """One labelled atom of nuclide ``key`` at position ``idx`` (the atom ``_max_atoms_at_
    position`` reads the capacity of the position from)."""
    atom = mol.GetAtomWithIdx(idx)
    if key[1] != "H":
        return atom
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() == 1 and nb.GetIsotope() == key[0]:
            return nb
    return atom


# ---------------------------------------------------------------------------
# The skeleton with its multiplied groups cited apart
# ---------------------------------------------------------------------------

#: A multiplied group: its locants (optional), the multiplying prefix, then a simple prefix word
#: ('2,4-dimethyl', '4,11,18-tri-tert-butyl', "N,N'-dimethyl", 'dimethylamino') or an enclosed
#: group ('3,4,5-tris(acetyloxy)', 'N,N-di([1,1'-biphenyl]-4-yl)', 'bis(2-chloroethyl)').
_GROUP_RE = re.compile(
    rf"(?:^|(?<=[-(\[{{ )\]}}]))(?:(?P<locs>{_iso._GROUP_LOCANT}(?:,{_iso._GROUP_LOCANT})+)-)?"
    r"(?:(?P<cmult>bis|tris|tetrakis|pentakis|hexakis|di|tri|tetra|penta|hexa)(?=[(\[{{])"
    r"|(?P<mult>di|tri|tetra|penta|hexa)(?P<word>(?:-(?:tert|sec)-)?[a-z]+))")
_COMPOUND_COUNT = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6,
                   "bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6}
#: Bases that are not substituent prefixes of a multiplied group ('2,3-dihydro').
_NOT_A_GROUP_BASE = frozenset({"hydro"})


def _group_spans(skeleton: str) -> List[Tuple[int, int, Tuple[str, ...]]]:
    """``(start, end, texts)`` of every multiplied group of ``skeleton``: the text of the group
    cited apart in each of the spellings tried (``texts``), outermost group first."""
    from ..assembly.naming_utils import _SIMPLE_PREFIX_VOCABULARY, matching_close
    spans: List[Tuple[int, int, Tuple[str, ...]]] = []
    for m in _GROUP_RE.finditer(skeleton):
        locs = m.group("locs").split(",") if m.group("locs") else []
        if m.group("mult"):
            copies = _COMPOUND_COUNT[m.group("mult")]
            if locs and len(locs) != copies:
                continue
            word = m.group("word")
            stem = re.match(r"-(tert|sec)-", word)
            lead = f"{stem.group(1)}-" if stem else ""
            rest = word[stem.end():] if stem else word
            base = next((rest[:k] for k in range(len(rest), 0, -1)
                         if rest[:k] in _SIMPLE_PREFIX_VOCABULARY), None)
            if base is None or base in _NOT_A_GROUP_BASE:
                continue
            group, end = lead + base, m.start("word") + (stem.end() if stem else 0) + len(base)
            wrapped, kind = f"({group})", "simple"
        else:
            copies = _COMPOUND_COUNT[m.group("cmult")]
            if locs and len(locs) != copies:
                continue
            close = matching_close(skeleton, m.end(), "([{")
            if close is None:
                continue
            group, end = skeleton[m.end():close + 1], close + 1
            wrapped, kind = group, "complex"
        if locs:
            texts = ("-".join(f"{loc}-{group}" for loc in locs),)
        elif copies >= 3:
            # without locants, one copy and the others multiplied: '(13C)methylpentamethylbenzene'
            reduced = group + multiplying_prefix(copies - 1, kind) + group
            texts = (reduced, wrapped * copies, group * copies)
        else:
            # (heading '', the Blue Book): "For mononuclear parent hydrides with
            # two or more substituents the first cited substituent never has enclosing marks unless it
            # includes a locant. The second and further substituents are each enclosed with parentheses even
            # for simple substituents." A copy that carries a descriptor includes a locant, and its marks
            # step up: '[(4-2H)phenyl](phenyl)methanol', not '(4-2H)phenylphenylmethanol'.
            # The copies enclosed come first; the bare spelling stays as the second one.
            texts = (wrapped * copies, group * copies)
        spans.append((m.start(), end, tuple(dict.fromkeys(texts))))
    spans.sort(key=lambda sp: (sp[0], -sp[1]))
    kept: List[Tuple[int, int, Tuple[str, ...]]] = []
    for span in spans:
        if not kept or span[0] >= kept[-1][1]:       # not inside a group already cited apart
            kept.append(span)
    return kept


def _apart_variants(skeleton: str, stripped: Chem.Mol) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """The spellings of ``skeleton`` with every multiplied group cited once per copy ('1,4-dimethyl'
    as '1-methyl-4-methyl', '3,4-bis(acetyloxy)' as '3-(acetyloxy)-4-(acetyloxy)', 'dimethylamino'
    as 'methylmethylamino' or '(methyl)(methyl)amino'), each with the ``(start, end, text)`` of its
    groups as they stood in ``skeleton``, in the coordinates of the new name. Only a spelling
    that OPSIN reads as the molecule without labels is returned.

     Different isotopic modifications on otherwise identical substituents: "When two
    substituent groups are isotopically modified in different ways so that they cannot be
    combined together using multiplicative terms such as 'di-', 'bis-', etc., they are cited
    separately." A group that holds only some of its copies' labels, or labels of several
    kinds, has no scope of its own while its copies are multiplied; cited apart each copy is a
    slot. A group cited apart that OPSIN does not read as the molecule (a word boundary that is
    not one) is left as it was."""
    kept = _group_spans(skeleton)
    if not kept:
        return []

    def build(chosen: Sequence[Tuple[int, int, str]]) -> Tuple[str, List[Tuple[int, int, str]]]:
        out, groups, delta = skeleton, [], 0
        for start, end, text in chosen:
            groups.append((start + delta, start + delta + len(text), skeleton[start:end]))
            delta += len(text) - (end - start)
        for start, end, text in sorted(chosen, reverse=True):
            out = out[:start] + text + out[end:]
        return out, groups

    found: List[Tuple[str, List[Tuple[int, int, str]]]] = []
    for pick in range(max(len(texts) for _s, _e, texts in kept)):
        whole = build([(s, e, texts[min(pick, len(texts) - 1)]) for s, e, texts in kept])
        if all(whole[0] != f[0] for f in found) and _iso._isotope_round_trips(whole[0], stripped):
            found.append(whole)
    if found:
        return found
    good = []
    for s, e, texts in kept:
        for text in texts:
            if _iso._isotope_round_trips(build([(s, e, text)])[0], stripped):
                good.append((s, e, text))
                break
    return [build(good)] if good else []


#: The retained prefix 'tert-butyl' (and 'tert-butoxy'), with the hyphen that joins it to a multiplier
#: ('3,5-di-tert-butyl').
_TERT_BUTYL_RE = re.compile(r"(?:(?<=[a-z])-)?tert-but(?P<kind>oxy|yl)")


def _respell_tert_butyl(skeleton: str, stripped: Chem.Mol) -> Optional[Tuple[str, List[Tuple[int, int, str]]]]:
    """``skeleton`` with 'tert-butyl' written as '(2-methylpropan-2-yl)' ('tert-butoxy' as
    '(2-methylpropan-2-yl)oxy'), with the ``(start, end, text)`` of each spelling as it stood in
    ``skeleton``, in the coordinates of the new name; None when it has none or the spelling does not
    read back as the molecule without labels.

     (the Blue Book, "When the nuclide is located at a position in a retained name that
    is not numbered a systematic name that identifies separately the relevant atom is used for the
    IUPAC preferred name"): the retained prefix names no position of its methyl groups, the
    systematic prefix does."""
    out, groups, delta = skeleton, [], 0
    for m in _TERT_BUTYL_RE.finditer(skeleton):
        text = "(2-methylpropan-2-yl)" + ("oxy" if m.group("kind") == "oxy" else "")
        start = m.start() + delta
        groups.append((start, start + len(text), m.group(0)))
        delta += len(text) - (m.end() - m.start())
    if not groups:
        return None
    for m in reversed(list(_TERT_BUTYL_RE.finditer(skeleton))):
        out = out[:m.start()] + "(2-methylpropan-2-yl)" + ("oxy" if m.group("kind") == "oxy" else "") + out[m.end():]
    return (out, groups) if _iso._isotope_round_trips(out, stripped) else None


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


#: An edit that removes the character at its offset (the hyphen a unit's opening mark replaces).
_DELETE = "\x00"


class _Slot:
    """A place of the name where a descriptor can be spliced.

    ``pos`` is the offset of the descriptor in the skeleton. A slot of a multiplied simple
    prefix that the skeleton writes without marks ('dimethyl', 'di-tert-butyl') also encloses the
    prefix word ('di[(13C)methyl]', 'di[(2H9)tert-butyl]'; the marks are stepped up by the
    escalation of the assembly)."""

    __slots__ = ("idx", "pos", "unit", "join", "quality")

    def __init__(self, idx: int, pos: int, unit: Optional[Tuple[int, int, bool]], quality: int):
        self.idx = idx
        self.pos = pos
        # (offset of the opening mark, offset of the closing mark, whether the opening mark takes the
        # place of the hyphen at its offset) for a bare multiplied word
        self.unit = unit
        self.join = ""              # '-' when the descriptor stands in front of a locant digit
        self.quality = quality

    def edits(self, desc: str) -> List[Tuple[int, str]]:
        if self.unit is None:
            return [(self.pos, desc + self.join)]
        opening, closing, drop = self.unit
        return ([(opening, "(")] + ([(opening, _DELETE)] if drop else [])
                + [(self.pos, desc), (closing, ")")])

    def sort_key(self) -> tuple:
        """Innermost slot first: the descriptor stands "before the part of the
        compound that is isotopically substituted"): a slot directly before the part it
        labels, then the later slot."""
        return (self.quality, -self.pos, self.unit is None)


def _splice(skeleton: str, edits: Sequence[Tuple[int, str]]) -> str:
    """``skeleton`` with every ``(offset, text)`` inserted (the texts of one offset keep their
    order) and the character at the offset of every ``_DELETE`` removed."""
    by_off: Dict[int, List[str]] = {}
    for off, text in edits:
        by_off.setdefault(off, []).append(text)
    out: List[str] = []
    pos = 0
    for off in sorted(by_off):
        out.append(skeleton[pos:off])
        drop = False
        for text in by_off[off]:
            if text == _DELETE:
                drop = True
            else:
                out.append(text)
        pos = off + (1 if drop else 0)
    out.append(skeleton[pos:])
    return "".join(out)


def _candidate_offsets(skeleton: str) -> List[int]:
    """Offsets where a descriptor may be offered: the offsets of the decorator, the stem of a
    ring assembly written with primes and colons, and the start of every later word
    ."""
    offs = {0}
    offs.update(_iso._insertion_offsets(skeleton))
    offs.update(m.start() for m in _ASSEMBLY_STEM_RE.finditer(skeleton))
    offs.update(m.start() for m in _PRIMED_STEM_AFTER_MARK_RE.finditer(skeleton))
    offs.update(i for i in range(1, len(skeleton))
                if skeleton[i - 1] == " " and skeleton[i] != " ")
    return sorted(offs)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _other_alternation(a: Chem.Mol, b: Chem.Mol) -> bool:
    """True iff the heavy-atom molecules ``a`` and ``b`` are one compound drawn in two Kekule alternations
    (the same atoms and bonds without their orders, ``rules.isotopes._kekule_forms_of_one_compound``)."""
    try:
        return (_iso._bond_order_free_key(a) == _iso._bond_order_free_key(b)
                and _iso._kekule_forms_of_one_compound(a, b))
    except Exception:  # noqa: BLE001 - a frame that cannot be judged is no frame
        return False


#: Isomorphisms of the frame (the skeleton as OPSIN writes it) onto the input that are enumerated, and the
#: distinct ones (by what they say about the labels) that are tried, the first one found first.
_FRAME_MATCHES = 4096
_FRAME_LIMIT = 16


class _Prober:
    """Parses the probe names of one spelling of the skeleton and reads the atoms each labels,
    counting the parses. ``ok`` is False when the skeleton itself does not give a frame."""

    def __init__(self, model: _Model, skeleton: str, left: int):
        self.model = model
        self.left = left
        self.spent = 0
        self.exhausted = False
        self.dropped = 0
        self.ok = False
        self._memo: Dict[Tuple[str, Key], Optional[Tuple[int, ...]]] = {}
        self._frame_key = ""
        self._phi: Tuple[int, ...] = ()
        self.frames: List[Tuple[int, ...]] = []
        self._frame_of(skeleton)

    def _frame_of(self, skeleton: str) -> None:
        """The frame of ``skeleton``: the key of the order of its heavy atoms and the isomorphism of
        that frame onto the heavy atoms of the input (chirality kept when it can be)."""
        smi = self._parse(skeleton)
        mol = Chem.MolFromSmiles(smi) if smi else None
        view = _heavy_view(mol) if mol is not None else None
        if view is None or view[1]:
            return
        work = view[0]
        if Chem.MolToSmiles(work) == self.model.base_key:
            # ``work`` comes in OPSIN's atom order: matched in its canonical order (``ordered_matches``), or a
            # molecule with many symmetric leaf groups never returns
            matches = (ordered_matches(self.model.heavy, work, use_chirality=True, uniquify=False,
                                       max_matches=_FRAME_MATCHES)
                       or ordered_matches(self.model.heavy, work, uniquify=False, max_matches=_FRAME_MATCHES))
        elif _other_alternation(work, self.model.heavy):
            # OPSIN writes a conjugated ring RDKit does not perceive as aromatic in the other Kekule
            # alternation than the input's (the periphery of '10b,10c-dihydropyren-1-yl', a 4n+2 circuit,
            # ``rules.isotopes._same_fixed_h_inchi``): one compound, so the frame is the same, matched on
            # the graphs without bond orders.
            matches = ordered_matches(_iso._single_bond_graph(self.model.heavy), _iso._single_bond_graph(work),
                                      uniquify=False, max_matches=_FRAME_MATCHES)
        else:
            return
        frames = {}
        for match in matches:
            if len(match) != work.GetNumAtoms():
                continue
            # what a frame says about the labels is which atoms of the skeleton the labelled atoms of the
            # input are: two isomorphisms that agree on it give one name
            where = {atom: a for a, atom in enumerate(match)}
            pattern = tuple(sorted((where[p], record) for p, record in self.model.need.items()))
            frames.setdefault(pattern, tuple(match))
        if not frames:
            return
        self._frame_key = Chem.MolToSmiles(work, canonical=False)
        self.frames = list(frames.values())[:_FRAME_LIMIT]
        self.use_frame(0)
        self.ok = True

    def use_frame(self, index: int) -> None:
        """Read the labels through the ``index``-th isomorphism of the frame onto the input."""
        self._phi = self.frames[index]

    def _parse(self, text: str) -> Optional[str]:
        if self.left <= 0:
            self.exhausted = True
            return None
        self.left -= 1
        self.spent += 1
        return _iso._opsin_parse(text)

    def positions(self, text: str, key: Key) -> Optional[Tuple[int, ...]]:
        """The sorted positions (heavy indices of the input) the labels of ``text`` sit on, or None
        when OPSIN does not read ``text`` as the skeleton with labels of nuclide ``key`` only."""
        ident = (text, key)
        if ident not in self._memo:
            if self.left <= 0:
                self.exhausted = True
                return None
            read: Optional[Tuple[int, ...]] = None      # the atoms of the frame the labels sit on
            smi = self._parse(text)
            mol = Chem.MolFromSmiles(smi) if smi else None
            view = _heavy_view(mol) if mol is not None else None
            if view is not None and view[1] and all(k == key for k, _p in view[1]):
                if Chem.MolToSmiles(view[0], canonical=False) == self._frame_key:
                    read = tuple(p for _k, p in view[1])
                else:
                    self.dropped += 1
            self._memo[ident] = read
        read = self._memo[ident]
        return None if read is None else tuple(sorted(self._phi[p] for p in read))


def _probe_descriptor(key: Key, token: Optional[str]) -> str:
    mass, element = key
    return f"({'' if token is None else token + '-'}{mass}{element}1)"


#: A multiplier that takes 'tert-' or 'sec-' behind a hyphen ('3,5-di-tert-butyl').
_HYPHENATED_UNIT_RE = re.compile(
    r"(?:^|(?<=[-(\[{ )\]},]))(?:di|tri|tetra|penta|hexa)(?P<hyphen>-)(?P<lead>tert|sec)-(?P<word>[a-z]+)")


def _hyphenated_units(skeleton: str) -> List[Tuple[int, Tuple[int, int, bool], int]]:
    """The slots of a multiplied 'tert-butyl' or 'sec-butyl' ('3,5-di-tert-butylphenyl'): the book writes
    the group in marks when it carries a descriptor, '1,2-di[(13C)methyl]benzene (PIN)',
    , and the hyphen of the skeleton gives way to the opening mark."""
    from ..assembly.naming_utils import _SIMPLE_PREFIX_VOCABULARY
    out: List[Tuple[int, Tuple[int, int, bool], int]] = []
    for m in _HYPHENATED_UNIT_RE.finditer(skeleton):
        word = m.group("word")
        base = next((word[:k] for k in range(len(word), 0, -1) if word[:k] in _SIMPLE_PREFIX_VOCABULARY), None)
        if base is None:
            continue
        opening = m.start("hyphen")
        closing = m.start("word") + len(base)
        out.append((opening + 1, (opening, closing, True), 0))
    return out


def _slots_of(skeleton: str, prober: _Prober) -> List[_Slot]:
    """The slots of ``skeleton`` where a locant-free probe of some nuclide of the input
    parses (the slots a Blue Book descriptor can use; the mid-word offsets fall away)."""
    found: List[_Slot] = []
    keys = prober.model.keys
    raw: List[Tuple[int, Optional[Tuple[int, int]], int]] = []
    for off in _candidate_offsets(skeleton):
        raw.append((off, None, _iso._placement_quality(skeleton, off)))
    for _copies, head, base, tail, enclosed in _iso._multiplied_unit_slots(skeleton):
        if enclosed:
            continue                     # a slot inside marks is a plain slot, read per copy
        for o in _iso._unit_descriptor_offsets(base):
            raw.append((len(head) + o, (len(head), len(head) + len(base), False), 0))
    raw.extend(_hyphenated_units(skeleton))
    seen = set()
    for off, unit, quality in raw:
        if (off, unit) in seen:
            continue
        seen.add((off, unit))
        slot = _Slot(len(found), off, unit, quality)
        for join in (("-", "") if skeleton[off:off + 1].isdigit() and unit is None else ("",)):
            slot.join = join
            if any(prober.positions(_splice(skeleton, slot.edits(_probe_descriptor(k, None))), k)
                   for k in keys):
                found.append(slot)
                break
    for i, slot in enumerate(found):
        slot.idx = i
    return found


def _letter_elements(candidates: Sequence[str]) -> List[str]:
    """The element letters ('N', 'O') of the letter locants the decorator would offer
    (``rules.isotopes._letter_locant_candidates``, which says where a hydrogen nuclide or an
    oxygen nuclide sits on a heteroatom that integer locants do not number); the composer offers
    each one bare and numbered, and decides there which it may cite."""
    return sorted({m.group(0) for c in candidates for m in [re.match(r"[A-Z][a-z]?", c)] if m})


def _token_key(token: str) -> tuple:
    """Order of the locants of one descriptor, (the Blue Book): "Italic capital and
    lower-case letter locants are lower than Greek letter locants, which, in turn, are lower than
    numerals"; a cited unit ('*n') first."""
    if token.startswith("*"):
        return (-1,)
    m = re.fullmatch(r"([A-Z][a-z]?)(\d*)", token)
    if m:
        return (0, m.group(1), int(m.group(2) or 0), 0)
    return (1,) + _iso._locant_sort_key(token)


def _prime_depth(skeleton: str) -> int:
    runs = re.findall(r"'+", skeleton)
    return max((len(r) for r in runs), default=0)


def _tokens_at(skeleton: str, slot: _Slot, key: Key, prober: _Prober, letters: Sequence[str],
               fusion: bool, depth: int, limit: int, deep: int = 0) -> Dict[str, Tuple[int, ...]]:
    """Every locant token that labels a position at ``slot`` with nuclide ``key``, with the
    positions it labels: integers, then primed integers up to the prime depth of the skeleton,
    then ``Na`` / ``Nb`` for a molecule that labels a fusion atom, then, for each element letter of
    ``letters`` ('N', 'O'), the bare letter and the numbered forms ('O1', 'O4'); the bare letter is
    kept only where the unit holds no second atom the numbered forms reach.

    A unit ends at the first run of four positions that no label can sit on; a large bridged
    or fused parent has such runs inside it, and ``deep`` goes on past them, in steps of two
    (1), then position by position (2)."""
    found: Dict[str, Tuple[int, ...]] = {}

    def attempt(token: str) -> bool:
        text = _splice(skeleton, slot.edits(_probe_descriptor(key, token)))
        got = prober.positions(text, key)
        if got:
            found[token] = got
            return True
        return False

    def run(make, stop: int) -> int:
        """Probe ``make(k)`` for k = 1.. and return the highest k that parsed (0 when none
        did). The first positions of a fused or bridged parent have no hydrogen, so the search
        for the first parsing position goes on in steps of two (a run of parsing positions is
        never shorter), and once the unit is entered the search ends after four failures."""
        first = 0
        k = 1
        while k <= stop:
            if attempt(make(k)):
                first = k
                break
            k += 1 if k < 4 else 2
        if not first:
            return 0
        if first > 1:
            attempt(make(first - 1))
        top, fails, k = first, 0, first + 1
        while k <= stop:
            if attempt(make(k)):
                top, fails, k = k, 0, k + 1
                continue
            fails += 1
            if fails >= 4 and not deep:
                break
            k += 2 if fails >= 4 and deep == 1 else 1
        return top

    top = run(str, limit)
    bound = max(top, 6) + 2
    for primes in range(1, depth + 1):
        if not run(lambda k, p="'" * primes: f"{k}{p}", bound):
            break
    if fusion:
        for suffix in ("a", "b"):
            for k in range(1, top + 2):
                attempt(f"{k}{suffix}")
    for element in letters:
        bare = attempt(element)
        numbered: set = set()
        first = fails = 0
        for k in range(1, _NUMBERED_LETTER_CAP + 1):
            if attempt(f"{element}{k}"):
                numbered.update(found[f"{element}{k}"])
                first, fails = k, 0
            else:
                fails += 1
                if first and fails >= 8:
                    break
        # a letter names only an atom ``rules.isotope_letters.denotable_atoms`` names: a ring nitrogen
        # ('1H-indole') takes its ring numeral, never the letter
        denotable = prober.model.denotable(element)
        for token in [t for t in found if re.fullmatch(rf"{element}\d*", t)]:
            if any(p not in denotable for p in found[token]):
                del found[token]
        numbered = {p for p in numbered if p in denotable}
        bare = element in found
        if bare and not _bare_letter_ok(prober.model, element, numbered | set(found[element])):
            # (the Blue Book) "if isotopic modification requires a locant to specify its
            # position, then all locants must be specified and none are omitted", and
            # (:44202) "Locants are not omitted when there is a possibility of isomers": a unit that
            # holds atoms of more than one symmetry class the letter could denote needs the number
            # (``rules.isotope_letters``, the decorator's predicate); OPSIN reads the bare letter as the
            # first.
            del found[element]
    return found


def _complete_unit(skeleton: str, slot: _Slot, key: Key, prober: _Prober, reach: Dict) -> None:
    """The unit at ``slot`` cited without locants, completely labelled with nuclide ``key``.

     (the Blue Book, under ' Omission of locants'): "Locants are omitted in
    compounds or substituent groups in which all positions are completely isotopically substituted or
    modified in the same way": '(2H5)phenyl', '(2H9)tert-butyl'. OPSIN reads ``(2Hn)`` for n up to the
    number of positions the unit holds, so the greatest n it reads is the unit complete; the atoms it
    labels (one position per atom, n atoms in every copy of a multiplied group) are the positions the
    piece covers. A unit that holds more than ``_COMPLETE_LIMIT`` atoms has none."""
    got, last = None, 0
    per_n: Dict[int, Tuple[int, ...]] = {}
    for n in range(1, _COMPLETE_LIMIT + 1):
        desc = _iso.format_isotope_descriptor([(None, key[0], key[1], n)], force_show=True)
        cur = prober.positions(_splice(skeleton, slot.edits(desc)), key)
        if not cur:
            break
        got, last = cur, n
        per_n[n] = cur
    else:
        return
    if got is not None:
        reach[(slot.idx, f"*{last}", key)] = got
        # Fewer atoms than the unit holds, cited without locants, when the positions of the unit are one
        # symmetry class: (the Blue Book) "if isotopic modification requires a locant to
        # specify its position, then all locants must be specified and none are omitted", and none is
        # required when every atom the unit could carry the label on is the same position of the molecule
        #,:44202, "Locants are not omitted when there is a possibility of isomers": there is
        # none): '(2H1)' on one phosphono group, which holds two equivalent hydroxyl oxygens.
        if len({prober.model.cls[p] for p in got}) == 1:
            for n, cur in per_n.items():
                if n < last:
                    reach[(slot.idx, f"*{n}", key)] = cur


# ---------------------------------------------------------------------------
# Covering the labelled atoms
# ---------------------------------------------------------------------------


Piece = Tuple[int, str]                 # (slot index, locant token)
Chosen = Tuple[Piece, Tuple[int, ...]]  # a piece and the positions it labels
Reach = Dict[Tuple[int, str, Key], Tuple[int, ...]]


def _class_positions(model: _Model, record_of, position: int,
                     positions: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """The labelled atoms of the input a piece that labels ``positions`` (read through the frame
    isomorphism) labels modulo the symmetry of the molecule, with ``position`` among them: each
    position takes a labelled atom of its own class that carries the record ``record_of`` asks of it,
    ``position`` for the first atom of its class; None when the classes hold too few such atoms."""
    wanted = model.cls[position]
    taken: set = set()
    chosen: List[int] = []
    first_done = False
    for p in positions:
        record = record_of(p)
        pool = [q for q in sorted(model.need)
                if model.cls[q] == model.cls[p] and model.need[q] == record and q not in taken]
        if not first_done and model.cls[p] == wanted and position in pool:
            q, first_done = position, True
        elif pool:
            q = pool[0]
        else:
            return None
        taken.add(q)
        chosen.append(q)
    return tuple(sorted(chosen)) if first_done else None


def _options(model: _Model, reach: Reach, slots: List[_Slot], position: int,
             extras: bool = False) -> List[Tuple[Piece, Tuple[int, ...]]]:
    """The pieces that label ``position`` with every nuclide it carries, innermost slot first and
    lowest locants first (a unit cited without locants first of all), each with the positions it
    labels (one per copy of a multiplied group). Every position a piece labels must be a labelled
    position of the input that carries the same record; a unit cited without locants
    (``*n``, n atoms of one nuclide in every copy) labels every position of the unit, each with
    the atoms the unit puts there.

    After those come the pieces that label the position only modulo the symmetry of the molecule, and
    the ``*n`` that cites fewer atoms than its unit holds (a unit of one symmetry class,
    ``_complete_unit``): OPSIN puts a label cited once on one of several atoms the graph cannot tell
    apart (the hydroxyl oxygens of one phosphono group, the two methyls of one atom), and the frame
    isomorphism maps that atom onto another atom of its class than the input labels. The label sits
    where the input has it modulo the symmetry, and the read-back of the whole name decides, so those
    pieces take the input's atoms of the classes. They are offered only when ``extras`` (the composer's
    second pass, after no spelling of the skeleton had a cover without them), so no name found without
    them changes."""
    record = model.need[position]
    first = record[0][0]
    out: List[Tuple[tuple, Piece, Tuple[int, ...]]] = []
    extra: List[Tuple[tuple, Piece, Tuple[int, ...]]] = []
    holds: Dict[Tuple[int, Key], int] = {}      # the atoms a unit holds: its greatest ``*n``
    for (slot_idx, token, key) in reach:
        if token.startswith("*"):
            holds[(slot_idx, key)] = max(holds.get((slot_idx, key), 0), int(token[1:]))
    for (slot_idx, token, key), positions in reach.items():
        if key != first:
            continue
        exact = position in positions
        later = False
        if token.startswith("*"):
            counts: Dict[int, int] = {}
            for p in positions:
                counts[p] = counts.get(p, 0) + 1
            later = int(token[1:]) < holds[(slot_idx, key)]
            if exact and not later:
                if any(model.need.get(p) != ((key, n),) for p, n in counts.items()):
                    continue
                positions = tuple(sorted(counts))
            else:
                positions = _class_positions(
                    model, lambda p, counts=counts, key=key: ((key, counts[p]),), position,
                    tuple(sorted(counts)))
                if positions is None:
                    continue
                later = True
        else:
            if any(reach.get((slot_idx, token, other)) != positions for other, _n in record[1:]):
                continue
            if exact and all(model.need.get(p) == record for p in positions):
                pass
            else:
                positions = _class_positions(model, lambda p, record=record: record, position, positions)
                if positions is None:
                    continue
                later = True
        (extra if later else out).append(((slots[slot_idx].sort_key(), _token_key(token)),
                                          (slot_idx, token), positions))
    out.sort(key=lambda o: o[0])
    extra.sort(key=lambda o: o[0])
    return [(piece, positions) for _sk, piece, positions in out + (extra if extras else [])]


def _covers(model: _Model, reach: Reach, slots: List[_Slot], stats: Dict[str, int],
            extras: bool = False, mixed: bool = False) -> Iterator[Tuple[Chosen, ...]]:
    """Covers of the labelled atoms, nearest to the all-preferred cover first (limited
    discrepancy search: a cover that takes the k-th best piece for an atom costs k - 1), each
    cover once, in a fixed order."""
    items = sorted(model.need)
    opts = {p: _options(model, reach, slots, p, extras) for p in items}
    if any(not opts[p] for p in items):
        return
    yielded = set()

    def walk(done: frozenset, chosen: List[Chosen], budget: int):
        stats["nodes"] += 1
        if stats["nodes"] > _NODE_CAP:
            return
        item = next((p for p in items if p not in done), None)
        if item is None:
            sig = tuple(sorted(chosen))
            kinds: Dict[Tuple[int, Key], set] = {}
            for (slot_idx, token), positions in sig:
                # One nuclide is cited with its locants or without them in a slot, not both. With ``mixed`` a
                # slot may cite one nuclide with locants and another without ('(2-13C,15N)acetamido': the one
                # nitrogen of the unit needs no locant,, the carbon of it does); without it a slot
                # holds one form of citation for every nuclide, as before.
                for key, _n in (model.need[positions[0]] if mixed else ((None, 0),)):
                    kinds.setdefault((slot_idx, key), set()).add(token.startswith("*"))
            if sig not in yielded and not any(len(k) > 1 for k in kinds.values()):
                yielded.add(sig)
                yield sig
            return
        skipped = 0
        for piece, positions in opts[item]:
            if skipped > budget:
                break
            if any(p in done for p in positions):
                continue
            yield from walk(done | frozenset(positions), chosen + [(piece, positions)], budget - skipped)
            skipped += 1

    for budget in range(_DISCREPANCY_CAP + 1):
        yield from walk(frozenset(), [], budget)
        if stats["nodes"] > _NODE_CAP:
            return


# ---------------------------------------------------------------------------
# Spelling and assembling a cover
# ---------------------------------------------------------------------------


def _is_complete(skeleton: str, slot: _Slot, key: Key, count: int, prober: _Prober) -> bool:
    """ for ``count`` atoms of ``key`` cited without locants at ``slot``: the unit
    holds exactly that many positions, so the unit with one more atom does not read
    ("Locants are omitted in compounds or substituent groups in which all positions are
    completely isotopically substituted or modified in the same way"). Asked of OPSIN, as the
    decorator asks it of a unit (``rules.isotopes._unit_completely_labelled``)."""
    def read(n: int) -> bool:
        desc = _iso.format_isotope_descriptor([(None, key[0], key[1], n)], force_show=True)
        return bool(prober.positions(_splice(skeleton, slot.edits(desc)), key))
    return read(count) and not read(count + 1)


def _descriptor(model: _Model, pieces: Sequence[Tuple[str, Tuple[int, ...]]],
                force_show: bool, omit_locants: bool) -> str:
    """One descriptor for the pieces of a slot: each nuclide cites its locants once per atom,
    in order, and the nuclides follow (alphabetical, then increasing mass)."""
    by_key: Dict[Key, List[str]] = {}
    capacity: Dict[Key, int] = {}
    counted: Dict[Key, Tuple[int, int]] = {}      # a unit cited without locants: nuclide -> (atoms, capacity)
    groups = []
    for token, positions in pieces:
        if token.startswith("*"):
            key = model.need[positions[0]][0][0]
            cap = max(model.maxpos.get((p, key), 0) for p in positions)
            atoms, top = counted.get(key, (0, 0))
            counted[key] = (atoms + int(token[1:]), max(top, cap))      # two pieces of one nuclide: '(2H6)'
            continue
        for key, count in model.need[positions[0]]:
            by_key.setdefault(key, []).extend([token] * count)
            for p in positions:
                capacity[key] = max(capacity.get(key, 0), model.maxpos.get((p, key), 0))
    groups.extend((None, key[0], key[1], atoms, cap) for key, (atoms, cap) in counted.items())
    for key, tokens in by_key.items():
        if omit_locants:
            groups.append((None, key[0], key[1], len(tokens), capacity.get(key)))
        else:
            tokens = sorted(tokens, key=_token_key)
            groups.append((tuple(tokens), key[0], key[1], len(tokens), capacity.get(key)))
    return _iso.format_isotope_descriptor(groups, force_show=force_show)


def _assemble(skeleton: str, slots: List[_Slot], model: _Model, cover: Sequence[Chosen],
              force_show: bool, complete: frozenset) -> str:
    """The skeleton with the descriptors of ``cover`` spliced in, one per slot (a slot in
    ``complete`` cites no locants), the enclosing marks of every group that received one
    stepped up."""
    by_slot: Dict[int, List[Tuple[str, Tuple[int, ...]]]] = {}
    for (slot_idx, token), positions in cover:
        by_slot.setdefault(slot_idx, []).append((token, positions))
    inserts: List[Tuple[int, int, int, str, bool]] = []     # (offset, order, sequence, text, a descriptor)
    marks = set()
    for slot_idx in sorted(by_slot):
        slot = slots[slot_idx]
        desc = _descriptor(model, by_slot[slot_idx], force_show, slot_idx in complete)
        for off, text in slot.edits(desc):
            if slot.unit is not None and text in ("(", ")", _DELETE):
                if (off, text) in marks:
                    continue
                marks.add((off, text))
                # at one offset the closing mark of a unit comes first, then an opening mark
                inserts.append((off, {")": 0, "(": 1}.get(text, 2), len(inserts), text, False))
            else:
                inserts.append((off, 3, len(inserts), text, True))
    inserts.sort(key=lambda e: e[:3])
    parts: List[str] = []
    length = pos = 0
    starts: List[int] = []
    deletes = {off for off, _o, _q, text, _d in inserts if text == _DELETE}
    for off in sorted({e[0] for e in inserts}):
        parts.append(skeleton[pos:off])
        length += off - pos
        for _at, _order, _seq, text, is_desc in (e for e in inserts if e[0] == off):
            if text == _DELETE:
                continue
            if is_desc:
                starts.append(length)
            parts.append(text)
            length += len(text)
        pos = off + (1 if off in deletes else 0)
    parts.append(skeleton[pos:])
    out = "".join(parts)
    for at in starts:
        out = _iso._escalate_marks_for_descriptor(out, at)
    return out


def _complete_slots(skeleton: str, slots: List[_Slot], model: _Model, cover: Sequence[Chosen],
                    prober: _Prober) -> frozenset:
    """The slots of ``cover`` whose every nuclide group fills its unit (:func:`_is_complete`)."""
    counts: Dict[int, Dict[Key, int]] = {}
    for (slot_idx, token), positions in cover:
        if token.startswith("*"):
            continue
        for key, n in model.need[positions[0]]:
            counts.setdefault(slot_idx, {})[key] = counts.get(slot_idx, {}).get(key, 0) + n
    return frozenset(
        slot_idx for slot_idx, by_key in counts.items()
        if all(_is_complete(skeleton, slots[slot_idx], key, n, prober) for key, n in by_key.items()))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


class _Found:
    """A name that reads back, with what built it (to spell it again on a shorter skeleton)."""

    def __init__(self, name: str, slots: List[_Slot], cover: Sequence[Chosen],
                 force_show: bool, omit: frozenset, locants: tuple = ()):
        self.name, self.slots, self.cover, self.force_show, self.omit = name, slots, cover, force_show, omit
        self.locants = locants      # the locants cited, one per labelled atom, in increasing order


def compose_scopes(skeleton: str, original: Chem.Mol, label_map: Dict[int, int], n_pos: int,
                   budget: int = PARSE_BUDGET) -> Optional[str]:
    """A name of ``original``: ``skeleton`` with one descriptor per labelled part, read back
    whole; None when no cover of the labels reads back. Never raises."""
    started = time.perf_counter()
    stats: Dict[str, object] = {"parses": 0, "slots": 0, "covers": 0, "readbacks": 0, "dropped": 0,
                                "nodes": 0, "seconds": 0.0, "exhausted": False, "outcome": "none"}
    try:
        return _compose(skeleton, original, label_map, n_pos, budget, stats)
    except Exception as exc:  # noqa: BLE001 - a composer that cannot read the molecule declines
        stats["outcome"] = f"error:{type(exc).__name__}"
        return None
    finally:
        stats["seconds"] = round(time.perf_counter() - started, 2)
        LAST_STATS.clear()
        LAST_STATS.update(stats)
        if len(CALLS) < 8:
            CALLS.append(dict(stats))


def _compose(skeleton: str, original: Chem.Mol, label_map: Dict[int, int], n_pos: int,
             budget: int, stats: Dict[str, object]) -> Optional[str]:
    model = _Model(original)
    variants: List[Tuple[str, Optional[List[Tuple[int, int, str]]]]] = [(skeleton, None)]
    stripped = _iso.strip_isotopes(original)[0]
    for text, groups in _apart_variants(skeleton, stripped):
        if text != skeleton:
            variants.append((text, groups))
    respelled = _respell_tert_butyl(skeleton, stripped)
    if respelled is not None:
        variants.append(respelled)
        for text, groups in _apart_variants(respelled[0], stripped):
            if text != respelled[0]:
                variants.append((text, groups))
    stats["variants"] = len(variants)
    probers: Dict[str, _Prober] = {}
    spent = [budget]

    def search(spellings) -> Optional[str]:
        # Passes over the spellings: the second offers the pieces that label an atom modulo the symmetry of
        # the molecule (``_options``), the third and fourth the covers that cite one nuclide of a slot with
        # locants and another without, so no name an earlier pass finds changes.
        for mixed, extras in ((False, False), (False, True), (True, False), (True, True)):
            for text, groups in spellings:
                prober = probers.get(text)
                if prober is None:
                    prober = probers[text] = _Prober(model, text, spent[0])
                else:
                    prober.left, prober.spent, prober.dropped = spent[0], 0, 0
                try:
                    if not prober.ok:
                        stats["outcome"] = "no-frame"
                        continue
                    found = _lowest_locants(text, original, label_map, n_pos, model, prober, stats, extras, mixed)
                finally:
                    spent[0] = prober.left
                    stats["parses"] = int(stats["parses"]) + prober.spent
                    stats["dropped"] = int(stats["dropped"]) + prober.dropped
                    stats["exhausted"] = bool(stats["exhausted"]) or prober.exhausted
                if found is not None:
                    if groups is None:
                        stats["outcome"] = "named"
                        return found.name
                    stats["outcome"] = "named-apart"
                    return _merge_back((text, groups), found, model, original) or found.name
        return None

    return search(variants)


def _merge_back(apart: Tuple[str, List[Tuple[int, int, str]]], found: _Found, model: _Model,
                original: Chem.Mol) -> Optional[str]:
    """The name of ``found`` with the groups that received no descriptor spelled as in the skeleton
    again ('2,2-dimethyl', not '2-methyl-2-methyl'; 'tert-butyl', not '(2-methylpropan-2-yl)'), when
    that name reads back; None otherwise.

    A group cited apart is needed only where a copy carries a descriptor of its own,
    and the systematic spelling of a retained prefix only where a position of it is labelled
    ; a group none of whose copies is the unmodified group of the skeleton."""
    skeleton, groups = apart
    used = [off for slot in (found.slots[idx] for (idx, _t), _p in found.cover)
            for off, _text in slot.edits("")]
    back = [g for g in groups if not any(g[0] <= off < g[1] for off in used)]
    if not back:
        return None
    new_skeleton, delta, moved = skeleton, 0, []
    for start, end, text in back:
        new_skeleton = new_skeleton[:start - delta] + text + new_skeleton[end - delta:]
        moved.append((end, (end - start) - len(text)))
        delta += (end - start) - len(text)

    def shifted(off: int) -> int:
        return off - sum(shrink for end, shrink in moved if end <= off)

    slots = []
    for slot in found.slots:
        copy = _Slot(slot.idx, shifted(slot.pos), None if slot.unit is None else
                     (shifted(slot.unit[0]), shifted(slot.unit[1]), slot.unit[2]), slot.quality)
        copy.join = slot.join
        slots.append(copy)
    name = _assemble(new_skeleton, slots, model, found.cover, found.force_show, found.omit)
    return name if _reads_back(name, original) else None


def _reads_back(name: str, original: Chem.Mol) -> bool:
    """The gate of every name the composer returns: OPSIN itself reads ``name`` (a spelling OPSIN
    has no grammar for, read through its twin by the decorator's gate, is not returned) and the
    reading is the input with its isotopes (``rules.isotopes._isotope_round_trips``: canonical
    SMILES and the full InChIKey with its isotope layer)."""
    return _iso._opsin_parse(name) is not None and _iso._isotope_round_trips(name, original)


def _lowest_locants(skeleton: str, original: Chem.Mol, label_map: Dict[int, int], n_pos: int,
                    model: _Model, prober: _Prober, stats: Dict[str, object], extras: bool,
                    mixed: bool) -> Optional[_Found]:
    """The name of ``original`` on one spelling of the skeleton, under the isomorphism of its frame onto the
    input that cites the lowest locants.

     NUMBERING criterion (i) (the Blue Book): "the starting point and the direction of numbering
    of the analogous isotopically substituted compound are chosen so as to give lowest locants to the modified
    atoms or groups considered together as a set in increasing numerical order". A skeleton with a symmetry
    (a phenyl ring, two equivalent substituents) can be read onto the input in several ways, each citing other
    locants; the first isomorphism the name is found for is the lowest only by chance, so the others are tried
    when the first has a name, and a name replaces it only when it cites a strictly lower set of as many locants
    (a unit cited without locants because it holds every position of its nuclide is another spelling, not
    another numbering). Of readings with the same locants, the one whose senior nuclides sit in the prefixes of
    the lowest parent locants wins, (the Blue Book, "the senior parent structure that has the
    lowest locant(s) for nuclides of higher mass number": '4-(2-13C)ethyl-2-[2-(2-14C)ethylpentyl]heptan-1-ol',
    not the reading that puts 14C at 4), then the one with its descriptors earlier in the name: of two identical
    groups the modified one is cited first, (the Blue Book, "The isotopically modified
    substituent is preferred alphabetically to the unmodified substituent"). A name found on the first
    isomorphism is never changed otherwise."""
    best = None
    for index in range(len(prober.frames)):
        prober.use_frame(index)
        # every reading is tried: which one the matcher gives first is no property of the molecule, and a
        # reading of a symmetric skeleton can have a cover the first one has not (the two equally labelled
        # phenyls of '(2,3,4,5-2H4)phenyldi(2H5)phenylsilyl' fall into the 'di' unit under some readings only)
        found = _compose_on(skeleton, original, label_map, n_pos, model, prober, stats, extras, mixed)
        if found is not None and (best is None or (len(found.locants) == len(best.locants)
                                                    and _reading_key(found, model, skeleton)
                                                    < _reading_key(best, model, skeleton))):
            best = found
    prober.use_frame(0)
    return best


def _reading_key(found: _Found, model: _Model, skeleton: str) -> tuple:
    """What ``_lowest_locants`` compares two readings by: the locants cited, the parent locants of the nuclides
    from the senior one down, the offsets of the descriptors."""
    held = []
    for (slot_idx, _token), positions in found.cover:
        top = _parent_locants(skeleton, found.slots[slot_idx].pos)
        for p in positions:
            for (mass, symbol), _n in model.need[p]:
                held.append((-Chem.GetPeriodicTable().GetAtomicNumber(symbol), -mass, top))
    return found.locants, tuple(top for _z, _a, top in sorted(held)), _descriptor_offsets(found)


def _parent_locants(skeleton: str, pos: int) -> tuple:
    """The locants of the prefix of the parent (outside every enclosing mark) that ``pos`` lies in: (2,) for a
    slot of '2-[2-(...)ethylpentyl]',  for a prefix cited without locants."""
    depth, last, i = 0, (), 0
    while i < pos:
        c = skeleton[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and c.isdigit() and (i == 0 or skeleton[i - 1] in "-,)]} "):
            m = _PARENT_LOCANTS_RE.match(skeleton, i)
            if m:
                last = tuple(int(x) for x in re.findall(r"\d+", m.group(0)))
                i = m.end()
                continue
        i += 1
    return last


_PARENT_LOCANTS_RE = re.compile(r"\d+[a-z]?'*(?:,\d+[a-z]?'*)*-")


def _descriptor_offsets(found: _Found) -> tuple:
    """The offsets in the skeleton of the slots that hold the descriptors of ``found``, in increasing order."""
    return tuple(sorted(found.slots[slot_idx].pos for (slot_idx, _token), _positions in found.cover))


def _locants_cited(model: _Model, cover: Sequence[Chosen]) -> tuple:
    """The locants a cover cites, one per labelled atom, in increasing numerical order (a unit cited without
    locants cites none): the set NUMBERING criterion (i) (the Blue Book) compares, "the modified
    atoms or groups considered together as a set in increasing numerical order"."""
    cited = []
    for (_slot_idx, token), positions in cover:
        if not token.startswith("*"):
            atoms = sum(n for p in positions for _k, n in model.need[p])
            cited.extend([_token_key(token)] * atoms)
    return tuple(sorted(cited))


def _compose_on(skeleton: str, original: Chem.Mol, label_map: Dict[int, int], n_pos: int,
                model: _Model, prober: _Prober, stats: Dict[str, object],
                extras: bool = False, mixed: bool = False) -> Optional[_Found]:
    """The name of ``original`` on one spelling of the skeleton, or None."""
    slots = _slots_of(skeleton, prober)
    stats["slots"] = len(slots)
    letters = _letter_elements(_iso._letter_locant_candidates(original, label_map, n_pos, skeleton=skeleton))
    depth = _prime_depth(skeleton)
    limit = max(1, min(n_pos, _TOKEN_LIMIT))
    reach: Reach = {}
    ordered = sorted(slots, key=lambda s: s.sort_key())
    for slot in ordered:
        for key in model.keys:
            _complete_unit(skeleton, slot, key, prober, reach)
    for deep in (0, 1, 2):
        for slot in ordered:
            for key in model.keys:
                for token, positions in _tokens_at(skeleton, slot, key, prober, letters, model.fusion,
                                                   depth, limit, deep).items():
                    reach[(slot.idx, token, key)] = positions
        if all(_options(model, reach, slots, p, extras) for p in model.need):
            break
        stats["deep"] = True
    tried = set()
    cstats = {"nodes": 0}
    covers = 0
    try:
        for cover in _covers(model, reach, slots, cstats, extras, mixed):
            covers += 1
            complete = _complete_slots(skeleton, slots, model, cover, prober)
            for force_show, omit in ((False, complete), (False, frozenset()), (True, frozenset())):
                name = _assemble(skeleton, slots, model, cover, force_show, omit)
                if name in tried:
                    continue
                tried.add(name)
                stats["readbacks"] = int(stats["readbacks"]) + 1
                if len(tried) <= 3:
                    stats.setdefault("tried", []).append(name[:160])
                if _reads_back(name, original):
                    return _Found(name, slots, cover, force_show, omit, _locants_cited(model, cover))
            if covers >= COVER_CAP:
                break
        stats["outcome"] = "unread" if covers else "no-cover"
        return None
    finally:
        stats["covers"] = int(stats["covers"]) + covers
        stats["nodes"] = int(stats["nodes"]) + cstats["nodes"]


def compose_last_resort(sink: Dict[str, object], original: Chem.Mol,
                        label_map: Dict[int, int]) -> Optional[str]:
    """The composer on the skeleton ``sink`` holds (left there by the decorator when it
    declined), the name labelled below the PIN.

    The descriptors are placed by what OPSIN reads and not by the grammar of the name, and a
    spelling the Blue Book prints for a part is not always the one built, so the name is
    recorded as a non-PIN fragment (the default tier declines it; the best-effort tier ships
    it as a verified systematic name) and keeps the non-PIN record of the skeleton it is
    built on."""
    skeleton = sink.get("skeleton")
    if not isinstance(skeleton, str) or not skeleton:
        return None
    from ..metrics.provenance import (
        record_derived_non_pin_fragment, record_non_pin_fragment, restore_provenance)
    restore_provenance(sink["prov"])
    name = compose_scopes(skeleton, original, label_map, int(sink["n_pos"]))
    if name is None:
        return None
    record_derived_non_pin_fragment(skeleton, name)
    record_non_pin_fragment(name)
    return name
