"""The stereocentres a nuclide makes: their descriptors for any name of the decorator (IUPAC 2013,,.

 STEREOISOMERIC ISOTOPICALLY SUBSTITUTED COMPOUNDS: "Two types of stereoisomeric isotopically
substituted compounds are possible: (1) those in which the stereoisomerism results from isotopic
modification; (2) those whose analogous unmodified compounds are themselves stereoisomers."... "When they
must be inserted into the name at the same place as isotopic descriptors, the stereodescriptors are cited
first." ('(1R)-(1-2H1)ethan-1-ol (PIN)',.) NAMING OF STEREOISOMERS: "In preferred IUPAC names,
stereodescriptors are placed immediately at the front of the part of the name to which they relate. They are
placed at the front of the complete name when related to the parent structure; they are cited in
parentheses followed by a hyphen. When they relate to substituent groups, they are cited at the front of the
corresponding prefix." (the Blue Book)

A centre a nuclide makes (a CHD, a carbon beside a CH2D) is no stereocentre of the molecule without its
nuclides, so the skeleton of the decorator carries no descriptor for it, and every candidate of its strategies
reads back without that stereo and fails the full read-back. ``decorate_isotope_stereo`` finds the descriptor
for one skeleton spelling and a one-descriptor placement, by trying every locant and configuration of the
centres together; ``decorate_induced`` finds it for the name ANY strategy builds: the strategies run on the
molecule with those centres left unspecified, and the name that reads back is completed.

Where a descriptor stands: at the front of the part of the name its centre belongs to. For a centre
of a substituent that is the front of the innermost group that holds the isotope descriptor; for a centre of
the acid of an ester, of an anhydride or of a salt it is the front of the word that holds the isotope
descriptor, not the front of the complete name, which is the front of the first word and the part of the alkyl
group: 'ethyl (R)-4-nitrobenzene-1-sulfinate (PIN)' The chirality symbols 'R/S',
the Blue Book), 'methyl (S)-[methyl(phenyl)phosphinate] (PIN)' Phosphates, phosphonates, and
related compounds,:46022). The first spelling of the descriptor that reads back wins, innermost first. A
descriptor left at the front of a name whose centre is in a later word, or inside marks, is below the PIN: the
validity check of the placement (``validation/spelling/checks_placement.py``) declines it only when its locant
is higher than the positions of the first word, so it would pass 'ethyl' and 'butyl' and refuse 'methyl'.

Every centre is found alone, against the input with the chiral tags of the other induced centres removed
(the name carries no descriptor for those either), so the OPSIN reads one centre costs are ``scopes x
locants x 2`` and not their product; the descriptors then go into their scopes together and the name is read
back whole (canonical SMILES and the full InChIKey with its isotope layer, ``_isotope_round_trips``).
Nothing is returned that does not read back as the input.
"""
from __future__ import annotations

import re
from itertools import product
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from . import isotopes as _iso

#: the highest locant a stereodescriptor of an induced centre is tried at
_LOCANT_CAP = 64

#: an isotope descriptor: optional locants, then nuclide items ('(2H1)', '(1-2H1)', '(O-2H)', '(2,3-2H2,15N)')
_DESCRIPTOR_RE = re.compile(
    r"\((?:(?:[A-Z][a-z]?\d*'*|\d+[a-z]?'*)(?:,(?:[A-Z][a-z]?\d*'*|\d+[a-z]?'*))*-)?"
    r"\d+[A-Z][a-z]?\d*(?:,\d+[A-Z][a-z]?\d*)*\)")
_NUCLIDE_RE = re.compile(r"(\d+)([A-Z][a-z]?)\d*$")


def _element_numbers() -> Dict[str, int]:
    """Atomic number of every element symbol (a lookup that cannot make RDKit log a missing element)."""
    table = Chem.GetPeriodicTable()
    return {table.GetElementSymbol(z): z for z in range(1, 119)}


_ELEMENT_NUMBERS = _element_numbers()


def _is_nuclide(item: str) -> bool:
    """A mass number and an element symbol that can be a nuclide ('2H', '13C'), not a stereodescriptor
    that reads like one ('2S', '3Z')."""
    m = _NUCLIDE_RE.match(item)
    z = _ELEMENT_NUMBERS.get(m.group(2)) if m else None
    return z is not None and int(m.group(1)) >= z


# ---------------------------------------------------------------------------
# (BB:~43855) -- isotope-INDUCED stereocentre, one skeleton spelling, one-descriptor placement
# ---------------------------------------------------------------------------
def _cip_labeled_count(mol: Chem.Mol) -> int:
    """Number of atoms carrying an assigned CIP label under the project's
    canonical stereo path (``perception.stereo.assign_stereochemistry`` -- the
    ``centres`` engine reads isotope mass as CIP Rule 1b, confirmed by spike).

    The mol is RE-PARSED from its canonical SMILES first: an isotope-stripped copy
    keeps the ORIGINAL's chiral tag on a carbon that is no longer a stereocentre
    (two H's now identical), and the canonical writer drops exactly those phantom
    tags -- so the difference original-minus-stripped counts only the centres the
    nuclide actually induces. Fail-soft to 0 (a missing JVM must never hard-fail a
    name)."""
    try:
        m = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if m is None:
            return 0
        from ..perception.stereo import assign_stereochemistry
        assign_stereochemistry(m)
    except Exception:
        return 0
    return sum(1 for a in m.GetAtoms() if a.HasProp("_CIPCode"))


_STEREO_TOKEN_RE = re.compile(r"(\d*)([a-z]?)('{0,2})([RSrsEZez])$")


def _stereo_token_key(tok: str):
    """Total order over stereodescriptor tokens by ASCENDING locant
    'cited according to the ascending order of their corresponding locants').
    Key: (int locant, letter suffix e.g. 'a' in 7a, prime count, descriptor)."""
    m = _STEREO_TOKEN_RE.match(tok)
    if not m:
        return (float("inf"), "", 0, tok)
    num, letter, primes, desc = m.groups()
    return (int(num) if num else 0, letter, len(primes), desc)


def _merge_leading_stereo(induced, base: str) -> str:
    """: merge the isotope-INDUCED stereodescriptors with any genuine
    stereodescriptor group already leading ``base`` into ONE parenthesised group,
    cited in ascending locant order — ``(1S)-`` + ``(2R)-…`` -> ``(1S,2R)-…``.
    ``induced`` is a list of ``(int_locant, 'R'|'S')``. When ``base`` carries no
    leading stereo group the result is byte-identical to the old ``prefix+base``,
    so single-induced rows (``(1R)-(1-2H1)ethan-1-ol``) are unchanged. The isotope
    descriptor ``(1-2H1)`` is NOT a stereodescriptor and never matches
    ``_STEREO_PREFIX_RE``, so it is never swept into the merge."""
    from .stereochemistry import _STEREO_PREFIX_RE  # lazy: avoid import cycle
    m = _STEREO_PREFIX_RE.match(base)
    base_tokens, rest = [], base
    if m:
        base_tokens = m.group(0)[1:-2].split(",")  # strip '(' and ')-'
        rest = base[m.end():]
    induced_tokens = [f"{loc}{cfg}" for loc, cfg in induced]
    merged = sorted(base_tokens + induced_tokens, key=_stereo_token_key)
    return "(" + ",".join(merged) + ")-" + rest


def decorate_isotope_stereo(skeleton, keys, original, stripped, n_pos,
                            extra_locants) -> Optional[str]:
    """: an isotope-INDUCED stereocentre -- a centre that is a stereocentre
    only because a nuclide breaks a local symmetry (``(1R)-(1-2H1)ethan-1-ol``,
    achiral once the D is collapsed). The skeleton is named from the STRIPPED mol
    and carries no such descriptor, and the isotope descriptor alone reproduces
    the constitution but not the stereo, so the full-InChIKey oracle rejects it
    and the row abstains.

    Fully oracle-driven (no atom->locant map needed, consistent with the rest of
    this module):
      1. find the isotope-descriptor placement reproducing the CONSTITUTION
         (``stereo_blind`` -- e.g. ``(1-2H1)ethan-1-ol``);
      2. enumerate stereodescriptor prefixes over (locant-subset x R/S) for the
         ``k`` induced centres, cost-capped;
      3. accept the first (lowest-locant, deterministic) prefixed candidate whose
         FULL InChIKey (stereo + isotope) reproduces the original.

    : "the stereodescriptors are cited first" -- the prefix leads the part of the name its centre
    belongs to: the winner, found at the front of the complete name, is moved to the front of the word or
    group that holds the isotope descriptor when that reads back too (``cite_where_it_belongs``).
    Every emission is FULL-oracle gated, so a wrong config/locant fails closed;
    fails closed (None) when there is no induced centre, no constitution placement,
    the enumeration would exceed the cap, or nothing round-trips.
    """
    k = _cip_labeled_count(original) - _cip_labeled_count(stripped)
    if k < 1:
        return None
    base, _off, _desc, _lr = _iso._find_best_placement(
        skeleton, keys, original, n_pos, allow_front_hyphen=True,
        extra_locants=extra_locants, stereo_blind=True)
    if base is None:
        # The single-descriptor placement above cannot SPLIT a symmetry-broken
        # multiplied substituent -- e.g. the two iodomethyl arms of
        # 1,3-diiodopropan-2-ol, made distinct by one heavy-iodine nuclide, which
        # is exactly what induces the C2 centre. Obtain that de-multiplied
        # constitution stereo-blind (connectivity only) from the demux helpers so
        # it passes the constitution gate; the stereodescriptor prefix and the
        # full-InChIKey acceptance below still apply, so 0-wrong is preserved.
        for _demux_fn in (_iso._decorate_demultiplied,
                          _iso._decorate_uniform_multiplier,
                          _iso._decorate_letter_multiplier):
            base = _demux_fn(skeleton, keys, original, stripped, stereo_blind=True)
            if base is not None:
                break
    if base is None:
        return None
    from itertools import combinations, product
    from math import comb
    # Cost cap (bounded oracle budget): C(n_pos, k) * 2**k fresh OPSIN parses.
    # 24 covers the single-centre rows (2*n_pos) comfortably; a multi-centre
    # symmetric parent that blows it up fails closed rather than spend minutes.
    if comb(n_pos, k) * (2 ** k) > 24:
        return None
    winners = []
    for locs in combinations(range(1, n_pos + 1), k):
        for cfg in product("RS", repeat=k):
            #: merge the induced descriptors with any genuine
            # stereo group `base` already carries into ONE ascending-locant
            # group (`(1S,2R)-`), not two adjacent `(1S)-(2R)-` groups.
            cand = _merge_leading_stereo(list(zip(locs, cfg)), base)
            if _iso._isotope_round_trips(cand, original):  # FULL stereo + isotope
                winners.append((locs, cfg, cand))
    if not winners:
        return None
    # lowest-locant-first; deterministic tiebreak on the candidate string.
    winners.sort(key=lambda w: (w[0], w[1], w[2]))
    locs, cfg, front = winners[0]
    return cite_where_it_belongs(base, original, list(zip(locs, cfg)), front)


def descriptor_spans(name: str) -> List[Tuple[int, int]]:
    """``(start, end)`` of every isotope descriptor in ``name``."""
    return [(m.start(), m.end()) for m in _DESCRIPTOR_RE.finditer(name)
            if all(_is_nuclide(i) for i in m.group(0)[1:-1].rsplit("-", 1)[-1].split(","))]


def induced_centres(original: Chem.Mol) -> Optional[List[int]]:
    """The atoms (indices of ``original``) of the tetrahedral stereocentres the nuclides make: specified in
    ``original`` and not stereogenic once every nuclide is an ordinary atom. Empty when there are none; None
    when a stereo element of another kind (a double bond) is made by a nuclide too (it takes a descriptor
    this does not build)."""
    try:
        # The molecule without its nuclides collapses its (now ordinary) deuterium and tritium atoms into
        # hydrogen counts, as ``strip_isotopes`` does: an explicit hydrogen atom is a node of the graph the
        # stereo perception reads, so a CH2D left as an atom would still tell its carbon from a methyl.
        plain = Chem.Mol(original)
        for atom in plain.GetAtoms():
            atom.SetIntProp("_orig", atom.GetIdx())
            atom.SetIsotope(0)
        plain = Chem.RemoveHs(plain)

        def element(mol: Chem.Mol, info) -> tuple:
            """The stereo element in the numbering of ``original`` (removing the hydrogens renumbers
            ``plain``)."""
            def number(atom: Chem.Atom) -> int:
                return atom.GetIntProp("_orig") if atom.HasProp("_orig") else atom.GetIdx()
            if info.type == Chem.StereoType.Atom_Tetrahedral:
                return (info.type, number(mol.GetAtomWithIdx(info.centeredOn)))
            bond = mol.GetBondWithIdx(info.centeredOn)
            return (info.type, frozenset(number(a) for a in (bond.GetBeginAtom(), bond.GetEndAtom())))

        before = {element(plain, info) for info in Chem.FindPotentialStereo(plain)}
        out: List[int] = []
        for info in Chem.FindPotentialStereo(original):
            if info.specified != Chem.StereoSpecified.Specified or element(original, info) in before:
                continue
            if info.type != Chem.StereoType.Atom_Tetrahedral:
                return None
            out.append(info.centeredOn)
        return sorted(out)
    except Exception:  # noqa: BLE001 - a molecule RDKit cannot judge has no induced centre to name
        return []


def _without_chirality(mol: Chem.Mol, atoms) -> Chem.Mol:
    """``mol`` with the chiral tag of ``atoms`` removed."""
    work = Chem.RWMol(mol)
    for idx in atoms:
        work.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return work.GetMol()


def _word_start(name: str, pos: int) -> int:
    """The offset where the word of ``name`` that holds ``pos`` begins: a word ends at a space outside
    every enclosing mark."""
    depth = start = 0
    for i, ch in enumerate(name[:pos]):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == " " and depth == 0:
            start = i + 1
    return start


def _scopes(name: str) -> List[int]:
    """The offsets of ``name`` a stereodescriptor of an induced centre can stand at: the front of the
    part its centre belongs to): right inside the opening mark of every group that holds an isotope
    descriptor and the front of the word that holds it ('ethyl (2S)-2-phenyl(2-2H1)acetate'), the innermost
    first, then the front of the complete name."""
    pairs = _iso._enclosing_pairs(name)
    scopes: List[int] = []
    for start, end in descriptor_spans(name):
        for o, _c in sorted((p for p in pairs if p[0] < start and p[1] >= end), key=lambda p: -p[0]):
            if o + 1 not in scopes:
                scopes.append(o + 1)
        word = _word_start(name, start)
        if word and word not in scopes:
            scopes.append(word)
    scopes.sort(reverse=True)
    return scopes + [0]


def _front_is_misplaced(name: str) -> bool:
    """True when a descriptor cited at the front of the complete name is not at the front of the part its
    centre belongs to: an isotope descriptor of ``name`` stands inside enclosing marks or in a later word."""
    return any(_iso._bracket_depth(name, s) > 0 or _word_start(name, s) for s, _e in descriptor_spans(name))


def _cited(name: str, placed: Dict[int, List[Tuple[object, str]]]) -> str:
    """``name`` with the stereodescriptors ``placed`` (scope offset -> ``[(locant, 'R'|'S')]``) cited at the
    front of their scopes, merged into the group of stereodescriptors a scope already opens with, in
    ascending locant order: ``(2R,3R)-`` and ``5S`` give ``(2R,3R,5S)-``), the enclosing marks
    of a scope that gained a group stepped up."""
    for off in sorted(placed, reverse=True):
        name = name[:off] + _merge_leading_stereo(placed[off], name[off:])
    for off in sorted(placed):
        if off:
            name = _iso._escalate_marks_for_descriptor(name, off)
    return name


def complete_stereo(base: str, original: Chem.Mol, induced: List[int], n_pos: int) -> Optional[str]:
    """``base`` -- a name of ``original`` without the stereodescriptors of its induced centres -- with those
    descriptors cited, or None. A scope is tried innermost first, locants ascending, ``R`` before ``S``;
    the first that reads back, for each centre, is taken, and the name is read back whole."""
    scopes = _scopes(base)
    ring_info = original.GetRingInfo()
    fusion = any(ring_info.NumAtomRings(a) >= 2 for a in range(original.GetNumAtoms()))
    locants: List[object] = list(range(1, max(1, min(n_pos, _LOCANT_CAP)) + 1))
    if fusion:
        locants = [t for k in locants for t in (k, f"{k}a")]
    per_centre: List[List[Tuple[int, object, str]]] = []
    for centre in induced:
        alone = _without_chirality(original, [c for c in induced if c != centre])
        found: List[Tuple[int, object, str]] = []
        for off in scopes:
            hit = None
            for loc in locants:
                for cfg in "RS":
                    cand = base[:off] + _merge_leading_stereo([(loc, cfg)], base[off:])
                    if off:
                        cand = _iso._escalate_marks_for_descriptor(cand, off)
                    if cfg == "R" and _iso._opsin_parse(cand) is None:
                        break                      # OPSIN reads no stereocentre at this locant
                    if _iso._isotope_round_trips(cand, alone):
                        hit = (off, loc, cfg)
                        break
                if hit:
                    break
            if hit:
                found.append(hit)
                if len(found) >= 2:                # one more scope, in case the first does not read whole
                    break
        if not found:
            return None
        per_centre.append(found)
    for pick in product(*per_centre):
        placed: Dict[int, List[Tuple[object, str]]] = {}
        for off, loc, cfg in pick:
            placed.setdefault(off, []).append((loc, cfg))
        out = _cited(base, placed)
        if _iso._isotope_round_trips(out, original):
            if all(off == 0 for off, _l, _c in pick) and _front_is_misplaced(base):
                # a descriptor of a substituent or of a later word cited at the front of the complete name: a
                # spelling below the PIN
                from ..metrics.provenance import record_non_pin_fragment
                record_non_pin_fragment(out)
            return out
    return None


def cite_where_it_belongs(base: str, original: Chem.Mol, tokens: List[Tuple[object, str]],
                          front: str) -> str:
    """The name ``front`` -- ``base`` with the stereodescriptors ``tokens`` cited at the front of the complete
    name, read back as ``original`` -- with them cited at the front of the part their centre belongs to, when
    that reads back: the innermost scope of ``_scopes`` at which all of ``tokens`` read back whole. ``front``
    when none does; it is then below the PIN if the centre is not in the first word or inside marks."""
    for off in _scopes(base):
        if off == 0:
            break
        out = _cited(base, {off: list(tokens)})
        if _iso._isotope_round_trips(out, original):
            return out
    if _front_is_misplaced(base):
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(front)
    return front


def decorate_induced(style: str, namer, original: Chem.Mol, induced: List[int]) -> Optional[str]:
    """The name of ``original`` when its induced centres have no descriptor in the name the strategies of the
    decorator build: those strategies on ``original`` with the centres unspecified, then
    :func:`complete_stereo`."""
    unspecified = _without_chirality(original, induced)
    bare, labels = _iso.strip_isotopes(unspecified)
    n_pos = bare.GetNumHeavyAtoms()
    return _iso._decorate_labelled(
        Chem.MolToSmiles(unspecified), style, namer, unspecified, bare, labels,
        finish=lambda base: complete_stereo(base, original, induced, n_pos))
