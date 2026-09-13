"""Isotopically substituted compound names (IUPAC +.

Fail-closed decorator layer. Orthonym's core pipeline strips isotope labels
(RDKit perception ignores GetIsotope for skeleton naming), so a labeled mol
would otherwise emit the UNLABELED name — a wrong PIN. This module runs as an
EARLY branch in Orthonym.name: it strips the labels, names the skeleton in
systematic style (locanted parents), then re-derives the isotopic descriptor
by an INTERNAL ORACLE (OPSIN-parse a candidate name; compare rdkit canonical
SMILES — with isotopes — to the original). If no candidate round-trips, it
returns None and the label is never mis-placed.

BB (the Blue Book): the nuclide symbol(s) in parentheses,
preceded by any necessary locant(s), are inserted before the isotopically
substituted part; polysubstitution count is a right subscript to the symbol.
BB /.4.2/.4.3 (the Blue Book-22232): lowest locants to modified
positions; then to higher atomic number; then to higher mass number.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def has_isotopes(mol: Optional[Chem.Mol]) -> bool:
    """True iff any atom of ``mol`` carries a non-zero isotope label."""
    if mol is None:
        return False
    return any(a.GetIsotope() != 0 for a in atoms_of(mol))


def strip_isotopes(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, int]]:
    """Return an isotope-cleared COPY of ``mol`` and a {atom_idx: mass} map.

    Atom indices are preserved by RWMol copy, so the returned map indexes the
    stripped mol AND the original identically. Used to name the skeleton and
    later to place isotopic descriptors at the correct positions.
    """
    copy = Chem.Mol(mol)
    label_map: Dict[int, int] = {}
    for atom in copy.GetAtoms():
        iso = atom.GetIsotope()
        if iso != 0:
            label_map[atom.GetIdx()] = iso
            atom.SetIsotope(0)
    # Deuterium/tritium ([2H]/[3H]) are EXPLICIT hydrogen atoms; clearing their
    # isotope leaves plain explicit [H] atoms, which change the canonical SMILES
    # ([H]C([H])([H])CO != CCO) and could perturb skeleton perception. Collapse
    # them back into the implicit H count so the skeleton is the true unlabeled
    # parent. RemoveHs is a no-op for heavy-atom labels (14C etc). label_map
    # keys index the ORIGINAL mol (element symbols are read there in Task 3), so
    # the index shift RemoveHs introduces on the returned copy is irrelevant —
    # the copy is used only for its canonical skeleton SMILES + heavy-atom count.
    try:
        copy = Chem.RemoveHs(copy)
    except Exception:
        pass
    return copy, label_map


# ---------------------------------------------------------------------------
# — isotopic descriptor formatting
# ---------------------------------------------------------------------------


def nuclide_symbol(mass: int, element: str) -> str:
    """Nuclide symbol with the mass number as a leading integer (ASCII form).

    Orthonym emits ASCII (no <sup>); OPSIN accepts the leading-digit form
    (probe: (2-14C)ethan-1-ol -> C([14CH3])O)..
    """
    return f"{mass}{element}"


def _max_atoms_at_position(atom: Chem.Atom) -> int:
    """Atoms of this element that could occupy ``atom``'s position in the
    UNMODIFIED parent -- the "polysubstitution at a single position"
    quantity that decides whether the count subscript is shown.

    A labelled HEAVY atom (13C, 18O, 15N, 131I,...) IS a skeleton position and a
    covalent position holds exactly one such atom -> 1. A labelled H (D/T) shares
    its single heavy neighbour with every other H that neighbour can bear, so the
    governing quantity is that neighbour's total-H capacity in the parent
    (``GetTotalNumHs(includeNeighbors=True)`` -- counts the D/T themselves, which
    RDKit keeps as explicit isotope-labelled H, plus any implicit H, so it reads
    the true parent complement directly off the ORIGINAL mol without a
    stripped-index remap). ``atom`` is indexed into the ORIGINAL (label-bearing)
    mol -- the ``label_map`` keys index there.
    """
    if atom.GetSymbol() == "H":
        nbrs = atom.GetNeighbors()
        if not nbrs:
            return 1
        return nbrs[0].GetTotalNumHs(includeNeighbors=True)
    return 1


def format_isotope_descriptor(groups, force_show: bool = False,
                              bracket: bool = False) -> str:
    """Build the parenthesized isotopic descriptor from grouped labels.

    ``bracket=True`` (FIX-F, / renders the SPECIFICALLY-LABELLED
    form ``[2H1]`` / ``[13C]`` in square brackets instead of the substituted
    form ``(2H1)`` / ``(13C)``. It is OPT-IN and OFF on the default single-call
    path by design: a labelled compound is STRUCTURALLY IDENTICAL to its
    substituted counterpart (``[13CH4]`` is one graph), so nothing in a bare SMILES
    selects between the two conventions -- the whole of chapter is therefore a
    permanent RIGHT_MOL_NONPIN ceiling under the default parenthetical form, not a
    reclaimable gap. The renderer exists and is testable so a caller that KNOWS it
    wants the labelled convention (e.g. a future explicit-mode API) can request it;
    the swap of the enclosing marks around a nested complex prefix is a
    documented extension point, not built here. See internal notes

    ``force_show=True`` restores the pre-FIX-A always-emit-the-subscript form
    regardless of ``max_at_pos``. The placement search uses it as a FALLBACK: the
    BB-preferred omitted form ``(2H)`` / ``(13C)`` is tried first, but OPSIN 2.9.0
    does not parse every omitted spelling in every position (e.g. an O-bound
    single ``(2H)`` before an ``-oic acid`` suffix), so when the omitted form does
    not round-trip the forced form ``(2H1)`` is tried before abstaining -- the
    right molecule at a non-preferred spelling beats silence (never a wrong name;
    every candidate stays OPSIN-RT gated).

    ``groups`` = list of (locant, mass, element, count[, max_at_pos]):
      locant None -> no leading locant (single-position parent / front
                      descriptor, e.g. (2H3)methoxybenzene, (12C)methane).
      locant int -> the count locants are repeated then hyphen-joined to the
                      nuclide, e.g. (2,2,2-2H3), (2-14C).
      max_at_pos -> OPTIONAL 5th field, the polysubstitution quantity
                      (:func:`_max_atoms_at_position`). Absent (legacy 4-tuple) ->
                      the count subscript is kept unconditionally (the pre-FIX-A
                      behaviour), so only a caller that supplies max_at_pos gets
                      the omission.

    Multiple groups at (possibly) the same place are cited alphabetically by
    element then by mass number /; groups are comma-joined.

     (the Blue Book-43720): the count subscript is shown "when polysubstitution
    at a single position is possible". So it is emitted iff ``count > 1`` (more
    than one atom of the nuclide is cited together) OR ``max_at_pos > 1`` (the
    single position could carry a second atom of that element) -- e.g. a lone D on
    a CH3 keeps ``(2H1)`` (3 H can occupy) but a lone 13C keeps no subscript
    ``(13C)`` (a carbon position holds one carbon) and a lone D on a CH keeps none
    ``(2H)`` (1 H can occupy). Verbatim witnesses: ``trichloro(12C)methane``,
    ``(2-13C)ethan-1-ol``, ``(2S)-(2-2H)butan-2-ol``.
    """
    def _one(group):
        locant, mass, element, count = group[0], group[1], group[2], group[3]
        max_at_pos = group[4] if len(group) > 4 else None
        sym = nuclide_symbol(mass, element)
        show_sub = force_show or count > 1 or max_at_pos is None or max_at_pos > 1
        sub = str(count) if show_sub else ""
        if locant is None:
            return f"{sym}{sub}"
        # BB: the locant is repeated once per substituted atom at that position
        # for a grouped multi-count token (2,2,2-2H3); for count 1 a single
        # locant + hyphen (2-14C).
        loc_part = ",".join(str(locant) for _ in range(count))
        return f"{loc_part}-{sym}{sub}"

    #: alphabetical by element symbol, then by mass number, then locant.
    ordered = sorted(groups, key=lambda g: (g[2], g[1], g[0] if g[0] is not None else -1))
    inner = ",".join(_one(g) for g in ordered)
    open_mark, close_mark = ("[", "]") if bracket else ("(", ")")
    return f"{open_mark}{inner}{close_mark}"


# ---------------------------------------------------------------------------
# / — internal-oracle isotopic-descriptor placement
# ---------------------------------------------------------------------------
# The skeleton name fixes the parent numbering, but Orthonym does NOT expose a
# reliable atom->locant map for acyclic parents (retrieve_confidence
# ['atom_to_locant'] is None for CCO). Rather than re-perceive the chain
# independently (which risks diverging from the name's own numbering — a
# fail-closed violation), the decorator treats the skeleton name's locant space
# as opaque and lets OPSIN adjudicate: it enumerates candidate descriptor
# placements, OPSIN-parses each candidate, and accepts the one whose relabeled
# structure is rdkit-canonical-identical (isotopes retained) to the original.
# Because OPSIN itself renumbers to give the isotope its lowest legal locant,
# the round-trip test IS the / (i) implementation — the
# wrong-locant candidate simply fails to reproduce the structure.

_ELEMENT_Z = {  # atomic numbers for tie-break
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def _opsin_parse(candidate_name: str) -> Optional[str]:
    """Parse ``candidate_name`` to SMILES via the repo's shared OPSIN CLI
    helper (validation.opsin_roundtrip.opsin_parse — the SAME loader the name
    validity gate uses). Fail-closed (None) on any error / no JAR."""
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        return opsin_parse(candidate_name)
    except Exception:
        return None


def _isotope_round_trips(candidate_name: str, original_mol: Chem.Mol,
                         stereo_blind: bool = False) -> bool:
    """True iff OPSIN parses ``candidate_name`` to a structure that is rdkit-
    canonical-identical (isotopes retained) to ``original_mol``. Fail-closed
    on parse failure / no JAR.

    ``stereo_blind=True`` compares CONSTITUTION only (stereochemistry stripped
    from both sides, InChIKey layer not consulted). Used by the isotope-induced
    stereocentre path (FIX-D, to find the descriptor placement that
    reproduces the connectivity before the stereodescriptor prefix is chosen --
    the full-stereo compare is then re-applied to the prefixed candidate, so no
    isotopologue or stereoisomer is ever shipped on a stereo-blind pass alone.
    """
    smi = _opsin_parse(candidate_name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        if stereo_blind:
            a, b = Chem.Mol(parsed), Chem.Mol(original_mol)
            Chem.RemoveStereochemistry(a)
            Chem.RemoveStereochemistry(b)
            return Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
        if Chem.MolToSmiles(parsed) != Chem.MolToSmiles(original_mol):
            return False
        # (a review P4 related note): RDKit CanonSmiles can call a charged /
        # aromatic multi-component pair equal while InChI distinguishes them (the
        # same weakness fixed in ``namer._rt_match`` — measured on a cyanine dye).
        # Back the isomeric-CanonSmiles compare with the full InChIKey (isotope
        # layer included) so a wrong isotopologue on such a species cannot pass.
        # Strictly tightening; fall back to the CanonSmiles match only when InChI is
        # genuinely unavailable for a side (no regression).
        from rdkit.Chem import inchi as _inchi
        _ik_p = _inchi.MolToInchiKey(parsed)
        _ik_o = inchikey_of(original_mol)
        if _ik_p and _ik_o:
            return _ik_p == _ik_o
        return True
    except Exception:
        return False


#: A parent hydride's indicated-hydrogen prefix, e.g. the ``1H-`` of
#: ``1H-pyrrole``. The hyphen is part of the match because the descriptor goes
#: *in place of* it -- see ``_insertion_offsets``.
_INDICATED_H_PREFIX_RE = re.compile(r"-\d+H-")

#: An indicated hydrogen at the START of a bracketed sub-name, e.g. the ``1H-`` of
#: ``(1H-indol-2-yl)…`` / ``[1H-inden-…]``. Here the ``\d+H-`` follows an OPENING
#: BRACKET, not a hyphen, so ``_INDICATED_H_PREFIX_RE`` misses it and the descriptor
#: slot (right after the bracket, before the digit) was never offered -- a labelled
#: fused-ring substituent (indol-2-yl carrying ¹³C) had nowhere legal to place its
#: descriptor and failed closed (a review review 2026-09-08).
_BRACKET_INDICATED_H_RE = re.compile(r"[(\[{]\d+H-")

#: An interior sub-parent introduced by its OWN locant set, e.g. the ``-1,4-`` of
#: ``3,6-dimethyl-1,4-dioxane-2,5-dione`` or the ``-1,3-`` of
#: ``2,4-dioxo-1,3-diazaspiro[4.7]dodecane``. A nuclide ON that sub-parent puts its
#: descriptor immediately before the sub-parent: "inserted before the
#: part... that is isotopically substituted"), replacing the introducing hyphen:
#: ``3,6-dimethyl`` + ``(2H2)`` + ``-1,4-dioxane-2,5-dione``. That slot begins with
#: a digit, so it was covered by neither the alphabetic-start rule nor the
#: indicated-H rules, and a labelled interior ring failed closed. More targeted
#: than "every hyphen before a digit": the locant set must itself be followed by a
#: hyphen and a stem letter (``[\d,]+-[a-z]``), so a plain substituent locant like
#: the ``-2-`` of ``prop-2-en`` is not offered. Every candidate stays OPSIN-RT
#: gated by the caller, so any surplus offset can only be discarded, never shipped.
_INTERIOR_LOCANT_STEM_RE = re.compile(r"-(?=[\d,]+-[a-z])")


def _insertion_offsets(skel: str) -> List[int]:
    """Offsets in ``skel`` at which an isotopic descriptor may be inserted.

    : the descriptor "is inserted before the part of the compound that is
    isotopically substituted". The enumeration does not parse the skeleton's
    grammar -- it offers every plausible offset and lets the OPSIN round-trip
    oracle pick the one that is right -- so the only thing that matters here is
    that no legal placement is left out of the candidate set.

    Alphabetic starts cover a bare parent stem. They do NOT cover a parent
    carrying an indicated hydrogen: the slot before ``1H-pyrrole`` begins with a
    digit, so a labelled molecule that also has substituent prefixes had nowhere
    legal to put the descriptor and failed closed -- ``[2H]C1=C(N(C=C1)C)[N+](=O)[O-]``
    named nothing at all, though its unlabelled skeleton names fine.

     prints the construction four times, every one a PIN
    (``the Blue Book Blue Book``-``:43796``)::

        (15N)-1H-indole (PIN)
        2,3-dihydro(15N)-1H-indole (PIN)
        2,3-dihydro(2,3-2H2,15N)-1H-indole (PIN)

    -- descriptor, then a hyphen, then the indicated-hydrogen prefix. So the
    descriptor replaces the hyphen that introduces that prefix, and the offset to
    offer is the hyphen's own index: ``1-methyl-2-nitro`` + ``(3-2H1)`` +
    ``-1H-pyrrole``.

    A third gap is an interior sub-parent introduced by its own locant set
    (``-1,4-dioxane``): a nuclide on that ring needs the slot before its locant
    set, which begins with a digit -- see ``_INTERIOR_LOCANT_STEM_RE``.

    Every candidate built from these offsets is still OPSIN-round-trip gated by
    the caller, so a surplus offset can only ever be discarded, never shipped.
    """
    offsets = [i for i in range(1, len(skel)) if skel[i].isalpha()]
    offsets.extend(m.start() for m in _INDICATED_H_PREFIX_RE.finditer(skel))
    # A bracketed indicated-H opening: offer the slot right AFTER the bracket
    # (before the digit), so '(1H-indol-2-yl)…' can take '((desc)-1H-indol-2-yl)…'
    # -> escalated to '[(desc)-1H-indol-2-yl]…'.
    offsets.extend(m.start() + 1 for m in _BRACKET_INDICATED_H_RE.finditer(skel))
    # An interior sub-parent introduced by its own locant set: offer the slot at
    # the introducing hyphen so a nuclide on that ring can place its descriptor
    # (e.g. '3,6-dimethyl' + '(2H2)' + '-1,4-dioxane-2,5-dione'). RT-gated.
    offsets.extend(m.start() for m in _INTERIOR_LOCANT_STEM_RE.finditer(skel))
    return sorted(set(offsets))


def _p4542_p4543_key(groups) -> tuple:
    """ then ordering key for a candidate placement.

    Prefer the placement giving the lowest locant to the HIGHER atomic number
    , then to the HIGHER mass number. Implemented as: for
    each group sorted by locant ascending, emit (-Z, -mass); the lexicographically
    smallest key is the preferred placement. the Blue Book / the Blue Book / the Blue Book(i).
    """
    key = []
    for group in sorted(
        groups, key=lambda g: (g[0] if g[0] is not None else 1 << 30)
    ):
        # groups are (locant, mass, element, count[, max_at_pos]) -- index the
        # first three so a 4- or 5-tuple both unpack.
        _locant, mass, el = group[0], group[1], group[2]
        z = _ELEMENT_Z.get(el, 0)
        key.append((-z, -mass))
    return tuple(key)


def _bracket_depth(skel: str, off: int) -> int:
    """Enclosing-mark nesting depth at ``off`` in ``skel`` -- the count of ``(``
    or ``[`` opened before ``off`` and not yet closed. Depth 0 means the
    descriptor modifies the PARENT hydride; depth > 0 means it is nested inside a
    SUBSTITUENT's own enclosing marks (``(1,1,2,2,2-pentafluoro(13C1)ethyl)…``).
    The parent-scope locant gate applies only at depth 0 -- a
    substituent-scoped descriptor is a separate rule this fix leaves
    untouched (PREP-T3: the ``(13C1)ethyl`` rows are out of scope)."""
    head = skel[:off]
    return (head.count("(") - head.count(")")
            + head.count("[") - head.count("]"))


def _placement_quality(skel: str, off: int) -> int:
    """ placement rank for an isotope descriptor spliced at ``off``:
    ``0`` when the descriptor sits directly before the AFFIX / parent name it
    modifies (an alphabetic character, or the ``-nH-`` indicated-hydrogen
    prefix), ``1`` when it is DETACHED from that affix by an intervening locant
    (or stereo prefix) at the front of the name.

    OPSIN parses both ``5-(81Br)bromo…`` (rank 0) and the front-detached
    ``(81Br)5-bromo…`` (rank 1) to the same structure, so the round-trip oracle
    cannot separate them; the Blue Book uses only the former -- the Blue Book
    ``(2R)-1-(131I)iodo-3-iodopropan-2-ol`` is locant, then descriptor, then
    affix, with the stereo descriptor kept at the front. Ranking the detached
    form BELOW the adjacent one lets the correct placement win the offset
    tie-break without EXCLUDING the detached form, so a molecule whose only
    round-tripper is the detached form still emits rather than failing closed.
    (The legitimate front-of-parent hyphen-joined placement for a
    skeletal-replacement locant, ``_front_hyphen_candidate``, is scored 0
    explicitly by its caller -- it is the parent-scope placement.)"""
    rest = skel[off:]
    if rest[:1].isalpha():
        return 0
    if _INDICATED_H_PREFIX_RE.match(rest):
        return 0
    return 1


def _isotopomer_key(cand: str) -> Optional[str]:
    """The distinct-structure key OPSIN assigns a candidate name (isotope-aware
    InChIKey, SMILES fallback); None when it does not parse. Shared by the
    uniqueness oracle so the seed and the probes are compared at one layer."""
    smi = _opsin_parse(cand)
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        from rdkit.Chem import inchi as _inchi
        return _inchi.MolToInchiKey(m) or Chem.MolToSmiles(m)
    except Exception:
        return Chem.MolToSmiles(m)


def _placement_unambiguous(skel: str, off: int, keys, n_pos: int,
                           accepted_cand: str) -> bool:
    """: may the isotope descriptor's locant be OMITTED at this
    placement? True iff no OTHER isotopomer can be expressed at the same slot by
    stating a locant on ANY ONE nuclide group -- i.e. the omitted form denotes a
    unique structure.

    Verbatim (the Blue Book): "if isotopic modification requires a locant
    to specify its position, then all locants must be specified and none are
    omitted"; (:44202) "Locants are not omitted when there is a
    possibility of isomers". A locant-free descriptor that merely round-trips is
    not proof of uniqueness -- OPSIN's *default* placement can coincidentally
    match the true atom on an asymmetric parent (``3-(2H1)oxatricyclo…``).

    Two corrections over the prior shared-single-locant version (which spliced the
    SAME locant into EVERY nuclide group and so, for a MULTI-nuclide descriptor,
    built a form OPSIN rejects for all n -- ``(n-13C,n-15N)`` -- leaving ``seen``
    empty and FAILING OPEN via ``len(∅) <= 1``):
      1. SEED ``seen`` with the structure the accepted OMITTED-locant candidate
         actually denotes (``accepted_cand``, the string that just round-tripped to
         ``original`` in the caller). It must be the caller's exact candidate, not
         a re-rendered one: the omitted subscript form ``(2H)benzene`` parses to a
         DIFFERENT structure than the force-show ``(2H1)benzene`` the engine emits,
         so re-rendering the seed mis-anchored it (over-cited monodeuterobenzene).
         An empty probe set then means "no distinct alternative" -> unique -> omit
         (correct for ``(2H6)benzene``, ``(2H3)methanol``, ``(13C1)methyl
         acetate``), not fail-open.
      2. Probe each nuclide group INDEPENDENTLY (only that group carries the
         locant, the others stay None -- the mixed form ``(1-13C,15N)`` the builder
         must emit) in the force-show subscript spelling (maximally parseable; the
         InChIKey is spelling-independent). Any distinct isotopomer -> ambiguous ->
         keep the locant.

    For a single-nuclide descriptor this is the seed + a per-locant sweep, so
    single-nuclide omit/keep decisions are unchanged. Still NOT a whole-molecule
    symmetry count (the WITHDRAWN-2026-07-28 over-broadening trap): a
    substituent-scoped or 1-atom-parent descriptor stays trivially unique because
    OPSIN rejects an out-of-scope locant, adding nothing beyond the seed."""
    seed = _isotopomer_key(accepted_cand)
    seen: set = {seed} if seed else set()
    for gi in range(len(keys)):
        for n in range(1, n_pos + 1):
            groups_n = [(n if j == gi else None, mass, el, count, maxpos)
                        for j, ((mass, el), count, maxpos) in enumerate(keys)]
            desc_n = format_isotope_descriptor(groups_n, force_show=True)
            key = _isotopomer_key(skel[:off] + desc_n + skel[off:])
            if key is None:
                continue
            seen.add(key)
            if len(seen) > 1:
                return False
    return len(seen) <= 1


#: A parent hydride opening with a skeletal-replacement / added locant, e.g. the
#: ``3-`` of ``3-oxatricyclo…`` or ``1-`` of ``1-azabicyclo…``, optionally behind
#: leading stereo / configuration prefixes (``(2R,4S)-``, ``rel-``, ``rac-``). The
#: isotope descriptor for the WHOLE parent goes at the front, hyphen-joined to that
#: locant: "inserted before the part … isotopically substituted") --
#: ``(2R,4S)-(2-2H1)-3-oxatricyclo…``, NOT the mid-parent ``…-3-(2-2H1)oxa…`` a
#: pure alpha-offset splice yields. The ``\dH`` indicated-H opening is EXCLUDED --
#: ``_INDICATED_H_PREFIX_RE`` already reuses that hyphen.
_LEADING_STEREO_PREFIX_RE = re.compile(r"^(?:\([^)]*\)-|rel-|rac-|cis-|trans-)*")

# A leading locant is a SKELETAL-REPLACEMENT locant (front-hyphen placement
# applies) only when a nondetachable 'a'-prefix stem follows it -- ``3-oxa…``,
# ``1-aza…``, ``2,4-dioxa…`` -- never a detachable-substituent locant such as
# ``1-methyl-`` or ``4-amino-`` (a review I2). The former regex ``^\d+-(?!\d*H-)``
# matched EVERY leading locant, so a wrong front candidate
# (``(3-2H1)-1-methyl-2-nitro-1H-pyrrole``) was built and RT-tested; it lost only
# because OPSIN 2.9 rejects it, i.e. correctness rested on the grammar rather than
# the rule -- exactly the fragility this fix exists to remove. The 'a'-prefix stem
# set is taken from HW_PREFIXES so it stays in sync with the data (all end in 'a',
# so they cannot false-match the common detachable prefixes oxo-/azido-/thio-).
from ..data.hw_heteroatoms import HW_PREFIXES as _HW_PREFIXES
from ..perception.molcache import atoms_of, inchikey_of
_A_PREFIX_ALT = "|".join(sorted(set(_HW_PREFIXES.values()), key=len, reverse=True))
_LEADING_SKELETAL_LOCANT_RE = re.compile(
    r"^\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?"
    r"(?:" + _A_PREFIX_ALT + r")")


def _front_hyphen_candidate(skel: str, desc: str) -> Optional[Tuple[int, str]]:
    """``(insertion_point, candidate)`` placing ``desc`` at the front of the
    parent hydride, hyphen-joined, when ``skel`` opens (after any leading stereo
    prefix) with a skeletal-replacement locant; else ``None``. See
    ``_LEADING_SKELETAL_LOCANT_RE``. The candidate is OPSIN-RT gated by the
    caller, so a wrong guess is discarded, never shipped."""
    pre = _LEADING_STEREO_PREFIX_RE.match(skel).group(0)
    rest = skel[len(pre):]
    if not _LEADING_SKELETAL_LOCANT_RE.match(rest):
        return None
    p = len(pre)
    return p, skel[:p] + desc + "-" + skel[p:]


def _letter_locant_candidates(original: Chem.Mol, label_map: Dict[int, int]) -> List[str]:
    """Letter locants the placement search should offer in addition to
    the integer locants -- currently the amide/amine nitrogen ``N``.

    A labelled H (D/T) on a non-ring nitrogen sits on a position the parent's
    suffix cites by the letter locant ``N`` (``acetamide`` -> ``(N-2H1)acetamide``),
    not by a ring/chain number, so the integer-only search cannot place it and the
    row abstains. Offering ``N`` is safe: every candidate is still OPSIN-RT gated,
    so a spurious offer (a D on a nitrogen the parent does NOT letter-locant) is
    discarded, never shipped.
    """
    out: List[str] = []
    for idx in label_map:
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "H":
            continue
        nbrs = atom.GetNeighbors()
        if nbrs and nbrs[0].GetSymbol() == "N" and not nbrs[0].GetIsAromatic():
            if "N" not in out:
                out.append("N")
    return out


def _find_best_placement(
    skel: str, keys, original: Chem.Mol, n_pos: int,
    allow_front_hyphen: bool = False,
    extra_locants: Optional[List] = None,
    stereo_blind: bool = False,
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    """Best ``(candidate, offset, descriptor, loc_rank)`` for ONE skeleton
    spelling and ONE ``(mass, element) -> count`` key list, OPSIN-RT gated
    against ``original``; ``(None, None, None, None)`` if nothing round-trips.

    Extracted (A3) from what was a same-shaped closure named ``_enumerate``
    inside ``_decorate_isotopic_name_inner`` so BOTH the single-attachment
    path there (unchanged behaviour -- it now calls this with its own
    ``keys``/``original``) and the multi-attachment placement below
    (:func:`_decorate_multi_position`, called once per attachment GROUP with a
    masked ``original``) share ONE oracle-driven search, instead of two
    independently-maintained copies that could drift apart.

    ``loc_rank`` is -1 when the winning descriptor needs NO locant
    lowest-locant-first: the locant-free form is tried before any integer),
    else the locant that turned out to be required.

    ``allow_front_hyphen`` enables the front-of-parent hyphen-joined placement
    (:func:`_front_hyphen_candidate`) for a LOCANTED descriptor on a parent that
    opens with a skeletal-replacement locant. It is OFF for the multi-attachment
    caller, whose ``(off, desc)`` pair is re-spliced verbatim and cannot express
    the added hyphen.
    """
    offs_raw = [0] + _insertion_offsets(skel)
    seen: set = set()
    offs: List[int] = []
    for x in offs_raw:
        if x not in seen:
            seen.add(x)
            offs.append(x)
    won = []
    # letter locants (e.g. amide ``N``) are tried after the locant-free
    # form and before the integer locants -- an amide/amine D belongs on ``N``,
    # which no integer locant can express. RT-gated like every other candidate.
    for locant in [None] + list(extra_locants or []) + list(range(1, n_pos + 1)):
        groups_desc = [(locant, mass, el, count, maxpos)
                       for (mass, el), count, maxpos in keys]
        p4542 = _p4542_p4543_key(groups_desc)
        loc_rank = -1 if locant is None else locant
        # Subscript axis (FIX-A): the BB-preferred omitted form (sub_rank 0) is
        # tried first; the forced always-show form (sub_rank 1) is a parseability
        # fallback so an OPSIN-unparseable omission degrades to the right molecule
        # at a non-preferred spelling rather than abstaining. Identical strings
        # (the predicate already keeps the subscript) collapse to one variant.
        desc_pref = format_isotope_descriptor(groups_desc)
        desc_force = format_isotope_descriptor(groups_desc, force_show=True)
        variants = [(0, desc_pref)]
        if desc_force != desc_pref:
            variants.append((1, desc_force))
        for sub_rank, desc in variants:
            for off in offs:
                candidate = skel[:off] + desc + skel[off:]
                if _isotope_round_trips(candidate, original, stereo_blind=stereo_blind):
                    #: a locant-free descriptor may only be OMITTED when
                    # its position is genuinely unambiguous. A round-trip alone is
                    # not proof -- OPSIN's default placement can coincidentally match
                    # the true atom on an asymmetric parent. Verify uniqueness at this
                    # slot before accepting the omission; otherwise fall through to
                    # the explicit-locant search (and the forced_locant_scope
                    # re-render below), which produces the required locanted form.
                    # applies at ANY enclosing-mark depth: a locant-free
                    # descriptor inside a substituent bracket
                    # ('[(13C6)naphthalen-2-yl]…' -- 6 of 10 ring C, no locants) is
                    # just as ambiguous as one on the parent, and round-trips only by
                    # OPSIN's default-placement coincidence. Run the uniqueness oracle
                    # regardless of depth; when ambiguous, fall through to the explicit
                    # locant search (and, for distinct multi-position labels, to
                    # _decorate_distinct_multi_locant). (Was gated to depth 0, which
                    # shipped the ambiguous nested form -- a review review 2026-09-08.)
                    if (locant is None
                            and not _placement_unambiguous(
                                skel, off, keys, n_pos, candidate)):
                        continue
                    won.append((p4542, loc_rank, sub_rank,
                                _placement_quality(skel, off), off, desc, candidate))
            # front placement: a REQUIRED locant on a parent that opens
            # with a skeletal-replacement locant belongs at the front, hyphen-
            # joined, not spliced mid-parent. Offer that candidate (RT-gated) with
            # the lowest possible offset so it wins the tie-break over the
            # mid-parent splice.
            if allow_front_hyphen and locant is not None:
                fh = _front_hyphen_candidate(skel, desc)
                if fh is not None and _isotope_round_trips(
                        fh[1], original, stereo_blind=stereo_blind):
                    # front-hyphen is the parent-scope placement -- it is
                    # adjacent to the part it modifies, so it is quality 0.
                    won.append((p4542, locant, sub_rank, 0, fh[0], desc, fh[1]))
        if won:
            #: the first locant-form that yields ANY round-tripper is
            # the lowest-locant form; do not consider higher-locant forms.
            break
    if not won:
        return None, None, None, None
    # Tie-break order: /.3 (w[0]) -> locant (w[1]) -> subscript form
    # (w[2]) -> placement quality (w[3], adjacent-to-affix beats
    # front-detached) -> insertion offset (w[4], deterministic final key).
    won.sort(key=lambda w: (w[0], w[1], w[2], w[3], w[4]))
    _, loc_rank, _sub_rank, _q, off, desc, candidate = won[0]
    # (the Blue Book "1,2-di[(13C)methyl]benzene"): a descriptor spliced
    # INSIDE a substituent's own enclosing marks makes that substituent compound,
    # so the marks step up  ->  -> { } ('1-(amino(14C)methyl)cyclopentan-
    # 1-ol' -> '1-[amino(14C)methyl]cyclopentan-1-ol'). Escalate for THIS
    # single-descriptor placement; the multi-attachment caller
    # (:func:`_decorate_multi_position`) discards this candidate and runs its own
    # cascade after splicing every group, so it is unaffected. A descriptor at
    # parent scope (no enclosing pair) leaves the name untouched.
    candidate = _escalate_marks_for_descriptor(candidate, off)
    # (the Blue Book) / the Blue Book "(15N)-1*H*-indole (PIN)": a parenthetical
    # isotope descriptor placed at the FRONT of a parent that opens with a locant
    # digit (an indicated-hydrogen locant '1H-', a leading skeletal locant) is
    # separated from that locant by a hyphen -- a name-part is always separated
    # from a following locant by '-'. The bare front concatenation
    # '(15N)1H-indole' is a spelling defect even though it round-trips. Re-render
    # hyphenated when the descriptor is the plain front splice (off==0) and the
    # next char is a digit, RT-gated so a non-parsing hyphenation falls back to
    # the bare form (the front-hyphen candidates above already carry their '-').
    if (off == 0 and desc and candidate.startswith(desc)
            and candidate[len(desc):len(desc) + 1].isdigit()):
        hyphenated = candidate[:len(desc)] + "-" + candidate[len(desc):]
        if _isotope_round_trips(hyphenated, original, stereo_blind=stereo_blind):
            candidate = hyphenated
    return candidate, off, desc, loc_rank


def _mask_isotopes(mol: Chem.Mol, keep_indices) -> Chem.Mol:
    """A COPY of ``mol`` with every isotope label cleared except at
    ``keep_indices``. Used to test ONE attachment-group's placement in
    isolation -- see:func:`_decorate_multi_position`.

    A cleared D/T is an EXPLICIT plain-1H atom (same reasoning as
    ``strip_isotopes``): left alone, its canonical SMILES prints a literal
    ``[H]`` that can never match an OPSIN-parsed candidate (which has an
    ordinary IMPLICIT hydrogen there), so every group but the last would
    spuriously fail its round-trip. ``RemoveHs`` collapses exactly the
    now-plain explicit H atoms into implicit H count while leaving any
    STILL-labelled (isotope != 0) explicit H untouched -- RDKit treats an
    isotopically-labelled H as "special" and keeps it explicit.
    """
    copy = Chem.Mol(mol)
    keep = set(keep_indices)
    for atom in copy.GetAtoms():
        if atom.GetIdx() not in keep and atom.GetIsotope() != 0:
            atom.SetIsotope(0)
    try:
        copy = Chem.RemoveHs(copy)
    except Exception:
        pass
    return copy


def _partition_by_attachment(
    original: Chem.Mol, label_map: Dict[int, int]
) -> List[Dict[int, int]]:
    """Partition ``label_map`` into groups that share ONE skeleton attachment
    point, so each group's descriptor can be placed at its OWN insertion
    offset independently: the descriptor is inserted "before the
    part of the compound that is isotopically substituted" -- different
    PARTS of the compound take different insertion points).

    An explicit-H label (D/T, ``strip_isotopes`` never removes them from
    ``label_map`` even though ``RemoveHs`` collapses the STRIPPED copy) is
    attached to exactly one heavy neighbour; every D/T sharing that neighbour
    (the two D's of an NH2, the two of a CH2,...) is one group. A labelled
    HEAVY atom (13C, 18F, 11C,...) is its own attachment point.

    A single-group molecule (everything shares one neighbour, or there is
    only one labelled atom) returns a length-1 list, so the existing
    single-descriptor path is exercised exactly as before for every
    previously-working case -- this function only SPLITS when the labelled
    atoms sit at structurally distinct positions, and the OPSIN round-trip
    oracle still has the final say on every candidate it enables.
    """
    groups: Dict[int, Dict[int, int]] = {}
    for idx, mass in label_map.items():
        atom = original.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "H":
            nbrs = atom.GetNeighbors()
            key = nbrs[0].GetIdx() if nbrs else idx
        else:
            key = idx
        groups.setdefault(key, {})[idx] = mass
    return list(groups.values())


def _enclosing_pairs(name: str) -> List[Tuple[int, int]]:
    """Every matched enclosing-mark pair ``(open_idx, close_idx)`` in ``name``,
    across ````, ```` and ``{}`` alike (they interleave by nesting level, not
    by type). Unmatched marks are ignored -- a name string is always balanced by
    construction."""
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(name):
        if ch in "([{":
            stack.append(i)
        elif ch in ")]}" and stack:
            pairs.append((stack.pop(), i))
    return pairs


def _escalate_marks_for_descriptor(name: str, desc_start: int) -> str:
    """: an isotope descriptor spliced INSIDE a substituent's
    existing enclosing marks makes that substituent compound, so its marks must
    step up ```` -> ```` -> ``{}`` (the Blue Book ``1,2-di[(13C)methyl]benzene``).

    ``desc_start`` is where the descriptor's own ``(`` now sits in ``name``. The
    immediately-enclosing detachable pair (the one with the largest open index
    strictly before ``desc_start`` whose close is after it) is re-enclosed by its
    content's nesting depth via the shared ``apply_enclosing_marks`` primitive;
    the step-up may over-fill THAT group's own parent, so the cascade repeats
    outward until an ancestor is already at the right mark. A descriptor at parent
    scope (no enclosing pair) leaves ``name`` untouched.

    OPSIN-RT is blind to the bracket LEVEL, so this must be deterministically
    correct rather than oracle-selected. Each rewrite only swaps the two bracket
    characters of one pair, so string length -- and therefore every other index
    -- is preserved across the cascade.
    """
    from ..assembly.naming_utils import apply_enclosing_marks

    pos = desc_start
    while True:
        enclosing = [(o, c) for (o, c) in _enclosing_pairs(name)
                     if o < pos <= c]
        if not enclosing:
            return name
        o, c = max(enclosing, key=lambda p: p[0])
        rewrapped = apply_enclosing_marks(name[o + 1:c], -1)
        if rewrapped == name[o:c + 1]:
            return name
        name = name[:o] + rewrapped + name[c + 1:]
        pos = o  # cascade to the parent of the group just stepped up


def _decorate_multi_position(
    skeleton: str, groups: List[Dict[int, int]], original: Chem.Mol, n_pos: int
) -> Optional[str]:
    """Place ONE descriptor per attachment GROUP, each at its own insertion
    offset + /.2/.3 -- different labelled parts of a
    compound each get their own descriptor scoped to that part, e.g. BB's own
    ``benzene(13C)carbonitrile`` / ``3-[ethyl(2-34S)trisulfanyl]propanoic
    acid`` pattern of several independently-scoped parenthetical descriptors
    in one name).

    A single combined descriptor (the existing enumeration, one shared
    locant for every labelled atom of a nuclide) cannot express labelled
    atoms sitting at MORE than one structurally distinct position -- e.g.
    per-deuterated glycine has D on the amino N, the alpha C, and the
    carboxyl O, three unrelated "parts". Each group's own winning
    ``(offset, descriptor)`` is found INDEPENDENTLY by:func:`_find_best_placement`
    against an isotope-MASKED copy of ``original`` carrying ONLY that group's
    labels -- valid because OPSIN scopes each parenthesized descriptor to the
    substring it precedes, independent of any other descriptor elsewhere in
    the name (verified 2026-08-23: the three-way glycine split round-trips
    when the three independently-found descriptors are combined). The
    combined candidate is still OPSIN-RT gated against the TRUE original
    before being returned, so any compositionality failure (e.g. two groups
    wanting the identical offset) simply fails closed -- never a wrong name.
    """
    if len(groups) < 2:
        return None
    placements: List[Tuple[int, str]] = []
    for group in groups:
        by_key: Dict[Tuple[int, str], int] = {}
        maxpos_by_key: Dict[Tuple[int, str], int] = {}
        for idx, mass in group.items():
            atom = original.GetAtomWithIdx(idx)
            el = atom.GetSymbol()
            by_key[(mass, el)] = by_key.get((mass, el), 0) + 1
            maxpos_by_key[(mass, el)] = max(
                maxpos_by_key.get((mass, el), 0), _max_atoms_at_position(atom))
        group_keys = sorted(
            (((mass, el), by_key[(mass, el)], maxpos_by_key[(mass, el)])
             for (mass, el) in by_key),
            key=lambda kv: (kv[0][1], kv[0][0]))
        masked = _mask_isotopes(original, group.keys())
        _cand, off, desc, _loc_rank = _find_best_placement(skeleton, group_keys, masked, n_pos)
        if off is None:
            return None
        placements.append((off, desc))
    # Colliding offsets have no principled merge order here -> fail closed
    # rather than guess (this has not been observed on any verified witness).
    offsets_seen = [p[0] for p in placements]
    if len(set(offsets_seen)) != len(offsets_seen):
        return None
    # Splice from the HIGHEST offset down so an earlier offset is never
    # shifted by a later insertion.
    placements.sort(key=lambda p: -p[0])
    out = skeleton
    for off, desc in placements:
        out = out[:off] + desc + out[off:]
    #: step up the enclosing marks of any substituent that just
    # received a descriptor INSIDE its own marks (``(1-bromopropyl)`` +
    # ``(81Br)`` -> ``[1-(81Br)bromopropyl]``). The final start index of each
    # descriptor in ``out`` is its own offset plus the total length of every
    # descriptor spliced at a LOWER offset (each rewrite below preserves length,
    # so these indices stay valid through the cascade).
    shift = 0
    for off, desc in sorted(placements):
        out = _escalate_marks_for_descriptor(out, off + shift)
        shift += len(desc)
    if _isotope_round_trips(out, original):
        return out
    return None


#: Heavy-atom cap for the distinct-position multi-locant sweep. The sweep does
#: up to (offsets x n) single-label OPSIN parses; 60 bounds the cost on a large
#: cage/ring parent while covering every realistic labelled skeleton.
_MULTI_LOCANT_SWEEP_CAP = 60
#: Bound on the number of candidate locant-SETS the sweep RT-gates per offset.
#: An asymmetric parent yields ~1 (each atom's environment is unique); a highly
#: symmetric one can combinatorially blow up, so a set count over this fails
#: closed rather than spending unbounded OPSIN parses.
_MULTI_LOCANT_COMBO_CAP = 4000


def _decorate_distinct_multi_locant(
    skeleton: str, keys, original: Chem.Mol,
    label_map: Dict[int, int], n_pos: int,
) -> Optional[str]:
    """ /: ONE nuclide labelled at MULTIPLE structurally
    DISTINCT positions on ONE parent -> a single ascending multi-locant
    descriptor, e.g. D at cage/ring positions 1,3,7 -> ``(1,3,7-2H3)``.

    Neither existing path expresses this. ``_find_best_placement`` (the single
    combined enumeration) repeats ONE shared locant ``count`` times
    (``2,2,2-2H3`` -- three atoms all at position 2, via
    :func:`format_isotope_descriptor`), and:func:`_decorate_multi_position`
    splits by distinct ATTACHMENT GROUP, not by distinct position within one
    parent. So a partially-labelled von Baeyer cage or a multiply-labelled ring
    (the residual isotope-abstain mass, measured 2026-09-08) fell through to
    ``None``.

    Orthonym exposes no atom->locant map for these parents, so -- exactly as the
    single-descriptor enumeration does -- this treats the skeleton's locant space
    as opaque and lets OPSIN adjudicate. For each insertion offset it sweeps a
    single-label probe ``(L-<sym>1)`` over every candidate locant TOKEN ``L``
    (parse -> canonical SMILES), building a locant->structure map, then groups
    locants by the single-label isotopomer they produce and the labelled atoms by
    their own single-label reference (:func:`_mask_isotopes`). Each reference class
    must be filled from its matching locant class; the labelled SET (not the
    individual atom, which is indistinguishable on a symmetric parent) is what the
    isotopomer fixes, so it enumerates the candidate locant sets in lowest-locant
    order and returns the first that OPSIN-RT gates against the true
    original.

    The candidate locants include PRIMED forms (``4'``, ``1''``) when the skeleton
    itself carries a prime (a ring assembly ``1,1'-biphenyl``, a multiparent fused
    system): those parents number across sub-units with primed locants, which a
    plain integer sweep cannot reach, so a D on the second ring of biphenyl had no
    placeable locant. The prime depth is bounded by the skeleton's own prime count,
    so a skeleton with NO prime sweeps the identical integer set as before (no
    behaviour change for those parents).

    Fail-closed (None) outside its scope: exactly one nuclide, count >= 2, every
    reference class fillable from its locant class, the candidate-set count within
    :data:`_MULTI_LOCANT_COMBO_CAP`, and a round-tripping candidate. A wrong
    locant set simply fails to reproduce the structure, so no wrong or
    label-dropping name is ever emitted.
    """
    import itertools

    if len(keys) != 1:
        return None
    (mass, el), count, _maxpos = keys[0]
    my_atoms = [idx for idx, m in label_map.items()
                if m == mass and original.GetAtomWithIdx(idx).GetSymbol() == el]
    if count < 2 or len(my_atoms) != count:
        return None
    # Single-label canonical reference (isotopes kept) for each labelled atom,
    # grouped: {reference_smiles: how many atoms carry it}. Symmetric atoms share
    # one reference and must be filled from the SAME locant class.
    try:
        ref_count: Dict[str, int] = {}
        for idx in my_atoms:
            r = Chem.MolToSmiles(_mask_isotopes(original, [idx]))
            ref_count[r] = ref_count.get(r, 0) + 1
    except Exception:
        return None
    sym = nuclide_symbol(mass, el)
    n = min(n_pos, _MULTI_LOCANT_SWEEP_CAP)
    # Prime depth: 0 when the skeleton carries no prime (then the token set is the
    # identical integer sweep as before -> no behaviour change). A ring assembly /
    # multiparent parent numbers across sub-units with primed locants; sweep up to
    # its own prime count (bounded to 3 to keep the probe count sane).
    maxprime = min(3, skeleton.count("'"))
    loc_tokens = [f"{base}{chr(39) * p}"
                  for p in range(maxprime + 1) for base in range(1, n + 1)]
    # Fusion / ring-junction carbons carry a LETTER-suffixed locant (indole 3a, 7a;
    # a labelled fused-ring bridgehead has no plain integer locant). Add '<base>a'
    # tokens when the molecule has a ring-fusion atom (a carbon shared by >=2 rings);
    # OPSIN rejects the invalid ones (they never enter loc_canon) and the RT gate
    # keeps 0-wrong. A non-fused parent keeps the identical integer sweep.
    ri = original.GetRingInfo()
    if any(ri.NumAtomRings(a) >= 2 for a in range(original.GetNumAtoms())):
        loc_tokens += [f"{base}a" for base in range(1, n + 1)]
    seen: set = set()
    offs = [o for o in ([0] + _insertion_offsets(skeleton))
            if not (o in seen or seen.add(o))]
    for off in offs:
        # locant token -> single-label isotopomer at this offset
        loc_canon: Dict[str, str] = {}
        for tok in loc_tokens:
            probe = f"({tok}-{sym}1)"
            s = _opsin_parse(skeleton[:off] + probe + skeleton[off:])
            if not s:
                continue
            pm = Chem.MolFromSmiles(s)
            if pm is not None:
                loc_canon[tok] = Chem.MolToSmiles(pm)
        if not loc_canon:
            continue
        # invert: reference_smiles -> locant tokens that reproduce it (lowest-first)
        by_ref: Dict[str, List[str]] = {}
        for tok in sorted(loc_canon, key=_locant_sort_key):
            by_ref.setdefault(loc_canon[tok], []).append(tok)
        # each reference class must be fillable from its own locant class
        if any(ref not in by_ref or len(by_ref[ref]) < need
               for ref, need in ref_count.items()):
            continue
        # count candidate locant-set combinations; fail closed if unbounded
        combos = 1
        for ref, need in ref_count.items():
            combos *= _n_choose_k(len(by_ref[ref]), need)
            if combos > _MULTI_LOCANT_COMBO_CAP:
                break
        if combos > _MULTI_LOCANT_COMBO_CAP:
            continue
        # per-reference choices; the full locant set is one pick from each class
        per_ref_choices = [
            list(itertools.combinations(by_ref[ref], need))
            for ref, need in ref_count.items()
        ]
        candidate_sets = []
        for pick in itertools.product(*per_ref_choices):
            locs = sorted((tok for group in pick for tok in group),
                          key=_locant_sort_key)
            candidate_sets.append(tuple(locs))
        # lowest-locant-first, unprimed before primed
        candidate_sets.sort(key=lambda s: [_locant_sort_key(t) for t in s])
        # /: a descriptor spliced immediately before a locant
        # digit (a front skeletal locant '1,1'-biphenyl', or a bracketed/leading
        # indicated-H '1H-indol-...') is hyphen-separated from it:
        # '(2,4',5-2H3)-1,1'-biphenyl', '(15N)-1H-indole'. Try the hyphenated form
        # first when the splice point is a digit; RT-gated, so a non-parsing
        # hyphenation falls back to the bare splice that already round-tripped.
        before_digit = off < len(skeleton) and skeleton[off:off + 1].isdigit()
        for locs in candidate_sets:
            loc_part = ",".join(locs)
            # count-bearing subscript first, bare form as a parseability fallback.
            for desc in (f"({loc_part}-{sym}{count})", f"({loc_part}-{sym})"):
                joins = ["-", ""] if before_digit else [""]
                for join in joins:
                    cand = skeleton[:off] + desc + join + skeleton[off:]
                    if _isotope_round_trips(cand, original):
                        return _escalate_marks_for_descriptor(cand, off)
    return None


def _locant_sort_key(tok: str) -> tuple:
    """ (the Blue Book) locant order: 'Primed locants are placed
    immediately after the corresponding unprimed locants in a set arranged in
    ascending order; locants consisting of a number and a lower-case letter...
    are placed immediately after the corresponding numeric locant'.

    So order by NUMERIC value first, then by (letter-suffix, prime-count) as a
    tie-break -- i.e. {2,5,4'} -> 2,4',5 (not 2,5,4'); 4a immediately after 4;
    4' immediately after 4. Key = (base_number, has_letter, letter, nprime).
    ``"2"`` -> (2,0,'',0); ``"4'"`` -> (4,0,'',1); ``"3a"`` -> (3,1,'a',0);
    ``"4'a"`` -> (4,1,'a',1)."""
    nprime = tok.count("'")
    core = tok.replace("'", "")
    # split trailing lower-case letter(s) (fusion locant like 3a / 4a)
    i = len(core)
    while i > 0 and core[i - 1].isalpha():
        i -= 1
    num_part, letter = core[:i], core[i:]
    try:
        base = int(num_part)
    except ValueError:
        base = 0
    return (base, 1 if letter else 0, letter, nprime)


def _n_choose_k(n: int, k: int) -> int:
    """C(n, k), 0 for k > n. Small local helper for the combination bound."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


# ---------------------------------------------------------------------------
# (the Blue Book) -- systematic-parent fallback
# ---------------------------------------------------------------------------
# "When the nuclide is located at a position in a retained name that is not
# numbered[,] a systematic name that identifies separately the relevant atom
# is used for the IUPAC preferred name." style='systematic' does not always
# deliver that: measured 2026-08-23, on a substituted 2-carbon carboxylic
# acid it still keeps the RETAINED "acetic acid" stem ("aminoacetic acid",
# not "aminoethanoic acid") -- omits the substituent's own locant and
# the retained "acid" suffix has no "-oic acid" slot to insert a descriptor
# before, unlike the fully systematic form. The amino-acid class already
# carries its own fully systematic namer for exactly this shape
# (`_name_amino_acid_systematic`); this fallback is scoped to it. When it
# does not apply, this returns None and the caller's fallback is a no-op --
# never a forced or guessed alternate parent.
def _systematic_parent_fallback(stripped: Chem.Mol) -> Optional[str]:
    """An alternate, more fully systematic parent name for ``stripped``.

    Currently covers the amino-acid class only (the verified per-D-glycine
    gap); returns None when inapplicable.
    """
    try:
        from .amino_acids import _name_amino_acid_systematic, detect_amino_acid
    except Exception:
        return None
    try:
        if not detect_amino_acid(stripped):
            return None
        return _name_amino_acid_systematic(stripped)
    except Exception:
        return None


_MULT_TO_COUNT = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}
_COUNT_TO_MULT = {v: k for k, v in _MULT_TO_COUNT.items()}

#: A repeated IDENTICAL letter locant (``N``, ``O``,...) followed by one of
#: the simple multiplier stems, e.g. the ``N,N,N-tri`` of
#: ``N,N,N-trimethylethanaminium``. Anchored to start-of-string or a
#: preceding hyphen so it cannot match mid-word.
_LETTER_MULT_RE = re.compile(
    r"(?:^|-)(?P<letter>[A-Z])(?:,(?P=letter))+-(?P<mult>di|tri|tetra|penta|hexa)"
)


def _decorate_letter_multiplier(skeleton, keys, original, stripped,
                                stereo_blind: bool = False) -> Optional[str]:
    """ de-multiplication for a REPEATED-LETTER-LOCANT substituent
    cluster (e.g. ``N,N,N-trimethyl``), found ANYWHERE in the skeleton -- not
    only a leading multiplier at the very front of the whole name (that
    narrower shape, with a NUMERIC locant list, is:func:`_decorate_demultiplied`).

    A letter locant (``N``, ``O``,...) cites several substituents on the
    SAME heteroatom; unlike a numeric gem-locant pair there is no alternative
    numbered position to disambiguate, so the copies are chemically
    INTERCHANGEABLE before labelling -- splitting off any ONE of them for the
    label yields the unique labelled structure, and no permutation search
    (unlike:func:`_decorate_demultiplied`'s distinct-numeric-position search)
    is needed.

    Scoped to a single labelled copy of a single nuclide (the verified
    witness, 11C-choline's ``N,N,N-trimethyl`` -> one 11C-methyl); every
    candidate is still OPSIN-RT gated against the true original AND the
    intermediate re-worded-but-unlabelled split is gated against the
    STRIPPED skeleton (to find the correct substituent-word boundary), so a
    wrong split point or a wrong final structure can only be discarded, never
    shipped.
    """
    if len(keys) != 1:
        return None
    (mass, el), count, maxpos = keys[0]
    if count != 1:
        return None
    for m in _LETTER_MULT_RE.finditer(skeleton):
        letter = m.group("letter")
        mult_count = _MULT_TO_COUNT[m.group("mult")]
        prefix = skeleton[: m.start("letter")]
        after = skeleton[m.end():]
        remaining = mult_count - 1
        for k in range(1, len(after)):
            base, tail = after[:k], after[k:]
            if remaining == 0:
                bare_seg = ""
            elif remaining == 1:
                bare_seg = f"{letter}-{base}-"
            else:
                bare_seg = (
                    ",".join([letter] * remaining) + "-"
                    + _COUNT_TO_MULT[remaining] + base + "-"
                )
            unlabelled_split = f"{prefix}{bare_seg}{letter}-{base}{tail}"
            if not _isotope_round_trips(unlabelled_split, stripped):
                continue
            # Locant None: the descriptor sits immediately after the "N-"
            # substituent citation already spliced into ``candidate`` below,
            # so 's "before the part... substituted" scope is already
            # unambiguous without also repeating the letter INSIDE the
            # descriptor itself (that would be a different, unverified
            # spelling -- e.g. ``(N-11C1)`` -- not the one this function is
            # built to produce).
            desc = format_isotope_descriptor([(None, mass, el, 1, maxpos)])
            candidate = f"{prefix}{bare_seg}{letter}-{desc}{base}{tail}"
            # ``stereo_blind`` caller): compare CONSTITUTION only so the
            # stereodescriptor-less split reproduces the connectivity; the caller
            # re-gates the stereodescriptor-prefixed candidate full-InChIKey. The
            # intermediate ``unlabelled_split`` probe stays FULL (it locates the
            # substituent-word boundary against the already-achiral ``stripped``).
            if _isotope_round_trips(candidate, original, stereo_blind=stereo_blind):
                return candidate
    return None


def _decorate_demultiplied(skeleton, keys, original, stripped,
                           stereo_blind: bool = False) -> Optional[str]:
    """Generalized de-multiplication (single OR mixed nuclides).

    Split a leading-locant simple multiplier (e.g. ``1,2-dimethoxyethane``) and
    scope a DISTINCT nuclide descriptor to each labeled copy, leaving at most one
    copy bare. Handles both:
      * single-group — one labeled copy among bare copies
                         +, e.g. 1-(13C1)methoxy-2-methoxyethane)
      * mixed — DISTINCT nuclides on DISTINCT copies, where — because the
                        parent is symmetric — BOTH numbering directions round-trip
                        and /.4.3 must choose (e.g.
                        1-(18O1)methoxy-2-(13C1)methoxyethane: 18O Z=8 > 13C Z=6).

    Every candidate is OPSIN-RT gated with isotopes retained; among the
    round-trippers, (lowest locants to the modified copies) then
    /.4.3 (higher atomic number then higher mass number at the lower
    locant) selects deterministically. The single-combined-descriptor enumeration
    in decorate_isotopic_name CANNOT place distinct nuclides on distinct copies,
    and would exhaust its JVM-per-candidate search fruitlessly on a de-mult
    skeleton, so this runs FIRST and that enumeration is the fall-through.

    Fail-closed (None) — never a wrong OR non-PIN separated name — when: the
    skeleton is not a simple-multiplier form; locants repeat (gem copies, which a
    locant-keyed map cannot tell apart); more than one copy is left bare or a
    nuclide repeats across copies (the unmodified/identical copies must stay
    grouped under a reduced multiplier — a DIFFERENT name shape, not this split,
    e.g. `3,5-dimethyl` not `3-methyl-5-methyl`; NOT built here); the candidate
    budget would blow up; or nothing round-trips. This is a DELIBERATE PIN-safe
    scope, not a superset of the single-group-only fallback it replaces (that
    fallback exploded every copy — non-PIN for >1 bare copy).
    """
    import re
    from itertools import combinations, permutations
    from math import comb, factorial

    m = re.match(
        r"^(?P<locs>\d+(?:,\d+)+)-(?P<mult>di|tri|tetra|penta|hexa)(?P<tail>[a-z].+)$",
        skeleton,
    )
    if not m:
        return None
    mult_count = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}[m.group("mult")]
    locs = sorted(int(x) for x in m.group("locs").split(","))
    if len(locs) != mult_count:
        return None
    if len(set(locs)) != len(locs):
        # Repeated locants (gem-disubstitution, e.g. 1,1,2-...): a locant-keyed
        # copy map (loc_to_label below) cannot distinguish two copies sharing a
        # locant, so a single labeled gem-copy is not constructible -> fail closed.
        return None
    tail = m.group("tail")

    # One nuclide instance per labeled copy (count 1 each). ``maxpos_of`` carries
    # the polysubstitution quantity per nuclide out to the descriptor
    # build below WITHOUT entering ``labels`` -- the set-distinctness / repeated-
    # nuclide guards must key on (mass, el) alone (two copies of one nuclide could
    # differ in max_at_pos yet must still be treated as identical here).
    maxpos_of = {(mass, el): maxpos for (mass, el), count, maxpos in keys}
    labels = []
    for (mass, el), count, maxpos in keys:
        labels.extend([(mass, el)] * count)
    # Scope (PIN-safe de-multiplication): every labeled copy carries a DISTINCT
    # nuclide and AT MOST ONE copy is left bare. Then no two copies share a
    # descriptor, so nothing needs re-multiplying and the split is a genuine PIN
    # de-multiplication. Repeated identical labels / >1 bare copy would require
    # re-collecting copies under a multiplier (a different name shape) -> fail
    # closed rather than emit a non-PIN separated form.
    if (not labels
            or len(set(labels)) != len(labels)
            or (mult_count - len(labels)) not in (0, 1)):
        return None
    # Cost cap (bounded oracle budget): the assignment search is
    # C(mult_count, len(labels)) * len(labels)! candidates, EACH OPSIN-RT gated
    # via a fresh JVM (~1.5-2s). Reject the combinatorial blow-up of high
    # multipliers with many distinct nuclides (penta/hexa reach 720) -> fail
    # closed rather than spend minutes; 24 covers di/tri/tetra, the verified
    # target being di.
    if comb(mult_count, len(labels)) * factorial(len(labels)) > 24:
        return None

    # Find the substituent/parent split point(s) against the UNLABELED skeleton
    # FIRST (bounded: <= len(tail) probes). A non-de-mult skeleton or a
    # parent-labeled skeleton exits here cheaply instead of exhausting the oracle.
    valid_ks = []
    for k in range(1, len(tail)):
        base = tail[:k]
        bare = "-".join(f"{L}-{base}" for L in locs) + tail[k:]
        if _isotope_round_trips(bare, stripped):
            valid_ks.append(k)
    if not valid_ks:
        return None

    winners = []  # (p4541_key, p4542_key, k, candidate)
    # labels are all distinct (guard above), so permutations yields no duplicates
    # and need not be de-duped; iteration order does not affect the RESULT because
    # the final sort key ends in the candidate string (see the sort comment).
    for chosen in combinations(locs, len(labels)):
        for perm in permutations(labels):
            loc_to_label = dict(zip(chosen, perm))
            for k in valid_ks:
                base, parent = tail[:k], tail[k:]
                segs = []
                for L in locs:
                    if L in loc_to_label:
                        mass, el = loc_to_label[L]
                        desc = format_isotope_descriptor(
                            [(None, mass, el, 1, maxpos_of[(mass, el)])])
                        segs.append(f"{L}-{desc}{base}")
                    else:
                        segs.append(f"{L}-{base}")
                cand = "-".join(segs) + parent
                # ``stereo_blind`` caller): compare CONSTITUTION only so a
                # de-multiplied constitution that lacks the stereodescriptor still
                # passes; the caller re-gates the stereo-prefixed candidate on the
                # full InChIKey. The ``bare`` vs ``stripped`` probe above stays FULL
                # (``stripped`` is achiral -- it only fixes the word boundary).
                if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
                    groups = [(L, loc_to_label[L][0], loc_to_label[L][1], 1,
                               maxpos_of[loc_to_label[L]])
                              for L in sorted(loc_to_label)]
                    p4541 = tuple(sorted(loc_to_label))
                    winners.append((p4541, _p4542_p4543_key(groups), k, cand))
    if not winners:
        return None
    # (lowest locants to modified copies) then /.4.3; the
    # trailing candidate STRING term (w[3]) makes the winner fully deterministic
    # regardless of assignment-iteration order -- it is load-bearing, not a
    # decorative tiebreak. Do not drop it.
    winners.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
    return winners[0][3]


def _decorate_uniform_multiplier(skeleton, keys, original, stripped,
                                 stereo_blind: bool = False) -> Optional[str]:
    """ + — a UNIFORM multiplied substituent: every copy of a
    de-multipliable prefix carries the IDENTICAL nuclide descriptor.

    ``_decorate_demultiplied`` deliberately DECLINES the repeated-identical-nuclide
    case (its docstring: identical copies must stay GROUPED under the multiplier, a
    DIFFERENT name shape). This builds that shape. It keeps the ``di``/``tri``/...
    multiplier and scopes ONE descriptor to the repeated substituent, then steps the
    enclosing marks up ```` -> ```` because the substituent already carries the
    descriptor's parentheses, the Blue Book; verbatim example the Blue Book
    ``1,2-di[(13C)methyl]benzene``). The step-up reuses the shared
    ``apply_enclosing_marks`` primitive -- no string post-processing.

    ⚠ SCOPE — NON-hydrogen nuclides ONLY. ``apply_enclosing_marks`` routes a leading
    ``(2H)``/``(3H)`` through ``_INDICATED_H_RE`` (naming_utils.py:1127) and returns
    ``((2H)methyl)`` (parentheses, NO step-up) instead of ``[(2H)methyl]``; OPSIN-RT
    is blind to that bracket-level spelling error and would NOT reject it. A
    deuterium/tritium uniform multiplier therefore fails closed here and falls
    through to the enumeration path (an RT-valid, non-bracket-stepped spelling),
    never shipping a wrong bracket.

    Every candidate is OPSIN-RT gated (isotopes retained), so a wrong split fails
    closed. Fail-closed (None) when the skeleton is not a leading-locant simple
    multiplier, the labels are not ONE uniform non-hydrogen nuclide with exactly one
    atom per copy (``count == mult_count``), or nothing round-trips.
    """
    from ..assembly.naming_utils import apply_enclosing_marks

    m = re.match(
        r"^(?P<locs>\d+(?:,\d+)+)-(?P<mult>di|tri|tetra|penta|hexa)(?P<tail>[a-z].+)$",
        skeleton,
    )
    if not m:
        return None
    mult_count = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6}[m.group("mult")]
    locs = sorted(int(x) for x in m.group("locs").split(","))
    if len(locs) != mult_count or len(set(locs)) != len(locs):
        # gem/repeated locants: not the distinct-copy uniform shape -> fail closed.
        return None
    tail = m.group("tail")

    # UNIFORM scope: exactly one distinct nuclide, one atom on EACH of the
    # mult_count copies (count == mult_count), and NOT hydrogen (bracket hazard
    # above). A single labeled copy among bare ones, or count != mult_count, is a
    # different (or non-uniform) shape -> fail closed.
    if len(keys) != 1:
        return None
    (mass, el), count, maxpos = keys[0]
    if el == "H" or count != mult_count:
        return None

    # One descriptor for the repeated copy (count 1 per copy). ``maxpos`` decides
    # the subscript (FIX-A): a carbon position holds one carbon -> ``(13C)``.
    desc = format_isotope_descriptor([(None, mass, el, 1, maxpos)])

    # Split ``tail`` into the repeated substituent (``methyl``) and the parent
    # (``benzene``). The grammar is opaque to this module, so every split is offered
    # and the OPSIN round-trip oracle keeps only the one that reproduces the mol
    # descriptor scope). ``apply_enclosing_marks(..., -1)`` steps ->
    #. Deterministic: lowest split index, then candidate string.
    winners = []
    for k in range(1, len(tail)):
        base, parent = tail[:k], tail[k:]
        sub = apply_enclosing_marks(f"{desc}{base}", -1)
        cand = f"{m.group('locs')}-{m.group('mult')}{sub}{parent}"
        # ``stereo_blind`` caller): constitution-only gate; the caller
        # re-gates the stereo-prefixed candidate full-InChIKey (0-wrong preserved).
        if _isotope_round_trips(cand, original, stereo_blind=stereo_blind):
            winners.append((k, cand))
    if not winners:
        return None
    winners.sort()
    return winners[0][1]


# ---------------------------------------------------------------------------
# (BB:~43855) -- isotope-INDUCED stereocentre
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


_MIXED_LOCANT_COMBO_CAP = 256  # (n_pos+1)**n_groups product bound; fail closed above


def _decorate_mixed_locant(skeleton: str, keys, original: Chem.Mol,
                           n_pos: int, stripped_smiles: str,
                           namer) -> Optional[str]:
    """ mixed descriptor: a multi-nuclide label where SOME nuclides
    need a locant and others do not — an ambiguous nuclide beside a unique one,
    e.g. ``C[13CH2][15NH2]`` -> ``(1-13C,15N)`` (¹³C ambiguous among the two ring/
    chain carbons, ¹⁵N unique). ``_find_best_placement`` shares ONE locant across
    every nuclide group, so it can build only the all-None ``(13C,15N)`` (ambiguous,
    now rejected by the uniqueness oracle) or the all-same ``(1-13C,1-15N)`` (OPSIN
    rejects the ¹⁵N locant); neither is the mixed form. This enumerates an
    INDEPENDENT locant per nuclide group (each None or 1..n_pos), RT-gated,
    lowest-locants-first. Bounded / fail-closed on a large product.

    Isolated fallback: only reached from the ``best is None`` cascade AFTER the
    single-descriptor and distinct-multi-locant paths return None, and only for a
    MULTI-nuclide descriptor, so it cannot change any row those paths already
    name."""
    if len(keys) < 2:
        return None
    if (n_pos + 1) ** len(keys) > _MIXED_LOCANT_COMBO_CAP:
        return None  # fail closed rather than spend minutes on a large product
    from itertools import product
    # A mixed descriptor ALWAYS cites at least one locant, so forces
    # the parent to restore its own omitted locants too (…ethan-1-amine, not
    # …ethanamine). Re-name the skeleton inside the forced-locant scope, as the
    # single-descriptor path does at the restoration block below; keep
    # the free-elided skeleton as a fallback if the re-render fails.
    from ..assembly.locant_omission import forced_locant_scope
    with forced_locant_scope("isotope"):
        skel_forced = _flagged_systematic_namer(namer).name(stripped_smiles)
    if skel_forced and "unknown" not in skel_forced.lower():
        skeleton = skel_forced
    seen_off: set = set()
    offs = [o for o in ([0] + _insertion_offsets(skeleton))
            if not (o in seen_off or seen_off.add(o))]
    choices = [[None] + list(range(1, n_pos + 1)) for _ in keys]
    winners = []
    for combo in product(*choices):
        # all-None and all-same forms are already covered by _find_best_placement;
        # this pass contributes only genuinely MIXED assignments.
        if len(set(combo)) <= 1:
            continue
        groups_desc = [(loc, mass, el, count, maxpos)
                       for loc, ((mass, el), count, maxpos) in zip(combo, keys)]
        # Subscript axis (as in _find_best_placement): the BB-preferred omitted-
        # subscript form (sub_rank 0, `(1-13C,15N)`) is tried first; the force-show
        # form (sub_rank 1, `(1-13C1,15N1)`) is a parseability fallback.
        desc_pref = format_isotope_descriptor(groups_desc)
        desc_force = format_isotope_descriptor(groups_desc, force_show=True)
        variants = [(0, desc_pref)]
        if desc_force != desc_pref:
            variants.append((1, desc_force))
        for sub_rank, desc in variants:
            for off in offs:
                cand = skeleton[:off] + desc + skeleton[off:]
                if _isotope_round_trips(cand, original):
                    cited = sorted(c for c in combo if c is not None)
                    winners.append((cited, sub_rank, off, cand))
    if not winners:
        return None
    # lowest locants -> BB-preferred subscript -> deterministic offset/string
    winners.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
    return winners[0][3]


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


def _decorate_isotope_stereo(skeleton, keys, original, stripped, n_pos,
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

    : "the stereodescriptors are cited first" -- the prefix leads the name.
    Every emission is FULL-oracle gated, so a wrong config/locant fails closed;
    fails closed (None) when there is no induced centre, no constitution placement,
    the enumeration would exceed the cap, or nothing round-trips.
    """
    k = _cip_labeled_count(original) - _cip_labeled_count(stripped)
    if k < 1:
        return None
    base, _off, _desc, _lr = _find_best_placement(
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
        for _demux_fn in (_decorate_demultiplied,
                          _decorate_uniform_multiplier,
                          _decorate_letter_multiplier):
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
            if _isotope_round_trips(cand, original):  # FULL stereo + isotope
                winners.append((locs, cfg, cand))
    if not winners:
        return None
    # lowest-locant-first; deterministic tiebreak on the candidate string.
    winners.sort(key=lambda w: (w[0], w[1], w[2]))
    return winners[0][2]


# ─────────────────────────────────────────────────────────────────────────────────────
# WITHDRAWN 2026-07-28: a whole-molecule isotopomer-ambiguity test.
#
# It was built and wired here, and it REGRESSED a correct name, so it is recorded rather
# than kept. It asked "can this nuclide multiset be placed on the skeleton in more than
# one non-isomorphic way?" and forced a locant when so. That ignores ****
# (``:44182``, verbatim): the descriptor is inserted *"before the part of the compound
# that is isotopically substituted"* -- so the descriptor's POSITION carries its scope,
# and ambiguity has to be judged inside that scope, not over the whole molecule.
#
# The witness that refuted it: ``[13CH3]OC(C)=O`` -> ``(13C1)methyl acetate``. Methyl
# acetate has three distinct carbons, so a whole-molecule test says "ambiguous" and forces
# ``(1-13C1)methyl acetate``. But the descriptor sits before ``methyl``, whose scope is a
# single carbon, so the name is already unique. ``:7492`` settles it:
# ``1,2-di[(13C)methyl]benzene (PIN. `` -- descriptor inside the brackets,
# scoped to ``methyl``, no locant on the label.
#
# Judging scope correctly needs awareness of the name's grammar, which this module does
# not have (it enumerates insertion offsets and lets the round-trip oracle choose). So the
# general rule is NOT implemented. What IS implemented below is the half with a verbatim
# Blue Book anchor: when the descriptor itself needs a locant, restores the
# parent's locants. See the open-defect note in
# internal notes
# ─────────────────────────────────────────────────────────────────────────────────────


def decorate_isotopic_name(smiles: str, style: str, namer) -> Optional[str]:
    """Fail-closed isotopic-substitution PIN +.

    1. Parse + strip isotopes; if none, None.
    2. Name the skeleton in systematic style (locanted parents).
    3. Enumerate candidate descriptors (lowest-locant-first, and
       accept the first whose OPSIN round-trip reproduces the original mol.
    4. None if nothing round-trips (never a wrong labeled name).
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or not has_isotopes(original):
        return None
    stripped, label_map = strip_isotopes(original)
    if not label_map:
        return None
    # (``:44180``). Everything below names the isotope-STRIPPED molecule,
    # so no structural test further down can see that a label exists -- measured: at
    # the substituent licences every ``GetIsotope`` in scope reads 0 even
    # for a 13C input. Declare it ambiently instead, UNCONDITIONALLY, so a licence
    # about to empty a scope of all its locants can decline. This is deliberately
    # weaker than ``forced_locant_scope`` (entered further down only once a locant is
    # known to be REQUIRED): the weaker flag must not gag the omissions that stay
    # correct under, i.e. ``(13C1)benzenehexol`` and ``(2H6)benzene``.
    from ..assembly.locant_omission import isotopic_naming_scope
    with isotopic_naming_scope("isotope"):
        return _decorate_isotopic_name_inner(
            smiles, style, namer, original, stripped, label_map)


def _flagged_systematic_namer(namer):
    """A fresh ``style="systematic"`` engine that INHERITS the caller's best-effort
    flags.

     (p4-trace): the isotope decorator named the isotope-STRIPPED skeleton with a
    FLAG-LESS ``Orthonym(style="systematic")``. The default top-level style is
    ``"pin"``, so the flag-carrying ``namer.name`` path below is bypassed and this
    fresh engine ran WITHOUT the caller's ``general_fallback`` /
    ``general_fallback_unverified`` / ``allow_aromatic_general`` — so a skeleton the
    general engine could only name at the best-effort tier returned
    ``'unknown organic compound'`` and the decoration failed closed
    (``isotope_decorator_failed``), dropping ~26% of the reclaimable isotope rows.
    The skeleton must be named at the SAME tier the caller runs at. Best-effort tier
    only (the flags are False at the PIN default), so PIN/complete output is
    byte-identical; ``_isotope_round_trips`` still gates every emission (0-wrong)."""
    from ..namer import Orthonym
    return Orthonym(
        style="systematic",
        general_fallback=getattr(namer, "_general_fallback", False),
        general_fallback_unverified=getattr(
            namer, "_general_fallback_unverified", False),
        allow_aromatic_general=getattr(namer, "_allow_aromatic_general", False),
    )


def _decorate_isotopic_name_inner(smiles, style, namer, original, stripped,
                                  label_map) -> Optional[str]:
    """Body of:func:`decorate_isotopic_name`, run inside the ambient isotopic scope."""
    stripped_smiles = Chem.MolToSmiles(stripped)
    # Skeleton in systematic style so parents carry locants (ethan-1-ol, not
    # ethanol) — the descriptor's locant needs a locanted parent.
    skeleton = namer.name(stripped_smiles) if style == "systematic" else None
    if skeleton is None:
        # Re-name in systematic regardless (the descriptor needs locants), at the
        # caller's tier so a best-effort skeleton is reachable (, p4-trace).
        skeleton = _flagged_systematic_namer(namer).name(stripped_smiles)
    from ..errors import is_failure_name
    if not skeleton or "unknown" in skeleton.lower() or is_failure_name(skeleton):
        return None

    # Group labels by (mass, element) -> count; element read from the ORIGINAL
    # mol at each labeled atom index (label_map keys index the original).
    # ``maxpos_by_key`` carries the polysubstitution quantity per nuclide
    # (:func:`_max_atoms_at_position`), taken as the MAX over the group's atoms so
    # a count-1 nuclide contributes its single atom's capacity -- it decides
    # whether the count subscript is shown (FIX-A). ``keys`` entries are enriched
    # to ((mass, el), count, max_at_pos).
    by_key: Dict[Tuple[int, str], int] = {}
    maxpos_by_key: Dict[Tuple[int, str], int] = {}
    for idx, mass in label_map.items():
        atom = original.GetAtomWithIdx(idx)
        el = atom.GetSymbol()
        by_key[(mass, el)] = by_key.get((mass, el), 0) + 1
        maxpos_by_key[(mass, el)] = max(
            maxpos_by_key.get((mass, el), 0), _max_atoms_at_position(atom))

    n_pos = stripped.GetNumHeavyAtoms()

    # Alphabetical by element then mass citation order).
    keys = sorted(
        (((mass, el), by_key[(mass, el)], maxpos_by_key[(mass, el)])
         for (mass, el) in by_key),
        key=lambda kv: (kv[0][1], kv[0][0]))

    # de-multiplication FIRST (single OR mixed nuclides). This is the
    # ONLY path that can place DISTINCT nuclides on DISTINCT copies of a
    # de-multiplied substituent (the mixed 18O/13C case), and it is bounded, so
    # routing it ahead of the single-combined-descriptor enumeration below both
    # (a) fixes that case and (b) avoids the fruitless JVM-per-candidate
    # exhaustion that enumeration performs on a de-mult skeleton. Fail-through to
    # the enumeration when the skeleton is not a de-mult form.
    demux = _decorate_demultiplied(skeleton, keys, original, stripped)
    if demux is not None:
        return demux

    # +: a UNIFORM multiplied substituent, every copy carrying
    # the IDENTICAL nuclide descriptor (which _decorate_demultiplied declines by
    # design -- identical copies stay grouped under the multiplier). Keep the
    # multiplier and scope one bracket-stepped descriptor to the repeated copy:
    # 1,2-di[(13C)methyl]benzene, NOT 1,2-di(1,1-13C2)methylbenzene. Runs before the
    # single-combined-descriptor enumeration (which would place a whole-molecule
    # (1,1-13C2) descriptor before the multiplier). Non-hydrogen nuclides only
    # (deuterium bracket hazard -- see the function's docstring); fails through
    # otherwise.
    uniform = _decorate_uniform_multiplier(skeleton, keys, original, stripped)
    if uniform is not None:
        return uniform

    # A3 Task 3, letter-locant class): a repeated IDENTICAL
    # letter-locant multiplier ANYWHERE in the skeleton (e.g. the
    # ``N,N,N-trimethyl`` of a quaternary ammonium salt cation) -- distinct
    # from the numeric leading-multiplier shape ``_decorate_demultiplied``
    # already handles. See that function's docstring for why.
    letter_demux = _decorate_letter_multiplier(skeleton, keys, original, stripped)
    if letter_demux is not None:
        return letter_demux

    #: the descriptor "is inserted before the part of the compound that
    # is isotopically substituted". For a whole-parent label that is the front of
    # the name; for a labeled part after substituent prefixes (the labeled C of
    # trichloro(12C1)methane) it sits before the parent stem. Orthonym does not
    # parse the skeleton's grammar, so it enumerates every insertion offset and
    # lets the OPSIN round-trip oracle pick the unique correct one (probe: only
    # ONE offset round-trips per (structure, descriptor)). Front (0) is tried
    # first so the front placement is preferred when it is valid.
    #
    # Locant-form order /: the no-locant form (None) is
    # tried first — locants are omitted when the position is unambiguous; then
    # integer locants ascending (lowest-locant-first). Among round-trippers,
    # /.4.3 (higher Z then higher mass at the lower locant) breaks ties.
    # (A3: the search itself now lives in the module-level
    # ``_find_best_placement`` so the multi-attachment path below can reuse it
    # unchanged; this closure is a thin adapter preserving the existing
    # ``(candidate, loc_rank)`` contract used by the rest of this function.)
    letter_locants = _letter_locant_candidates(original, label_map)

    def _enumerate(skel):
        """Best (candidate, loc_rank) for one skeleton spelling, or (None, None).

        ``loc_rank`` is -1 when the winning descriptor needs NO locant, else the
        locant. That value is the trigger: see below.
        """
        candidate, _off, _desc, loc_rank = _find_best_placement(
            skel, keys, original, n_pos, allow_front_hyphen=True,
            extra_locants=letter_locants)
        return candidate, loc_rank

    best, loc_rank = _enumerate(skeleton)
    if best is None:
        # A3 Task 1 + multi-position placement): the single
        # combined descriptor above shares ONE locant across every labelled
        # atom of a nuclide, which cannot express labels sitting at more than
        # one structurally distinct position (per-D-glycine: D on the amino
        # N, the alpha C, AND the carboxyl O). Try, in order:
        # (a) one descriptor per attachment group, on THIS skeleton;
        # (b) if the skeleton itself has no slot at all for some part of
        # the label (a retained parent with an unnumbered position,
        #, retry every placement strategy against a MORE
        # fully systematic alternate parent.
        groups = _partition_by_attachment(original, label_map)
        if len(groups) > 1:
            multi = _decorate_multi_position(skeleton, groups, original, n_pos)
            if multi is not None:
                return multi
        # (b') ONE nuclide at several DISTINCT positions on this parent
        # (1,3,7-2H3 on a von Baeyer cage / a multiply-labelled ring) -- neither
        # the single-shared-locant enumeration nor the attachment-group split
        # above expresses it.
        ml = _decorate_distinct_multi_locant(skeleton, keys, original, label_map, n_pos)
        if ml is not None:
            return ml
        # (b'') MIXED per-nuclide locants: an ambiguous nuclide that
        # needs a locant beside a unique one that does not -- (1-13C,15N) -- which
        # neither the single-shared-locant enumeration nor the same-nuclide
        # distinct-multi-locant path above can express.
        mx = _decorate_mixed_locant(skeleton, keys, original, n_pos,
                                    stripped_smiles, namer)
        if mx is not None:
            return mx
        alt_skeleton = _systematic_parent_fallback(stripped)
        if alt_skeleton and alt_skeleton != skeleton:
            alt_demux = _decorate_demultiplied(alt_skeleton, keys, original, stripped)
            if alt_demux is not None:
                return alt_demux
            alt_letter_demux = _decorate_letter_multiplier(
                alt_skeleton, keys, original, stripped)
            if alt_letter_demux is not None:
                return alt_letter_demux
            alt_best, _alt_loc_rank = _enumerate(alt_skeleton)
            if alt_best is not None:
                return alt_best
            if len(groups) > 1:
                alt_multi = _decorate_multi_position(alt_skeleton, groups, original, n_pos)
                if alt_multi is not None:
                    return alt_multi
            alt_ml = _decorate_distinct_multi_locant(
                alt_skeleton, keys, original, label_map, n_pos)
            if alt_ml is not None:
                return alt_ml
        # isotope-INDUCED stereocentre: the constitution round-trips but
        # the full-stereo oracle does not, because a nuclide made an otherwise
        # achiral centre chiral. Enumerate the stereodescriptor prefix (oracle-
        # driven, full-InChIKey gated). Tried last so a plain (non-stereo) row is
        # never charged its cost.
        stereo = _decorate_isotope_stereo(
            skeleton, keys, original, stripped, n_pos, letter_locants)
        if stereo is not None:
            return stereo
        # De-multiplication (single AND mixed nuclides) was already attempted
        # ahead of this enumeration by _decorate_demultiplied; nothing else to
        # try -> fail closed (never a wrong labeled name).
        return None

    # ── (the Blue Book), the conditional locant restoration ────────────────
    # Verbatim: "In preferred IUPAC names, locants are omitted if no locants are
    # necessary in unmodified names. However, if isotopic modification requires a locant
    # to specify its position, then all locants must be specified and none are omitted."
    # Its own example prints the elided form as the REJECTED one (the Blue Book):
    # 13CH3-CH2-OH (2-13C)ethan-1-ol [not (2-13C)ethanol]
    #
    # `loc_rank >= 1` is precisely "the isotopic modification requires a locant": the
    # enumeration tries the locant-FREE descriptor first, so a locant is only reached
    # when the locant-free form failed to round-trip -- i.e. when the position genuinely
    # has to be stated. When `loc_rank == -1` no locant was needed, the condition in the
    # rule is not met, and the parent keeps its licensed omission. That is what makes
    # `(13C1)benzenehexol` correct, the Blue Book -- all six ring positions are one
    # orbit, so there is only one isotopomer) while `(13C2)benzenehexol` is not.
    #
    # The skeleton above was named from the isotope-STRIPPED molecule, so the
    # licences could not see the label and elided freely. Re-name it inside the ambient
    # scope so they decline, then re-enumerate against the locanted spelling.
    # A letter locant amide ``N``) is a non-int ``loc_rank``; it is
    # already forced INTO the descriptor, so the parent-relocanting re-render below
    # (which handles integer positions on the parent chain/ring) does not apply --
    # skip it for non-int ranks.
    if isinstance(loc_rank, int) and loc_rank >= 1:
        from ..assembly.locant_omission import forced_locant_scope
        with forced_locant_scope("isotope"):
            skel2 = _flagged_systematic_namer(namer).name(stripped_smiles)
        if skel2 and "unknown" not in skel2.lower() and skel2 != skeleton:
            best2, _ = _enumerate(skel2)
            # Fail toward the locanted spelling only if it still round-trips; never
            # trade a verified name for an unverified one.
            if best2 is not None:
                return best2
    return best
