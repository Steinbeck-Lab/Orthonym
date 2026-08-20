"""Isotopically substituted compound names (IUPAC P-82.2.1 + P-45.4).

Fail-closed decorator layer. Orthonym's core pipeline strips isotope labels
(RDKit perception ignores GetIsotope for skeleton naming), so a labeled mol
would otherwise emit the UNLABELED name — a wrong PIN. This module runs as an
EARLY branch in Orthonym.name(): it strips the labels, names the skeleton in
systematic style (locanted parents), then re-derives the isotopic descriptor
by an INTERNAL ORACLE (OPSIN-parse a candidate name; compare rdkit canonical
SMILES — with isotopes — to the original). If no candidate round-trips, it
returns None and the label is never mis-placed.

BB P-82.2.1 (BlueBookV2.md:43718): the nuclide symbol(s) in parentheses,
preceded by any necessary locant(s), are inserted before the isotopically
substituted part; polysubstitution count is a right subscript to the symbol.
BB P-45.4.1/.4.2/.4.3 (BlueBookV2.md:22212-22232): lowest locants to modified
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
    return any(a.GetIsotope() != 0 for a in mol.GetAtoms())


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
# P-82.2.1 — isotopic descriptor formatting
# ---------------------------------------------------------------------------


def nuclide_symbol(mass: int, element: str) -> str:
    """Nuclide symbol with the mass number as a leading integer (ASCII form).

    Orthonym emits ASCII (no <sup>); OPSIN accepts the leading-digit form
    (probe: (2-14C)ethan-1-ol -> C([14CH3])O). P-82.2.1.
    """
    return f"{mass}{element}"


def format_isotope_descriptor(groups) -> str:
    """Build the parenthesized isotopic descriptor from grouped labels.

    ``groups`` = list of (locant, mass, element, count):
      locant None  -> no leading locant (single-position parent / front
                      descriptor, e.g. (2H3)methoxybenzene, (12C1)methane).
      locant int   -> the count locants are repeated then hyphen-joined to the
                      nuclide, e.g. (2,2,2-2H3), (2-14C1).
    Multiple groups at (possibly) the same place are cited alphabetically by
    element then by mass number (P-82.2.1 / P-82.3); groups are comma-joined.
    The count subscript is ALWAYS emitted (P-82.2.1 line 43720 + tritium
    parseability, Task 4).
    """
    def _one(locant, mass, element, count):
        sym = nuclide_symbol(mass, element)
        if locant is None:
            return f"{sym}{count}"
        # BB: the locant is repeated once per substituted atom at that position
        # for a grouped multi-count token (2,2,2-2H3); for count 1 a single
        # locant + hyphen (2-14C1).
        loc_part = ",".join(str(locant) for _ in range(count))
        return f"{loc_part}-{sym}{count}"

    # P-82.2.1: alphabetical by element symbol, then by mass number, then locant.
    ordered = sorted(groups, key=lambda g: (g[2], g[1], g[0] if g[0] is not None else -1))
    inner = ",".join(_one(*g) for g in ordered)
    return f"({inner})"


# ---------------------------------------------------------------------------
# P-45.4.1 / P-82.2.1 — internal-oracle isotopic-descriptor placement
# ---------------------------------------------------------------------------
# The skeleton name fixes the parent numbering, but Orthonym does NOT expose a
# reliable atom->locant map for acyclic parents (retrieve_confidence()
# ['atom_to_locant'] is None for CCO). Rather than re-perceive the chain
# independently (which risks diverging from the name's own numbering — a
# fail-closed violation), the decorator treats the skeleton name's locant space
# as opaque and lets OPSIN adjudicate: it enumerates candidate descriptor
# placements, OPSIN-parses each candidate, and accepts the one whose relabeled
# structure is rdkit-canonical-identical (isotopes retained) to the original.
# Because OPSIN itself renumbers to give the isotope its lowest legal locant,
# the round-trip test IS the P-45.4.1 / P-31.1.4.3.4(i) implementation — the
# wrong-locant candidate simply fails to reproduce the structure.

_ELEMENT_Z = {  # atomic numbers for P-45.4.2 tie-break
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Se": 34, "Br": 35, "I": 53,
}


def _opsin_parse(candidate_name: str) -> Optional[str]:
    """Parse ``candidate_name`` to SMILES via the repo's shared OPSIN CLI
    helper (validation.opsin_roundtrip.opsin_parse — the SAME loader the name()
    validity gate uses). Fail-closed (None) on any error / no JAR."""
    try:
        from ..validation.opsin_roundtrip import opsin_parse
        return opsin_parse(candidate_name)
    except Exception:
        return None


def _isotope_round_trips(candidate_name: str, original_mol: Chem.Mol) -> bool:
    """True iff OPSIN parses ``candidate_name`` to a structure that is rdkit-
    canonical-identical (isotopes retained) to ``original_mol``. Fail-closed
    on parse failure / no JAR."""
    smi = _opsin_parse(candidate_name)
    if not smi:
        return False
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return False
    try:
        return Chem.MolToSmiles(parsed) == Chem.MolToSmiles(original_mol)
    except Exception:
        return False


#: A parent hydride's indicated-hydrogen prefix, e.g. the ``1H-`` of
#: ``1H-pyrrole``. The hyphen is part of the match because the descriptor goes
#: *in place of* it -- see ``_insertion_offsets``.
_INDICATED_H_PREFIX_RE = re.compile(r"-\d+H-")


def _insertion_offsets(skel: str) -> List[int]:
    """Offsets in ``skel`` at which an isotopic descriptor may be inserted.

    P-82.2.1: the descriptor "is inserted before the part of the compound that is
    isotopically substituted".  The enumeration does not parse the skeleton's
    grammar -- it offers every plausible offset and lets the OPSIN round-trip
    oracle pick the one that is right -- so the only thing that matters here is
    that no legal placement is left out of the candidate set.

    Alphabetic starts cover a bare parent stem.  They do NOT cover a parent
    carrying an indicated hydrogen: the slot before ``1H-pyrrole`` begins with a
    digit, so a labelled molecule that also has substituent prefixes had nowhere
    legal to put the descriptor and failed closed -- ``[2H]C1=C(N(C=C1)C)[N+](=O)[O-]``
    named nothing at all, though its unlabelled skeleton names fine.

    P-82 prints the construction four times, every one a PIN
    (``BlueBookV2/BlueBookV2.md:43790``-``:43796``)::

        (15N)-1H-indole (PIN)
        2,3-dihydro(15N)-1H-indole (PIN)
        2,3-dihydro(2,3-2H2,15N)-1H-indole (PIN)

    -- descriptor, then a hyphen, then the indicated-hydrogen prefix.  So the
    descriptor replaces the hyphen that introduces that prefix, and the offset to
    offer is the hyphen's own index: ``1-methyl-2-nitro`` + ``(3-2H1)`` +
    ``-1H-pyrrole``.

    Every candidate built from these offsets is still OPSIN-round-trip gated by
    the caller, so a surplus offset can only ever be discarded, never shipped.
    """
    offsets = [i for i in range(1, len(skel)) if skel[i].isalpha()]
    offsets.extend(m.start() for m in _INDICATED_H_PREFIX_RE.finditer(skel))
    return sorted(offsets)


def _p4542_p4543_key(groups) -> tuple:
    """P-45.4.2 then P-45.4.3 ordering key for a candidate placement.

    Prefer the placement giving the lowest locant to the HIGHER atomic number
    (P-45.4.2), then to the HIGHER mass number (P-45.4.3). Implemented as: for
    each group sorted by locant ascending, emit (-Z, -mass); the lexicographically
    smallest key is the preferred placement. BB:22220 / BB:22226 / BB:3336(i).
    """
    key = []
    for locant, mass, el, _count in sorted(
        groups, key=lambda g: (g[0] if g[0] is not None else 1 << 30)
    ):
        z = _ELEMENT_Z.get(el, 0)
        key.append((-z, -mass))
    return tuple(key)


def _decorate_demultiplied(skeleton, keys, original, stripped) -> Optional[str]:
    """Generalized P-82.2.2.1 de-multiplication (single OR mixed nuclides).

    Split a leading-locant simple multiplier (e.g. ``1,2-dimethoxyethane``) and
    scope a DISTINCT nuclide descriptor to each labeled copy, leaving at most one
    copy bare. Handles both:
      * single-group  — one labeled copy among bare copies
                        (P-82.2.2.1 + P-45.4.1, e.g. 1-(13C1)methoxy-2-methoxyethane)
      * mixed         — DISTINCT nuclides on DISTINCT copies, where — because the
                        parent is symmetric — BOTH numbering directions round-trip
                        and P-45.4.2/.4.3 must choose (e.g.
                        1-(18O1)methoxy-2-(13C1)methoxyethane: 18O Z=8 > 13C Z=6).

    Every candidate is OPSIN-RT gated with isotopes retained; among the
    round-trippers, P-45.4.1 (lowest locants to the modified copies) then
    P-45.4.2/.4.3 (higher atomic number then higher mass number at the lower
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

    # One nuclide instance per labeled copy (count 1 each).
    labels = []
    for (mass, el), count in keys:
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
                        desc = format_isotope_descriptor([(None, mass, el, 1)])
                        segs.append(f"{L}-{desc}{base}")
                    else:
                        segs.append(f"{L}-{base}")
                cand = "-".join(segs) + parent
                if _isotope_round_trips(cand, original):
                    groups = [(L, loc_to_label[L][0], loc_to_label[L][1], 1)
                              for L in sorted(loc_to_label)]
                    p4541 = tuple(sorted(loc_to_label))
                    winners.append((p4541, _p4542_p4543_key(groups), k, cand))
    if not winners:
        return None
    # P-45.4.1 (lowest locants to modified copies) then P-45.4.2/.4.3; the
    # trailing candidate STRING term (w[3]) makes the winner fully deterministic
    # regardless of assignment-iteration order -- it is load-bearing, not a
    # decorative tiebreak. Do not drop it.
    winners.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
    return winners[0][3]


# ─────────────────────────────────────────────────────────────────────────────────────
# WITHDRAWN 2026-07-28: a whole-molecule isotopomer-ambiguity test.
#
# It was built and wired here, and it REGRESSED a correct name, so it is recorded rather
# than kept. It asked "can this nuclide multiset be placed on the skeleton in more than
# one non-isomorphic way?" and forced a locant when so. That ignores **P-82.2.1**
# (``:44182``, verbatim): the descriptor is inserted *"before the part of the compound
# that is isotopically substituted"* -- so the descriptor's POSITION carries its scope,
# and ambiguity has to be judged inside that scope, not over the whole molecule.
#
# The witness that refuted it: ``[13CH3]OC(C)=O`` -> ``(13C1)methyl acetate``. Methyl
# acetate has three distinct carbons, so a whole-molecule test says "ambiguous" and forces
# ``(1-13C1)methyl acetate``. But the descriptor sits before ``methyl``, whose scope is a
# single carbon, so the name is already unique. ``:7492`` settles it:
# ``1,2-di[(13C)methyl]benzene (PIN. P-82.2.1)`` -- descriptor inside the brackets,
# scoped to ``methyl``, no locant on the label.
#
# Judging scope correctly needs awareness of the name's grammar, which this module does
# not have (it enumerates insertion offsets and lets the round-trip oracle choose). So the
# general rule is NOT implemented. What IS implemented below is the half with a verbatim
# Blue Book anchor: when the descriptor itself needs a locant, P-82.6.1.1 restores the
# parent's locants. See the open-defect note in
# 
# ─────────────────────────────────────────────────────────────────────────────────────


def decorate_isotopic_name(smiles: str, style: str, namer) -> Optional[str]:
    """Fail-closed isotopic-substitution PIN (P-82.2.1 + P-45.4).

    1. Parse + strip isotopes; if none, None.
    2. Name the skeleton in systematic style (locanted parents).
    3. Enumerate candidate descriptors (lowest-locant-first, P-45.4.1) and
       accept the first whose OPSIN round-trip reproduces the original mol.
    4. None if nothing round-trips (never a wrong labeled name).
    """
    original = Chem.MolFromSmiles(smiles)
    if original is None or not has_isotopes(original):
        return None
    stripped, label_map = strip_isotopes(original)
    if not label_map:
        return None
    # P-82.6.1.1 (``:44180``). Everything below names the isotope-STRIPPED molecule,
    # so no structural test further down can see that a label exists -- measured: at
    # the P-14.3.4 substituent licences every ``GetIsotope()`` in scope reads 0 even
    # for a 13C input. Declare it ambiently instead, UNCONDITIONALLY, so a licence
    # about to empty a scope of all its locants can decline. This is deliberately
    # weaker than ``forced_locant_scope`` (entered further down only once a locant is
    # known to be REQUIRED): the weaker flag must not gag the omissions that stay
    # correct under P-82.6.1.3, i.e. ``(13C1)benzenehexol`` and ``(2H6)benzene``.
    from ..assembly.locant_omission import isotopic_naming_scope
    with isotopic_naming_scope("isotope"):
        return _decorate_isotopic_name_inner(
            smiles, style, namer, original, stripped, label_map)


def _decorate_isotopic_name_inner(smiles, style, namer, original, stripped,
                                  label_map) -> Optional[str]:
    """Body of :func:`decorate_isotopic_name`, run inside the ambient isotopic scope."""
    stripped_smiles = Chem.MolToSmiles(stripped)
    # Skeleton in systematic style so parents carry locants (ethan-1-ol, not
    # ethanol) — the descriptor's locant needs a locanted parent.
    skeleton = namer.name(stripped_smiles) if style == "systematic" else None
    if skeleton is None:
        # Re-name in systematic regardless (the descriptor needs locants).
        from ..namer import Orthonym
        skeleton = Orthonym(style="systematic").name(stripped_smiles)
    from ..errors import is_failure_name
    if not skeleton or "unknown" in skeleton.lower() or is_failure_name(skeleton):
        return None

    # Group labels by (mass, element) -> count; element read from the ORIGINAL
    # mol at each labeled atom index (label_map keys index the original).
    by_key: Dict[Tuple[int, str], int] = {}
    for idx, mass in label_map.items():
        el = original.GetAtomWithIdx(idx).GetSymbol()
        by_key[(mass, el)] = by_key.get((mass, el), 0) + 1

    n_pos = stripped.GetNumHeavyAtoms()

    # Alphabetical by element then mass (P-82.2.1 citation order).
    keys = sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0]))

    # P-82.2.2.1 de-multiplication FIRST (single OR mixed nuclides). This is the
    # ONLY path that can place DISTINCT nuclides on DISTINCT copies of a
    # de-multiplied substituent (the mixed 18O/13C case), and it is bounded, so
    # routing it ahead of the single-combined-descriptor enumeration below both
    # (a) fixes that case and (b) avoids the fruitless JVM-per-candidate
    # exhaustion that enumeration performs on a de-mult skeleton. Fail-through to
    # the enumeration when the skeleton is not a de-mult form.
    demux = _decorate_demultiplied(skeleton, keys, original, stripped)
    if demux is not None:
        return demux

    def _descriptor(locant):
        groups = [(locant, mass, el, count) for (mass, el), count in keys]
        return format_isotope_descriptor(groups)

    # P-82.2.1: the descriptor "is inserted before the part of the compound that
    # is isotopically substituted". For a whole-parent label that is the front of
    # the name; for a labeled part after substituent prefixes (the labeled C of
    # trichloro(12C1)methane) it sits before the parent stem. Orthonym does not
    # parse the skeleton's grammar, so it enumerates every insertion offset and
    # lets the OPSIN round-trip oracle pick the unique correct one (probe: only
    # ONE offset round-trips per (structure, descriptor)). Front (0) is tried
    # first so the front placement is preferred when it is valid.
    insert_offsets = [0] + _insertion_offsets(skeleton)
    # dedupe preserving order (front-first)
    seen_off = set()
    offsets = []
    for o in insert_offsets:
        if o not in seen_off:
            seen_off.add(o)
            offsets.append(o)

    # Locant-form order (P-45.4.1 / P-14.3.4): the no-locant form (None) is
    # tried first — locants are omitted when the position is unambiguous; then
    # integer locants ascending (lowest-locant-first). Among round-trippers,
    # P-45.4.2/.4.3 (higher Z then higher mass at the lower locant) breaks ties.
    def _enumerate(skel):
        """Best (candidate, loc_rank) for one skeleton spelling, or (None, None).

        ``loc_rank`` is -1 when the winning descriptor needs NO locant, else the
        locant. That value is the P-82.6.1.1 trigger: see below.
        """
        _offs = [0] + _insertion_offsets(skel)
        _seen, _o = set(), []
        for x in _offs:
            if x not in _seen:
                _seen.add(x)
                _o.append(x)
        won = []
        for locant in [None] + list(range(1, n_pos + 1)):
            desc = _descriptor(locant)
            for off in _o:
                candidate = skel[:off] + desc + skel[off:]
                if _isotope_round_trips(candidate, original):
                    groups = [(locant, mass, el, count) for (mass, el), count in keys]
                    # locant None ranks as lowest for P-45.4.1 (unambiguous omit).
                    loc_rank = -1 if locant is None else locant
                    won.append((_p4542_p4543_key(groups), loc_rank, off, candidate))
            if won:
                # P-45.4.1: the first locant-form that yields ANY round-tripper is
                # the lowest-locant form (None precedes integers, integers ascend);
                # do not consider higher-locant forms once a lower one succeeds.
                break
        if not won:
            return None, None
        # Among equally-low-locant round-trippers, P-45.4.2/.4.3 then front-first.
        won.sort(key=lambda w: (w[0], w[1], w[2]))
        return won[0][3], won[0][1]

    best, loc_rank = _enumerate(skeleton)
    if best is None:
        # De-multiplication (single AND mixed nuclides) was already attempted
        # ahead of this enumeration by _decorate_demultiplied; nothing else to
        # try -> fail closed (never a wrong labeled name).
        return None

    # ── P-82.6.1.1 (BB:44180), the conditional locant restoration ────────────────
    # Verbatim: "In preferred IUPAC names, locants are omitted if no locants are
    # necessary in unmodified names. However, if isotopic modification requires a locant
    # to specify its position, then all locants must be specified and none are omitted."
    # Its own example prints the elided form as the REJECTED one (BB:44186):
    #     13CH3-CH2-OH   (2-13C)ethan-1-ol   [not (2-13C)ethanol]
    #
    # `loc_rank >= 1` is precisely "the isotopic modification requires a locant": the
    # enumeration tries the locant-FREE descriptor first, so a locant is only reached
    # when the locant-free form failed to round-trip -- i.e. when the position genuinely
    # has to be stated. When `loc_rank == -1` no locant was needed, the condition in the
    # rule is not met, and the parent keeps its licensed omission. That is what makes
    # `(13C1)benzenehexol` correct (P-82.6.1.3, BB:44202 -- all six ring positions are one
    # orbit, so there is only one isotopomer) while `(13C2)benzenehexol` is not.
    #
    # The skeleton above was named from the isotope-STRIPPED molecule, so the P-14.3.4
    # licences could not see the label and elided freely. Re-name it inside the ambient
    # P-14.3.3 scope so they decline, then re-enumerate against the locanted spelling.
    if loc_rank is not None and loc_rank >= 1:
        from ..assembly.locant_omission import forced_locant_scope
        from ..namer import Orthonym
        with forced_locant_scope("isotope"):
            skel2 = Orthonym(style="systematic").name(stripped_smiles)
        if skel2 and "unknown" not in skel2.lower() and skel2 != skeleton:
            best2, _ = _enumerate(skel2)
            # Fail toward the locanted spelling only if it still round-trips; never
            # trade a verified name for an unverified one.
            if best2 is not None:
                return best2
    return best
