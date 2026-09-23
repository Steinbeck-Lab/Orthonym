""" -- omission of locants. The ONE place the licences are decided.

 Phase C tranche B. Pure module: RDKit mols in, booleans out. No I/O, no naming.

⚠ **DENY BY DEFAULT.** ```` "Citation of locants"
(``the Blue Book Blue Book``) is the rule:

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

```` (``:2871``) then grants narrow *licences*, and its own preamble says
*"for absolute clarity in preferred IUPAC names it is necessary to be prescriptive
about when omission of locants is permissible."* So:

* **never** write a locant-stripping pass -- every omission is a positively-licensed
  structural predicate;
* **fail toward retaining the locant** -- anything this module cannot positively
  establish returns ``False`` (or ``True`` from:func:`scope_forces_locants`, which is
  the same direction);
* a licence is evaluated **per enclosing-mark scope**. One essential locant anywhere in
  a scope restores *every* locant in that scope -- which is why
  ``1-chloro-2-(pentafluoroethyl)benzene`` elides inside the parentheses and keeps
  ``1,2`` outside.

★ **THE SELF-VALIDATING BOUNDARY PAIR** -- the reason these predicates count
HYDROGENS, never positions::

    :7625 benzenehexol (PIN, (not benzenehexaol) -> OMITS
    :54823 "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
              group of cyclitols." -> RETAINS

Same six OH, same ring size. A benzene ring carbon has ONE substitutable H, so six OH
*completely* substitutes the ring and ```` fires. A cyclohexane ring carbon
has TWO, so six OH is *partial* substitution and the ``:3009`` counter-clause restores
every locant. A predicate that counts positions gets this pair wrong.

The class is **OPEN** (ring size, ring element and substituent identity are all
unbounded), so a finite table of names or spellings would be wrong on its complement by
construction. Structural predicates only.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from rdkit import Chem

__all__ = [
    "CHALCOGENS",
    "substitutable_h_count",
    "substitutable_positions",
    "l5_uniform_complete",
    "l3_one_kind_of_substitutable_h",
    "l4_no_isomer_by_relocation",
    "l3_monosubstituted_locant_omitted",
    "l3_locant_omitted_for_parent_atoms",
    "l6_all_substitutable_h_share_one_locant",
    "scope_forces_locants",
    "forced_locant_scope",
    "locants_are_forced",
    "forced_locant_reason",
    "isotopic_naming_scope",
    "scope_has_isotopic_modification",
    "isotope_parent_positional_scope",
    "parent_scope_has_positional_isotope",
]

#: ``:3007`` -- "Except for hydrogen atoms attached to chalcogen atoms, such as in
#: acids, alcohols,...". Group 16: O, S, Se, Te (Po is not a nomenclature case).
CHALCOGENS = frozenset({"O", "S", "Se", "Te"})


def _is_aldehyde_formyl_carbon(atom) -> bool:
    """``:3007`` -- "and to the carbon atoms of formyl groups (aldehydes)".

    Deliberately **NARROW** (derivation F1 finding 5 -- the earlier carve-out was too
    broad and swallowed acids/amides, whose C-H is a different case). All four must
    hold: carbon; exactly one H; exactly one double bond to O; and **no** single-bonded
    O/N/S neighbour. The last clause is what keeps formic acid's carbon substitutable.
    """
    if atom.GetSymbol() != "C":
        return False
    if atom.GetTotalNumHs() != 1:
        return False
    n_carbonyl = 0
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        sym = other.GetSymbol()
        if bond.GetBondType() == Chem.BondType.DOUBLE and sym == "O":
            n_carbonyl += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and sym in ("O", "N", "S"):
            # An acid / ester / amide / thioacid carbon, not a formyl carbon.
            return False
    return n_carbonyl == 1


def substitutable_h_count(mol, idx: int) -> int:
    """How many hydrogens on atom ``idx`` are *substitutable* per ``:3007``.

    Zero for a hydrogen on a chalcogen (acid/alcohol/thiol OH, SH, SeH, TeH) and zero
    for the H of an aldehyde formyl carbon; otherwise the atom's total H count.

    Returns ``0`` -- never raises -- for a missing mol or an out-of-range index, so a
    caller that fails to establish the structure gets the deny-by-default answer.
    """
    if mol is None:
        return 0
    try:
        atom = mol.GetAtomWithIdx(int(idx))
    except (OverflowError, RuntimeError, ValueError, TypeError):
        return 0
    n_h = atom.GetTotalNumHs()
    if n_h <= 0:
        return 0
    if atom.GetSymbol() in CHALCOGENS:
        return 0
    if _is_aldehyde_formyl_carbon(atom):
        return 0
    return n_h


def substitutable_positions(mol) -> FrozenSet[int]:
    """``:3007``. Atom indices carrying at least one substitutable hydrogen."""
    if mol is None:
        return frozenset()
    return frozenset(
        a.GetIdx() for a in mol.GetAtoms() if substitutable_h_count(mol, a.GetIdx())
    )


def l5_uniform_complete(
    mol,
    *,
    decoration_of: Optional[Mapping[int, Optional[str]]],
    counts: Optional[Mapping[int, int]] = None,
) -> bool:
    """```` (``:3007``) plus its counter-clause (``:3009``).

        "All locants are omitted in compounds or substituent groups in which all
         substitutable positions are completely substituted or modified, for example,
         by hydro, in the same way."

        "In case of partial substitution or modification, all numerical prefixes must
         be indicated."

    ``mol`` is the **parent hydride / parent compound**, not the decorated molecule --
    the rule speaks of *its* substitutable positions. ``decoration_of`` maps an atom
    index of that parent to a decoration key (the prefix or suffix morpheme).

    True iff **every** substitutable position of ``mol`` has **all** of its
    substitutable hydrogens decorated, and every decoration is the SAME KIND.

    ``counts`` is an optional per-atom decoration multiplicity, defaulting to 1. It
    exists because ``decoration_of`` maps one key per atom and cannot otherwise express
    a doubly decorated position -- ``heptafluorobutanoic acid`` (``:3017``) puts two F
    on each of two CH2 carbons. Omitting it means "one decoration per atom", so a
    position with two substitutable hydrogens is *partial* and the licence is denied:
    that is exactly the ``cyclohexane-1,2,3,4,5,6-hexols`` boundary (``:54823``).

    Decorations at NON-substitutable positions are permitted and still count toward the
    "in the same way" test -- ``decahydronaphthalene`` (``:3013``) hydrogenates the two
    bridgeheads, which carry no substitutable H of their own.
    """
    if mol is None or not decoration_of:
        return False
    positions = substitutable_positions(mol)
    if not positions:
        # Nothing to "completely substitute" -- fail toward retaining.
        return False

    # "in the same way": one decoration kind across the whole scope. Checked over ALL
    # entries, including any at non-substitutable positions, because a differing kind
    # anywhere makes the substitution non-uniform.
    kinds = set(decoration_of.values())
    if len(kinds) != 1:
        return False
    only = next(iter(kinds))
    if only is None or not str(only).strip():
        return False

    n_atoms = mol.GetNumAtoms()
    for idx in decoration_of:
        if not isinstance(idx, int) or isinstance(idx, bool):
            return False
        if idx < 0 or idx >= n_atoms:
            return False

    for idx in positions:
        if idx not in decoration_of:
            return False                       # partial ->:3009
        n_dec = 1 if counts is None else counts.get(idx, 1)
        if not isinstance(n_dec, int) or n_dec != substitutable_h_count(mol, idx):
            return False                       # partial AT this position ->:3009
    return True


def _orbits(mol) -> Optional[list]:
    """Canonical-rank orbits, ties NOT broken -- the constitutional equivalence classes.

    Global constraint 5: equivalence is ``CanonicalRankAtoms(breakTies=False)``, never a
    SMARTS-symmetry heuristic.
    """
    try:
        return list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        return None


def l3_one_kind_of_substitutable_h(mol) -> bool:
    """```` (``:2939``).

        "The locant is omitted in monosubstituted symmetrical parent hydrides or parent
         compounds where there is only one kind of substitutable hydrogen."

    True iff every substitutable hydrogen of ``mol`` lies in ONE
    ``CanonicalRankAtoms(breakTies=False)`` orbit -- benzene, cyclohexane, urea
    (``:2943`` ``methylurea``), pyrazine (``:2949`` ``pyrazinecarboxylic acid``).

    This speaks only about the parent. The caller must separately establish that the
    substitution really is *mono* and that nothing else in the scope forces locants
    (:func:`scope_forces_locants`).
    """
    positions = substitutable_positions(mol)
    if not positions:
        return False
    ranks = _orbits(mol)
    if ranks is None:
        return False
    return len({ranks[i] for i in positions}) == 1


# --------------------------------------------------------------------------------- #
# -- the ISOMER-COUNT licence #
# --------------------------------------------------------------------------------- #
_BOND_ORDER_TO_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

#: Hard ceiling on the placement enumeration. Exceeding it DENIES (fail closed)
#: rather than costing unbounded time; every Blue Book example of this rule has at
#: most 3 candidate positions and at most 2 decorations.
_L4_MAX_ASSIGNMENTS = 4096


def _l4_components(mol, parent_set: FrozenSet[int]
                   ) -> Optional[List[Tuple[int, int, int]]]:
    """The decorations, read STRUCTURALLY off ``mol`` as ``(anchor, position, order)``.

    A "decoration" is one connected component of the atoms OUTSIDE ``parent_set``
    -- which is what makes this predicate cover *"suffixes and/or prefixes"* alike
    (``:2953``) without being told which is which: a ``-one`` oxygen and a
    ``methyl`` carbon are both simply atoms outside the parent.

    ``anchor`` is the component atom bonded to the parent, ``position`` the parent
    atom it is bonded to, ``order`` the integer bond order (how many parent
    hydrogens that decoration consumes -- 1 for ``chloro``, 2 for ``ethylidene``,
    3 for ``methylidyne``).

    Returns None -- deny -- unless EVERY component attaches by exactly one bond to
    exactly one parent atom. A component joined twice is a bridge or a fused ring,
    for which "moving it to another position" is not the operation ``:2953``
    describes, and the single-vertex relocation below would not be faithful to it.
    """
    n = mol.GetNumAtoms()
    outside = [i for i in range(n) if i not in parent_set]
    if not outside:
        return None
    outside_set = set(outside)

    # Connected components of the outside atoms.
    unseen = set(outside_set)
    comps: List[Tuple[int, int, int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        comp = {start}
        unseen.discard(start)
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                j = nb.GetIdx()
                if j in outside_set and j in unseen:
                    unseen.discard(j)
                    comp.add(j)
                    stack.append(j)
        # Exactly one bond from this component into the parent.
        links = []
        for i in sorted(comp):
            for bond in mol.GetAtomWithIdx(i).GetBonds():
                j = bond.GetOtherAtomIdx(i)
                if j in parent_set:
                    order = bond.GetBondTypeAsDouble()
                    if order not in (1.0, 2.0, 3.0):
                        return None          # aromatic/dative -> cannot relocate
                    links.append((i, j, int(order)))
        if len(links) != 1:
            return None
        comps.append(links[0])
    return comps


def _l4_relocated_key(mol, comps, assignment, capacity) -> Optional[str]:
    """Canonical SMILES of ``mol`` with every decoration moved to ``assignment``.

    ★ The relocation is performed on the **REAL molecule**: the attachment bond is
    detached from its current parent atom and re-attached, at the same bond order,
    to the assigned one. Nothing is modelled by a placeholder, so the constitution
    compared is the constitution the rule speaks about, with the decorations' own
    structure, isotopes and charges intact.

    Parent hydrogen counts are frozen explicitly first (``SetNoImplicit``), so the
    surgery cannot silently move a hydrogen and make two genuinely different
    constitutions compare equal.

    Returns None on any failure to build or sanitize -- the caller denies.
    """
    rw = Chem.RWMol(mol)
    used: Dict[int, int] = {}
    for (_anchor, _old, order), new_pos in zip(comps, assignment):
        used[new_pos] = used.get(new_pos, 0) + order
    for pos, cap in capacity.items():
        left = cap - used.get(pos, 0)
        if left < 0:
            return None
        atom = rw.GetAtomWithIdx(pos)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(left)
    for (anchor, old_pos, order), new_pos in zip(comps, assignment):
        if old_pos == new_pos:
            continue
        rw.RemoveBond(old_pos, anchor)
        rw.AddBond(new_pos, anchor, _BOND_ORDER_TO_TYPE[order])
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:                          # noqa: BLE001 -- deny-by-default
        return None


def l4_no_isomer_by_relocation(
    mol,
    parent_atoms,
    *,
    prefix_locants,
    suffix_locants,
    stereo_text,
    has_indicated_h,
    has_isotope,
    is_multiplicative: bool = False,
    is_ring_assembly: bool = False,
    has_skeletal_replacement: bool = False,
) -> bool:
    """§**** (``:2953``) -- the ISOMER-COUNT licence.

        "Locants are omitted when no isomer can be generated by moving suffixes
         and/or prefixes (if any) from their position to another or by
         interchanging them between two different positions."

    True => every locant in this scope is omitted.

    ``mol`` is the DECORATED molecule and ``parent_atoms`` the atom indices of the
    parent hydride / parent compound within it; the decorations are then read off
    structurally (:func:`_l4_components`) rather than taken on trust.

    The test is exactly the rule's own two operations, run to exhaustion: place the
    decoration multiset over every position of the parent that bears a hydrogen, in
    every way the positions' hydrogen counts allow, and compare the resulting
    constitutions. One distinct constitution => no isomer can be generated => omit.

    ⚠⚠ **``substitutable_positions`` IS DELIBERATELY NOT REUSED HERE**, and that
    is the single most important line of this function. That helper implements
    ``:3007``'s carve-out (*"Except for hydrogen atoms attached to chalcogen atoms,
    such as in acids, alcohols, and to the carbon atoms of formyl groups"*), which
    is a sentence of **** and has no counterpart in -- this
    rule counts ISOMERS, and says nothing whatever about which hydrogens are
    "substitutable". Routing through it would make the whole polysulfane family
    deny by construction: trisulfane is ``HS-S-SH``, so EVERY hydrogen it has is on
    a chalcogen, ``substitutable_positions`` is **empty**, and
    ``l3_one_kind_of_substitutable_h`` /:func:`l5_uniform_complete` /
    :func:`l6_all_substitutable_h_share_one_locant` therefore all deny -- yet
    ``:39335`` prints ``CH3-S-S-SH methyltrisulfane (PIN)`` under §****
    *"Compounds with three or more contiguous identical chalcogen atoms are treated
    as parent hydrides in substitutive nomenclature"*. Positions here are those
    bearing a hydrogen in the ORDINARY sense, and the reason no isomer exists for
    methyltrisulfane is a hydrogen COUNT, not a carve-out: the middle sulfur bears
    **zero** hydrogens, so S1/S3 are the only placements and they are one orbit.
    (⚠ And the ``:3007`` carve-out must NOT be loosened to make this work: it is
    load-bearing for, where propanedioic acid's two acid O-H must not
    count or ``chloropropanedioic acid`` (``:2951``) loses its licence.)

    Derived against **every** example printed in the rule's own block, positives
    and negatives, which is what fixes the two design points a simpler reading
    misses:

    ========================================== ====================================
    ``:39335`` ``methyltrisulfane`` S2 has no H => S1/S3 only, one orbit
    ``:39339`` ``dimethyltrisulfane`` both placements forced, one result
    ``:39341`` ``methyl(phenyl)triselane`` ★ HETEROGENEOUS: interchange gives
                                                the same molecule
    ``:2979`` ``not 1-bromo-2-methyldisulfane`` heterogeneous disulfane, likewise
    ``:2959`` ``ethylidenehydrazinyl`` ethylidene needs 2 H => only N2 can
                                                host it: ONE placement
    ``:2983`` ``oxoethenyl`` same, oxo needs 2 H
    ``:2975`` ``chloro(silylidene)hydrazine`` 2+1 > N's 2 H, so no co-location;
                                                the two swaps agree
    ``:2995`` ``1-ethylidene-2-propylidene- ★ interchange DOES give an isomer
               disilan-1-yl`` (locants needed) -- a single-orbit test would have
                                                wrongly omitted here
    ``:2999`` ``1-chloro-2-ethylidenedisilane`` ★ the BB's reason is *"moving the
               (locants needed) Cl atom to the other Si atom"*, and
                                                Si has 3 H, so CO-LOCATION on one
                                                position is a legal placement and
                                                must be enumerated
    ``:3003`` ``2-chloroethen-1-yl`` C1 and C2 both have H and differ
    ========================================== ====================================

    So single-orbit equivalence is NOT the test on either flank: it is too weak for
    a heterogeneous multiset (``:2995``) and too narrow to notice co-location
    (``:2999``).

    ⚠ The four spellings of ``:3005`` -- *"As an exception the locant is not omitted
    from propan-2-one, butan-2-one, prop-2-enoic acid and prop-2-ynoic acid although
    unambiguous without a locant"* -- are exceptions to THIS licence (that sentence
    closes 's block). They are NOT enumerated here, deliberately: the set
    is demonstrably OPEN -- ``prop-2-enamide (PIN)`` (``:32724``),
    ``...prop-2-enenitrile (PIN)`` (``:46790``) and ``prop-2-enoyl (preferred
    prefix)`` (``:30570``) are all equally unambiguous without a locant and all
    printed with one. (``prop-2-enal``'s locanted spelling appears at ``:35068``
    only inside a ``[not 2-butylprop-2-enal...]`` clause that rejects the name on a
    different ground -- parent selection -- so it evidences the SPELLING, not a PIN
    endorsement; the three above carry the tag themselves.) A four-row table would
    therefore be wrong on its complement by construction. They stay
    correct because nothing routes an unsaturated chain suffix through this licence;
    a caller that ever does must handle the class, not the four names.

    **Deny-by-default**, per this module's docstring. It declines on all THREE of
    the things ``ARCH-a-licence-can-be-evaluated-on-the-wrong-molecule.md`` requires
    of a licence::func:`locants_are_forced` and
    :func:`scope_has_isotopic_modification` here, and the fragment-boundary
    observation at its call site (kept in ``handlers/_handler_shared.py`` so this
    module stays a pure leaf). It also denies on any defined stereochemistry, since
    relocating a bond cannot be shown to preserve a descriptor.
    """
    # ARCH-a, the two AMBIENT declarations. This licence empties its scope of ALL
    # locants, which is precisely what (``:44180``) forbids while an
    # isotopic modification is being spliced in -- and the isotope path names an
    # isotope-STRIPPED skeleton, so no structural test here can see the label.
    if locants_are_forced():
        return False
    if scope_has_isotopic_modification():
        return False

    if mol is None or parent_atoms is None:
        return False
    try:
        parent_set = frozenset(int(i) for i in parent_atoms)
    except (TypeError, ValueError):
        return False
    n_atoms = mol.GetNumAtoms()
    if not parent_set or any(i < 0 or i >= n_atoms for i in parent_set):
        return False

    if scope_forces_locants(
        prefix_locants=prefix_locants,
        suffix_locants=suffix_locants,
        stereo_text=stereo_text,
        has_indicated_h=has_indicated_h,
        has_isotope=has_isotope,
        is_multiplicative=is_multiplicative,
        is_ring_assembly=is_ring_assembly,
        has_skeletal_replacement=has_skeletal_replacement,
    ):
        return False

    # Relocating a bond cannot be shown to preserve a stereodescriptor, so any
    # defined stereochemistry denies rather than being silently rewired.
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return False
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return False

    comps = _l4_components(mol, parent_set)
    if not comps:
        return False

    # Capacity = the parent HYDRIDE's hydrogen count at each parent atom: what it
    # carries now, plus whatever the decorations sitting on it displaced.
    capacity: Dict[int, int] = {}
    for pos in sorted(parent_set):
        capacity[pos] = mol.GetAtomWithIdx(pos).GetTotalNumHs()
    for (_anchor, old_pos, order) in comps:
        capacity[old_pos] = capacity.get(old_pos, 0) + order

    positions = [p for p in sorted(parent_set) if capacity[p] > 0]
    if not positions:
        return False
    if len(positions) ** len(comps) > _L4_MAX_ASSIGNMENTS:
        return False                           # fail closed rather than run long

    keys = set()
    for assignment in itertools.product(positions, repeat=len(comps)):
        load: Dict[int, int] = {}
        for (_anchor, _old, order), pos in zip(comps, assignment):
            load[pos] = load.get(pos, 0) + order
        if any(load[p] > capacity[p] for p in load):
            continue                           # not a placement at all
        key = _l4_relocated_key(mol, comps, assignment, capacity)
        if key is None:
            return False                       # could not establish -> deny
        keys.add(key)
        if len(keys) > 1:
            return False                       # an isomer exists -> cite
    return len(keys) == 1


def l6_all_substitutable_h_share_one_locant(
    mol, atom_to_locant: Optional[Mapping[int, object]]
) -> bool:
    """```` (``:3031``).

        "All locants are omitted for parent compounds when all substitutable hydrogen
         atoms have the same locant."

    Example ``:3037`` ``difluoroacetic acid (PIN) (not 2,2-difluoroacetic acid)`` --
    acetic acid's only substitutable hydrogens are the three on C-2, because the acid
    OH is a chalcogen H excluded by ``:3007``.

    Fail-closed: a substitutable atom missing from ``atom_to_locant`` denies, because
    its locant cannot be shown to coincide.
    """
    if mol is None or not atom_to_locant:
        return False
    positions = substitutable_positions(mol)
    if not positions:
        return False
    locants = set()
    for idx in positions:
        if idx not in atom_to_locant:
            return False
        loc = atom_to_locant[idx]
        if loc is None:
            return False
        locants.add(loc)
    return len(locants) == 1


def _has_letter_locant(locants: Iterable) -> bool:
    """A non-plain-numeric locant (``N``, ``N1``, ``1'``, ``O``) is always essential.

    ``N,N'-dimethylurea`` and ``N-methylthiourea`` keep their italic-``N`` locants,
    and a primed locant belongs to a ring assembly or multiplicative name, both of
    which always cite. (The MONOsubstituted urea ``methylurea`` omits its locant, but
    by the separate composer-level rule (``:2943``) that never lets a
    letter locant reach this scope, not by this predicate.)
    """
    for loc in locants:
        if isinstance(loc, bool):
            return True
        if isinstance(loc, int):
            continue
        if loc is None:
            return True
        if not str(loc).strip().isdigit():
            return True
    return False


# --------------------------------------------------------------------------------- #
# as an AMBIENT SCOPE #
# --------------------------------------------------------------------------------- #
# Some essential-locant facts are known only OUTSIDE the naming call that has to honour
# them. The isotope path is the measured case: ``rules/isotopes.py`` strips every label,
# names the isotope-FREE skeleton, then splices the descriptor into the finished string.
# The licences run deep inside that skeleton naming and receive a molecule with
# ``GetIsotope == 0`` everywhere, so every ``has_isotope`` argument they compute is
# structurally False. Measured with a trace validated on 2 known positives + 1 negative:
# ``scope_forces_locants`` IS reached for a labelled benzene and receives
# ``has_isotope=False``.
#
# So we shipped ``hexamethyl(13C1)benzene`` and ``(2-13C1)cyclohexanol`` against
# **** (``:44180``): *"In preferred IUPAC names, locants are omitted if no
# locants are necessary in unmodified names. However, if isotopic modification requires a
# locant to specify its position, then all locants must be specified and none are
# omitted."* -- whose own example prints the elided form as the rejected one:
# ``(2-13C)ethan-1-ol [not (2-13C)ethanol]`` (``:44186``).
#
# Patching each guard cannot work: by the time a licence runs, the isotope is gone. The
# fact has to be carried DOWN from where it is known. A ContextVar is the mechanism --
# it is exactly "this whole naming scope contains an essential locant", which is the
# sentence is made of, and it is async/thread-safe so concurrent naming cannot
# leak state between callers.
#
# ⚠ IT IS DELIBERATELY CONDITIONAL, NOT A BLANKET "isotope => cite". fires
# only when the modification *requires a locant to specify its position*. When every
# candidate position is equivalent, no locant is needed and the parent keeps its licensed
# omission -- **** (``:44202``) prints ``(2H6)benzene (PIN)``, and
# ``(13C1)benzenehexol`` is correct today for exactly that reason (all six ring positions
# are one orbit, so there is only one isotopomer). ``rules/isotopes.py`` therefore enters
# this scope only after its own descriptor enumeration has *established* that a locant is
# required -- i.e. only when the locant-free descriptor failed to round-trip.
_FORCED_LOCANT_REASON: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "orthonym_forced_locant_reason", default=None
)


@contextlib.contextmanager
def forced_locant_scope(reason: str):
    """Declare that this naming scope contains an essential locant (````).

    Every licence must consult:func:`locants_are_forced` and decline while
    this is active. ``reason`` is free text for debugging (e.g. ``"isotope"``); it is
    never parsed.
    """
    token = _FORCED_LOCANT_REASON.set(reason)
    try:
        yield
    finally:
        _FORCED_LOCANT_REASON.reset(token)


def locants_are_forced() -> bool:
    """True when an enclosing:func:`forced_locant_scope` is active.

    A licence that does not consult this will silently elide a locant the Blue Book
    requires, and neither (``namer.py`` states verbatim that it *"ignores
    isotopes"*) nor the gold set can see it.
    """
    return _FORCED_LOCANT_REASON.get() is not None


def forced_locant_reason() -> "Optional[str]":
    """The active reason, or None. Diagnostics only."""
    return _FORCED_LOCANT_REASON.get()


# --------------------------------------------------------------------------------- #
# "THIS NAMING SCOPE IS ISOTOPICALLY MODIFIED" -- weaker than forced_locant_scope #
# --------------------------------------------------------------------------------- #
# ⚠ MEASURED 2026-07-29 (Phase C Task 5a). ``forced_locant_scope`` above is NOT
# sufficient for every licence, because ``rules/isotopes.py`` enters it *conditionally*
# -- only once ``_enumerate`` has established that the descriptor needs a locant
# (``loc_rank >= 1``). A validated trace at the two live substituent sites recorded, for
# ``FC(F)(F)[13C](F)(F)C1CCCCC1``:
#
# locants_are_forced == False and every GetIsotope in the scope == 0
#
# i.e. BOTH signals a licence could consult are negative, yet the finished name really
# does carry ``(13C1)``. Wiring on ``locants_are_forced`` alone therefore
# turned ``(1,1,2,2,2-pentafluoro(13C1)ethyl)cyclohexane`` into
# ``(pentafluoro(13C1)ethyl)cyclohexane`` -- stripping the only locants left in a scope
# that carries an isotopic modification, which is what **** (``:44180``)
# forbids: *"if isotopic modification requires a locant to specify its position, then
# all locants must be specified and none are omitted."* The ethyl group's two carbons
# are NOT equivalent, so the position does have to be stated, and the Blue Book's own
# locant-free precedent ``(2H6)benzene (PIN)`` (``:44202``) is licensed precisely
# because all six positions ARE equivalent.
#
# Nothing could catch this: "ignores isotopes" (``namer.py`` verbatim), the
# name round-trips cleanly through OPSIN either way, and gold exposure for the whole
# class is zero.
#
# So this is a SEPARATE, WEAKER declaration: *"an isotopic descriptor will be spliced
# into this scope"*, made unconditionally by the isotope decorator. A licence consults
# it when it is about to empty a scope of ALL its locants. It is deliberately NOT
# folded into ``forced_locant_scope``, because that would also gag the licences whose
# locant-free form is CORRECT under -- ``(13C1)benzenehexol`` and
# ``(2H6)benzene`` -- which is why ``rules/benzene.py`` still consults only
# ``locants_are_forced``.
_ISOTOPIC_NAMING_SCOPE: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "orthonym_isotopic_naming_scope", default=None
)


@contextlib.contextmanager
def isotopic_naming_scope(reason: str = "isotope"):
    """Declare that an isotopic descriptor will be spliced into this naming scope.

    Entered UNCONDITIONALLY by ``rules/isotopes.py`` around its skeleton naming,
    because the labels are stripped before naming and are therefore invisible to
    every structural test further down.
    """
    token = _ISOTOPIC_NAMING_SCOPE.set(reason)
    try:
        yield
    finally:
        _ISOTOPIC_NAMING_SCOPE.reset(token)


def scope_has_isotopic_modification() -> bool:
    """True when an enclosing:func:`isotopic_naming_scope` is active.

    A licence that would leave a scope with ZERO locants must decline on this
    , ``:44180``). A licence whose locant-free form stays correct under
     (all candidate positions in one orbit) must NOT -- see the comment
    above for why this is separate from:func:`locants_are_forced`.
    """
    return _ISOTOPIC_NAMING_SCOPE.get() is not None


# --------------------------------------------------------------------------------- #
# "THE PARENT HYDRIDE ITSELF CARRIES A POSITIONAL ISOTOPE" -- stronger than #
# forced_locant_scope, for the licences that restore a PARENT's OWN omitted locant #
# --------------------------------------------------------------------------------- #
# ⚠ MEASURED 2026-09-22 (v52 a phase Task 4).:func:`locants_are_forced` above is
# TRUE whenever ANY locant in the whole naming is forced -- including a locant that a
# label buried in a SUBSTITUENT needs (``(1,1,2,2,2-pentafluoro(1-13C)ethyl)benzene``:
# the ¹³C sits on the ethyl's C1, so ``forced_locant_scope`` is entered, yet the
# benzene ring is symmetric and its monosubstituted-prefix locant must stay omitted).
# The licences that only ever restore a SUFFIX/ambient locant already in the parent's
# own scope (L3, L4, ``handlers/_handler_shared.py:440``, ``should_omit_locant_one``
# Rule 3b) may consult the weaker:func:`locants_are_forced`. But a licence that
# restores the PARENT's OWN substituent-position locant -- ``should_omit_locant_one``
# Rule 4 (monosubstituted carbocyclic ring) and the ylidene-hydrazine handler -- must
# fire ONLY when the isomerism is real, i.e. when the PARENT SKELETON ITSELF carries
# the positional label, ``:44202`` "Locants are not omitted when there is a
# possibility of isomers"): ``1-(79Br)bromo(2-13C)benzene`` (ring carbon labelled) but
# NOT ``(pentafluoro(1-13C)ethyl)benzene`` (substituent carbon labelled). This flag is
# entered by ``rules/isotopes.py`` alongside ``forced_locant_scope``, but ONLY when a
# labelled atom sits on the parent skeleton (see ``_parent_carries_positional_isotope``).
_ISOTOPE_PARENT_POSITIONAL: "contextvars.ContextVar[Optional[str]]" = (
    contextvars.ContextVar("orthonym_isotope_parent_positional", default=None))


@contextlib.contextmanager
def isotope_parent_positional_scope(reason: str = "isotope"):
    """Declare that the PARENT skeleton being named carries a positional isotope
    label. Entered by ``rules/isotopes.py`` alongside
    :func:`forced_locant_scope`, and only when a labelled atom sits on the parent
    hydride itself, so a licence that restores the parent's OWN
    substituent-position locant can decline without over-citing the cases where the
    label sits in a substituent (see the module comment above)."""
    token = _ISOTOPE_PARENT_POSITIONAL.set(reason)
    try:
        yield
    finally:
        _ISOTOPE_PARENT_POSITIONAL.reset(token)


def parent_scope_has_positional_isotope() -> bool:
    """True when an enclosing:func:`isotope_parent_positional_scope` is active."""
    return _ISOTOPE_PARENT_POSITIONAL.get() is not None


def scope_forces_locants(
    *,
    prefix_locants,
    suffix_locants,
    stereo_text,
    has_indicated_h,
    has_isotope,
    is_multiplicative,
    is_ring_assembly,
    has_skeletal_replacement,
) -> bool:
    """```` (``:2869``) -- the deny-default itself.

    True => cite EVERY locant in this scope, licence or not.

    Plain numeric locants alone never force: those are exactly what a ````
    licence is permitted to omit. What forces is anything *essential* in the same
    scope -- a letter or primed locant, a stereodescriptor that needs a locant, or one
    of the hard overrides of derivation Part C (indicated/added hydrogen; isotopic
    labels -- ``:44180`` §"" *"if isotopic modification requires a locant to
    specify its position"*; multiplicative names; ring assemblies; skeletal
    replacement -- ``:6446`` §"", whose sentence *"Once a structure modified
    by skeletal replacement ('a') prefixes has been named and numbered, it is
    considered to be a new parent hydride. As locants assigned to heteroatoms are
    essential, all locants must be cited as defined in "* is the override;
    the same line *opens* with the unrelated element-seniority order, ``F > Cl > Br >
    ...``, but that is not the clause being cited here).

    **Fail-closed:** any argument that is ``None`` -- i.e. the caller could not
    establish it -- returns True.
    """
    args = (prefix_locants, suffix_locants, stereo_text, has_indicated_h,
            has_isotope, is_multiplicative, is_ring_assembly,
            has_skeletal_replacement)
    if any(a is None for a in args):
        return True

    if (has_indicated_h or has_isotope or is_multiplicative
            or is_ring_assembly or has_skeletal_replacement):
        return True
    if str(stereo_text).strip():
        return True
    if _has_letter_locant(prefix_locants) or _has_letter_locant(suffix_locants):
        return True
    return False


def l3_monosubstituted_locant_omitted(
    parent,
    *,
    n_substitutions,
    prefix_locants,
    suffix_locants,
    parent_cites_locants,
    stereo_text,
    has_indicated_h,
    has_isotope,
    is_multiplicative: bool = False,
    is_ring_assembly: bool = False,
    has_skeletal_replacement: bool = False,
) -> bool:
    """§**** (``:2939``) -- the WHOLE licence, for ONE substitution.

        "The locant is omitted in monosubstituted symmetrical parent hydrides or
         parent compounds where there is only one kind of substitutable hydrogen."

    True => that single locant is omitted. The orbit test itself is NOT re-derived
    here: it is:func:`l3_one_kind_of_substitutable_h`, the one place the licence
    lives. This function adds the three things the orbit test deliberately leaves to
    its caller (its own docstring says so): that the substitution really is *mono*,
    that nothing else in the scope forces locants, and the two ambient
    scopes.

    ``parent`` is the **parent hydride or parent compound** -- the molecule with the
    ONE substitution REMOVED -- because that is what the rule's own examples measure:

    ========================== ================================== =============
    printed PIN ``parent`` orbit test
    ========================== ================================== =============
    ``pyrazinecarboxylic acid`` pyrazine (parent **hydride**) 4 CH, 1 orbit
    ``chloropropanedioic acid`` propanedioic acid (parent **cpd**) C2 only
    ``chlorobutanedioic acid`` butanedioic acid C2/C3, 1 orbit
    ``methylurea`` urea 2 NH2, 1 orbit
    ========================== ================================== =============

    ★ **The boundary the whole task turns on**, and it falls out of ``:3007`` with no
    special case: propanedioic acid's two acid O-H are on a chalcogen and are NOT
    substitutable, leaving C2 as the only kind, so the licence fires. Propane**diamide**
    has C2 *and* two amide N-H -- neither a chalcogen H nor a formyl H, so they count --
    giving two kinds, so it is denied and ``2-methylpropanediamide`` (``:2887``) keeps
    its locant. ``:2889`` ``N1,N3-dimethylpropanediamide (PIN)`` proves independently
    that an amide N-H is substitutable. ⚠ Do NOT "fix" the chalcogen exclusion to make
    trisulfane work: it is load-bearing HERE, and ``methyltrisulfane`` is licensed by a
    different sub-rule, unimplemented) -- see
    `internal notes`.

    ⚠ **This licence is orthogonal to (c)** (``_ring_suffix_locant_is_trivial``),
    which is restricted to saturated all-carbon monocycles and therefore cannot reach a
    heteroarene. L3 is what licenses ``pyrazinecarboxylic acid`` while
    ``piperidine-1-carbonitrile`` (``:34730``) correctly keeps its locant -- piperidine's
    N-H, C2/C6, C3/C5 and C4 are FOUR orbits. Neither rule may be widened into the
    other; the orbit predicate is the only thing separating those two rings.

    Args:
        parent: the parent hydride / parent compound, substitution removed.
        n_substitutions: how many decorations sit on ``parent``. The rule says
            *"monosubstituted"*, so anything but exactly 1 denies. Passed
            separately from the locant lists because a decoration whose locant was
            already elided upstream contributes no locant, and counting locants
            alone would read a disubstituted parent as mono.
        prefix_locants: locants of substituent prefixes in this scope.
        suffix_locants: locants of suffixes in this scope.
        parent_cites_locants: True when the parent name itself already cites a
            locant -- a heteroatom locant set (``1,4-dioxane``), an added/indicated
            hydrogen, or an unsaturation locant. ```` (``:2869``) then
            restores every locant in the scope, so the licence declines. Every one of
            the Blue Book's five printed L3 positives has a locant-free parent name
            (pyrazine, urea, disiloxane, coronene, propanedioic acid), so this is the
            deny-by-default side of a boundary the source does not print, and it is
            recorded as such rather than as a verified rule.
    """
    # (``:2869``) as an AMBIENT scope -- ``rules/isotopes.py`` names an
    # isotope-STRIPPED skeleton, so every structural isotope test further down reads
    # 0 even for a labelled input and cannot be the guard.
    if locants_are_forced():
        return False
    # ⚠ AND the weaker declaration. Measured 2026-07-29 (Task 5a): the isotope
    # decorator enters ``forced_locant_scope`` only CONDITIONALLY, so
    # ``locants_are_forced`` alone is False for a labelled molecule whose finished
    # name still carries ``(13C1)``. This licence empties its scope of ALL locants,
    # which is exactly what **** (``:44180``) forbids when an isotopic
    # modification needs a locant, so it must decline on the weaker flag too.
    if scope_has_isotopic_modification():
        return False

    if parent is None:
        return False
    if isinstance(n_substitutions, bool) or not isinstance(n_substitutions, int):
        return False
    if n_substitutions != 1:
        return False

    prefix_locants = list(prefix_locants or [])
    suffix_locants = list(suffix_locants or [])
    # Exactly ONE cited locant in the scope -- the one this licence would omit.
    # Emptying a scope that cites two locants is not what:2939 licenses.
    if len(prefix_locants) + len(suffix_locants) != 1:
        return False

    if parent_cites_locants:
        return False

    if scope_forces_locants(
        prefix_locants=prefix_locants,
        suffix_locants=suffix_locants,
        stereo_text=stereo_text,
        has_indicated_h=has_indicated_h,
        has_isotope=has_isotope,
        is_multiplicative=is_multiplicative,
        is_ring_assembly=is_ring_assembly,
        has_skeletal_replacement=has_skeletal_replacement,
    ):
        return False

    return l3_one_kind_of_substitutable_h(parent)


def _one_substituent_removed(mol, parent_atoms: FrozenSet[int]) -> bool:
    """Structural proof of *"monosubstituted"* for ````.

    True iff the atoms OUTSIDE ``parent_atoms`` form exactly **one** connected
    component joined to the parent by exactly **one** bond -- i.e. the parent bears
    exactly one substituent group.

    This is what makes ``n_substitutions`` a measurement rather than a caller's
    belief. It also rejects the two shapes that would otherwise read as "one":
    several separate decorations (``3,3,3-trifluoro...`` removes THREE fluorines =
    three components) and a group bonded twice (a fused/bridging ring, two
    attachment bonds), neither of which ``:2939`` licenses.
    """
    if mol is None:
        return False
    n = mol.GetNumAtoms()
    outside = [i for i in range(n) if i not in parent_atoms]
    if not outside:
        return False                      # nothing substituted -> no locant to omit
    outside_set = set(outside)

    # Exactly one attachment bond between the parent and everything outside it.
    attachments = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in parent_atoms) != (b in parent_atoms):
            attachments += 1
            if attachments > 1:
                return False
    if attachments != 1:
        return False

    #...and the outside atoms are ONE connected component.
    seen = {outside[0]}
    stack = [outside[0]]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in outside_set and j not in seen:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(outside_set)


def l3_locant_omitted_for_parent_atoms(
    mol,
    parent_atoms,
    *,
    prefix_locants,
    suffix_locants,
    parent_cites_locants,
    stereo_text,
    is_multiplicative: bool = False,
    is_ring_assembly: bool = False,
    has_skeletal_replacement: bool = False,
) -> bool:
    """§**** (``:2939``) where the parent is given as an ATOM SET.

    The one entry point both live call sites use. It performs the parent surgery --
    delete every atom outside ``parent_atoms``, letting RDKit restore the implicit
    hydrogens that the substituent had displaced -- and derives the two arguments a
    caller must not be trusted with:

    * ``n_substitutions``, proven by:func:`_one_substituent_removed`;
    * ``has_isotope``, read off the real (undeleted) molecule, because the parent
      copy may not carry the label.

    Returns False -- retain the locant -- on any failure to establish the structure,
    including a sanitization failure on the reconstructed parent.

    ⚠ ``parent_atoms`` is the parent **hydride or compound** in the rule's sense: for
    ``pyrazinecarboxylic acid`` it is the six RING atoms only (the ``-carboxylic acid``
    carbon is the substitution), while for ``chloropropanedioic acid`` it is the chain
    **plus** both ``-COOH`` groups (the chloro is the substitution). Getting that
    boundary wrong silently changes which molecule the orbit test measures.
    """
    if mol is None or parent_atoms is None:
        return False
    try:
        parent_set = frozenset(int(i) for i in parent_atoms)
    except (TypeError, ValueError):
        return False
    n_atoms = mol.GetNumAtoms()
    if not parent_set or any(i < 0 or i >= n_atoms for i in parent_set):
        return False

    if not _one_substituent_removed(mol, parent_set):
        return False

    rw = Chem.RWMol(mol)
    for idx in sorted(range(n_atoms), reverse=True):
        if idx not in parent_set:
            rw.RemoveAtom(idx)
    parent = rw.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return False
    if parent.GetNumAtoms() != len(parent_set):
        return False

    has_isotope = any(a.GetIsotope() for a in mol.GetAtoms())

    return l3_monosubstituted_locant_omitted(
        parent,
        n_substitutions=1,
        prefix_locants=prefix_locants,
        suffix_locants=suffix_locants,
        parent_cites_locants=parent_cites_locants,
        stereo_text=stereo_text,
        has_indicated_h=False,
        has_isotope=has_isotope,
        is_multiplicative=is_multiplicative,
        is_ring_assembly=is_ring_assembly,
        has_skeletal_replacement=has_skeletal_replacement,
    )
