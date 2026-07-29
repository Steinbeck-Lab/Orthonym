"""P-14.3.4 -- omission of locants. The ONE place the licences are decided.

v29 Phase C tranche B. Pure module: RDKit mols in, booleans out. No I/O, no naming.

⚠ **DENY BY DEFAULT.** ``P-14.3.3`` "Citation of locants"
(``BlueBookV2/BlueBookV2.md:2869``) is the rule:

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure as defined by its appropriate
     enclosing marks, then all locants must be cited for the parent structure or that
     structural unit."

``P-14.3.4`` (``:2871``) then grants narrow *licences*, and its own preamble says
*"for absolute clarity in preferred IUPAC names it is necessary to be prescriptive
about when omission of locants is permissible."* So:

* **never** write a locant-stripping pass -- every omission is a positively-licensed
  structural predicate;
* **fail toward retaining the locant** -- anything this module cannot positively
  establish returns ``False`` (or ``True`` from :func:`scope_forces_locants`, which is
  the same direction);
* a licence is evaluated **per enclosing-mark scope**. One essential locant anywhere in
  a scope restores *every* locant in that scope -- which is why
  ``1-chloro-2-(pentafluoroethyl)benzene`` elides inside the parentheses and keeps
  ``1,2`` outside.

★ **THE SELF-VALIDATING BOUNDARY PAIR** -- the reason these predicates count
HYDROGENS, never positions::

    :7625    benzenehexol (PIN, P-63.1.2) (not benzenehexaol)            -> OMITS
    :54823   "Inositols, cyclohexane-1,2,3,4,5,6-hexols, are a specific
              group of cyclitols."                                       -> RETAINS

Same six OH, same ring size. A benzene ring carbon has ONE substitutable H, so six OH
*completely* substitutes the ring and ``P-14.3.4.5`` fires. A cyclohexane ring carbon
has TWO, so six OH is *partial* substitution and the ``:3009`` counter-clause restores
every locant. A predicate that counts positions gets this pair wrong.

The class is **OPEN** (ring size, ring element and substituent identity are all
unbounded), so a finite table of names or spellings would be wrong on its complement by
construction. Structural predicates only.
"""
from __future__ import annotations

import contextlib
import contextvars

from typing import Dict, FrozenSet, Iterable, Mapping, Optional

from rdkit import Chem

__all__ = [
    "CHALCOGENS",
    "substitutable_h_count",
    "substitutable_positions",
    "l5_uniform_complete",
    "l3_one_kind_of_substitutable_h",
    "l3_monosubstituted_locant_omitted",
    "l3_locant_omitted_for_parent_atoms",
    "l6_all_substitutable_h_share_one_locant",
    "scope_forces_locants",
    "forced_locant_scope",
    "locants_are_forced",
    "forced_locant_reason",
    "isotopic_naming_scope",
    "scope_has_isotopic_modification",
]

#: ``:3007`` -- "Except for hydrogen atoms attached to chalcogen atoms, such as in
#: acids, alcohols, ...". Group 16: O, S, Se, Te (Po is not a nomenclature case).
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
    """``P-14.3.4.5`` (``:3007``) plus its counter-clause (``:3009``).

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
            return False                       # partial -> :3009
        n_dec = 1 if counts is None else counts.get(idx, 1)
        if not isinstance(n_dec, int) or n_dec != substitutable_h_count(mol, idx):
            return False                       # partial AT this position -> :3009
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
    """``P-14.3.4.3`` (``:2939``).

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


def l6_all_substitutable_h_share_one_locant(
    mol, atom_to_locant: Optional[Mapping[int, object]]
) -> bool:
    """``P-14.3.4.6`` (``:3031``).

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

    ``N-methylurea`` keeps its ``N`` even though ``methylurea`` drops the numeral
    (``:2943``), and a primed locant belongs to a ring assembly or multiplicative name,
    both of which always cite.
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
# P-14.3.3 as an AMBIENT SCOPE                                                        #
# --------------------------------------------------------------------------------- #
# Some essential-locant facts are known only OUTSIDE the naming call that has to honour
# them. The isotope path is the measured case: ``rules/isotopes.py`` strips every label,
# names the isotope-FREE skeleton, then splices the descriptor into the finished string.
# The licences run deep inside that skeleton naming and receive a molecule with
# ``GetIsotope() == 0`` everywhere, so every ``has_isotope`` argument they compute is
# structurally False. Measured with a spy validated on 2 known positives + 1 negative:
# ``scope_forces_locants`` IS reached for a labelled benzene and receives
# ``has_isotope=False``.
#
# So we shipped ``hexamethyl(13C1)benzene`` and ``(2-13C1)cyclohexanol`` against
# **P-82.6.1.1** (``:44180``): *"In preferred IUPAC names, locants are omitted if no
# locants are necessary in unmodified names. However, if isotopic modification requires a
# locant to specify its position, then all locants must be specified and none are
# omitted."* -- whose own example prints the elided form as the rejected one:
# ``(2-13C)ethan-1-ol [not (2-13C)ethanol]`` (``:44186``).
#
# Patching each guard cannot work: by the time a licence runs, the isotope is gone. The
# fact has to be carried DOWN from where it is known. A ContextVar is the mechanism --
# it is exactly "this whole naming scope contains an essential locant", which is the
# sentence P-14.3.3 is made of, and it is async/thread-safe so concurrent naming cannot
# leak state between callers.
#
# ⚠ IT IS DELIBERATELY CONDITIONAL, NOT A BLANKET "isotope => cite". P-82.6.1.1 fires
# only when the modification *requires a locant to specify its position*. When every
# candidate position is equivalent, no locant is needed and the parent keeps its licensed
# omission -- **P-82.6.1.3** (``:44202``) prints ``(2H6)benzene (PIN)``, and
# ``(13C1)benzenehexol`` is correct today for exactly that reason (all six ring positions
# are one orbit, so there is only one isotopomer). ``rules/isotopes.py`` therefore enters
# this scope only after its own descriptor enumeration has *established* that a locant is
# required -- i.e. only when the locant-free descriptor failed to round-trip.
_FORCED_LOCANT_REASON: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "orthonym_forced_locant_reason", default=None
)


@contextlib.contextmanager
def forced_locant_scope(reason: str):
    """Declare that this naming scope contains an essential locant (``P-14.3.3``).

    Every P-14.3.4 licence must consult :func:`locants_are_forced` and decline while
    this is active. ``reason`` is free text for debugging (e.g. ``"isotope"``); it is
    never parsed.
    """
    token = _FORCED_LOCANT_REASON.set(reason)
    try:
        yield
    finally:
        _FORCED_LOCANT_REASON.reset(token)


def locants_are_forced() -> bool:
    """True when an enclosing :func:`forced_locant_scope` is active.

    A licence that does not consult this will silently elide a locant the Blue Book
    requires, and neither SELF-01 (``namer.py`` states verbatim that it *"ignores
    isotopes"*) nor the gold set can see it.
    """
    return _FORCED_LOCANT_REASON.get() is not None


def forced_locant_reason() -> "Optional[str]":
    """The active reason, or None. Diagnostics only."""
    return _FORCED_LOCANT_REASON.get()


# --------------------------------------------------------------------------------- #
# "THIS NAMING SCOPE IS ISOTOPICALLY MODIFIED" -- weaker than forced_locant_scope     #
# --------------------------------------------------------------------------------- #
# ⚠ MEASURED 2026-07-29 (v29 Phase C Task 5a). ``forced_locant_scope`` above is NOT
# sufficient for every licence, because ``rules/isotopes.py`` enters it *conditionally*
# -- only once ``_enumerate`` has established that the descriptor needs a locant
# (``loc_rank >= 1``). A validated spy at the two live substituent sites recorded, for
# ``FC(F)(F)[13C](F)(F)C1CCCCC1``:
#
#     locants_are_forced() == False   and   every GetIsotope() in the scope == 0
#
# i.e. BOTH signals a licence could consult are negative, yet the finished name really
# does carry ``(13C1)``. Wiring P-14.3.4.5 on ``locants_are_forced()`` alone therefore
# turned ``(1,1,2,2,2-pentafluoro(13C1)ethyl)cyclohexane`` into
# ``(pentafluoro(13C1)ethyl)cyclohexane`` -- stripping the only locants left in a scope
# that carries an isotopic modification, which is what **P-82.6.1.1** (``:44180``)
# forbids: *"if isotopic modification requires a locant to specify its position, then
# all locants must be specified and none are omitted."* The ethyl group's two carbons
# are NOT equivalent, so the position does have to be stated, and the Blue Book's own
# locant-free precedent ``(2H6)benzene (PIN)`` (``:44202``) is licensed precisely
# because all six positions ARE equivalent.
#
# Nothing could catch this: SELF-01 "ignores isotopes" (``namer.py`` verbatim), the
# name round-trips cleanly through OPSIN either way, and gold exposure for the whole
# class is zero.
#
# So this is a SEPARATE, WEAKER declaration: *"an isotopic descriptor will be spliced
# into this scope"*, made unconditionally by the isotope decorator. A licence consults
# it when it is about to empty a scope of ALL its locants. It is deliberately NOT
# folded into ``forced_locant_scope``, because that would also gag the licences whose
# locant-free form is CORRECT under P-82.6.1.3 -- ``(13C1)benzenehexol`` and
# ``(2H6)benzene`` -- which is why ``rules/benzene.py`` still consults only
# ``locants_are_forced()``.
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
    """True when an enclosing :func:`isotopic_naming_scope` is active.

    A licence that would leave a scope with ZERO locants must decline on this
    (P-82.6.1.1, ``:44180``). A licence whose locant-free form stays correct under
    P-82.6.1.3 (all candidate positions in one orbit) must NOT -- see the comment
    above for why this is separate from :func:`locants_are_forced`.
    """
    return _ISOTOPIC_NAMING_SCOPE.get() is not None


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
    """``P-14.3.3`` (``:2869``) -- the deny-default itself.

    True => cite EVERY locant in this scope, licence or not.

    Plain numeric locants alone never force: those are exactly what a ``P-14.3.4``
    licence is permitted to omit. What forces is anything *essential* in the same
    scope -- a letter or primed locant, a stereodescriptor that needs a locant, or one
    of the hard overrides of derivation Part C (indicated/added hydrogen; isotopic
    labels -- ``:44180`` §"P-82.6.1.1" *"if isotopic modification requires a locant to
    specify its position"*; multiplicative names; ring assemblies; skeletal
    replacement -- ``:6446`` §"P-15.4.1.2", whose sentence *"Once a structure modified
    by skeletal replacement ('a') prefixes has been named and numbered, it is
    considered to be a new parent hydride. As locants assigned to heteroatoms are
    essential, all locants must be cited as defined in P-14.3.3"* is the override;
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
    """§**P-14.3.4.3** (``:2939``) -- the WHOLE licence, for ONE substitution.

        "The locant is omitted in monosubstituted symmetrical parent hydrides or
         parent compounds where there is only one kind of substitutable hydrogen."

    True => that single locant is omitted. The orbit test itself is NOT re-derived
    here: it is :func:`l3_one_kind_of_substitutable_h`, the one place the licence
    lives. This function adds the three things the orbit test deliberately leaves to
    its caller (its own docstring says so): that the substitution really is *mono*,
    that nothing else in the scope forces locants, and the two ambient P-14.3.3
    scopes.

    ``parent`` is the **parent hydride or parent compound** -- the molecule with the
    ONE substitution REMOVED -- because that is what the rule's own examples measure:

    ==========================  ==================================  =============
    printed PIN                 ``parent``                          orbit test
    ==========================  ==================================  =============
    ``pyrazinecarboxylic acid`` pyrazine (parent **hydride**)       4 CH, 1 orbit
    ``chloropropanedioic acid`` propanedioic acid (parent **cpd**)  C2 only
    ``chlorobutanedioic acid``  butanedioic acid                    C2/C3, 1 orbit
    ``methylurea``              urea                                2 NH2, 1 orbit
    ==========================  ==================================  =============

    ★ **The boundary the whole task turns on**, and it falls out of ``:3007`` with no
    special case: propanedioic acid's two acid O-H are on a chalcogen and are NOT
    substitutable, leaving C2 as the only kind, so the licence fires. Propane**diamide**
    has C2 *and* two amide N-H -- neither a chalcogen H nor a formyl H, so they count --
    giving two kinds, so it is denied and ``2-methylpropanediamide`` (``:2887``) keeps
    its locant. ``:2889`` ``N1,N3-dimethylpropanediamide (PIN)`` proves independently
    that an amide N-H is substitutable. ⚠ Do NOT "fix" the chalcogen exclusion to make
    trisulfane work: it is load-bearing HERE, and ``methyltrisulfane`` is licensed by a
    different sub-rule (P-14.3.4.4, unimplemented) -- see
    ``.

    ⚠ **This licence is orthogonal to P-14.3.4.2(c)** (``_ring_suffix_locant_is_trivial``),
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
            hydrogen, or an unsaturation locant. ``P-14.3.3`` (``:2869``) then
            restores every locant in the scope, so the licence declines. Every one of
            the Blue Book's five printed L3 positives has a locant-free parent name
            (pyrazine, urea, disiloxane, coronene, propanedioic acid), so this is the
            deny-by-default side of a boundary the source does not print, and it is
            recorded as such rather than as a verified rule.
    """
    # P-14.3.3 (``:2869``) as an AMBIENT scope -- ``rules/isotopes.py`` names an
    # isotope-STRIPPED skeleton, so every structural isotope test further down reads
    # 0 even for a labelled input and cannot be the guard.
    if locants_are_forced():
        return False
    # ⚠ AND the weaker declaration. Measured 2026-07-29 (Task 5a): the isotope
    # decorator enters ``forced_locant_scope`` only CONDITIONALLY, so
    # ``locants_are_forced()`` alone is False for a labelled molecule whose finished
    # name still carries ``(13C1)``. This licence empties its scope of ALL locants,
    # which is exactly what **P-82.6.1.1** (``:44180``) forbids when an isotopic
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
    # Emptying a scope that cites two locants is not what :2939 licenses.
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
    """Structural proof of *"monosubstituted"* for ``P-14.3.4.3``.

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

    # ...and the outside atoms are ONE connected component.
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
    """§**P-14.3.4.3** (``:2939``) where the parent is given as an ATOM SET.

    The one entry point both live call sites use. It performs the parent surgery --
    delete every atom outside ``parent_atoms``, letting RDKit restore the implicit
    hydrogens that the substituent had displaced -- and derives the two arguments a
    caller must not be trusted with:

    * ``n_substitutions``, proven by :func:`_one_substituent_removed`;
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
