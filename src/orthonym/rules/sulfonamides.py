"""N-substituted sulfonamides — P-66.1.1.3.1.1 (chalcogen-acid amides).

THE DEFECT THIS CLOSES
----------------------
``secondary_sulfonamide`` / ``tertiary_sulfonamide`` were perceived
(``perception/functional_groups.py:236-238``) and carried a SUFFIX_FORMS row
(``rules/seniority.py:562-564``), but no producer ever rendered the
N-substituent. Every path that reached one emitted a name for a DIFFERENT
molecule and was killed downstream by the OPSIN self-consistency gate:

* ``CS(=O)(=O)NC``        -> ``methanesulfonamide``   (N-methyl carbon DROPPED)
* ``c1ccccc1S(=O)(=O)NC`` -> ``sulfanylbenzene``      (both =O, N and C dropped)
* ``C1CCCCC1S(=O)(=O)NC`` -> ``carbamoylcyclohexane-1-sulfonamide`` (fabricated)

So the class did not "fail closed by design" — it fail-closed by ACCIDENT, one
gate downstream of a wrong-constitution producer.

BLUE BOOK AUTHORITY
-------------------
**P-66.1.1.2** "Sulfonamides, sulfinamides, and related selenium and tellurium
amides" (``BlueBookV2/BlueBookV2.md:32746``):

    "Sulfonamides, sulfinamides, and the analogous selenium and tellurium
    amides are named substitutively using the following suffixes: -SO2-NH2
    sulfonamide (preselected suffix) ... These suffixes may be assigned to any
    position of a parent hydride."

with ``CH3-SO2-NH2 methanesulfonamide (PIN)`` (``:32752``).

**P-66.1.1.3.1** "*N*-Substitution" / **P-66.1.1.3.1.1** (``:32774``) — the
sentence that governs the italic locant, quoted with its operative clause:

    "Substituted primary amides, with general structures such as R-CO-NHR' and
    R-CO-NR'R'', AND THE CORRESPONDING AMIDES DERIVED FROM CHALCOGEN ACIDS are
    named by citing the substituents R' and R'' as prefixes preceded by the
    locant *N* when one amide group is present."

A sulfonamide is an amide of a sulfur (chalcogen) acid, so the ``N-`` locant is
REQUIRED — this is derived from the rule, not inferred from the carbamate
pattern. The Blue Book's own sulfonamide ``(PIN)`` rows confirm the spelling:
``N^3-ethyl-N^1-methylnaphthalene-1,3-disulfonamide (PIN)`` (``:32805``),
``3-chloro-N-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)`` (``:32881``),
``N-carbamoylbenzenesulfonamide (PIN)`` (``:33362``),
``N-hydroxymethanesulfonamide (PIN)`` (``:38338``).

``:32881`` also fixes the negative boundary: it cites
``2',3-dichloronaphthalene-2-sulfonanilide`` as the NON-preferred alternative,
so the anilide contraction must never be emitted for an N-phenyl sulfonamide.

DESIGN — no count stands in for a structure proof
-------------------------------------------------
Per ``.planning/audit-v29/FINDING-count-based-naming-sites.md`` the parent is
never derived from an atom count. Instead the N-substituent branches are
EXCISED from the real molecule and the residual R-SO2-NH2 is named by
re-entering the naming pipeline, exactly as ``handlers/hydroximic_acid.py:92``
does. That has three consequences worth stating:

* the parent spelling is whatever the already-verified PRIMARY sulfonamide path
  produces — ``methanesulfonamide``, ``benzenesulfonamide``,
  ``cyclohexane-1-sulfonamide`` — so the parent name is never spelled twice
  here (the second documented anti-pattern);
* any parent the pipeline cannot name makes the whole name fail closed;
* recursion terminates: the excised parent is a strictly smaller molecule whose
  principal group is ``primary_sulfonamide``, a different class from the one
  this module owns.

MEASURED CLASS BOUNDARY (fails closed outside it)
-------------------------------------------------
* acyclic amide N only. A RING nitrogen (``CS(=O)(=O)N1CCCCC1``) is refused —
  its PIN is ``1-(methanesulfonyl)piperidine``, a sulfonyl PREFIX on a ring
  parent, not an N-substituted sulfonamide.
* exactly one sulfonamide unit. Di- and polysulfonamides need the superscripted
  ``N^1``/``N^3`` locants of P-66.1.1.3.1.1, which are not built here.
* sulfur only. Sulfinamides (-SO-NH2) and the Se/Te analogues have NO functional
  group perception in this tree at all, so even the UNSUBSTITUTED
  ``methanesulfinamide`` is unnameable; that is a separate unbuilt class.
* carbon-attached N-substituents only.

MEASURED GUARD REDUNDANCY (do not "simplify" this away without re-measuring)
---------------------------------------------------------------------------
12 mutations were applied to this module. Only THREE are killable at all --
disabling ``_parent_hydride_is_unsubstituted``, dropping the ``N-`` prefix, and
allowing a heteroatom N-substituent. The other nine (ring N, >1 sulfonamide
unit, the parent-suffix check, the empty-branch check, the sultam loop-back, >2
branches, branch disjointness, the two-``=O`` count, branch ORDER) are
EQUIVALENT MUTANTS: over a 33-molecule probe set covering every one of their
target shapes, no input distinguishes mutant from original, because each is
masked by an earlier guard -- e.g. a ring nitrogen is rejected by
``IsInRing`` before the sultam loop-back can see it, and branch order is
irrelevant because ``format_n_substitution`` sorts internally.

They are kept deliberately: each names a distinct refusal reason at the point a
reader would look for it. But NONE of them is individually load-bearing today,
so a test asserting one of them in isolation cannot fail, and a future edit that
removes an EARLIER guard silently promotes a later one to load-bearing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

# The sulfonamide S-N unit, proved structurally: a four-connected sulfur
# carrying two TERMINAL doubly-bonded oxygens and a trivalent nitrogen.
_SULFONAMIDE_SN = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")

_PARENT_SUFFIX = "sulfonamide"


def _sulfonyl_nitrogen(mol: Any) -> Optional[Tuple[int, int]]:
    """Return ``(s_idx, n_idx)`` for the single sulfonamide unit, else None.

    Every condition is a structural assertion; nothing is inferred from a count
    of carbons. Refuses (returns None) whenever the shape is not exactly the one
    P-66.1.1.3.1.1 governs.
    """
    matches = mol.GetSubstructMatches(_SULFONAMIDE_SN)
    if len(matches) != 1:
        # 0 -> not a sulfonamide; >1 -> a di/polysulfonamide needing the
        # superscripted N-locants of P-66.1.1.3.1.1, which are not built here.
        return None
    s_idx, _o1, _o2, n_idx = matches[0]

    s_atom = mol.GetAtomWithIdx(s_idx)
    n_atom = mol.GetAtomWithIdx(n_idx)

    # -- sulfur must be a clean sulfonyl: exactly two terminal =O, exactly one
    #    amide N, and exactly one other (the R of R-SO2-).
    dbl_o = [
        nb.GetIdx() for nb in s_atom.GetNeighbors()
        if nb.GetAtomicNum() == 8
        and nb.GetDegree() == 1
        and mol.GetBondBetweenAtoms(s_idx, nb.GetIdx()).GetBondType()
        == Chem.BondType.DOUBLE
    ]
    if len(dbl_o) != 2:
        return None
    if s_atom.GetDegree() != 4 or s_atom.GetFormalCharge() != 0:
        return None
    n_neighbours = [nb.GetIdx() for nb in s_atom.GetNeighbors()
                    if nb.GetAtomicNum() == 7]
    if n_neighbours != [n_idx]:
        return None

    # -- nitrogen must be a neutral, acyclic amide N bonded to exactly one S.
    if n_atom.GetFormalCharge() != 0 or n_atom.GetIsotope():
        return None
    if n_atom.IsInRing():
        # PIN is a sulfonyl PREFIX on the ring parent
        # (1-(methanesulfonyl)piperidine), a different construction.
        return None
    if len([nb for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 16]) != 1:
        return None
    return s_idx, n_idx


def _sulfonyl_on_benzene(mol: Any, s_idx: int) -> bool:
    """True iff the sulfonyl sulfur is bonded to a benzene ring carbon.

    F-B: the benzene suffix path owns the N,ring-locant merge, but only when the
    RING is the parent hydride. An N-aryl substituent on a chain parent
    (``N-phenylmethanesulfonamide``, S bonded to a methyl) must NOT be rerouted to
    a benzene parent, so this checks the atom the sulfonyl is actually attached to.
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    ring_info = mol.GetRingInfo()
    for nb in s_atom.GetNeighbors():
        if nb.GetAtomicNum() != 6 or not nb.GetIsAromatic():
            continue
        for ring in ring_info.AtomRings():
            if nb.GetIdx() not in ring or len(ring) != 6:
                continue
            # An ISOLATED benzene only: every ring atom aromatic carbon AND a member
            # of exactly one ring. A FUSED arene (naphthalene, indane, ...) has a
            # shared bond -> NumAtomRings > 1 for the bridgeheads, and delegating it
            # to the benzene namer would rename it as benzene, DROPPING the fused
            # carbons (a wrong molecule; F-B fable BLOCKER 1). Reject those here.
            if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                   and mol.GetAtomWithIdx(i).GetIsAromatic()
                   and ring_info.NumAtomRings(i) == 1
                   for i in ring):
                return True
    return False


def _branch_atoms(mol: Any, start: int, blocked: int) -> List[int]:
    """Atoms of the substituent branch rooted at ``start``, in BFS order.

    ``blocked`` (the amide nitrogen) is never crossed, so the walk stays inside
    the branch. The attachment atom is first, which is the ordering
    ``_name_n_substituent`` requires.
    """
    from collections import deque

    seen = {blocked}
    order: List[int] = []
    queue = deque([start])
    while queue:
        idx = queue.popleft()
        if idx in seen:
            continue
        seen.add(idx)
        order.append(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in seen:
                queue.append(nb.GetIdx())
    return order


def _collect_n_substituents(
    mol: Any, s_idx: int, n_idx: int,
) -> Optional[List[List[int]]]:
    """Branches hanging off the amide N, or None if the shape is refused."""
    n_atom = mol.GetAtomWithIdx(n_idx)
    branches: List[List[int]] = []
    for nb in n_atom.GetNeighbors():
        if nb.GetIdx() == s_idx:
            continue
        if nb.GetAtomicNum() != 6:
            # An N-heteroatom substituent (N-hydroxy, N-amino, N-N) is a
            # different construction (P-66.1.1.3.2 hydroxamic etc.).
            return None
        branch = _branch_atoms(mol, nb.GetIdx(), n_idx)
        if s_idx in branch:
            # The branch loops back to the sulfonyl: a cyclic sulfonamide
            # (sultam). Its parent is the ring, not an N-substituted chain.
            return None
        branches.append(branch)

    if not branches:
        return None  # primary sulfonamide -- not this class
    if len(branches) > 2:
        return None  # a trivalent N cannot carry three C plus the S
    if len(branches) == 2 and set(branches[0]) & set(branches[1]):
        # The two branches share atoms, i.e. N sits in a ring closed through
        # them. Refuse rather than spell the shared atoms twice.
        return None
    return branches


def _parent_hydride_is_unsubstituted(mol: Any, s_idx: int) -> bool:
    """True iff the R of R-SO2-N< carries no substituent of its own.

    WHY THIS GUARD EXISTS (it is not conservatism -- it prevents a wrong name).
    When the parent hydride is itself substituted, the N-locant and the parent's
    numerical locants belong to ONE alphanumerically ordered prefix list, not to
    two concatenated ones. The Blue Book spells this out at ``:32879``:

        ``N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)``

    -- ``N,4-dimethyl``, NOT ``N-methyl-4-methyl``. The ordering is
    alphanumerical across the MERGED list, not "N first": ``:32881`` names a
    substituted-parent SULFONAMIDE
    ``3-chloro-N-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)``, citing the
    parent's ``3-chloro`` BEFORE the N-prefix. This module names the parent by
    delegating to the pipeline and prefixing the result, so it CANNOT merge or
    re-order the two lists; for 4-methylbenzenesulfonamide + N-methyl it
    produced the malformed ``N-methyl4-methylbenzenesulfonamide``. OPSIN
    round-tripped that string to the correct InChIKey and reported OK -- a round
    trip proves the STRUCTURE, never the preferred SPELLING -- so only this
    structural refusal keeps the wrong name off the output.

    THE INVARIANT (state it as the invariant, not as a shape list)
    --------------------------------------------------------------
    True only when the delegated parent name is guaranteed to carry NO
    DETACHABLE SUBSTITUENT PREFIX. Suffix locants and unsaturation locants are
    fine -- they are part of the parent hydride's own name and nothing has to be
    merged with them, so ``N-methylbut-2-ene-1-sulfonamide`` is legal.

    The acyclic test used to be "an unbranched chain rooted at the attachment",
    and that was wrong in BOTH directions:

    * TOO PERMISSIVE -- a terminal heteroatom has degree 1, so it read as a
      chain terminus and passed. Measured on 2026-08-02, five malformed names
      were shipping, every one of them reported OK by OPSIN + InChIKey:
      ``N-methyl2-chloroethane-1-sulfonamide``,
      ``N-methyl2-methoxyethane-1-sulfonamide``,
      ``N-methyl2-fluoroethane-1-sulfonamide``,
      ``N-methyl2-cyanoethane-1-sulfonamide``,
      ``N,N-dimethyl3-bromopropane-1-sulfonamide``. A heteroatom in an acyclic R
      ALWAYS surfaces as a detachable prefix (chloro/methoxy/cyano...), so R
      must be all-carbon.
    * TOO RESTRICTIVE -- it forced the sulfonyl onto a chain TERMINUS by giving
      the root a degree limit of 1, refusing ``CC(C)S(=O)(=O)NC`` and
      ``CCC(C)S(=O)(=O)NC``. Their parents are ``propane-2-sulfonamide`` and
      ``butane-2-sulfonamide`` -- a suffix locant, no prefix -- so they were
      always safe. Interior attachment is fine; a BRANCH POINT is not.

    So the acyclic condition is exactly: R is all-carbon AND R is a PATH graph.
    r_atoms is connected by construction (BFS from the root) and ring-free in
    this branch, so "every degree within R is <= 2" is equivalent to "R is a
    path", and a path attached to S at any position is a parent hydride whose
    name has no prefix. A branch point (degree 3) means a real substituent --
    ``CC(C)(C)S(=O)(=O)NC`` is ``N,2-dimethylpropane-2-sulfonamide``, a merged
    list -- and stays refused.

    The ring case is left alone deliberately. Widening it to fused ring SYSTEMS
    was measured to have ZERO reachable targets: the PRIMARY parents themselves
    already fail upstream (``NS(=O)(=O)c1ccc2ccccc2c1`` and
    ``c1ccncc1S(=O)(=O)N`` both name ``unknown organic compound``), so
    ``_excised_parent_name`` would refuse regardless.

    The proof is structural throughout, never a test on the parent's name
    string.
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    roots = [nb.GetIdx() for nb in s_atom.GetNeighbors()
             if nb.GetAtomicNum() not in (7, 8)]
    if len(roots) != 1:
        return False
    r_atoms = set(_branch_atoms(mol, roots[0], s_idx))
    if not r_atoms:
        return False

    ring_info = mol.GetRingInfo()
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in r_atoms):
        # R is ring-based: it must be EXACTLY one ring and nothing else, so the
        # ring bears no substituent other than the sulfonyl. A heteroatom INSIDE
        # the ring is part of the parent hydride's own name (pyridine...), not a
        # prefix, so it is not excluded here.
        for ring in ring_info.AtomRings():
            if r_atoms == set(ring):
                return True
        return False

    # R is acyclic. (a) all-carbon: any heteroatom becomes a detachable prefix.
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r_atoms):
        return False

    # (b) a PATH graph: no atom may branch. The root's limit is 2, not 1 --
    # attachment at an INTERIOR chain position is still an unsubstituted parent.
    for idx in r_atoms:
        neighbours_in_r = sum(
            1 for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nb.GetIdx() in r_atoms
        )
        if neighbours_in_r > 2:
            return False
    return True


def _excised_parent_name(
    mol: Any, branches: List[List[int]], style: str,
) -> Optional[str]:
    """Name R-SO2-NH2 by deleting the N-substituents and re-entering naming."""
    from ..errors import is_failure_name

    doomed = sorted({idx for branch in branches for idx in branch}, reverse=True)
    try:
        rw = Chem.RWMol(mol)
        for idx in doomed:
            rw.RemoveAtom(idx)
        parent = rw.GetMol()
        for atom in parent.GetAtoms():
            # Let RDKit restore the N-H hydrogens the excision freed up.
            if atom.GetAtomicNum() == 7:
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(parent)
        parent_smiles = Chem.MolToSmiles(parent)
    except Exception:
        return None

    from ..namer import name_compound
    parent_name = name_compound(parent_smiles, style=style)
    if not parent_name or is_failure_name(parent_name):
        return None
    # Structural check on the produced name: the excised parent MUST still be a
    # sulfonamide. Anything else means the excision changed the class, and
    # prefixing 'N-...' onto it would name a different molecule.
    if not parent_name.endswith(_PARENT_SUFFIX):
        return None
    return parent_name


def n_substituted_sulfonamide_name(
    mol: Any, style: str = "pin",
) -> Optional[str]:
    """Return the P-66.1.1.3.1.1 PIN for an N-substituted sulfonamide, else None.

    ``N-methylmethanesulfonamide``, ``N,N-dimethylbenzenesulfonamide``,
    ``N-phenylmethanesulfonamide``. Fails closed on every shape outside the
    class documented in this module's docstring.
    """
    if mol is None:
        return None
    found = _sulfonyl_nitrogen(mol)
    if found is None:
        return None
    s_idx, n_idx = found

    branches = _collect_n_substituents(mol, s_idx, n_idx)
    if branches is None:
        return None

    # F-B (P-66.1.1.3.1.1 + P-14.3.4.2): when the parent hydride is a BENZENE ring
    # the italic-N and ring locants must be merged into ONE alphanumerical prefix
    # list and the suffix '1' cited on a di-substituted ring
    # (`N,4-dimethylbenzene-1-sulfonamide`). Prefixing a delegated parent name
    # cannot merge the two lists, but the benzene suffix path builds it
    # structurally -- the same machinery that produces `N,4-dimethylbenzamide` --
    # so delegate to it. Restricted to a sulfonyl bonded to a benzene ring so an
    # N-aryl substituent on a CHAIN parent (`N-phenylmethanesulfonamide`) is never
    # misrouted to a benzene parent. Fail-safe: accept only a result that is still
    # a sulfonamide name (the excision-class check, mirrored here).
    if _sulfonyl_on_benzene(mol, s_idx):
        from .benzene import name_benzene_derivative
        ring_name = name_benzene_derivative(mol)
        if ring_name and ring_name.endswith(_PARENT_SUFFIX):
            return ring_name
        # Fall through: the bare-ring excise path below is the proven backstop and
        # a substituted-ring parent is refused there anyway (never a wrong name).

    # A substituted parent hydride needs the N and numerical locants merged into
    # one ordered prefix list (P-66.1.1.3.1.1; `N,4-dimethyl...` at :32879),
    # which prefixing a delegated parent name cannot do. Refuse.
    if not _parent_hydride_is_unsubstituted(mol, s_idx):
        return None

    # Name each branch through the SHARED N-substituent primitive, which owns
    # the decorated-ring path and the double-prefix guard. A branch it cannot
    # name fails the whole name closed -- never skipped, never counted.
    from .amides import _name_n_substituent, format_n_substitution

    subs: List[Dict[str, Any]] = []
    for branch in branches:
        carbon_count = sum(
            1 for idx in branch if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
        )
        sub_name = _name_n_substituent(mol, branch, carbon_count)
        if not sub_name:
            return None
        subs.append({"atoms": branch, "name": sub_name,
                     "carbon_count": carbon_count})

    parent_name = _excised_parent_name(mol, branches, style)
    if parent_name is None:
        return None

    n_prefix = format_n_substitution(subs)
    if not n_prefix:
        return None
    return f"{n_prefix}{parent_name}"


__all__ = ["n_substituted_sulfonamide_name"]
