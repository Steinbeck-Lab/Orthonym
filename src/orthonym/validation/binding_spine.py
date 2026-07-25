"""v29 Phase 1: the recursion-safe name<->graph binding spine.

WHY this module exists
----------------------
v29 pushes best-effort naming breadth far past the ``general_engine``'s
flat, chain-shaped emissions, and that is only safe if an emitted name can
be PROVEN to spell exactly the input graph -- cheaply, in-process, with no
Java. The existing E1 certificate (``e1_certificate.py``) proves a flat
atom->token partition, which is sound only while every token is atomic. It
is not: a substituent prefix token like ``"2-chloroethyl"`` is a COMPOSITE
that spells a sub-name, and under a flat model such a token may legitimately
claim a whole subtree -- including atoms whose morphemes live in its own
inner name. A flat partition therefore cannot distinguish "this token says
those atoms" from "this token was handed those atoms".

The spine closes that hole with one invariant:

    **EXCLUSIVE CLAIM** -- a binding's ``atom_ids`` are the atoms that this
    token *itself* spells, and never include an atom spelled by one of its
    ``children``. Nesting in the name is nesting in the spine, so
    ``subtree_atoms() == atom_ids | union(child.subtree_atoms())``.

Consequence: a composite token can no longer claim a subgraph it does not
say. Every atom in the molecule must be claimed by exactly one binding
*somewhere* in the tree, at the depth whose morpheme actually spells it.

Scope and status (Phase 1)
--------------------------
AUDIT-ONLY and PURE. This module is wired into no naming path, gates
nothing, and changes no emitted name. ``verify_spine`` runs proofs **P1**
(atom partition: exactly-once, no unbound, no phantom), **P2** (bond
totality: every bond claimed exactly once), **P3** (charge totality),
**P4** (every token anchored to real spans of the FINAL name, with
multiplicity), **P5** (name morphemes no binding accounts for -- the dual
of P1's atom coverage) and **P6** (each token's claimed atom count checked
against an INDEPENDENT estimate of what its text spells). Each appends to
the SAME ``findings`` list, so the proof object grows without changing
shape.

P1-P3 all take the producer at its word about WHICH atoms a token covers:
they check the claims are mutually consistent and total, never that the
token's TEXT says those atoms. P4 and P6 are the two halves of closing
that: P4 proves the token is spelled in the name at all (and the right
number of times), P6 proves the number of atoms it claims is the number
its morphemes spell. P6 is the falsifier -- it is what makes a token
unable to claim atoms it does not say.

Atom coverage alone is not structure: a name can spell exactly the right
atom set while describing the wrong bonds (``propylpropane`` and
``cyclohexane`` claim the same six carbons). P2 is therefore not a
refinement of P1 but the other half of it -- P1 fixes *which* atoms the
name accounts for, P2 fixes *how they are joined*.

It SUPERSEDES NOTHING yet: ``e1_certificate.py`` is untouched and remains
the production emission gate for this phase.

Fail-closed discipline
----------------------
A proof must never be confidently wrong. A finding is an ``"error"`` only
when the spine and the graph provably disagree; anything the current proofs
cannot decide is reported as its own code (``*_UNVERIFIED``) or not claimed
at all -- never silently folded into ``ok``. ``ok`` means "no error-severity
finding was raised by the proofs that actually ran", so callers that need a
stronger guarantee must check which proofs ran (``stats["proofs"]``).

``verify_spine`` requires a real RDKit ``mol``; it raises rather than
returning a verdict when handed ``None``, because a crash is safe and a
fabricated pass is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (Any, Dict, Iterable, Iterator, List, Literal, NamedTuple,
                    Optional, Sequence, Tuple)

from orthonym.validation.name_morphemes import token_arity

# --------------------------------------------------------------------------
# Finding codes. Each constant's value equals its name so a code can be
# logged, grouped and compared without a lookup table.
# --------------------------------------------------------------------------
# P1 -- atom partition
ATOM_DOUBLE_BOUND = "ATOM_DOUBLE_BOUND"
ATOM_UNBOUND = "ATOM_UNBOUND"
ATOM_PHANTOM = "ATOM_PHANTOM"
# P2 -- bond totality
BOND_UNCLAIMED = "BOND_UNCLAIMED"
BOND_DOUBLE_CLAIMED = "BOND_DOUBLE_CLAIMED"
BOND_AMBIGUOUS_LINKAGE = "BOND_AMBIGUOUS_LINKAGE"
# P3 -- charge totality
CHARGE_UNCLAIMED = "CHARGE_UNCLAIMED"
CHARGE_UNVERIFIED = "CHARGE_UNVERIFIED"
CHARGE_DOUBLE_CLAIMED = "CHARGE_DOUBLE_CLAIMED"
NET_CHARGE_OUT_OF_SCOPE = "NET_CHARGE_OUT_OF_SCOPE"
# P4 -- name-span anchoring
TOKEN_ABSENT = "TOKEN_ABSENT"
TOKEN_SPAN_OVERLAP = "TOKEN_SPAN_OVERLAP"
TOKEN_SUBSTRING_ONLY = "TOKEN_SUBSTRING_ONLY"
MULTIPLICITY_MISMATCH = "MULTIPLICITY_MISMATCH"
UNBOUND_MORPHEME = "UNBOUND_MORPHEME"
# P5 -- independent arity check
ARITY_MISMATCH = "ARITY_MISMATCH"
ARITY_UNVERIFIED = "ARITY_UNVERIFIED"


class BindingKind(str, Enum):
    """What a token contributes to the name.

    Values are the lowercase member names so a kind serialises readably
    into stats and logs. ``PARENT``/``SUFFIX``/``PREFIX`` are the kinds the
    legacy flat bindings can express; the rest exist for the later phases
    (functional replacement, fusion assembly, charge suffixes, added
    hydrogen) so that spine consumers never need a second vocabulary.
    """

    PARENT = "parent"
    SUFFIX = "suffix"
    PREFIX = "prefix"
    REPLACEMENT = "replacement"
    FUSION = "fusion"
    CHARGE = "charge"
    HYDRO = "hydro"


# Legacy ``TokenBinding.role`` -> kind. Anything absent coerces to PREFIX
# (the only role a substituent-shaped token can safely be assumed to play)
# and is recorded, never silently absorbed.
_LEGACY_ROLE_KINDS: Dict[str, BindingKind] = {
    "parent": BindingKind.PARENT,
    "suffix": BindingKind.SUFFIX,
    "prefix": BindingKind.PREFIX,
}


@dataclass(frozen=True)
class SpineBinding:
    """One token and the atoms it *itself* spells.

    ``atom_ids`` is EXCLUSIVE: atoms spelled by a nested token belong to
    that child, not here. ``bond_ids`` and ``charge_atom_ids`` follow the
    same exclusivity and stay empty until P2/P3 populate them.
    ``attachment`` is an atom OF THIS FRAGMENT -- a member of this binding's
    own ``atom_ids`` -- namely the one bonded to the enclosing token. It is
    NOT the atom on the parent/enclosing side of that bond (``None`` for a
    parent/root token, which has no enclosing token to attach to).
    """

    token: str
    kind: BindingKind
    atom_ids: frozenset[int]
    bond_ids: frozenset[int] = frozenset()
    charge_atom_ids: frozenset[int] = frozenset()
    children: Tuple["SpineBinding", ...] = ()
    attachment: Optional[int] = None

    def subtree_atoms(self) -> frozenset[int]:
        """Atoms spelled by this token together with its whole subtree."""
        atoms = set(self.atom_ids)
        for child in self.children:
            atoms |= child.subtree_atoms()
        return frozenset(atoms)

    def walk(self) -> Iterator["SpineBinding"]:
        """Pre-order traversal of this subtree, self first."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class BindingSpine:
    """The full name<->graph binding tree for one emission.

    ``legacy_role_coerced`` carries the raw ``TokenBinding.role`` strings
    that ``from_token_bindings`` could not map, so ``verify_spine`` can
    report them in ``stats`` instead of the adapter having to log out of
    band. It defaults to empty for hand-built spines.
    """

    roots: Tuple[SpineBinding, ...]
    legacy_role_coerced: Tuple[str, ...] = ()

    def walk(self) -> Iterator[SpineBinding]:
        """Pre-order traversal of every binding in every root."""
        for root in self.roots:
            yield from root.walk()

    @classmethod
    def from_token_bindings(cls, bindings) -> "BindingSpine":
        """Adapt legacy flat ``TokenBinding``s into a (flat) spine.

        The legacy producers claim a substituent's WHOLE branch subtree
        under a single composed token string, so no nesting can be
        recovered without re-deriving the sub-name: every adapted binding
        is therefore a root with ``children=()``. That is a faithful
        widening -- the flat spine proves exactly what the flat E1
        certificate proved, no more -- and it is deliberately NOT an
        attempt to infer structure the legacy binding never recorded.
        """
        roots = []
        coerced = []
        for binding in bindings:
            role = (getattr(binding, "role", "") or "").strip().lower()
            kind = _LEGACY_ROLE_KINDS.get(role)
            if kind is None:
                kind = BindingKind.PREFIX
                coerced.append(getattr(binding, "role", ""))
            roots.append(SpineBinding(
                token=binding.token,
                kind=kind,
                atom_ids=frozenset(binding.atom_ids),
            ))
        return cls(roots=tuple(roots),
                   legacy_role_coerced=tuple(coerced))


Severity = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class Finding:
    """One thing a proof observed. Only ``"error"`` blocks ``ok``."""

    code: str
    detail: str
    severity: str


@dataclass(frozen=True)
class SpineProof:
    """Result of running the proofs that are implemented so far.

    ``stats["proofs"]`` names them, so a caller can tell "passed every
    proof there is" from "passed the two that ran".
    """

    ok: bool
    findings: Tuple[Finding, ...]
    stats: Dict[str, Any]

    def codes(self) -> Tuple[str, ...]:
        """Finding codes in the order they were raised."""
        return tuple(f.code for f in self.findings)


def _p1_atom_partition(mol, spine, allow_charged, findings, stats):
    """P1: every heavy atom claimed by EXACTLY ONE binding in the tree.

    Walks the whole spine, so a child's exclusive claim counts toward the
    partition and an overlap between a parent and its own child is caught
    as a double bind -- the hole the flat certificate could not see.

    ``allow_charged`` is accepted (and unused) because charge is proved by
    P3, not here; keeping it in the signature keeps the proof helpers
    uniform for the later phases.
    """
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    owner = {}
    for b in spine.walk():
        for idx in b.atom_ids:
            if idx in owner:
                findings.append(Finding(
                    ATOM_DOUBLE_BOUND,
                    f"atom {idx} bound twice ({owner[idx]!r} and {b.token!r})",
                    "error"))
            else:
                owner[idx] = b.token
    unbound = sorted(heavy - set(owner))
    if unbound:
        findings.append(Finding(ATOM_UNBOUND,
                                f"unbound heavy atoms: {unbound}", "error"))
    phantom = sorted(set(owner) - heavy)
    if phantom:
        findings.append(Finding(ATOM_PHANTOM,
                                f"bindings reference non-heavy/missing atoms: {phantom}",
                                "error"))
    stats["atoms_heavy"] = len(heavy)
    stats["atoms_bound"] = len(owner)
    return owner


class _Node(NamedTuple):
    """One binding plus the tree facts P2 needs, computed once."""

    binding: SpineBinding
    depth: int
    root: int
    subtree: frozenset


def _flatten(spine: BindingSpine) -> List[_Node]:
    """Every binding with its depth, owning root index and subtree atoms.

    ``subtree_atoms()`` is recursive, so it is evaluated once per node here
    rather than once per (node, bond) pair inside P2's loop.
    """
    nodes: List[_Node] = []

    def descend(binding: SpineBinding, depth: int, root: int) -> None:
        nodes.append(_Node(binding, depth, root, binding.subtree_atoms()))
        for child in binding.children:
            descend(child, depth + 1, root)

    for root_idx, root_binding in enumerate(spine.roots):
        descend(root_binding, 0, root_idx)
    return nodes


def _deepest_owners(nodes: List[_Node], i: int, j: int) -> List[_Node]:
    """The maximal-depth bindings whose subtree contains both endpoints.

    A bond internal to a subtree is spelled by the DEEPEST binding that
    contains it: every ancestor also contains it, but an ancestor's own
    morpheme does not say that join -- the nested token's does. Under the
    EXCLUSIVE CLAIM invariant the containing bindings form an ancestor chain
    (two bindings off different branches cannot share an endpoint), so the
    maximum is unique and the returned list has one element.

    Empty list = no binding contains both endpoints, i.e. the bond crosses
    two subtrees. More than one element = the impossible tie; it is RETURNED
    rather than resolved so the caller can refuse instead of guessing which
    token spells the bond.
    """
    owners = [n for n in nodes if i in n.subtree and j in n.subtree]
    if not owners:
        return []
    deepest = max(n.depth for n in owners)
    return [n for n in owners if n.depth == deepest]


def _piece_of_atom(in_scope: Dict[int, Tuple[int, int]],
                   subtrees: List[frozenset]) -> Dict[int, int]:
    """Map each atom to the internally connected piece of its root subtree.

    A root subtree need not be connected: the single suffix token
    ``-oic acid`` claims two oxygens that are bonded to the parent carbon but
    not to each other, so it contributes TWO pieces. Contracting pieces
    rather than whole subtrees is what makes the linkage acyclicity test
    correct for multivalent tokens.
    """
    adjacency: Dict[int, List[int]] = {}
    for i, j in in_scope.values():
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)
    piece: Dict[int, int] = {}
    next_id = 0
    for subtree in subtrees:
        for start in sorted(subtree):
            if start in piece:
                continue
            piece[start] = next_id
            stack = [start]
            while stack:
                for neighbour in adjacency.get(stack.pop(), ()):
                    if neighbour in subtree and neighbour not in piece:
                        piece[neighbour] = next_id
                        stack.append(neighbour)
            next_id += 1
    return piece


class _Merges:
    """Union-find over contracted pieces; ``join`` reports cycle closure."""

    def __init__(self, ids: Iterable[int]) -> None:
        self._parent: Dict[int, int] = {i: i for i in ids}

    def _find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def join(self, a: int, b: int) -> bool:
        """True if ``a`` and ``b`` were separate (so this edge was forced)."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return False
        # Lower id wins so the result never depends on iteration order.
        self._parent[max(ra, rb)] = min(ra, rb)
        return True


def _p2_bond_totality(mol, spine, mode, findings, stats):
    """P2: every heavy-atom bond claimed exactly once.

    Three ways a bond can be claimed:

    1. **Declared** -- some binding lists it in ``bond_ids``. Two bindings
       listing the same bond is ``BOND_DOUBLE_CLAIMED``.
    2. **Internal** -- both endpoints lie inside one binding's subtree. The
       claim belongs to the DEEPEST such binding, because that is the token
       whose own morpheme spells the join; an ancestor merely contains it.
       Under the EXCLUSIVE CLAIM invariant the candidates form an
       ancestor chain, so "deepest" is unique -- see the tie note below.
    3. **Linkage** -- the endpoints lie in two different root subtrees, so
       the bond is the attachment that joins a substituent to its parent.
       Production producers almost never declare ``bond_ids``, so in
       ``mode="audit"`` this is the NORMAL path, not an exotic fallback. In
       ``mode="strict"`` nothing may be inferred: ``BOND_UNCLAIMED``.

    When is an undeclared linkage safe to infer? Contract every internally
    connected piece of every root subtree to a single node and add the cross
    bonds as edges. A cross bond whose endpoints are NOT yet connected is
    FORCED -- the atom claims admit no other way to join those pieces, so
    inferring it invents nothing. A cross bond whose endpoints are ALREADY
    connected closes a cycle, and a cycle is a ring, bridge or fusion that no
    substituent prefix spells: ``BOND_AMBIGUOUS_LINKAGE``. Declared cross
    bonds are contracted first, since the name states them outright.

    Acyclicity, NOT a per-pair bond count, is the criterion, because counting
    is wrong in both directions:

    * Too strict: ``-oic acid`` is ONE token spelling TWO bonds (C=O and
      C-OH) to the SAME parent carbon. Its two claimed oxygens are separate
      pieces, so both bonds are forced and nothing is ambiguous -- a count
      rule refuses every carboxylic acid.
    * Too lax: cyclohexane spelled as three disjoint 2-carbon bindings has
      only ONE bond between each pair, yet the three together close the ring.
      A count rule passes a name that does not spell the graph, which is the
      exact failure P2 exists to prevent.

    Only bonds between two heavy atoms are in scope, matching P1's heavy-atom
    partition -- a mol carrying explicit hydrogens must not manufacture
    unclaimed bonds for atoms no binding is expected to name.

    P2 is SKIPPED when P1 already failed: with a broken partition, "which
    binding owns this endpoint" is meaningless and every bond would produce a
    second, derivative finding. ``stats["p2_skipped"]`` records that.
    """
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    in_scope = {}
    hydrogen_bonds = 0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in heavy and j in heavy:
            in_scope[bond.GetIdx()] = (i, j)
        else:
            hydrogen_bonds += 1
    stats["bonds_total"] = len(in_scope)
    stats["bonds_hydrogen"] = hydrogen_bonds

    nodes = _flatten(spine)
    declarers: Dict[int, List[str]] = {}
    for node in nodes:
        for bond_idx in node.binding.bond_ids:
            declarers.setdefault(bond_idx, []).append(node.binding.token)
    # A declared id that is not an in-scope bond of this mol cannot be
    # checked for totality (it claims nothing that exists). It is surfaced
    # in stats rather than as a finding: P1's ATOM_PHANTOM already catches
    # the atom-level form of this producer bug, and P2 has no code for it.
    stats["bonds_declared_unknown"] = sum(
        1 for bond_idx in declarers if bond_idx not in in_scope)

    # Atom -> root subtree. Sound only because P1 passed (every heavy atom
    # is claimed exactly once, so root subtrees are disjoint and total).
    root_of: Dict[int, int] = {}
    for node in nodes:
        if node.depth == 0:
            for atom_idx in node.subtree:
                root_of[atom_idx] = node.root

    bonds_claimed = 0
    linkages_inferred = 0
    declared_cross: List[Tuple[int, int, int]] = []
    undeclared_cross: List[Tuple[int, int, int, int, int]] = []

    for bond_idx, (i, j) in sorted(in_scope.items()):
        declared_by = declarers.get(bond_idx, ())
        if len(declared_by) > 1:
            findings.append(Finding(
                BOND_DOUBLE_CLAIMED,
                f"bond {bond_idx} ({i}-{j}) declared by "
                f"{sorted(declared_by)!r}",
                "error"))
            continue

        deepest = _deepest_owners(nodes, i, j)
        if declared_by:
            bonds_claimed += 1
            if not deepest:
                # Stated outright by the name, so it constrains what the
                # remaining undeclared cross bonds can still be inferred to be.
                declared_cross.append((bond_idx, i, j))
            continue

        if deepest:
            if len(deepest) > 1:
                # Unreachable while P1 passes: two bindings at equal depth
                # both containing both endpoints would have to share an
                # atom, which the partition forbids. Reported, not
                # asserted -- a proof may not crash on its own blind spot.
                findings.append(Finding(
                    BOND_DOUBLE_CLAIMED,
                    f"bond {bond_idx} ({i}-{j}) has {len(deepest)} equally "
                    f"deep owning bindings "
                    f"{sorted(n.binding.token for n in deepest)!r}",
                    "error"))
                continue
            bonds_claimed += 1
            continue

        left, right = root_of.get(i), root_of.get(j)
        if left is None or right is None:
            findings.append(Finding(
                BOND_UNCLAIMED,
                f"bond {bond_idx} ({i}-{j}) has an endpoint in no binding "
                f"subtree",
                "error"))
            continue
        undeclared_cross.append((bond_idx, i, j, left, right))

    if mode == "strict":
        for bond_idx, i, j, left, right in undeclared_cross:
            findings.append(Finding(
                BOND_UNCLAIMED,
                f"bond {bond_idx} ({i}-{j}) joins root subtrees "
                f"{left} and {right} and is not declared by any binding",
                "error"))
    else:
        piece = _piece_of_atom(
            in_scope, [n.subtree for n in nodes if n.depth == 0])
        merges = _Merges(set(piece.values()))
        for _, i, j in declared_cross:
            merges.join(piece[i], piece[j])
        # Ascending bond index: which bond of a cycle gets reported must not
        # depend on dict ordering.
        for bond_idx, i, j, left, right in undeclared_cross:
            if merges.join(piece[i], piece[j]):
                linkages_inferred += 1
            else:
                findings.append(Finding(
                    BOND_AMBIGUOUS_LINKAGE,
                    f"undeclared bond {bond_idx} ({i}-{j}) joining root "
                    f"subtrees {left} and {right} closes a cycle; the atoms "
                    f"are already joined, so this bond is a ring, bridge or "
                    f"fusion that no substituent prefix spells",
                    "error"))

    stats["bonds_claimed"] = bonds_claimed
    stats["linkages_inferred"] = linkages_inferred


def _p3_charge_totality(mol, spine, mode, allow_charged, findings, stats):
    """P3: every formal charge claimed by exactly one token, plus E1 parity.

    Three independent statements, kept as separate codes on purpose:

    * A charged atom the name does not account for, where SOME binding in
      the spine does declare at least one charge claim (just not one that
      covers this atom), means the name provably is not spelling this
      species -- ``CHARGE_UNCLAIMED`` (or ``CHARGE_DOUBLE_CLAIMED`` when two
      tokens both claim it).
    * A charged atom the name does not account for, where NO binding
      anywhere in the spine declares any ``charge_atom_ids`` at all, is a
      different, weaker statement: the 12 legacy production binding
      producers (``from_token_bindings``) structurally cannot populate
      ``charge_atom_ids`` -- they never emit charge claims -- so this shape
      is indistinguishable from "the producer never wires charge evidence
      through" rather than "the token failed to claim a charge it should
      have". Reporting it as an ``error`` would be confidently wrong: it
      would make P3 refuse every charged legacy-adapted spine even when the
      name is correct. So it gets its own code, ``CHARGE_UNVERIFIED``, at
      ``"warn"`` severity in ``mode="audit"`` (unproven, not disproven) and
      ``"error"`` in ``mode="strict"`` (nothing may be left unproven).
    * ``allow_charged=False`` is a SCOPE limit, not a spine defect. It
      reproduces the legacy E1 net-charge refusal exactly, under its own
      code ``NET_CHARGE_OUT_OF_SCOPE``, so a caller can tell "we decline to
      name ions here" from "this spine is wrong".

    Only HEAVY atoms are in scope for the charge-claim check, matching P1's
    and P2's heavy-atom scoping (documented at the top of ``_p2_bond_totality``
    and ``_p1_atom_partition``): under the EXCLUSIVE CLAIM invariant a binding
    can only ever legitimately claim a heavy atom, so a charged explicit
    hydrogen could never be claimed by any token and would be permanently
    unverifiable if counted here. The net charge total below is deliberately
    NOT scoped this way -- it must sum every atom's formal charge regardless
    of heaviness to reproduce the true net charge of the species.

    The net charge is summed over the graph's atoms rather than read via
    ``Chem.GetFormalCharge``, which is the same number by definition and
    keeps this module free of an RDKit import (``mol`` stays duck-typed).
    """
    charged = {a.GetIdx() for a in mol.GetAtoms()
               if a.GetAtomicNum() > 1 and a.GetFormalCharge() != 0}
    owner: Dict[int, str] = {}
    declared_count = 0
    for binding in spine.walk():
        for atom_idx in binding.charge_atom_ids:
            declared_count += 1
            if atom_idx in owner:
                findings.append(Finding(
                    CHARGE_DOUBLE_CLAIMED,
                    f"charge on atom {atom_idx} claimed twice "
                    f"({owner[atom_idx]!r} and {binding.token!r})",
                    "error"))
            else:
                owner[atom_idx] = binding.token
    stats["charge_claims_declared"] = declared_count
    unclaimed = sorted(charged - set(owner))
    if unclaimed:
        if declared_count == 0:
            # No binding anywhere even attempted a charge claim -- see the
            # CHARGE_UNVERIFIED docstring section above. Not a provable
            # disagreement, so it must never reach "error" in audit mode.
            findings.append(Finding(
                CHARGE_UNVERIFIED,
                f"formally charged atoms present but no binding in the "
                f"spine declares any charge_atom_ids: {unclaimed}",
                "error" if mode == "strict" else "warn"))
        else:
            findings.append(Finding(
                CHARGE_UNCLAIMED,
                f"formally charged atoms claimed by no token: {unclaimed}",
                "error"))
    # Claims on uncharged atoms cannot make the name wrong about a charge
    # that is not there; counted, not raised (P3 has no code for it).
    stats["charges_claimed_uncharged"] = len(set(owner) - charged)
    stats["charges_total"] = len(charged)
    stats["charges_claimed"] = len(set(owner) & charged)

    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    stats["net_charge"] = net_charge
    if not allow_charged and net_charge != 0:
        findings.append(Finding(
            NET_CHARGE_OUT_OF_SCOPE,
            f"net formal charge {net_charge:+d} and allow_charged is False",
            "error"))


# --------------------------------------------------------------------------
# P4/P5 morpheme lexicons.
#
# These are BOUNDARY evidence, not a nomenclature model. A name is a single
# unspaced string, so "is this token actually spelled here?" reduces to "does
# the token start and end at a morpheme boundary?", and the only cheap
# evidence for a boundary is that what sits against it is itself a recognised
# morpheme. The lists are therefore deliberately generous: a missing entry
# costs a FALSE ERROR on a correct name, which is the one outcome this
# milestone cannot ship, while a superfluous entry only costs a missed
# detection that P5/P6 may still catch.
#
# The four groupings below record where each morpheme came from and what it
# does in a name; they are NOT four different rules. Both boundary sides and
# the P5 residue scan all consult their union, ``_GLUE_MORPHEMES`` -- see
# ``_left_ok`` for why filing a morpheme as "left" or "right" turned out to be
# a source of false errors rather than a source of evidence.
# --------------------------------------------------------------------------
_MULTIPLIER_WORDS = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6,
                     "hepta": 7, "octa": 8, "nona": 9, "deca": 10,
                     "undeca": 11, "dodeca": 12,
                     "bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5,
                     "hexakis": 6}
# P-16.3.3: a multiplying prefix drops its terminal vowel before a
# vowel-initial suffix -- "butane-1,2,3,4-tetraol" is written "...-tetrol",
# "...-pentaol" is written "...-pentol". The elided spelling is the SAME
# morpheme with the same value, so it is DERIVED here rather than listed:
# hand-listing would invite the two tables to drift apart.
_MULTIPLIERS = {**_MULTIPLIER_WORDS,
                **{word[:-1]: value
                   for word, value in _MULTIPLIER_WORDS.items()
                   if word.endswith("a")}}
_LEFT_GLUE = frozenset(_MULTIPLIERS) | {
    "cyclo", "bicyclo", "tricyclo", "spiro", "iso", "neo", "sec", "tert",
    "bi", "ter", "o", "a", "n", "hydro"}
_RIGHT_GLUE = frozenset({
    "yl", "ylidene", "ylidyne", "oxy", "thio", "sulfanyl", "amino", "imino",
    "an", "ane", "en", "ene", "yn", "yne", "e", "ol", "one", "al", "oic",
    "ic", "ate", "amide", "amine", "nitrile", "ium", "ide", "ylium", "uide",
    "carbo", "sulfo", "sulfon", "phosph", "o", "a", "idene", "hydro"})
_STEREO_WORDS = frozenset({"r", "s", "e", "z", "rel", "rac", "cis", "trans",
                           "endo", "exo", "syn", "anti", "alpha", "beta",
                           "seqcis", "seqtrans"})
_CONNECTIVES = frozenset({"acid", "ester", "ether", "anhydride", "oxide",
                          "hydrate", "and", "of", "yl", "ylidene"})
_GLUE_MORPHEMES = _LEFT_GLUE | _RIGHT_GLUE | _STEREO_WORDS | _CONNECTIVES


def _is_letter(char: str) -> bool:
    """ASCII letter. Non-ASCII name characters (primes, italics) are not."""
    return char.isascii() and char.isalpha()


def _ends_with_any(text: str, words: Iterable[str]) -> bool:
    return any(text.endswith(word) for word in words)


def _starts_with_any(text: str, words: Iterable[str]) -> bool:
    return any(text.startswith(word) for word in words)


def _left_ok(lowered: str, i: int, token: str, tokens: Iterable[str]) -> bool:
    """Is position ``i`` a left morpheme boundary?

    Four ways to be one: nothing precedes; what precedes is not a letter; the
    TOKEN'S OWN first character is not a letter; or the text before ends with
    a recognised morpheme or with ANOTHER BINDING'S TOKEN.

    Both sides consult the SAME lexicon (``_GLUE_MORPHEMES``). Splitting it
    into a left list and a right list treats "which side of a token this
    morpheme usually sits on" as evidence about boundaries, which it is not:
    the saturation endings were filed as right-glue only, so every suffix
    sitting after one ('propan|amide', 'octadec-9-yne|nitrile') was refused
    ``TOKEN_SUBSTRING_ONLY`` -- the third distinct class of false error the
    asymmetry produced. A boundary is evidenced by a KNOWN MORPHEME abutting
    it, full stop.

    The other-token clause mirrors the right-hand rule and is load-bearing,
    not a convenience: in ``methylbenzene`` the parent token ``benzene`` is
    preceded by the prefix token ``methyl``, whose spelling ends in no glue
    morpheme. Without it every parent that follows a substituent prefix --
    very nearly every real emission -- would be reported
    ``TOKEN_SUBSTRING_ONLY``, a confidently WRONG error on a correct name.
    Four already-shipped P2 tests assert ``ok`` on exactly that name.

    The non-letter-token-edge clause exists because the rule is about a
    token's letters BLEEDING into a longer word. A token that itself begins
    with ``(`` or ends with ``]`` -- every von Baeyer parent the producers
    emit -- has no letters at that edge to bleed, so no glue evidence is
    needed there and demanding some was refusing correct names outright
    (measured: 5 of 24 real emissions).
    """
    if (i == 0 or not _is_letter(lowered[i - 1])
            or not _is_letter(token[0])):
        return True
    head = lowered[:i]
    return _ends_with_any(head, _GLUE_MORPHEMES) or _ends_with_any(head, tokens)


def _right_ok(lowered: str, j: int, token: str, tokens: Iterable[str]) -> bool:
    """Is position ``j`` a right morpheme boundary? (mirror of ``_left_ok``)"""
    if (j == len(lowered) or not _is_letter(lowered[j])
            or not _is_letter(token[-1])):
        return True
    tail = lowered[j:]
    return (_starts_with_any(tail, _GLUE_MORPHEMES)
            or _starts_with_any(tail, tokens))


def _occurrence_weight(lowered: str, i: int) -> int:
    """How many bindings one occurrence at ``i`` can be spelling.

    ``dimethylbenzene`` writes ``methyl`` ONCE for TWO bindings, so an
    occurrence is worth the multiplier immediately in front of it. The
    longest matching multiplier wins, so ``undeca`` reads 11 rather than the
    ``deca`` (10) it ends with.

    Enclosing marks are stepped over first because that is where the
    COMPLEX multiplicative prefixes live: ``bis``/``tris``/``tetrakis`` are
    by convention (P-16.3.2) always followed by enclosing marks, so the
    alphabetic run immediately against the token is empty and a naive scan
    scores every ``bis(...)`` prefix as a single occurrence -- measured as a
    false ``MULTIPLICITY_MISMATCH`` on real emissions.
    """
    end = i
    while end > 0 and lowered[end - 1] in "([{":
        end -= 1
    start = end
    while start > 0 and _is_letter(lowered[start - 1]):
        start -= 1
    run = lowered[start:end]
    if not run:
        return 1
    best = max((word for word in _MULTIPLIERS if run.endswith(word)),
               key=len, default=None)
    return _MULTIPLIERS[best] if best else 1


def _overlaps(span: Tuple[int, int], taken: Sequence[Tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end
               for other_start, other_end in taken)


def _p4_token_spans(spine, name, findings, stats) -> List[Tuple[int, int, str]]:
    """P4: every token anchored to real spans of the FINAL name.

    P1-P3 never look at the name, so a spine whose tokens spell something
    else entirely passes all three. P4 is the first proof that reads the
    string, and it asks two questions per token: is it SPELLED here (at a
    morpheme boundary, not merely present as a substring), and is it spelled
    the right NUMBER of times?

    Multiplicity is the half that is easy to miss. ``k`` bindings sharing one
    token text need not produce ``k`` occurrences: the name writes
    ``dimethyl`` once and the multiplier says it twice. So an occurrence
    carries a WEIGHT (its preceding multiplier's value, else 1) and the group
    is satisfied when the accumulated weight reaches ``k``. Three methyl
    bindings against a ``di`` therefore fail, which is a claim about the graph
    no atom-level proof can make.

    Groups are processed longest-token-first so that a token which is a
    prefix of another (``meth`` vs ``methyl``) cannot steal the longer one's
    span; a group left short only because its occurrences were already taken
    reports ``TOKEN_SPAN_OVERLAP`` rather than a multiplicity finding, since
    the name may well spell it and the collision is what blocked the proof.

    Matching is case-insensitive: names carry italic capitals (``N,N-``,
    ``O-``) that no producer puts in a token. That is a WIDENING -- it can
    only remove false ``TOKEN_ABSENT`` errors, never manufacture one.

    Empty-token bindings are skipped (counted in ``stats["empty_tokens"]``):
    ``CHARGE``/``HYDRO`` kinds may legitimately carry no text of their own,
    and an empty string matches everywhere.

    Spans chosen before a finding is raised are still recorded, so P5 does not
    then re-report the same text as unexplained.
    """
    lowered = (name or "").lower()
    counts: Dict[str, int] = {}
    empty = 0
    for binding in spine.walk():
        token = binding.token.strip().lower()
        if not token:
            empty += 1
            continue
        counts[token] = counts.get(token, 0) + 1
    stats["empty_tokens"] = empty
    tokens = frozenset(counts)

    assigned: List[Tuple[int, int, str]] = []
    taken: List[Tuple[int, int]] = []

    # Longest token first, then alphabetical: the outcome must not depend on
    # spine order.
    for token in sorted(counts, key=lambda t: (-len(t), t)):
        wanted = counts[token]
        occurrences: List[Tuple[int, int]] = []
        at = lowered.find(token)
        while at >= 0:
            end = at + len(token)
            if (_left_ok(lowered, at, token, tokens)
                    and _right_ok(lowered, end, token, tokens)):
                occurrences.append((at, end))
            at = lowered.find(token, at + 1)

        blocked = [span for span in occurrences if _overlaps(span, taken)]
        chosen: List[Tuple[int, int]] = []
        weight = 0
        for span in occurrences:
            if weight >= wanted:
                break
            if _overlaps(span, taken) or _overlaps(span, chosen):
                continue
            chosen.append(span)
            weight += _occurrence_weight(lowered, span[0])
        for start, end in chosen:
            taken.append((start, end))
            assigned.append((start, end, token))

        if weight == wanted:
            continue
        if not occurrences:
            if token in lowered:
                findings.append(Finding(
                    TOKEN_SUBSTRING_ONLY,
                    f"token {token!r} occurs in the name only inside a larger "
                    f"morpheme, never at a morpheme boundary",
                    "error"))
            else:
                findings.append(Finding(
                    TOKEN_ABSENT,
                    f"token {token!r} does not occur in the name at all",
                    "error"))
        elif weight > wanted:
            findings.append(Finding(
                MULTIPLICITY_MISMATCH,
                f"the name spells token {token!r} {weight} times "
                f"(counting multiplier prefixes) but {wanted} binding(s) "
                f"claim it",
                "error"))
        elif blocked:
            findings.append(Finding(
                TOKEN_SPAN_OVERLAP,
                f"token {token!r} is claimed by {wanted} binding(s) but only "
                f"{weight} occurrence(s) are free; {len(blocked)} overlap a "
                f"span already assigned to another token",
                "error"))
        else:
            findings.append(Finding(
                MULTIPLICITY_MISMATCH,
                f"the name spells token {token!r} {weight} times "
                f"(counting multiplier prefixes) but {wanted} binding(s) "
                f"claim it",
                "error"))

    assigned.sort()
    stats["spans"] = tuple(assigned)
    return assigned


def _longest_glue(text: str) -> int:
    """Length of the longest glue morpheme at position 0, or 0 if none."""
    best = 0
    for word in _GLUE_MORPHEMES:
        if len(word) > best and text.startswith(word):
            best = len(word)
    return best


def _p5_name_residue(name, spans, mode, findings, stats) -> None:
    """P5: name text that no binding accounts for -- the dual of P1.

    P1 asks whether every ATOM is spelled by some token. P5 asks the mirror
    question: is every MORPHEME backed by some binding? A name can partition
    the graph perfectly and still carry a whole extra functional suffix that
    no token claims, and only this direction sees it.

    Each maximal run of ASCII letters left uncovered by P4's spans is consumed
    greedily longest-match against ``_GLUE_MORPHEMES``; whatever cannot be
    consumed is reported. Severity is ``"warn"`` in ``mode="audit"`` and
    ``"error"`` in ``"strict"``, exactly as P3 escalates ``CHARGE_UNVERIFIED``:
    unexplained text is missing evidence rather than a proved disagreement,
    because the glue lexicon is a finite hand-list and an unlisted connective
    is a gap in the LEXICON, not in the name.

    KNOWN BLIND SPOT, deliberate: ``_GLUE_MORPHEMES`` includes the whole of
    ``_RIGHT_GLUE``, which carries atom-bearing morphemes (``sulfon``, ``ol``,
    ``amide``). An unclaimed sulfonic acid therefore consumes as glue and goes
    unreported here. Splitting the boundary lexicon from the residue lexicon
    would tighten this, at the price of false errors on every name whose
    connective morphemes are not yet listed -- the wrong trade for a proof
    that must never be confidently wrong. P6 covers part of the same ground
    from the atom side.
    """
    covered = bytearray(len(name))
    for start, end, _token in spans:
        for position in range(start, min(end, len(name))):
            covered[position] = 1

    remainders: List[str] = []
    index = 0
    while index < len(name):
        if not _is_letter(name[index]) or covered[index]:
            index += 1
            continue
        end = index
        while (end < len(name) and _is_letter(name[end])
               and not covered[end]):
            end += 1
        rest = name[index:end].lower()
        while rest:
            step = _longest_glue(rest)
            if not step:
                break
            rest = rest[step:]
        if rest:
            remainders.append(rest)
            findings.append(Finding(
                UNBOUND_MORPHEME,
                f"name text no binding accounts for: {rest!r}",
                "error" if mode == "strict" else "warn"))
        index = end

    stats["residue_runs"] = tuple(remainders)


def _p6_arity(spine, findings, stats) -> None:
    """P6: what a token CLAIMS, checked against what its text SPELLS.

    ``token_arity`` reads the token string against the project's own naming
    tables and knows nothing of the graph, the binding, or the producer that
    made either; ``len(binding.atom_ids)`` is the producer's bookkeeping.
    Their agreement is therefore evidence rather than tautology, and their
    disagreement means the name does not say what the spine claims it says.

    The comparison is against ``atom_ids``, the EXCLUSIVE claim, never
    ``subtree_atoms()``: a token's morphemes spell its own atoms, and its
    children's morphemes spell theirs. That is precisely how "``phenyl``
    claiming eight atoms" is caught while "``benzene`` with two nested
    ``methyl`` children" passes.

    The oracle is sound but partial -- it refuses anything it cannot decide,
    and on real production tokens it is confident about three quarters of
    them (von Baeyer descriptors dominate the refusals). ``ARITY_UNVERIFIED``
    is therefore COMMON and must never be an error: an unconfident estimate
    is the absence of evidence, and failing on it would refuse correct names
    wholesale.
    """
    confident = 0
    unverified = 0
    for binding in spine.walk():
        if not binding.token.strip():
            continue
        estimate = token_arity(binding.token, binding.kind)
        if estimate.confident:
            confident += 1
            claimed = len(binding.atom_ids)
            if estimate.heavy_atoms != claimed:
                findings.append(Finding(
                    ARITY_MISMATCH,
                    f"token {binding.token!r} claims {claimed} heavy atom(s) "
                    f"but its morphemes spell {estimate.heavy_atoms} "
                    f"({estimate.basis})",
                    "error"))
        else:
            unverified += 1
            findings.append(Finding(
                ARITY_UNVERIFIED,
                f"token {binding.token!r} arity not decidable: "
                f"{estimate.basis}",
                "info"))
    stats["arity_confident"] = confident
    stats["arity_unverified"] = unverified


def verify_spine(mol, spine: BindingSpine, name: str, *,
                 mode: str = "audit", allow_charged: bool = False) -> SpineProof:
    """Prove that ``spine`` binds ``name`` to exactly the graph of ``mol``.

    Runs P1 (atom partition), P2 (bond totality), P3 (charge totality),
    P4 (token spans on ``name``), P5 (name residue) and P6 (token arity).
    ``name`` must be the FINAL post-processed string, since that is what P4
    and P5 read. ``mode`` selects P2's linkage policy and the severity of the
    two unproven-not-disproven codes: ``"audit"`` infers a unique undeclared
    cross-subtree bond as an attachment (what today's producers actually
    emit) and warns on unexplained name text, ``"strict"`` requires every
    bond to be declared or internal and leaves nothing unproven.

    ``stats["proofs"]`` lists the proofs that actually ran, so ``ok`` is never
    mistaken for a stronger guarantee than was computed. A failed P1 drops
    both P2 (bond ownership becomes undecidable) and P5 (a token set that
    does not partition the graph makes the span map unreliable, so uncovered
    text says nothing). P4 and P6 read only the name and the tokens, so they
    still run -- and are often exactly what explains the P1 failure.
    """
    findings: list = []
    stats: Dict[str, Any] = {
        "mode": mode,
        "allow_charged": allow_charged,
        "name_len": len(name or ""),
        "bindings": sum(1 for _ in spine.walk()),
        "legacy_role_coerced": tuple(spine.legacy_role_coerced),
    }
    _p1_atom_partition(mol, spine, allow_charged, findings, stats)
    proofs = ["P1"]

    # P2 reads "which binding's subtree owns this endpoint", which only means
    # something once P1 has established a partition. On a broken partition it
    # is skipped rather than guessed -- reporting bond findings derived from a
    # known-bad ownership map would be a confidently wrong proof.
    p1_failed = any(f.severity == "error" for f in findings)
    stats["p2_skipped"] = p1_failed
    if not p1_failed:
        _p2_bond_totality(mol, spine, mode, findings, stats)
        proofs.append("P2")

    _p3_charge_totality(mol, spine, mode, allow_charged, findings, stats)
    proofs.append("P3")

    spans = _p4_token_spans(spine, name or "", findings, stats)
    proofs.append("P4")

    stats["p5_skipped"] = p1_failed
    if not p1_failed:
        _p5_name_residue(name or "", spans, mode, findings, stats)
        proofs.append("P5")

    _p6_arity(spine, findings, stats)
    proofs.append("P6")
    stats["proofs"] = tuple(proofs)
    return SpineProof(
        ok=not any(f.severity == "error" for f in findings),
        findings=tuple(findings),
        stats=stats,
    )
