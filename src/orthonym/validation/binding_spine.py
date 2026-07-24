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
totality: every bond claimed exactly once) and **P3** (charge totality);
later phases append P4 name-span anchoring on the final post-processed
name and P5 an independent arity check -- each appending to the SAME
``findings`` list, so the proof object grows without changing shape.

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
from typing import (Any, Dict, Iterator, List, Literal, NamedTuple, Optional,
                    Tuple)

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
    ``attachment`` is the atom index this fragment bonds to in its parent
    (``None`` for a parent/root token).
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

    def __init__(self, ids):
        self._parent = {i: i for i in ids}

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


def _p3_charge_totality(mol, spine, allow_charged, findings, stats):
    """P3: every formal charge claimed by exactly one token, plus E1 parity.

    Two independent statements, kept as separate codes on purpose:

    * A charged atom the name does not account for means the name is not
      spelling this species -- ``CHARGE_UNCLAIMED`` (or
      ``CHARGE_DOUBLE_CLAIMED`` when two tokens both claim it).
    * ``allow_charged=False`` is a SCOPE limit, not a spine defect. It
      reproduces the legacy E1 net-charge refusal exactly, under its own
      code ``NET_CHARGE_OUT_OF_SCOPE``, so a caller can tell "we decline to
      name ions here" from "this spine is wrong".

    The net charge is summed over the graph's atoms rather than read via
    ``Chem.GetFormalCharge``, which is the same number by definition and
    keeps this module free of an RDKit import (``mol`` stays duck-typed).
    """
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() != 0}
    owner: Dict[int, str] = {}
    for binding in spine.walk():
        for atom_idx in binding.charge_atom_ids:
            if atom_idx in owner:
                findings.append(Finding(
                    CHARGE_DOUBLE_CLAIMED,
                    f"charge on atom {atom_idx} claimed twice "
                    f"({owner[atom_idx]!r} and {binding.token!r})",
                    "error"))
            else:
                owner[atom_idx] = binding.token
    unclaimed = sorted(charged - set(owner))
    if unclaimed:
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


def verify_spine(mol, spine: BindingSpine, name: str, *,
                 mode: str = "audit", allow_charged: bool = False) -> SpineProof:
    """Prove that ``spine`` binds ``name`` to exactly the graph of ``mol``.

    Runs P1 (atom partition), P2 (bond totality) and P3 (charge totality);
    ``name`` is accepted but not yet inspected, because token-span anchoring
    must run against the FINAL post-processed name and is built in a later
    phase. ``mode`` selects P2's linkage policy: ``"audit"`` infers a unique
    undeclared cross-subtree bond as an attachment (what today's producers
    actually emit), ``"strict"`` requires every bond to be declared or
    internal.

    ``stats["proofs"]`` lists the proofs that actually ran, so ``ok`` is never
    mistaken for a stronger guarantee than was computed -- P2 drops out of
    that tuple when a failed P1 makes bond ownership undecidable.
    """
    findings: list = []
    stats: Dict[str, Any] = {
        "mode": mode,
        "allow_charged": allow_charged,
        "name_len": len(name or ""),
        "bindings": sum(1 for _ in spine.walk()),
        "legacy_role_coerced": tuple(spine.legacy_role_coerced),
        "proofs": ("P1",),
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

    _p3_charge_totality(mol, spine, allow_charged, findings, stats)
    proofs.append("P3")
    stats["proofs"] = tuple(proofs)
    return SpineProof(
        ok=not any(f.severity == "error" for f in findings),
        findings=tuple(findings),
        stats=stats,
    )
