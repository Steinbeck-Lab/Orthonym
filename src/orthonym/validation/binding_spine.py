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
nothing, and changes no emitted name. ``verify_spine`` runs proof **P1**
(atom partition: exactly-once, no unbound, no phantom) and nothing else;
later phases append P2 bond totality, P3 charge totality, P4 name-span
anchoring on the final post-processed name, and P5 an independent arity
check -- each appending to the SAME ``findings`` list, so the proof object
grows without changing shape.

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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, Literal, Optional, Tuple

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
    legacy_role_coerced: Tuple[str, ...] = field(default=())

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


def verify_spine(mol, spine: BindingSpine, name: str, *,
                 mode: str = "audit", allow_charged: bool = False) -> SpineProof:
    """Prove that ``spine`` binds ``name`` to exactly the graph of ``mol``.

    Phase 1 runs P1 (atom partition) only; ``name`` is accepted but not yet
    inspected, because token-span anchoring must run against the FINAL
    post-processed name and is built in a later phase. ``mode`` is recorded
    for that future branch ("audit" = observe, never veto).
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
    # The owner map is returned for the later proofs (P2 needs "which token
    # owns each bond endpoint"); P1 itself only needs its findings.
    _p1_atom_partition(mol, spine, allow_charged, findings, stats)
    return SpineProof(
        ok=not any(f.severity == "error" for f in findings),
        findings=tuple(findings),
        stats=stats,
    )
