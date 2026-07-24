"""v29 breadth metrics — the pinned definitions behind the breadth instrument.

Why this module exists: every breadth figure quoted before v29 came from an
ad-hoc harness in a session , and at least one was a ~4x mirage
because the harness and the production producer disagreed about what counts as
an emission. These functions are the single definition of each metric, unit
tested in ``tests/unit/metrics/test_breadth.py``, so a number measured in one
session is comparable to a number measured weeks later.

Pure logic: no naming, no OPSIN, no Java. The driver
(``) supplies per-molecule observations; this module
defines what they mean.
"""
from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Sequence, Set

from ..errors import is_failure_name

__all__ = [
    "ring_systems",
    "classify_outcome",
    "parse_refusal_codes",
    "molecule_components",
    "aggregate",
]


# --------------------------------------------------------------------- rings

def ring_systems(mol) -> List[Set[int]]:
    """Connected ring systems of ``mol`` as disjoint atom-index sets.

    Two SSSR rings belong to the same system when they share at least one atom,
    so ortho-fused (naphthalene), bridged and spiro systems each collapse to ONE
    system, while rings joined only by a bond or a linker (biphenyl) stay
    separate. This is the ring-SYSTEM grouping; the v1 census classifier that
    labelled linker-joined rings as 'fused' produced a badly wrong topology
    distribution, so the union-find is deliberate.
    """
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    parent = list(range(len(rings)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                parent[find(i)] = find(j)

    groups: Dict[int, Set[int]] = {}
    for i, ring in enumerate(rings):
        groups.setdefault(find(i), set()).update(ring)
    return list(groups.values())


# ------------------------------------------------------------------- outcome

def classify_outcome(result: Dict[str, Any]) -> str:
    """``EMIT`` or ``ABSTAIN`` for one ``name_tiered()`` result.

    Uses the production failure predicate (``errors.is_failure_name``) rather
    than a local truthiness check, because the PIN tier signals abstention with
    a DESCRIPTIVE string ('unknown organic compound') while the best-effort tier
    signals it with ``None``. Counting the descriptive string as an emission is
    precisely how a breadth number becomes a mirage.
    """
    return "ABSTAIN" if is_failure_name(result.get("name")) else "EMIT"


# ----------------------------------------------------------- refusal parsing

# The reason= field is OPTIONAL: real call sites log DROP-17 and DROP-HYG04 with
# no reason at all, and DROP-HYG04's suffix is not numeric. Requiring either would
# erase those sites from the census, making a genuinely blocking site rank as zero
# because of its log format rather than its behaviour.
_DROP_RE = re.compile(r"(DROP-[A-Za-z0-9]+)\b(?:[^|]*?\breason=([A-Za-z0-9_]+))?")
# [^|] guard mirrors _DROP_RE so a capture can never run past its own log record.
_REFUSE_RE = re.compile(r"general_engine refused:\s*([^|]+?)\s*(?:\(tier|$)")


def parse_refusal_codes(log_lines: Iterable[str]) -> List[str]:
    """Distinct refusal codes for ONE molecule, in first-seen order.

    Deduplicated on purpose: a code firing five times while the engine probes
    five substituents is still a single blocker for that molecule, and the
    milestone question is 'how many molecules does this site block', not 'how
    often does it fire'. Un-deduplicated counts would overstate the dominant
    site and mis-rank the build order.
    """
    seen: List[str] = []

    def add(code: str) -> None:
        if code not in seen:
            seen.append(code)

    for line in log_lines:
        for m in _DROP_RE.finditer(line):
            add(f"{m.group(1)}:{m.group(2) or '<no_reason>'}")
        for m in _REFUSE_RE.finditer(line):
            add(f"REFUSE:{m.group(1).strip()}")
    return seen


# ---------------------------------------------------------------- components

def molecule_components(mol) -> List[Dict[str, Any]]:
    """Partition heavy atoms into the pieces that must ALL name for the whole
    molecule to name.

    Whole-molecule success is approximately the PRODUCT of per-component
    success, so this partition is the denominator of ``per_fragment_p`` and the
    reason breadth is multiplicative: at ~3.3 components per molecule, 99%
    whole-molecule needs p ~= 0.997 per component.

    Components are the connected ring systems plus each connected run of
    acyclic atoms. The partition is total and disjoint by construction.
    """
    systems = ring_systems(mol)
    comps: List[Dict[str, Any]] = [
        {"kind": "ring_system", "atoms": sorted(s)} for s in systems
    ]

    ring_atoms = {a for s in systems for a in s}
    # Heavy atoms only. RDKit keeps ISOTOPIC hydrogens as explicit graph atoms
    # (unlike implicit H), so on a deuterated input GetAtoms() and
    # GetNumHeavyAtoms() disagree; without this filter the partition silently
    # covers H nodes and breaks this function's documented contract.
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    acyclic = [i for i in sorted(heavy) if i not in ring_atoms]

    # connected runs among the acyclic atoms (edges through ring atoms excluded,
    # so a linker bridging two rings splits into its own piece per side only if
    # genuinely disconnected without the rings)
    remaining = set(acyclic)
    while remaining:
        seed = remaining.pop()
        blob = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                idx = nb.GetIdx()
                if idx in remaining and idx in heavy:
                    remaining.discard(idx)
                    blob.add(idx)
                    stack.append(idx)
        comps.append({"kind": "acyclic", "atoms": sorted(blob)})

    return comps


# ----------------------------------------------------------------- aggregate

def aggregate(rows: Sequence[Dict[str, Any]],
              components_measured: bool = True) -> Dict[str, Any]:
    """Roll per-molecule observations into the milestone metrics.

    ``rows`` entries carry: ``outcome``, ``tier``, ``refusal_codes``,
    ``structure_wrong``, ``opsin_unparseable``, ``n_components``,
    ``n_components_named``.

    Two separable loss terms are reported, because they map to DIFFERENT build
    phases and conflating them hides which one is responsible for a flat number:

    * **component loss** — ``per_fragment_p`` < 1: a component cannot be named
      even standalone (ring / fragment naming gaps).
    * **context loss** — ``context_loss``: components that name standalone but
      whose molecule still abstains (the assembly gap).

    ``projected_emit_independent`` is E[p**n] over the observed component-count
    distribution, NOT p**E[n]: the latter is convex in n and understates the
    independence model materially on any real corpus (8pp on live data). Treat it
    as an upper bound, not a calibrated predictor — p is measured on components
    cut out and capped with implicit H, which is easier than naming them in
    context, so the independence model overshoots observed emit.
    """
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "emit_rate": 0.0, "tiers": {}, "structure_wrong": 0,
            "opsin_unparseable": 0,
            "refusal_census": {}, "refusal_census_abstain": {},
            "per_fragment_p": 0.0, "mean_components": 0.0,
            "projected_emit_independent": 0.0, "context_loss": 0.0,
        }

    emits = sum(1 for r in rows if r.get("outcome") == "EMIT")

    tiers: Dict[str, int] = collections.Counter(
        r.get("tier") for r in rows if r.get("outcome") == "EMIT"
    )

    census: Dict[str, int] = collections.Counter()
    # Restricted to abstainers: a DROP code can fire while the molecule still
    # names via another path, so the all-molecule census over-counts sites that
    # are already survivable and would mis-rank the build order. This is the
    # census that answers "which site actually costs us breadth".
    census_abstain: Dict[str, int] = collections.Counter()
    for r in rows:
        codes = set(r.get("refusal_codes") or ())
        for code in codes:
            census[code] += 1
            if r.get("outcome") != "EMIT":
                census_abstain[code] += 1

    total_comp = sum(r.get("n_components", 0) for r in rows)
    named_comp = sum(r.get("n_components_named", 0) for r in rows)
    # Pooled, not the mean of per-molecule ratios: pooling keeps p comparable
    # across corpora whose molecules carry different component counts.
    p = (named_comp / total_comp) if total_comp else 0.0
    mean_comp = total_comp / n
    emit_rate = emits / n

    if components_measured:
        # E[p**n] over the OBSERVED component-count distribution (see docstring:
        # p**E[n] is the wrong estimator and understates by ~8pp on live data).
        projected = sum(p ** r.get("n_components", 0) for r in rows) / n
        context_loss = projected - emit_rate
    else:
        # Never report a real-looking 0.0 for something that was not measured.
        p = projected = context_loss = None

    return {
        "n": n,
        "emit_rate": emit_rate,
        "tiers": dict(tiers),
        "structure_wrong": sum(1 for r in rows if r.get("structure_wrong")),
        # T6: emitted names OPSIN could not parse at all.
        "opsin_unparseable": sum(1 for r in rows if r.get("opsin_unparseable")),
        "refusal_census": dict(census.most_common()),
        "refusal_census_abstain": dict(census_abstain.most_common()),
        "per_fragment_p": p,
        "mean_components": mean_comp,
        "projected_emit_independent": projected,
        "context_loss": context_loss,
    }
