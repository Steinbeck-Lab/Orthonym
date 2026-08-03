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
    "residual_refusal_code",
    "refusal_structure",
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

# v30 P0-T3. The two grammars above see PRODUCER refusals only, and measurement
# showed that is not where these molecules die: of the 11 abstainers that logged
# no code at all on the v30 P0 best-effort run, NINE were terminated by a
# post-hoc GATE — the name was built, then suppressed. The gate's own log line
# was being read as noise (the superseded
# ``test_parse_refusal_codes_ignores_unrelated_log_noise`` asserted exactly that),
# so the census could not see the site that actually cost the breadth. This
# matches the standing project finding that the 0-wrong margin is the GATE, not
# the producers.
#
# Only the two lines that genuinely REFUSE are matched. ``namer.py:568``
# "OPSIN grammar validation failed" is deliberately NOT here: it logs and then
# ``return name`` (``namer.py:572``), so it is a warning, not a blocker. Same for
# the stereo backstop, which is detection-only.
_GATE_RES = (
    # namer.py:951 — SELF-01 rejected the candidate as a DIFFERENT molecule.
    (re.compile(r"SELF-01 suppressed \(different molecule\)"),
     "GATE-SELF01:different_molecule"),
    # namer.py:1224 — the pre-emission OPSIN-parse validity gate.
    (re.compile(r"OPSIN validity gate suppressed unparseable name"),
     "GATE-OPSIN:unparseable"),
)


def parse_refusal_codes(log_lines: Iterable[str]) -> List[str]:
    """Distinct refusal codes for ONE molecule, in first-seen order.

    Deduplicated on purpose: a code firing five times while the engine probes
    five substituents is still a single blocker for that molecule, and the
    milestone question is 'how many molecules does this site block', not 'how
    often does it fire'. Un-deduplicated counts would overstate the dominant
    site and mis-rank the build order.

    Covers producer refusals (``DROP-*``, ``general_engine refused:``) AND the
    two post-hoc gates that suppress a finished name (``GATE-SELF01``,
    ``GATE-OPSIN``) — see ``_GATE_RES`` for why omitting the gates made 9 of 11
    uncoded abstainers unattributable.
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
        for pat, code in _GATE_RES:
            if pat.search(line):
                add(code)
    return seen


# ------------------------------------------------------ residual attribution

def residual_refusal_code(row: Dict[str, Any]) -> str | None:
    """A SPECIFIC named terminal site for an abstainer that logged no code.

    Deliberately NOT a catch-all bucket: a single "unattributed" bin would
    absorb precisely the future instrument gaps this attribution exists to
    expose, so every branch names its own mechanism and anything unrecognised
    returns ``None`` and is counted as a hard instrument gap.

    The three residual mechanisms are structurally unable to produce a log-derived
    code, which is why they are read out of stored result fields instead:

    * ``SKIP`` — RDKit could not parse the input; the namer never ran.
    * ``EXC`` — an unhandled exception. NOT a refusal: a crash the instrument's
      own ``except`` turned into an abstainer row. On the v30 P0 run this was one
      row, ``TypeError: '<' not supported between instances of 'str' and 'int'``.
    * ``TIMEOUT`` — the instrument's own per-molecule SIGALRM, not an engine
      decision.
    * ``LIMIT:<code>`` — the ``errors.LIMIT_CATALOG`` classification carried back
      in ``limit_code``. The unsupported-element classifier returns before the
      naming machinery is touched (measured: the ``[99Tc]``-labelled sorbitol row
      produced ZERO log lines of any kind), so no log stream exists to parse.

    Used ONLY where ``refusal_codes`` is empty, so it can never displace or
    inflate a producer-attributed site.
    """
    if row.get("refusal_codes"):
        return None
    outcome = row.get("outcome")
    if outcome == "EMIT":
        return None
    if outcome == "SKIP":
        return f"SKIP:{row.get('reason') or 'unknown'}"
    if outcome == "EXC":
        exc = str(row.get("exc") or "")
        return f"EXC:{exc.split(':', 1)[0] or 'unknown'}"
    if outcome == "TIMEOUT":
        return "TIMEOUT:per_molecule_alarm"
    limit = row.get("limit_code")
    if limit:
        return f"LIMIT:{limit}"
    return None


# ----------------------------------------------------- ONLY-ranked structure

def refusal_structure(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rank refusal sites by ``ONLY`` — what a ONE-SITE fix can actually convert.

    The ``touched`` census cannot size a fix and has already mis-aimed a
    milestone: per-site ``touched`` percentages sum to ~408% of abstainers
    because a molecule blocked by four sites needs all four cleared, so
    ``DROP-24`` looked like the top target at 116 touched / 95 first-refusals
    while being the SOLE blocker on exactly 1 molecule. ``pg='ester'`` is 12th by
    touched and the largest single-blocked site at 10.

    Three counts per site, all over the SAME denominator (every abstainer, never
    a filtered subset — a signature computed over a subset is how a lead gets
    promoted to a defect class):

    * ``touched`` — abstainers on which the site fired at all. NOT additive.
    * ``first``   — abstainers where it fired first. Systematically over-credits
      whichever site happens to sit earliest in the pipeline.
    * ``only``    — abstainers naming this site and NO other. The honest ceiling
      of a one-site fix.

    ``single_site_ceiling`` = ``(emitted + Σ only) / n`` and is an OPTIMISTIC
    UPPER BOUND: ``len(refusal_codes) == 1`` only bounds single-blocked-ness,
    because an early bail-out hides downstream blockers (project record:
    "total_refuse=1 does NOT mean single-blocked … 1 of 10 emitted").

    Depth-0 abstainers are reported as ``uncoded_abstainers`` and attributed
    separately via :func:`residual_refusal_code`. They are NEVER folded into
    ``only`` or the ceiling: uncoded is the opposite of known-single, and
    counting them would inflate the ceiling with rows whose blocker is unknown.
    """
    n = len(rows)
    emits = [r for r in rows if r.get("outcome") == "EMIT"]
    abst = [r for r in rows if r.get("outcome") != "EMIT"]
    n_ab = len(abst)

    # DISTINCT sites per molecule. Must use the same dedup'd basis as `only`,
    # or a row carrying ["X","X"] reports depth 2 while being counted
    # single-blocked, and multi_blocked_total stops reconciling.
    depth: Dict[int, int] = collections.Counter(
        len(set(r.get("refusal_codes") or ())) for r in abst)

    touched: Dict[str, int] = collections.Counter()
    first: Dict[str, int] = collections.Counter()
    only: Dict[str, int] = collections.Counter()
    for r in abst:
        codes = list(r.get("refusal_codes") or ())
        for code in set(codes):
            touched[code] += 1
        if codes:
            first[codes[0]] += 1
        # dedup guard: parse_refusal_codes already dedups, but a hand-built or
        # legacy row could repeat a code, and ["X","X"] is single-blocked.
        if len(set(codes)) == 1:
            only[codes[0]] += 1

    # ONLY descending. Tie-breaks are total and value-based (touched, then
    # first, then the site name) so the ordering is deterministic across runs —
    # Counter iteration order is insertion order, which follows corpus order.
    sites = sorted(
        ({"site": s,
          "touched": touched[s],
          "first": first.get(s, 0),
          "only": only.get(s, 0),
          "only_pct_of_corpus": (only.get(s, 0) / n) if n else 0.0}
         for s in touched),
        key=lambda d: (-d["only"], -d["touched"], -d["first"], d["site"]),
    )

    residual: Dict[str, int] = collections.Counter()
    residual_rows: List[Dict[str, Any]] = []
    unattributed: List[str] = []
    for r in abst:
        if r.get("refusal_codes"):
            continue
        code = residual_refusal_code(r)
        if code is None:
            unattributed.append(r.get("smiles") or "<no smiles>")
            continue
        residual[code] += 1
        residual_rows.append({"smiles": r.get("smiles"), "code": code})

    solo = sum(only.values())
    site_hits = sum(touched.values())
    return {
        "n": n,
        "n_emit": len(emits),
        "n_abstain": n_ab,
        "total_site_hits": site_hits,
        # Derived from the SAME dedup'd basis as total_site_hits, so the printed
        # "touched percentages sum to ~408%" is consistent by construction rather
        # than by the rows happening to carry no duplicate codes.
        "mean_blockers_per_abstainer": (site_hits / n_ab) if n_ab else 0.0,
        "depth_histogram": {str(k): depth[k] for k in sorted(depth)},
        # EVERY site, never a truncated head: the run JSON is the queryable
        # record, so truncation belongs to the printer, not to the data.
        "sites": sites,
        "sites_total": len(sites),
        "single_blocked_total": solo,
        "multi_blocked_total": n_ab - solo - depth.get(0, 0),
        "uncoded_abstainers": depth.get(0, 0),
        "single_site_ceiling": ((len(emits) + solo) / n) if n else 0.0,
        "residual_attribution": dict(residual.most_common()),
        "residual_rows": residual_rows,
        "unattributed_abstainers": len(unattributed),
        "unattributed_smiles": unattributed,
    }


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

    ``refusal_structure`` carries the ONLY-ranked build order (see
    :func:`refusal_structure`); the two flat ``refusal_census`` dicts are kept
    because they are ``touched`` counts and existing run records quote them.

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
            "opsin_unparseable": 0, "tautomer_differs": 0,
            "refusal_census": {}, "refusal_census_abstain": {},
            "refusal_structure": refusal_structure(rows),
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
        # Tracked APART from structure_wrong: a mobile-H tautomer difference is
        # not a wrong structure. The first v29 baseline reported 2 "wrong"
        # names that were both tautomers (benzimidazole NH, guanidine); folding
        # those into T3 would manufacture phantom 0-wrong violations.
        "tautomer_differs": sum(1 for r in rows if r.get("tautomer_differs")),
        "refusal_census": dict(census.most_common()),
        "refusal_census_abstain": dict(census_abstain.most_common()),
        # v30 P0-T2: the ONLY-ranked structure. Carried into the run JSON so the
        # build order is queryable without re-running a 675 s measurement.
        "refusal_structure": refusal_structure(rows),
        "per_fragment_p": p,
        "mean_components": mean_comp,
        "projected_emit_independent": projected,
        "context_loss": context_loss,
    }
