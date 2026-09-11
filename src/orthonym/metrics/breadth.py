""" breadth metrics — the pinned definitions behind the breadth instrument.

Why this module exists: every breadth figure quoted before came from an
ad-hoc harness in a session a temp dir, and at least one was a ~4x mirage
because the harness and the production producer disagreed about what counts as
an emission. These functions are the single definition of each metric, unit
tested in ``tests/unit/metrics/test_breadth.py``, so a number measured in one
session is comparable to a number measured weeks later.

Pure logic: no naming, no OPSIN, no Java. The driver
(``scripts/measure_breadth.py``) supplies per-molecule observations; this module
defines what they mean.
"""
from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from ..errors import is_failure_name

__all__ = [
    "ring_systems",
    "classify_outcome",
    "parse_refusal_codes",
    "parse_suppressed_candidates",
    "residual_refusal_code",
    "terminal_site",
    "terminal_stage",
    "terminal_basis_available",
    "row_attribution",
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
    """``EMIT`` or ``ABSTAIN`` for one ``name_tiered`` result.

    Uses the production failure predicate (``errors.is_failure_name``) rather
    than a local truthiness check, because the PIN tier signals abstention with
    a DESCRIPTIVE string ('unknown organic compound') while the best-effort tier
    signals it with ``None``. Counting the descriptive string as an emission is
    precisely how a breadth number becomes a mirage.
    """
    return "ABSTAIN" if is_failure_name(result.get("name")) else "EMIT"


# ----------------------------------------------------------- refusal parsing

# The reason= field is OPTIONAL: real call sites log
# substituent_all_candidates_filtered and suppress_duplicate_bare_amino_prefix
# with no reason at all. Requiring it would erase those sites from the census,
# making a genuinely blocking site rank as zero because of its log format
# rather than its behaviour.
#
# The leading token is a compiled alternation of every canonical producer
# refusal slug (formerly the cryptic numbered producer-refusal codes) actually
# emitted on the naming path, so a slug that is not one of these is never
# mistaken for a refusal code. Longest-first ordering is not required: every
# slug is a
# complete, disjoint identifier (no slug is a prefix of another), and the
# trailing ``\b`` guards against a partial-word match regardless.
_DROP_CODES = (
    "substituent_is_bare_functional_group",
    "substituent_ring_atom_overlap",
    "principal_group_branch_overlap",
    "ring_substituent_bare_functional_group",
    "universal_pipeline_unnameable_substituent",
    "substituent_smiles_extraction_failure",
    "substituent_recursion_depth_exceeded",
    "substituent_no_prefix_form",
    "substituent_all_candidates_filtered",
    "polyfunctional_producer_returned_none",
    "ring_fragment_declined_by_ring_engine",
    "amine_n_substituent_unnameable",
    "charged_fragment_not_directly_nameable",
    "suppress_duplicate_bare_amino_prefix",
    "ring_substituent_fragment_unnameable",
    "substituent_extraction_exception",
    "n_branch_ring_substituent_unnameable",
    "c_branch_ring_substituent_unnameable",
    "ring_substituent_recursive_naming_fallback",
)
_DROP_RE = re.compile(
    r"\b(" + "|".join(_DROP_CODES) + r")\b(?:[^|]*?\breason=([A-Za-z0-9_]+))?"
)
# [^|] guard mirrors _DROP_RE so a capture can never run past its own log record.
_REFUSE_RE = re.compile(r"general_engine_declined:\s*([^|]+?)\s*(?:\(tier|$)")

# -T3. The two grammars above see PRODUCER refusals only, and measurement
# showed that is not where these molecules die: of the 11 abstainers that logged
# no code at all on the best-effort run, NINE were terminated by a
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
    # namer.py:951 — rejected the candidate as a DIFFERENT molecule.
    (re.compile(r"self_consistency rejected \(different molecule\)"),
     "self_consistency_rejected:different_molecule"),
    # namer.py:1224 — the pre-emission OPSIN-parse validity gate.
    (re.compile(r"OPSIN validity gate suppressed unparseable name"),
     "opsin_unparseable:"),
)


# The line's payload. ``_GATE_RES`` above matches the same line to
# ATTRIBUTE the abstention but has no capture groups, so the built candidate and
# the constitution OPSIN read it as were both discarded -- which is why the 162
# ``GATE_SUPPRESSED:self01_mismatch`` rows could be counted but never split by
# mechanism.
#
# The name group is greedy and terminated by "' (opsin=", NOT ``[^']+``: IUPAC
# names routinely contain primes (2,2'-bi-3,1,5-benzoxadiarsepine), and a
# first-apostrophe match would report a truncated string as the suppressed
# candidate. ``re.search`` rather than ``match`` so a logger prefix
# ("WARNING:orthonym.namer:") does not defeat it.
_SELF01_PAYLOAD_RE = re.compile(
    r"self_consistency rejected \(different molecule\): '(.*)' \(opsin=(.*)\)\s*$"
)


def parse_suppressed_candidates(log_lines: Iterable[str]) -> Dict[str, Any]:
    """The candidate(s) rejected for ONE molecule, from its log lines.

    Returns ``{}`` when nothing was suppressed, so a caller can ``row.update``
    it without introducing null keys onto rows the gate never touched.

    The LAST suppression is reported as *the* suppressed candidate because the
    retry cascade (``namer.py:2323-2354``) re-submits alternatives and each is
    re-gated, so the final rejection is the one that ended the molecule. All of
    them are kept under ``suppressed_all`` as well: "one bad candidate" and "the
    producer kept offering variants of the same wrong structure" need different
    fixes, and only the full list distinguishes them.

    Scope: this reads the line ONLY. The OPSIN validity gate suppresses
    names too, but for a different reason (unparseable, not wrong molecule), and
    conflating the two would size a grammar defect as a constitution defect.

    ⚠ The keys are ``self01_``-prefixed ON PURPOSE, and the prefix is the guard.
    A molecule can be suppressed by mid-cascade and then terminate at a
    DIFFERENT gate, so a row whose terminal cause is ``opsin_unparseable`` can
    still carry a payload from earlier in its own cascade. Under the
    earlier generic names (``suppressed_name``) that read as "the candidate this
    row died on", and an investigator consuming it had to monkeypatch
    ``record_suppression`` to recover the real one. Measured on the 500-row
    census: 39 of 200 payload-carrying rows terminated somewhere other than
    ``self01_mismatch`` (6 ``opsin_unparseable``, 5 ``enumerator_last_resort``,
    1 ``enumerator_ring_fallback``, 27 with no terminal cause recorded).
    **Only trust this payload when ``terminal_detail == 'self01_mismatch'``.**
    """
    found: List[Tuple[str, str]] = []
    for line in log_lines:
        m = _SELF01_PAYLOAD_RE.search(line)
        if m:
            found.append((m.group(1), m.group(2)))
    if not found:
        return {}
    return {
        "self01_suppressed_name": found[-1][0],
        "self01_suppressed_opsin_smiles": found[-1][1],
        "self01_suppressed_count": len(found),
        "self01_suppressed_all": found,
    }


def parse_refusal_codes(log_lines: Iterable[str]) -> List[str]:
    """Distinct refusal codes for ONE molecule, in first-seen order.

    Deduplicated on purpose: a code firing five times while the engine probes
    five substituents is still a single blocker for that molecule, and the
    milestone question is 'how many molecules does this site block', not 'how
    often does it fire'. Un-deduplicated counts would overstate the dominant
    site and mis-rank the build order.

    Covers producer refusals (the canonical refusal slugs, ``general_engine_declined:``)
    AND the two post-hoc gates that suppress a finished name (the self-consistency
    rejection, ``opsin_unparseable:``) — see ``_GATE_RES`` for why omitting the
    gates made 9 of 11 uncoded abstainers unattributable.
    """
    seen: List[str] = []

    def add(code: str) -> None:
        if code not in seen:
            seen.append(code)

    for line in log_lines:
        for m in _DROP_RE.finditer(line):
            add(f"{m.group(1)}:{m.group(2) or '<no_reason>'}")
        for m in _REFUSE_RE.finditer(line):
            add(f"producer_refused:{m.group(1).strip()}")
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
      own ``except`` turned into an abstainer row. On the run this was one
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


# -------------------------------------------------- TERMINAL attribution (T4)

#: The two attribution bases, and WHY there are two rather than one.
#:
#: ``log`` — every producer-refusal-slug / ``producer_refused:*`` / gate
#: code scraped from the engine's own log stream for that
#: molecule.
#: ``terminal`` — the ONE mechanism the typed abstention channel
#: (``metrics.abstention``) recorded as having ended the naming.
_BASES = ("log", "terminal")

#: Prefix for a site attributed by the typed abstention channel.
_TERM_PREFIX = "TERM:"
#: Prefix for a site attributed from the LOG because the channel RAN and
#: recorded nothing specific. Deliberately a DIFFERENT prefix: it is a weaker
#: inference and a reader must be able to tell the two apart at a glance.
_TERMGAP_PREFIX = "TERMGAP:"
#: Prefix for a site attributed from the LOG because the channel was never read
#: for that row at all (a pre- run, or a crash/timeout). A THIRD prefix, not a
#: reuse of TERMGAP:, because "the instrument did not look" and "the engine did
#: not record" are different findings and only the second is an engine gap.
_TERMUNMEASURED_PREFIX = "TERMGAP-UNMEASURED:"


def terminal_site(row: Dict[str, Any]) -> str | None:
    """The single TERMINAL site for one abstaining ``row``, or ``None``.

    Why a terminal basis exists at all — and why it is not merely "the log
    codes, deduplicated". Producer refusal-slug / ``producer_refused:*`` codes
    are **EXPLORATORY**: they fire while the engine searches candidates (five
    substituent tiers, a ring engine, a chain engine) and are frequently NOT
    the mechanism that ended the molecule. Measured instance, the class this
    function exists for::

        [I-](CCO)c1ccccc1
          log codes: substituent_is_bare_functional_group:fg_only,
                      ring_fragment_declined_by_ring_engine:ring_fragment_declined…,
                      n_branch_ring_substituent_unnameable,
                      producer_refused:branch unnameable
          terminal: GATE_SUPPRESSED / charge_dropped (namer.py:3916)

    A name WAS built for that molecule; a structure-conservation veto removed
    it. On the log basis four innocent sites collect ``first``/``ONLY`` credit
    they did not earn, and ``ONLY`` is what the build order is ranked by — so
    the log basis aims a milestone at four sites whose fix would convert
    nothing. Hence: ``ONLY`` and ``first`` are computed on TERMINAL
    attribution; ``touched`` stays the UNION of both bases, because the
    exploratory codes did fire and the multi-blocking depth histogram (the
    evidence that the census is non-additive) is built from that union.

    Three honest caveats, stated because they bound what the terminal basis
    can be used for:

    1. "Terminal" means *the channel's single attribution*, not
       *chronologically last*. ``metrics.abstention`` is first-writer-wins
       among generation-stage codes and among post-generation codes, with one
       documented override: a post-generation site holding a REAL (not
       failure-marked) candidate overrides a speculative generation-stage
       record, because the candidate's existence proves generation completed.
    2. It is coarse — five ``AbstentionCode`` values. ``detail`` is therefore
       part of the site key, which splits ``GATE_SUPPRESSED`` into the six
       structure-conservation vetoes plus and the OPSIN validity gate.
    3. Not every post-generation termination is instrumented. The general
       engine's inline E1-certificate rejection (``namer.py:3586``) and its
       no-jar / dropped-stereo discards (``namer.py:3634``) throw a generated
       candidate away without recording, leaving the channel at ``OTHER`` with
       no detail. Those rows get a ``TERMGAP:`` site naming the best available
       log code — never folded into a silent catch-all.

    Returns ``None`` when the row carries no channel observation at all (an
    EMIT, a run recorded before the channel was wired, or a crash/timeout in
    which the namer never returned). Those rows are attributed by the existing
    :func:`residual_refusal_code` machinery instead.
    """
    if row.get("outcome") == "EMIT":
        return None
    code = row.get("terminal_code")
    if not code:
        return None
    detail = row.get("terminal_detail")
    if code == "OTHER" and not detail:
        # The channel ran and recorded NOTHING specific: an uninstrumented
        # post-generation termination (caveat 3). Name the gap with the log's
        # own first code rather than inventing a bucket; a bare
        # 'TERM:OTHER' bin would absorb exactly the instrument gaps this
        # attribution exists to expose.
        codes = list(row.get("refusal_codes") or ())
        return f"{_TERMGAP_PREFIX}{codes[0]}" if codes else None
    return f"{_TERM_PREFIX}{code}:{detail or 'no_detail'}"


#: Site prefix -> the operational bucket the abstention lands in. The names are
#: taken VERBATIM from ``scripts/abstention_census.py::_stage`` so the two
#: censuses cannot drift into two vocabularies for the same axis:
#:
#: * ``needs_engine`` — the pipeline produced NO candidate. Only new naming
#: CAPABILITY recovers these.
#: * ``suppressed`` — a candidate WAS built and a gate/downgrade removed it.
#: * ``other`` — the channel fired but named no specific mechanism.
#: * ``uninstrumented`` — no channel record; the site came from the log stream.
#:
#: ⚠ ``suppressed`` is NOT the same as "recoverable by re-emitting". A
#: ``self01_mismatch`` row means the built name denoted a DIFFERENT molecule, so
#: its lever is an upstream CORRECTNESS fix, never a looser gate. Reading
#: ``suppressed`` as free breadth is how a milestone gets aimed at the gate that
#: is the only thing holding 0-wrong.
_STAGE_BY_CODE = {
    "NO_PARENT": "needs_engine",
    "BRANCH_UNNAMEABLE": "needs_engine",
    "GATE_SUPPRESSED": "suppressed",
    "COVERAGE_DOWNGRADE": "suppressed",
    "OTHER": "other",
}


def terminal_stage(site: str) -> str:
    """The operational bucket for a terminal ``site`` label."""
    if site.startswith(_TERM_PREFIX):
        return _STAGE_BY_CODE.get(site[len(_TERM_PREFIX):].split(":", 1)[0],
                                  "other")
    if site.startswith((_TERMGAP_PREFIX, _TERMUNMEASURED_PREFIX)):
        return "uninstrumented"
    return "other"


def terminal_basis_available(rows: Sequence[Dict[str, Any]]) -> bool:
    """True when ``rows`` were measured with the abstention channel wired.

    Checked explicitly rather than inferred from "are there any terminal
    codes": a run recorded before carries none, and censusing it on the
    terminal basis would print a clean-looking all-residual zero — the
    project's standing "a PERFECT harness result means it did not RUN"
    failure mode. ``run_worker`` stamps ``terminal_measured`` on every row it
    processes, so its ABSENCE is the signal.
    """
    return any("terminal_measured" in r for r in rows)


def row_attribution(row: Dict[str, Any],
                    basis: str = "log") -> tuple[List[str], List[str]]:
    """``(union, attribution)`` site lists for one row under ``basis``.

    ``union`` feeds ``touched`` and the depth histogram; ``attribution`` feeds
    ``first`` (its head) and ``ONLY`` (when it names exactly one site). On the
    ``log`` basis the two are the same list, which is why the log basis
    credits ``ONLY`` to whichever exploratory code happened to fire alone.
    """
    if basis not in _BASES:
        raise ValueError(f"basis must be one of {_BASES}, got {basis!r}")
    log_codes = list(row.get("refusal_codes") or ())
    if basis == "log":
        return log_codes, log_codes
    if row.get("outcome") == "EMIT":
        # An emission has no terminal mechanism by definition. Guarded here as
        # well as in terminal_site so a direct caller cannot get an emitted row
        # labelled as an unmeasured gap.
        return log_codes, []
    term = terminal_site(row)
    if term is None and log_codes:
        # No channel observation at all, but the log named something. Attribute
        # to the log's first code under a DISTINCT prefix so it can never be
        # mistaken for a channel attribution: this is the pre- shape (a run
        # recorded before the channel was wired) and the crash/timeout shape.
        term = f"{_TERMUNMEASURED_PREFIX}{log_codes[0]}"
    if term is None:
        # Nothing from either source: claim NO attribution and let the residual
        # machinery name the mechanism (SKIP / EXC / TIMEOUT / LIMIT).
        return log_codes, []
    union = log_codes + ([term] if term not in log_codes else [])
    return union, [term]


# ----------------------------------------------------- ONLY-ranked structure

def refusal_structure(rows: Sequence[Dict[str, Any]],
                      basis: str = "log") -> Dict[str, Any]:
    """Rank refusal sites by ``ONLY`` — what a ONE-SITE fix can actually convert.

    The ``touched`` census cannot size a fix and has already mis-aimed a
    milestone: per-site ``touched`` percentages sum to ~408% of abstainers
    because a molecule blocked by four sites needs all four cleared, so
    ``ring_fragment_declined_by_ring_engine`` looked like the top target at 116
    touched / 95 first-refusals while being the SOLE blocker on exactly 1
    molecule. ``pg='ester'`` is 12th by touched and the largest single-blocked
    site at 10.

    Three counts per site, all over the SAME denominator (every abstainer, never
    a filtered subset — a signature computed over a subset is how a lead gets
    promoted to a defect class):

    * ``touched`` — abstainers on which the site fired at all. NOT additive.
    * ``first`` — abstainers where it fired first. Systematically over-credits
      whichever site happens to sit earliest in the pipeline.
    * ``only`` — abstainers naming this site and NO other. The honest ceiling
      of a one-site fix.

    ``single_site_ceiling`` = ``(emitted + Σ only) / n`` and is an OPTIMISTIC
    UPPER BOUND: ``len(refusal_codes) == 1`` only bounds single-blocked-ness,
    because an early bail-out hides downstream blockers (project record:
    "total_refuse=1 does NOT mean single-blocked … 1 of 10 emitted").

    Depth-0 abstainers are reported as ``uncoded_abstainers`` and attributed
    separately via:func:`residual_refusal_code`. They are NEVER folded into
    ``only`` or the ceiling: uncoded is the opposite of known-single, and
    counting them would inflate the ceiling with rows whose blocker is unknown.

    ``basis`` (-T4) selects the ATTRIBUTION source; see
    :func:`row_attribution` and:func:`terminal_site` for why there are two.

    * ``"log"`` (default, unchanged) — every code the engine logged. ``first``
      and ``ONLY`` therefore go to whichever EXPLORATORY producer code fired
      alone, even when a post-hoc gate is what actually killed the molecule.
    * ``"terminal"`` — ``touched`` stays the UNION (log ∪ terminal) so the
      depth histogram keeps its multi-blocking evidence, but ``first`` and
      ``ONLY`` are credited to the single terminal mechanism.

    ⚠ ``single_site_ceiling`` is VACUOUS on the terminal basis and the returned
    ``ceiling_is_vacuous`` flag says so. Terminal attribution names exactly one
    site per row by construction, so every attributed abstainer is
    "single-blocked" and the ceiling collapses to
    ``(emitted + attributed abstainers) / n`` — a property of the attribution,
    not a finding about the engine. Use the log-basis ceiling for the
    multi-blocking bound and the terminal-basis per-site ``ONLY`` for the build
    order; that split is the whole point of carrying both.
    """
    n = len(rows)
    emits = [r for r in rows if r.get("outcome") == "EMIT"]
    abst = [r for r in rows if r.get("outcome") != "EMIT"]
    n_ab = len(abst)

    # Parallel to `abst` by INDEX, not keyed by id: a caller may legitimately
    # pass the same row dict twice, and an id key would silently merge them.
    attributions = [row_attribution(r, basis) for r in abst]

    # DISTINCT sites per molecule, over the UNION. Must use the same dedup'd
    # basis as total_site_hits, or a row carrying ["X","X"] reports depth 2
    # while being counted single-blocked, and multi_blocked_total stops
    # reconciling.
    depth: Dict[int, int] = collections.Counter(
        len(set(union)) for union, _ in attributions)

    touched: Dict[str, int] = collections.Counter()
    first: Dict[str, int] = collections.Counter()
    only: Dict[str, int] = collections.Counter()
    for union, attribution in attributions:
        for code in set(union):
            touched[code] += 1
        if attribution:
            first[attribution[0]] += 1
        # dedup guard: parse_refusal_codes already dedups, but a hand-built or
        # legacy row could repeat a code, and ["X","X"] is single-blocked.
        if len(set(attribution)) == 1:
            only[attribution[0]] += 1

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
    for r, (_union, attribution) in zip(abst, attributions):
        # Keyed on "this basis produced NO attribution", not on "refusal_codes
        # is empty": on the terminal basis a row can carry log codes and still
        # have nothing to attribute, and such a row must not vanish.
        if attribution:
            continue
        code = residual_refusal_code(r)
        if code is None:
            unattributed.append(r.get("smiles") or "<no smiles>")
            continue
        residual[code] += 1
        residual_rows.append({"smiles": r.get("smiles"), "code": code})

    solo = sum(only.values())
    site_hits = sum(touched.values())
    n_unattributed_rows = sum(1 for _u, a in attributions if not a)
    return {
        "basis": basis,
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
        # Subtracts rows THIS basis could not attribute, which on the log basis
        # is exactly depth-0 (identical numbers) but on the terminal basis also
        # covers a row that logged codes yet carries no channel observation.
        "multi_blocked_total": n_ab - solo - n_unattributed_rows,
        "uncoded_abstainers": depth.get(0, 0),
        "no_attribution_rows": n_unattributed_rows,
        "single_site_ceiling": ((len(emits) + solo) / n) if n else 0.0,
        # ⚠ see the docstring: on the terminal basis every attributed abstainer
        # is single-blocked BY CONSTRUCTION, so the ceiling is a restatement of
        # the attribution rate and must never be quoted as a build-order bound.
        "ceiling_is_vacuous": basis == "terminal",
        # Molecules per operational bucket, on the terminal basis only. This is
        # the number P1 must be sized from: `needs_engine` is the only bucket a
        # new naming capability converts.
        "terminal_stage_rollup": (
            dict(collections.Counter(
                terminal_stage(a[0]) for _u, a in attributions if a
            ).most_common()) if basis == "terminal" else None),
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
    # (unlike implicit H), so on a deuterated input GetAtoms and
    # GetNumHeavyAtoms disagree; without this filter the partition silently
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
            "refusal_structure_terminal": None,
            "terminal_basis_available": False,
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
        # not a wrong structure. The first baseline reported 2 "wrong"
        # names that were both tautomers (benzimidazole NH, guanidine); folding
        # those into would manufacture phantom 0-wrong violations.
        "tautomer_differs": sum(1 for r in rows if r.get("tautomer_differs")),
        "refusal_census": dict(census.most_common()),
        "refusal_census_abstain": dict(census_abstain.most_common()),
        # -T2: the ONLY-ranked structure. Carried into the run JSON so the
        # build order is queryable without re-running a 675 s measurement.
        "refusal_structure": refusal_structure(rows),
        # -T4: the same structure on TERMINAL attribution. `None`, not an
        # empty-looking structure, when the rows were measured before the
        # abstention channel was wired -- a zero here would be indistinguishable
        # from "the channel found nothing".
        "refusal_structure_terminal": (
            refusal_structure(rows, basis="terminal")
            if terminal_basis_available(rows) else None),
        "terminal_basis_available": terminal_basis_available(rows),
        "per_fragment_p": p,
        "mean_components": mean_comp,
        "projected_emit_independent": projected,
        "context_loss": context_loss,
    }
