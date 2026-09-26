"""Scoped-per-call memoization infrastructure (M1, Levers C1 + E).

A byte-identity-preserving cache whose lifetime is exactly ONE top-level naming
call. The engine re-names the same substituent fragments many times inside one
molecule -- the dispatch cascade, the ``_retry_cascade_on_gate_rejection``
re-entries and the recursive Tier-4 cascade all re-derive the same fragment
prefixes -- so a per-call cache removes that redundant work without changing any
emitted name. See `internal notes` Part 2.

Design invariants (each is load-bearing for the 0-wrong / byte-identity contract):

* **Scope is per top-level call.** A ``contextvars.ContextVar`` holds the cache
  dict; it is created at the outermost naming entry and torn down on exit, so
  cross-molecule staleness, tier drift and env-flag changes are structurally
  impossible and memory stays bounded.
* **Fail-open, never fail-wrong.** With no active scope (or ``ORTHONYM_MEMO=off``)
  every call recomputes. A cache MISS can never corrupt output; only a false HIT
  could, and that is exactly what a COMPLETE key and ``verify`` mode prevent.
* **``verify`` mode is the continuous completeness check.** It always recomputes
  and raises:class:`MemoMismatch` the instant a stored value disagrees with a
  fresh one for the same key -- turning an incomplete key into a loud failure
  rather than a silent wrong name.

Dependency-light on purpose (``contextvars`` + ``os`` only).
"""
import contextvars
import os
from collections import OrderedDict

#: verify-mode mismatch log (process-global, across scopes). A ``MemoMismatch`` is
#: RAISED as a loud failure, but the recursive naming cascade wraps the hot loops in
#: broad ``except Exception`` and CATCHES it (degrading to a von-Baeyer name), so the
#: raise alone is invisible from a full-engine ``verify`` run -- "0 MemoMismatch
#: exceptions" is NOT "0 incomplete-key events" (a review P3 Crit-2). This list is
#: appended BEFORE the raise, so a validation harness can COUNT incomplete-key events
#: over a whole corpus even when every raise is swallowed. Read/reset via the helpers.
_VERIFY_MISMATCHES = []


def verify_mismatch_count():
    """Number of verify-mode same-key-different-result events since the last reset."""
    return len(_VERIFY_MISMATCHES)


def reset_verify_mismatches():
    """Clear the verify-mode mismatch log (call before a validation corpus run)."""
    _VERIFY_MISMATCHES.clear()

# Read the mode ONCE at import. The A/B / gate harnesses set ORTHONYM_MEMO in the
# subprocess environment before the engine imports, so an import-time read is
# correct for them. Unit tests that must exercise all three modes in one process
# override the module global ``_MODE`` directly (the functions below read the
# global on every call, so a monkeypatch takes effect immediately).
_MODE = os.environ.get("ORTHONYM_MEMO", "on").strip().lower()
if _MODE not in ("on", "off", "verify"):
    _MODE = "on"

# The per-call cache dict, keyed by ``(namespace, key)``. ``None`` means "no scope
# is open" -> fail-open (recompute, never cache).
_cache_var = contextvars.ContextVar("orthonym_memo_cache", default=None)


class MemoMismatch(Exception):
    """Raised in ``verify`` mode when a cached result disagrees with a fresh
    recompute for the same key -- i.e. the cache key is INCOMPLETE and a hit
    would have shipped a wrong name. Carries the offending namespace/key and both
    values so the missing key component can be identified."""

    def __init__(self, namespace, key, stored, fresh):
        self.namespace = namespace
        self.key = key
        self.stored = stored
        self.fresh = fresh
        super().__init__(
            f"MEMO VERIFY MISMATCH in namespace {namespace!r}: a cached result "
            f"differs from a fresh recompute for the SAME key -- the key is "
            f"incomplete and a cache hit would ship a wrong name.\n"
            f"  key={key!r}\n  stored={stored!r}\n  fresh={fresh!r}"
        )


def push_scope():
    """Open a memo scope for this (and nested) calls, unless one is already open.

    Returns a ContextVar token when THIS call created the scope (the caller owns
    teardown and must pass the token to:func:`pop_scope`), or ``None`` when a
    scope already existed (a nested re-entry -- it shares the outer cache and must
    NOT reset it).
    """
    if _cache_var.get() is not None:
        return None
    return _cache_var.set({})


def pop_scope(token):
    """Tear down the scope created by the matching:func:`push_scope`. A ``None``
    token (nested re-entry) is a no-op, so only the outermost frame tears down."""
    if token is not None:
        _cache_var.reset(token)


def push_sandbox():
    """Open a throwaway COPY of the current scope for a speculative computation.

    Reads still hit what the enclosing scope already holds; writes land in the
    copy and are discarded by:func:`pop_sandbox`, so the enclosing naming can
    never be served a value the speculation computed in its own context. With no
    scope open, nothing changes (no caching, as before). Always returns a token
    for:func:`pop_sandbox`.
    """
    cur = _cache_var.get()
    return _cache_var.set(dict(cur) if cur is not None else None)


def pop_sandbox(token):
    """Discard the sandbox opened by the matching:func:`push_sandbox`."""
    _cache_var.reset(token)


# Scope-bound SIDE store (a performance pass a lever verify-honesty). The nested replay-memo
# needs to keep per-key metadata (the provenance vars a call touched + the budget
# units it charged) that is internal notes-RELATIVE, so it must NOT go through
# ``cache_or_compute`` (``verify`` mode would recompute and flag it as a mismatch
# even though the emitted NAME is identical). It lives here instead: a plain dict
# kept inside the scope cache under a reserved non-tuple key, so it is torn down
# with the scope and is never compared by ``verify`` (which only ever inspects the
# ``(namespace, key)`` tuple keys it was called with). Absent a scope, there is no
# side store and no cache hit ever occurs, so a replay is never needed.
_SIDE_KEY = "__nested_memo_side__"


def side_get(namespace, key):
    """Return the scope-bound side value for ``(namespace, key)``, or ``None``."""
    cache = _cache_var.get()
    if cache is None:
        return None
    side = cache.get(_SIDE_KEY)
    return None if side is None else side.get((namespace, key))


def side_put(namespace, key, value) -> None:
    """Store a scope-bound side value for ``(namespace, key)`` (no verify compare).
    A no-op with no active scope (fail-open: no scope -> no hit -> no replay)."""
    cache = _cache_var.get()
    if cache is None:
        return
    side = cache.get(_SIDE_KEY)
    if side is None:
        side = {}
        cache[_SIDE_KEY] = side
    side[(namespace, key)] = value


# ---------------------------------------------------------------------------
# Perf lever A8 (2026-09-13): process-wide cache for the PURE namespaces
# ---------------------------------------------------------------------------
# The scope cache above dies with each top-level ``name`` call, so work that is a
# pure function of a STRUCTURE or a NAME is redone for every molecule. Measured over
# 300 fixed-seed molecules: of the values computed inside a scope, 34 % of
# ``fg_detect``, 32 % of ``sugar_c_substituted`` and 26 % of the in-process OPSIN
# parses had already been computed for an earlier molecule.
#
# ONLY namespaces whose key is complete may use this. ``name_substituent`` and
# ``fused_core`` are NOT promoted here. Their keys USED to carry no tier/breadth flags
# (2026-09-12 a lever finding, one ChEBI row flipped pin_unverified -> pin_verified);
# a performance pass a lever added ``general_fallback`` / ``allow_aromatic_general`` /
# ``full_coverage`` (and ``best_effort``) to both keys, so the completeness bar is now
# met — but promoting them to this cross-molecule cache is a separate, unmeasured
# optimization left for later; they stay scope-only for now.
# In ``verify`` mode nothing is served from here; the value is recomputed and compared
# exactly as in:func:`cache_or_compute`.
_PROCESS_MAX = int(os.environ.get("ORTHONYM_PROCESS_CACHE", "20000") or 0)
_process_cache: "OrderedDict[tuple, object]" = OrderedDict()


def process_cache_stats() -> dict:
    """``{"size", "maxsize", "hits", "misses"}`` for the process-wide pure cache."""
    return {"size": len(_process_cache), "maxsize": _PROCESS_MAX,
            "hits": _PROCESS_HITS[0], "misses": _PROCESS_MISSES[0]}


_PROCESS_HITS = [0]
_PROCESS_MISSES = [0]


def clear_process_cache() -> None:
    """Drop every process-wide entry (tests; a worker that changes engine flags)."""
    _process_cache.clear()
    _PROCESS_HITS[0] = _PROCESS_MISSES[0] = 0


def pure_cache_or_compute(namespace, key, compute_fn):
    """Memoise a value that is a pure function of ``key`` for the LIFE OF THE PROCESS.

    Same contract as:func:`cache_or_compute` except that the entry outlives the naming
    scope. Bounded LRU (``ORTHONYM_PROCESS_CACHE`` entries, default 20,000; ``0``
    disables). ``ORTHONYM_MEMO=off``/``verify`` bypass it exactly as they bypass the
    scope cache. The caller guarantees the key determines the value.
    """
    if _MODE != "on" or _PROCESS_MAX <= 0:
        return cache_or_compute(namespace, key, compute_fn)
    ck = (namespace, key)
    try:
        val = _process_cache[ck]
    except KeyError:
        pass
    except TypeError:  # unhashable key -> scope semantics
        return cache_or_compute(namespace, key, compute_fn)
    else:
        _process_cache.move_to_end(ck)
        _PROCESS_HITS[0] += 1
        return val
    val = cache_or_compute(namespace, key, compute_fn)
    _PROCESS_MISSES[0] += 1
    try:
        _process_cache[ck] = val
        if len(_process_cache) > _PROCESS_MAX:
            _process_cache.popitem(last=False)
    except TypeError:
        pass
    return val


def cache_or_compute(namespace, key, compute_fn):
    """Return the memoized value for ``(namespace, key)``, computing it via
    ``compute_fn`` on a miss.

    * ``ORTHONYM_MEMO=off`` OR no active scope -> always ``compute_fn``
      (fail-open; never caches).
    * ``on`` -> return the cached value if present, else compute + store + return.
    * ``verify`` -> ALWAYS recompute; if a value is already stored for the key and
      differs, raise:class:`MemoMismatch`; then store + return the fresh value.
    """
    if _MODE == "off":
        return compute_fn()
    cache = _cache_var.get()
    if cache is None:
        return compute_fn()
    ck = (namespace, key)
    if _MODE == "verify":
        val = compute_fn()
        if ck in cache and cache[ck] != val:
            # Record BEFORE raising: the naming cascade swallows the exception, so
            # this counter is the only observable signal of an incomplete key over a
            # full-engine corpus run (a review P3 Crit-2).
            _VERIFY_MISMATCHES.append((namespace, key))
            raise MemoMismatch(namespace, key, cache[ck], val)
        cache[ck] = val
        return val
    # mode == "on"
    if ck in cache:
        return cache[ck]
    val = compute_fn()
    cache[ck] = val
    return val
