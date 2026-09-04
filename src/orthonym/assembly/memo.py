"""Scoped-per-call memoization infrastructure.

A byte-identity-preserving cache whose lifetime is exactly ONE top-level naming
call. The engine re-names the same substituent fragments many times inside one
molecule -- the dispatch cascade, the ``_retry_cascade_on_gate_rejection``
re-entries and the recursive Tier-4 cascade all re-derive the same fragment
prefixes -- so a per-call cache removes that redundant work without changing any
emitted name. See Part 2.

Design invariants (each is load-bearing for the 0-wrong / byte-identity contract):

* **Scope is per top-level call.** A ``contextvars.ContextVar`` holds the cache
  dict; it is created at the outermost naming entry and torn down on exit, so
  cross-molecule staleness, tier drift and env-flag changes are structurally
  impossible and memory stays bounded.
* **Fail-open, never fail-wrong.** With no active scope (or ``ORTHONYM_MEMO=off``)
  every call recomputes. A cache MISS can never corrupt output; only a false HIT
  could, and that is exactly what a COMPLETE key and ``verify`` mode prevent.
* **``verify`` mode is the continuous completeness check.** It always recomputes
  and raises :class:`MemoMismatch` the instant a stored value disagrees with a
  fresh one for the same key -- turning an incomplete key into a loud failure
  rather than a silent wrong name.

Dependency-light on purpose (``contextvars`` + ``os`` only).
"""
import contextvars
import os

#: verify-mode mismatch log (process-global, across scopes). A ``MemoMismatch`` is
#: RAISED as a loud failure, but the recursive naming cascade wraps the hot loops in
#: broad ``except Exception`` and CATCHES it (degrading to a von-Baeyer name), so the
#: raise alone is invisible from a full-engine ``verify`` run -- "0 MemoMismatch
#: exceptions" is NOT "0 incomplete-key events". This list is
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
    teardown and must pass the token to :func:`pop_scope`), or ``None`` when a
    scope already existed (a nested re-entry -- it shares the outer cache and must
    NOT reset it).
    """
    if _cache_var.get() is not None:
        return None
    return _cache_var.set({})


def pop_scope(token):
    """Tear down the scope created by the matching :func:`push_scope`. A ``None``
    token (nested re-entry) is a no-op, so only the outermost frame tears down."""
    if token is not None:
        _cache_var.reset(token)


def cache_or_compute(namespace, key, compute_fn):
    """Return the memoized value for ``(namespace, key)``, computing it via
    ``compute_fn`` on a miss.

    * ``ORTHONYM_MEMO=off`` OR no active scope -> always ``compute_fn()``
      (fail-open; never caches).
    * ``on`` -> return the cached value if present, else compute + store + return.
    * ``verify`` -> ALWAYS recompute; if a value is already stored for the key and
      differs, raise :class:`MemoMismatch`; then store + return the fresh value.
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
            # full-engine corpus run.
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
