"""Unit tests for the scoped-per-call memo infra (M1, Levers C1 + E).

Covers the memo contract itself (off/on/verify, scope teardown, nesting) and the
COMPLETENESS of the Lever C1 ``name_substituent`` cache key -- every key component
is a proven wrong-name mechanism if omitted (PERF-OPTIMIZATION-PLAN Part 1 Lever C),
so each test pins that two structurally different calls do NOT collide.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import memo
from orthonym.assembly.memo import (
    push_scope, pop_scope, cache_or_compute, MemoMismatch,
)


@pytest.fixture
def memo_mode():
    """Give each test a clean memo baseline: no leaked scope, restored _MODE.

    Yields a setter so a test can select ``on`` / ``off`` / ``verify`` in-process
    (the module reads the ``_MODE`` global on every call, so the override is live).
    """
    saved = memo._MODE
    base_tok = memo._cache_var.set(None)  # guarantee "no scope" baseline

    def _set(mode):
        memo._MODE = mode

    try:
        yield _set
    finally:
        memo._MODE = saved
        memo._cache_var.reset(base_tok)


# --------------------------------------------------------------------------- #
# memo contract
# --------------------------------------------------------------------------- #

def test_off_mode_always_computes(memo_mode):
    memo_mode("off")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "X"

    tok = push_scope()
    try:
        assert cache_or_compute("ns", ("k",), fn) == "X"
        assert cache_or_compute("ns", ("k",), fn) == "X"
    finally:
        pop_scope(tok)
    assert calls["n"] == 2  # off never caches, even with a scope open


def test_on_mode_caches_same_key_and_separates_different_keys(memo_mode):
    memo_mode("on")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    tok = push_scope()
    try:
        v1 = cache_or_compute("ns", ("k",), fn)
        v2 = cache_or_compute("ns", ("k",), fn)      # HIT -> no compute
        v3 = cache_or_compute("ns", ("other",), fn)  # MISS -> compute
    finally:
        pop_scope(tok)
    assert v1 == v2 == 1
    assert v3 == 2
    assert calls["n"] == 2


def test_on_mode_without_scope_is_fail_open(memo_mode):
    memo_mode("on")
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "Y"

    # No push_scope: no active cache -> always compute.
    assert cache_or_compute("ns", ("k",), fn) == "Y"
    assert cache_or_compute("ns", ("k",), fn) == "Y"
    assert calls["n"] == 2


def test_verify_mode_raises_on_incomplete_key(memo_mode):
    memo_mode("verify")
    seq = iter([1, 2])

    def fn():
        return next(seq)

    tok = push_scope()
    try:
        assert cache_or_compute("ns", ("k",), fn) == 1
        with pytest.raises(MemoMismatch):
            cache_or_compute("ns", ("k",), fn)  # recompute -> 2 != stored 1
    finally:
        pop_scope(tok)


def test_verify_mode_stable_value_does_not_raise(memo_mode):
    memo_mode("verify")

    def fn():
        return "stable"

    tok = push_scope()
    try:
        assert cache_or_compute("ns", ("k",), fn) == "stable"
        assert cache_or_compute("ns", ("k",), fn) == "stable"  # no raise
    finally:
        pop_scope(tok)


def test_scope_teardown_leaves_second_cycle_empty(memo_mode):
    memo_mode("on")
    tok = push_scope()
    try:
        cache_or_compute("ns", ("k",), lambda: "A")
        assert memo._cache_var.get()  # non-empty while scope is open
    finally:
        pop_scope(tok)
    # A fresh cycle starts from an empty cache -- no cross-call staleness.
    tok2 = push_scope()
    try:
        assert memo._cache_var.get() == {}
    finally:
        pop_scope(tok2)


def test_nested_push_shares_outer_cache(memo_mode):
    memo_mode("on")
    outer = push_scope()
    assert outer is not None
    try:
        cache_or_compute("ns", ("k",), lambda: "V")
        inner = push_scope()
        assert inner is None  # nested -> no new scope created
        pop_scope(inner)      # no-op; must NOT tear down the outer cache
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "V2"

        assert cache_or_compute("ns", ("k",), fn) == "V"  # still the outer hit
        assert calls["n"] == 0
    finally:
        pop_scope(outer)


# --------------------------------------------------------------------------- #
# Lever C1 -- name_substituent cache-key completeness
# --------------------------------------------------------------------------- #

def _be():
    from orthonym.metrics.provenance import best_effort_ctx
    return best_effort_ctx.get()


def test_c1_key_distinguishes_attachment_position():
    # Same 5-carbon fragment attached at C1 vs C2 names differently
    # (pentyl vs pentan-2-yl); the keys MUST differ.
    from orthonym.assembly.substituent_enumerator import _substituent_memo_key
    mol = Chem.MolFromSmiles("CCCCC")
    frag = [0, 1, 2, 3, 4]
    k1 = _substituent_memo_key(mol, frag, 0, False, _be())
    k2 = _substituent_memo_key(mol, frag, 1, False, _be())
    assert k1 is not None and k2 is not None
    assert k1 != k2


def test_c1_key_distinguishes_free_valence_order():
    # Same single-carbon fragment, external free valence single vs double
    # (methyl -yl vs methylidene -ylidene); the keys MUST differ.
    from orthonym.assembly.substituent_enumerator import _substituent_memo_key
    mol_single = Chem.MolFromSmiles("CC")
    mol_double = Chem.MolFromSmiles("C=C")
    ks = _substituent_memo_key(mol_single, [0], 0, False, _be())
    kd = _substituent_memo_key(mol_double, [0], 0, False, _be())
    assert ks is not None and kd is not None
    assert ks != kd


def test_c1_key_distinguishes_cip_parity():
    # [C@@H] vs [C@H] fragments carry different descriptors; the keys MUST differ.
    from orthonym.assembly.substituent_enumerator import _substituent_memo_key
    mol1 = Chem.MolFromSmiles("C[C@@H](O)CC")
    mol2 = Chem.MolFromSmiles("C[C@H](O)CC")
    Chem.AssignStereochemistry(mol1, cleanIt=True, force=True)
    Chem.AssignStereochemistry(mol2, cleanIt=True, force=True)
    frag = [0, 1, 2, 3, 4]
    k1 = _substituent_memo_key(mol1, frag, 1, False, _be())
    k2 = _substituent_memo_key(mol2, frag, 1, False, _be())
    assert k1 is not None and k2 is not None
    assert k1 != k2


def test_c1_key_distinguishes_allow_mancude():
    # allow_mancude False vs True changes what Tier 4.5 may return, so the two
    # must NOT share a cache entry.
    from orthonym.assembly.substituent_enumerator import _substituent_memo_key
    mol = Chem.MolFromSmiles("C1CC2CCC1CC2")
    frag = list(range(mol.GetNumAtoms()))
    kf = _substituent_memo_key(mol, frag, 0, False, _be())
    kt = _substituent_memo_key(mol, frag, 0, True, _be())
    assert kf is not None and kt is not None
    assert kf != kt


# --------------------------------------------------------------------------- #
# Lever C1 -- wired name_substituent path (memo actually engaged)
# --------------------------------------------------------------------------- #

def test_c1_wired_no_false_collision_on_attachment(memo_mode):
    # With the memo engaged, two attachment positions on the SAME fragment must
    # still return DIFFERENT names -- a collision would return the cached first.
    memo_mode("on")
    from orthonym.assembly.substituent_enumerator import name_substituent
    mol = Chem.MolFromSmiles("CCCCC")
    tok = push_scope()
    try:
        n1 = name_substituent(mol, [0, 1, 2, 3, 4], 0)
        n2 = name_substituent(mol, [0, 1, 2, 3, 4], 1)
    finally:
        pop_scope(tok)
    assert n1 and n2 and n1 != n2


def test_c1_wired_verify_mode_stable(memo_mode):
    # Naming the same fragment repeatedly under verify mode must not raise
    # (the key is complete + the result is deterministic).
    memo_mode("verify")
    from orthonym.assembly.substituent_enumerator import name_substituent
    mol = Chem.MolFromSmiles("CCCCCCCC")
    tok = push_scope()
    try:
        for _ in range(3):
            name_substituent(mol, list(range(8)), 0)
    finally:
        pop_scope(tok)
