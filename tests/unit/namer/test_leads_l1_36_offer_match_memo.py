"""Leads item 36: the process-wide memo of the full-InChIKey offer compare keeps no verdict
that is not a function of its key.

``namer._full_inchikey_offer_match`` is memoised for the life of the process through
``assembly.memo.pure_cache_or_compute``, whose contract is "the caller guarantees the key
determines the value". Its compute function returned the fail-open ``True`` from an
``except Exception`` branch, and the memo stored that verdict, so a transient failure (a
MemoryError, a protonation-check exception) would have been replayed for the life of the
process. The compute function now raises ``_InconclusiveCompare`` (the pattern of
``validation.opsin_roundtrip.extended_smiles_or_unavailable``: "raised, not returned: the memo
stores nothing when its compute function raises") and the wrapper gives the fail-open ``True``
outside the memo, un-memoised. No nomenclature rule is involved.
"""
import pytest

from orthonym import namer
from orthonym.assembly import memo

NAME = "leads-l1-item-36 probe"       # a name no other test uses, so the process memo is fresh
SAME = ("OCC", "CCO")                  # one molecule, two spellings: the compare reaches the
                                       # protonation check
OTHER = ("CCO", "CCC")                 # two molecules: False


def _key(triple):
    return memo._ck("full_inchikey_offer_match", (triple[0], triple[1], NAME))


@pytest.fixture(autouse=True)
def _memo_on():
    if memo._MODE != "on" or memo._PROCESS_MAX <= 0:
        pytest.skip("the process memo is off in this run")
    yield
    for triple in (SAME, OTHER):
        memo._process_cache.pop(_key(triple), None)


def test_a_compare_that_raised_is_not_memoised_and_is_recomputed(monkeypatch):
    import orthonym.validation.protonation_identity as pi
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("transient")

    monkeypatch.setattr(pi, "protonation_site_verdict", boom)
    assert namer._full_inchikey_offer_match(*SAME, NAME) is True       # fail-open, as before
    assert _key(SAME) not in memo._process_cache                        #... and not stored
    assert namer._full_inchikey_offer_match(*SAME, NAME) is True
    assert len(calls) == 2                                              # recomputed, not replayed
    monkeypatch.undo()
    # once the compare can be made its verdict is stored (the memo itself still works)
    assert namer._full_inchikey_offer_match(*SAME, NAME) is True
    assert _key(SAME) in memo._process_cache
    assert memo._process_cache[_key(SAME)] is True


def test_the_verdict_of_a_conclusive_compare_is_still_memoised():
    assert namer._full_inchikey_offer_match(*OTHER, NAME) is False
    assert memo._process_cache[_key(OTHER)] is False


def test_the_scope_memo_stores_nothing_for_a_raising_compare(monkeypatch):
    import orthonym.validation.protonation_identity as pi
    monkeypatch.setattr(memo, "_PROCESS_MAX", 0)                        # the scope memo only
    monkeypatch.setattr(pi, "protonation_site_verdict",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("transient")))
    scope = memo.push_scope()
    try:
        ck = memo._ck("full_inchikey_offer_match", (SAME[0], SAME[1], NAME))
        assert namer._full_inchikey_offer_match(*SAME, NAME) is True
        assert ck not in (memo._cache_var.get() or {})
    finally:
        memo.pop_scope(scope)


def test_the_impl_raises_instead_of_returning_the_fail_open_verdict(monkeypatch):
    import orthonym.validation.protonation_identity as pi
    monkeypatch.setattr(pi, "protonation_site_verdict",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("transient")))
    with pytest.raises(namer._InconclusiveCompare):
        namer._full_inchikey_offer_match_impl(*SAME, NAME)
    # an unparseable string is a deterministic answer (RDKit's), and stays fail-open
    assert namer._full_inchikey_offer_match_impl("not a smiles", "CCO", NAME) is True
