"""Perf lever A8 (2026-09-13): the process-wide cache for pure namespaces serves repeats
across naming scopes, stays bounded, and is bypassed in off/verify memo modes."""
import pytest

from orthonym.assembly import memo as M


@pytest.fixture(autouse=True)
def _clean():
    M.clear_process_cache()
    yield
    M.clear_process_cache()


def test_value_survives_the_scope(monkeypatch):
    calls = []

    def compute():
        calls.append(1)
        return "v"

    tok = M.push_scope()
    assert M.pure_cache_or_compute("ns", ("k",), compute) == "v"
    M.pop_scope(tok)
    tok = M.push_scope()          # a new molecule: the scope cache is empty
    assert M.pure_cache_or_compute("ns", ("k",), compute) == "v"
    M.pop_scope(tok)
    assert len(calls) == 1
    assert M.process_cache_stats()["hits"] == 1


def test_bounded_lru(monkeypatch):
    monkeypatch.setattr(M, "_PROCESS_MAX", 3)
    for i in range(5):
        M.pure_cache_or_compute("ns", i, lambda i=i: i)
    assert M.process_cache_stats()["size"] == 3
    # the two oldest were evicted, the newest survive
    assert ("ns", 4) in M._process_cache and ("ns", 0) not in M._process_cache


def test_off_and_verify_modes_bypass(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "_MODE", "off")
    for _ in range(2):
        M.pure_cache_or_compute("ns", "k", lambda: calls.append(1))
    assert len(calls) == 2 and M.process_cache_stats()["size"] == 0
    monkeypatch.setattr(M, "_MODE", "verify")
    tok = M.push_scope()
    for _ in range(2):
        M.pure_cache_or_compute("ns", "k2", lambda: "same")
    M.pop_scope(tok)
    assert M.process_cache_stats()["size"] == 0


def test_disabled_by_zero_size(monkeypatch):
    monkeypatch.setattr(M, "_PROCESS_MAX", 0)
    calls = []
    for _ in range(2):
        M.pure_cache_or_compute("ns", "k", lambda: calls.append(1))
    assert len(calls) == 2
