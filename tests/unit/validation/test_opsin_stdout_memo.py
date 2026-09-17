"""Lever I (a performance pass): identical in-process OPSIN parses within one naming scope are served from the memo."""
from orthonym import jvm_bridge
from orthonym.assembly import memo


def test_opsin_stdout_memo_hits_within_scope(monkeypatch):
    n = {"calls": 0}
    def fake(name, allow_radicals, jar_path=None):
        n["calls"] += 1
        return ("CCO\n", True)
    monkeypatch.setattr(jvm_bridge, "_opsin_stdout_uncached", fake)
    tok = memo.push_scope()
    try:
        assert jvm_bridge.opsin_stdout("ethanol", False) == ("CCO\n", True)
        assert jvm_bridge.opsin_stdout("ethanol", False) == ("CCO\n", True)
        assert jvm_bridge.opsin_stdout("ethanol", True) == ("CCO\n", True)   # different flag = different key
    finally:
        memo.pop_scope(tok)
    assert n["calls"] == 2


def test_opsin_stdout_no_scope_and_unserved_not_cached(monkeypatch):
    n = {"calls": 0}
    def fake(name, allow_radicals, jar_path=None):
        n["calls"] += 1
        return (None, False)
    monkeypatch.setattr(jvm_bridge, "_opsin_stdout_uncached", fake)
    assert memo._cache_var.get() is None
    jvm_bridge.opsin_stdout("x", False); jvm_bridge.opsin_stdout("x", False)
    tok = memo.push_scope()
    try:
        jvm_bridge.opsin_stdout("x", False); jvm_bridge.opsin_stdout("x", False)   # unserved -> never cached
    finally:
        memo.pop_scope(tok)
    assert n["calls"] == 4
