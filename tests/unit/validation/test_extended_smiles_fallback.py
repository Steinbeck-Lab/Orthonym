"""Lever A: a definitive in-process OPSIN rejection must not start a Java subprocess,
and the same name must not be looked up twice within one naming scope."""
import subprocess
import pytest
from orthonym import jvm_bridge
from orthonym.validation import opsin_roundtrip as orr
from orthonym.assembly import memo


def test_rejected_name_skips_subprocess(monkeypatch):
    calls = []
    monkeypatch.setattr(orr, "opsin_extended_smiles", lambda name, jar_path=None: (jvm_bridge.REJECTED, True))
    monkeypatch.setattr(orr, "_find_opsin_jar", lambda v="2.9.0": "/fake/opsin.jar")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or pytest.fail("subprocess.run must not be called"))
    assert orr._extended_smiles("not-a-name") is None
    assert calls == []


def test_unavailable_still_falls_back(monkeypatch):
    class R:  # what subprocess.run returns
        stdout = "CCO |$_AV:1;2;$|\n"
    calls = []
    monkeypatch.setattr(orr, "opsin_extended_smiles", lambda name, jar_path=None: (None, False))
    monkeypatch.setattr(orr, "_find_opsin_jar", lambda v="2.9.0": "/fake/opsin.jar")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a) or R())
    assert orr._extended_smiles("ethanol") == "CCO |$_AV:1;2;$|"
    assert len(calls) == 1


def test_memoised_within_scope(monkeypatch):
    n = {"calls": 0}
    def fake(name, jar_path=None):
        n["calls"] += 1
        return ("C |$_AV:1$|", True)
    monkeypatch.setattr(orr, "opsin_extended_smiles", fake)
    monkeypatch.setattr(orr, "_find_opsin_jar", lambda v="2.9.0": "/fake/opsin.jar")
    tok = memo.push_scope()
    try:
        assert orr._extended_smiles("methane") == "C |$_AV:1$|"
        assert orr._extended_smiles("methane") == "C |$_AV:1$|"
    finally:
        memo.pop_scope(tok)
    assert n["calls"] == 1


def test_real_rejection_is_tristate():
    """Live JVM: a nonsense name is a definitive rejection, not 'cannot serve'."""
    if not jvm_bridge.opsin_available():
        pytest.skip("no in-process OPSIN")
    ext, served = jvm_bridge.opsin_extended_smiles("this is not a chemical name", jar_path=jvm_bridge._OPSIN_JAR)
    assert served is True and ext == jvm_bridge.REJECTED
