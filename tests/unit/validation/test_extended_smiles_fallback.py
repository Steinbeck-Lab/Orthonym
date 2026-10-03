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
        returncode = 0
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


class _Stdout:
    """What ``subprocess.run`` returns: OPSIN's stdout and the exit status (OPSIN exits 0
    on a name it reads and on one it rejects; non-zero when the JVM did not start or
    crashed)."""
    def __init__(self, text, returncode=0):
        self.stdout = text
        self.returncode = returncode


def _subprocess_path(monkeypatch, answers):
    """The in-process path cannot serve; ``subprocess.run`` gives ``answers`` in turn (an
    exception instance is raised)."""
    calls = []

    def run(*a, **k):
        calls.append(a)
        got = answers[min(len(calls), len(answers)) - 1]
        if isinstance(got, BaseException):
            raise got
        return got if isinstance(got, _Stdout) else _Stdout(got)
    monkeypatch.setattr(orr, "opsin_extended_smiles", lambda name, jar_path=None: (None, False))
    monkeypatch.setattr(orr, "_find_opsin_jar", lambda v="2.9.0": "/fake/opsin.jar")
    monkeypatch.setattr(subprocess, "run", run)
    memo.clear_process_cache()
    return calls


def test_a_subprocess_timeout_is_not_memoised(monkeypatch):
    # a timeout is a state of the process, not of the name: the next call asks again
    calls = _subprocess_path(monkeypatch, [subprocess.TimeoutExpired("java", 15), "CCC |$_AV:1;2;3$|\n"])
    tok = memo.push_scope()
    try:
        assert orr._extended_smiles("propane") is None
        assert orr._extended_smiles("propane") == "CCC |$_AV:1;2;3$|"
        assert orr._extended_smiles("propane") == "CCC |$_AV:1;2;3$|"
    finally:
        memo.pop_scope(tok)
        memo.clear_process_cache()
    assert len(calls) == 2


def test_the_strict_helper_raises_on_a_timeout(monkeypatch):
    _subprocess_path(monkeypatch, [OSError("no java")])
    try:
        with pytest.raises(orr.OpsinUnavailable):
            orr.extended_smiles_or_unavailable("butane")
    finally:
        memo.clear_process_cache()


def test_a_subprocess_rejection_is_still_memoised(monkeypatch):
    # OPSIN answered (an empty line): a property of the name, kept as before
    calls = _subprocess_path(monkeypatch, ["\n"])
    tok = memo.push_scope()
    try:
        assert orr._extended_smiles("not-a-name-either") is None
        assert orr._extended_smiles("not-a-name-either") is None
    finally:
        memo.pop_scope(tok)
        memo.clear_process_cache()
    assert len(calls) == 1


def test_a_subprocess_crash_is_not_memoised(monkeypatch):
    # a JVM that did not start or crashed exits non-zero with no answer: a state of the
    # process, not a rejection of the name, so the next call asks again
    calls = _subprocess_path(monkeypatch, [_Stdout("", returncode=1), "CCCC |$_AV:1;2;3;4$|\n"])
    tok = memo.push_scope()
    try:
        assert orr._extended_smiles("butane") is None
        assert orr._extended_smiles("butane") == "CCCC |$_AV:1;2;3;4$|"
        assert orr._extended_smiles("butane") == "CCCC |$_AV:1;2;3;4$|"
    finally:
        memo.pop_scope(tok)
        memo.clear_process_cache()
    assert len(calls) == 2


def test_the_strict_helper_raises_on_a_crash(monkeypatch):
    _subprocess_path(monkeypatch, [_Stdout("", returncode=1)])
    try:
        with pytest.raises(orr.OpsinUnavailable):
            orr.extended_smiles_or_unavailable("pentane")
    finally:
        memo.clear_process_cache()
