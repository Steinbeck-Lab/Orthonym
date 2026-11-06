"""Serve the OPSIN command-line calls of the test code in process (PR runs).

About 60 public test files round-trip names through OPSIN by running
``java -jar opsin.jar [-r] -osmi`` (or ``-o smi``) once per case. A JVM start costs about
a second, the in-process parse about a millisecond, and
``orthonym.jvm_bridge._opsin_stdout_uncached`` is defined to return exactly what that
command writes to stdout for one name ("byte for byte", its docstring). So, in the PR
run, this module answers such a call through the in-process JVM instead of starting a new
one. A ``-ostdinchikey`` call (OPSIN's own StdInChIKey of the name) is answered by the
persistent JVM of ``tests/support/indep_oracle``, which makes the same OPSIN call
(``NameToInchi.convertResultToStdInChIKey``). ``tests/unit/validation/test_opsin_cli_inproc.py``
compares both with the command line on 40 names.

Scope, so that a call is never served when its result could differ:

* Only ``subprocess.run`` calls whose argv is ``java [jvm flags] -jar <the OPSIN jar the
  suite resolves> [-r] (-osmi | -o smi | -ostdinchikey | -o stdinchikey)``, with ``input`` a ``str`` in text mode and the
  output captured. Every other call (other output modes, files, bytes mode, another jar,
  every other program) goes to the real ``subprocess.run`` unchanged.
* Only when the immediate caller is test code (a file under ``tests/`` or ``tests_local/``).
  The three ``src`` call sites that start the OPSIN command line themselves
  (``validation/opsin_roundtrip.py``, ``rules/parent_correctness.py``,
  ``assembly/retained_substitution.py``) keep the real command line in PR runs.
* Not when ``ORTHONYM_TEST_REAL_OPSIN_CLI=1`` (the nightly run sets it, so the real
  command line still runs against every case once a night), and not in a test marked
  ``real_opsin_cli`` (a test that compares something with the real command line).

``REAL_RUN`` keeps the original ``subprocess.run``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REAL_RUN = subprocess.run

_REPO = Path(__file__).resolve().parents[2]
_TEST_ROOTS = tuple(str(_REPO / d) + os.sep for d in ("tests", "tests_local"))
_ENV_FORCE_REAL = "ORTHONYM_TEST_REAL_OPSIN_CLI"

#: set by the ``real_opsin_cli`` marker fixture (tests/conftest.py) for the length of one test
SUSPENDED = False
#: number of calls served in process; the equality test uses it to see the shim at work
SERVED = 0

_UNSET = object()
_jar_real = _UNSET


def forced_real() -> bool:
    """True when the environment asks for the real command line everywhere."""
    return os.environ.get(_ENV_FORCE_REAL, "").strip().lower() in ("1", "true", "yes", "on")


def _suite_jar():
    """Real path of the OPSIN jar the suite resolves, or None."""
    global _jar_real
    if _jar_real is _UNSET:
        from tests.support.jars import _resolve
        path, _reason = _resolve("opsin")
        _jar_real = os.path.realpath(path) if path else None
    return _jar_real


def _opsin_smi_call(args):
    """``(allow_radicals, output)`` when ``args`` is a plain OPSIN SMILES or StdInChIKey call on the
    suite's jar (``output`` 'smi' or 'stdinchikey')."""
    if not isinstance(args, (list, tuple)) or "-jar" not in args:
        return None
    argv = [str(a) for a in args]
    i = argv.index("-jar")
    if i + 1 >= len(argv) or _suite_jar() is None:
        return None
    try:
        if os.path.realpath(argv[i + 1]) != _suite_jar():
            return None
    except (OSError, ValueError):
        return None
    rest, radicals, output, k = argv[i + 2:], False, None, 0
    while k < len(rest):
        flag = rest[k]
        if flag == "-r":
            radicals = True
        elif flag in ("-osmi", "-ostdinchikey") and output is None:
            output = flag[2:]
        elif flag == "-o" and k + 1 < len(rest) and rest[k + 1] in ("smi", "stdinchikey") and output is None:
            output = rest[k + 1]
            k += 1
        else:
            return None
        k += 1
    return (radicals, output) if output else None


def stdinchikey_stdout(name: str, radicals: bool):
    """What ``-ostdinchikey`` writes for one name (the key, or nothing for a name OPSIN does not read,
    then a newline), from the persistent JVM of ``indep_oracle``; None when it cannot answer."""
    from tests.support import indep_oracle
    got = indep_oracle.keys("", name, radicals)
    if got is None or got[1] in ("", "ERR"):
        return None
    return "\n" if got[1] == "NOPARSE" else got[1] + "\n"


def _caller_is_test_code(frame) -> bool:
    try:
        return os.path.realpath(frame.f_code.co_filename).startswith(_TEST_ROOTS)
    except (OSError, ValueError):
        return False


def _serve(args, kwargs, frame):
    """The in-process ``CompletedProcess`` for this call, or None to run the real command."""
    global SERVED
    text = kwargs.get("input")
    if not isinstance(text, str):
        return None
    if not (kwargs.get("text") or kwargs.get("universal_newlines")):
        return None
    if not (kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE):
        return None
    if kwargs.get("stdin") is not None:
        return None
    call = _opsin_smi_call(args)
    if call is None or not _caller_is_test_code(frame):
        return None
    from orthonym.jvm_bridge import _opsin_stdout_uncached
    jar = _suite_jar()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out = []
    for line in lines:
        if call[1] == "stdinchikey":
            # the command line reads a name up to its first tab
            stdout = stdinchikey_stdout(line.split("\t")[0], call[0])
            served = stdout is not None
        else:
            stdout, served = _opsin_stdout_uncached(line, call[0], jar)
        if not served:
            return None          # the bridge cannot answer this one: run the real command
        out.append(stdout)
    SERVED += 1
    return subprocess.CompletedProcess(args, 0, "".join(out), "")


def _run(*popenargs, **kwargs):
    if not SUSPENDED:
        args = popenargs[0] if popenargs else kwargs.get("args")
        served = _serve(args, kwargs, sys._getframe(1))
        if served is not None:
            return served
    return REAL_RUN(*popenargs, **kwargs)


def install() -> bool:
    """Patch ``subprocess.run`` for this process; False (nothing patched) when forced real."""
    if forced_real():
        return False
    subprocess.run = _run
    return True


def uninstall() -> None:
    subprocess.run = REAL_RUN
