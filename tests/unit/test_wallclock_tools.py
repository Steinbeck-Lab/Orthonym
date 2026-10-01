"""Every per-molecule alarm in the tools raises a BaseException.

A SIGALRM handler that raises an ``Exception`` subclass is caught by whichever of the
naming path's ``except Exception`` handlers is active when the alarm fires, and the
naming carries on from a half-finished choice (``orthonym.wallclock``). This scans
the tool trees (``eval/``, ``scripts/``, ``benchmarks/``, ``tests/``), including
scripts a tool or test writes out as a string and runs in a subprocess, and checks:

* every handler installed for ``SIGALRM`` raises only ``BaseException`` subclasses
  that are not ``Exception`` subclasses (``orthonym.wallclock.WallClockTimeout``, or
  a class of the same file that derives from ``BaseException``);
* a file that arms an alarm (``signal.alarm`` / ``signal.setitimer``) also installs
  such a handler itself, so the arming cannot fall back on a handler that raises an
  ``Exception``.

The tools use ``orthonym.wallclock.wall_clock_limit``, which satisfies both.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL_TREES = ("eval", "scripts", "benchmarks", "tests")

# Exception classes a handler may raise: BaseException subclasses outside Exception.
_ALLOWED_BUILTIN = {"BaseException", "KeyboardInterrupt", "SystemExit", "GeneratorExit"}
_ALLOWED_IMPORTED = {"WallClockTimeout", "PerfBudgetExceeded"}


def _is_sigalrm(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "SIGALRM") or \
        (isinstance(node, ast.Name) and node.id == "SIGALRM")


def _call_name(node) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _class_is_base_not_exception(name: str, classes: dict, seen=()) -> bool:
    if name in _ALLOWED_BUILTIN or name in _ALLOWED_IMPORTED:
        return True
    cls = classes.get(name)
    if cls is None or name in seen:
        return False
    bases = []
    for b in cls.bases:
        if isinstance(b, ast.Name):
            bases.append(b.id)
        elif isinstance(b, ast.Attribute):
            bases.append(b.attr)
        else:
            return False
    return bool(bases) and all(
        _class_is_base_not_exception(b, classes, seen + (name,)) for b in bases)


def _raised_names(func) -> list:
    """Names of the classes a handler raises (a bare ``raise`` re-raises, kept as
    ``<reraise>``; anything not a plain name or call of a name as ``<expr>``)."""
    out = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if exc is None:
            out.append("<reraise>")
        elif isinstance(exc, ast.Call) and isinstance(exc.func, (ast.Name, ast.Attribute)):
            out.append(exc.func.id if isinstance(exc.func, ast.Name) else exc.func.attr)
        elif isinstance(exc, ast.Name):
            out.append(exc.id)
        else:
            out.append("<expr>")
    return out


def _embedded_scripts(tree):
    """Python source a file carries in a string literal that installs a SIGALRM
    handler (an f-string's placeholders read as ``0``)."""
    in_fstring = {id(v) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)
                  for v in n.values}
    for node in ast.walk(tree):
        text = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in in_fstring:
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) else "0"
                           for v in node.values)
        if text and "signal.signal(" in text and "SIGALRM" in text:
            yield text


def _problems_in_source(source: str, where: str) -> list:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return [f"{where}: carries `signal.signal(... SIGALRM ...)` in text that does "
                "not parse as Python, so its handler cannot be checked"]
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    funcs = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(n.name, []).append(n)
    problems, installs, arms = [], 0, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "signal" and len(node.args) >= 2 and _is_sigalrm(node.args[0]):
            handler = node.args[1]
            if isinstance(handler, ast.Lambda):
                bodies = [handler]
            elif isinstance(handler, ast.Name) and handler.id in funcs:
                bodies = funcs[handler.id]
            elif isinstance(handler, ast.Attribute) and handler.attr in funcs:
                bodies = funcs[handler.attr]
            else:
                continue  # putting back a saved handler (SIG_DFL, a variable)
            installs += 1
            for body in bodies:
                for raised in _raised_names(body):
                    if not _class_is_base_not_exception(raised, classes):
                        problems.append(
                            f"{where}:{node.lineno}: SIGALRM handler raises {raised}, "
                            "which an `except Exception` on the naming path can absorb; "
                            "use orthonym.wallclock.wall_clock_limit")
        elif name in ("alarm", "setitimer") and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "signal":
            if node.args and isinstance(node.args[0] if name == "alarm" else node.args[-1],
                                        ast.Constant) and \
                    (node.args[0] if name == "alarm" else node.args[-1]).value == 0:
                continue  # disarming
            arms.append(node.lineno)
    if arms and not installs:
        problems.append(f"{where}:{arms[0]}: arms an alarm without installing a "
                        "BaseException handler in the same file; use "
                        "orthonym.wallclock.wall_clock_limit")
    return problems


def _tool_files():
    this_file = Path(__file__).resolve()
    for tree in TOOL_TREES:
        for path in sorted((REPO / tree).rglob("*.py")):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            if path.resolve() == this_file:  # its own known positives
                continue
            yield path


def _all_problems():
    problems = []
    for path in _tool_files():
        source = path.read_text(encoding="utf-8", errors="replace")
        if "SIGALRM" not in source and "signal.alarm" not in source \
                and "setitimer" not in source:
            continue
        rel = str(path.relative_to(REPO))
        tree = ast.parse(source)
        problems += _problems_in_source(source, rel)
        for k, script in enumerate(_embedded_scripts(tree)):
            problems += _problems_in_source(script, f"{rel}<embedded script {k}>")
    return problems


def test_every_tool_alarm_raises_a_base_exception():
    problems = _all_problems()
    assert not problems, "\n".join(problems)


def test_the_scan_sees_an_exception_raising_handler():
    """Known positives: the shapes the tools used before ``wall_clock_limit``."""
    shapes = {
        "local class": (
            "import signal\n"
            "class _Timeout(Exception):\n    pass\n"
            "def _on_alarm(s, f):\n    raise _Timeout()\n"
            "signal.signal(signal.SIGALRM, _on_alarm)\nsignal.alarm(30)\n"),
        "builtin": (
            "import signal\n"
            "def run():\n"
            "    def _bail(signum, frame):\n        raise TimeoutError('t')\n"
            "    signal.signal(signal.SIGALRM, _bail)\n    signal.alarm(30)\n"),
        "arm only": "import signal\nsignal.setitimer(signal.ITIMER_REAL, 90)\n",
    }
    for label, source in shapes.items():
        assert _problems_in_source(source, label), label
    ok = ("import signal\n"
          "class _Stop(BaseException):\n    pass\n"
          "def _h(s, f):\n    raise _Stop()\n"
          "signal.signal(signal.SIGALRM, _h)\nsignal.alarm(5)\nsignal.alarm(0)\n")
    assert _problems_in_source(ok, "ok") == []


def test_the_scan_reads_scripts_embedded_in_strings():
    source = (
        "import textwrap\n"
        "def build(t):\n"
        "    return textwrap.dedent(f'''\n"
        "        import signal\n"
        "        class _T(Exception):\n"
        "            pass\n"
        "        def _h(signum, frame):\n"
        "            raise _T()\n"
        "        signal.signal(signal.SIGALRM, _h)\n"
        "        signal.alarm({t})\n"
        "        ''')\n")
    tree = ast.parse(source)
    scripts = list(_embedded_scripts(tree))
    assert len(scripts) == 1
    assert _problems_in_source(scripts[0], "embedded")
