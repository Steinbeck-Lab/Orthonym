"""Measure function and file length in ``src/orthonym``, and compare it with an allow-list.

This is the engine behind ``tests/unit/test_size_budget.py``. It is pure standard library
(``ast``): it never imports ``orthonym`` and never starts a JVM, so it runs in any CI job.

The rule is a ratchet on NEW code, not a clean-up of old code:

* no function longer than ``MAX_FUNCTION_LINES`` and no file longer than ``MAX_FILE_LINES``,
* unless it is listed in ``tests/support/size_budget_baseline.py`` -- and a listed one may not grow.

An entry that shrank, or whose function or file is gone, is not a failure; it is reported as
"lower the entry" so the allow-list gets tighter over time and never looser.

Measuring conventions (they are what the baseline was generated with, so do not change them
without regenerating it):

* a function's length is ``end_lineno - lineno + 1`` of its ``def`` (decorators not counted);
  nested functions and methods are measured on their own and also inside their parent,
* a file's length is its number of lines (a last line without a newline counts),
* a function is keyed ``<file>::<qualname>#<n>``: the path relative to the repository root, the
  dotted qualified name (``Class.method``, ``outer.<locals>.inner``), and ``n`` counting the
  functions of that file with the same qualified name in source order, from 1. A key holds no line
  number, so edits above a function do not change its key.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

MAX_FUNCTION_LINES = 200
MAX_FILE_LINES = 5000
SRC_PACKAGE = "src/orthonym"

BASELINE_MODULE = "tests/support/size_budget_baseline.py"


def count_lines(data: bytes) -> int:
    """Number of lines in ``data`` (like ``wc -l``, but a last line without a newline counts)."""
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def _collect(node: ast.AST, scope: List[str], out: List[Tuple[int, str, int]]) -> None:
    """Append ``(first line, qualified name, length)`` for every function below ``node``."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join(scope + [child.name])
            out.append((child.lineno, qualname, child.end_lineno - child.lineno + 1))
            _collect(child, scope + [child.name, "<locals>"], out)
        elif isinstance(child, ast.ClassDef):
            _collect(child, scope + [child.name], out)
        else:
            _collect(child, scope, out)


def measure_source(relpath: str, data: bytes) -> Tuple[int, Dict[str, int]]:
    """``(file length, {function key: function length})`` for one file's bytes."""
    found: List[Tuple[int, str, int]] = []
    _collect(ast.parse(data, filename=relpath), [], found)
    found.sort(key=lambda item: item[0])
    seen: Dict[str, int] = {}
    functions: Dict[str, int] = {}
    for _lineno, qualname, length in found:
        seen[qualname] = seen.get(qualname, 0) + 1
        functions[f"{relpath}::{qualname}#{seen[qualname]}"] = length
    return count_lines(data), functions


def measure_tree(root: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Measure every ``.py`` file under ``<root>/src/orthonym``.

    Returns ``(functions, files)``: every function's length by key, every file's length by its
    path relative to ``root`` (POSIX separators).
    """
    functions: Dict[str, int] = {}
    files: Dict[str, int] = {}
    for path in sorted((root / SRC_PACKAGE).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relpath = path.relative_to(root).as_posix()
        length, found = measure_source(relpath, path.read_bytes())
        files[relpath] = length
        functions.update(found)
    return functions, files


@dataclass
class Verdict:
    """What the comparison found. ``*_failures`` fail the test; ``lower`` only informs."""

    function_failures: List[str] = field(default_factory=list)
    file_failures: List[str] = field(default_factory=list)
    lower: List[str] = field(default_factory=list)


def _bare_name(key: str) -> str:
    return key.split("::", 1)[1].rsplit("#", 1)[0].rsplit(".", 1)[-1]


def _rename_candidates(key: str, stale: List[str]) -> List[str]:
    """Baseline entries that no longer exist and may be ``key`` renamed or moved."""
    path = key.split("::", 1)[0]
    return [
        old for old in stale if old.split("::", 1)[0] == path or _bare_name(old) == _bare_name(key)
    ]


def check(
    functions: Mapping[str, int],
    files: Mapping[str, int],
    baseline_functions: Mapping[str, int],
    baseline_files: Mapping[str, int],
    max_function: int = MAX_FUNCTION_LINES,
    max_file: int = MAX_FILE_LINES,
) -> Verdict:
    """Compare a measurement with the allow-list. Pure: no file access."""
    verdict = Verdict()
    stale = sorted(key for key in baseline_functions if key not in functions)

    for key, length in sorted(functions.items()):
        if length <= max_function:
            continue
        allowed = baseline_functions.get(key)
        if allowed is None:
            msg = (
                f"NEW function over budget: {key} is {length} lines (limit {max_function}). "
                f"Keep a new function within {max_function} lines, or split it."
            )
            moved = _rename_candidates(key, stale)
            if moved:
                msg += (
                    f" If it is a baselined function renamed or moved (baseline entries that no "
                    f"longer exist: {', '.join(moved[:3])}), re-key that entry in "
                    f"{BASELINE_MODULE}, keeping its line count or lowering it."
                )
            verdict.function_failures.append(msg)
        elif length > allowed:
            verdict.function_failures.append(
                f"GREW baselined function: {key} is {length} lines, baseline {allowed} "
                f"(+{length - allowed}). A baselined function may not grow; keep it at "
                f"{allowed} lines or fewer."
            )

    for path, length in sorted(files.items()):
        if length <= max_file:
            continue
        allowed = baseline_files.get(path)
        if allowed is None:
            verdict.file_failures.append(
                f"NEW file over budget: {path} is {length} lines (limit {max_file}). "
                f"Put new code in a new module."
            )
        elif length > allowed:
            verdict.file_failures.append(
                f"GREW baselined file: {path} is {length} lines, baseline {allowed} "
                f"(+{length - allowed}). A baselined file may not grow; put new code in a "
                f"new module."
            )

    for key, allowed in sorted(baseline_functions.items()):
        length = functions.get(key)
        if length is None:
            verdict.lower.append(
                f"lower the entry: {key} no longer exists; delete it from {BASELINE_MODULE}."
            )
        elif length <= max_function:
            verdict.lower.append(
                f"lower the entry: {key} is now {length} lines, within the {max_function}-line "
                f"limit; delete it from {BASELINE_MODULE}."
            )
        elif length < allowed:
            verdict.lower.append(
                f"lower the entry: {key} is {length} lines, baseline {allowed}; set it to "
                f"{length} in {BASELINE_MODULE}."
            )

    for path, allowed in sorted(baseline_files.items()):
        length = files.get(path)
        if length is None:
            verdict.lower.append(
                f"lower the entry: {path} no longer exists; delete it from {BASELINE_MODULE}."
            )
        elif length <= max_file:
            verdict.lower.append(
                f"lower the entry: {path} is now {length} lines, within the {max_file}-line "
                f"limit; delete it from {BASELINE_MODULE}."
            )
        elif length < allowed:
            verdict.lower.append(
                f"lower the entry: {path} is {length} lines, baseline {allowed}; set it to "
                f"{length} in {BASELINE_MODULE}."
            )

    return verdict
