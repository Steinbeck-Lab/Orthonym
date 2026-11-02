"""Count the hand-written comma joins of locants in ``src/orthonym`` (for ``tests/unit/test_locant_rendering.py``).

A hand-written join is a call ``",".join(arg)`` or ``", ".join(arg)`` whose argument, as
``ast.unparse`` writes it, names a locant (the regex ``LOCANTISH``). The joins are counted per
function, keyed ``<file>::<qualified name>``. ``src/orthonym/validation/`` is not counted: the
spelling auditors keep their own code, so that a bug of the shared function cannot hide itself.
``data/locant_text.py`` holds ``render_locants`` itself and is not counted either.

``python -m tests.support.locant_rendering`` prints the baseline dictionary of the current tree.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SRC_PACKAGE = "src/orthonym"
LOCANTISH = re.compile(r"(?<![a-z])loc(?!k|al|at)|locant|[a-z]locs\b|_l\b", re.I)
SEPARATORS = {",", ", "}
NOT_COUNTED = ("src/orthonym/validation/", "src/orthonym/data/locant_text.py")


class _Joins(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.counts: Counter = Counter()

    def _scope(self, node) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope  # noqa: N815

    def visit_Call(self, node) -> None:
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "join" and isinstance(f.value, ast.Constant)
                and f.value.value in SEPARATORS and len(node.args) == 1
                and LOCANTISH.search(ast.unparse(node.args[0]))):
            self.counts[".".join(self.stack) or "<module>"] += 1
        self.generic_visit(node)


def count_source(relpath: str, source: bytes) -> dict[str, int]:
    v = _Joins()
    v.visit(ast.parse(source))
    return {f"{relpath}::{q}": n for q, n in v.counts.items()}


def count_tree(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in sorted((root / SRC_PACKAGE).rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(NOT_COUNTED):
            continue
        out.update(count_source(rel, p.read_bytes()))
    return out


@dataclass
class Verdict:
    failures: list[str] = field(default_factory=list)
    lower: list[str] = field(default_factory=list)


def check(counts: dict[str, int], baseline: dict[str, int]) -> Verdict:
    v = Verdict()
    for key, n in sorted(counts.items()):
        allowed = baseline.get(key, 0)
        if n > allowed:
            v.failures.append(
                f"{key}: {n} hand-written comma join(s) of locants, {allowed} allowed. Write the locants with "
                "render_locants (orthonym.data.locant_text); never add or raise a baseline entry.")
        elif n < allowed:
            v.lower.append(f"lower the entry {key} to {n}")
    for key in sorted(set(baseline) - set(counts)):
        v.lower.append(f"delete the entry {key}")
    return v


if __name__ == "__main__":
    counts = count_tree(Path(__file__).resolve().parents[2])
    print("LOCANT_JOINS = {")
    for key, n in sorted(counts.items()):
        print(f"    {key!r}: {n},")
    print("}")
