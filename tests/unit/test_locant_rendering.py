"""A new hand-written comma join of locants fails; locants are written with ``render_locants``.

``orthonym.data.locant_text.render_locants`` is the one home of the comma rule for locants
(IUPAC 2013 "Commas", (a)). The functions that still join locants by hand are listed in
``tests/support/locant_rendering_baseline.py``; a listed one may not gain a join, and the list may
only shrink. See ``tests/support/locant_rendering.py`` for what is counted.

AST only: no ``orthonym`` import, no JVM, a few seconds.
"""

import ast
import warnings
from pathlib import Path

import pytest

from tests.support.locant_rendering import SRC_PACKAGE, check, count_source, count_tree
from tests.support.locant_rendering_baseline import LOCANT_JOINS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_no_new_hand_written_locant_join():
    counts = count_tree(ROOT)
    assert counts, "the count found nothing; the check would be vacuous"
    v = check(counts, LOCANT_JOINS)
    if v.failures:
        pytest.fail("\n" + "\n".join(v.failures), pytrace=False)
    if v.lower:
        warnings.warn("the locant-join baseline can be tightened:\n  " + "\n  ".join(v.lower))


def test_validation_does_not_import_the_renderer():
    """The spelling auditors stay independent of the code they audit."""
    for p in sorted((ROOT / SRC_PACKAGE / "validation").rglob("*.py")):
        for node in ast.walk(ast.parse(p.read_bytes())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith("locant_text"), f"{p}:{node.lineno}"
                assert not any(a.name == "render_locants" for a in node.names), f"{p}:{node.lineno}"


# ---------------------------------------------------------------------------
# The checker itself (so the ratchet cannot silently stop failing)
# ---------------------------------------------------------------------------

REL = "src/orthonym/rules/m.py"


def test_new_hand_written_locant_join_fails():
    counts = count_source(REL, b'def f(locs):\n    return ",".join(str(loc) for loc in locs)\n')
    v = check(counts, {})
    assert len(v.failures) == 1 and f"{REL}::f" in v.failures[0] and "render_locants" in v.failures[0]


def test_listed_function_may_not_gain_a_join():
    src = b'def f(locs):\n    return ",".join(locs) + "-" + ", ".join(map(str, locs))\n'
    assert check(count_source(REL, src), {f"{REL}::f": 1}).failures
    assert not check(count_source(REL, src), {f"{REL}::f": 2}).failures


def test_a_removed_join_only_says_lower_the_entry():
    v = check(count_source(REL, b'def f(locs):\n    return ",".join(locs)\n'), {f"{REL}::f": 2, f"{REL}::g": 1})
    assert not v.failures
    assert v.lower == [f"lower the entry {REL}::f to 1", f"delete the entry {REL}::g"]


def test_render_locants_calls_are_not_counted():
    assert count_source(REL, b"def f(locs):\n    return render_locants(sorted(locs))\n") == {}


def test_joins_without_a_locant_are_not_counted():
    assert count_source(REL, b'def f(names):\n    return ",".join(names) + "-".join(locs)\n') == {}


def test_validation_is_not_scanned(tmp_path):
    pkg = tmp_path / SRC_PACKAGE
    (pkg / "validation").mkdir(parents=True)
    (pkg / "validation" / "a.py").write_text('def f(locs):\n    return ",".join(locs)\n')
    (pkg / "b.py").write_text('def g(locs):\n    return ",".join(locs)\n')
    assert count_tree(tmp_path) == {f"{SRC_PACKAGE}/b.py::g": 1}
