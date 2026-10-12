"""Size budget for ``src/orthonym``: no new function over 200 lines, no new file over 5,000.

A ratchet on new code. Functions and files that were already over the limit are listed in
``tests/support/size_budget_baseline.py``; a listed one may not grow, and the list may only
shrink. Everything else must stay within the limit. See ``tests/support/size_budget.py`` for how
length is measured and how a function is keyed.

The tests that read the source tree use ``ast`` only: no ``orthonym`` import, no JVM, a few
seconds. The tests of the checker itself build tiny trees in a temporary directory.

If a test here fails, fix the code (split the function, put new code in a new module); do not
raise a baseline count or add a baseline entry.
"""

import textwrap
import warnings
from pathlib import Path

import pytest

from tests.support.size_budget import (
    MAX_FILE_LINES,
    MAX_FUNCTION_LINES,
    SRC_PACKAGE,
    check,
    count_lines,
    measure_source,
    measure_tree,
)
from tests.support.size_budget_baseline import FILES, FUNCTIONS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The real tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def verdict():
    assert (ROOT / SRC_PACKAGE).is_dir(), f"{SRC_PACKAGE} not found under {ROOT}"
    functions, files = measure_tree(ROOT)
    assert functions and files, "the measurement found nothing; the check would be vacuous"
    return check(functions, files, FUNCTIONS, FILES)


def _report_lower(verdict):
    if verdict.lower:
        warnings.warn("size budget allow-list can be tightened:\n  " + "\n  ".join(verdict.lower))


def test_no_new_or_grown_function_over_budget(verdict):
    """No function over 200 lines unless baselined, and a baselined one may not grow."""
    if verdict.function_failures:
        pytest.fail("\n" + "\n".join(verdict.function_failures), pytrace=False)
    _report_lower(verdict)


def test_no_new_or_grown_file_over_budget(verdict):
    """No file over 5,000 lines unless baselined, and a baselined one may not grow."""
    if verdict.file_failures:
        pytest.fail("\n" + "\n".join(verdict.file_failures), pytrace=False)
    _report_lower(verdict)


def test_baseline_entries_are_over_the_limit():
    """An allow-list entry at or below the limit is dead weight and hides head-room."""
    short = [k for k, n in FUNCTIONS.items() if n <= MAX_FUNCTION_LINES]
    short += [k for k, n in FILES.items() if n <= MAX_FILE_LINES]
    assert not short, f"delete these entries from the baseline, they are within the limit: {short}"


# ---------------------------------------------------------------------------
# The checker itself (so the budget cannot silently stop failing)
# ---------------------------------------------------------------------------


def _function_source(name, body_lines, indent=""):
    """Source of ``def name`` that is exactly ``body_lines + 1`` lines long."""
    body = "\n".join(f"{indent}    x{i} = {i}" for i in range(body_lines - 1))
    return f"{indent}def {name}():\n{body}\n{indent}    return 0\n"


def _measure(source, relpath="src/orthonym/m.py"):
    return measure_source(relpath, source.encode())


def test_function_length_is_def_to_end():
    length, functions = _measure(_function_source("f", MAX_FUNCTION_LINES - 1))
    assert functions == {"src/orthonym/m.py::f#1": MAX_FUNCTION_LINES}
    assert length == MAX_FUNCTION_LINES


def test_limit_is_strict():
    """Exactly 200 lines passes; 201 fails, naming the entry."""
    _, ok = _measure(_function_source("f", MAX_FUNCTION_LINES - 1))
    assert not check(ok, {}, {}, {}).function_failures
    _, big = _measure(_function_source("f", MAX_FUNCTION_LINES))
    failures = check(big, {}, {}, {}).function_failures
    assert len(failures) == 1
    assert "src/orthonym/m.py::f#1" in failures[0] and "201 lines" in failures[0]


def test_baselined_function_may_not_grow_but_may_shrink():
    key = "src/orthonym/m.py::f#1"
    _, now = _measure(_function_source("f", 249))  # 250 lines
    assert key in now and now[key] == 250
    assert not check(now, {}, {key: 250}, {}).function_failures
    grew = check(now, {}, {key: 249}, {}).function_failures
    assert len(grew) == 1 and key in grew[0] and "+1" in grew[0]
    shrank = check(now, {}, {key: 260}, {})
    assert not shrank.function_failures
    assert (
        len(shrank.lower) == 1
        and shrank.lower[0].startswith("lower the entry")
        and key in shrank.lower[0]
    )


def test_deleted_or_now_small_entry_only_says_lower_the_entry():
    key = "src/orthonym/m.py::f#1"
    gone = check({}, {}, {key: 250}, {})
    assert not gone.function_failures and gone.lower and gone.lower[0].startswith("lower the entry")
    _, small = _measure(_function_source("f", 10))
    now_small = check(small, {}, {key: 250}, {})
    assert not now_small.function_failures
    assert now_small.lower and now_small.lower[0].startswith("lower the entry")


def test_file_limit_is_strict_and_names_the_file():
    path = "src/orthonym/big.py"
    assert not check({}, {path: MAX_FILE_LINES}, {}, {}).file_failures
    new = check({}, {path: MAX_FILE_LINES + 1}, {}, {}).file_failures
    assert len(new) == 1 and path in new[0]
    assert not check({}, {path: 5400}, {}, {path: 5400}).file_failures
    grew = check({}, {path: 5401}, {}, {path: 5400}).file_failures
    assert len(grew) == 1 and path in grew[0] and "+1" in grew[0]
    shrank = check({}, {path: 5300}, {}, {path: 5400})
    assert not shrank.file_failures and shrank.lower[0].startswith("lower the entry")


def test_count_lines():
    assert count_lines(b"") == 0
    assert count_lines(b"a\n") == 1
    assert count_lines(b"a\nb") == 2
    assert count_lines(b"a\n\nb\n") == 3


def test_keys_ignore_line_numbers_and_number_same_named_functions():
    source = textwrap.dedent("""\
        class A:
            def run(self):
                def helper():
                    return 1
                return helper

        def run():
            return 0

        if True:
            def run():
                return 1
        """)
    _, found = _measure(source)
    assert set(found) == {
        "src/orthonym/m.py::A.run#1",
        "src/orthonym/m.py::A.run.<locals>.helper#1",
        "src/orthonym/m.py::run#1",
        "src/orthonym/m.py::run#2",
    }
    _, shifted = _measure("# a comment\n\n\n" + source)
    assert set(shifted) == set(found)


def test_a_renamed_or_moved_baselined_function_points_at_its_old_entry():
    old = "src/orthonym/a.py::big#1"
    moved = check({"src/orthonym/b.py::big#1": 250}, {}, {old: 250}, {})
    assert len(moved.function_failures) == 1 and old in moved.function_failures[0]
    renamed = check({"src/orthonym/a.py::bigger#1": 250}, {}, {old: 250}, {})
    assert len(renamed.function_failures) == 1 and old in renamed.function_failures[0]
    unrelated = check({"src/orthonym/c.py::other#1": 250}, {}, {old: 250}, {})
    assert len(unrelated.function_failures) == 1 and old not in unrelated.function_failures[0]
