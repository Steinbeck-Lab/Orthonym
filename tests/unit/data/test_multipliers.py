"""The multiplying-prefix tables and ``multiplier_table`` (``orthonym.data.multipliers``).

The words are the basic numerical terms of IUPAC 2013 (Table 1.4).
``OLD_SIMPLE`` and ``OLD_COMPLEX`` below are a fixed copy of those words, so the tests
compare the shared tables and the table builder against an independent reference (same
keys, same values, same order), not against the code under test.
"""

import ast
import inspect

import pytest

from orthonym.assembly import naming_utils
from orthonym.data import multipliers
from orthonym.data.multipliers import (
    COMPLEX_MULTIPLIERS,
    SIMPLE_MULTIPLIERS,
    multiplier_table,
)

OLD_SIMPLE = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
    17: "heptadeca",
    18: "octadeca",
    19: "nonadeca",
    20: "icosa",
}

OLD_COMPLEX = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
    7: "heptakis",
    8: "octakis",
    9: "nonakis",
    10: "decakis",
    11: "undecakis",
    12: "dodecakis",
    13: "tridecakis",
    14: "tetradecakis",
    15: "pentadecakis",
    16: "hexadecakis",
    17: "heptadecakis",
    18: "octadecakis",
    19: "nonadecakis",
    20: "icosakis",
}

OLD = {"simple": OLD_SIMPLE, "complex": OLD_COMPLEX}

# The ranges the tables had at the sites that carried their own copy (max key per site):
# 3, 4, 5, 6, 7, 8, 10, 12, 15. Every max_n from 2 to 20 is tested, which covers them.
RANGES = tuple(range(2, 21))


@pytest.mark.unit
class TestSharedTables:
    def test_simple_items_equal_the_old_literal(self):
        assert list(SIMPLE_MULTIPLIERS.items()) == list(OLD_SIMPLE.items())

    def test_complex_items_equal_the_old_literal(self):
        assert list(COMPLEX_MULTIPLIERS.items()) == list(OLD_COMPLEX.items())

    def test_naming_utils_exports_the_same_objects(self):
        assert naming_utils.SIMPLE_MULTIPLIERS is SIMPLE_MULTIPLIERS
        assert naming_utils.COMPLEX_MULTIPLIERS is COMPLEX_MULTIPLIERS

    def test_module_is_a_leaf(self):
        """No import of ``orthonym`` (absolute or relative): any module may import it at load time."""
        tree = ast.parse(inspect.getsource(multipliers))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"relative import at line {node.lineno}"
                assert not (node.module or "").startswith("orthonym")
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("orthonym") for a in node.names)


@pytest.mark.unit
class TestMultiplierTable:
    @pytest.mark.parametrize("kind", ["simple", "complex"])
    @pytest.mark.parametrize("one", [None, "", "mono"])
    @pytest.mark.parametrize("max_n", RANGES)
    def test_items_equal_the_old_literal_slice(self, kind, one, max_n):
        expected = ([(1, one)] if one is not None else []) + [
            (k, v) for k, v in OLD[kind].items() if k <= max_n
        ]
        got = multiplier_table(max_n, one, kind)
        assert type(got) is dict
        assert list(got.items()) == expected  # same keys, same values, same order

    def test_default_form_is_simple_with_no_key_1(self):
        assert list(multiplier_table(4).items()) == [(2, "di"), (3, "tri"), (4, "tetra")]

    def test_one_and_kind_can_be_keywords(self):
        got = multiplier_table(3, one="", kind="complex")
        assert list(got.items()) == [(1, ""), (2, "bis"), (3, "tris")]

    def test_full_range_equals_the_shared_table(self):
        assert list(multiplier_table(20).items()) == list(SIMPLE_MULTIPLIERS.items())
        assert list(multiplier_table(20, kind="complex").items()) == list(
            COMPLEX_MULTIPLIERS.items()
        )

    def test_every_call_returns_a_fresh_dict(self):
        a = multiplier_table(5, "")
        b = multiplier_table(5, "")
        assert a is not b
        assert a == b and list(a.items()) == list(b.items())

    def test_editing_a_result_changes_nothing_else(self):
        a = multiplier_table(5)
        a[2] = "changed"
        del a[3]
        a[99] = "extra"
        assert list(multiplier_table(5).items()) == [
            (k, v) for k, v in OLD_SIMPLE.items() if k <= 5
        ]
        assert list(SIMPLE_MULTIPLIERS.items()) == list(OLD_SIMPLE.items())
        c = multiplier_table(5, kind="complex")
        c[2] = "changed"
        assert COMPLEX_MULTIPLIERS[2] == "bis"
        assert multiplier_table(5, kind="complex")[2] == "bis"

    @pytest.mark.parametrize("max_n", [-1, 0, 1, 21, 100])
    def test_max_n_outside_2_to_20_is_refused(self, max_n):
        with pytest.raises(ValueError):
            multiplier_table(max_n)

    @pytest.mark.parametrize("kind", ["", "Simple", "bis", None])
    def test_unknown_kind_is_refused(self, kind):
        with pytest.raises(ValueError):
            multiplier_table(3, kind=kind)

    @pytest.mark.parametrize("one", [1, 0, b"", ["mono"]])
    def test_non_string_one_is_refused(self, one):
        with pytest.raises(TypeError):
            multiplier_table(3, one)
