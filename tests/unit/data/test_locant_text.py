"""``render_locants`` (``orthonym.data.locant_text``): the comma-separated text of a list of locants.

IUPAC 2013 "Commas", (a): commas separate locants, numerals and italicized letters
(``1,2-dibromoethane``). The old expressions below are the hand-written joins that call sites
used; the function must give the same text for each of them.
"""

import ast
import inspect

import pytest

from orthonym.assembly import naming_utils
from orthonym.data import locant_text
from orthonym.data.locant_text import render_locants

CORPUS = [[], [1], [1, 2], [3, 1, 2], ["4a", "2H", "N1", "1'", "5" + chr(39)], [10, 2, 2], ["N", "N'", "O"]]


@pytest.mark.unit
class TestRenderLocants:
    def test_commas_without_a_space(self):
        assert render_locants([1, 2]) == "1,2"
        assert render_locants(["N", "N'"]) == "N,N'"

    def test_order_is_the_callers(self):
        assert render_locants([3, 1, 2]) == "3,1,2"
        assert render_locants({5: "a", 2: "b"}) == "5,2"
        s = {7, 3}
        assert render_locants(s) == ",".join(str(x) for x in s)

    def test_empty_and_single(self):
        assert render_locants([]) == ""
        assert render_locants([5]) == "5"

    def test_text_tokens_pass_unchanged(self):
        tokens = ["4a", "2H", "N1", "1'", "5" + chr(39)]
        assert render_locants(tokens) == "4a,2H,N1,1',5'"

    def test_items_that_are_not_text_are_written_with_str(self):
        """``",".join(...)`` of the same list raises ``TypeError``; the function writes each item with ``str``."""
        items = [1, None, 2.5]
        assert render_locants(items) == ",".join(str(x) for x in items) == "1,None,2.5"

    def test_generator_is_consumed_once_in_order(self):
        pulled = []

        def gen():
            for x in (3, 1, 2):
                pulled.append(x)
                yield x

        assert render_locants(gen()) == "3,1,2"
        assert pulled == [3, 1, 2]

    def test_str_of_each_item_changes_nothing(self):
        class Loc(str):
            pass

        for items in ([1, 2, 3], ["2", "4a"], [Loc("1"), Loc("1'")]):
            assert render_locants(str(x) for x in items) == render_locants(items)

    @pytest.mark.parametrize("items", CORPUS)
    def test_equals_every_old_shape(self, items):
        text = [str(x) for x in items]
        assert render_locants(items) == ",".join(str(x) for x in items)
        assert render_locants(items) == ",".join(map(str, items))
        assert render_locants(text) == ",".join(text)
        assert render_locants(sorted(text)) == ",".join(str(x) for x in sorted(text))
        assert render_locants(f"N{x}" for x in items) == ",".join(f"N{x}" for x in items)

    def test_naming_utils_exposes_the_same_function(self):
        assert naming_utils.render_locants is render_locants

    def test_module_is_a_leaf(self):
        """No import of ``orthonym`` (absolute or relative): any module may import it at load time."""
        tree = ast.parse(inspect.getsource(locant_text))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"relative import at line {node.lineno}"
                assert not (node.module or "").startswith("orthonym")
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("orthonym") for a in node.names)
