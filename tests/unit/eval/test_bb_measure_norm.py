"""The Blue Book conformance measure compares names after one normalisation on both sides.

The expected names of `benchmarks/bb_conformance/bb_conformance_report.json` went through
`eval/bb_conformance/bb_examples.normalise`, which folds the en dash, the em dash and the
minus sign to '-'. `bb_measure.norm` must fold the engine's name the same way:
adduct names carry an em dash ('9H-fluorene—1-methylnaphthalene (1/1) (PIN)',
the Blue Book; '4-nitrobenzoic acid—quinolin-8-ol (1/1) (PIN)',:3944;
'methanesulfonic acid—cyclopentane-1,3-diamine (1/1) (PIN)',:26718).

`bb_measure` reads its report and argv at import time, so the test takes `norm` out of the
file with `ast` and runs it alone.
"""
import ast
import re
import unicodedata
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "eval" / "bb_conformance" / "bb_measure.py"


def _load_norm():
    tree = ast.parse(_SRC.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "norm")
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_SUP"
                                                  for t in n.targets))]
    ns = {"re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_SRC), "exec"), ns)
    return ns["norm"]


def test_em_dash_adduct_name_equals_its_folded_expected_string():
    norm = _load_norm()
    for engine, expected in [
        ("9H-fluorene—1-methylnaphthalene (1/1)", "9H-fluorene-1-methylnaphthalene (1/1)"),
        ("4-nitrobenzoic acid—quinolin-8-ol (1/1)", "4-nitrobenzoic acid-quinolin-8-ol (1/1)"),
        ("methanesulfonic acid—cyclopentane-1,3-diamine (1/1)",
         "methanesulfonic acid-cyclopentane-1,3-diamine (1/1)"),
        ("coronene—1,3,5-trinitrobenzene (1/1)", "coronene—1,3,5-trinitrobenzene (1/1)"),
    ]:
        assert norm(engine) == norm(expected)


def test_norm_still_separates_different_names():
    norm = _load_norm()
    assert norm("(2R)-butan-2-ol") != norm("(2r)-butan-2-ol")   # case-sensitive
    assert norm("ethanol—pyridine (1/1)") != norm("ethanol—pyridine (1/2)")
    assert norm("1-methylnaphthalene") != norm("2-methylnaphthalene")
