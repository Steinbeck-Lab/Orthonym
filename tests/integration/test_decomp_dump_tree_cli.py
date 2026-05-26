"""Phase 160 Plan-04 integration tests: --dump-tree CLI end-to-end.

Per CONTEXT D-19 + DECOMP-04: ``python -m orthonym --dump-tree <smi>``
emits the NameTreeNode IR (text or JSON format). This file invokes the
CLI via subprocess and verifies the output is parseable + correct.

Test coverage:
- text format: --dump-tree CCO produces parseable indented tree.
- json format: --dump-tree --format json CCO produces parseable JSON.
- name field matches programmatic Orthonym.name_with_tree(smi).name.
- tree=None case: legacy fragment_legacy renderer is exercised.
- tree-populated case: explicit-field renderer (currently no extracted
  handlers emit trees per CONTEXT D-05 first-wave; will be added when
  v19+1 handlers ship).
- error handling: invalid SMILES exits non-zero with stderr message.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from orthonym import Orthonym


def _run_cli(args, expected_returncode=0):
    """Helper: run ``python -m orthonym`` with args; return (stdout, stderr, rc).

    Sets PYTHONPATH=src to use the worktree codebase (matches the canary
    harness pattern).
    """
    import os
    env = os.environ.copy()
    cwd = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # Ensure the worktree src/ is on the path so the CLI uses our code.
    env["PYTHONPATH"] = os.path.join(cwd, "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "orthonym"] + args,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )
    return result.stdout, result.stderr, result.returncode


def test_dump_tree_text_format_simple():
    """--dump-tree CCO emits text-format tree to stdout."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "CCO"])
    assert rc == 0, f"stderr={stderr!r}"
    # Text format header line includes 'NameTree:' + name
    assert "NameTree" in stdout
    assert "ethanol" in stdout


def test_dump_tree_text_renders_populated_tree():
    """Phase 165: CCO routes to general_acyclic which now emits a populated tree;
    the text dump renders parent_stem with NO first-wave tree=None placeholder."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "CCO"])
    assert rc == 0
    assert "parent_stem" in stdout
    assert "first-wave" not in stdout.lower()


def test_dump_tree_json_format_parseable():
    """--dump-tree --format json CCO emits parseable JSON."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "--format", "json", "CCO"])
    assert rc == 0, f"stderr={stderr!r}"
    data = json.loads(stdout)
    assert "name" in data
    assert "tree" in data
    assert "atom_to_locant_hint" in data


def test_dump_tree_json_name_field():
    """JSON dump's name field matches programmatic name_with_tree().name."""
    stdout, _stderr, rc = _run_cli(
        ["--dump-tree", "--format", "json", "CCO"],
    )
    assert rc == 0
    data = json.loads(stdout)
    namer = Orthonym()
    expected = namer.name_with_tree("CCO").name
    assert data["name"] == expected


def test_dump_tree_json_tree_populated():
    """Phase 165: CCO emits a populated tree; JSON dump renders it as a dict
    with the structured fields."""
    stdout, _stderr, rc = _run_cli(
        ["--dump-tree", "--format", "json", "CCO"],
    )
    assert rc == 0
    data = json.loads(stdout)
    assert isinstance(data["tree"], dict)
    assert "parent_stem" in data["tree"]


def test_dump_tree_text_format_default():
    """Default format (no --format arg) is text."""
    stdout_default, _, rc1 = _run_cli(["--dump-tree", "CCO"])
    stdout_text, _, rc2 = _run_cli(
        ["--dump-tree", "--format", "text", "CCO"],
    )
    assert rc1 == 0 and rc2 == 0
    assert stdout_default == stdout_text


def test_dump_tree_complex_smiles_text():
    """Non-trivial SMILES (cyclohexanol) renders correctly."""
    stdout, _, rc = _run_cli(["--dump-tree", "OC1CCCCC1"])
    assert rc == 0
    # Output contains 'NameTree' + the produced name (cyclohexanol or similar).
    assert "NameTree" in stdout


def test_dump_tree_invalid_smiles_exit_nonzero():
    """Invalid SMILES should produce a non-zero return code."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "X!Y!Z?ABC"])
    assert rc != 0
    # Error message is on stderr.
    assert "Error" in stderr or "error" in stderr.lower()


def test_dump_tree_help():
    """--help mentions the --dump-tree flag."""
    stdout, _stderr, rc = _run_cli(["--help"])
    assert rc == 0
    assert "--dump-tree" in stdout


def test_dump_tree_format_help():
    """--help mentions the --format flag."""
    stdout, _stderr, rc = _run_cli(["--help"])
    assert rc == 0
    assert "--format" in stdout


def test_dump_tree_recursive_multiprefix_text():
    """Phase 165 D-04: a multi-prefix molecule renders nested prefixes[] subtrees
    (the recursive renderer emits a parent_stem line per node)."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "CC(C)CC(C)CO"])
    assert rc == 0, f"stderr={stderr!r}"
    # Root + >=1 prefix subtree -> at least 2 parent_stem lines.
    assert stdout.count("parent_stem") >= 2, stdout


def test_dump_tree_json_multiprefix_no_typeerror():
    """Pitfall 6: JSON recursion over a multi-prefix tree survives — str
    fragment_legacy is serializable, no NameFragment TypeError."""
    stdout, stderr, rc = _run_cli(
        ["--dump-tree", "--format", "json", "CC(C)CC(C)CO"],
    )
    assert rc == 0, f"stderr={stderr!r}"
    data = json.loads(stdout)
    assert isinstance(data["tree"], dict)


def test_dump_tree_any_smiles_ion():
    """SC-3: --dump-tree works for ANY SMILES including ions (boundary coarse
    fallback) — text + JSON, no crash."""
    stdout, stderr, rc = _run_cli(["--dump-tree", "CC(=O)[O-]"])
    assert rc == 0, f"stderr={stderr!r}"
    assert "NameTree" in stdout
    stdout_j, _stderr_j, rc_j = _run_cli(
        ["--dump-tree", "--format", "json", "CC(=O)[O-]"],
    )
    assert rc_j == 0
    data = json.loads(stdout_j)
    assert isinstance(data["tree"], dict)
