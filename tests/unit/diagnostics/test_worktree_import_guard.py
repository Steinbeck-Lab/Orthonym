"""A bare pytest in a git worktree must not test the main tree's code.

``tests/conftest.py::_worktree_import_guard``: the shared.venv is an editable install of the main
tree, so in a worktree `import orthonym` finds main's src unless PYTHONPATH names the worktree's.
The guard stops the run then, and is silent where `.git` is not a file (the main tree, CI, a
published copy).
"""
import pytest

import orthonym
from tests.conftest import _worktree_import_guard

pytestmark = pytest.mark.unit


def test_a_tree_whose_git_is_not_a_file_is_not_checked(tmp_path):
    (tmp_path / ".git").mkdir()
    assert _worktree_import_guard(tmp_path) is None
    assert _worktree_import_guard(tmp_path / "published") is None      # no.git at all


def test_a_worktree_that_imports_another_trees_code_stops_the_run(tmp_path):
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n")
    with pytest.raises(pytest.exit.Exception, match="PYTHONPATH="):
        _worktree_import_guard(tmp_path)


def test_a_worktree_that_imports_its_own_src_passes(tmp_path, monkeypatch):
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n")
    monkeypatch.setattr(orthonym, "__file__", str(tmp_path / "src" / "orthonym" / "__init__.py"))
    assert _worktree_import_guard(tmp_path) is None
