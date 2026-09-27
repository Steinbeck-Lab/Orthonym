"""Skip a test that reads a local-only file when that file is absent.

``docs/`` and ``a temp dir/`` are gitignored (``.gitignore:110-111``; the whole
``docs/`` folder is kept local by decision, 5229d80c3) and most of ``.planning/``
is untracked, so a clean checkout -- a git worktree, CI, the published tree --
does not have them. A test that reads such a file used to raise
FileNotFoundError there (TRIAGE g7 C02). The old guard in ``tests/conftest.py``
keyed on ``.planning/`` being absent as a proxy for "not the dev tree", which
stopped working once part of ``.planning/`` became tracked.

Key the skip on the file itself::

    from tests.support.local_only import local_only

    @local_only("docs/cip_known_limitations.md")
    def test_reads_the_doc:...

The test runs wherever the file exists (the dev tree) and is skipped, with the
missing path in the reason, where it does not. It is never a blanket skip.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def local_only(*relpaths: str):
    """``skipif`` marker for a test that needs the given repo-relative files."""
    missing = [p for p in relpaths if not (REPO_ROOT / p).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=("local-only file absent from this checkout (gitignored / untracked, "
                f"not published): {', '.join(missing)}"))
