"""a phase: _should_bypass_fused_guard is deleted.

Per a phase internal notes, the function is removed entirely (no
feature-flagged code path). The ORTHONYM_USE_V18_WEIGHTS env var is the
documented rollback mechanism (a phase). This file is the
import-failure regression scaffold — if any future change reintroduces
the bypass it must also delete this test.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html cascade
Source: AUTONOM 1990 (Wisniewski J. Chem. Inf. Comput. Sci. 30, 324-332)
        — full seniority cascade on ALL structures, no bypass.
Source: a phase internal notes (delete bypass entirely; no feature flag).
"""
import pytest


def test_bypass_function_is_deleted():
    """: importing the deleted bypass must raise ImportError.

    Per a phase internal notes, ``_should_bypass_fused_guard`` is removed
    entirely (no feature-flagged code path). The ORTHONYM_USE_V18_WEIGHTS
    env var is the documented rollback mechanism (a phase).
    """
    with pytest.raises(ImportError):
        from orthonym.namer import _should_bypass_fused_guard  # noqa: F401
