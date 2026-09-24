"""Test-side access to the pinned OPSIN / centres jars.

The jars are no longer in the repository; ``orthonym.jars.find_jar`` is the
only way to locate them. Tests use one of two helpers:

*:func:`jar_or_skip` -- inside a test: the jar path, or SKIP the test.
*:func:`jar_or_none` -- for module-level constants and legacy
  ``if jar is None: skip`` guards: the jar path, or None.

Both refuse to hide a missing jar when ``ORTHONYM_REQUIRE_JARS=1``: the test
(or, at import time, the module's collection) FAILS instead, so a run that is
meant to exercise the jars can never go green-but-blind by skipping them.
"""

import os
from typing import Optional

import pytest

REQUIRE_JARS = os.environ.get("ORTHONYM_REQUIRE_JARS", "").strip().lower() in ("1", "true", "yes", "on")


def jar_unavailable(reason: str) -> None:
    """Skip the current test for ``reason`` -- or fail it under ORTHONYM_REQUIRE_JARS=1."""
    if REQUIRE_JARS:
        pytest.fail(f"{reason} (ORTHONYM_REQUIRE_JARS=1: a missing jar is a failure, not a skip)",
                    pytrace=False)
    pytest.skip(reason)


def _resolve(kind: str):
    """(path, reason): the jar path, or None plus why it is unavailable."""
    from orthonym.jars import JarUnavailable, find_jar
    try:
        path = find_jar(kind)
    except JarUnavailable as exc:
        return None, f"{kind} jar unavailable: {exc}"
    if path is None:
        return None, f"{kind} jar unavailable (ORTHONYM_ALLOW_REDUCED=1)"
    return path, ""


def jar_or_none(kind: str = "opsin") -> Optional[str]:
    """Resolved jar path, or None when unavailable -- FAILS under ORTHONYM_REQUIRE_JARS=1."""
    path, reason = _resolve(kind)
    if path is None and REQUIRE_JARS:
        pytest.fail(f"{reason} (ORTHONYM_REQUIRE_JARS=1)", pytrace=False)
    return path


def jar_or_skip(kind: str = "opsin") -> str:
    """Resolved jar path; skip the test when unavailable, or FAIL when ORTHONYM_REQUIRE_JARS=1."""
    path, reason = _resolve(kind)
    if path is None:
        jar_unavailable(reason)
    return path
