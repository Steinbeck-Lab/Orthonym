"""Phase 160 Plan-04 per-handler test STUB for ``benzene`` (DEFERRED).

This handler is one of the 8 NOT YET EXTRACTED handlers per Plan-02 +
Plan-03 honest-fail (CONTEXT D-27): byte-identical extraction requires
architectural changes beyond Phase 160 scope.

Per ADR-19-02: a future v19.x follow-up plan must:
  - Extract the handler from composer.py:_assemble_name_impl inline branch.
  - Add an InnerDispatchEntry registration at the appropriate priority.
  - Update this test file to exercise the real handler.

Until then, every test in this file is SKIPPED with a loud signal so the
test pyramid reports the gap clearly (test pyramid passes; pytest
collection still finds the file).
"""
from __future__ import annotations

import pytest


SKIP_REASON = (
    "Plan-160-FOLLOWUP: handler 'benzene' not yet extracted to "
    "src/orthonym/assembly/handlers/benzene.py per CONTEXT D-27 "
    "honest-fail; see ADR-19-02 for the architectural blocker and the "
    "v19.x extraction plan."
)


@pytest.mark.skip(reason=SKIP_REASON)
def test_handler_extracted():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_predicate_signature():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_handler_signature():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_inner_dispatch_registration():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_byte_identical_name_via_namer():
    pass
