"""Reload modules for the length of one test, then put them back.

A few integration tests reload ``orthonym.assembly.coverage_scoring`` and
``orthonym.assembly.candidate_pool`` so that the modules' import-time reads of
``ORTHONYM_USE_V18_WEIGHTS`` and ``ORTHONYM_SELECTION_MODE`` see the test's
environment. ``importlib.reload`` re-executes a module inside the SAME module
object, so the reloaded constants (the V18 factor weights) and the new class
objects stay there after the test; ``monkeypatch`` restoring the environment does
not undo them. Every later test on that pytest worker then named with the V18
weights -- the canary oxime row '(3Z,6E)-2,4,4,7-tetramethylnona-6,8-dien-3-one
oxime' became '...-3-oxime' -- and saw another ``CandidateName`` class, so
``isinstance`` checks against the class it imported failed (TRIAGE 'Canary oxime --
test-order flake').

``reloaded`` keeps the reload for the body of the test and afterwards puts each
module's original namespace back into the module object, so the rest of the run
sees exactly the objects it imported: the default constants and the original
classes and functions. The functions defined in the module read their globals from
that same namespace, so they see the restored values too. ``tests/conftest.py``
errors any test that reloads an orthonym module and leaves it changed.
"""
from __future__ import annotations

import contextlib
import importlib
from types import ModuleType
from typing import Iterator


@contextlib.contextmanager
def reloaded(*modules: ModuleType) -> Iterator[None]:
    """Reload ``modules`` in order for the body; restore their namespaces on exit.

    Set the environment the reload should read (``monkeypatch.setenv``) before
    entering. The namespaces are restored in reverse order even when the reload or
    the body raises.
    """
    saved = [(module, dict(vars(module))) for module in modules]
    try:
        for module in modules:
            importlib.reload(module)
        yield
    finally:
        for module, namespace in reversed(saved):
            live = vars(module)
            live.clear()
            live.update(namespace)
