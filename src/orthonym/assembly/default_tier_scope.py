"""A producer of the best-effort tier that has a default-tier route asks the default tier's route first.

The best-effort tier publishes its tier through the context variables of ``metrics.provenance``
(``best_effort_ctx`` and the three others ``Orthonym.name`` sets), and the producers read them to turn
their wider vocabulary on. A producer whose wider reading names a part with a token that the default
tier's cascade declines makes the route the producer sits on a route that only the wider tier has, and the
route pre-empts the route the default tier takes after its decline: a molecule the default tier names
``pin_verified`` came out of the best-effort tier with another name or a lower label.

:func:`default_tier_scope` is the one home of the context a producer is asked under when it is asked the
default tier's way: the four breadth variables off, as in a default-tier call (the promotion re-run of the
default tier is not entered). ``wider_tier_ctx`` stays as published: it selects the spelling OPSIN reads at a
wider tier (``validation.twin_spellings.pin_spellings_on``), which the name of the wider tier must keep.
"""
from __future__ import annotations

import contextlib
from typing import Iterator


def wider_tier_running() -> bool:
    """True inside a call of the best-effort tier, the one whose producers widen in front of the default tier's."""
    from ..metrics.provenance import best_effort_ctx
    return bool(best_effort_ctx.get())


@contextlib.contextmanager
def default_tier_scope() -> Iterator[None]:
    """The four breadth context variables off, as in a default-tier call, and put back on exit."""
    from ..metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx, full_coverage_ctx,
                                      general_fallback_ctx)
    published = [(var, var.set(False)) for var in (general_fallback_ctx, best_effort_ctx,
                                                  allow_aromatic_general_ctx, full_coverage_ctx)]
    try:
        yield
    finally:
        for var, token in reversed(published):
            var.reset(token)
