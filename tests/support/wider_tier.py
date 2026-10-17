"""A wider-tier call (complete / best-effort) as the producers of the PIN spellings see it.

``validation.twin_spellings.pin_spellings_on`` reads ``wider_tier_ctx``, which ``Orthonym.name`` sets for
a call of a wider tier; a test that calls a producer directly sets it with this context manager."""
import contextlib

from orthonym.validation.twin_spellings import wider_tier_ctx


@contextlib.contextmanager
def wider_tier():
    token = wider_tier_ctx.set(True)
    try:
        yield
    finally:
        wider_tier_ctx.reset(token)
