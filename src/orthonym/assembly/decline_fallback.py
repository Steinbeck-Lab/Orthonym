"""The names tried for a caller's input only when the drawing as written gets no name.

Each fallback names another reading or drawing of the same compound and ships only what reads back to the input; its
label is systematic_verified. A row the drawing as written names is never replaced, so no name or tier of today
changes, and the default tier keeps its decline:

* ``sulfoxide_drawing``: a charge-separated S(IV) oxide, named through its S=O drawing;
* ``aziridine_nitrogen``: an aziridine nitrogen whose tag RDKit drops, cited in the name of RDKit's reading.

``assembly.halogen_drawing.halogen_drawing_scope`` (the decorator of the three public naming methods) calls
:func:`has_fallback` and:func:`fallback_row`.
"""
import contextlib
import contextvars
import logging
from typing import Optional, Tuple

#: True while a caller's input is named as drawn: a call inside that naming (``name_tiered`` calls ``name``, and names
#: the standard spelling of a lone-pair centre RDKit misreads) names its string as drawn and does not come back here,
#: so only the caller's own call takes a fallback and its row says so
_AS_DRAWN: "contextvars.ContextVar[bool]" = contextvars.ContextVar("orthonym_decline_fallback_as_drawn", default=False)

#: the ``source`` of a fallback row
SOURCES = frozenset({"sulfoxide_drawing", "aziridine_nitrogen"})
#: the tiers a name can have, best first: a fallback row never carries a better one than the row it was built on
_TIERS = ("pin_verified", "pin_unverified", "systematic_verified", "best_effort")
#: declines a fallback does not retry: the call ran out of its time
_NOT_RETRIED = frozenset({"TIME_LIMIT"})


@contextlib.contextmanager
def as_drawn():
    """While this block runs,:func:`has_fallback` is False."""
    token = _AS_DRAWN.set(True)
    try:
        yield
    finally:
        _AS_DRAWN.reset(token)


def has_fallback(smiles) -> bool:
    """True for a caller's SMILES string that a fallback applies to (cheap on any other input), outside:func:`as_drawn`."""
    if _AS_DRAWN.get() or not isinstance(smiles, str):
        return False
    from .aziridine_nitrogen import has_dropped_aziridine_nitrogen
    from .sulfoxide_drawing import has_charge_separated_sulfoxide
    return has_charge_separated_sulfoxide(smiles) or has_dropped_aziridine_nitrogen(smiles)


def capped(row: dict, inner: dict) -> dict:
    """``row`` with the lower of its tier and the tier of ``inner`` (the row of the other reading or drawing it was
    built on), and ``inner``'s spelling failures: a fallback never labels a name better than its producer did."""
    tier = row['tier']
    if inner.get('tier') in _TIERS and _TIERS.index(inner['tier']) > _TIERS.index(tier):
        tier = inner['tier']
    return {**row, 'tier': tier, 'spelling_failures': list(inner.get('spelling_failures') or [])}


def fallback_row(namer, smiles: str, written: dict) -> Optional[dict]:
    """The first fallback row for ``smiles`` given the row of the drawing as written, or None (also when a fallback
    fails: the decline of the drawing as written stands)."""
    from .aziridine_nitrogen import aziridine_nitrogen_row, has_dropped_aziridine_nitrogen
    from .sulfoxide_drawing import has_charge_separated_sulfoxide, sulfoxide_drawing_row
    if written.get('tier') != 'abstain' or written.get('limit_code') in _NOT_RETRIED:
        return None
    try:
        if has_charge_separated_sulfoxide(smiles):
            row = sulfoxide_drawing_row(namer, smiles, written)
            if row is not None:
                return row
        if has_dropped_aziridine_nitrogen(smiles):
            return aziridine_nitrogen_row(namer, smiles, written)
    except Exception:  # noqa: BLE001 -- a fallback that fails is no name
        logging.getLogger(__name__).debug("decline fallback failed for %s", smiles, exc_info=True)
    return None


def named_or_fallback(namer, kind: str, fn, smiles: str, args, kwargs) -> Tuple[bool, object]:
    """Name ``smiles`` as drawn through ``fn`` (``kind`` is its name: ``name``, ``name_tiered`` or
    ``name_with_confidence``) and return ``(True, result)`` when it is named, or when no fallback gives a name; for a
    fallback row, ``(False, row)``, which the caller turns into the result ``fn`` gives (``name_tiered`` returns it).
    The drawing as written is named once; the fallback runs only after its decline."""
    from ..errors import OrthonymLimitError
    limit = None
    with as_drawn():
        try:
            out = fn(namer, smiles, *args, **kwargs)
        except OrthonymLimitError as exc:        # ``name(..., raise_on_limit=True)``
            out, limit = None, exc
    if kind == 'name_tiered':
        written = out
    elif kind == 'name_with_confidence':
        written = {'tier': 'abstain' if out.get('tier') is None else out['tier'],
                   'limit_code': (out.get('limit') or {}).get('code')}
    else:
        from ..errors import is_failure_name
        declined = limit is not None or is_failure_name(out)
        written = {'tier': 'abstain' if declined else 'named',
                   'limit_code': getattr(limit, 'code', None)}
    row = fallback_row(namer, smiles, written)
    if row is None:
        if limit is not None:
            raise limit
        return True, out
    return (True, row) if kind == 'name_tiered' else (False, row)
