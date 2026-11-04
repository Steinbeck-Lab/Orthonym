"""A wider tier gives the PIN tier's name for a molecule its own naming names in a lower class.

The default of the PIN style is a cascade (user decision 2026-10-25): the PIN, then a valid, a complete and a
best-effort name. The wider tiers name the molecule in one pass of their own, with the breadth producers in it, so
a producer of a wider tier can name the molecule in a lower class where the PIN tier builds the PIN. A ring parent
of the ring cascade cites a principal characteristic group that sits on no ring as a prefix:
'4-[(1Z)-2-formamidoethen-1-yl]phenol' for 'N-[(Z)-2-(4-hydroxyphenyl)ethenyl]formamide': the parent has
the maximum number of principal characteristic groups; the amide is senior to the alcohol,.

Asking the PIN tier for every molecule is no option: it costs 39 % to 78 % more time on the evaluation sets, most
of it on the molecules the PIN tier declines. The producer that names a molecule in such a lower class notes it
(:func:`note_pin_tier_asked`), and only such a molecule is named by the PIN tier (``default_tier_engine``, the
engine of the strict twin) after the wider naming. The PIN tier's row is taken when it is ``pin_verified`` and its
own string read back (``opsin`` ``verified``). A PIN that only a twin spelling verifies (``verified_twin``:
'propane-1,2-diyl... diester' names, 'methanaminiumyl') is not taken: the wider tiers ship the name OPSIN reads
(user decision 2026-10-22, "Best-effort ships readable"). Every other molecule keeps the wider tier's name.

:func:`pin_tier_checked` is the step at the three entry points (``name``, ``name_tiered``,
``name_with_confidence``). The outermost one watches the naming; the entry points it calls on the way run as before.
The PIN tier's naming runs in a copy of the context, so nothing it records reaches the wider naming's state.
"""
import contextvars
from typing import Callable, Optional

#: The watch of the outermost entry point: a one-item list (True once a producer asked for the PIN tier), else None
_WATCH: contextvars.ContextVar = contextvars.ContextVar("orthonym_pin_tier_watch", default=None)


def note_pin_tier_asked() -> None:
    """A producer of a wider tier named the whole molecule in a lower class than the PIN tier may (see the module
    docstring): the outermost entry point asks the PIN tier for its name."""
    watch = _WATCH.get()
    if watch is not None:
        watch[0] = True


def _applies(namer) -> bool:
    from ..namer import _BOOK_RETRY_OWNED
    from ..tiers import verification_active
    from .fragment_naming import name_scope_depth
    return (getattr(namer, "style", None) == "pin"
            and any(getattr(namer, flag, False) for flag in (
                "_general_fallback", "_general_fallback_unverified", "_allow_aromatic_general", "_full_coverage"))
            and getattr(namer, "_principal_group_override", None) is None and not getattr(namer, "_forced_parent_rank", 0)
            and _WATCH.get() is None and not _BOOK_RETRY_OWNED.get()
            and name_scope_depth() == 0 and verification_active())


def _pin_tier_row(namer, smiles) -> Optional[dict]:
    """The PIN tier's row of ``smiles`` when it is a PIN read back directly; else None."""
    from .default_tier_offer import default_tier_engine
    tw = getattr(namer, "_pin_twin", None)
    if tw is None:
        tw = namer._pin_twin = default_tier_engine(namer)
    try:
        row = contextvars.copy_context().run(tw.name_tiered, smiles)
    except Exception:  # noqa: BLE001 - the PIN tier's failure is no row: the wider tier's name stands
        return None
    if row.get("tier") == "pin_verified" and row.get("opsin") == "verified":
        return row
    return None


def pin_tier_checked(namer, smiles, run: Callable, pick: Callable):
    """``run``, the wider tier's naming of ``smiles``; when a producer asked for the PIN tier in it
    (:func:`note_pin_tier_asked`) and the PIN tier names the molecule with a PIN read back directly,
    ``pick(row)`` of that row instead (unless ``pick`` gives None). Inside an outer entry point, and for every naming the module does not
    apply to, ``run`` alone."""
    if not _applies(namer):
        return run()
    watch = [False]
    token = _WATCH.set(watch)
    try:
        result = run()
    finally:
        _WATCH.reset(token)
    if not watch[0]:
        return result
    from ..input_mol import handles, smiles_for
    if not isinstance(smiles, str) and handles(smiles):
        smiles = smiles_for(smiles)
    row = _pin_tier_row(namer, smiles)
    picked = None if row is None else pick(row)
    return result if picked is None else picked


def pin_tier_confidence(namer, smiles: str, row: dict) -> dict:
    """The record of ``name_with_confidence`` for the PIN tier's row ``row``: the PIN tier's own record, with the row's
    label (None, the wider tier's record stands, when the record names the molecule otherwise)."""
    metadata = namer._pin_twin.name_with_confidence(smiles)
    if not isinstance(metadata, dict) or metadata.get('name') != row.get("name"):
        return None
    metadata['tier'] = row.get("tier")
    return metadata
