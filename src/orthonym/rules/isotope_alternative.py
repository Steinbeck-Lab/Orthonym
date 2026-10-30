"""The second skeleton the isotope decorator is given when the first one takes no descriptor.

The decorator names the label-free skeleton of a labelled molecule and splices the isotopic descriptor into that
name; every candidate is read back with the labels (``rules.isotopes._isotope_round_trips``). Two producers give
skeleton names that no candidate can pass, although another producer names the same skeleton in a way that can:

- a member of a hydrocarbon series whose rings OPSIN draws in the other Kekule set than the input
  (``tetraphenylene``: OPSIN puts the double bonds in the eight-membered ring), which the read-back refuses
  for a labelled name (``_kekule_forms_of_one_compound``, a 4n circuit is two compounds);
- a senior mononuclear hub, ``bis[...]di(phenyl)silane``) whose like substituents the labels make unlike,
  so that no multiplied spelling can carry the descriptor.

The namer then runs the decorator once more inside ``alternative_skeleton``, where those two producers make no
offer and the skeleton gets the name the next producer gives. Outside that block nothing changes.
"""
import contextlib
import contextvars

_ACTIVE: "contextvars.ContextVar[bool]" = contextvars.ContextVar("orthonym_isotope_alternative_skeleton",
                                                                 default=False)


@contextlib.contextmanager
def alternative_skeleton():
    """Name the skeleton without the series and senior-hub producers while this block runs."""
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def has_alternative(mol) -> bool:
    """True when ``mol`` (the label-free skeleton) has a producer this scope withholds: a series member or a
    senior mononuclear hub. Otherwise the second pass would give the skeleton the name it had."""
    from .hydrocarbon_series import identify_series
    from .mononuclear_hydrides import _BELOW_HUB, _scan_senior_hubs
    try:
        return identify_series(mol) is not None or bool(_scan_senior_hubs(mol, _BELOW_HUB))
    except Exception:  # noqa: BLE001 -- a skeleton these readers refuse has no alternative here
        return False


def alternative_skeleton_active() -> bool:
    """True inside:func:`alternative_skeleton` (the memo keys its caches apart on it)."""
    return _ACTIVE.get()
