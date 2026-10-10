"""Spelling checks of a pin_verified name, independent of the round trip.

A round trip proves the molecule, never the spelling: OPSIN reads ``pyrido[1,2-b]pyridazin-6-one``
and even ``9H-pyrido[1,2-b]pyridazin-6-one`` back to the same structure as the PIN
``6H-pyrido[1,2-b]pyridazin-6-one``. The label ``pin_verified`` also claims that the spelling is
the Blue Book PIN form. Each check here decides one spelling rule from two inputs only: the
structure (an RDKit Mol) and the name read lexically. A check never calls the engine's naming
code or OPSIN, so it cannot inherit a defect of the code that built the name.

Interface (binding for every lane that adds a check):

*:class:`SpellingFailure` -- ``rule`` (the Blue Book rule id the name breaks) and ``detail``.
*:func:`register` -- ``@register("")`` on a function ``fn(mol, name) -> SpellingFailure | None``.
  Several functions may share one rule id. A function returns a failure only when it has decided
  that the name breaks its rule; pass and abstain (a name or structure the reader cannot read with
  certainty) both return None.
*:func:`check_pin_spelling` -- every registered check on ``(mol, name)``; the list of failures.

The built-in checks live in the modules of:data:`BUILTIN_CHECK_MODULES`, imported on first use.
A lane adds a check by adding its module there.

The caller (``Orthonym._tier_row_and_pin_form``) runs the checks on a name it would label
``pin_verified``; a non-empty result lowers the label (the default tier then declines the name,
the wider tiers keep it as ``systematic_verified``) and never changes the name.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpellingFailure:
    """A name breaks the Blue Book rule ``rule`` (for example ``""``); ``detail`` says how."""
    rule: str
    detail: str


#: rule id -> the check functions registered under it, in registration order
_REGISTRY: dict = {}

#: modules whose import registers the built-in checks
BUILTIN_CHECK_MODULES = (
    "orthonym.validation.spelling.checks_hydrogen",
    "orthonym.validation.spelling.checks_marks",
    "orthonym.validation.spelling.checks_locants",
    "orthonym.validation.spelling.checks_parent",
    "orthonym.validation.spelling.checks_phane",
    "orthonym.validation.spelling.checks_assembly",
)

_loaded = False


def register(rule_id: str):
    """Decorator: register ``fn(mol, name) -> SpellingFailure | None`` under ``rule_id``."""
    def deco(fn: Callable):
        fns = _REGISTRY.setdefault(rule_id, [])
        if fn not in fns:
            fns.append(fn)
        return fn
    return deco


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    for mod in BUILTIN_CHECK_MODULES:
        importlib.import_module(mod)
    _loaded = True


def registered_rules() -> tuple:
    """The rule ids that have at least one check."""
    _ensure_loaded()
    return tuple(_REGISTRY)


def check_pin_spelling(mol, name: Optional[str], *, strict: bool = False) -> list:
    """Every registered check on ``(mol, name)``: the list of:class:`SpellingFailure`.

    A check that raises is a defect of the check, not of the name: it is logged and counts as an
    abstention, so a reader bug can never lower a label by itself. ``strict=True`` re-raises
    instead (the validation runs and the tests use it, so a crash is seen there). Without a
    structure or a name there is nothing to check, and no check module is loaded."""
    if mol is None or not name:
        return []
    _ensure_loaded()
    out = []
    for rule_id, fns in _REGISTRY.items():
        for fn in fns:
            try:
                res = fn(mol, name)
            except Exception:
                if strict:
                    raise
                logger.warning("spelling check %s (%s) failed on %r", rule_id,
                               getattr(fn, "__name__", fn), name, exc_info=True)
                continue
            if res is not None:
                out.append(res)
    return out
