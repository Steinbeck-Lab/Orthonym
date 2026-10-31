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
  A check may say more (DT3 step B): it returns:data:`PASS` when it read the name and decided that
  the rule holds,:data:`ABSTAIN` when it could not read what the rule is about, and None keeps its
  old meaning, "no failure, and the check does not say whether it decided" (an UNDECIDED check in
  :func:`pin_spelling_report`). A label never reads the difference: PASS, ABSTAIN and None are all
  "no failure".
*:func:`check_pin_spelling` -- every registered check on ``(mol, name)``; the list of failures.
*:func:`pin_spelling_report` -- the same run, kept check by check as PASS / FAIL / ABSTAIN /
  UNDECIDED, with the coverage of the name (``spelling.coverage``: the units of the name whose
  parent a reader models, of all the units). A pass of a check on a name no reader models is
  not evidence, and the report is what says so.

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


#: what one check said about one name (:func:`pin_spelling_report`). A check returns PASS when it read the
#: name and decided that its rule holds, ABSTAIN when it could not read what the rule is about, a
#::class:`SpellingFailure` for FAIL; None is the old contract, which does not say whether the check
#: decided (UNDECIDED).
PASS = "pass"
FAIL = "fail"
ABSTAIN = "abstain"
UNDECIDED = "undecided"


@dataclass(frozen=True)
class SpellingFailure:
    """A name breaks the Blue Book rule ``rule`` (for example ``""``); ``detail`` says how."""
    rule: str
    detail: str


#: rule id -> the check functions registered under it, in registration order
_REGISTRY: dict = {}

#: modules whose import registers the built-in checks
BUILTIN_CHECK_MODULES = (
    "orthonym.validation.spelling.checks_ring_prefix",
    "orthonym.validation.spelling.checks_hydrogen",
    "orthonym.validation.spelling.checks_marks",
    "orthonym.validation.spelling.checks_locants",
    "orthonym.validation.spelling.checks_parent",
    "orthonym.validation.spelling.checks_phane",
    "orthonym.validation.spelling.checks_assembly",
    "orthonym.validation.spelling.checks_prefixes",
    "orthonym.validation.spelling.checks_pseudohalide",
    "orthonym.validation.spelling.checks_one_carbon_parent",
    "orthonym.validation.spelling.checks_heterone",
    "orthonym.validation.spelling.checks_placement",
    "orthonym.validation.spelling.checks_identical_prefixes",
    "orthonym.validation.spelling.checks_thioperoxol",
    "orthonym.validation.spelling.checks_amine_parent",
    "orthonym.validation.spelling.checks_group_count",
    "orthonym.validation.spelling.checks_acid_cyanide",
    "orthonym.validation.spelling.checks_polyol_ester",
    "orthonym.validation.spelling.checks_chalcone",
    "orthonym.validation.spelling.checks_amide_chains",
    "orthonym.validation.spelling.checks_skeletal_chain",
    "orthonym.validation.spelling.checks_natural_products",
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


def _run(mol, name, strict: bool):
    """``[(rule_id, check_name, verdict, failure)]`` of every registered check on ``(mol, name)``.

    A check that raises is a defect of the check, not of the name: it is logged and counts as an
    abstention, so a reader bug can never lower a label by itself. ``strict`` re-raises instead."""
    _ensure_loaded()
    out = []
    for rule_id, fns in _REGISTRY.items():
        for fn in fns:
            label = getattr(fn, "__name__", repr(fn))
            try:
                res = fn(mol, name)
            except Exception:
                if strict:
                    raise
                logger.warning("spelling check %s (%s) failed on %r", rule_id, label, name, exc_info=True)
                out.append((rule_id, label, ABSTAIN, None))
                continue
            if isinstance(res, SpellingFailure):
                out.append((rule_id, label, FAIL, res))
            elif res == PASS or res == ABSTAIN:
                out.append((rule_id, label, res, None))
            else:
                out.append((rule_id, label, UNDECIDED, None))
    return out


def check_pin_spelling(mol, name: Optional[str], *, strict: bool = False) -> list:
    """Every registered check on ``(mol, name)``: the list of:class:`SpellingFailure`.

    A check that raises is a defect of the check, not of the name: it is logged and counts as an
    abstention, so a reader bug can never lower a label by itself. ``strict=True`` re-raises
    instead (the validation runs and the tests use it, so a crash is seen there). Without a
    structure or a name there is nothing to check, and no check module is loaded."""
    if mol is None or not name:
        return []
    return [failure for _r, _c, _v, failure in _run(mol, name, strict) if failure is not None]


@dataclass(frozen=True)
class CheckOutcome:
    """One check's verdict on one name (``failure`` is set for FAIL only)."""
    rule: str
    check: str
    verdict: str
    failure: Optional[SpellingFailure] = None


@dataclass(frozen=True)
class SpellingReport:
    """Every check on one name, check by check, and the coverage of the name (``coverage`` is None
    when the marks of the name do not pair, so no reader can read it)."""
    outcomes: tuple
    coverage: Optional[object]

    @property
    def failures(self) -> list:
        return [o.failure for o in self.outcomes if o.failure is not None]

    def count(self, verdict: str) -> int:
        return sum(1 for o in self.outcomes if o.verdict == verdict)


def pin_spelling_report(mol, name: Optional[str], *, strict: bool = False) -> SpellingReport:
    """:func:`check_pin_spelling` kept check by check (PASS / FAIL / ABSTAIN / UNDECIDED) with the
    coverage of ``name`` (``validation.spelling.coverage.name_coverage``). It runs the same checks the
    same way and changes no label: a name with no structure or no name has no outcome."""
    if mol is None or not name:
        return SpellingReport((), None)
    coverage = importlib.import_module("orthonym.validation.spelling.coverage")
    return SpellingReport(
        tuple(CheckOutcome(r, c, v, f) for r, c, v, f in _run(mol, name, strict)), coverage.name_coverage(name))
