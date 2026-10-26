"""How much of a name a reader of the spelling checks can model (DT3 step B, ``pin_spelling_report``).

A spelling check abstains on a parent it does not know (:func:`.models.ring_or_chain_model` models an
unbranched chain, a carbocyclic, retained or Hantzsch-Widman monocycle, naphthalene and anthracene), so
"no spelling failure" on a name whose parents no reader models says nothing about its numbering, its order
of prefixes or its locants. The coverage of a name is the share of its units (a unit is the parent
structure or a substituent group in its own enclosing marks,, the Blue Book) whose parent
a reader models. A name whose every unit is decided is *fully covered*; only for such a name is a pass of
the numbering checks a statement about the name and not about the part of it the checks could read.

No naming code and no OPSIN: the name is read lexically (``models.read_units``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import P10, read_units, ring_or_chain_model


@dataclass(frozen=True)
class NameCoverage:
    """``units`` parent-bearing units of the name, ``decided`` of them modelled by a reader;
    ``unmodelled`` the parent tokens of the others; ``p10`` the name carries a lexical mark of a
    Chapter class (the substitutive checks abstain on it, ``models.P10``)."""
    units: int
    decided: int
    unmodelled: tuple
    p10: bool

    @property
    def full(self) -> bool:
        """Every unit is modelled, the name has at least one unit, and it is no name."""
        return self.units > 0 and self.decided == self.units and not self.p10

    @property
    def fraction(self) -> float:
        return self.decided / self.units if self.units else 0.0


def name_coverage(name: Optional[str]) -> Optional[NameCoverage]:
    """The coverage of ``name``, or None when its enclosing marks do not pair (no reader reads it)."""
    if not name:
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, units = parsed
    total = decided = 0
    unmodelled = []
    for p in units:
        if not p.parent:                # the content of an enclosure with no parent token is text
            continue
        total += 1
        enclosed = p.unit.encl is not None
        model = _model(p, enclosed)
        if model is None and enclosed:
            model = _model(p, False)
        if model is not None:
            decided += 1
        else:
            unmodelled.append(p.parent)
    return NameCoverage(total, decided, tuple(unmodelled), bool(P10.search(name)))


def _model(p, enclosed: bool):
    try:
        return ring_or_chain_model(p, enclosed)
    except Exception:  # noqa: BLE001 - a reader defect is a unit nobody decides
        return None
