"""A substituent prefix name that carries how its writer built it.

 (c) (the Blue Book): "any component which is substituted automatically
requires use of the multiplicative forms 'bis', 'tris', etc." Whether a prefix is
substituted is a fact of the derivation its writer chose ('tert-butyl' is simple,
'2-methylpropan-2-yl' is substituted; 'bicyclo[3.2.1]octan-3-yl' is simple although it
holds brackets, (f):7104), so the writer records it with the name it returns.

The value is a ``str`` subclass: it equals, hashes, sorts and serialises as its text, so
dict keys, sort keys, JSON and the memo are unchanged. Pickle and ``copy`` keep the
record; any string operation (slice, ``strip``, f-string, concatenation) returns a plain
``str`` without it, and a name without a record is decided as before
(``derivation_of`` returns ``None``). A writer step that adds marks or a descriptor to a
recorded name keeps the record with:func:`carried`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["PrefixDerivation", "PrefixName", "built", "carried", "derivation_of",
           "substituted_prefix", "with_derivation"]


@dataclass(frozen=True)
class PrefixDerivation:
    """``substituted``: the writer attached at least one detachable substituent prefix to
    the parent group of this prefix (c), (a); compound and complex
    groups, /.
    ``enclosed``: the prefix takes enclosing marks wherever it is cited,
    the Blue Book), as its writer decided; ``None`` leaves the marks to the
    predicates of ``naming_utils``."""

    substituted: bool
    enclosed: Optional[bool] = None


class PrefixName(str):
    """The text of a substituent prefix and the ``PrefixDerivation`` of its writer."""

    def __new__(cls, text: str, derivation: PrefixDerivation):
        obj = super().__new__(cls, text)
        obj.derivation = derivation
        return obj

    def __getnewargs__(self):
        return (str(self), self.derivation)


def built(text: Optional[str], *, substituted: bool,
          enclosed: Optional[bool] = None) -> Optional[str]:
    """``text`` as returned by its writer, with the writer's facts; ``None`` stays None."""
    if text is None:
        return None
    return PrefixName(text, PrefixDerivation(substituted=substituted, enclosed=enclosed))


def retained_benzyl(mol, ring_name: str, methylene_atoms) -> bool:
    """True when a group of ``ring_name`` on the carbons ``methylene_atoms`` is written with the retained prefix
    'benzyl': the bare phenyl ring, and no label on the methylene. under ' Location of nuclides
    on positions not normally denoted by locants' (the Blue Book): "When the nuclide is located at a position
    in a retained name that is not numbered a systematic name that identifies separately the relevant atom is used
    for the IUPAC preferred name" ('[phenyl(13C)methyl]hydrazine (PIN)',:44259); the retained 'benzyl' numbers no
    position of its methylene, so a label there takes 'phenylmethyl'."""
    from .locant_omission import label_on_unnumbered_position
    return ring_name == 'phenyl' and not label_on_unnumbered_position(mol, methylene_atoms)


def with_derivation(text: Optional[str], rec: Optional[PrefixDerivation]) -> Optional[str]:
    """``text`` with the record ``rec`` (unchanged when either is ``None``)."""
    if text is None or rec is None:
        return text
    return PrefixName(text, rec)


def derivation_of(name) -> Optional[PrefixDerivation]:
    """The writer's record of ``name``, or ``None`` for a name without one."""
    return getattr(name, "derivation", None)


def carried(text: Optional[str], like) -> Optional[str]:
    """``text`` is ``like`` with marks or a descriptor added by the same writer chain: it
    keeps the record of ``like`` (none when ``like`` has none)."""
    rec = derivation_of(like)
    if rec is None or text is None or derivation_of(text) is not None:
        return text
    return PrefixName(text, rec)


def substituted_prefix(text: Optional[str], step) -> Optional[str]:
    """``step(token)`` for ``text`` recorded as substituted: a ring prefix with decorations is a
    substituted component whatever stem its spelling ends in (c), the Blue Book), and
    the multiplier of a repeated copy reads this record and not the spelling; the record is kept
    through ``step`` (a stereodescriptor prepended, marks added)."""
    token = built(text, substituted=True)
    return carried(step(token), token)
