"""Slots of a name where an isotope descriptor can be inserted that ``rules.isotopes._insertion_offsets`` offers
besides the start of every alphabetic run (IUPAC 2013,.

 (the Blue Book): the descriptor "is inserted before the part of the compound that is
isotopically substituted", and "when the name, or a part of a name, includes a preceding locant, a hyphen is
inserted". Every candidate built from a slot is read back by OPSIN before it is used, so a slot that offers
too much costs a parse and nothing else.
"""
import re

from ..assembly.naming_utils import _REPLACEMENT_FRONT_RE as _REPLACEMENT_STEM_RE

#: The heteroatom locants of a Hantzsch-Widman ring opening a SUBSTITUENT's name inside its enclosing marks, the
#: ``1,3-`` of ``(1,3-dioxolan-2-yl)`` or of ``(1,3-thiazol-2-yl)``: as for the benzo-fused rings
#: (``rules.isotopes._BRACKET_BENZO_LOCANT_RE``) the locant set belongs to the ring, and a nuclide on it takes the
#: slot right after the mark, hyphen-joined: ``{2-[(2-13C)-1,3-dioxolan-2-yl]ethenyl}``. The index after the
#: mark is the slot.
BRACKET_HW_LOCANT_RE = re.compile(
    r"[(\[{](?=\d+(?:,\d+)*-" + _REPLACEMENT_STEM_RE.pattern[1:] + r")")

#: The start of a later component of a multi-part name that opens with a locant: the ``1H-`` of
#: ``ethanol—1H-imidazole (1/1)`` adducts, salts). The alphabetic slots of
#: ``rules.isotopes._insertion_offsets`` start inside the stem and the indicated-hydrogen slot needs a hyphen before
#: it, so a nuclide on such a component had no slot: ``ethanol—(2-2H)-1H-imidazole (1/1)`` named nothing although
#: ``(2-2H)-1H-imidazole`` alone was named.: the name is formed by "inserting the nuclide symbol(s)... before the part of the
#: compound that is isotopically substituted" (the Blue Book); the hyphen before the locant is added afterwards, as for
#: the whole-name front slot ('(15N)-1H-indole (PIN)',:43790). The index of the digit is the slot.
COMPONENT_LOCANT_START_RE = re.compile(r"(?<=[—\s])(?=\d)")

#: a nuclide descriptor that closes a component's label: '(4-2H)', '(13C)', '(2H3)', '(1,1-2H2)'
_NUCLIDE_RE = re.compile(r"\d+(?:H|He|Li|B|C|N|O|F|Na|Si|P|S|Cl|Se|Br|I)\d*\)")


def unlabelled_components_kept(name: str, skeleton: str, original) -> str:
    """``name`` (re-rendered with every locant cited, with each component that carries no nuclide spelled
    as in the unmodified ``skeleton``: (the Blue Book) "if isotopic modification requires a locant
    to specify its position, then all locants must be specified" concerns the compound that is modified, and the other
    components of an adduct keep their names ('ethanol—(4-2H)-1H-imidazole (1/1)', not 'ethan-1-ol—'). Kept only when
    the two names have the same components in the same order and OPSIN reads the result back to ``original``;
    otherwise ``name`` is returned and recorded as no PIN."""
    parts, plain = name.split("—"), skeleton.split("—")
    if len(parts) < 2:
        return name
    if len(parts) == len(plain):
        kept = "—".join(p if _NUCLIDE_RE.search(p) else q for p, q in zip(parts, plain))
        if kept == name:
            return name
        from .isotopes import _isotope_round_trips
        if _isotope_round_trips(kept, original):
            return kept
    from ..metrics.provenance import record_non_pin_label
    record_non_pin_label(name, rule="P-82.6.1.1",
                         detail="a component that carries no nuclide is spelled with locants its unmodified name omits")
    return name
