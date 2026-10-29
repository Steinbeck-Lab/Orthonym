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
