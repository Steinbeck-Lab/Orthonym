"""A multiplied unit named by an acyl group and a class word: the class word is multiplied too.

The multiplicative producer names the unit of a bridge as a free molecule ('benzoyl chloride',
'benzenesulfonyl cyanide') and multiplies that name. A unit that is an acid is one word to
multiply ('benzoic acid' -> '4,4'-oxydibenzoic acid',, the Blue Book), but the
name of the halide or pseudohalide of an acid is two parts, the acyl group and the class word:

 "Acyl halides from suffix acids" (the Blue Book, text:31414): the acyl group
"followed by the name(s) of the specific class(es) as a separate word(s), in alphabetical
order, each preceded by a multiplicative prefix, as needed" (Table 6.3,:31416, lists
fluoride, chloride, bromide, iodide and the pseudohalides); examples 'benzene-1,4-dicarbonyl
dichloride (PIN)' (:31450) and 'ethane-1,2-disulfonyl dibromide (PIN)' (:31452).
 "Acyl pseudohalides from suffix acids" (:31460) says the same of the pseudohalides
('oxalyl diisothiocyanate (PIN)',:31468). The acyl group of an acid named by 'oic acid' takes
the ending 'oyl',:30608), so the acyl group of the multiplied acid
'4,4'-oxydibenzoic acid' is '4,4'-oxydibenzoyl', and the halide is '4,4'-oxydibenzoyl
dichloride': the multiplied acyl group, then the class word with its own multiplicative
prefix. The esters of a multiplied acid are built the same way, the organyl groups cited
before the multiplied anion ('dimethyl 3,3'-oxydibenzoate (PIN)',,:31794).

Splitting the unit at its space and multiplying the first word alone gave
'4,4'-oxydibenzoyl chloride', a name that cites one halide for two. The unit of a multiplied
suffix acid with a locant is enclosed, '4,4'-oxydi(benzene-1-sulfonic acid) (PIN)' (:5827,
 (e),:7085); its acyl group takes the same marks ('4,4'-oxydi(benzene-1-sulfonyl)
dichloride'), so ``ACYL_UNIT_SUFFIXES`` lists the acyl endings next to the acid endings of
``multiplicative._P1634_PARENS_SUFFIXES``.

The Blue Book gives no example of the multiplied acyl group of an acyl halide: the name is the
reading of and for the acyl group of a multiplied acid, as the ester is
, and the claim that it is a PIN rests on that reading.

A unit that holds more than one acyl group or class word ('benzene-1,3-dicarbonyl dichloride',
'butanedioyl bromide chloride') is not multiplied here: the Blue Book gives no multiplied name
for it, and the producer declines it.
"""
import re
from typing import Callable, Optional, Tuple

from ..assembly.naming_utils import simple_multiplier_word
from ..data.oxoacid_cyanides import ACYL_ENDING_OF_ACID_ENDING

# Table 6.3 "Acyl halide and pseudohalide classes" (the Blue Book): the class words.
ACYL_CLASS_WORDS = (
    "fluoride", "chloride", "bromide", "iodide",
    "azide", "cyanide", "isocyanide", "isocyanate", "isothiocyanate",
    "isoselenocyanate", "isotellurocyanate",
)

# The endings of the acyl groups of the suffix acids that take a locant inside a multiplied
# unit: 'carbonyl' (carboxylic acid) and the sulfonyl, sulfinyl, selenonyl,... endings.
ACYL_UNIT_SUFFIXES = ("carbonyl",) + tuple(ACYL_ENDING_OF_ACID_ENDING.values())

_CLASS_WORD = re.compile(
    r"^(?P<multiplier>(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?)(?P<word>%s)$"
    % "|".join(ACYL_CLASS_WORDS))


def multiply_class_word_unit(
    assemble: Callable[..., Optional[str]],
    locant: Optional[int],
    bridge_name: str,
    parent_name: str,
    unit_count: int,
) -> Tuple[bool, Optional[str]]:
    """``(True, name)`` when ``parent_name`` is an acyl group and a class word, else ``(False, None)``.

    ``assemble`` is ``multiplicative._assemble_multiplicative_name``: it multiplies the acyl group,
    as it does the name of an acid, and the class word follows with the multiplicative prefix of the
    count ('4,4'-oxydibenzoyl dichloride', '4,4',4''-nitrilotribenzoyl trichloride'). ``name`` is
    ``None`` (the unit is declined) when the unit holds more than one acyl group or class word (the
    module docstring), or when the acyl group or the count has no multiplied name."""
    acyl, _, last = parent_name.rpartition(" ")
    match = _CLASS_WORD.match(last)
    if not acyl or match is None:
        return False, None
    if match.group("multiplier") or " " in acyl:
        return True, None
    multiplied = assemble(locant, bridge_name, acyl, unit_count)
    prefix = simple_multiplier_word(unit_count)
    if multiplied is None or prefix is None:
        return True, None
    return True, f"{multiplied} {prefix}{match.group('word')}"
