"""The acyl prefix of a carboxylic acid whose =O is replaced by =S, =Se or =Te: 'ethanethioyl'.

 Acyl groups derived from carboxylic acids modified by functional replacement,
(the Blue Book): "Names of acyl groups derived from carboxylic acids modified by functional
replacement are all denoted by the ending 'oyl'"; (:30626): "the suffix 'carbothioic acid' is
changed to 'carbothioyl'": 'propanethioyl (preferred prefix)' (:30620), 'sulfanyl(sulfanylidene)ethanethioyl
(preferred prefix)' (:30604), '4-(ethanethioyl)benzoic acid (PIN)' (:30246). The acid of an acyl group
R-C(=X)- (-OH added) is the X-acid 'R...thioic O-acid'.
"""
import re
from typing import Optional

#: the endings of the O-acids of the chalcogen acids and the acyl endings they give
_O_ACID_TO_ACYL = (('thioic o-acid', 'thioyl'),
                   ('selenoic o-acid', 'selenoyl'),
                   ('telluroic o-acid', 'telluroyl'))


def is_chalcogen_o_acid(low: str) -> bool:
    """True when the lower-cased acid name ends in 'thioic O-acid', 'selenoic O-acid' or 'telluroic
    O-acid' and is not a peroxy acid."""
    return low.endswith(tuple(acid for acid, _ in _O_ACID_TO_ACYL)) and 'perox' not in low


def chalcogen_o_acid_to_acyl(name: str) -> Optional[str]:
    """The acyl prefix of the chalcogen O-acid ``name`` ('ethanethioic O-acid' -> 'ethanethioyl'); None
    (fail closed) for a name with a space before the ending, a di- or poly-acid, which needs a divalent or
    multiplied prefix."""
    low = name.lower()
    for acid, acyl in _O_ACID_TO_ACYL:
        if low.endswith(acid):
            stem = name[:-len(acid)]
            if (' ' in stem or 'dithio' in low or 'diseleno' in low or 'ditelluro' in low
                    or re.search(r'(?:di|tri|tetra|penta|hexa)[-,\d]*(?:carbo)?$', stem.lower())):
                return None
            return stem + acyl
    return None
