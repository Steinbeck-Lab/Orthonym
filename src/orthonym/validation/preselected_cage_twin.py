"""Twin of a preselected cage name: the 'a' name of the same cage, which OPSIN reads.

 (the Blue Book) gives the preselected name of a von Baeyer or spiro system of one
noncarbon element as the descriptor, the multiplied hydride name: 'bicyclo[4.2.1]nonasilane (preselected
name)', and "alternative names for use in general nomenclature are formed by skeletal replacement ('a')
nomenclature": 'nonasilabicyclo[4.2.1]nonane' (:23618;,:9797, example:9810). OPSIN has no grammar
for the first, and reads the second to the same structure (no locants are needed:, "it is not
necessary to give the location of the heteroatoms because all positions are modified by the same
heteroatom",:9806). The rewrite is of the text of the name alone, as the other twins are; the producer
(``rules.preselected_cage``) owns the spelling, the twin only carries the read-back.
"""
import re
from functools import lru_cache
from typing import Dict

from ..data.chain_names import get_chain_name, multiplying_prefix

#: hydride stem of the preselected name -> the 'a' prefix of the twin (Table 2.8 'a' terms,
_A_PREFIX = {'sil': 'sila', 'germ': 'germa', 'stann': 'stanna', 'plumb': 'plumba', 'az': 'aza',
             'phosph': 'phospha', 'ars': 'arsa', 'stib': 'stiba', 'bismuth': 'bisma'}

_NAME = re.compile(
    r"^(?P<descriptor>(?:[a-z]*cyclo|[a-z]*spiro)\[[^\]]+\])(?P<count>[a-z]+?)"
    r"(?P<stem>" + "|".join(sorted(_A_PREFIX, key=len, reverse=True)) + r")ane$")


@lru_cache(maxsize=1)
def _counts() -> Dict[str, int]:
    return {multiplying_prefix(n): n for n in range(2, 201)}


def preselected_cage_twin(name: str) -> str:
    """'bicyclo[4.2.1]nonasilane' -> 'nonasilabicyclo[4.2.1]nonane'; any other name unchanged."""
    m = _NAME.match(name)
    count = _counts().get(m.group("count")) if m else None
    if count is None:
        return name
    return f"{m.group('count')}{_A_PREFIX[m.group('stem')]}{m.group('descriptor')}{get_chain_name(count)}"
