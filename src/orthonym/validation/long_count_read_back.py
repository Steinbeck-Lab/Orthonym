"""A name with a count word above 50, read back without OPSIN, by the reconstructor.

OPSIN 2.9.0 has a numerical term up to 50 ('pentaconta') and none above: ``pentacontane``, ``hectane`` and
``pentacontafluorohectane`` are read, and ``tetrapentacontafluorohexacosane`` or a cumulene of 55 carbon atoms is
not. The producer spells these names as the Blue Book does, and until this module the gate withdrew every one of them
(``opsin_unparseable``): 24 rows of the BR study, acyclic cumulenes C53 to C113, perfluoro and perbromo alkanes with
51 and more halogen atoms, polyynes, and a few with an alkyl branch.

 Basic multiplicative prefixes (the Blue Book, Table 1.4) lists the numerical terms, and
Derivation of basic numerical terms (:2811) composes the others: "composite numerical terms are formed
systematically by citing the basic terms in the order opposite to that of the constituent digits in the arabic
numbers... formed by direct joining of the basic terms, without hyphen(s). The letter 'i' in 'icosa' is elided
after a vowel." (examples 52 dopentaconta, 111 undecahecta, 363 trihexacontatricta). Hydrocarbons (:7984)
names the saturated unbranched ones the same way, 'heptacontane (PIN)'.

The check reads the NAME TEXT and nothing else: the parent chain (its numerical term gives its length), the
unsaturation (locants and 'ene' / 'yne' with its multiplier) and the substituent prefixes (locants, multiplier,
name) become ``reconstruct.NameFacts``; the reconstructor builds the structure from the facts alone and compares it
with the input's full canonical SMILES, which ``reconstruct.verify_from_name_facts`` does with no stereo (it
declines an input that asserts any). Nothing is read from the input and nothing from the producer: the count words
are composed here again from the Blue Book table, and the producer's own table (``data.chain_names``) is only
compared with them in a test.

The reader abstains on any token it does not understand, so a name that has anything left over, any locant out of
the chain, a locant list that does not match its multiplier, an unknown prefix, a second parse, or no count word
above 50 gives no verdict. A false confirmation would be a wrong name: the reader never guesses, and the compare is
a full constitution compare. It proves the structure and never the spelling (the order of the prefixes, the
enclosing marks, the choice of the parent): the producer owns those, as for the twin spellings
(``validation.twin_spellings``).

The grammar (hydrocarbon chains, hydrogen replaced by halogen or alkyl prefixes):

* the parent: ``<term>ane`` (the numerical term of the chain without its 'a',,
  ``<term>a-<locants>-<multiplier>ene`` or ``yne`` for several multiple bonds, and ``<term>-<locant>-ene`` / ``yne``
  for one; the locant of a multiple bond is the lower of the two atoms it joins;
* the prefixes: ``<locants>-<multiplier><name>`` blocks in any order, one locant for each multiplied name, a
  name between enclosing marks when it is compound (``(propan-2-yl)``), and no locants at all only for halogen
  prefixes that replace every hydrogen atom of the parent: Omission of locants, "All locants are
  omitted in compounds or substituent groups in which all substitutable positions are completely substituted or
  modified... In case of partial substitution or modification, all numerical prefixes must be indicated."
  (:3007).
"""
import functools
import re
from typing import Dict, Iterator, List, Optional, Tuple

from .reconstruct import NameFacts, verify_from_name_facts

#: the rule id of the outcome (``reconstructed:long_count``)
LONG_COUNT_RULE = "long_count"

#: the smallest count word OPSIN 2.9.0 does not read ('hen' + 'pentaconta' = 51)
_FIRST_UNREAD_COUNT = 51

#: Table 1.4 (the Blue Book): the numerical terms in association (units, tens, hundreds)
_UNITS = {1: "hen", 2: "do", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa", 9: "nona"}
_TENS = {1: "deca", 2: "icosa", 3: "triaconta", 4: "tetraconta", 5: "pentaconta", 6: "hexaconta",
         7: "heptaconta", 8: "octaconta", 9: "nonaconta"}
_HUNDREDS = {1: "hecta", 2: "dicta", 3: "tricta", 4: "tetracta", 5: "pentacta", 6: "hexacta",
             7: "heptacta", 8: "octacta", 9: "nonacta"}

#: the prefixes of the substituents the reader knows, each the name of ``reconstruct._SUBSTITUENT_TABLE``;
#: a halogen is replaced by a halogen atom, an alkyl is a carbon chain. Bare names and enclosed names are
#: separate: 'propan-2-yl' states a locant and is cited between enclosing marks, Parentheses,
#: (the Blue Book; '1-(butan-2-yl)-3-tert-butylbenzene (PIN)',:3507).
_HALOGENS = ("fluoro", "chloro", "bromo", "iodo")
_BARE = _HALOGENS + ("methyl", "ethyl", "propyl", "butyl")
_ENCLOSED = ("propan-2-yl",)

#: an upper bound on the multiplying word before a name: the term of 999 plus 'kis'
#: ('nonanonacontanonactakis') has 23 letters, so 48 leaves a wide margin
_MAX_WORD = 48


def _below_hundred(n: int) -> str:
    """The term of 1..99 in association (units before tens,; 11 is 'undeca'."""
    if n == 11:
        return "undeca"
    tens, units = divmod(n, 10)
    unit, ten = _UNITS.get(units, ""), _TENS.get(tens, "")
    if ten.startswith("i") and unit.endswith(("a", "o", "i", "e", "u")):
        ten = ten[1:]                                       # "The letter 'i' in 'icosa' is elided after a vowel"
    return unit + ten


@functools.lru_cache(maxsize=None)
def _terms() -> Tuple[Dict[int, str], Dict[str, int]]:
    """The numerical terms of 2..999 by number, and the numbers of the words (a word names one number).

    Alone, 2 is 'di'; in association it is 'do', "except in the cases of 'dicta' and 'dilia'")."""
    by_number: Dict[int, str] = {}
    for n in range(2, 1000):
        hundreds, low = divmod(n, 100)
        low_word = _below_hundred(low) if low else ""
        by_number[n] = low_word + _HUNDREDS.get(hundreds, "")
    by_number[2] = "di"
    by_word = {word: n for n, word in by_number.items()}
    if len(by_word) != len(by_number):
        raise AssertionError("two numbers share a numerical term")
    return by_number, by_word


def numerical_term(n: int) -> Optional[str]:
    """The Blue Book numerical term of ``n`` (2..999), composed from Table 1.4, or None."""
    return _terms()[0].get(n)


def _count(word: str) -> Optional[int]:
    """The number a simple multiplying term or a count word stands for, or None."""
    return _terms()[1].get(word)


def _multiplier(word: str, enclosed: bool) -> Optional[int]:
    """The number of a multiplying prefix before a name: 'di', 'tri',... and, before an enclosed name,
    'bis', 'tris' and the term with 'kis' "formed by adding the ending 'kis' to the basic multiplicative
    prefix ending in 'a'", 4 'tetrakis')."""
    if not enclosed:
        return _count(word)
    if word == "bis":
        return 2
    if word == "tris":
        return 3
    if word.endswith("kis"):
        n = _count(word[:-3])
        return n if n is not None and n >= 4 else None
    return _count(word)


class _Block:
    """One prefix of the name: its locants (None when the name has none), its multiplier and its name."""

    __slots__ = ("locants", "count", "name")

    def __init__(self, locants: Optional[List[int]], count: int, name: str):
        self.locants, self.count, self.name = locants, count, name


#: locants are written without leading zeros ('01-fluoro' is not a locant)
_LOCANTS = re.compile(r"([1-9]\d*(?:,[1-9]\d*)*)-")


def _ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",")]


def _terms_at(text: str, start: int) -> Iterator[Tuple[int, int, str]]:
    """Every reading of a multiplied prefix at ``start``: (end, count, name)."""
    for stop in range(start, min(len(text), start + _MAX_WORD) + 1):
        word = text[start:stop]
        count = 1 if not word else _multiplier(word, False)
        if count is not None:
            for name in _BARE:
                if text.startswith(name, stop):
                    yield stop + len(name), count, name
        count = 1 if not word else _multiplier(word, True)
        if count is not None:
            for name in _ENCLOSED:
                if text.startswith("(" + name + ")", stop):
                    yield stop + len(name) + 2, count, name


def _blocks(text: str, start: int) -> Iterator[List[_Block]]:
    """Every reading of ``text[start:]`` as prefix blocks, each ending at the end or before a hyphen."""
    if start == len(text):
        yield []
        return
    found = _LOCANTS.match(text, start)
    locants, begin = (_ints(found.group(1)), found.end()) if found else (None, start)
    for end, count, name in _terms_at(text, begin):
        block = _Block(locants, count, name)
        if end == len(text):
            yield [block]
        elif text[end] == "-" and _LOCANTS.match(text, end + 1):
            # a hyphen inside the prefixes only precedes the locants of the next prefix;
            # '1-fluoro-henpentacontane' (a hyphen before the parent) is not read
            for rest in _blocks(text, end + 1):
                yield [block] + rest


_PARENT_MULTIPLE = re.compile(r"(?P<term>[a-z]+)-(?P<locants>\d+(?:,\d+)*)-(?P<count>[a-z]+)(?P<kind>ene|yne)")
_PARENT_ONE = re.compile(r"(?P<term>[a-z]+)-(?P<locant>\d+)-(?P<kind>ene|yne)")
_ORDER = {"ene": 2, "yne": 3}


def _parent(text: str) -> Optional[Tuple[int, int, List[Tuple[int, int]]]]:
    """The parent chain a tail of the name reads as: (its length, the largest count it cites, its multiple bonds
    as (locant, order)), or None. A chain of five or more carbon atoms: its term is the numerical term with the
    'a' elided before 'ane' and before one 'ene' / 'yne',, kept before a multiplied one."""
    if text.endswith("ane"):
        n = _count(text[:-3] + "a")
        return (n, n, []) if n is not None and n >= 5 else None
    found = _PARENT_MULTIPLE.fullmatch(text)
    if found:
        n, count = _count(found["term"]), _multiplier(found["count"], False)
        locants = _ints(found["locants"])
        if n is None or n < 5 or count is None or count != len(locants) or len(set(locants)) != count:
            return None
        return n, max(n, count), [(locant, _ORDER[found["kind"]]) for locant in locants]
    found = _PARENT_ONE.fullmatch(text)
    if found:
        n = _count(found["term"] + "a")
        return (n, n, [(int(found["locant"]), _ORDER[found["kind"]])]) if n is not None and n >= 5 else None
    return None


def _free_hydrogens(length: int, multiple_bonds: List[Tuple[int, int]]) -> List[int]:
    """The hydrogen atoms of each carbon atom of the parent hydride, position 1 first."""
    valence = [4] * length
    for index in range(length):
        valence[index] -= (index > 0) + (index < length - 1)
    for locant, order in multiple_bonds:
        valence[locant - 1] -= order - 1
        valence[locant] -= order - 1
    return valence


def _facts_of(prefix: str, parent: Tuple[int, int, List[Tuple[int, int]]]) -> Optional[Tuple[NameFacts, int]]:
    """The facts of one reading of the name and the largest count it cites, or None."""
    length, largest, multiple_bonds = parent
    if any(not 1 <= locant < length for locant, _ in multiple_bonds):
        return None
    readings = list(_blocks(prefix, 0))
    if len(readings) != 1:
        return None                                         # a second reading of the prefixes is no reading
    substituents: List[Tuple[str, int]] = []
    blocks = readings[0]
    if blocks and blocks[0].locants is None:
        #: every substitutable position substituted by the same halogen, and the count says so
        hydrogens = _free_hydrogens(length, multiple_bonds)
        if len(blocks) != 1 or blocks[0].name not in _HALOGENS or blocks[0].count != sum(hydrogens):
            return None
        substituents = [(blocks[0].name, locant + 1) for locant, free in enumerate(hydrogens) for _ in range(free)]
    else:
        for block in blocks:
            if block.locants is None or len(block.locants) != block.count:
                return None
            if any(not 1 <= locant <= length for locant in block.locants):
                return None
            substituents += [(block.name, locant) for locant in block.locants]
    largest = max([largest] + [block.count for block in blocks])
    facts = NameFacts(parent_kind="chain", parent_length=length, unsaturations=tuple(multiple_bonds),
                      substituents=tuple(substituents))
    return facts, largest


#: the characters of a name this reader can understand, and the longest name it reads
_READABLE = re.compile(r"[a-z0-9,()-]+")
_MAX_NAME = 2000
#: the multiple bonds a name cites at its end: ``-<locants>-<multiplier>ene`` or ``yne``
_TAIL = re.compile(r"-\d+(?:,\d+)*-[a-z]*(?:ene|yne)$")


def _splits(name: str) -> range:
    """The places the parent of ``name`` can begin: the numerical term of the chain stands just before the
    ending 'ane' or before the locants of the multiple bonds, and is at most ``_MAX_WORD`` letters long."""
    if name.endswith("ane"):
        end = len(name) - 3
    else:
        tail = _TAIL.search(name)
        if tail is None:
            return range(0)
        end = tail.start()
    return range(max(0, end - _MAX_WORD), end)


def name_facts_from_text(name: str, least_count: int = _FIRST_UNREAD_COUNT) -> Optional[NameFacts]:
    """The ``NameFacts`` of ``name``, read from its text alone, or None when the name is not wholly understood
    or cites no count of ``least_count`` and above (the class of this reader: names OPSIN has no numerical term
    for, 51 and above; a lower floor serves the comparison with OPSIN's own reading, which a test makes)."""
    if not name or len(name) > _MAX_NAME or _READABLE.fullmatch(name) is None:
        return None
    found = None
    for split in _splits(name):
        parent = _parent(name[split:])
        if parent is None:
            continue
        facts = _facts_of(name[:split], parent)
        if facts is None:
            continue
        if found is not None and found != facts:
            return None                                     # two different readings: no reading
        found = facts
    if found is None or found[1] < least_count:
        return None
    return found[0]


def long_count_round_trip(name: str, smiles: Optional[str]) -> Optional[str]:
    """``LONG_COUNT_RULE`` when the structure the reader builds from the text of ``name`` is the constitution of
    ``smiles``, else None (a name it does not wholly understand, another structure, an input with stereo)."""
    if not smiles:
        return None
    facts = name_facts_from_text(name)
    if facts is None:
        return None
    from rdkit import Chem, rdBase
    with rdBase.BlockLogs():                                # a wrong name is a valence error of RDKit's, not news
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not verify_from_name_facts(facts, mol):
            return None
    return LONG_COUNT_RULE
