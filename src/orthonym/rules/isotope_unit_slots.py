"""The places of a name where a multiplied unit takes the descriptor of an isotopic modification.

Split out of ``rules.isotopes`` (the size budget of ``tests/unit/test_size_budget.py``): the multiplying prefixes that
can open a multiplied substituent, and the generator of the slots of a skeleton name that the per-unit search of
``isotopes._decorate_uniform_multiplier`` walks.
"""
import re

#: The multiplying prefixes that can open a multiplied substituent and the number
#: of copies each states /: 'di', 'tri',... for a simple
#: substituent, 'bis', 'tris',... for a compound one). Longest first, so 'tetrakis'
#: is not read as 'tetra'.
UNIT_MULTIPLIER_COUNT = {
    "tetrakis": 4, "tris": 3, "bis": 2, "hepta": 7, "octa": 8, "nona": 9,
    "deca": 10, "tetra": 4, "penta": 5, "hexa": 6, "tri": 3, "di": 2,
}
UNIT_MULTIPLIER_RE = re.compile(
    "(?:" + "|".join(UNIT_MULTIPLIER_COUNT) + r")(?=[a-z(\[{]|-(?:tert|sec)-)")


def multiplied_unit_slots(skeleton: str):
    """Every place of ``skeleton`` where a multiplying prefix opens a multiplied
    substituent, as ``(copies, head, base, tail, enclosed)``: the name up to and
    including the multiplier, the substituent word, and what follows it.

     (the Blue Book): the descriptor is inserted "before the part of the
    compound that is isotopically substituted"; for a multiplied substituent that is
    the substituent word, inside the multiplier's marks:43770; the verbatim
    '1,2-di[(13C)methyl]benzene (PIN)',:43734). The multiplier may lead the name
    ('tetramethylsilane', 'dimethyl sulfate'), follow locants ('1,2-diethylbenzene',
    'N,N,N-trimethyl...') or sit inside the marks of an outer group ('...-2,3-bis(
    hexadecanoyloxy)propoxy...'). A multiplied word is enclosed ('bis(hexadecanoyloxy)')
    or, for a simple prefix, bare ('diethyl'); the multiplied anion that closes a
    functional class name ('ethane-1,2-diyl diacetate') is one unit, the whole anion word
    ('di[(2-13C)acetate]', counts the descriptor's parentheses). A
    multiplier-like run inside a word ('hexadecanoyl', 'dihydro') is not offered, and a
    slot that reads no unit fails the caller's round trip, so a surplus slot costs a
    parse, never a wrong name."""
    from ..assembly.naming_utils import _SIMPLE_PREFIX_VOCABULARY, matching_close
    for m in UNIT_MULTIPLIER_RE.finditer(skeleton):
        p = m.start()
        if p and skeleton[p - 1] not in "-([{ ,":
            continue
        copies = UNIT_MULTIPLIER_COUNT[m.group(0)]
        head, rest = skeleton[:m.end()], skeleton[m.end():]
        if rest[:1] in ("(", "[", "{"):
            close = matching_close(rest, 0, "([{")
            if close is not None:
                yield copies, head, rest[1:close], rest[close + 1:], True
            continue
        run = re.match(r"[a-z]+", rest)
        if not run:
            continue
        word = run.group(0)
        if p and skeleton[p - 1] == " " and rest == word and word.endswith("ate"):
            yield copies, head, word, "", False         # the anion word of 'zinc diacetate', 'ethane-1,2-diyl diacetate'
            continue
        for k in range(len(word), 0, -1):
            if word[:k] in _SIMPLE_PREFIX_VOCABULARY:
                yield copies, head, word[:k], rest[k:], False
