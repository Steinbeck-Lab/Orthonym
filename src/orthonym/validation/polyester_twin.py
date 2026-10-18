"""The twin of a functional-class polyol polyester name: the same ester written with acyloxy prefixes.

``propane-1,2,3-triyl 1,3-diacetate 2-propanoate`` is the preferred name of a polyester of one polyol and
different acids method (1), the Blue Book,:31833-:31836: "When anions are different, two
methods are used: (1) names of anions are cited in alphanumerical order preceded by a locant, when
required; multiplicative prefixes are used to denote a multiplicity of identical anionic components;
(2) one anion is chosen as principal anion and all other ester groups are expressed as prefixes in the name
of the organyl group. [...] Method (1) generates preferred IUPAC names but names formed by using method (2)
are acceptable in general nomenclature", the example 'propane-1,2,3-triyl 1,2-diacetate 3-propanoate (PIN)
(2) 2,3-bis(acetyloxy)propyl propanoate',:31840). OPSIN 2.9.0 cannot read the multi-anion form, and the
name used to ship with no read-back at all (the ``polyol_polyester`` grammar-gap carve-out).

The twin writes every anion word as an acyloxy prefix on the parent hydride of the polyol:
``1,3-bis[acetyloxy]-2-[propanoyloxy]propane``. Substitutive nomenclature names the same structure
,:31968: "Substitutive nomenclature is senior to functional class nomenclature for preferred
IUPAC for esters", '4-(acetyloxy)benzoic acid (PIN)'), so the twin reads to the structure of the name if
the anion words are what the name says. It is a rewrite of the tokens of the name: the polyol parent
('propane-1,2,3-triyl' to 'propane'), each anion ('acetate' to 'acetyloxy', 'octadecanoate' to
'octadecanoyloxy', 'carbamate' to 'carbamoyloxy', 'cyclopropane-1-carboxylate' to
'cyclopropane-1-carbonyloxy') with the locants it is cited under, the multiplying prefix of a repeated
anion as 'bis'/'tris', and the stereodescriptor of the parent in front. A half ester ('2-pentanedioate')
has no acyloxy prefix OPSIN reads ('pentanedioyloxy' has a free valence), so the twin is not read and the
name does not ship: that name states the dianion, and (:31935) gives the substitutive name of
the acid as the preferred one.

The twin proves the structure and never the spelling: the producer owns the spelling.
"""
import re
from typing import List, Optional

from rdkit import Chem

_MULTIPLIERS = {2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis"}
_WORD_MULTIPLE = {"di": 2, "tri": 3, "tetra": 4, "penta": 5}
_BIS = {"bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5}

#: ``[(2S)-]<parent>ane-<locants>-<di|tri|tetra|penta>yl <anion words>``
_SHAPE = re.compile(
    r"^(?P<descriptor>\([^()\s]*\)-)?(?P<parent>[a-z]+ane)-(?P<locants>\d+(?:,\d+)*)-"
    r"(?P<count>di|tri|tetra|penta)yl (?P<words>\S.*)$")
#: an anion word: the locants it is cited under, then the anion (a multiplied anion with its prefix)
_WORD = re.compile(r"^(?P<locants>\d+(?:,\d+)*)-(?P<anion>.+)$")
_WORD_SPLIT = re.compile(r" (?=\d+(?:,\d+)*-)")


def _enclosed(text: str) -> Optional[str]:
    """The inside of ``text`` when one pair of enclosing marks spans all of it, else None."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    if len(text) < 2 or text[0] not in pairs or text[-1] != pairs[text[0]]:
        return None
    depth = 0
    for i, ch in enumerate(text):
        if ch in pairs:
            depth += 1
        elif ch in pairs.values():
            depth -= 1
            if depth == 0 and i != len(text) - 1:
                return None
    return text[1:-1] if depth == 0 else None


def _acyloxy(anion: str) -> Optional[str]:
    """The acyloxy prefix of an anion name: 'acetate' 'acetyloxy', 'octadecanoate' 'octadecanoyloxy',
    'cyclopropane-1-carboxylate' 'cyclopropane-1-carbonyloxy', 'carbamate' 'carbamoyloxy'; None for a
    name that is no '-ate' anion."""
    for ending, prefix in (("carboxylate", "carbonyloxy"), ("carbamate", "carbamoyloxy"),
                           ("idate", "idoyloxy"), ("oate", "oyloxy"), ("ate", "yloxy")):
        if anion.endswith(ending):
            return anion[:-len(ending)] + prefix
    return None


def _anion_prefix(locants: List[str], anion: str) -> Optional[str]:
    """``'1,3-bis[acetyloxy]'`` for the word ``1,3-diacetate`` / ``1,3-bis(acetate)``; None when the word is
    not one the twin reads (a multiplier that does not count the locants, an anion without an acyloxy)."""
    count = len(locants)
    inner = _enclosed(anion)
    if count > 1:
        multiple = re.match(r"^(bis|tris|tetrakis|pentakis)(.+)$", anion)
        plain = re.match(r"^(di|tri|tetra|penta)(.+)$", anion)
        if multiple and _BIS[multiple.group(1)] == count and _enclosed(multiple.group(2)) is not None:
            anion = _enclosed(multiple.group(2))
        elif plain and _WORD_MULTIPLE[plain.group(1)] == count:
            anion = plain.group(2)
        else:
            return None
    elif inner is not None:
        anion = inner
    acyloxy = _acyloxy(anion)
    if acyloxy is None:
        return None
    mark = f"[{acyloxy}]"
    return f"{','.join(locants)}-{_MULTIPLIERS[count]}{mark}" if count > 1 else f"{locants[0]}-{mark}"


def polyester_twin(name: str) -> str:
    """The acyloxy spelling of a polyol polyester name, or ``name`` itself when it is no such name or a
    word of it is not one the twin reads. Pure and deterministic."""
    found = _SHAPE.match(name or "")
    if found is None:
        return name
    parent_locants = found["locants"].split(",")
    prefixes = []
    cited: List[str] = []
    words = _WORD_SPLIT.split(found["words"])
    if len(words) < 2:
        return name
    for word in words:
        parsed = _WORD.match(word)
        if parsed is None:
            return name
        locants = parsed["locants"].split(",")
        prefix = _anion_prefix(locants, parsed["anion"])
        if prefix is None:
            return name
        prefixes.append(prefix)
        cited.extend(locants)
    if sorted(cited, key=int) != sorted(parent_locants, key=int):
        return name
    return f"{found['descriptor'] or ''}{'-'.join(prefixes)}{found['parent']}"


_FREE_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def polyester_has_free_acid(smiles: Optional[str]) -> bool:
    """Whether the structure holds a free carboxylic acid group (or carboxylate): the residue of a
    polybasic acid that is esterified once, a half ester.

    OPSIN reads the acyloxy twin of 'propane-1,2,3-triyl 1,3-diacetate 2-glutamate' to the structure of the
    glycerol diacetate glutamate, whose acyl keeps its free COOH ('glutamyloxy'), so the read-back alone would
    verify an anion word that states the dianion. A polyester name of one polyol cites fully esterified anions
    of acids; a residue with another free acid group makes the compound an acid, the senior class, and
     Partial esters of polybasic acids and their salts (the Blue Book): "(1) substitutively on
    the basis of the anion, the free acid group(s) and the ester group(s) being cited as prefixes" and "Method
    (1) generates preferred IUPAC names" (:31941). The twin is not offered for such a structure: only the
    monobasic acid residues of a fully esterified polyol are read back, whatever the word for the anion is
    (the retained 'glutamate', 'aspartate', 'malate' and every systematic or substituted one)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol is not None and mol.HasSubstructMatch(_FREE_ACID)
