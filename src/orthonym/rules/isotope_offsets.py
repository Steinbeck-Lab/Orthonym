"""Where in the skeleton name of a mixture an isotope descriptor can stand.

The placement search of ``rules.isotopes`` builds a candidate for every offset of the skeleton name and reads each
back with OPSIN. A descriptor is read onto the part of the name that follows it, in the one component of a mixture
name ('ethan-1-ol—(2H1)methane (1/1)',: the components are named one after the other, joined by long dashes,
and the amounts follow in a bracket), and a component of amount ``n`` is read as ``n`` identical molecules (OPSIN 2.9.0:
'(2H2)ethan-1-ol—propane—methane (1/1/2)' is read as C.CCC.C(C)(O)([2H])[2H].C). So two things hold of every candidate
that the search builds from ONE descriptor, and a candidate that breaks one of them cannot be read back to the
labelled input:

* the labelled fragments of the input are all one species (the same labelled structure): the descriptor labels the
  copies of one component and nothing else;
* the descriptor stands in a component whose structure is the structure of a labelled fragment, without its labels.

A search over a mixture of a dozen components, labelled in two of them, built every pair of an offset of the whole name
and a locant up to the heavy atoms of the whole mixture (a CD3 of a five-component mixture: 42,098 OPSIN parses in 90 s,
none of them read to the input).:func:`restrict_offsets` leaves out the offsets that cannot give a candidate that is read
back, and asks nothing of the search beyond it: where the name is no mixture, or the structure of a component cannot be
decided, every offset stays.

The locants are bounded the same way. A locant is looked up among the atoms of the part that follows the descriptor
(``Fragment.getAtomByLocantOrThrow``), and OPSIN names the locant of every atom it reads (``-o extendedsmi``, ``$_AV``), so a
locant above the highest one it gives the atoms of the component is refused ("Could not find the atom with locant N.").
:func:`locant_limit` is that highest locant and a margin for the gaps in the numbering of OPSIN's own fragments
(``isotope_search.GAP_WINDOW``): the search asks for no locant above it, where it asked for one for every heavy atom of
the whole mixture.
"""
import re
from typing import List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import inchi

#: the skeleton name of a mixture: the names of the components, joined by em dashes, then the amounts in a bracket
_MIXTURE_NAME = re.compile(r"^(?P<body>.+) \((?P<amounts>\d+(?:/\d+)+)\)$", re.S)
_DASH = "—"

#: the atom values OPSIN writes after an extended SMILES, one locant per atom ('1', "2'", '3a', 'N', 'N7', or none)
_ATOM_VALUES = re.compile(r"\|\$_AV:(.*)\$\|")
_NUMBERED_LOCANT = re.compile(r"^[A-Za-z]?[a-z]?(\d+)'*[a-z]?$")
_LETTER_LOCANT = re.compile(r"^[A-Za-z]+'*$")


def _first_block(mol: Chem.Mol) -> Optional[str]:
    """The first block of the standard InChIKey (formula and connectivity, no isotopes, charge or stereo), or None."""
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001 -- no key: undecided
        return None
    return key[:14] if key and len(key) >= 14 else None


def labelled_species(original: Chem.Mol) -> Optional[List[Chem.Mol]]:
    """The labelled fragments of ``original``, one per species (the same canonical SMILES, isotopes kept); None when
    the fragments cannot be read."""
    try:
        species = {}
        for fragment in Chem.GetMolFrags(original, asMols=True):
            if any(atom.GetIsotope() for atom in fragment.GetAtoms()):
                species.setdefault(Chem.MolToSmiles(fragment), fragment)
        return list(species.values())
    except Exception:  # noqa: BLE001 -- undecided
        return None


def _component_spans(skel: str) -> Optional[List[Tuple[int, int]]]:
    """The ``(start, end)`` of the name of each component of a mixture skeleton name, or None for any other name."""
    found = _MIXTURE_NAME.match(skel)
    if not found or _DASH not in found.group("body"):
        return None
    parts = found.group("body").split(_DASH)
    if len(parts) != len(found.group("amounts").split("/")):
        return None
    spans, start = [], 0
    for part in parts:
        spans.append((start, start + len(part)))
        start += len(part) + 1
    return spans


def _can_carry(text: str, blocks: Set[str]) -> bool:
    """False only when OPSIN reads the component ``text`` and none of the fragments it gives has the first InChIKey block
    of a labelled fragment of the input."""
    from ..validation.opsin_roundtrip import opsin_parse
    from .isotope_compare import RADICAL_READING
    smiles = opsin_parse(text, allow_radicals=RADICAL_READING.get())
    parsed = Chem.MolFromSmiles(smiles) if smiles else None
    if parsed is None:
        return True
    keys = [_first_block(fragment) for fragment in Chem.GetMolFrags(parsed, asMols=True)]
    return any(key is None or key in blocks for key in keys)


def restrict_offsets(offsets: List[int], skel: str, original: Chem.Mol) -> List[int]:
    """The offsets of ``offsets`` (the insertion offsets of ``skel``) at which a descriptor of the labels of ``original``
    can be read back to ``original``; ``offsets`` itself when that cannot be decided (see the module docstring)."""
    labelled = labelled_species(original)
    if not labelled:
        return offsets
    if len(labelled) > 1:
        return []
    spans = _component_spans(skel)
    if spans is None:
        return offsets
    block = _first_block(labelled[0])
    if block is None:
        return offsets
    parts = [skel[start:end] for start, end in spans]
    reachable = [span for span, text in zip(spans, parts) if _can_carry(text, {block})]
    # A labelled species sits in some component: when none matches (a component named through another drawing of the
    # same compound, 'zinc diacetate' read as zinc(2+) and two acetates), the comparison is what failed, and every
    # offset stays.
    if not reachable or len(reachable) == len(spans):
        return offsets
    return [off for off in offsets if any(start <= off <= end for start, end in reachable)]


def _highest_locant(text: str) -> Optional[int]:
    """The highest number among the locants OPSIN gives the atoms of the name ``text``; None when OPSIN does not
    read it or writes a locant of a shape not read here (the search then keeps every locant)."""
    from ..validation.opsin_roundtrip import _extended_smiles
    from .isotope_compare import RADICAL_READING
    extended = _extended_smiles(text, allow_radicals=RADICAL_READING.get())
    found = _ATOM_VALUES.search(extended or "")
    if not found:
        return None
    top = 0
    for token in found.group(1).split(";"):
        token = token.strip()
        if not token or _LETTER_LOCANT.match(token):
            continue
        number = _NUMBERED_LOCANT.match(token)
        if number is None:
            return None
        top = max(top, int(number.group(1)))
    return top


def search_space(skel: str, original: Chem.Mol, offsets: List[int], n_pos: int) -> Tuple[List[int], int]:
    """``([0] + the offsets, the locants 1..n)`` that the placement search asks for ``skel``: the offsets of
    :func:`restrict_offsets`, and ``n_pos`` (the heavy atoms of the whole molecule) lowered to the highest locant of the
    parts of the name that can take a descriptor, with the margin of the gaps in OPSIN's numbering."""
    from .isotope_search import GAP_WINDOW
    kept = restrict_offsets(offsets, skel, original)
    spans = _component_spans(skel)
    texts = [skel] if spans is None else [skel[start:end] for start, end in spans
                                          if any(start <= off <= end for off in kept)]
    tops = [_highest_locant(text) for text in texts]
    if tops and None not in tops:
        n_pos = min(n_pos, max(tops) + GAP_WINDOW)
    return list(dict.fromkeys([0] + kept)), n_pos
