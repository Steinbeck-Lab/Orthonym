"""An isotopically substituted compound named from OPSIN's locants of its unlabelled name (the all-organic program,
lane I).

 "nuclide descriptors" (the Blue Book): the nuclide symbols, with the locants of the labelled
positions, are enclosed in parentheses and placed before the name, '(1,1,1,3,3-2H5)pentan-2-one',
:44208); the symbols are cited in the alphanumerical order of the element, then of the mass number, and an integer
locant is repeated once per labelled atom.

``rules.isotopes.decorate_isotopic_name`` searches the offsets of the skeleton name for a descriptor and asks OPSIN
which one reproduces the input; on 197 declined PubChem rows that name without their labels it found none. Here the
locants come from OPSIN itself: ``validation.opsin_roundtrip.opsin_atom_locant_map`` maps each atom of the unlabelled
molecule to the locant OPSIN gives it in the skeleton name, so a front descriptor with those locants names the
labelled positions, and the name ships only when OPSIN reads it back to the input's full standard InChIKey, whose
isotopic layer holds every label. A labelled atom with no locant in the map (an atom of a substituent that the name
numbers on its own, an oxygen of a suffix) declines: the old search stays the only route there.

Called only after the old search finds nothing, so no name it ships changes; the name is recorded as no PIN
(``record_non_pin_fragment``): a front descriptor is not always where the book cites it.
"""
import re
from collections import defaultdict
from typing import Optional

from rdkit import Chem

#: a leading stereodescriptor group ('(2S)-', '(1R,2E)-'): the isotope descriptor follows it, '(2S)-(2-2H)butan-2-ol'
#: the most placements read back for one name
_MAX_CANDIDATES = 120
_LEADING_STEREO = re.compile(r"^\((?:[0-9,'a-z]*[RSEZrs]\*?,?)+\)-")


def _token(locant) -> Optional[str]:
    if locant is None:
        return None
    if isinstance(locant, (tuple, list)):
        return f"{locant[0]}{locant[1]}"
    return str(locant)


def _sort_key(token: str):
    m = re.match(r"(\d+)([a-z]?)('*)$", token)
    return (0, int(m.group(1)), m.group(2), len(m.group(3))) if m else (1, 0, token, 0)


def locant_map_descriptor(original: Chem.Mol, skeleton_name: str) -> Optional[str]:
    """The front descriptor of ``original`` on ``skeleton_name`` ('(2,2,4-2H3)'), or None."""
    from ..validation.opsin_roundtrip import opsin_atom_locant_map
    heavy = [a.GetIdx() for a in original.GetAtoms() if a.GetAtomicNum() > 1]
    work = Chem.RWMol(original)
    for atom in work.GetAtoms():
        atom.SetIsotope(0)
    try:
        bare = Chem.RemoveHs(work.GetMol())
    except Exception:  # noqa: BLE001 -- a skeleton RDKit cannot write has no map
        return None
    if bare.GetNumAtoms() != len(heavy) or any(
            bare.GetAtomWithIdx(new).GetAtomicNum() != original.GetAtomWithIdx(old).GetAtomicNum()
            for new, old in enumerate(heavy)):
        return None
    locants = opsin_atom_locant_map(skeleton_name, bare)
    if not locants:
        return None
    new_index = {old: new for new, old in enumerate(heavy)}
    cited = defaultdict(list)       # (element, mass) -> locant tokens, one per labelled atom
    may_hold_more = defaultdict(bool)
    for atom in original.GetAtoms():
        if not atom.GetIsotope():
            continue
        if atom.GetAtomicNum() == 1:
            carriers = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
            if len(carriers) != 1:
                return None
            carrier = carriers[0]
            hydrogens = carrier.GetTotalNumHs() + sum(1 for nb in carrier.GetNeighbors() if nb.GetAtomicNum() == 1)
            key = ('H', atom.GetIsotope())
            may_hold_more[key] |= hydrogens > 1
        else:
            carrier = atom
            key = (atom.GetSymbol(), atom.GetIsotope())
        token = _token(locants.get(new_index[carrier.GetIdx()]))
        if token is None:
            return None
        cited[key].append(token)
    if not cited:
        return None
    parts = []
    for (element, mass) in sorted(cited, key=lambda k: (k[0], k[1])):
        tokens = sorted(cited[(element, mass)], key=_sort_key)
        count = len(tokens)
        sub = str(count) if count > 1 or may_hold_more[(element, mass)] else ""
        parts.append(f"{','.join(tokens)}-{mass}{element}{sub}")
    return "(" + ",".join(parts) + ")"


def _insertion_points(name: str):
    """The offsets of ``name`` a descriptor is tried at, in order: the start, then each word that a mark or a hyphen
    opens, then the parent of each unit the spelling parser reads ('trichloro|methane'), left to right."""
    points = {0}
    points.update(i for i in range(1, len(name)) if name[i - 1] in "([{-" and name[i].isalnum())
    try:
        from ..validation.spelling.units import parse_name
        _s, _e, units = parse_name(name)
        for unit in units:
            if unit.parent and unit.parent.isalpha():
                start = name.find(unit.parent)
                while start >= 0:
                    points.add(start)
                    start = name.find(unit.parent, start + 1)
    except Exception:  # noqa: BLE001 -- the parser adds offsets; without it the marks still give some
        pass
    return sorted(points)


def decorate_by_locant_map(original: Chem.Mol, skeleton_name: Optional[str]) -> Optional[str]:
    """``skeleton_name`` with the front descriptor of ``original``'s labels, when OPSIN reads it back to the full
    standard InChIKey of ``original``; else None. One component only (an adduct keeps the old route)."""
    if not skeleton_name or "—" in skeleton_name or original is None or len(Chem.GetMolFrags(original)) > 1:
        return None
    desc = locant_map_descriptor(original, skeleton_name)
    if desc is None:
        return None
    stereo = _LEADING_STEREO.match(skeleton_name)
    first = stereo.end() if stereo else 0
    key = Chem.MolToInchiKey(original)
    from ..namer import _validity_gate_name_to_smiles
    from .isotopes import _escalate_marks_for_descriptor
    candidates = []
    for i in _insertion_points(skeleton_name):
        if i < first or (i > first and skeleton_name[i - 1:i] == "("
                         and _LEADING_STEREO.match(skeleton_name[i - 1:])):
            continue
        head, body = skeleton_name[:i], skeleton_name[i:]
        #: a closing mark before a locant takes a hyphen, '(2S)-(2-2H)butan-2-ol';: the
        # descriptor inside a substituent's marks steps them up, '1,2-di[(13C)methyl]benzene' (the Blue Book)
        spliced = f"{head}{desc}-{body}" if body[:1].isdigit() else f"{head}{desc}{body}"
        candidates.append(_escalate_marks_for_descriptor(spliced, i))
    for candidate in candidates[:_MAX_CANDIDATES]:
        read = _validity_gate_name_to_smiles(candidate)
        mol = Chem.MolFromSmiles(read) if read else None
        if mol is not None and key and Chem.MolToInchiKey(mol) == key:
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(candidate)
            return candidate
    return None
