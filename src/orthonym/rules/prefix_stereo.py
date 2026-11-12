"""The input's stereo composed on a stereo-free name with each descriptor at the front of the part it relates to.

 NAMING OF STEREOISOMERS (the Blue Book),:44643: "In preferred IUPAC names, stereodescriptors are
placed immediately at the front of the part of the name to which they relate.... When they relate to substituent
groups, they are cited at the front of the corresponding prefix. They are preceded by a numerical or letter locant
to describe the position of the stereogenic unit when such locants are present". The composer of the best-effort
tier (``namer._try_compose_input_stereo``) cites every descriptor at the front of the complete name, in OPSIN's
numbering of the stereo-free name; a double bond or a centre of a substituent then carries the substituent's own
locant at the front of the name, where it is read on the parent (``(1Z)-4-[(2-aminophenyl)...methyl]pyridazine``),
and the name does not read back. Here each stereo unit is placed on its own: at the front of the name or at the
front of an enclosed prefix (``4-[(Z)-(2-aminophenyl)...methyl]pyridazine``), with the locant OPSIN gives its atom
in the stereo-free name or with none, and a place is kept only when OPSIN reads the name with that one descriptor
to the input with only that unit's stereo. The units then go into one name, one block per place, the enclosing
marks around each block are derived again,:7446: a descriptor block nests,:7478), and the
name must read back to the input as an exact isomeric match and pass the enclosing-marks spelling check. The name
is recorded as non-PIN: the caller labels it below the PIN. Nothing is guessed: a unit no place reads back leaves
the name to the caller's decline.
"""
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: bounds of the search: the stereo units of one molecule, the places of one name and the OPSIN readings tried
_MAX_UNITS = 12
_MAX_PLACES = 60
_MAX_READINGS = 200
_OPEN = "([{"
_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _units(mol) -> Optional[List[Tuple[str, int, str]]]:
    """The stereo units of ``mol`` (CIP-labelled), as ``('a', atom, code)`` and ``('b', bond, code)``; None when
    a defined unit has no CIP label or is not a tetrahedral centre or a double bond (a square-planar or
    octahedral metal centre, an atropisomeric axis: its descriptor is not written here, and the standard InChIKey
    would not see it missing)."""
    units = []
    for a in mol.GetAtoms():
        tag = a.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        code = a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None
        if tag not in _TETRAHEDRAL or code not in ("R", "S", "r", "s"):
            return None
        units.append(("a", a.GetIdx(), code))
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        code = b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None
        if code not in ("E", "Z"):
            return None
        units.append(("b", b.GetIdx(), code))
    return units


def _single_unit_smiles(mol, unit) -> str:
    """``mol`` with the stereo of ``unit`` only."""
    m = Chem.Mol(mol)
    kind, idx, _ = unit
    for a in m.GetAtoms():
        if not (kind == "a" and a.GetIdx() == idx):
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        if not (kind == "b" and b.GetIdx() == idx) and b.GetStereo() != Chem.BondStereo.STEREONONE:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(m)


def _places(name: str) -> List[int]:
    """The front of every enclosed part that begins with a letter, a locant or another enclosed part, the
    innermost first (a descriptor stands at the front of the smallest part it relates to), then the front of each
    word, the last word first: in an ester the acid word holds the unit of the acid ('methyl (2Z)-but-2-enoate';,
    the Blue Book, the descriptor at the front of the part it relates to), and a front OPSIN reads only on the
    groups of its own word."""
    places, words, depth = [], [], 0
    for i, ch in enumerate(name[:-1]):
        if ch in _OPEN:
            depth += 1
            if name[i + 1].isalnum() or name[i + 1] in _OPEN:
                places.append((-depth, i + 1))
        elif ch in ")]}":
            depth -= 1
        elif ch == " " and depth == 0:
            words.append(i + 1)
    return [i for _, i in sorted(places)] + words[::-1] + [0]


def _word_front(name: str, place: int) -> int:
    """The front of the top-level word of ``name`` that holds ``place``."""
    depth, front = 0, 0
    for i, ch in enumerate(name[:place]):
        depth += ch in _OPEN
        depth -= ch in ")]}"
        if ch == " " and depth == 0:
            front = i + 1
    return front


def _texts(mol, unit, locants: Dict[int, object]) -> List[str]:
    """The descriptors tried for ``unit``: with the locant of its atom (of a double bond, the lower locant first:
    , the Blue Book, "Only the lower locant for a multiple bond is cited"); bare only when no
    atom of the unit has a locant,:44643: a locant "when such locants are present")."""
    from .stereochemistry import _render_locant_token
    kind, idx, code = unit
    atoms = [idx] if kind == "a" else [mol.GetBondWithIdx(idx).GetBeginAtomIdx(),
                                       mol.GetBondWithIdx(idx).GetEndAtomIdx()]
    tokens = sorted({_render_locant_token(locants[a]) for a in atoms if a in locants}, key=_descriptor_key)
    return [f"{t}{code}" for t in tokens] if tokens else [code]


def _attachment_texts(name: str, place: int, code: str) -> Tuple[int, List[str], str]:
    """``(host, texts, attached)`` for the double bond of the enclosed prefix that opens at ``place``, when that
    prefix ends in 'ylidene' or 'imino': ``host`` is the front of the part that holds the prefix (its word, or the
    enclosed prefix around it), ``texts`` one descriptor per locant cited before the prefix ('2-(propylidene)' ->
    '2E'; '1,5-di(...ylidene)' -> '1E', '5E'), bare without one ('[(...-5-ylidene)methyl]' -> 'E'), ``attached`` the
    prefix's own locant of its attachment atom ('prop-2-en-1-ylidene' -> '1', 'propan-2-ylidene' -> '2',
    'ethylidene' -> '1') or 'N' for an 'imino' group. Else ``(0, , '')``."""
    end, opens = None, []
    for i, ch in enumerate(name[:place - 1]):
        if ch in _OPEN:
            opens.append(i)
        elif ch in ")]}" and opens:
            opens.pop()
    host = opens[-1] + 1 if opens else _word_front(name, place)
    level = 1
    for j in range(place, len(name)):
        level += name[j] in _OPEN
        level -= name[j] in ")]}"
        if level == 0:
            end = j
            break
    if end is None or not name[place:end].endswith(("ylidene", "imino")):
        return 0, [], ""
    own = re.search(r"-(\d+)-ylidene$", name[place:end])
    attached = "N" if name[place:end].endswith("imino") else (own.group(1) if own else "1")
    before = name[:place - 1]
    for multiplier in ("di", "tri", "tetra", "bis", "tris", "tetrakis"):
        if before.endswith("-" + multiplier):
            before = before[:-len(multiplier)]
            break
    if not before.endswith("-"):
        return host, ([code] if not before or before[-1] in " ([{" else []), attached
    start = len(before) - 1
    while start > 0 and before[start - 1] not in " -([{":
        start -= 1
    return host, [f"{loc}{code}" for loc in before[start:-1].split(",") if loc], attached


def _is_attachment_bond(mol, bond_idx: int, locants: Dict[int, object], attached: str) -> bool:
    """Whether the double bond is the one that joins the '-ylidene' (or '-imino') prefix to its host: one of its
    atoms carries the prefix's attachment locant in OPSIN's labels (``attached``), or is the nitrogen of an 'imino'
    group. Inside a '1-ylidene' chain no other double bond can reach C-1, which already has its double bond."""
    from .stereochemistry import _render_locant_token
    bond = mol.GetBondWithIdx(bond_idx)
    atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
    if attached == "N":
        return any(a.GetSymbol() == "N" for a in atoms)
    return attached in {_render_locant_token(locants[a.GetIdx()]) for a in atoms if a.GetIdx() in locants}


def _with_block(name: str, place: int, block: str) -> str:
    """``name`` with the descriptor block ``(block)-`` at ``place`` and the enclosing marks of the part it opens,
    and of every part around that, derived again."""
    from ..assembly.naming_utils import renest_group_and_ancestors
    out = f"{name[:place]}({block})-{name[place:]}"
    return renest_group_and_ancestors(out, place - 1) if place > 0 else out


def place_stereo_by_scope(flat_name: str, smiles: str) -> Optional[str]:
    """``flat_name`` with the stereo of ``smiles`` placed unit by unit (see the module docstring), or None. The
    name is recorded as a non-PIN fragment."""
    from ..assembly.memo import cache_or_compute
    name = cache_or_compute("prefix_stereo", (flat_name, smiles), lambda: _placed(flat_name, smiles))
    if name:
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
    return name


def scoped_over_front(composed: Optional[str], flat_name: str, smiles: str, in_key: str) -> Optional[str]:
    """``composed`` (the front composition of ``namer._try_compose_input_stereo``), or the name of
    :func:`placed_in_prefixes` in its place when that name reads back to ``in_key`` too and ``composed`` does."""
    if not composed or composed == flat_name or not composed_reads_back(composed, smiles, in_key):
        return composed
    scoped = placed_in_prefixes(flat_name, smiles)
    return scoped if scoped and composed_reads_back(scoped, smiles, in_key) else composed


def placed_in_prefixes(flat_name: str, smiles: str) -> Optional[str]:
    """``flat_name`` with the stereo of ``smiles`` placed unit by unit, when that places a descriptor at the front of
    an enclosed prefix; None when every descriptor stands at the front of the name (the caller's front composition,
    byte for byte the same places) or when no place is found.

    The front composition (``namer._try_compose_input_stereo``) numbers the stereo-free name with OPSIN's labels,
    which give every substituent atom its own locant: a double bond inside a substituent was cited at the front with
    the substituent's locant ('(1E)-4-{...[(...-5-ylidene)methyl]phenoxy}butanoic acid', C-1 of 'methyl'). OPSIN
    reads it back, because a front descriptor that finds no bond in the parent is tried on every group of the name
    (StereochemistryHandler.assignStereoBond), so no read-back sees it; (the Blue Book): "When they
    relate to substituent groups, they are cited at the front of the corresponding prefix"."""
    if not any(ch in _OPEN for ch in flat_name):
        return None
    from ..assembly.memo import cache_or_compute
    name = cache_or_compute("prefix_stereo", (flat_name, smiles), lambda: _placed(flat_name, smiles))
    if not isinstance(name, str):
        return None
    if _without_word_front_blocks(name) == flat_name:
        return None
    return name


def _without_word_front_blocks(name: str) -> str:
    """``name`` without the descriptor blocks that stand at the front of one of its top-level words."""
    fronts = [0] + [i + 1 for i, ch in enumerate(name) if ch == " " and _word_front(name, i + 1) == i + 1]
    out, last = [], 0
    for front in fronts:
        m = re.match(r"\([^()\[\]{}]*\)-", name[front:])
        if m:
            out.append(name[last:front])
            last = front + m.end()
    out.append(name[last:])
    return "".join(out)


def place_pair_by_scope(flat_name: str, smiles: str, labelled, swapped: frozenset) -> Optional[Tuple[str, str]]:
    """``(name, twin)`` for the CIP dialect judge (``validation.cip_dialect``): ``labelled`` is the input labelled
    with the letters OPSIN reads right (the record's, with the letter of each unit of ``swapped`` -- ``('a', atom)``
    or ``('b', bond)`` -- changed to the other one). The places are found exactly as:func:`place_stereo_by_scope`
    finds them, with those letters, and the name with them (the twin) must read back to the input; the name returned
    first cites the same places and locants with the record's letter at the units of ``swapped``. None when a unit
    has no place or the twin does not read back."""
    found = _placed(flat_name, smiles, labelled=labelled, swapped=swapped)
    return found if isinstance(found, tuple) else None


def _placed(flat_name: str, smiles: str, labelled=None, swapped: frozenset = frozenset()):
    from ..namer import (Orthonym, _has_lone_pair_stereocentre, _lone_pair_as_written, _self_consistency_full_key,
                         _validity_gate_name_to_opsin_smiles)
    from ..perception.stereo import assign_stereochemistry
    from ..validation.opsin_roundtrip import opsin_atom_locant_map
    from .stereochemistry import _STEREO_EMBEDDED_RE
    if _has_lone_pair_stereocentre(_lone_pair_as_written(smiles)):
        # the full key does not settle a lone-pair centre (namer._lone_pair_configuration_verified, which this
        # path does not run): such a molecule keeps the caller's decline
        return None
    mol = Chem.MolFromSmiles(smiles) if labelled is None else Chem.Mol(labelled)
    if mol is None or not flat_name or _STEREO_EMBEDDED_RE.search(flat_name):
        return None                         # a name that already cites stereo is not placed again
    if labelled is None:
        assign_stereochemistry(mol)
    units = _units(mol)
    places = _places(flat_name)
    if not units or len(units) > _MAX_UNITS or len(places) > _MAX_PLACES:
        return None
    locants = opsin_atom_locant_map(flat_name, mol)
    if not locants:
        return None                         # without OPSIN's numbering no locant is known
    chosen: Dict[int, List[str]] = {}
    record: Dict[int, List[str]] = {}       # the same places with the record's letters (``swapped``)
    readings = 0
    for unit in units:
        target = _self_consistency_full_key(_single_unit_smiles(mol, unit))
        if not target:
            return None
        hit = None
        for place in places:
            for text in _texts(mol, unit, locants):
                readings += 1
                if readings > _MAX_READINGS:
                    return None
                read = _validity_gate_name_to_opsin_smiles(_with_block(flat_name, place, text))
                if read and _self_consistency_full_key(read) == target:
                    hit = (place, text)
                    break
            if hit:
                break
        if hit is None:
            return None
        if unit[0] == "b" and hit[0] != 0:
            # the double bond that joins a '-ylidene' or '-imino' prefix to the part that holds it:
            # (1)(a) (the Blue Book) gives the PIN form, the descriptor at the front of that part with the
            # locant of the point of attachment ('(1Z)-1-(...azepan-4-ylidene)propan-2-one'), not (b) at the front
            # of the '-ylidene' prefix
            host, texts, attached = _attachment_texts(flat_name, hit[0], unit[2])
            if not _is_attachment_bond(mol, unit[1], locants, attached):
                texts = []          # a double bond inside the '-ylidene' group ('prop-2-en-1-ylidene', C-2=C-3) stays
            for text in texts:
                readings += 1
                if readings > _MAX_READINGS:
                    return None
                read = _validity_gate_name_to_opsin_smiles(_with_block(flat_name, host, text))
                if read and _self_consistency_full_key(read) == target:
                    hit = (host, text)
                    break
        chosen.setdefault(hit[0], []).append(hit[1])
        if (unit[0], unit[1]) in swapped:
            other = {"R": "S", "S": "R", "r": "s", "s": "r", "E": "Z", "Z": "E"}[hit[1][-1]]
            record.setdefault(hit[0], []).append(hit[1][:-1] + other)
        else:
            record.setdefault(hit[0], []).append(hit[1])
    for texts in chosen.values():
        # a bare descriptor only alone in its block, and no locant twice in one block: OPSIN gives the
        # descriptors of such a block to the centres in the order it reads them, so the name would rest on
        # that order (CID10929611: '(R,S,1R,24S)-' read back only by that order)
        locs = [t[:-1] for t in texts]
        if (len(texts) > 1 and "" in locs) or len(set(locs)) != len(locs):
            return None
    name = flat_name
    for place in sorted(chosen, reverse=True):
        name = _with_block(name, place, ",".join(sorted(chosen[place], key=_descriptor_key)))
    from ..validation.spelling.checks_marks import enclosing_marks_check
    read = _validity_gate_name_to_opsin_smiles(name)
    if (not read or not Orthonym._rt_match(smiles, read, False)
            or enclosing_marks_check(mol, name) is not None):
        # the exact isomeric comparison sees the E/Z of a C=N the key cannot (amidines, amidoximes)
        return None
    if labelled is None:
        return name
    first = flat_name
    for place in sorted(record, reverse=True):
        first = _with_block(first, place, ",".join(sorted(record[place], key=_descriptor_key)))
    return first, name


def _descriptor_key(text: str):
    """ (the Blue Book): "Multiple stereodescriptors are cited in the increasing order of their
    respective locants" (a bare descriptor has none and stands first)."""
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else -1, text)


def composed_reads_back(composed: str, smiles: str, in_key: str) -> bool:
    """OPSIN reads ``composed`` to ``in_key``, with the input's protonation site and radical graph."""
    from ..namer import _self_consistency_full_key, _validity_gate_name_to_smiles
    composed_smiles = _validity_gate_name_to_smiles(composed)
    if composed_smiles is None or _self_consistency_full_key(composed_smiles) != in_key:
        return False
    # the key does not place a hydron (validation/protonation_identity.py)
    from ..validation.protonation_identity import protonation_site_verdict
    if protonation_site_verdict(smiles, composed_smiles, composed) == "mismatch":
        return False
    # nor the radical graph (validation/radical_identity.py)
    from ..validation.radical_identity import radical_identity_verdict
    return radical_identity_verdict(smiles, composed_smiles) != "mismatch"
