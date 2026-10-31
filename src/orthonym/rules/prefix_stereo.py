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
    innermost first (a descriptor stands at the front of the smallest part it relates to), then the front of
    the name."""
    places, depth = [], 0
    for i, ch in enumerate(name[:-1]):
        if ch in _OPEN:
            depth += 1
            if name[i + 1].isalnum() or name[i + 1] in _OPEN:
                places.append((-depth, i + 1))
        elif ch in ")]}":
            depth -= 1
    return [i for _, i in sorted(places)] + [0]


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


def _placed(flat_name: str, smiles: str) -> Optional[str]:
    from ..namer import (Orthonym, _has_lone_pair_stereocentre, _lone_pair_as_written, _self_consistency_full_key,
                         _validity_gate_name_to_opsin_smiles)
    from ..perception.stereo import assign_stereochemistry
    from ..validation.opsin_roundtrip import opsin_atom_locant_map
    from .stereochemistry import _STEREO_EMBEDDED_RE
    if _has_lone_pair_stereocentre(_lone_pair_as_written(smiles)):
        # the full key does not settle a lone-pair centre (namer._lone_pair_configuration_verified, which this
        # path does not run): such a molecule keeps the caller's decline
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not flat_name or _STEREO_EMBEDDED_RE.search(flat_name):
        return None                         # a name that already cites stereo is not placed again
    assign_stereochemistry(mol)
    units = _units(mol)
    places = _places(flat_name)
    if not units or len(units) > _MAX_UNITS or len(places) > _MAX_PLACES:
        return None
    locants = opsin_atom_locant_map(flat_name, mol)
    if not locants:
        return None                         # without OPSIN's numbering no locant is known
    chosen: Dict[int, List[str]] = {}
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
        chosen.setdefault(hit[0], []).append(hit[1])
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
    return name


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
