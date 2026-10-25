"""CIP descriptors of a phane name,, (j)).

the Blue Book, each line read with ``sed -n``:

* "NAMING OF STEREOISOMERS" (:44639): a descriptor "is preceded by a numerical or letter locant to
  describe the position of the stereogenic unit"; descriptors "are placed at the front of the complete name
  when related to the parent structure" and, "when they relate to substituent groups, they are cited at the
  front of the corresponding prefix". A stereogenic unit inside a substituent is therefore named by the
  substituent namer, which cites it in its prefix (``(1R)-1-bromoethyl``); only the units of the skeleton are
  cited here. The round trip of the producer tells whether the whole name describes the input.
* (:15111): an atom of an amplificant has a composite locant, the locant of its superatom with the
  locant of the atom in the amplificant as superscript (``5^1S``); the descriptors of the examples at:50787
  and:50791 are in the order of the primary locants and then of the superscripts: ``(2E,5^1S,5^2R,5^5S,8E,
  11E,15^1S,15^2R,15^4R,17E)``.
* "Specification of double bonds" (:49684): "The omission of stereodescriptors 'E' and 'Z' to denote
  the configuration of double bonds in cyclophanes is related to the total number of nodes in the system; for
  up to seven nodes, no descriptor is necessary. When more than seven nodes are present, all descriptors must
  be cited." A cyclic skeleton of seven nodes or fewer with a specified double bond is not named: OPSIN does not
  read the omitted descriptor as the configuration, so no read-back could show that the name is the input's.
* (j) (:3346): "When there is a choice for lower locants related to the presence of stereogenic centers or
  stereoisomers, the lower locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that
  are preferred to E, S, P, and s, respectively". The rule is applied by ``numbering``.

The labels come from ``perception.stereo.assign_stereochemistry`` (the ``centres`` engine first), never from a
raw RDKit call. A structure with an axial or planar element, a pseudoasymmetric label other than ``r`` and
``s``, or a double bond between a skeleton atom and a substituent or inside an amplificant is not named.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem

from ...perception.stereo import assign_stereochemistry, detect_axial_chirality
from .skeleton import PhaneSkeleton, SkeletonKind

#: the labels of a stereogenic centre and of a double bond that a descriptor block cites
_CENTRE_CODES = frozenset({"R", "S", "r", "s"})
_BOND_CODES = frozenset({"E", "Z"})
#:: a cyclic skeleton cites the descriptors of its double bonds from eight nodes on
_MIN_NODES_CITING_BONDS = 8


@dataclass(frozen=True)
class Unit:
    """A stereogenic unit of the skeleton: the atom of a centre, or the two atoms of a double bond."""

    code: str
    atoms: Tuple[int, ...]


def _units(skel: PhaneSkeleton) -> Optional[Tuple[Unit, ...]]:
    mol = skel.mol
    assign_stereochemistry(mol)
    if any(element.get("cip") is not None for element in detect_axial_chirality(mol)):
        return None
    skeleton = {atom for atom in range(mol.GetNumAtoms()) if skel.node_of(atom) is not None}
    units: List[Unit] = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        code = atom.GetProp("_CIPCode")
        if code not in _CENTRE_CODES:
            return None
        if atom.GetIdx() in skeleton:
            units.append(Unit(code, (atom.GetIdx(),)))
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or not bond.HasProp("_CIPCode"):
            continue
        code = bond.GetProp("_CIPCode")
        if code not in _BOND_CODES:
            return None
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        inside = (begin in skeleton, end in skeleton)
        if inside == (False, False):
            continue                       # inside a substituent: its prefix cites it
        nodes = (skel.node_of(begin), skel.node_of(end)) if all(inside) else None
        if nodes is None or nodes[0][0] != "atom" or nodes[1][0] != "atom":
            return None                    # an ylidene bond, or a bond of an amplificant
        if skel.kind is not SkeletonKind.PATH and len(skel.nodes) < _MIN_NODES_CITING_BONDS:
            return None
        units.append(Unit(code, (begin, end)))
    return tuple(units)


def units_of(skel: PhaneSkeleton) -> Optional[Tuple[Unit, ...]]:
    """The stereogenic units of the skeleton that the name cites, or None when the structure has one the
    name cannot cite. Computed once for a skeleton."""
    cached = skel.__dict__.get("_stereo_cache")
    if cached is None:
        cached = (_units(skel),)
        object.__setattr__(skel, "_stereo_cache", cached)
    return cached[0]


def unit_locant(skel: PhaneSkeleton, pos, chosen: Mapping[int, Mapping[int, Tuple[int, str]]], unit: Unit):
    """The locant of a unit: the atom of a centre, the lower of the two atoms of a double bond."""
    from .numbering import atom_locant
    return min(atom_locant(skel, pos, chosen, atom) for atom in unit.atoms)


def descriptors(skel: PhaneSkeleton, pos, chosen, units: Sequence[Unit]) -> List[Tuple[tuple, str]]:
    """``(locant, code)`` of every unit, in the order of the locants (primary, then superscript)."""
    return sorted((unit_locant(skel, pos, chosen, unit), unit.code) for unit in units)


_DESCRIPTOR_BLOCK = re.compile(r"^\(([^()]*)\)-")
_DESCRIPTOR_CODE = re.compile(r"(?:^|,)[\d']*[a-z]?([RSrsEZMP])(?=,|$)")


def prefix_codes(name: str) -> List[str]:
    """The descriptors at the front of the name of a substituent prefix ('(1E)-prop-1-en-1-yl' gives ['E']; '(1R,2S)-...'
    gives ['R', 'S']); none for a descriptor of another form."""
    block = _DESCRIPTOR_BLOCK.match(name)
    return _DESCRIPTOR_CODE.findall(block.group(1)) if block else []


def rank_key(skel: PhaneSkeleton, pos, chosen, units: Sequence[Unit], extra: Sequence[tuple] = ()) -> tuple:
    """ (j): the key of one numbering, lower is preferred (see ``cip_locant_rank_key``). ``extra`` are the
    ``(locant, code)`` pairs of the descriptors inside the prefixes, each at the locant of the atom its prefix is attached to
     (j),:3346 and its example '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)',:3399: the R prefix has
    the lower locant)."""
    if not units and not extra:
        return ()
    from ...assembly.naming_utils import cip_locant_rank_key
    return cip_locant_rank_key(sorted(descriptors(skel, pos, chosen, units) + list(extra)))


def block(skel: PhaneSkeleton, pos, chosen, units: Sequence[Unit]) -> str:
    """``(2E,5^1S,5^2R)-``, or nothing."""
    if not units:
        return ""
    from .numbering import render_locant
    return "(" + ",".join(f"{render_locant(loc)}{code}" for loc, code in descriptors(skel, pos, chosen, units)) + ")-"


def stereo_atoms(units: Sequence[Unit]) -> Dict[int, None]:
    """Every atom a unit is read from (an ordered set)."""
    return {atom: None for unit in units for atom in unit.atoms}
