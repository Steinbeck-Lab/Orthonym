"""E/Z of a double bond that joins a ring atom to a ring atom of a '-ylidene' substituent.

``rules.stereochemistry.ring_ylidene_bonds`` names the bonds and carries the Blue Book rule
, the Blue Book-48279): a stereo double bond with a ring atom at each end is
cited by neither scope of the floor's name, so the name reads back to the right skeleton with that
bond's configuration undefined and the full-key gate refuses it. This module is the floor's use of
that licence: which scope cites the bond, and which bonds the gate can check.

The unit is ONE double bond, so exactly one scope cites it (two descriptors for one bond read back
as no structure). Two modes, each a candidate the floor offers beside its existing rungs:

* ``'host'`` method (a): the scope that holds the bond's parent-side atom cites it with that
               atom's locant, at the front of its own name (the whole name for the top spine, the
               host's prefix for a branch);
* ``'branch'`` method (b): the branch cites it with the locant of its own ylidene atom, at the
               front of its prefix ('[(5Z)-...-5-ylidene]').

Only the bonds the standard InChIKey can check are cited (``key_visible_ring_ylidene_bonds``).
For a bond the key does not encode (a charge-separated or tautomer-normalised drawing, such as a
squaraine dye) a wrong sense and the right one give the same key, so the gate cannot tell them
apart and the descriptor would be unchecked; such a bond is left uncited, as before. For a bond the
key does encode, a wrong locant or a wrong sense changes the key and the full-InChIKey gate
(``validation.reconstruct.verify_or_none``) refuses the name.

Nothing here runs unless the molecule has such a bond, so every other molecule keeps its
candidates byte for byte. For a molecule that has one, the best-effort ladder (``t4_coverage``)
runs its existing spellings first, with this rung suspended (``ring_ylidene_suspended``), and
offers it only to a molecule those leave without a verified name: a named row keeps its name.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional, Tuple

from rdkit import Chem

from ..rules.stereochemistry import (
    collect_stereodescriptors,
    format_stereodescriptor_string,
    merge_stereodescriptors,
    ring_ylidene_bonds,
)

HOST = "host"
BRANCH = "branch"

#: True while the floor must not offer its ring=ring rung (rung 1d) nor compose the stereo of a
#: rung name: the best-effort ladder's first pass, which is the behaviour before this lane.
_SUSPENDED: ContextVar[bool] = ContextVar("ring_ylidene_suspended", default=False)


@contextlib.contextmanager
def ring_ylidene_suspended() -> Iterator[None]:
    """Run the floor as it was before the ring=ring rung existed."""
    token = _SUSPENDED.set(True)
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


def is_suspended() -> bool:
    return _SUSPENDED.get()


@dataclass(frozen=True)
class RingYlidene:
    """One rebuild's citing rule: the scope that cites (``mode``) and the indices of the
    ring=ring bonds it cites (the ones the key can check)."""
    mode: str
    bonds: FrozenSet[int]


def key_sees_bond(mol, bond) -> bool:
    """True iff the standard InChIKey of ``mol`` changes when the stereo of ``bond`` is removed,
    i.e. the key encodes this bond's configuration and a name that gets it wrong cannot pass a
    full-key comparison."""
    rw = Chem.RWMol(mol)
    target = rw.GetBondWithIdx(bond.GetIdx())
    target.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in (target.GetBeginAtom(), target.GetEndAtom()):
        for nb in atom.GetBonds():
            if nb.GetIdx() != target.GetIdx():
                nb.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolToInchiKey(rw) != Chem.MolToInchiKey(mol)


def key_visible_ring_ylidene_bonds(mol) -> List["Chem.Bond"]:
    """The ring=ring stereo double bonds of ``mol`` whose configuration the InChIKey encodes."""
    return [b for b in ring_ylidene_bonds(mol) if key_sees_bond(mol, b)]


def ring_ylidene_descriptors(mol, atom_to_locant: Dict[int, object], spec: RingYlidene,
                             attach_bond: Optional["Chem.Bond"] = None
                             ) -> List[Tuple[object, str]]:
    """The ``(locant, 'E'|'Z')`` pairs a scope cites for the ring=ring bonds of ``spec``.

    ``atom_to_locant`` is the numbering of the scope (a spine); ``attach_bond`` is the bond that
    joins the scope, as a branch, to its host (``None`` for the top spine). A bond with exactly
    one end in the numbering leaves the scope: in ``'host'`` mode the scope cites it unless it is
    its own attachment bond (the host cites that one); in ``'branch'`` mode the scope cites only
    its own attachment bond (the child cites the others)."""
    attach_idx = attach_bond.GetIdx() if attach_bond is not None else None
    out = []
    for bond in ring_ylidene_bonds(mol):
        if bond.GetIdx() not in spec.bonds:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in atom_to_locant) == (b in atom_to_locant):
            continue  # both ends here (a bond of the spine) or neither (a deeper scope's)
        is_attach = bond.GetIdx() == attach_idx
        if is_attach != (spec.mode == BRANCH):
            continue
        out.append((atom_to_locant[a if a in atom_to_locant else b], bond.GetProp("_CIPCode")))
    return out


def scope_block(mol, atom_to_locant: Dict[int, object], spec: RingYlidene, skip_bonds=(),
                attach_bond: Optional["Chem.Bond"] = None) -> str:
    """The descriptor block ('(2R,5Z)-') of one scope with its ring=ring double bonds cited as
    ``spec`` says; '' when the scope has no descriptor."""
    base = collect_stereodescriptors(mol, atom_to_locant, skip_bonds=skip_bonds)
    extra = ring_ylidene_descriptors(mol, atom_to_locant, spec, attach_bond)
    return format_stereodescriptor_string(merge_stereodescriptors(base, extra))
