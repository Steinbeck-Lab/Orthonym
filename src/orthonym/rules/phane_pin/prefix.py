"""A linear phane as a substituent group,,,.

the Blue Book, each line read with ``sed -n``:

* (:16148): "Substituent groups derived from cyclophanes are formed by applying the principles described
  above"; the book prints one: ``trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane (PIN)``
  (:19325, under: the silane is the parent because Si is senior to O).
* (:18875) and (:18917) decide which parent is senior. The reader of the spelling check
  (``validation.spelling.linear_phane``) applies them to the whole structure: its verdict ``as_prefix`` says that the
  phane is not the parent, so the PIN cites it in a prefix. A prefix is built here only for such a structure.
*: the suffix of a substituent group is its free valence ('yl'); the characteristic groups in it are prefixes.
  The free valence is a suffix for the numbering (c) to (e)), which ``numbering`` already reads for a suffix group.

The fragment of the structure that is the substituent group is read as a molecule of its own, with a dummy atom for the
parent. The chain is the longest linear phane chain in it (``linear_phane_candidate_chains``), the free valence is on
one of its skeleton atoms, and everything else in the fragment is a prefix. The labels of the stereogenic units are the
ones of the whole structure,:44639: the descriptors of a substituent group are cited at the front of its
prefix, and a name states the configuration of the whole structure). A fragment whose attachment is a stereogenic unit
is not built. The name is read back by OPSIN as a radical and compared, with a hydrogen atom for the free valence, to the
fragment.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple

from rdkit import Chem

from ...perception.stereo import _CIP_ASSIGNED_PROP, assign_stereochemistry
from ...validation.spelling.linear_phane import (
    linear_phane_candidate_chains,
    linear_phane_pin_expected,
)
from . import _kekule, _plain

_CACHE: dict = {"mol": None, "prefix": False}
#: the prefix of each (fragment, attachment) of the last few molecules: the substituent namers ask for the same fragment
#: of one molecule many times (108 calls for 4 distinct fragments on one porphyrin), and building a name costs a
#: perception of every chain of the fragment. The molecule is kept so that its id is not reused.
_MEMO_MOLECULES = 8
_MEMO: "OrderedDict[int, Tuple[Chem.Mol, Tuple[int, int], Dict[tuple, Optional[str]]]]" = OrderedDict()


def _whole_structure_cites_a_prefix(mol: Chem.Mol) -> bool:
    """The reader's verdict for the whole structure is ``as_prefix`` (the PIN is not named by the phane parent) and the
    PIN is not a multiplicative name with the phane as its central group,:23178: two identical parents hold every
    principal group). Memoised for the last molecule."""
    if _CACHE["mol"] is not mol:
        verdict = linear_phane_pin_expected(mol)
        _CACHE["mol"], _CACHE["prefix"] = mol, bool(verdict is not None and verdict.as_prefix
                                                    and verdict.criterion != "P-51.3.1")
    return _CACHE["prefix"]


def _fragment(mol: Chem.Mol, frag: frozenset, attach: int) -> Optional[Tuple[Chem.Mol, int]]:
    """The fragment as a molecule with a dummy atom bonded to ``attach`` in the place of the parent, and the index of
    the dummy atom; None when the fragment is bonded to the rest of the structure by anything but one single bond or
    when the attachment atom is a stereogenic unit (the dummy atom would change what its descriptor reads)."""
    outside = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors() if n.GetIdx() not in frag]
    if len(outside) != 1:
        return None
    if mol.GetBondBetweenAtoms(attach, outside[0]).GetBondType() != Chem.BondType.SINGLE:
        return None
    atom = mol.GetAtomWithIdx(attach)
    if atom.HasProp("_CIPCode") or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return None
    if any(bond.HasProp("_CIPCode") for bond in atom.GetBonds()):
        return None
    for index in frag:
        if any(n.GetIdx() not in frag and index != attach for n in mol.GetAtomWithIdx(index).GetNeighbors()):
            return None                 # the fragment is bonded to the structure at another atom
    rw = Chem.RWMol(mol)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(attach, dummy, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(dummy).SetIntProp("_phane_free_valence", 1)
    for index in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in frag and a.GetIdx() != dummy),
                        reverse=True):
        rw.RemoveAtom(index)
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:  # noqa: BLE001 -- RDKit raises several types
        return None
    sub.SetProp(_CIP_ASSIGNED_PROP, "1")        # the labels are those of the whole structure
    marked = [a.GetIdx() for a in sub.GetAtoms() if a.HasProp("_phane_free_valence")]
    return (sub, marked[0]) if len(marked) == 1 else None


def _closed(sub: Chem.Mol, dummy: int) -> Optional[Chem.Mol]:
    """The hydride of the fragment: the dummy atom replaced by a hydrogen atom."""
    rw = Chem.RWMol(sub)
    rw.RemoveAtom(dummy)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        return None
    return out


def _radical_hydride(smiles: str) -> Optional[Chem.Mol]:
    """The hydride of the radical OPSIN reads from a substituent name: a hydrogen atom on the radical centre."""
    read = Chem.MolFromSmiles(smiles)
    if read is None:
        return None
    centres = [a for a in read.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(centres) != 1:
        return None
    rw = Chem.RWMol(read)
    atom = rw.GetAtomWithIdx(centres[0].GetIdx())
    hydrogens = atom.GetTotalNumHs()
    atom.SetNumRadicalElectrons(0)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(hydrogens + 1)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        return None
    return out


def _read_back(name: str, sub: Chem.Mol, dummy: int) -> bool:
    from ...validation.opsin_roundtrip import opsin_parse
    parsed = opsin_parse(name, allow_radicals=True)
    if not parsed:
        return False
    given, read = _closed(sub, dummy), _radical_hydride(parsed)
    if given is None or read is None:
        return False
    if any(a.HasProp("_CIPCode") for a in sub.GetAtoms()) or any(b.HasProp("_CIPCode") for b in sub.GetBonds()):
        # the configuration of a group is read in the structure it is part of: OPSIN gives a descriptor in the
        # radical alone other CIP priorities (the free valence has no neighbour), so the constitution is compared
        # here and the full InChIKey of the whole name is the engine's read-back
        return (Chem.MolToSmiles(given, isomericSmiles=False) == Chem.MolToSmiles(read, isomericSmiles=False)
                and Chem.MolToInchiKey(given)[:14] == Chem.MolToInchiKey(read)[:14])
    return (Chem.MolToSmiles(given) == Chem.MolToSmiles(read)
            and Chem.MolToInchiKey(given) == Chem.MolToInchiKey(read))


def _name(sub: Chem.Mol, dummy: int) -> Optional[str]:
    from .linear import perceive_linear
    from .naming import phane_candidate
    from .substituents import plan_substituents
    if not _plain(sub):
        return None
    chains = linear_phane_candidate_chains(sub)
    if not chains:
        return None
    kekule = _kekule(sub)
    candidates = []
    for chain in chains:
        skel = perceive_linear(kekule, chain)
        plan = plan_substituents(skel, free_valence=dummy) if skel is not None else None
        got = phane_candidate(skel, plan) if plan is not None else None
        if got is not None:
            candidates.append(got)
    return min(candidates)[1] if candidates else None


def _memo_of(mol: Chem.Mol) -> Dict[tuple, Optional[str]]:
    """The memo of ``mol``: dropped when the molecule has changed size since it was filled."""
    shape = (mol.GetNumAtoms(), mol.GetNumBonds())
    held = _MEMO.get(id(mol))
    if held is None or held[0] is not mol or held[1] != shape:
        held = (mol, shape, {})
        _MEMO[id(mol)] = held
        while len(_MEMO) > _MEMO_MOLECULES:
            _MEMO.popitem(last=False)
    return held[2]


def _prefix(mol: Chem.Mol, frag: frozenset, attach_idx: int) -> Optional[str]:
    if attach_idx not in frag:
        attach_idx = next((n.GetIdx() for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors() if n.GetIdx() in frag), None)
        if attach_idx is None:
            return None
    if sum(1 for ring in mol.GetRingInfo().AtomRings() if frag.issuperset(ring)) < 4:
        return None                     # fewer than four rings in the fragment: no linear phane
    if not _whole_structure_cites_a_prefix(mol):
        return None
    assign_stereochemistry(mol)
    built = _fragment(mol, frag, attach_idx)
    if built is None:
        return None
    sub, dummy = built
    name = _name(sub, dummy)
    return name if name is not None and _read_back(name, sub, dummy) else None


def phane_prefix(mol: Optional[Chem.Mol], frag_atoms, attach_idx: int) -> Optional[str]:
    """The substituent prefix of the linear phane in the fragment ``frag_atoms`` of ``mol``, attached to the rest of the
    structure at ``attach_idx`` (the atom of the fragment or the atom of the parent, as the substituent namers pass it),
    or None: with an OPSIN that reads no phane name, when the whole structure does not cite a phane prefix, when the
    fragment holds no linear phane chain or is bonded to the structure in another way than by one single bond, when a
    group of it is not built, or when the name does not read back to the fragment."""
    from ...opsin_capabilities import phane_enabled
    if mol is None or not phane_enabled():
        return None
    frag = frozenset(frag_atoms)
    if len(frag) < 20:
        return None                     # four rings, three atoms each, and three chain atoms
    memo = _memo_of(mol)
    key = (frag, attach_idx)
    if key not in memo:
        memo[key] = _prefix(mol, frag, attach_idx)
    return memo[key]
