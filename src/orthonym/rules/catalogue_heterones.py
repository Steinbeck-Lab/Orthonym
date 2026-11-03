"""The heterone of a catalogued fused ring system with =O on a ring chalcogen in the lambda state: the sultone and the
sultam of naphthalene ('2H-2λ6-naphtho[1,8-cd][1,2]oxathiole-2,2-dione', '1λ6-naphtho[1,8-cd][1,2]thiazole-1,1(2H)-
dione').

``###...`` (the Blue Book): "Sultones and sultines are intramolecular esters of hydroxy
sulfonic acids and sulfinic acids, respectively, and may be named in three ways: (1) as heterocyclic heterones by
adding the suffix 'one', 'dione',... to the name of the heterocyclic parent hydride;... Method (1) gives preferred
IUPAC names", '2H-2λ6-naphtho[1,8-cd][1,2]oxathiole-2,2-dione (PIN)' (:32160); the sultams likewise (:33248),
'1λ6-naphtho[1,8-cd][1,2]thiazole-1,1(2H)-dione (PIN)' (:33254). (:12452): the lambda symbol follows
the locant of the atom in the numbering of the fused system, cited at the front of the name.

The parent and its numbering come from the fused-ring catalogue (``data.fused_heterocycles``, OPSIN-derived locants);
the only atoms off the ring system are the =O groups of one chalcogen. The hydrogen follows the two book names: with a
ring N-H the parent is mancude without indicated hydrogen and the N-H is added hydrogen after the suffix locants
('1,1(2H)-dione'); without one the lambda atom carries the indicated hydrogen ('2H-2λ6-'). Each spelling is read back
by its full InChIKey before it is returned, else None. A parent whose indicated hydrogen stands on a ring carbon
('3H-2,1-benzoxathiole') keeps it, and the name is recorded below the PIN (no book name of that kind).
"""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem

from ..data.locant_text import render_locants

_CHALCOGENS = frozenset({"S", "Se", "Te"})
_DI = {1: "", 2: "di"}


def catalogue_heterone_name(mol: Optional[Chem.Mol]) -> Optional[str]:
    from ..data.fused_heterocycles import match_fused_heterocycle_core
    from .lambda_convention import fused_lambda_entries, format_lambda_token, merge_lambda_into_benzo_locants
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol):
        return None
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    outside = [a for a in mol.GetAtoms() if a.GetIdx() not in ring_atoms]
    hosts = {a.GetNeighbors()[0].GetIdx() for a in outside if a.GetDegree() == 1}
    if (not outside or len(hosts) != 1 or any(
            a.GetSymbol() != "O" or a.GetDegree() != 1 or a.GetIsotope()
            or a.GetBonds()[0].GetBondType() != Chem.BondType.DOUBLE for a in outside)):
        return None
    host = mol.GetAtomWithIdx(next(iter(hosts)))
    if host.GetSymbol() not in _CHALCOGENS or len(outside) > 2:
        return None
    core = match_fused_heterocycle_core(mol) or _core_of_the_bare_parent(mol, outside)
    if core is None:
        return None
    parent, atom_to_locant = core[0], core[1]
    if set(atom_to_locant) != ring_atoms:
        return None
    entries = fused_lambda_entries(mol, ring_atoms, atom_to_locant)
    if not entries or len(entries) != 1:
        return None
    loc, lam = entries[0]
    held = re.match(r"^(\d+[a-z]?H)-", parent)
    stem = parent[held.end():] if held else parent
    if not stem.endswith("e"):
        return None
    suffix_locants = render_locants([loc] * len(outside))
    suffix = _DI[len(outside)] + "one"
    if suffix == "one":
        stem = stem[:-1]    # the final 'e' is elided before a vowel: '1H-1λ4-thiophen-1-one (PIN)' (:29440)
    nh = [atom_to_locant[i] for i in ring_atoms
          if mol.GetAtomWithIdx(i).GetSymbol() == "N" and mol.GetAtomWithIdx(i).GetTotalNumHs() == 1]
    token = format_lambda_token(loc, lam)
    if held and not nh and mol.GetAtomWithIdx(next(i for i, v in atom_to_locant.items()
                                                    if f"{v}H" == held.group(1))).GetSymbol() == "C":
        # the parent's indicated hydrogen on a ring carbon stays; the lambda token joins the benzo heteroatom
        # locants ('3H-2,1λ6-benzoxathiole-1,1-dione'). No book name of this kind is printed: valid, not the PIN.
        merged = merge_lambda_into_benzo_locants(stem, mol, ring_atoms, atom_to_locant) or f"{token}-{stem}"
        name = f"{held.group(1)}-{merged}-{suffix_locants}-{suffix}"
        from .charged_router import _full_inchikey_rt_ok
        if not _full_inchikey_rt_ok(mol, name):
            return None
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)
        return name
    if len(nh) == 1:
        name = f"{token}-{stem}-{suffix_locants}({nh[0]}H)-{suffix}"
        n_atom = next(i for i in ring_atoms if atom_to_locant[i] == nh[0])
        if mol.GetBondBetweenAtoms(n_atom, host.GetIdx()) is None:
            # the book's added-hydrogen spelling is the sultam's, an N-H beside the lambda atom (:33254); an N-H
            # elsewhere in the system ('5λ6-phenothiazine-5,5(10H)-dione') is spelled the same way, valid, not the PIN
            from .charged_router import _full_inchikey_rt_ok
            if not _full_inchikey_rt_ok(mol, name):
                return None
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(name)
            return name
    elif not nh and not any(mol.GetAtomWithIdx(i).GetTotalNumHs() and mol.GetAtomWithIdx(i).GetSymbol() != "C"
                            for i in ring_atoms):
        name = f"{loc}H-{token}-{stem}-{suffix_locants}-{suffix}"
    else:
        return None
    from .charged_router import _full_inchikey_rt_ok
    return name if _full_inchikey_rt_ok(mol, name) else None


def _core_of_the_bare_parent(mol: Chem.Mol, outside) -> Optional[tuple]:
    """The catalogue match of the ring system without its =O groups (the lambda atom at its standard bonding
    number), its locants carried back to the atoms of ``mol``; None when the bare system does not sanitize or match."""
    from ..data.fused_heterocycles import match_fused_heterocycle_core
    bare = Chem.RWMol(mol)
    for atom in bare.GetAtoms():
        atom.SetIntProp("_orig", atom.GetIdx())
    for idx in sorted((a.GetIdx() for a in outside), reverse=True):
        bare.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(bare)
    except Exception:  # noqa: BLE001 -- no bare parent, no name
        return None
    bare = bare.GetMol()
    back = {a.GetIdx(): a.GetIntProp("_orig") for a in bare.GetAtoms()}
    core = match_fused_heterocycle_core(bare)
    if core is not None:
        return (core[0], {back[i]: loc for i, loc in core[1].items()}) + tuple(core[2:])
    # the bare parent itself is a catalogue key: its own locant map, moved onto the atoms
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    key = Chem.MolToSmiles(bare)
    entry = FUSED_HETEROCYCLE_DATA.get(key)
    if entry and entry.get("iupac_locants"):
        match = bare.GetSubstructMatch(Chem.MolFromSmiles(key))
        if not match:
            return None
        return (entry["name"], {back[match[k]]: loc for k, loc in entry["iupac_locants"].items()}, key)
    # a two-ring system on its certified parent ('3H-2,1-benzoxathiole'): one numbering only, so that the
    # locants are not a choice made here, and the parent's name with its indicated hydrogen
    from .fused_rings import _certified_two_ring_parent, name_fused_heterocycle
    ring_atoms = {i for ring in bare.GetRingInfo().AtomRings() for i in ring}
    certified = _certified_two_ring_parent(bare, ring_atoms)
    named = name_fused_heterocycle(bare)
    if certified is None or len(certified.numberings) != 1 or not named:
        return None
    numbering = dict(certified.numberings[0])
    if set(numbering) != ring_atoms or not named[0].endswith(certified.name):
        return None
    return (named[0], {back[i]: loc for i, loc in numbering.items()}, key)
