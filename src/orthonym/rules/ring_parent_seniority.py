"""Label-only guard: a name with no suffix built on a ring system that loses to another ring system of the molecule.

 (the Blue Book): "general criteria for determining ring seniority given below are applied successively
until there are no alternatives remaining" -- (a) a heterocycle, (b) a nitrogenous one, (c) the senior heteroatom,
(d) more rings, (e) more ring atoms, (f) more heteroatoms, (g) the greater variety (:19412-19418); applied when "there
must not be a characteristic group present in the compound",:19422); '2-[(naphthalen-2-yl)methyl]pyridine
(PIN) (pyridine > naphthalene)' (:19426). So '2-(1H-indol-2-yl)-9H-fluorene' (PIN '2-(9H-fluoren-2-yl)-1H-indole') is a
correct name that is not the PIN.

The parent ring of the name is read from its parent word (the parent hydride with its hydro and indicated-hydrogen
parts) through OPSIN and matched to one ring system of the molecule by its ring atoms; the ring systems are compared
by criteria (a)-(f) of ``ring_selection.ring_system_score``. A ring system that wins only on a later criterion, a
parent that matches no ring system or more than one, a name with a suffix or a suffixable group, a charge, an
isotope, a second fragment, a phane or a primed name leaves the label as it was. Rows for the spelling step of
``Orthonym.name_tiered``.
"""
from __future__ import annotations

from collections import Counter

from rdkit import Chem

_SUFFIXABLE = Chem.MolFromSmarts('[$([#6]=[O,S,Se,Te]),$([#6;!R]=[#7]),$([OX2H1]),$([SX2H1]),$([NX3;H1,H2;!R]),$([#6]#[NX1])]')
_NITRO = Chem.MolFromSmarts('[N+](=O)[O-]')
#: an amine of any degree, a characteristic group cited as a suffix), an amide-type N-sulfonyl or N-phosphoryl
#: group, an azo group or an acyclic parent-hydride atom decides the parent before ring seniority
_DECIDED_EARLIER = [Chem.MolFromSmarts(q) for q in (
    '[NX3;!R;!$(N=*);!$(N#*)]', '[#7]-[SX4](=O)=O', '[#7]-[SX3]=O', '[#7]-[PX4]=O', '[#7;!R]=[#7;!R]',
    '[Si,Ge,Sn,Pb,P,As,Sb,Bi,B,Al;!R]')]
#: (a)-(f) occupy the first 6 places of the ring-system score tuple; criterion (g) is not read (the score
#: counts N first, the book's sequence for (g) puts N after Te,:19418)
_GENERAL_CRITERIA = 6


def _ring_systems(mol):
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        rest = [s for s in systems if not (s & merged)]
        for s in systems:
            if s & merged:
                merged |= s
        systems = rest + [merged]
    return systems


def _signature(mol, atoms):
    return Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms)


def _parent_word(name):
    """The parent hydride of a one-word suffix-free name, with its hydro and indicated-hydrogen parts, or None."""
    from ..data.locant_text import render_locants
    from ..validation.spelling.models import P10, read_units
    if not name or ' ' in name.strip() or "'" in name or 'phane' in name or P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    tops = [p for p in parsed[2] if p.unit.encl is None]
    if len(tops) != 1:
        return None
    top = tops[0]
    if not top.parent or top.endings or top.repl or top.multiplied:
        return None
    word = top.parent
    if top.parent_locs:
        word = f"{render_locants(top.parent_locs)}-{word}"
    if top.ih:
        parts = [x for item in top.ih for x in (item if isinstance(item, (tuple, list)) else (item,))]
        if not parts or not all(isinstance(x, str) for x in parts):
            return None
        word = f"{render_locants(parts)}-{word}"
    for locs, raw in reversed(top.hydro or []):
        word = (f"{render_locants(locs)}-{raw}" if locs else raw) + ('-' if word[:1].isdigit() else '') + word
    return word


def ring_parent_seniority_rows(smiles, name) -> list:
    """[{'rule': '', 'detail':...}] when the name's parent ring system loses on (a)-(f), else ."""
    try:
        word = _parent_word(name)
        if word is None:
            return []
        mol = Chem.MolFromSmiles(smiles)
        if (mol is None or len(Chem.GetMolFrags(mol)) != 1 or mol.HasSubstructMatch(_SUFFIXABLE)
                or any(mol.HasSubstructMatch(q) for q in _DECIDED_EARLIER)):
            return []
        nitro = {i for m in mol.GetSubstructMatches(_NITRO) for i in m}
        if any((a.GetFormalCharge() and a.GetIdx() not in nitro) or a.GetIsotope() or a.GetNumRadicalElectrons()
               for a in mol.GetAtoms()):
            return []
        systems = _ring_systems(mol)
        if len(systems) < 2:
            return []
        from ..validation.opsin_roundtrip import opsin_parse
        parent_smiles = opsin_parse(word)
        parent = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
        if parent is None:
            return []
        parent_systems = _ring_systems(parent)
        if len(parent_systems) != 1 or len(parent_systems[0]) != parent.GetNumAtoms():
            return []                   # the parent word must be one bare ring system (no unlocanted suffix)
        sig = _signature(parent, parent_systems[0])
        matches = [s for s in systems if _signature(mol, s) == sig]
        if len(matches) != 1:
            return []
        from .ring_selection import ring_system_score
        mine = ring_system_score(mol, matches[0])[:_GENERAL_CRITERIA]
        best = min(ring_system_score(mol, s)[:_GENERAL_CRITERIA] for s in systems)
        if best < mine:
            return [{'rule': 'P-44.2.1', 'detail': "another ring system of the molecule is senior by the general "
                                                   "criteria of ring seniority; it is the parent of the PIN"}]
        return []
    except Exception:  # noqa: BLE001 -- no answer: no label change
        return []
