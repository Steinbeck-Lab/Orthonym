"""Catenated Group-14 / chalcogen(+N) hydride namer (P-21.2.3 / P-52.1.3).

Alternating homonuclear a(ba)n catenated parent hydrides — a chain of identical
Group-14 atoms {Si, Ge, Sn, Pb} linked by identical bridge heteroatoms
{O, N, S, Se, Te}, terminated at both ends by a Group-14 atom, everything else
saturated with hydrogen::

    [SiH3]O[SiH3]              -> disiloxane        (2 Si, 1 O)
    [SiH3]O[SiH2]O[SiH3]       -> trisiloxane       (3 Si, 2 O)
    [SiH3]N[SiH3]              -> disilazane        (N bridge)
    [SiH3]S[SiH3]              -> disilathiane      (S bridge; linking 'a')
    [SnH3]O[SnH3]              -> distannoxane      (Sn)
    [GeH3]O[GeH3]              -> digermoxane       (Ge)

Every emitted name round-trips through OPSIN 2.9.0.

SCOPE (fail-closed, accuracy-first): a single unbranched chain that strictly
alternates one Group-14 element with one bridge element, terminated by two
Group-14 atoms, all H-saturated, neutral, non-radical, acyclic, single fragment.
ANY carbon, branch, mixed Group-14 kind, mixed bridge kind, or extra substituent
fails a guard and returns None (cascade-continuation). This is the boundary the
skeletal-replacement engine deliberately declines (Gate 3b) — pure homonuclear
Si-O-Si chains route here; mixed Si-O-C-S chains stay with skeletal 'a'.

Graph/atom classifier — no SMARTS broadening; pure (no mol mutation).
"""

from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix

_GROUP14_STEM = {'Si': 'sil', 'Ge': 'germ', 'Sn': 'stann', 'Pb': 'plumb'}
_BRIDGE_SUFFIX = {
    'O': 'oxane', 'N': 'azane', 'S': 'thiane', 'Se': 'selenane', 'Te': 'tellurane',
}
_VOWELS = frozenset('aeiou')


def name_catenated_hydride(mol) -> Optional[str]:
    """Return the catenated Group-14/bridge hydride PIN, else ``None``."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    g14_atoms = []
    bridge_atoms = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'H':
            continue
        if sym in _GROUP14_STEM:
            g14_atoms.append(atom)
        elif sym in _BRIDGE_SUFFIX:
            bridge_atoms.append(atom)
        else:
            return None  # any carbon / other element -> not this class

    # Need >= 2 Group-14 atoms and exactly one fewer bridge (a-b-a-b-...-a).
    if len(g14_atoms) < 2 or len(bridge_atoms) != len(g14_atoms) - 1:
        return None

    # Homogeneous element kinds.
    g14_sym = g14_atoms[0].GetSymbol()
    if any(a.GetSymbol() != g14_sym for a in g14_atoms):
        return None
    bridge_sym = bridge_atoms[0].GetSymbol()
    if any(a.GetSymbol() != bridge_sym for a in bridge_atoms):
        return None

    # Strict alternation + unbranched: every Group-14 atom bonds ONLY to bridge
    # atoms (1 if terminal, 2 if internal); every bridge atom bonds ONLY to two
    # Group-14 atoms. No Group-14--Group-14 or bridge--bridge bonds.
    for a in g14_atoms:
        heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy or len(heavy) > 2:
            return None
        if any(n.GetSymbol() != bridge_sym for n in heavy):
            return None
    for b in bridge_atoms:
        heavy = [n for n in b.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(heavy) != 2 or any(n.GetSymbol() != g14_sym for n in heavy):
            return None

    stem = _GROUP14_STEM[g14_sym]
    suffix = _BRIDGE_SUFFIX[bridge_sym]
    # Linking 'a' when the bridge suffix begins with a consonant (silathiane,
    # not silthiane; siloxane / silazane keep no linker — vowel-initial).
    link = '' if suffix[0] in _VOWELS else 'a'
    base = f"{stem}{link}{suffix}"
    multiplier = get_multiplier_prefix(len(g14_atoms), base)
    return f"{multiplier}{base}"


__all__ = ["name_catenated_hydride"]
