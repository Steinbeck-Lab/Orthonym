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


# ---------------------------------------------------------------------------
# P-68.3.2.2 — homonuclear Group-15 (pnictogen) catenated parent hydrides
# ---------------------------------------------------------------------------
# The pnictogen (P/As/Sb/Bi) analogue of the polyazane family (N -> polyazane):
# an unbranched chain of >=2 IDENTICAL Group-15 atoms, fully H-saturated,
# neutral, acyclic, standard valence -> <multiplier><stem> (P-21.2.2, no elision
# of the multiplier vowel):
#     PP -> diphosphane (BB 39081; not 'diphosphine')
#     AsAsAsAsAs -> pentaarsane (BB 39083)   H2Bi-BiH2 -> dibismuthane (BB 39278)
# Mirrors polyazane's saturated homonuclear-chain logic. Nitrogen deliberately
# routes to polyazane (higher functionality of amines, P-21.2.3.1); this family
# is P/As/Sb/Bi only.
_PNICTOGEN_STEM = {'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane',
                   'Bi': 'bismuthane'}
# Basic multiplying prefixes (Table 1.4). NO elision of the terminal vowel
# (P-21.2.2): penta+arsane -> "pentaarsane".
_PNICTOGEN_MULT = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                   7: 'hepta', 8: 'octa'}


def name_homonuclear_pnictogen_chain(mol) -> Optional[str]:
    """Return the PIN for a homonuclear Group-15 catenated parent hydride
    (P-68.3.2.2), else ``None`` (fail-closed cascade-continuation).

    Scope (a pure graph classifier, NOT SMARTS): a single unbranched chain of
    >=2 IDENTICAL pnictogen atoms {P, As, Sb, Bi}, every one H-saturated, neutral,
    non-radical, standard bonding number (3), acyclic, single fragment. ANY carbon
    (organyl phosphane -> name_phosphine), a mononuclear hub (PH3 -> mononuclear
    hydride), a heteronuclear chain (Si-As -> dinuclear hydride), nitrogen
    (-> polyazane), a branch, a ring, a multiple bond, a charge/radical, or a
    nonstandard valence fails a guard and cascades onward. Pure: no mol mutation.
    """
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None

    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    if len(heavy) < 2:
        return None                      # mononuclear PH3 -> mononuclear hydride
    syms = {a.GetSymbol() for a in heavy}
    if len(syms) != 1:
        return None                      # heteronuclear / has carbon -> not here
    element = next(iter(syms))
    stem = _PNICTOGEN_STEM.get(element)
    if stem is None:
        return None                      # not a Group-15 P/As/Sb/Bi element

    from .lambda_convention import nonstandard_bonding_number
    idxs = {a.GetIdx() for a in heavy}
    deg = {}
    for a in heavy:
        if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
            return None
        if nonstandard_bonding_number(mol, a.GetIdx()) is not None:
            return None                  # lambda5 phosphane chain not built here
        d = sum(1 for nb in a.GetNeighbors() if nb.GetIdx() in idxs)
        if d > 2:
            return None                  # branch
        deg[a.GetIdx()] = d

    # Every chain bond must be single (a P=P diphosphene is not built here).
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in idxs and b.GetEndAtomIdx() in idxs:
            if b.GetBondType() != Chem.BondType.SINGLE:
                return None

    ends = [i for i, d in deg.items() if d == 1]
    if len(ends) != 2:
        return None                      # ring (no endpoint) or malformed

    n = len(heavy)
    mult = _PNICTOGEN_MULT.get(n)
    if mult is None:
        return None                      # chain too long for the multiplier table
    return f"{mult}{stem}"


__all__ = ["name_catenated_hydride", "name_homonuclear_pnictogen_chain"]
