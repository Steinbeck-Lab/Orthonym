"""Amplificants of a phane: the ring or ring system that one superatom stands for.

the Blue Book, each line read with ``sed -n``:

* (:14892): the amplification prefix is the name of the allowed ring or ring system with
  the final letter 'e' changed to 'a', or an 'a' added when there is no final 'e'.
* "Allowed parent hydrides" (:14902): "An amplification prefix can be derived from
  monocyclic rings and polycyclic ring systems having the maximum number of noncumulative double
  bonds (mancude), bridged fused ring systems, saturated monocyclic rings, saturated bicycloalkanes
  and polycycloalkanes (von Baeyer hydrocarbons), and spiro alkanes."
* (a) "Disallowed names of parent hydrides" (:14910): spirobi names, "spiro ring systems
  with at least one fused ring system or polycycloalkane ring system", ring assembly names.
* (:14967): "Numbering of an amplificant is determined primarily by the numbering rules
  that apply to the parent name from which the amplification prefix is derived. When there is a
  choice, the general rule of lowest locants is used."
* (:14794): a heterocyclic amplificant with locants takes brackets (``[1,2]oxazola``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from rdkit import Chem

from ...validation.spelling.cyclophane import is_mancude_system
from ...validation.spelling.linear_phane import _CHAIN_ELEMENTS, ring_system_hydride
from ..skeletal_replacement import _A_CITATION_ORDER

#: (:15246) "Lowest locants are assigned to heteroatoms considered in the order of their
#: seniority: O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl"
HETERO_ORDER: Tuple[str, ...] = tuple(e for e in _A_CITATION_ORDER if e in _CHAIN_ELEMENTS and e != "C")

# a name with indicated hydrogen, a hydro prefix, the lambda convention, a spiro or a primed locant is no
# allowed parent of the cases built here (PN4c, PN5)
_NO_AMPLIFICANT_NAME = re.compile(r"\d[A-Za-z]*H-|hydro|λ|lambda|spiro|['′]")
_LEADING_LOCANTS = re.compile(r"^(\d+(?:,\d+)*)-")
_Locant = Tuple[int, str]


@dataclass(frozen=True)
class AmplificantHydride:
    name: str                    # engine PIN of the isolated hydride: "pyridine", "1,3-oxazole", "naphthalene"
    prefix: str                  # amplification prefix: "pyridina", "[1,3]oxazola"
    key: str                     # canonical SMILES of the hydride (the identity of the amplificant type)
    score: tuple                 # rules.ring_selection.ring_system_score of the hydride; lower = senior)
    numberings: Tuple[Mapping[int, _Locant], ...]   # every valid numbering: atom index IN THE INPUT MOLECULE -> (number, letter)


def amplification_prefix(name: str) -> str:
    """ and: the amplification prefix of the parent hydride ``name``."""
    from ..phane import _amplification_prefix
    return _amplification_prefix(_LEADING_LOCANTS.sub(r"[\1]", name))


def _plain_allowed_atoms(mol: Chem.Mol, atoms: FrozenSet[int]) -> bool:
    """No charge, radical or isotope on the ring system, and no multiple bond to an atom outside it."""
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope():
            return False
        for bond in atom.GetBonds():
            if bond.GetOtherAtomIdx(idx) not in atoms and bond.GetBondType() != Chem.BondType.SINGLE:
                return False
    return True


def _allowed_parent(hydride: Chem.Mol) -> bool:
    """: a mancude ring or ring system, or a saturated monocycle."""
    everything = frozenset(range(hydride.GetNumAtoms()))
    if is_mancude_system(hydride, everything):
        return True
    if hydride.GetRingInfo().NumRings() != 1:
        return False
    return all(b.GetBondType() == Chem.BondType.SINGLE for b in hydride.GetBonds())


def _needs_indicated_hydrogen(hydride: Chem.Mol) -> bool:
    """ (:3557): a mancude hydride that has a saturated ring atom (a pyrrole-type NH, a CH2 of 2H-pyran)
    needs indicated hydrogen. Read from the structure, because the engine's name of a fused hydride of this
    kind can come without it (``pyrido[3,4-b]pyrrolo[2,3-d]pyridine`` for a hydride that OPSIN reads otherwise)."""
    unsaturated = any(b.GetBondType() != Chem.BondType.SINGLE for b in hydride.GetBonds())
    if not unsaturated:
        return False
    for atom in hydride.GetAtoms():
        multiple = atom.GetIsAromatic() or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())
        if atom.GetIsAromatic():
            # a heteroatom with a hydrogen and no valence left for a pi bond (pyrrole N-H); a silabenzene Si-H has one
            if (atom.GetSymbol() != "C" and atom.GetTotalNumHs()
                    and atom.GetDegree() + atom.GetTotalNumHs()
                    >= Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())):
                return True
        elif not multiple and atom.GetTotalNumHs():
            return True
    return False


def _hetero_key(symbols: List[str]) -> tuple:
    """(locants of all heteroatoms, locants per element in the order of of a numbered ring."""
    hetero = tuple(i + 1 for i, s in enumerate(symbols) if s != "C")
    per_element = tuple(tuple(i + 1 for i, s in enumerate(symbols) if s == e) for e in HETERO_ORDER)
    return hetero, per_element


def _monocycle_numberings(hydride: Chem.Mol) -> List[Dict[int, _Locant]]:
    """Every numbering of a monocycle that gives its heteroatoms the lowest locants, then the
    general rule: heteroatoms as a set, then in the order of; a ring without heteroatom keeps
    all 2n of them."""
    ring = list(hydride.GetRingInfo().AtomRings()[0])
    n = len(ring)
    best, found = None, []
    for start in range(n):
        for step in (1, -1):
            order = [ring[(start + step * k) % n] for k in range(n)]
            key = _hetero_key([hydride.GetAtomWithIdx(a).GetSymbol() for a in order])
            if best is None or key < best:
                best, found = key, []
            if key == best:
                found.append({a: (k + 1, "") for k, a in enumerate(order)})
    return found


def _fused_numbering(hydride: Chem.Mol) -> Optional[Dict[int, _Locant]]:
    from ..fusion_numbering import compute_fused_numbering
    got = compute_fused_numbering(hydride, set(range(hydride.GetNumAtoms())))
    if got is None:
        return None
    return {a: ((loc, "") if isinstance(loc, int) else (loc[0], loc[1])) for a, loc in got.items()}


def _numberings(hydride: Chem.Mol) -> Optional[List[Dict[int, _Locant]]]:
    """Every numbering of the hydride, in hydride atom indices: a monocycle as above, a cata-fused system by
    the fusion numbering composed with every automorphism of the hydride."""
    if hydride.GetRingInfo().NumRings() == 1:
        return _monocycle_numberings(hydride)
    base = _fused_numbering(hydride)
    if base is None:
        return None
    seen = []
    for match in hydride.GetSubstructMatches(hydride, uniquify=False, useChirality=False, maxMatches=100000):
        numbering = {match[i]: base[i] for i in range(hydride.GetNumAtoms())}
        if numbering not in seen:
            seen.append(numbering)
    return seen


def amplificant_hydride(mol: Chem.Mol, atoms: FrozenSet[int]) -> Optional[AmplificantHydride]:
    """The amplificant hydride of the ring system ``atoms`` of ``mol``, or None when it is no parent the
    plain case builds (a charge, a radical, an isotope, a multiple bond to the outside, indicated hydrogen,
    hydro prefixes, a ring system that is neither mancude nor a saturated monocycle, and a system with no
    fusion numbering)."""
    from ...assembly.fragment_naming import name_fragment_recursively
    from ...errors import is_failure_name
    from ..ring_selection import ring_system_score
    if not _plain_allowed_atoms(mol, atoms):
        return None
    got = ring_system_hydride(mol, atoms)
    if got is None:
        return None
    hydride, atom_map = got
    if not _allowed_parent(hydride) or _needs_indicated_hydrogen(hydride):
        return None
    smiles = Chem.MolToSmiles(hydride, isomericSmiles=False)
    name = name_fragment_recursively(smiles, style="pin")
    if not name or is_failure_name(name) or _NO_AMPLIFICANT_NAME.search(name):
        return None
    numberings = _numberings(hydride)
    if not numberings:
        return None
    back = {i: a for a, i in atom_map.items()}
    return AmplificantHydride(
        name=name, prefix=amplification_prefix(name), key=smiles,
        score=ring_system_score(hydride, set(range(hydride.GetNumAtoms()))),
        numberings=tuple({back[i]: loc for i, loc in numbering.items()} for numbering in numberings))
