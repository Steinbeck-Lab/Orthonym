"""Benzo rings ortho-fused to a peri-fused retained parent: 'hexabenzo[a,d,g,l,o,r]ovalene'.

 (the Blue Book): the peripheral sides of the base component are lettered "a, b, c, etc.... the
side 1,2 being 'a'", and the letters are "as low as possible"; the benzo prefixes are multiplied ('dibenzo[a,j]-
anthracene'). The base component is the component with the most rings: ovalene (10 rings) and
coronene (7 rings) are named here, with the numbering ``data.polycyclic_data`` holds for them (OPSIN's own); a
system that also holds another component with as many rings (b), the Blue Book) keeps the name
below the PIN (its base is not decided here). Every candidate is a name for the caller to read back; the bare system only (no substituent).
"""
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

_PARENTS = ("ovalene", "coronene")
_MULTIPLIERS = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa"}


def _locant_key(loc) -> Tuple[int, str]:
    text = str(loc)
    digits = "".join(c for c in text if c.isdigit())
    return int(digits), text[len(digits):]


def _benzo_rings(mol) -> Optional[List[Tuple[int, int, List[int]]]]:
    """``(fusion atom, fusion atom, the four other atoms)`` of every benzo ring ortho-fused by one bond, or None."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings or any(len(r) != 6 for r in rings):
        return None
    member: Dict[int, int] = {}
    for r in rings:
        for a in r:
            member[a] = member.get(a, 0) + 1
    out = []
    for r in rings:
        shared = [a for a in r if member[a] > 1]
        outer = [a for a in r if member[a] == 1]
        if (len(shared) == 2 and len(outer) == 4 and mol.GetBondBetweenAtoms(*shared) is not None
                and all(mol.GetAtomWithIdx(a).GetDegree() == 2 for a in outer)):
            out.append((shared[0], shared[1], outer))
    return out


def benzo_peri_fused_name(mol) -> Optional[str]:
    """The fusion name of a bare PAH that is benzo rings on ovalene or coronene, or None."""
    from ..data.polycyclic_data import POLYCYCLIC_DATA
    if (mol is None or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetSymbol() != "C" or a.GetFormalCharge() or a.GetIsotope() or not a.GetIsAromatic()
                   for a in mol.GetAtoms())):
        return None
    benzo = _benzo_rings(mol)
    if not benzo:
        return None
    dropped = sorted({a for _x, _y, outer in benzo for a in outer}, reverse=True)
    rw = Chem.RWMol(mol)
    for a in dropped:
        rw.RemoveAtom(a)
    new_index, shift = {}, 0
    for old in range(mol.GetNumAtoms()):
        if old in set(dropped):
            shift += 1
        else:
            new_index[old] = old - shift
    try:
        parent = rw.GetMol()
        for atom in parent.GetAtoms():
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(parent)
    except Exception:  # noqa: BLE001 -- not this shape
        return None
    canonical = Chem.MolToSmiles(parent)
    stem = next((p for p in _PARENTS if POLYCYCLIC_DATA[p]["canonical_smiles"] == canonical), None)
    if stem is None:
        return None
    data = POLYCYCLIC_DATA[stem]
    senior = _other_component_with_as_many_rings(mol, stem, data["num_rings"])
    template = Chem.MolFromSmiles(data["canonical_smiles"])
    numbering = data["iupac_numbering"]
    member: Dict[int, int] = {}
    for r in template.GetRingInfo().AtomRings():
        for a in r:
            member[a] = member.get(a, 0) + 1
    periphery = sorted((i for i in range(template.GetNumAtoms()) if member[i] <= 2), key=lambda i: _locant_key(numbering[i]))
    best = None
    for match in parent.GetSubstructMatches(template, uniquify=False, maxMatches=1000):
        position = {match[t]: k for k, t in enumerate(periphery)}
        letters = []
        for x, y, _outer in benzo:
            px, py = position.get(new_index[x]), position.get(new_index[y])
            if px is None or py is None:
                break
            k = min(px, py) if abs(px - py) == 1 else (max(px, py) if {px, py} == {0, len(periphery) - 1} else None)
            if k is None:
                break
            letters.append(k)
        else:
            key = sorted(letters)
            if best is None or key < best:
                best = key
    if best is None or len(best) not in _MULTIPLIERS or max(best) >= 26:
        return None
    name = f"{_MULTIPLIERS[len(best)]}benzo[{','.join(chr(97 + k) for k in best)}]{stem}"
    if senior:
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(name, rule="P-25.3.2.4", detail=f"the component {senior} has as many rings or more")
    return name


#: the parent components of Table 2.7 (the Blue Book) with seven rings or more besides ovalene and coronene
_RETAINED_LARGE = ("pyranthrene", "rubicene", "trinaphthylene")
_SERIES = ("cene", "phene", "helicene")


@lru_cache(maxsize=None)
def _component_smiles(name: str) -> Optional[str]:
    from ..validation.opsin_roundtrip import opsin_parse
    return opsin_parse(name) or None


def _other_component_with_as_many_rings(mol, stem: str, rings: int) -> Optional[str]:
    """A parent component other than ``stem`` that ``mol`` contains with at least ``rings`` rings, or None: such a
    component decides the base by (b) "a component containing the greater number of rings" (or by a later
    criterion on a tie), so the name of ``stem`` is not the PIN. The retained names of Table 2.7 and the acenes,
    phenes and helicenes of up to the ring count of ``mol``; a component OPSIN gives no structure for is no
    evidence either way and is skipped."""
    from ..data.chain_names import multiplying_prefix
    names = [n for n in _RETAINED_LARGE + _PARENTS if n != stem]
    for n in range(max(rings, 7), mol.GetRingInfo().NumRings() + 1):
        prefix = multiplying_prefix(n)
        names += [prefix + ("" if suffix == "cene" and prefix.endswith("a") else "") + suffix for suffix in _SERIES]
    for name in names:
        smiles = _component_smiles(name)
        part = Chem.MolFromSmiles(smiles) if smiles else None
        if part is not None and part.GetRingInfo().NumRings() >= rings and mol.HasSubstructMatch(part):
            return name
    return None
