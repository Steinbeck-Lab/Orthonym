"""The fused parent of a bridged fused ring system: its name and every IUPAC numbering.

 (the Blue Book): "Bridges are selected so that a recommended fused ring
system as described in through is the parent fused ring system that is
bridged." (:14245): "The ring system that remains after removal of the
bridge(s) is named according to through." (:14177): "The fused
ring system to be bridged is numbered in the usual way." (:12493): "Anthracene,
phenanthrene, acridine, carbazole, xanthene and its chalcogen analogues, purine, and
cyclopenta[a]phenanthrene are exceptions; traditional numberings are retained."
 (:12501): numbering proceeds "around the system, including fusion
heteroatoms but not fusion carbon atoms. Each fusion carbon atom is given the same number
as the immediately preceding nonfusion skeletal atom, modified by a Roman letter".

``fused_parent(mol, residual)`` names the residual by the engine's own parent tables (a
fused ring system whose name the Blue Book prints as a PIN: ``PIN_PARENT_NAMES``, the
book's other printed names in ``parent_table.BB_PARENTS``, a benzo name, or a
name the rules give step by step for a skeleton the book does not print,
``derived_parents.DERIVED_PARENTS``) or,
when no table holds the skeleton, by a two-component carbocyclic fusion name
(``fusion_names``), and returns every numbering: one fixed numbering read off the tables
(or off OPSIN's reading of the fusion name), checked against every other source that
numbers the same skeleton, composed with each element-aware automorphism of the residual.
Any disagreement declines (None).
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem

from ...perception.automorphisms import skeleton_automorphisms
from ...perception.mancude import is_mancude_ring_system, max_matching_size
from .derived_parents import DERIVED_PARENTS
from .parent_table import BB_PARENTS

Locant = Union[int, str]

#: (the Blue Book): "Anthracene, phenanthrene, acridine, carbazole, xanthene and
#: its chalcogen analogues, purine, and cyclopenta[a]phenanthrene are exceptions; traditional
#: numberings are retained." Of these, the parents whose fusion carbon atoms carry plain
#: numbers (the steroid numbering 5, 8, 9, 10, 13, 14) are listed here; the others keep their
#: lettered fusion locants.
TRADITIONAL_NUMBERING = frozenset({"cyclopenta[a]phenanthrene"})

#: Fused parent names (indicated hydrogen dropped) that the Blue Book prints as PINs, with
#: the line of the Blue Book that shows it. A name outside this table, outside
#: ``parent_table.BB_PARENTS``, outside ``derived_parents.DERIVED_PARENTS`` and outside the
#: benzo-name class (``_BENZO_NAME``) is not used as a bridged fused parent (the
#: fusion names that ``fusion_names`` builds for skeletons no table holds aside):
#: the engine's tables also hold general-nomenclature names ('naphthacene',
#: '1,2-benzisoxazole', 'beta-carboline', 'imidazo[2,1-b]thiazole') that read back to the
#: right structure but are not PINs. Three names have no bare '(PIN)' line: 'heptalene'
#: cites the list of hydrocarbon parent components (:14776; the rule
#::11455 forms the name, and the PIN at:14599 is a dimethyl derivative of it);
#: 'dibenzo[b,d]thiophene' and 'benzo[7]annulene' cite a derived example, the
#: 2H-5-lambda4 PIN at:14555 and '6,7-dihydro-5H-benzo[7]annulene (PIN)' at:17032.
PIN_PARENT_NAMES: Dict[str, int] = {
    "acenaphthylene": 11493, "acridine": 11707, "anthracene": 11386,
    "as-indacene": 11408, "azulene": 11412, "benzo[7]annulene": 17032,
    "benzo[8]annulene": 19564, "benzo[g]quinoline": 13978, "biphenylene": 11465,
    "carbazole": 11543, "chrysene": 11364, "cinnoline": 11551, "coronene": 11346,
    "cyclopenta[8]annulene": 13974, "dibenzo[b,d]pyran": 13463,
    "dibenzo[b,d]thiophene": 14555, "fluorene": 11400, "furo[3,2-b]pyran": 12246,
    "heptalene": 14776, "hexahelicene": 11483, "indazole": 11599, "indene": 11317,
    "indole": 11605, "indolizine": 11709, "isoindole": 11611, "isoquinoline": 11711,
    "naphthalene": 11416, "oxanthrene": 11757, "pentacene": 11442, "perylene": 11354,
    "phenalene": 11396, "phenanthrene": 11390, "phenazine": 11517,
    "phenoselenazine": 11785, "phenotellurazine": 11787, "phenothiazine": 11783,
    "phenoxathiine": 11795, "phenoxazine": 11781, "phthalazine": 11567, "picene": 11356,
    "purine": 11595, "pyrene": 11368, "pyrrolizine": 11628, "quinazoline": 11553,
    "quinoline": 11713, "quinolizine": 11584, "quinoxaline": 11557, "s-indacene": 11404,
    "tetracene": 11442, "thianthrene": 11757, "thioxanthene": 11638,
    "triphenylene": 11467, "xanthene": 11634, "1-benzofuran": 11827,
    "2-benzofuran": 11829, "1-benzopyran": 11656, "2-benzopyran": 11682,
    "1-benzothiopyran": 11664, "3-benzoxepine": 11821, "1,3-benzoxathiole": 14575,
}

#: (the Blue Book, heading:11813): a benzene ring ortho-fused to a
#: heteromonocycle of five or more members is named by a 'benzo name' (examples:11821
#: '3-benzoxepine (PIN)',:11827 '1-benzofuran (PIN)',:11823 '4H-3,1-benzoxazine (PIN)'):
#: heteroatom locants ("for preferred IUPAC names locants must be cited",:11815), 'benzo'
#: (its final 'o' elided before a vowel), and the PIN name of the heteromonocycle. The stems are Hantzsch-Widman / retained PIN names of
#: (never 'isoxazole' / 'isothiazole', whose PINs are 1,2-oxazole and 1,2-thiazole).
_BENZO_STEMS = ("furan", "thiophene", "selenophene", "tellurophene", "pyran", "thiopyran",
                "selenopyran", "telluropyran", "oxepine", "thiepine", "azepine", "oxazole",
                "thiazole", "selenazole", "oxadiazole", "thiadiazole", "oxathiole", "dioxole",
                "dithiole", "dioxine", "oxazine", "thiazine", "diazepine", "oxazepine",
                "thiazepine", "triazole", "imidazole", "oxathiine", "dioxepine")
_BENZO_NAME = re.compile(
    r"^\d+(?:,\d+)*-(?:"
    + "|".join(("benz" + s) if s[0] in "aeiou" else ("benzo" + s) for s in _BENZO_STEMS)
    + r")$")

_IH = re.compile(r"^((?:\d+[a-z]*H,)*\d+[a-z]*H)-")


@dataclass(frozen=True)
class FusedParent:
    """A nameable fused parent: ``name`` without indicated hydrogen ('indene',
    '1-benzopyran'), ``numberings`` = every IUPAC numbering of the residual atoms (int
    for nonfusion atoms and every heteroatom, '4a'-type str for fusion carbon atoms),
    ``source`` = the table that named it."""
    name: str
    numberings: Tuple[Dict[int, Locant], ...]
    source: str


def is_pin_parent_name(name: str) -> bool:
    return (name in PIN_PARENT_NAMES or name in BB_PARENTS or name in DERIVED_PARENTS
            or bool(_BENZO_NAME.match(name)))


def normalize_locant(loc) -> Locant:
    """int stays int; (4, 'a') and '4a' become '4a'; '5' becomes 5."""
    if isinstance(loc, tuple):
        return f"{loc[0]}{loc[1]}"
    if isinstance(loc, int):
        return loc
    text = str(loc)
    return int(text) if text.isdigit() else text


def _skeleton(mol, atoms) -> Tuple[Chem.Mol, List[int]]:
    """The element-aware all-single-bond graph induced by ``atoms`` and its atom order."""
    idx = sorted(atoms)
    pos = {a: i for i, a in enumerate(idx)}
    rw = Chem.RWMol()
    for a in idx:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in pos and j in pos:
            rw.AddBond(pos[i], pos[j], Chem.BondType.SINGLE)
    sk = rw.GetMol()
    sk.UpdatePropertyCache(strict=False)
    return sk, idx


def _key_of(mol) -> str:
    sk, _ = _skeleton(mol, range(mol.GetNumAtoms()))
    return Chem.MolToSmiles(sk)


def _iso(source, key_mol) -> Optional[Tuple[int, ...]]:
    """One isomorphism source atom -> key_mol atom (bond orders ignored)."""
    if source.GetNumAtoms() != key_mol.GetNumAtoms() or source.GetNumBonds() != key_mol.GetNumBonds():
        return None
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    query = Chem.AdjustQueryProperties(source, params)
    match = key_mol.GetSubstructMatch(query)
    return tuple(match) if match else None


def _mancude(key_mol) -> Optional[Chem.Mol]:
    """A standalone mancude molecule on the skeleton: a maximum matching of the atoms that
    can carry a ring double bond, as double bonds (the unmatched atoms carry hydrogen)."""
    pt = Chem.GetPeriodicTable()
    n = key_mol.GetNumAtoms()
    elig = {i for i in range(n)
            if pt.GetDefaultValence(key_mol.GetAtomWithIdx(i).GetAtomicNum())
            - key_mol.GetAtomWithIdx(i).GetDegree() >= 1}
    adj = {i: {nb.GetIdx() for nb in key_mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in elig}
           for i in elig}
    target = max_matching_size(elig, adj)
    rw = Chem.RWMol(key_mol)
    used: Set[int] = set()
    chosen: List[Tuple[int, int]] = []

    def extend(order: List[int]) -> bool:
        if len(chosen) == target:
            return True
        free = [a for a in order if a not in used and a in elig]
        if not free:
            return False
        a = free[0]
        for b in sorted(adj[a]):
            if b in used:
                continue
            rest = {x for x in elig if x not in used and x not in (a, b)}
            sub = {x: adj[x] & rest for x in rest}
            if len(chosen) + 1 + max_matching_size(rest, sub) == target:
                used.update((a, b))
                chosen.append((a, b))
                if extend(order):
                    return True
                chosen.pop()
                used.difference_update((a, b))
        rest = {x for x in elig if x not in used and x != a}
        sub = {x: adj[x] & rest for x in rest}
        if len(chosen) + max_matching_size(rest, sub) == target:
            used.add(a)
            if extend(order):
                return True
            used.discard(a)
        return False

    if not extend(list(range(n))):
        return None
    for a, b in chosen:
        rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


@lru_cache(maxsize=1)
def _table_index() -> Dict[str, List[Tuple[str, Chem.Mol, Dict[int, Locant], str]]]:
    """skeleton key -> [(name, source molecule, fixed numbering on its atoms, source)] for
    every POLYCYCLIC_DATA entry with a full numbering and every mancude all-ring
    catalogue entry with a full numbering, whose name is a PIN parent name."""
    from ...data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    from ...data.polycyclic_data import POLYCYCLIC_DATA
    from ..fusion_numbering import compute_fused_numbering
    index: Dict[str, List[Tuple[str, Chem.Mol, Dict[int, Locant], str]]] = {}
    for name, entry in POLYCYCLIC_DATA.items():
        num = entry.get("iupac_numbering") or {}
        pm = Chem.MolFromSmiles(entry.get("canonical_smiles") or "")
        if pm is None or not is_pin_parent_name(name):
            continue
        source = "polycyclic_data"
        if not num:     # an empty table map: the fusion numbering of the same molecule
            num = compute_fused_numbering(pm, set(range(pm.GetNumAtoms()))) or {}
            source = "polycyclic_data+fusion_numbering"
        if len(num) != pm.GetNumAtoms():
            continue
        fixed = {int(a): normalize_locant(l) for a, l in num.items()}
        index.setdefault(_key_of(pm), []).append((name, pm, fixed, source))
    for smi, entry in FUSED_HETEROCYCLE_DATA.items():
        name = _IH.sub("", entry.get("name") or "")
        pm = Chem.MolFromSmiles(smi)
        locs = entry.get("iupac_locants") or {}
        if pm is None or not is_pin_parent_name(name):
            continue
        if any(not a.IsInRing() for a in pm.GetAtoms()):
            continue
        if sorted(k for k in locs if isinstance(k, int)) != list(range(pm.GetNumAtoms())):
            continue
        if is_mancude_ring_system(pm, set(range(pm.GetNumAtoms()))) is not True:
            continue
        fixed = {a: normalize_locant(locs[a]) for a in range(pm.GetNumAtoms())}
        index.setdefault(_key_of(pm), []).append((name, pm, fixed, "catalogue"))
    for name, (_line, _opsin_name, smiles, locants) in BB_PARENTS.items():
        pm = Chem.MolFromSmiles(smiles)
        if pm is None or len(locants) != pm.GetNumAtoms():
            continue
        fixed = {a: normalize_locant(loc) for a, loc in enumerate(locants)}
        index.setdefault(_key_of(pm), []).append((name, pm, fixed, "bb_table"))
    # user ruling D6 (2026-10-02): the rule-derived parents, each with its written
    # derivation (``derived_parents``); indexed as the printed ones, so the fusion
    # numbering still cross-checks any of them it can number (S2 decision 7)
    for name, entry in DERIVED_PARENTS.items():
        pm = Chem.MolFromSmiles(entry.smiles)
        if pm is None or len(entry.locants) != pm.GetNumAtoms():
            continue
        fixed = {a: normalize_locant(loc) for a, loc in enumerate(entry.locants)}
        index.setdefault(_key_of(pm), []).append((name, pm, fixed, "derived_table"))
    return index


def _locant_types_ok(key_mol, numbering: Dict[int, Locant], traditional: bool = False) -> bool:
    """ (:12501): every nonfusion atom and every heteroatom has a number,
    every fusion carbon atom a lettered locant; the numbers are distinct. ``traditional``:
    a parent of ``TRADITIONAL_NUMBERING`` numbers its fusion carbon atoms like the others
    ,:12493), so every locant is a number."""
    seen = set()
    if traditional:
        locs = [numbering.get(a.GetIdx()) for a in key_mol.GetAtoms()]
        return all(isinstance(x, int) for x in locs) and len(set(locs)) == len(locs)
    for a in key_mol.GetAtoms():
        loc = numbering.get(a.GetIdx())
        if loc is None or loc in seen:
            return False
        seen.add(loc)
        fusion = a.GetDegree() >= 3
        if a.GetAtomicNum() != 6 or not fusion:
            if not isinstance(loc, int):
                return False
        elif isinstance(loc, int):
            return False
    return True


def _same_orbit(key_mol, first: Dict[int, Locant], other: Dict[int, Locant]) -> bool:
    auts, exhaustive = skeleton_automorphisms(key_mol, set(range(key_mol.GetNumAtoms())),
                                              element_blind=False, cap=64)
    if not exhaustive:
        return False
    return any(all(first[sigma[a]] == other[a] for a in other) for sigma in auts)


@lru_cache(maxsize=1024)
def _parent_for_key(key: str) -> Optional[Tuple[str, Tuple[Tuple[int, Locant], ...], str]]:
    """(name, fixed numbering on the atoms of ``Chem.MolFromSmiles(key)``, source)."""
    from ...data import get_retained_name
    from ..fusion_numbering import compute_fused_numbering
    key_mol = Chem.MolFromSmiles(key)
    if key_mol is None:
        return None
    candidates: List[Tuple[str, Dict[int, Locant], str]] = []
    checks: List[Dict[int, Locant]] = []
    sources: List[Chem.Mol] = []
    for name, pm, fixed, source in _table_index().get(key, ()):
        iso = _iso(pm, key_mol)
        if iso is None:
            continue
        candidates.append((name, {iso[a]: l for a, l in fixed.items()}, source))
        sources.append((pm, iso))
    std = _mancude(key_mol)
    if std is not None and not candidates:
        retained = get_retained_name(Chem.MolToSmiles(std))
        if retained and is_pin_parent_name(_IH.sub("", retained)):
            cfn = compute_fused_numbering(std, set(range(std.GetNumAtoms())))
            if cfn and len(cfn) == std.GetNumAtoms():
                candidates.append((_IH.sub("", retained),
                                   {a: normalize_locant(l) for a, l in cfn.items()},
                                   "retained+fusion_numbering"))
    if not candidates:
        # (:11903): no retained or Blue Book name -- a two-component carbocyclic
        # fusion name (``fusion_names``), numbered by OPSIN's reading of it
        from .fusion_names import opsin_structure, two_component_name
        made = two_component_name(key_mol)
        struct = opsin_structure(made, key_mol.GetNumAtoms()) if made else None
        source = "fusion_name+opsin"
        if made is None:
            # slice S2c-1: a two-component name with a heterocyclic component, or a
            # benzo name (``hetero_fusion``), numbered the same way
            from . import hetero_fusion
            made = hetero_fusion.hetero_component_name(key_mol)
            struct = hetero_fusion.opsin_structure(made, key_mol.GetNumAtoms()) if made else None
            source = "hetero_fusion_name+opsin"
        pm = Chem.MolFromSmiles(struct[0]) if struct else None
        iso = _iso(pm, key_mol) if pm is not None else None
        if iso is not None:
            candidates.append((made, {iso[a]: normalize_locant(loc) for a, loc in enumerate(struct[1])},
                               source))
            sources.append((pm, iso))
    if not candidates or len({c[0] for c in candidates}) != 1:
        return None
    # every other numbering source for the same skeleton must agree (up to symmetry)
    for pm, iso in sources:
        cfn = compute_fused_numbering(pm, set(range(pm.GetNumAtoms())))
        if cfn and len(cfn) == pm.GetNumAtoms():
            checks.append({iso[a]: normalize_locant(l) for a, l in cfn.items()})
    # A map that breaks (a lettered fusion heteroatom, as the catalogue's
    # quinolizine N '4a') is discarded; every remaining map must agree up to symmetry.
    name = candidates[0][0]
    maps = [(c[1], c[2]) for c in candidates] + [(m, "fusion_numbering") for m in checks]
    maps = [(m, src) for m, src in maps
            if _locant_types_ok(key_mol, m, name in TRADITIONAL_NUMBERING)]
    if not maps:
        return None
    fixed, source = maps[0]
    for other, _ in maps[1:]:
        if not _same_orbit(key_mol, fixed, other):
            return None
    return name, tuple(sorted(fixed.items())), source


def fused_parent(mol, residual) -> Optional[FusedParent]:
    """The name and every numbering of the fused parent on ``residual``, or None."""
    sk, idx = _skeleton(mol, residual)
    key = Chem.MolToSmiles(sk)
    order = list(sk.GetPropsAsDict(True, True).get("_smilesAtomOutputOrder", ()))
    from ...validation.opsin_roundtrip import OpsinUnavailable
    try:
        got = _parent_for_key(key)
    except OpsinUnavailable:
        # OPSIN could not number a produced name just now (``fusion_names.opsin_structure``):
        # no parent this time, and neither cache keeps the failure, so a later call asks again
        return None
    if got is None or len(order) != len(idx):
        return None
    name, fixed_items, source = got
    base = {idx[order[k]]: loc for k, loc in fixed_items}
    residual = set(residual)
    auts, exhaustive = skeleton_automorphisms(mol, residual, element_blind=False, cap=64)
    if not exhaustive or set(base) != residual:
        return None
    out, seen = [], set()
    for sigma in auts:
        numb = {a: base[sigma[a]] for a in residual}
        sig = tuple(sorted(numb.items()))
        if sig not in seen:
            seen.add(sig)
            out.append(numb)
    return FusedParent(name, tuple(out), source)
