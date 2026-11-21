"""The prefix of a sulfur acid group cited as a substituent: 'sulfo', 'sulfino', 'sulfamoyl', 'chlorosulfonyl'.

Appendix 2 (preferred prefixes marked *): 'sulfo* (when unmodified)' (the Blue Book), 'sulfino* (when
unmodified)' (:56832), 'aminosulfonyl = sulfamoyl* = sulfuramidoyl' (:55489,, 'phenylsulfamoyl*'
(:55494), 'aminosulfinyl* (not sulfinamoyl)' (:55485); (:36472,:36482): '-SO2-Cl chlorosulfonyl (preselected
prefix)', '-S(=O)-Cl chlorosulfinyl (preselected prefix)'. The shared substituent namers had no case for a fragment
rooted at such a sulfur and returned their 'substituent' sentinel, so a producer that names its prefixes through
them dropped the group ('propanoyl chloride' for 3-sulfamoylpropanoyl chloride, rejected as another molecule).
"""
from typing import Iterable, Optional

#: the halogen prefix of a sulfonyl or sulfinyl halide group
_HALO = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


def sulfur_acid_group_prefix(mol, frag_atoms: Iterable[int], attach_idx: int, _ring_core_senior: bool = False) -> Optional[str]:
    """The prefix of ``frag_atoms`` when it is one sulfur ``attach_idx`` (two or one doubly bonded O) with one OH,
    NH2, N-substituted amino (sulfonyl only) or halogen group, joined to one atom outside the fragment; None for any
    other shape (a charge, a nuclide label, a ring sulfur or nitrogen, a branch the namer cannot name)."""
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    sulfur = mol.GetAtomWithIdx(attach_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetTotalNumHs() or sulfur.IsInRing():
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge() for i in frag):
        return None
    if sum(1 for nb in sulfur.GetNeighbors() if nb.GetIdx() not in frag) != 1:
        return None
    oxo, rest = 0, []
    for bond in sulfur.GetBonds():
        other = bond.GetOtherAtom(sulfur)
        if other.GetIdx() not in frag:
            continue
        if bond.GetBondTypeAsDouble() == 2 and other.GetSymbol() == "O" and other.GetDegree() == 1:
            oxo += 1
        elif bond.GetBondTypeAsDouble() == 1:
            rest.append(other)
        else:
            return None
    if oxo not in (1, 2) or len(rest) != 1:
        return None
    from .acyl_halide_prefix import holds_principal_group
    if holds_principal_group(mol, attach_idx) and not _ring_core_senior:
        return None                                  # the principal group is the suffix, never this prefix
    group, sym = rest[0], rest[0].GetSymbol()
    sulfonyl = oxo == 2
    if sym in _HALO and group.GetDegree() == 1 and len(frag) == oxo + 2:
        return f"{_HALO[sym]}{'sulfonyl' if sulfonyl else 'sulfinyl'}"
    if sym == "O" and group.GetDegree() == 1 and len(frag) == oxo + 2:
        return "sulfo" if sulfonyl else "sulfino"
    if sym == "N" and group.IsInRing() and sulfonyl and not group.GetIsAromatic():
        outside = [nb for nb in sulfur.GetNeighbors() if nb.GetIdx() not in frag]
        # on a carbon only: -NH-SO2-ring is the amido prefix '(pyrrolidine-1-sulfonamido)'
        return _ring_n_sulfonyl(mol, frag, attach_idx) if outside[0].GetSymbol() == "C" else None
    if sym != "N" or group.IsInRing() or any(b.GetBondTypeAsDouble() != 1 for b in group.GetBonds()):
        return None
    if group.GetDegree() == 1 and len(frag) == oxo + 2:
        return "sulfamoyl" if sulfonyl else "aminosulfinyl"
    if not sulfonyl:
        return None
    return _n_substituted_sulfamoyl(mol, frag, attach_idx, group.GetIdx(), oxo)


def _n_substituted_sulfamoyl(mol, frag, s_idx, n_idx, oxo) -> Optional[str]:
    """'(methylsulfamoyl)' for an -SO2-NHR or -SO2-NR2 fragment, each R named by the shared namer; None otherwise."""
    from .substituent_enumerator import name_substituent
    from ..rules.benzene import _build_n_substituted_sulfamoyl_prefix
    names, covered = [], {s_idx, n_idx}
    for branch in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        if branch.GetIdx() == s_idx:
            continue
        comp, stack = set(), [branch.GetIdx()]
        while stack:
            x = stack.pop()
            if x in comp or x in (s_idx, n_idx):
                continue
            if x not in frag:
                return None
            comp.add(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
        name = name_substituent(mol, sorted(comp), branch.GetIdx())
        if not name or name == "substituent":
            return None
        names.append(name)
        covered |= comp
    if len(covered) + oxo != len(frag):
        return None                                  # a ring through the nitrogen, or atoms the branches do not reach
    return _build_n_substituted_sulfamoyl_prefix(names)


def _ring_n_sulfonyl(mol, frag, s_idx) -> Optional[str]:
    """'pyrrolidine-1-sulfonyl' for an -SO2- joined to a ring nitrogen: the acyl group of the acid named with the suffix
    'sulfonic acid' on that nitrogen ('pyrrolidine-1-sulfonic acid'), acyl groups derived from sulfonic acids
    (the Blue Book, "ic acid" -> "yl", 'benzenesulfonyl (preferred prefix)'), like 'piperidine-1-carbonyl' (the Blue Book) for
    the carboxylic acid. The acid is named by the engine; None when its name has another acid word or no such suffix."""
    import re
    from rdkit import Chem
    if any(mol.GetAtomWithIdx(i).GetSymbol() in ("S", "Se", "Te", "P") and mol.GetAtomWithIdx(i).GetDegree() > 2
           for i in frag if i != s_idx):
        return None                                  # a lambda-convention ring atom ('1lambda6-thiomorpholine'): not here
    rw = Chem.RWMol(mol)
    hydroxy = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(s_idx, hydroxy, Chem.BondType.SINGLE)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in frag), reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
        acid = Chem.MolToSmiles(rw)
    except Exception:                                # noqa: BLE001 -- a fragment RDKit refuses has no acid name
        return None
    from ..rules.lipids import _name_fragment
    name = _name_fragment(acid)
    if not name or not re.search(r"-\d+(?:\(\d+H\))?-sulfonic acid$", name) or name.count("sulfo") != 1:
        return None
    from ..metrics.provenance import _NON_PIN_FRAGMENTS
    if any(f in name for f in _NON_PIN_FRAGMENTS.get()):
        return None                                  # an acid name below the PIN ('2H-benzo[b]pyridine-1-sulfonic acid')
    # (c) (the Blue Book): a substituted group takes 'bis' ('bis(4-methylpiperazine-1-sulfonyl)'); the
    # callers enclose the compound prefix,:31262)
    from .prefix_derivation import built
    return built(name[:-len("sulfonic acid")] + "sulfonyl", substituted=_ring_has_substituent(mol, frag, s_idx))


def _ring_has_substituent(mol, frag, s_idx) -> bool:
    """True when an atom of ``frag`` other than the sulfonyl group lies outside the ring system of the nitrogen."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    nitrogen = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(s_idx).GetNeighbors() if nb.GetIdx() in frag
                    and nb.GetSymbol() == "N")
    system, grown = {nitrogen}, True
    while grown:
        grown = False
        for ring in rings:
            if ring & system and not ring <= system:
                system |= ring
                grown = True
    own = {s_idx} | {nb.GetIdx() for nb in mol.GetAtomWithIdx(s_idx).GetNeighbors() if nb.GetSymbol() == "O"}
    return any(i not in system and i not in own for i in frag)


def ring_n_sulfonyl_group(mol, s_idx: int, core_atoms, record_type: Optional[str] = None) -> Optional[dict]:
    """The prefix record ('name', 'atoms', and 'type' when ``record_type`` is given) of an -SO2- group on the core
    ``core_atoms`` whose other end is a ring nitrogen ('(pyrrolidine-1-sulfonyl)'); None for any other sulfur group,
    which the ring producers name as before."""
    sulfur = mol.GetAtomWithIdx(s_idx)
    if sulfur.GetSymbol() != "S" or not any(nb.GetSymbol() == "N" and nb.IsInRing() and nb.GetIdx() not in core_atoms
                                            for nb in sulfur.GetNeighbors()):
        return None
    frag, stack = set(), [s_idx]
    while stack:
        x = stack.pop()
        if x in frag or x in core_atoms:
            continue
        frag.add(x)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
    prefix = sulfur_acid_group_prefix(mol, frag, s_idx, _core_outranks_ring_n(mol, s_idx, frag, set(core_atoms)))
    if not prefix or not prefix.endswith("-sulfonyl"):
        return None
    record = {"name": prefix, "atoms": sorted(frag), "is_complex": True}   # a compound prefix: '(...)', 'bis(...)'
    return {**record, "type": record_type} if record_type else record


def _core_outranks_ring_n(mol, s_idx: int, frag, core) -> bool:
    """True when the -SO2-N(ring) group is the one sulfonamide of the molecule and the ring system of ``core`` is senior
     (a)-(g), the Blue Book) to that of the nitrogen. A sulfonamide nitrogen in a ring has no suffix
    form, so the senior ring system is the parent and the group is its prefix ('5-(1,4-diazepane-1-sulfonyl)isoquinoline',
    as '1-(4-methylbenzene-1-sulfonyl)pyrrolidine' has the heterocycle for parent)."""
    from rdkit import Chem
    from ..rules.lactones import _ring_system_seniority_key
    if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]"))) != 1:
        return False
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]

    def system_of(seed):
        system, grown = set(seed), True
        while grown:
            grown = False
            for ring in rings:
                if ring & system and not ring <= system:
                    system |= ring
                    grown = True
        return system
    nitrogen = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(s_idx).GetNeighbors() if nb.GetIdx() in frag
                    and nb.GetSymbol() == "N")
    return (_ring_system_seniority_key(mol, system_of(core), rings)
            > _ring_system_seniority_key(mol, system_of({nitrogen}), rings))


def ring_n_sulfonamide(mol, group_class: str, atoms) -> bool:
    """True for a sulfonamide match whose nitrogen is a ring atom: it has no suffix form (as the 'hidden amide' of
    , the Blue Book, is a pseudoketone), so it is cited as a sulfonyl prefix and is no senior group the
    name leaves out ('5-(1,4-diazepane-1-sulfonyl)isoquinoline')."""
    if "sulfonamide" not in group_class:
        return False
    for a in atoms:                                  # C-SO2-N(ring) only: H2N-SO2-N(ring) is 'pyrrolidine-1-sulfonamide'
        sulfur = mol.GetAtomWithIdx(a)
        if sulfur.GetSymbol() == "S":
            near = [nb.GetSymbol() + ("R" if nb.IsInRing() else "") for nb in sulfur.GetNeighbors() if nb.GetSymbol() != "O"]
            return sorted(near) in (["C", "NR"], ["CR", "NR"])
    return False
