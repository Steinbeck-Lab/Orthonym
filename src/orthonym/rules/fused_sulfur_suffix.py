"""Sulfur acid groups as the suffix of a fused ring parent: 'quinoline-8-sulfonyl chloride', '1H-indole-5-sulfonamide'.

The fused ring producer recognised four ring suffixes (-carboxylic acid, -carboxamide, -carbonitrile, -carbaldehyde);
a sulfonic or sulfinic acid, a sulfonamide or a sulfonyl halide on a fused ring had no suffix, so these rings had no
name at the PIN tier ('1,3-benzothiazole-2-sulfonamide'). Suffixes: 'Substitutive nomenclature, suffix
mode, for sulfonic, sulfinic, etc., acids' (the Blue Book), sulfonamides (:32744), acyl
halides from suffix acids (:31412). Order, (:18162): 7a acids "carboxylic..., sulfonic, sulfinic",
10 acid halides "in the order of the corresponding acid... then... (-F > -Cl > -Br > -I)", 11 amides "in the
order of the corresponding acids", 14 nitriles, 15 aldehydes. Prefix forms when a senior group takes the suffix:
'sulfo*', 'sulfino*', 'sulfamoyl*' (Appendix 2,:56837,:56832,:55489), 'chlorosulfonyl (preselected prefix)'
,:36472).
"""
from typing import Any, Dict, Optional, Set

_HALIDE = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
_HALO = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}

#: suffix -> prefix form, for the fused assembler's suffix-to-prefix table
SULFUR_SUFFIX_PREFIX: Dict[str, str] = {
    "sulfonic acid": "sulfo", "sulfinic acid": "sulfino", "sulfonamide": "sulfamoyl",
    # a compound prefix takes enclosing marks: '4-(chlorosulfonyl)benzoic acid'
    **{f"sulfonyl {_HALIDE[x]}": f"({_HALO[x]}sulfonyl)" for x in _HALIDE},
}
#: suffix -> the seniority class it expresses (seniority.SENIORITY_ORDER)
SULFUR_SUFFIX_CLASS: Dict[str, str] = {
    "sulfonic acid": "sulfonic_acid", "sulfinic acid": "sulfinic_acid", "sulfonamide": "primary_sulfonamide",
    **{f"sulfonyl {_HALIDE[x]}": "sulfonyl_halide" for x in _HALIDE},
}
#: the detachable fused ring suffixes in decreasing seniority
FUSED_SUFFIX_PRIORITY = ["carboxylic acid", "sulfonic acid", "sulfinic acid",
                         *(f"sulfonyl {_HALIDE[x]}" for x in ("F", "Cl", "Br", "I")),
                         "carboxamide", "sulfonamide", "carbonitrile", "carbaldehyde"]


def fused_sulfur_suffix(mol, start_idx: int, excluded: Set[int]) -> Optional[Dict[str, Any]]:
    """The suffix record of an -SO3H, -SO2H, -SO2NH2 or -SO2X group whose sulfur ``start_idx`` is joined to the core
    (``excluded``) by one bond; None for any other shape, a charge, a nuclide label, or a second sulfonyl halide in
    the molecule (a multiplied halide suffix is left to the other ring writers)."""
    sulfur = mol.GetAtomWithIdx(start_idx)
    if sulfur.GetSymbol() != "S" or sulfur.IsInRing() or sulfur.GetFormalCharge() or sulfur.GetIsotope():
        return None
    if sum(1 for nb in sulfur.GetNeighbors() if nb.GetIdx() in excluded) != 1:
        return None
    ring_info = mol.GetRingInfo()
    if not any(ring_info.NumAtomRings(i) > 1 for i in excluded):
        return None                                  # a fused parent only; benzene and monocycles have their own writers
    oxo, rest = 0, []
    for bond in sulfur.GetBonds():
        other = bond.GetOtherAtom(sulfur)
        if other.GetIdx() in excluded:
            continue
        if other.GetFormalCharge() or other.GetIsotope() or (other.GetDegree() != 1 and other.GetSymbol() != "N"):
            return None
        if bond.GetBondTypeAsDouble() == 2 and other.GetSymbol() == "O":
            oxo += 1
        elif bond.GetBondTypeAsDouble() == 1:
            rest.append(other)
        else:
            return None
    if len(rest) != 1:
        return None
    group, sym = rest[0], rest[0].GetSymbol()
    if oxo == 2 and sym == "O":
        suffix = "sulfonic acid"
    elif oxo == 1 and sym == "O":
        suffix = "sulfinic acid"
    elif oxo == 2 and sym == "N" and group.GetTotalNumHs() == 2:
        suffix = "sulfonamide"
    elif oxo == 2 and sym == "N":
        return _n_substituted_sulfonamide(mol, sulfur, group, excluded)
    elif oxo == 2 and sym in _HALIDE and _sulfonyl_halide_count(mol) == 1:
        suffix = f"sulfonyl {_HALIDE[sym]}"
    else:
        return None
    from ..assembly.acyl_halide_prefix import holds_principal_group
    if not holds_principal_group(mol, start_idx):
        return None                                  # not the principal group: the prefix writers name it, as before
    atoms = [start_idx] + [nb.GetIdx() for nb in sulfur.GetNeighbors() if nb.GetIdx() not in excluded]
    if suffix == "sulfonamide" and amide_rival(mol, set(atoms)):
        return None
    return {"name": SULFUR_SUFFIX_PREFIX[suffix], "atoms": atoms, "functional_group": SULFUR_SUFFIX_CLASS[suffix],
            "type": "suffix", "suffix_name": suffix}


def amide_rival(mol, own: Set[int]) -> bool:
    """True when an acyclic -C(=O)-N or -C(=S)-N group (a carboxamide, a hydroxamic acid, an N-acyl group, a urea)
    lies outside the atoms ``own`` of the group claimed as suffix. Amides rank "in the order of the corresponding
    acids", the Blue Book; carboxylic before sulfonic,:18172), and N-hydroxy amides are amides
    (:17568), but the perception ranks a sulfonamide above a hydroxamic acid; a ring C(=O)-N is a lactam, a
    pseudoketone (class 16), and does not count."""
    from rdkit import Chem
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3;!R](=[O,S])[NX3]")):
        if not set(match) <= own:
            return True
    return False


def _sulfonyl_halide_count(mol) -> int:
    from rdkit import Chem
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts("[SX4](=O)(=O)[F,Cl,Br,I]")))


def _n_substituted_sulfonamide(mol, sulfur, nitrogen, excluded: Set[int]) -> Optional[Dict[str, Any]]:
    """'N-methyl...-sulfonamide': the suffix record of the one sulfonamide of the molecule whose nitrogen carries one or
    two groups (each named by the shared namer), in the form of ``fused_amide_suffix.fused_heterocycle_n_amide`` (its
    'n_substituents' join the producer's italic-N table); None for a ring nitrogen, a branch back to the core, a second
    sulfonamide, or a ring oxo or amine (the producer's oxo branch and amine N table take no amide N groups)."""
    from rdkit import Chem
    from .fused_amide_suffix import _oxo_or_amino_on
    from ..assembly.acyl_halide_prefix import holds_principal_group
    from ..assembly.substituent_enumerator import name_substituent
    from .benzene import _build_n_substituted_sulfamoyl_prefix
    s_idx, n_idx = sulfur.GetIdx(), nitrogen.GetIdx()
    if nitrogen.IsInRing() or nitrogen.GetFormalCharge() or any(b.GetBondTypeAsDouble() != 1 for b in nitrogen.GetBonds()):
        return None
    if any(nb.GetIdx() != s_idx and nb.GetSymbol() in ("N", "O") for nb in nitrogen.GetNeighbors()):
        return None                                  # N-N: a sulfonohydrazide,:33932); N-O: not this class
    if len(mol.GetSubstructMatches(Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]"))) != 1 or _oxo_or_amino_on(mol, excluded):
        return None
    if not holds_principal_group(mol, s_idx):
        return None
    names, atoms = [], [s_idx, n_idx] + [nb.GetIdx() for nb in sulfur.GetNeighbors()
                                          if nb.GetIdx() not in excluded and nb.GetIdx() != n_idx]
    for branch in nitrogen.GetNeighbors():
        if branch.GetIdx() == s_idx:
            continue
        comp, stack = set(), [branch.GetIdx()]
        while stack:
            x = stack.pop()
            if x in comp or x in (s_idx, n_idx):
                continue
            if x in excluded:
                return None                          # the branch reaches the core: a ring, not a substituent
            comp.add(x)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
        name = name_substituent(mol, sorted(comp), branch.GetIdx())
        if not name or name == "substituent":
            return None
        names.append(name)
        atoms += sorted(comp)
    if amide_rival(mol, set(atoms) - {n_idx} | {s_idx}):
        return None                                  # an N-acyl or another amide: the senior amide is the parent
    prefix = _build_n_substituted_sulfamoyl_prefix(names) if names else None
    if not prefix:
        return None
    return {"name": prefix, "atoms": atoms, "functional_group": "sulfonamide", "type": "suffix",
            "suffix_name": "sulfonamide", "n_substituents": names}
