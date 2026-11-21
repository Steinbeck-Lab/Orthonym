"""More ring suffixes for a partly saturated fused carbocycle: '5,6,7,8-tetrahydronaphthalene-2-carbonitrile'.

The partly saturated path spelled four suffixes ('-carboxylic acid', '-ol', '-peroxol', '-amine'); with any other
principal characteristic group it declined, so '2,3-dihydro-1H-indene-5-sulfonyl chloride' had no name below the
general engine. A functional suffix represents "the principal characteristic group", the Blue Book);
the ring suffixes of these groups are those of the ring rules: '-carbonitrile',:34720), '-carbaldehyde'
, '-carboxamide', '-sulfonic acid' / '-sulfinic acid', '-sulfonamide'
, '-sulfonyl chloride'. The group here is the one the polyfunctional producer perceives as
principal, and the suffix is claimed only when every match of it sits on a ring atom of the fused system, so no
chain or other ring can hold one and change the parent.
"""
from typing import Dict, Optional, Set, Tuple

#: principal group -> ring suffix (each consonant-initial: the parent keeps its terminal 'e')
GROUP_SUFFIX: Dict[str, str] = {
    "nitrile": "carbonitrile",
    "aldehyde": "carbaldehyde",
    "primary_amide": "carboxamide",
    "sulfonic_acid": "sulfonic acid",
    "sulfinic_acid": "sulfinic acid",
    "primary_sulfonamide": "sulfonamide",
    "sulfonyl_halide:F": "sulfonyl fluoride",
    "sulfonyl_halide:Cl": "sulfonyl chloride",
    "sulfonyl_halide:Br": "sulfonyl bromide",
    "sulfonyl_halide:I": "sulfonyl iodide",
}


def _principal_matches(mol):
    from ..perception.functional_groups import detect_functional_groups
    from .polyfunctional import get_principal_group
    groups = detect_functional_groups(mol)
    found = get_principal_group(mol, groups)
    found = found[0] if isinstance(found, tuple) else found
    return found, list(groups.get(found, ()) if found else ())


def group_pcg(mol, fused_ring_atoms: Set[int]) -> Tuple[Optional[str], Set[int]]:
    """``(kind, ring_atoms)`` for a principal group of:data:`GROUP_SUFFIX` whose every match is joined by one bond to
    one ring atom of ``fused_ring_atoms`` (at most four, one per ring atom; a sulfonyl halide only alone); ``(None,
    set)`` otherwise, and for a charged or radical species."""
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return (None, set())
    try:
        found, matches = _principal_matches(mol)
    except Exception:
        return (None, set())
    if not matches or len(matches) > 4:
        return (None, set())
    kind = found
    if found == "sulfonyl_halide":
        halogens = {mol.GetAtomWithIdx(i).GetSymbol() for i in matches[0]} & {"F", "Cl", "Br", "I"}
        if len(matches) != 1 or len(halogens) != 1:
            return (None, set())
        kind = f"sulfonyl_halide:{halogens.pop()}"
    if kind not in GROUP_SUFFIX:
        return (None, set())
    ring: Set[int] = set()
    for match in matches:
        joins = [(i, nb.GetIdx()) for i in match for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                 if nb.GetIdx() not in match]
        if len(joins) != 1 or joins[0][1] not in fused_ring_atoms or joins[0][1] in ring:
            return (None, set())
        ring.add(joins[0][1])
    from .fused_sulfur_suffix import amide_rival
    if kind not in ("sulfonic_acid", "sulfinic_acid") and not kind.startswith("sulfonyl_halide") and amide_rival(
            mol, {a for m in matches for a in m}):
        return (None, set())                         # an amide outside the group may be senior (fused_sulfur_suffix.amide_rival)
    return (kind, ring)


def group_pcg_atoms(mol, pcg_ring_atoms: Set[int]) -> Set[int]:
    """The off-ring atoms of the principal-group matches joined to ``pcg_ring_atoms`` (the suffix's own atoms)."""
    try:
        _found, matches = _principal_matches(mol)
    except Exception:
        return set()
    out: Set[int] = set()
    for match in matches:
        if any(nb.GetIdx() in pcg_ring_atoms for i in match for nb in mol.GetAtomWithIdx(i).GetNeighbors()):
            out.update(match)
    return out
