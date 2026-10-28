"""The tiers of NUMBERING that choose among the automorphic numberings of a fused heterocycle core
(``data.fused_heterocycles._select_lowest_locant_match``): (g) the prefix cited first at the lowest locant
(``the Blue Book``, by, which leaves the stereodescriptors out, ``:3446``), (i) the lowest locants to the
isotopically modified atoms (``:3336``), (j) the CIP stereodescriptors of the core and those cited inside its prefixes,
together, at their locants (``:3346``; ``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``, ``:3399``)."""
from typing import Dict, List, Tuple


def _prefix_names(mol, match, core_atoms, sub_atoms, iupac_locants, real: bool = False) -> List[Tuple[float, str]]:
    """``(locant, name)`` of the exocyclic prefix on each substituent-bearing core atom under this match's
    numbering; an unnameable fragment falls back to a carbon-count / element-symbol proxy. The name of a
    fragment without a ring is that proxy (``ethyl`` for a 1-chloroethyl) unless ``real``: then every fragment is
    named as it is cited, with its stereodescriptors, which (j) reads."""
    from ..assembly.naming_utils import get_alkyl_name
    from ..data.fused_heterocycles import _coerce_locant_for_compare, _loc_to_float

    entries: List[Tuple[float, str]] = []
    for pattern_idx, mol_idx in enumerate(match):
        if mol_idx not in sub_atoms:
            continue
        coerced = _coerce_locant_for_compare(iupac_locants.get(pattern_idx))
        if coerced is None:
            continue
        # Gather the exocyclic substituent fragment hanging off this core atom.
        frag, seen, stack = [], set(core_atoms), [
            nb.GetIdx() for nb in mol.GetAtomWithIdx(mol_idx).GetNeighbors()
            if nb.GetIdx() not in core_atoms
        ]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            frag.append(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        name = None
        if frag and (real or any(mol.GetAtomWithIdx(a).IsInRing() for a in frag)):
            try:
                attach = next(
                    (a for a in frag
                     if any(nbr.GetIdx() == mol_idx
                            for nbr in mol.GetAtomWithIdx(a).GetNeighbors())),
                    None,
                )
                if attach is not None:
                    from ..assembly.substituent_enumerator import name_substituent
                    name = name_substituent(mol, list(frag), attach)
            except Exception:
                name = None
        if not name:
            ccount = sum(1 for a in frag if mol.GetAtomWithIdx(a).GetSymbol() == 'C')
            if ccount > 0:
                name = get_alkyl_name(ccount)
            elif frag:
                name = mol.GetAtomWithIdx(frag[0]).GetSymbol().lower()
            else:
                name = 'zzz'
        entries.append((_loc_to_float(coerced), name or 'zzz'))
    return entries


def match_tiers(mol, match, core_atoms, sub_atoms, iupac_locants) -> Dict[str, tuple]:
    """``{'alpha': (g), 'isotopes': (i), 'cip': (j)}`` of one automorphic match, for
    ``rules.locants.compare_numbering``."""
    from ..assembly.locant_omission import isotope_locant_key
    from ..assembly.naming_utils import alpha_sort_key, numbering_cip_key
    from ..data.fused_heterocycles import _coerce_locant_for_compare, _loc_to_float
    prefixes = _prefix_names(mol, match, core_atoms, sub_atoms, iupac_locants)
    alpha = sorted((alpha_sort_key(name), loc) for loc, name in prefixes)
    at = {}
    for pattern_idx, mol_idx in enumerate(match):
        coerced = _coerce_locant_for_compare(iupac_locants.get(pattern_idx))
        if coerced is not None:
            at[mol_idx] = _loc_to_float(coerced)
    parent = []
    from rdkit import Chem
    if (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        prefixes = _prefix_names(mol, match, core_atoms, sub_atoms, iupac_locants, real=True)
        try:
            from ..perception.stereo import assign_stereochemistry
            assign_stereochemistry(mol)
            parent = [(loc, mol.GetAtomWithIdx(a).GetProp('_CIPCode')) for a, loc in at.items()
                      if mol.GetAtomWithIdx(a).HasProp('_CIPCode')]
        except Exception:
            parent = []
    return {'alpha': tuple(item for pair in alpha for item in pair),
            'isotopes': isotope_locant_key(mol, at),
            'cip': numbering_cip_key(parent, prefixes)}
