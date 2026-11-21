"""Sulfur acid groups on a polycyclic hydrocarbon parent: 'naphthalene-2-sulfonamide', '2-sulfamoylnaphthalene-...'.

The polycyclic producer knew one sulfur suffix, '-sulfonic acid'; a sulfinic acid or a sulfonamide on naphthalene,
anthracene or another parent of its table had no name at the PIN tier ('naphthalene-2-sulfonamide' fell to
'bicyclo[4.4.0]deca-1(10),2,4,6,8-pentaene-3-sulfonamide'). Suffixes: sulfonic and sulfinic acids
(the Blue Book), sulfonamides (:32744); when another group is the principal group the prefix forms
'sulfino*', 'sulfamoyl*' and 'chlorosulfonyl (preselected prefix)' are used (Appendix 2,:56832,:55489;,
:36472). The group records come from the writers the fused ring and substituent producers share
(``fused_sulfur_suffix``, ``sulfur_acid_group_prefix``).
"""
from typing import Dict, Iterable, Optional, Set

#: the suffixes this module gives the polycyclic producer (its '_SUFFIX_PRIORITY' ranks them,
PAH_SULFUR_SUFFIXES = ("sulfinic acid", "sulfonamide")


def pah_sulfur_group(mol, s_idx: int, core_atoms: Set[int], frag_atoms: Iterable[int]) -> Optional[Dict]:
    """The substituent record of the sulfur group rooted at ``s_idx`` on the polycyclic core ``core_atoms``: a suffix
    record for the principal sulfinic acid or primary sulfonamide, a prefix record for such a group (or a sulfonyl
    halide) that is not the principal group, None for any other shape."""
    from .fused_sulfur_suffix import fused_sulfur_suffix
    record = fused_sulfur_suffix(mol, s_idx, set(core_atoms))
    if record is not None:
        if record["suffix_name"] not in PAH_SULFUR_SUFFIXES:
            return None                              # a sulfonyl halide suffix has its own producer (rules.sulfur)
        # an N-substituted sulfonamide carries its italic-N groups ('N-methylnaphthalene-2-sulfonamide')
        return {"name": record["suffix_name"], "atoms": record["atoms"], "is_suffix": True,
                "suffix_type": record["functional_group"], "n_substituents": record.get("n_substituents") or []}
    from ..assembly.sulfur_acid_group_prefix import sulfur_acid_group_prefix
    frag = list(frag_atoms)
    prefix = sulfur_acid_group_prefix(mol, frag, s_idx)
    return {"name": prefix, "atoms": frag, "is_complex": "sulfonyl" in prefix or "sulfinyl" in prefix} if prefix else None
