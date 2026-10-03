"""Nitrate-ester handler -- functional-class naming.

An ester of nitric acid R-O-NO2 is named in the two-word functional-class form
``<R> nitrate``. (heading 'Esters of mononuclear noncarbon
oxoacids', the Blue Book),:35918: "Esters of mononuclear noncarbon acids
are named in the same way as esters of organic acids (see. Alkyl
groups, aryl groups, etc. are cited as separate words, in alphanumerical order
when more than one, and followed by the name of the appropriate anion." The
nitrous-acid sibling is 'pentyl nitrite (PIN)' (:35922), built by
handlers/nitrite_ester.py; nitric acid is monobasic and its anion is 'nitrate'.

The group is perceived as ``nitrooxy`` ([O-][N+](=O)O-C, the charge-separated
drawing). 'nitrooxy' is the preselected prefix of the ester group,
:36401, '-O-NO2 nitrooxy (preselected prefix)':36405), used when a senior class
names the parent ('3-(nitrooxy)propanoic acid', 'ethyl 3-(nitrooxy)propanoate');
it has no suffix, so a molecule whose only characteristic group is the ester
has no principal group, and without this producer it was named substitutively
as '(nitrooxy)ethane', which is not the PIN, Table 4.1,:18182: esters,
class 9, rank above every class a prefix-only group can express).

Scope, fail-closed (None) outside it:
  * the oxygen carries an alcohol component: an acyl, thioacyl or imidoyl carbon
    there makes a mixed anhydride,:32272), named by the anhydride
    producer ('acetic nitric anhydride');
  * one nitric-acid ester group, or several on one clean acyclic polyol, named
    with a multivalent organyl group, 'Polyesters formed from a
    single 'alcoholic' component',:31815;,:31819, 'ethane-1,2-
    diyl diacetate (PIN)':31823): 'propane-1,2,3-triyl trinitrate'. Any other
    shape with several nitrate groups keeps its prefix name below the PIN.
  * no principal characteristic group, or one of a class junior to the esters
    ,:18182; ``nitrite_ester.principal_group_is_junior_to_esters``), which
    the alcohol component then cites as a prefix ('2-hydroxyethyl nitrate',
    '2-acetamidoethyl nitrate'); and no nitrite ester, so any group that could
    compete with the ester for the parent defers;
  * the molecule is neutral apart from the N+/O- pair of the nitrate group, and
    the alcohol component plus the four atoms of the group are the whole
    molecule.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_nitrate_ester(features: Any) -> bool:
    """Predicate (pure): one nitric-acid ester group is the only
    characteristic group that could name the parent. NO mol/features mutation;
    NO module state."""
    from .nitrite_ester import principal_group_is_junior_to_esters
    fg = getattr(features, 'functional_groups', None) or {}
    matches = fg.get('nitrooxy') or []
    if not matches or fg.get('nitrite'):
        return False
    return principal_group_is_junior_to_esters(getattr(features, 'principal_group', None))


def name_nitrate_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Emit ``<R> nitrate`` for an ester of nitric acid R-O-NO2.

    From the ``nitrooxy`` match (O-, N+, =O, ester O, R carbon) the alcohol
    component R (the subgraph at the R carbon that does not cross the ester
    oxygen) is named by the universal substituent pipeline and joined as
    ``"<R> nitrate"``. Returns None (defer) when R cannot be named or the
    molecule is outside the scope of the module docstring.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ..substituent_enumerator import name_substituent
    from .nitrite_ester import _collect_subgraph

    from ..fragment_naming import is_top_level_naming

    # A fragment named inside another name ('CCO[N+](=O)[O-]', the alcohol part
    # of '2-(nitrooxy)ethyl...acetate') becomes a substituent prefix through
    # parent_to_prefix, which reads a substitutive parent name; a functional-class
    # name is not one, so the fragment keeps its substitutive name there.
    if not is_top_level_naming():
        return None
    matches = (getattr(features, 'functional_groups', None) or {}).get('nitrooxy', [])
    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
        return None
    if len(matches) >= 2:
        # Identical nitrate anions on one polyol: functional class multiplicative
        # ('propane-1,2,3-triyl trinitrate'); see esters.name_polyol_identical_
        # inorganic_ester, the Blue Book;,:35918).
        from ...rules.esters import name_polyol_identical_inorganic_ester
        group = {a for m in matches for a in m[:4]}
        name = name_polyol_identical_inorganic_ester(
            mol, [m[4] for m in matches], {m[3] for m in matches}, group, "nitrate")
        return _emit(features, name) if name else None
    if len(matches) != 1:
        return None
    match = matches[0]
    if len(match) < 5:
        return None
    o_minus, n_plus, o_double, ester_o, r_carbon = match[:5]
    group = {o_minus, n_plus, o_double, ester_o}

    # An acyl, thioacyl or imidoyl carbon on the oxygen makes a mixed anhydride of
    # nitric acid, not an ester: (the Blue Book), "Mixed
    # anhydrides with carbonic acid, cyanic acid, and inorganic acids are named as
    # anhydrides" ('acetic cyanic anhydride (PIN)',:32276). cites an
    # alcohol component ("Alkyl groups, aryl groups, etc.").
    for bond in mol.GetAtomWithIdx(r_carbon).GetBonds():
        other = bond.GetOtherAtom(mol.GetAtomWithIdx(r_carbon))
        if (bond.GetBondTypeAsDouble() == 2.0
                and other.GetSymbol() in ("O", "S", "Se", "Te", "N")):
            return None

    # Neutral apart from the group's own N+/O- pair.
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            return None
        if atom.GetIdx() not in group and atom.GetFormalCharge() != 0:
            return None

    r_atoms = _collect_subgraph(mol, r_carbon, exclude={ester_o})
    if not r_atoms or group & set(r_atoms):
        return None
    if set(r_atoms) | group != set(range(mol.GetNumAtoms())):
        return None
    try:
        r_word = name_substituent(mol, set(r_atoms), r_carbon)
    except Exception:
        r_word = None
    from ...errors import is_refusal_sentinel
    if not r_word or is_refusal_sentinel(r_word):
        return None             # 'substituent nitrate' is no name: decline

    name = f"{r_word} nitrate"
    # A second, different ester on the same alcohol component makes this method (2)
    # of ('2-(acetyloxy)ethyl nitrate'): correct, not the PIN.
    from ...rules.esters import record_polyol_mixed_anion_non_pin
    record_polyol_mixed_anion_non_pin(mol, r_carbon, ester_o, name)
    pool = get_current_pool()
    cand = pool.add(name, "nitrate_ester", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrate_ester",
            iupac_section_cite="P-67.1.3.2", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


def _emit(features: Any, name: str) -> Optional[NamingResult]:
    """Pool the name and return it as the handler result (stereo injected)."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    cand = get_current_pool().add(name, "nitrate_ester", features)
    if cand is None:
        return None
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="nitrate_ester",
            iupac_section_cite="P-67.1.3.2", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_nitrate_ester", "_is_nitrate_ester"]
