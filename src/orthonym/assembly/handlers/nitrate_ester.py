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
  * exactly one nitric-acid ester group. Several nitrate groups on one alcohol
    component are an ester of a polyol, named with a multivalent organyl
    group, 'Polyesters formed from a single 'alcoholic'
    component',:31815; 'ethane-1,2-diyl diacetate (PIN)':31823), which this
    producer does not build; the prefix name stays below the PIN.
  * no principal characteristic group (``principal_group`` None) and no nitrite
    ester, so any group that could compete with the ester for the parent defers;
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
    fg = getattr(features, 'functional_groups', None) or {}
    matches = fg.get('nitrooxy') or []
    if len(matches) != 1 or fg.get('nitrite'):
        return False
    return getattr(features, 'principal_group', None) is None


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
    if len(matches) != 1:
        return None
    if mol is None:
        mol = getattr(features, 'mol', None)
    if mol is None:
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
    if not r_word:
        return None

    name = f"{r_word} nitrate"
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


__all__ = ["name_nitrate_ester", "_is_nitrate_ester"]
