"""The functional class ester of one Group 13 or 14 hub: the pseudoester groups of one atom named as one multivalent group.

Split out of ``rules.esters`` (the size budget of ``tests/unit/test_size_budget.py``): the reading of a free carboxylic
acid as the anion of a multiplied functional class name (``anion_of_free_acid``, ``_acid_substitution``) and the namer of
a hub's ester groups (``name_polyvalent_group_ester``, 'dibutylstannanediyl diacetate'). The carbon polyol esters of
``esters._try_functional_class_diol_diester`` read their acid with the same primitive.
"""
from typing import List, Optional, Tuple

from rdkit import Chem

from ..perception.smarts_cache import compiled as _compiled_smarts
from .esters import (_acid_name_to_ate, _free_acid_name, _multiplied_anion, _noncarbon_ester_cut,
                     _severed_acid_smiles)


def _acid_substitution(acid_mol) -> Tuple[bool, bool]:
    """``(substituted, plain)`` of the anion of the free carboxylic acid ``acid_mol`` (one carboxyl, sanitised).

     "General methodology" (the Blue Book): the "simple components" ("unsubstituted parent hydrides...;
    functionalized parent hydrides, such as benzenesulfonic acid; or retained names, such as acetic acid") "are
    multiplied by the multiplicative prefixes 'di', 'tri', etc." (:7033) and "(c) any component which is substituted
    automatically requires use of the multiplicative forms 'bis', 'tris', etc." (:7035). Read from the structure of the
    acid, the one source of both the anion name and its multiplier:

    * ``substituted`` False: formic acid; an acid whose carboxyl sits on a hetero atom that holds no group (the retained
      'carbamic acid', 'carbonochloridic acid'); an unbranched chain of carbon atoms; a ring system or an assembly of
      identical rings that is the whole of the acid but its carboxyl. Any other acid carries a substituent prefix
      ('chloroacetic', '2-methylpropanoic', '4-chlorobenzoic', 'N-methylcarbamic').
    * ``plain``: the unsubstituted reading is certain (formic acid, a chain, a monocycle that is aromatic or has no ring
      double bond: no hydro prefix, no retained name that stands for a substituted acid). A ring system with several
      rings, and a hetero-atom acid, are unsubstituted by the rule above and not plain: a name that needs the
      spelling to be certain (the PIN label) leaves them out.
    """
    carboxyl = acid_mol.GetSubstructMatches(_compiled_smarts("[CX3](=O)[OX2H1]"))
    if len(carboxyl) != 1:
        return True, False
    carboxyl_c = carboxyl[0][0]
    rest = {a.GetIdx() for a in acid_mol.GetAtoms() if a.GetAtomicNum() > 1} - set(carboxyl[0])
    if not rest:
        return False, True
    attached = [n for n in acid_mol.GetAtomWithIdx(carboxyl_c).GetNeighbors() if n.GetIdx() in rest]
    if len(attached) != 1:
        return True, False
    first = attached[0]
    if first.GetAtomicNum() != 6:
        return len(rest) > 1, False
    atoms = [acid_mol.GetAtomWithIdx(i) for i in rest]
    if all(a.IsInRing() for a in atoms):
        rings = [set(r) for r in acid_mol.GetRingInfo().AtomRings()]
        systems: List[set] = []
        for ring in rings:
            for system in [x for x in systems if x & ring]:
                ring |= system
                systems.remove(system)
            systems.append(ring)
        if len(systems) == 1:
            monocycle = len(rings) == 1
            plain = monocycle and (
                all(a.GetIsAromatic() for a in atoms)
                or not any(b.GetBondType() == Chem.BondType.DOUBLE
                           for b in acid_mol.GetBonds() if b.IsInRing()))
            return False, plain
        if len(systems) > 1:
            # an assembly: the rings must be the same ring joined to itself ('[1,1'-biphenyl]-4-carboxylic acid')
            def shape(system):
                return Chem.MolFragmentToSmiles(acid_mol, atomsToUse=sorted(system), canonical=True)
            return len({shape(x) for x in systems}) != 1, False
    chain_ok = not any(a.IsInRing() or a.GetAtomicNum() != 6 for a in atoms) and all(
        sum(1 for n in a.GetNeighbors() if n.GetIdx() in rest) <= (1 if a.GetIdx() == first.GetIdx() else 2)
        for a in atoms)
    return (not chain_ok), chain_ok


def anion_of_free_acid(acid_smi: str) -> Optional[Tuple[str, bool, bool]]:
    """``(anion, substituted, plain)`` of the free carboxylic acid ``acid_smi`` (one carboxyl), else None: the acid named
    through the general pipeline (``_free_acid_name``), its ester anion (``_acid_name_to_ate``) and the reading of
    ``_acid_substitution`` that picks its multiplier. The one reading of an acid from its structure that the hub groups
    (``name_polyvalent_group_ester``) and the carbon polyol esters (``_try_functional_class_diol_diester``) share.

    ``plain`` also needs the anion not to begin with a multiplying prefix of: (c) "simple substituent
    prefixes and functionalized parent hydrides beginning with a multiplicative prefix" (the Blue Book,
    'di(dodecyl)', 'di(pentanal)') and (d) 'dec' enclose it, and the multiplied spelling is bare here
    (``_multiplied_anion``), so the spelling is not certain for 'dodecanoate' or 'docosanoate' alike. The prefixes are all
    of the table (``anhydrides._begins_with_multiplier``: 'henicosa', 'docosa', 'hentriaconta',... as well as 'di'
    to 'icosa')."""
    acid_mol = Chem.MolFromSmiles(acid_smi)
    if acid_mol is None or len(acid_mol.GetSubstructMatches(_compiled_smarts("[CX3](=O)[OX2H1]"))) != 1:
        return None
    acid_name = _free_acid_name(acid_smi)
    anion = _acid_name_to_ate(acid_name) if acid_name else None
    if anion is None:
        return None
    from .anhydrides import _begins_with_multiplier
    substituted, plain = _acid_substitution(acid_mol)
    return anion, substituted, plain and not (anion.startswith("dec") or _begins_with_multiplier(anion))


def name_polyvalent_group_ester(mol, group: list) -> Optional[str]:
    """ "Polyesters formed from a single 'alcoholic' component" (the Blue Book): "When anions are
    identical functional class multiplicative nomenclature is used. Names are formed by citing the multivalent group,
    the multiplicative prefix, and the multiplied anionic component name. Multiplicative prefixes 'di', 'tri', etc. are
    used when anions are unsubstituted; when substituted, prefixes 'bis', 'tris', etc." -- the organyl of the
    ester is one atom of a Group 13 or 14 hydride that bears every group of ``group`` ('dibutylstannanediyl diacetate',
    'alumanetriyl tris(pentafluoropropanoate)'; the book prints 'dimethyl dibutylstannanediyl dibutanedioate (PIN)',
    ,:38202, '1,1,3,3-tetramethyldistannoxane-1,3-diyl dipropanoate (PIN)',:38252, and 'alumanetriyl
    tri(octadecanoate) (a pseudoester)',,:37526).

    The group is named on the parent hydride (``mononuclear_hydrides.name_hub_polyvalent_group``), the acid is the one
    free acid of every group (the same SMILES, with one carboxyl): it gives the anion name and, through
    ``_acid_substitution``, the multiplier, which ``_multiplied_anion`` joins to the anion as it does for the carbon
    polyol esters (``_try_functional_class_diol_diester``). An acid the free-acid namer names is the acid of the name,
    a ring acid included.

    The label. "PRESELECTED NAMES" (:2062): "the names based on these parent hydrides [alumane, gallane,
    indigane and thallane] currently do not have PIN status", so those names are labelled below the PIN (the default
    tier gives no name, a wider tier gives this one). The hydrides of Si, Ge, Sn and Pb are the parents of PINs
     "PREFERRED IUPAC NAMES",:2056: "the label 'PIN' is added to the names of compounds whose parent hydride
    contains at least one of the following elements: B, Si, Ge, Sn, Pb... and that also contain at least one carbon
    atom"), the book prints the PIN of the stannanediyl and the distannoxanediyl esters above, and their groups are the
    ones of this rule, so the name of one of them is the PIN where every word of it is certain: an acid that
    ``_acid_substitution`` reads as plain (formic acid, an unbranched chain, an aromatic or saturated monocycle) whose
    anion does not begin with a multiplying prefix (``_multiplied_anion``). Any other acid leaves the name below the PIN.
    None for any shape outside the class: the groups of two hubs, a hub with a halogen or a heteroatom, a mixed set of
    anions, an acid with a second carboxyl.
    """
    from .isotope_alternative import alternative_skeleton_active
    if alternative_skeleton_active():
        return None             # no offer in the second skeleton of the isotope decorator (``rules.isotope_alternative``)
    cuts = [_noncarbon_ester_cut(mol, m[0]) for m in group]
    if any(c is None for c in cuts) or len({organyl for _, organyl in cuts}) != 1:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    from .mononuclear_hydrides import name_hub_polyvalent_group
    hub_group = name_hub_polyvalent_group(mol, cuts[0][1], [ester_o for ester_o, _ in cuts])
    if hub_group is None:
        return None
    acids = {_severed_acid_smiles(mol, [m], [cut]) for m, cut in zip(group, cuts)}
    if len(acids) != 1 or None in acids:
        return None
    read = anion_of_free_acid(acids.pop())
    if read is None:
        return None
    anion, substituted, plain = read
    name = f"{hub_group} {_multiplied_anion(anion, len(group), substituted)}"
    if not (plain and mol.GetAtomWithIdx(cuts[0][1]).GetSymbol() in _PIN_HYDRIDE_HUBS):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(name)
    return name


#: the hubs whose hydrides are the parents of PINs, the Blue Book): the polyvalent group of one of them
#: with a plain anion is the PIN of its pseudoester; the Group 13 metals are preselected parents only,:2062)
_PIN_HYDRIDE_HUBS = ("Si", "Ge", "Sn", "Pb")
