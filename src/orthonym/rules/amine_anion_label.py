"""The label and the aminide of a secondary amine anion that the suffix seam does not name.

 "Anions derived from amines and imines" (the Blue Book): the suffix 'aminide' gives the preferred
name of an amine anion ('methanaminide (PIN)', 'benzenaminide (PIN)'). (:41069, 'acetylazanide (PIN)')
covers the anions of other characteristic groups, an amide for one: the azanide of an acyl group is a PIN, the azanide
of the groups of an amine ('diphenylazanide', 'bis(2-hydroxyethyl)azanide') is a valid name that is not.

``ring_aminide`` builds the PIN where the parent is a ring of one kind of position ('N-phenylbenzenaminide',
'N-methylbenzenaminide'); ``record_non_pin_azanide`` labels the azanide of an amine below the PIN otherwise.
"""
from rdkit import Chem


def arm_is_acyl(atom) -> bool:
    """An atom with a double bond to a heteroatom: the carbon of C=O, C=S, C=N or the sulfur of S=O (the groups of an
    amide, a sulfonamide, an amidine): the nitrogen it holds is not an amine."""
    return any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() != 'C'
               for b in atom.GetBonds())


def record_non_pin_azanide(mol, anion_idx: int, name: str) -> None:
    """Record ``name`` as no PIN when none of the groups on the nitrogen is an acyl group: the anion is that of an
    amine, whose PIN is the 'aminide', the Blue Book)."""
    if name and not any(arm_is_acyl(n) for n in mol.GetAtomWithIdx(anion_idx).GetNeighbors()):
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment(name)


def _arm_atoms(mol, anion_idx: int):
    """The two groups of the nitrogen as ``(attachment atom, atoms of the group)``, or None when they share a ring."""
    neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(anion_idx).GetNeighbors()]
    rw = Chem.RWMol(mol)
    for nb in neighbours:
        rw.RemoveBond(anion_idx, nb)
    comps = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    arms = [(nb, list(next(c for c in comps if nb in c))) for nb in neighbours]
    return arms if arms[0][1] != arms[1][1] and anion_idx not in arms[0][1] + arms[1][1] else None


def ring_aminide(mol, anion_idx: int, style: str) -> str:
    """'N-R-benzenaminide' for R-[N-]-Ar where Ar is a ring whose parent hydride has one kind of position (no locant
    for the suffix,,:2939) and R is the other group; the ring is the parent. Read back to the
    full InChIKey; '' for any other anion."""
    a = mol.GetAtomWithIdx(anion_idx)
    if (a.GetSymbol() != 'N' or a.GetFormalCharge() != -1 or a.GetTotalNumHs() or a.GetDegree() != 2
            or a.IsInRing() or a.GetIsAromatic()):
        return ''
    arms = _arm_atoms(mol, anion_idx)
    if arms is None or any(mol.GetAtomWithIdx(nb).GetSymbol() != 'C' or arm_is_acyl(mol.GetAtomWithIdx(nb))
                           for nb, _ in arms):
        return ''
    rings = [i for i, (nb, _) in enumerate(arms) if mol.GetAtomWithIdx(nb).IsInRing()]
    if len(rings) == 2 and Chem.MolFragmentToSmiles(mol, arms[0][1]) == Chem.MolFragmentToSmiles(mol, arms[1][1]):
        rings = rings[:1]
    if len(rings) != 1:
        return ''
    (parent_nb, parent_atoms), (other_nb, other_atoms) = arms[rings[0]], arms[1 - rings[0]]
    keep = set(parent_atoms)
    rw = Chem.RWMol(mol)
    for i in sorted((i for i in range(mol.GetNumAtoms()) if i not in keep), reverse=True):
        rw.RemoveAtom(i)
    try:
        frag = rw.GetMol()
        Chem.SanitizeMol(frag)
    except (RuntimeError, ValueError):
        return ''
    from .charged_router import _full_inchikey_rt_ok, _reenter
    from .ions import apply_ion_suffix_to_name
    from .ring_position_kinds import one_kind_of_substitutable_position
    if any(x.GetFormalCharge() for x in frag.GetAtoms()) or not one_kind_of_substitutable_position(frag):
        return ''
    from ..assembly.naming_utils import enclose_if_compound
    from ..assembly.substituent_naming import name_substituent_fragment
    prefix = name_substituent_fragment(mol, other_atoms, other_nb, [anion_idx])
    parent = _reenter(Chem.MolToSmiles(frag), style)
    if not prefix or not parent or ' ' in parent or ',' in parent:
        return ''
    aminide = apply_ion_suffix_to_name((parent[:-1] if parent.endswith('e') else parent) + 'amine', -1)
    name = f"N-{enclose_if_compound(prefix)}{aminide}" if aminide else ''
    return name if name and _full_inchikey_rt_ok(mol, name) else ''
