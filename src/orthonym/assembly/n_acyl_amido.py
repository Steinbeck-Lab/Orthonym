"""The N-substituted amido prefix of a tertiary acyl nitrogen: '2-chloro-N-methylacetamido', 'N,2-dimethylpropanamido'.

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
amide is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the
complete name of the amide to 'o'... Method (1) generates preferred IUPAC names." (:32995-32998); the book's example
is '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040). The amide of the group -N(R')-CO-R is the amide
R-CO-NH-R', named as a whole: the prefixes of the acyl and the N-substituent R' are cited together in the order of
 ('2-chloro-N-methylacetamide', 'N,2-dimethylpropanamide'), so no string of the acyl name and the N block is
joined here.

``assembly.substituent_naming.n_substituted_acyl_amido_prefix`` builds the prefix of a plain acyl from the acyl and the
N-substituent; a decorated acyl (a ring, a halogen, a branch, a second group) comes here, where the amide of the group
is named by the engine and read back against the fragment.
"""
import re
from typing import Optional

from rdkit import Chem

#: an acylated nitrogen of any kind: N bound to a carbon, sulfur, phosphorus, selenium or tellurium that carries =O, =S,
#: =Se or =Te (a carboxamide, a thioamide, a sulfonamide, a phosphonamide, a urea, a carbamate, a lactam, an imide)
_ACYLATED_N = Chem.MolFromSmarts("[#7]-[#6,#16,#15,#34,#52]=[#8,#16,#34,#52]")

#: an amide name that holds two or more amide groups ('butanediamide'): its 'amido' would be a divalent or multiplied
#: prefix 'diamide' to 'diamido'), not the one acyl nitrogen of the fragment
_POLYAMIDE_RE = re.compile(r"(?:di|tri|tetra|penta|hexa)(?:carbox)?amide$")


def decorated_n_acyl_amido_prefix(mol, sub_atoms) -> Optional[str]:
    """The method-(1) prefix of the group ``sub_atoms`` (the acyl nitrogen, its acyl and its one other
    substituent) from the complete name of the amide it stands for: the group capped at the nitrogen with a hydrogen
    is named as a whole ('2-chloro-N-methylacetamide') and its final 'e' changed to 'o' ('2-chloro-N-methylacetamido').
    Returns the bare prefix, or None (the caller keeps its fall-through) unless every one of these holds:

    * no atom of the group is charged, radical or labelled, so the prefix cannot lose a label or a charge;
    * the group holds one acylated nitrogen, the attachment one (``_ACYLATED_N``): with a second the amide name could
      be the name of the other amide, and its 'amido' would attach the group at the wrong nitrogen;
    * the name is the complete name of one amide, a single word that ends in 'amide' ('N-methylacetamide', not an
      acid, an ester or a 'diamide');
    * OPSIN reads the name back to the capped group on the full InChIKey (constitution and stereo): an atom the namer
      dropped is not a prefix. Fails closed when OPSIN cannot be consulted.
    The PIN status of the amide name is carried to the prefix, so a name that cites it ships as a PIN only where the
    amide name is the PIN."""
    from ..errors import is_failure_name
    from .fragment_naming import name_fragment_recursively
    from .substituent_naming import fragment_acid_name_verified, record_prefix_from_acid_status
    atoms = sorted(set(sub_atoms))
    if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms):
        return None
    try:
        capped_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
        capped = Chem.MolFromSmiles(capped_smiles)
    except Exception:  # noqa: BLE001 - a fragment that does not close names nothing
        return None
    if capped is None or len({m[0] for m in capped.GetSubstructMatches(_ACYLATED_N)}) != 1:
        return None
    name = name_fragment_recursively(capped_smiles)
    if (not name or is_failure_name(name) or " " in name or not name.endswith("amide")
            or _POLYAMIDE_RE.search(name)):
        return None
    if not fragment_acid_name_verified(name, capped_smiles):
        return None
    prefix = name[:-1] + "o"
    record_prefix_from_acid_status(name, prefix)
    return prefix


def tertiary_acyl_amido_group(mol, n_idx: int, ring_atoms) -> Optional[dict]:
    """The substituent group of a benzene ring's nitrogen ``n_idx`` when it is a tertiary acyl nitrogen -N(R')-CO-R: the
    N-substituted amido prefix of (the Blue Book; '2-(N-methylpropanamido)benzene-1-sulfonic acid
    (PIN)',:33040), as ``{'name', 'atoms', 'is_complex'}`` (the shape ``_identify_nitrogen_group`` returns). None for any
    other nitrogen, and for one with a heteroatom on it: -N(NHR)-CO-R is the hydrazin-1-yl unit of ('4-(1-
    acetyl-2-ethylhydrazin-1-yl)benzoic acid'), cited by its own prefix. The amide is cited as a prefix only where a
    group senior to it is present, which the parent selection has decided before a group is asked for."""
    from ..rules.benzene import _bfs_substituent_atoms
    from .naming_utils import apply_enclosing_marks
    from .substituent_naming import n_substituted_acyl_amido_prefix
    nbrs = [nb for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors() if nb.GetIdx() not in ring_atoms]
    if any(nb.GetAtomicNum() != 6 for nb in nbrs):  # a hydrazine or hydroxylamine unit is cited by its own prefix
        return None
    if not any(nb.GetAtomicNum() == 6 and any(
            o.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2.0
            for o in nb.GetNeighbors()) for nb in nbrs):
        return None
    sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
    amido = n_substituted_acyl_amido_prefix(mol, n_idx, sub_atoms, ring_atoms)
    if not amido:
        return None
    return {'name': apply_enclosing_marks(amido, -1), 'atoms': sub_atoms, 'is_complex': True}


#: the principal groups that are carboxamides: the second amide of a molecule whose principal group is one of these is
#: a candidate parent beside it
_CARBOXAMIDE_GROUPS = frozenset({"primary_amide", "secondary_amide", "tertiary_amide"})


def _amide_rank(mol, carbonyl_c: int) -> tuple:
    """``(0, 0)`` for the amide of a ring carboxamide (a ring is senior to a chain, (1), the Blue Book),
    else ``(1, -n)`` for the acyl chain of ``n`` carbons that starts at the carbonyl carbon (the longer chain is senior,
     (b),:20938); the smaller rank is the senior amide. The chain runs through acyclic carbon atoms only."""
    atom = mol.GetAtomWithIdx(carbonyl_c)
    if any(nb.GetAtomicNum() == 6 and nb.IsInRing() for nb in atom.GetNeighbors()):
        return (0, 0)

    def longest(idx, seen):
        best = 0
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() == 6 and not nb.IsInRing() and nb.GetIdx() not in seen:
                best = max(best, longest(nb.GetIdx(), seen | {nb.GetIdx()}))
        return 1 + best
    return (1, -longest(carbonyl_c, {carbonyl_c}))


def parent_amide_is_senior(mol, acyl_c: int, group_atoms) -> bool:
    """False when the molecule's principal group is a carboxamide outside ``group_atoms`` (the amido group a caller would
    cite as a prefix, whose amide has its carbonyl carbon at ``acyl_c``) that the group's amide ties or outranks:
     "Maximum number of substituent groups cited as prefixes" (the Blue Book) then decides ('N,N,2-trimethyl-
    3-{...}propanamide (PIN)', the example (4) counting the N-substituents), and the group with more of them is the parent,
    not a prefix: '2-(N,2-dimethylpropanamido)-N-methylpropanamide' is not the PIN of 'CC(C)C(=O)N(C)C(C)C(=O)NC', which is
    'N,2-dimethyl-N-[1-(methylamino)-1-oxopropan-2-yl]propanamide'. The group is cited only where the amide parent is
    strictly senior (a ring, or the longer chain:,. True where the principal group is not a
    carboxamide (an acid, an ester, a sulfonate) or lies inside the group itself, and False where it is not decidable."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        group, matches = get_principal_group(mol, detect_functional_groups(mol))[:2]
        if group not in _CARBOXAMIDE_GROUPS:
            return True
        inside = set(group_atoms)
        carbonyls = [a for m in matches for a in m if a not in inside and mol.GetAtomWithIdx(a).GetAtomicNum() == 6
                     and any(nb.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                             for nb in mol.GetAtomWithIdx(a).GetNeighbors())]
        return not carbonyls or min(_amide_rank(mol, c) for c in carbonyls) < _amide_rank(mol, acyl_c)
    except Exception:  # noqa: BLE001 - not decidable: no prefix
        return False
