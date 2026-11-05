"""The amido-type prefixes of an acylated nitrogen, on any attachment: '2-chloro-N-methylacetamido', 'N,2-dimethylpropanamido',
'N-methylmethanesulfonamido', '(methylcarbamoyl)amino'.

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
amide is named in two ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the
complete name of the amide to 'o'... Method (1) generates preferred IUPAC names." (:32995-32998); the book's examples
are '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33040) and '(N-cyclopropylmethanesulfonamido)acetic acid'
(:33029). The amide of the group -N(R')-CO-R is the amide R-CO-NH-R', named as a whole: the prefixes of the acyl and
the N-substituent R' are cited together in the order of ('2-chloro-N-methylacetamide', 'N,2-dimethylpropanamide'),
so no string of the acyl name and the N block is joined here. A urea group -N(R')-CO-NR2 is the amino prefix that carries
the carbamoyl prefix of the other nitrogen,:33338; '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic
acid (PIN)',:33354).

``assembly.substituent_naming.n_substituted_acyl_amido_prefix`` builds the prefix of a plain acyl from the acyl and the
N-substituent; a decorated acyl (a ring, a halogen, a branch, a second group) comes here, where the amide of the group
is named by the engine and read back against the fragment. The producers of this module serve every attachment of the
group: a ring (``acylated_n_group``, the hook of ``rules.benzene``), a parent chain (``n_substituted_sulfonamido_prefix``,
``substituent_enumerator._name_amino_branch``) and a carbon of a chain substituent (``chain_acylated_n_prefix``, the
pass of ``chain_amide_prefixes``).
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


def sulfonyl_neighbour(mol, n_idx: int, among) -> Optional[int]:
    """The sulfur of the one neighbour of the nitrogen ``n_idx`` (of the atoms ``among``) that is a clean sulfonyl group
    -SO2-R of a sulfonamide: uncharged, unlabelled, two terminal =O and one carbon, the nitrogen the fourth neighbour.
    None for any other neighbour set (a sulfinyl, a sulfonimidoyl, a second sulfonyl, no sulfur)."""
    hits = []
    for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        if nb.GetIdx() not in among or nb.GetAtomicNum() != 16:
            continue
        if nb.GetFormalCharge() or nb.GetIsotope() or nb.GetNumRadicalElectrons() or nb.GetDegree() != 4:
            return None
        oxo = [x for x in nb.GetNeighbors() if x.GetAtomicNum() == 8 and x.GetDegree() == 1
               and mol.GetBondBetweenAtoms(nb.GetIdx(), x.GetIdx()).GetBondTypeAsDouble() == 2.0]
        carbon = [x for x in nb.GetNeighbors() if x.GetAtomicNum() == 6]
        if len(oxo) != 2 or len(carbon) != 1:
            return None
        hits.append(nb.GetIdx())
    return hits[0] if len(hits) == 1 else None


def acylated_n_group(mol, n_idx: int, ring_atoms) -> Optional[dict]:
    """The substituent group of a benzene ring's nitrogen ``n_idx`` when it is an acylated nitrogen that takes a prefix of
    its own: a tertiary -N(R')-CO-R or -N(R')-SO2-R, the N-substituted amido prefix of (the Blue Book;
    '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)',:33040, '(N-cyclopropylmethanesulfonamido)acetic acid',
    :33029), or a urea -NH-CO-NR2 / -N(R')-CO-NR2, the amino prefix that carries a carbamoyl prefix,
    :33338: '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)',:33354), as ``{'name', 'atoms',
    'is_complex'}`` (the shape ``_identify_nitrogen_group`` returns). None for any other nitrogen, and for one with a
    heteroatom on it: -N(NHR)-CO-R is the hydrazin-1-yl unit of ('4-(1-acetyl-2-ethylhydrazin-1-
    yl)benzoic acid'), cited by its own prefix. The group is cited as a prefix only where a group senior to it is
    present, which the parent selection has decided before a group is asked for."""
    from ..rules.benzene import _bfs_substituent_atoms
    from .naming_utils import apply_enclosing_marks
    from .substituent_naming import n_substituted_acyl_amido_prefix
    nbrs = [nb for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors() if nb.GetIdx() not in ring_atoms]
    sulfonyl = sulfonyl_neighbour(mol, n_idx, {nb.GetIdx() for nb in nbrs})
    ureas = [nb.GetIdx() for nb in nbrs if urea_carbonyl(mol, nb.GetIdx(), n_idx) is not None]
    if any(nb.GetAtomicNum() != 6 and nb.GetIdx() != sulfonyl for nb in nbrs):  # a hydrazine or hydroxylamine unit
        return None
    if len(nbrs) == 1 and not ureas and sulfonyl is None:  # -NH-CO-R has the prefix of the secondary amides
        return None
    if sulfonyl is None and not ureas and not any(nb.GetAtomicNum() == 6 and any(
            o.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(nb.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2.0
            for o in nb.GetNeighbors()) for nb in nbrs):
        return None
    sub_atoms = _bfs_substituent_atoms(mol, n_idx, ring_atoms)
    if ureas:
        parent = {nb.GetIdx() for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors()} - set(sub_atoms)
        amido = urea_prefix(mol, n_idx, sub_atoms, next(iter(parent))) if len(parent) == 1 else None
    elif sulfonyl is not None:
        amido = None
        if parent_sulfonamide_is_senior(mol, sub_atoms):
            amido = sulfonamido_prefix(mol, n_idx, sub_atoms, ring_atoms) or decorated_n_acyl_amido_prefix(mol, sub_atoms)
    else:
        amido = n_substituted_acyl_amido_prefix(mol, n_idx, sub_atoms, ring_atoms)
    if not amido:
        return None
    return {'name': apply_enclosing_marks(amido, -1), 'atoms': sub_atoms, 'is_complex': True}


#: the principal groups that are carboxamides: the second amide of a molecule whose principal group is one of these is
#: a candidate parent beside it
_CARBOXAMIDE_GROUPS = frozenset({"primary_amide", "secondary_amide", "tertiary_amide"})


def parent_amide_is_senior(mol, acyl_c: int, group_atoms) -> bool:
    """False when a carboxamide outside ``group_atoms`` (the amido group a caller would cite as a prefix, whose amide has
    its carbonyl carbon at ``acyl_c``) that the group's amide ties or outranks is a candidate parent: the amides are one
    class, and the senior parent among them is chosen by ``rules.amide_parent.amide_parent_key`` (a ring, the
    longer chain, the multiple bonds, "Maximum number of substituent groups cited as prefixes", the Blue Book,
    : the group with the senior amide is the parent, not a prefix, '2-(N,2-dimethylpropanamido)-N-methylpropanamide'
    is not the PIN of 'CC(C)C(=O)N(C)C(C)C(=O)NC', which is 'N,2-dimethyl-N-[1-(methylamino)-1-oxopropan-2-yl]propanamide'.
    The group is cited only where an amide parent outside it is strictly senior. True where the molecule's principal group
    is not a carboxamide (an acid, an ester, a sulfonate); False where the principal amide lies in the group itself (a
    group that holds the principal group is no prefix of a parent outside it), where no other amide exists, and where it
    is not decidable."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.amide_parent import _label_map, amide_parent_key, comparable_amide_units
        from ..rules.seniority import get_principal_group, offered_principal_group
        fg = detect_functional_groups(mol)
        group, matches = (offered_principal_group(mol) or get_principal_group(mol, fg))[:2]   # an ester being named outranks
        if group not in _CARBOXAMIDE_GROUPS or mol.HasSubstructMatch(_ION):   # an ion outranks the amides (the perception does not see it)
            return True
        inside = set(group_atoms)

        def carbonyl_carbons(found):
            return {a for m in found for a in m if mol.GetAtomWithIdx(a).GetAtomicNum() == 6
                    and any(nb.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(a, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                            for nb in mol.GetAtomWithIdx(a).GetNeighbors())}
        principal = carbonyl_carbons(matches)
        if principal & inside:
            return False
        # the principal amides, and every other amide that can be a parent: the senior of them is the parent
        units = {m[0] for _, m in (comparable_amide_units(mol, fg) or ())}
        candidates = principal | {c for c in units if c not in inside}
        if not candidates:
            return False
        labels = _label_map(mol)
        every = units | principal | {acyl_c}
        return (min(amide_parent_key(mol, c, labels, every) for c in candidates)
                < amide_parent_key(mol, acyl_c, labels, every))
    except Exception:  # noqa: BLE001 - not decidable: no prefix
        return False


_SULFONAMIDE_GROUPS = frozenset({"primary_sulfonamide", "secondary_sulfonamide", "tertiary_sulfonamide"})


def _principal_is_other_group_of(mol, group_atoms, classes) -> bool:
    """True when the molecule's principal group is one of ``classes`` and a match of it lies wholly outside
    ``group_atoms``: a second group of the class the group belongs to, which the group's class only ties. False
    where the principal group is any other class or lies inside the group itself, and None where it is not
    decidable."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        group, matches = get_principal_group(mol, detect_functional_groups(mol))[:2]
        inside = set(group_atoms)
        return group in classes and any(not (set(m) & inside) for m in matches)
    except Exception:  # noqa: BLE001 - not decidable: no prefix
        return None


def parent_sulfonamide_is_senior(mol, group_atoms) -> bool:
    """False when the molecule's principal group is a sulfonamide outside ``group_atoms`` (the sulfonamido group a
    caller would cite as a prefix): two sulfonamides tie, and "Maximum number of substituent groups cited as
    prefixes" (the Blue Book) then decides, as it does for two carboxamides (``parent_amide_is_senior``). True
    where the principal group is any other class (class 16 carboxamides are senior to class 19 sulfonamides,
    SENIORITY ORDER FOR CLASSES, the Blue Book) or lies inside the group itself, and False where it is not
    decidable."""
    return _principal_is_other_group_of(mol, group_atoms, _SULFONAMIDE_GROUPS) is False


def in_peptide(mol) -> bool:
    """True when ``mol``, or the molecule the caller asked ``name`` about (the engine names fragments it cuts from
    it, whose cut amide bonds no longer show a dipeptide unit), holds a dipeptide unit.

    The chain-substituent amido pass (``chain_amide_prefixes.consume_acylated_nitrogen``) is kept out of such a molecule.
    That is a scoping of the pass, not a consequence of: "INTRODUCTION" (the Blue Book) says "Preferred
    IUPAC names (PINs) are not identified for the compounds in this Chapter", which keeps the names the peptide handler
    builds from residues ('alanylaspartylglutamic acid') below the PIN, while the systematic substitutive name of a
    dipeptide is the expected PIN of the project "Systematic substitutive names",:54247; the E-AA ruling of
    2026-10-31) and is built by the other amido routes: '(2-acetamidoacetamido)acetic acid' is pin_verified. Measured on
    110 corpus rows with a dipeptide unit, with and without the gate: 109 rows have the same name and label at the PIN and
    best-effort tiers, and one (a glutathione disulfide, best-effort) keeps the method (2) '[(2S)-2-amino-4-carboxy-1-
    oxobutyl]amino' where the pass would cite '2-amino-4-carboxybutanamido' and label it pin_unverified."""
    from ..rules.peptides import peptide_like
    if peptide_like(mol):
        return True
    from .fragment_naming import outermost_name_input
    smiles = outermost_name_input()
    if not smiles:
        return False

    def _whole_is_peptide_like() -> bool:
        whole = Chem.MolFromSmiles(smiles)
        return whole is not None and peptide_like(whole)
    # asked for every chain substituent the call names: read once per naming call (the scope of ``assembly.memo``)
    from .memo import cache_or_compute
    return bool(cache_or_compute("outermost_input_is_peptide_like", smiles, _whole_is_peptide_like))


def _side_of(mol, start: int, blocked: int, sub_set) -> set:
    """The atoms of ``sub_set`` reached from ``start`` without crossing the atom ``blocked``."""
    seen, stack = set(), [start]
    while stack:
        i = stack.pop()
        if i in seen or i == blocked or i not in sub_set:
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors())
    return seen


def urea_carbonyl(mol, c_idx: int, n_p: int):
    """``(oxo, distal_n)`` when the carbon ``c_idx`` bonded to the nitrogen ``n_p`` is the carbonyl of a urea unit
    -N-CO-N<: one terminal =O and a second acyclic, uncharged, unlabelled nitrogen on single bonds, both uncharged and
    unlabelled; else None."""
    c = mol.GetAtomWithIdx(c_idx)
    if c.GetAtomicNum() != 6 or c.IsInRing() or c.GetFormalCharge() or c.GetIsotope() or c.GetDegree() != 3:
        return None
    oxo = [x.GetIdx() for x in c.GetNeighbors() if x.GetAtomicNum() == 8 and x.GetDegree() == 1 and not x.GetFormalCharge()
           and mol.GetBondBetweenAtoms(c_idx, x.GetIdx()).GetBondTypeAsDouble() == 2.0]
    ns = [x for x in c.GetNeighbors() if x.GetAtomicNum() == 7 and x.GetIdx() != n_p]
    if len(oxo) != 1 or len(ns) != 1:
        return None
    n_d = ns[0]
    if (n_d.IsInRing() or n_d.GetFormalCharge() or n_d.GetIsotope() or n_d.GetNumRadicalElectrons()
            or any(b.GetBondTypeAsDouble() != 1.0 for b in n_d.GetBonds())):
        return None
    return oxo[0], n_d.GetIdx()


def _organyl_entries(mol, n_idx: int, roots, group) -> Optional[list]:
    """``[(name, complex)]`` of the organyl branches ``roots`` of the nitrogen ``n_idx``, each named as the amino-prefix
    passes of the chain substituent namer name theirs (``carbon_free_valence_prefix``, then ``composed_prefix_organyl_name``);
    None when a branch is charged, not a carbon or unnameable."""
    import re
    from .naming_utils import has_structural_hyphen
    from .substituent_enumerator import carbon_free_valence_prefix, composed_prefix_organyl_name
    entries = []
    for r in roots:
        comp = _side_of(mol, r, n_idx, group)
        if mol.GetAtomWithIdx(r).GetAtomicNum() != 6 or any(mol.GetAtomWithIdx(v).GetFormalCharge() for v in comp):
            return None
        fv = carbon_free_valence_prefix(mol, comp, r)
        nm = fv.prefix
        if nm is None:
            if fv.must_fail_closed:
                return None
            nm = composed_prefix_organyl_name(mol, comp, r)
        if nm is None:
            return None
        entries.append((nm, bool(re.search(r"[()]", nm)) or has_structural_hyphen(nm) or nm[:1].isdigit()))
    return entries


def urea_prefix(mol, n_idx: int, group, host: int) -> Optional[str]:
    """The prefix of the pendant urea group -N(R')-CO-NR2 whose proximal nitrogen is ``n_idx`` (one bond to ``host``,
    the group the atoms ``group``): the amino prefix that carries the carbamoyl prefix of the distal nitrogen,
    '(methylcarbamoyl)amino', '[methyl(phenylcarbamoyl)amino]'.

     (the Blue Book): "The prefixes for appropriate substituent groups derived from urea are formed
    systematically. The prefixes 'ureido' and 'ureylene' are not recommended." (:33340: 'carbamoylamino'... recommended
    for preferred IUPAC names); '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)' (:33354). The distal
    nitrogen's own substituents are cited on 'carbamoyl', the proximal nitrogen's on 'amino',
    both through the one amino-prefix assembler. Returns the bare prefix core, or None unless every atom of the group is
    accounted for by the unit, the substituents on the two nitrogens and nothing else."""
    from .composer import _assemble_decorated_amino_prefix
    n = mol.GetAtomWithIdx(n_idx)
    if (n.IsInRing() or n.GetFormalCharge() or n.GetIsotope() or n.GetNumRadicalElectrons()
            or any(b.GetBondTypeAsDouble() != 1.0 for b in n.GetBonds())):
        return None
    branches = [nb.GetIdx() for nb in n.GetNeighbors() if nb.GetIdx() != host]
    found = [(c, urea_carbonyl(mol, c, n_idx)) for c in branches]
    found = [(c, u) for c, u in found if u is not None]
    if len(found) != 1:
        return None
    c_idx, (o_idx, d_idx) = found[0]
    r_p = [b for b in branches if b != c_idx]
    r_d = [nb.GetIdx() for nb in mol.GetAtomWithIdx(d_idx).GetNeighbors() if nb.GetIdx() != c_idx]
    if len(r_p) > 1 or len(r_d) > 2:
        return None
    group = set(group)
    p_entries = _organyl_entries(mol, n_idx, r_p, group)
    d_entries = _organyl_entries(mol, d_idx, r_d, group)
    if p_entries is None or d_entries is None:
        return None
    covered = {n_idx, c_idx, o_idx, d_idx}
    for r, owner in [(r, n_idx) for r in r_p] + [(r, d_idx) for r in r_d]:
        covered |= _side_of(mol, r, owner, group)
    if covered != group:
        return None
    carbamoyl = _assemble_decorated_amino_prefix(d_entries, enclose=True, head="carbamoyl") if d_entries else "carbamoyl"
    if not carbamoyl:
        return None
    return _assemble_decorated_amino_prefix([(carbamoyl, bool(d_entries))] + p_entries, enclose=False)


def n_substituted_sulfonamido_prefix(mol, n_idx: int, sub_atoms, parent_atoms) -> Optional[str]:
    """The method-(1) prefix of an N-substituted sulfonamido branch -N(R')-SO2-R whose one bond into the parent is the
    nitrogen's: the complete name of the sulfonamide with its final 'e' changed to 'o' ('N-cyclopropylmethanesulfonamido').

     (the Blue Book): "R-SO2-NH-... of an N-substituted amide is named... (1) substitutively, by using
    a prefix formed by changing the final letter 'e' in the complete name of the amide to 'o', thus changing the suffixes
    ... 'sulfonamide' to 'sulfonamido'"; "Method (1) generates preferred IUPAC names": '(N-cyclopropylmethanesulfonamido)
    acetic acid' (:33029). The unsubstituted -NH-SO2-R is ``substituent_naming.sulfonamido_prefix_from_n_branch``.
    None unless the nitrogen has exactly one parent bond and one other branch beside the sulfonyl, and the molecule's
    principal group is not a sulfonamide the group ties (``parent_sulfonamide_is_senior``)."""
    sub_set, parent_set = set(sub_atoms), set(parent_atoms)
    n = mol.GetAtomWithIdx(n_idx)
    if (n_idx not in sub_set or n.GetAtomicNum() != 7 or n.IsInRing() or n.GetIsAromatic() or n.GetFormalCharge()
            or n.GetIsotope() or n.GetNumRadicalElectrons() or any(b.GetBondTypeAsDouble() != 1.0 for b in n.GetBonds())):
        return None
    if sum(1 for nb in n.GetNeighbors() if nb.GetIdx() in parent_set) != 1:
        return None
    branches = {nb.GetIdx() for nb in n.GetNeighbors() if nb.GetIdx() not in parent_set and nb.GetIdx() in sub_set}
    if len(branches) != 2 or sulfonyl_neighbour(mol, n_idx, branches) is None:
        return None
    other = next(i for i in branches if i != sulfonyl_neighbour(mol, n_idx, branches))
    if mol.GetAtomWithIdx(other).GetAtomicNum() != 6 or not parent_sulfonamide_is_senior(mol, sub_set):
        return None
    return decorated_n_acyl_amido_prefix(mol, sub_set)


#: an ion (the nitro group and the N-oxide are no ion of the class): ranks the radical ions, anions,
#: zwitterions and cations (classes 2 to 6, the Blue Book-18169) above every class of suffix groups, and the principal group
#: of the perception does not see an ion at all (``rules.ions`` names it from its own reader)
_ION = Chem.MolFromSmarts("[+,-;!$([N+]~[O-]);!$([O-]~[N+])]")


def group_is_principal(mol, group) -> bool:
    """True when the group ``group`` (a set of atoms) is the molecule's principal characteristic group: every match of the
    principal class lies in it, so it is the suffix of the parent and never a prefix, the Blue Book;.
    A match outside the group is a second group of the class, which ``parent_amide_is_senior`` and
    ``parent_sulfonamide_is_senior`` rank. An acyclic ester of a phosphorus or sulfur oxoacid outranks the amides classes 9 and 11), and ``rules.seniority.get_principal_group`` ranks it when its functional-class name
    can be built; while that name is being built (its organyl groups cite the amido prefix) the ester is the principal
    group asked of (``rules.seniority.offered_principal_group``). False for an ion, which the perception does not rank
    and which outranks the amides (``_ION``). True when the perception cannot tell."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group, offered_principal_group
        principal, matches = (offered_principal_group(mol)
                              or get_principal_group(mol, detect_functional_groups(mol)))
    except Exception:  # noqa: BLE001 - not decidable: the group is the principal group
        return True
    group = set(group)
    if not principal or any(not (set(m) & group) for m in (matches or ())):
        return False
    return not mol.HasSubstructMatch(_ION)     # an anion, a zwitterion and a cation outrank the group


def chain_sulfonamido_prefix(mol, n_idx: int, sub_atoms, chain_set) -> Optional[str]:
    """The '(...sulfonamido)' prefix of the branch ``sub_atoms`` of an acyclic parent whose nitrogen ``n_idx`` is on the chain
    ``chain_set`` and bears a sulfonyl (-SO2-R), '' when it bears one the prefix cannot spell, None when it bears none
    (``composer._check_for_acylamino``).

     (the Blue Book), "Method (1) generates preferred IUPAC names": '3-(methanesulfonamido)propanoic acid'
    (:33019), '(N-cyclopropylmethanesulfonamido)acetic acid' (:33029). The -NH-SO2-R of a linear alkyl or an isolated benzene
    (heavy degree 2: chain and S) is spelled here, the N-substituted -N(R')-SO2-R by ``n_substituted_sulfonamido_prefix``;
    the alkylamino fallback of the caller counts the carbon branches of the nitrogen and has no place for the sulfonyl."""
    n = mol.GetAtomWithIdx(n_idx)
    sub_set = set(sub_atoms)
    if n.GetDegree() == 3 and any(s.GetSymbol() == 'S' and s.GetIdx() in sub_set for s in n.GetNeighbors()):
        prefix = n_substituted_sulfonamido_prefix(mol, n_idx, sub_set, chain_set)
        return f"({prefix})" if prefix else ''
    if n.GetDegree() != 2:
        return None
    sulfonyl = None
    for s in n.GetNeighbors():
        if s.GetSymbol() != 'S' or s.GetIdx() not in sub_set:
            continue
        dbl_o = [o.GetIdx() for o in s.GetNeighbors() if o.GetSymbol() == 'O' and o.GetIdx() in sub_set
                 and mol.GetBondBetweenAtoms(s.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2.0]
        rest = [x.GetIdx() for x in s.GetNeighbors() if x.GetIdx() != n_idx and x.GetIdx() not in dbl_o]
        if len(dbl_o) == 2 and len(rest) == 1:
            sulfonyl = (s.GetIdx(), rest[0])
            break
    if sulfonyl is None:
        return None
    s_idx, acyl = sulfonyl
    from ..data.chain_names import get_chain_prefix
    from .composer import _pure_linear_alkyl_len
    length = _pure_linear_alkyl_len(mol, acyl, {s_idx})
    if length is not None and length >= 1:
        return f"({get_chain_prefix(length)}anesulfonamido)"
    ra = mol.GetAtomWithIdx(acyl)
    if ra.GetIsAromatic() and ra.GetSymbol() == 'C':    # an isolated benzene wholly in the branch
        rings = mol.GetRingInfo().AtomRings()
        for ring in rings:
            if (acyl in ring and len(ring) == 6 and all(r in sub_set for r in ring)
                    and all(mol.GetAtomWithIdx(r).GetIsAromatic() and mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                    and not any(set(o) != set(ring) and set(o) & set(ring) for o in rings)):
                return "(benzenesulfonamido)"
    return ''                                           # a sulfonyl whose acyl is un-nameable: fail closed


def sulfonamido_prefix(mol, n_idx: int, sub_atoms, parent_atoms) -> Optional[str]:
    """The method-(1) prefix of the sulfonamido branch ``sub_atoms`` on the nitrogen ``n_idx``,
    the Blue Book): '{R}sulfonamido' of -NH-SO2-R (``substituent_naming.sulfonamido_prefix_from_n_branch``) and
    'N-{R'}{R}sulfonamido' of -N(R')-SO2-R (``n_substituted_sulfonamido_prefix``); None for any other branch."""
    from .substituent_naming import sulfonamido_prefix_from_n_branch
    return (sulfonamido_prefix_from_n_branch(mol, n_idx, sub_atoms, parent_atoms)
            or n_substituted_sulfonamido_prefix(mol, n_idx, sub_atoms, parent_atoms))


def sulfonamido_or_urea_prefix(mol, n_idx: int, sub_atoms, parent_atoms) -> Optional[str]:
    """The prefix of the branch ``sub_atoms`` on the nitrogen ``n_idx`` when it is a sulfonamido (``sulfonamido_prefix``) or a
    urea group -N(R')-CO-NR2, the amino prefix that holds the carbamoyl prefix of the distal nitrogen and R',
    the Blue Book; 'carbamoyl(methyl)amino',:33354), through the one bond into the parent; else None."""
    found = sulfonamido_prefix(mol, n_idx, sub_atoms, parent_atoms)
    if found or len(parent_atoms) != 1:
        return found
    return urea_prefix(mol, n_idx, sub_atoms, next(iter(parent_atoms)))


def _chain_carbons(mol, attach_idx: int, sub_set) -> set:
    """The acyclic carbons of ``sub_set`` joined to the free-valence carbon ``attach_idx`` through acyclic carbons
    only: the chain whose carbons carry the prefixes of the substituent. A carbon behind a nitrogen, an oxygen
    or a ring belongs to the prefix of that group, which names it."""
    seen, stack = set(), [attach_idx]
    while stack:
        i = stack.pop()
        a = mol.GetAtomWithIdx(i)
        if i in seen or i not in sub_set or a.GetAtomicNum() != 6 or a.IsInRing():
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in a.GetNeighbors())
    return seen


def chain_acylated_n_prefix(mol, sub_set, attach_idx: int, n_idx: int, consumed):
    """``(prefix, group_atoms, host)`` when the nitrogen ``n_idx`` of the chain substituent ``sub_set`` (free valence at
    ``attach_idx``) is a pendant acylated nitrogen -N(R')-CO-R or -N(R')-SO2-R on a chain carbon, with the method-(1)
    amido prefix of the whole group; else None.

     Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference
    for citation as a principal characteristic group is present, the group R-CO-NH-, or R-SO2-NH-... of an N-substituted
    amide is named... (1) substitutively, by using a prefix formed by changing the final letter 'e' in the complete name
    of the amide to 'o'... Method (1) generates preferred IUPAC names." ('4-(N-phenylacetamido)benzoic acid' is the
    ring-attached form; this is the same group on a chain carbon: '4-[(N-phenylacetamido)methyl]benzoic acid'.) The group
    is the nitrogen, the acyl side and the one other substituent R', all one pendant group of the chain: its prefix is
    built by the builders every other attachment of the group uses, so the chain, a ring and a parent chain name it the
    same way, and the host carbon cites it like any other prefix of the chain,.

    The nitrogen must be acyclic, uncharged, unlabelled, all bonds single and not the attachment; its neighbours are the
    host (an acyclic sp3 carbon on the free-valence side) and one or two branches that do not lead back to the chain,
    exactly one of them the acyl (a carboxamide carbon, ``substituent_naming._is_amide_acyl_carbon``, or a sulfonyl, with
    the senior-amide checks of ``parent_amide_is_senior`` / ``parent_sulfonamide_is_senior``)."""
    from .substituent_naming import (_is_amide_acyl_carbon, acyl_amido_prefix_from_branch,
                                     n_substituted_acyl_amido_prefix, sulfonamido_prefix_from_n_branch)
    n = mol.GetAtomWithIdx(n_idx)
    if (n.GetAtomicNum() != 7 or n_idx == attach_idx or n_idx in consumed or n.IsInRing() or n.GetFormalCharge()
            or n.GetIsotope() or n.GetNumRadicalElectrons()
            or any(b.GetBondTypeAsDouble() != 1.0 for b in n.GetBonds())):
        return None
    nbrs = [nb.GetIdx() for nb in n.GetNeighbors()]
    if len(nbrs) not in (2, 3) or any(i not in sub_set for i in nbrs):
        return None
    sides = {i: _side_of(mol, i, n_idx, sub_set) for i in nbrs}
    hosts = [i for i in nbrs if attach_idx in sides[i]]
    if len(hosts) != 1:
        return None
    host = hosts[0]
    if host not in _chain_carbons(mol, attach_idx, sub_set):
        return None  # behind a heteroatom: the prefix of a group that is itself a prefix of the chain names it
    host_atom = mol.GetAtomWithIdx(host)
    if (host_atom.GetAtomicNum() != 6 or host_atom.IsInRing() or host_atom.GetIsAromatic() or host_atom.GetFormalCharge()
            or any(b.GetBondTypeAsDouble() != 1.0 for b in host_atom.GetBonds())):
        return None
    branches = [i for i in nbrs if i != host]
    group = {n_idx}.union(*(sides[i] for i in branches))
    if len(group) != 1 + sum(len(sides[i]) for i in branches) or group & set(consumed):
        return None
    if group_is_principal(mol, group):
        return None  # the group is the molecule's principal group: the suffix of the parent, no prefix,
    carbonyls = [i for i in branches if _is_amide_acyl_carbon(mol, i, n_idx)]
    sulfonyl = sulfonyl_neighbour(mol, n_idx, set(branches))
    ureas = [i for i in branches if urea_carbonyl(mol, i, n_idx) is not None]
    if len(carbonyls) + len(ureas) + (sulfonyl is not None) != 1:
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in branches if i not in carbonyls and i != sulfonyl):
        return None  # a hydrazine or hydroxylamine unit has its own prefix
    if ureas:
        prefix = urea_prefix(mol, n_idx, group, host)
    elif sulfonyl is not None:
        if not parent_sulfonamide_is_senior(mol, group):
            return None
        prefix = ((sulfonamido_prefix_from_n_branch(mol, n_idx, group, {host}) if len(branches) == 1 else None)
                  or decorated_n_acyl_amido_prefix(mol, group))
    elif not parent_amide_is_senior(mol, carbonyls[0], group):
        return None
    elif len(branches) == 1:
        prefix = acyl_amido_prefix_from_branch(mol, n_idx, carbonyls[0], group, verify_acid=True)
    else:
        prefix = n_substituted_acyl_amido_prefix(mol, n_idx, group, {host})
    return (prefix, group, host) if prefix else None
