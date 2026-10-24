"""The substituent on the nitrogen of an imidic, sulfonimidic or hydroxamic acid group is a prefix like the others.

 "Substitutive nomenclature, suffix mode" (the Blue Book,:30096): "Acids in which the
carbonyl oxygen atom of a carboxylic acid group has been replaced by =N-OH are named as *N*-hydroxy
derivatives of imidic acids named as described in." The N-hydroxy of that name is one case of a
substituent on the N of the group, and the book prints the others: 'N-sulfanylcyclopentanecarboximidic acid
(PIN)' "Functional replacement in systematic names of carboxylic acids",:30271),
'N-{[({[(2-oxoethyl)amino]sulfanyl}methyl)amino]oxy}ethanimidothioic acid (PIN)',:30039),
'N-ethyl-N-methylcarbamimidic acid (PIN)',:30764), and for the acids of sulfur
'N-hydroxymethanesulfonimidic acid (PIN)' "Hydroximic and hydroxamic acids derived from sulfonic,
sulfinic, etc. acids",:31230,:31234). A hydroxamic acid is the N-hydroxy amide,:30144;
 "Hydroxamic acids",:32821: 'N-hydroxypropanamide (PIN)', 'N-hydroxycyclohexanecarboxamide (PIN)').
The substituent on the N of an imidic acid is a carbon group in the same way ('N-methylethanimidic acid',
'N-methylethanimidothioic acid': ASSUMED -- the book prints none with a carbon group on the N of a
carboximidic acid, only the carbamimidic acid above) and, as 'sulfanyl' is, the other terminal chalcogen
groups.

So each of these groups is written with the suffix of its own parent acid (the imidic acid of the group's
chalcogen, the amide, the sulfonimidic acid) and the substituent of its N is cited with the other prefixes in
ONE alphanumerical series "ALPHANUMERICAL ORDER",:3436,:3440; 'N,4-dimethyl-N-(3-methylphenyl)benzamide
(PIN)',:32879, puts the italic N and the numbers in one list), under the italic locant N, which is essential
 "Citation of locants",:2869) so that no licence withholds a locant beside it, and one that
the parent also carries is one multiplied prefix (b),:7067): 'N,2-dimethylpropanimidic acid'.

The groups were named by handlers that put 'N-hydroxy' in front of the finished name of the parent acid
(``handlers/hydroxamic_acid.py``, ``handlers/sulfonimidic_n_hydroxy.py``), which is right only when that name
has no prefix of its own: 'N-hydroxy1-chloromethanesulfonimidic acid', nothing for 'N-hydroxy-2-methylpropanamide',
and a group of an N with a carbon group was the name of an imine and a thiol, or of nothing. This module is the
one place that reads the N of the group; ``hydroximic_acids`` keeps what only the carboximidic acids have (the
configuration of the C=N, the prefixes of the demoted group, the benzene tables).

Not built, and left to the producers that name every structure: more than one group of the molecule that
has an N-substituent, a substituent on the N that is neither a terminal OH nor a carbon group, a carbon
group that has a functional group of its own (it would be cited twice: as the substituent and as a group of
the parent), a group whose acid carbon has a heteroatom beside it (a carbonic acid derivative), and the group
demoted by a senior one (no prefix for it is printed by the book for a carbon group on the N).
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

#: the imidic acid whose N-substituted derivative a group is, by the chalcogen of its acid group
IMIDIC_GROUP = {
    "O": "imidic_acid", "S": "imidothioic_acid", "Se": "imidoselenoic_acid", "Te": "imidotelluroic_acid",
}

#: group -> (index of its N in a match, index of its acid chalcogen X in a match, index of the O on N or None,
#: the parent acid the suffix is that of: a group name, or None for the chalcogen's imidic acid).
#: hydroximic_acid: (C, imino N, O on N, X). n_substituted_imidic_acid: (C, imino N, X).
#: hydroxamic_acid: (C, =O, N, O on N). sulfonimidic_acid: (S, =O, imino N, -OH on S). sulfinimidic_acid: (S, imino N, -OH).
_SHAPE: Dict[str, Tuple[int, Optional[int], Optional[int], Optional[str]]] = {
    "hydroximic_acid": (1, 3, 2, None),
    "n_substituted_imidic_acid": (1, 2, None, None),
    "hydroxamic_acid": (2, None, 3, "primary_amide"),
    "sulfonimidic_acid": (2, None, None, "sulfonimidic_acid"),
    "sulfinimidic_acid": (1, None, None, "sulfinimidic_acid"),
}
#: the terminal -XH on the N of a group and its prefix: 'N-hydroxy', the Blue Book) and
#: 'N-sulfanyl' ('N-sulfanylcyclopentanecarboximidic acid (PIN)',,:30271)
_CHALCOGEN_H_PREFIX = {"O": "hydroxy", "S": "sulfanyl", "Se": "selanyl", "Te": "tellanyl"}
#: the groups whose N-substituent is cited as a prefix
N_PREFIX_GROUPS = frozenset(_SHAPE)
#: the groups whose C=N carries a configuration the standard InChIKey does not see (its mobile hydrogen)
CN_GROUPS = frozenset({"hydroximic_acid", "n_substituted_imidic_acid"})


def suffix_group(mol, fg_name: Optional[str], matches) -> Optional[str]:
    """The group whose suffix ``fg_name`` is written with ('hydroximic_acid' -> 'imidic_acid', 'hydroxamic_acid' ->
    'primary_amide',...).

    The group's own table row has no suffix (``SUFFIX_FORMS['hydroximic_acid']`` is None: the
    '-hydroximic acid' suffix is general nomenclature only,:30100; the '-hydroxamic acid' suffix is not
    used for a PIN,:30144). ``matches`` are the group's atom tuples, of the first of which the chalcogen is
    read. Every other group is its own."""
    shape = _SHAPE.get(fg_name or "")
    if shape is None:
        return fg_name
    if shape[3] is not None:
        return shape[3]
    match = (matches or [None])[0]
    if mol is None or match is None or len(match) <= shape[1]:
        return fg_name
    return IMIDIC_GROUP.get(mol.GetAtomWithIdx(match[shape[1]]).GetSymbol(), fg_name)


def suffix_group_atoms(fg_name: Optional[str], matches) -> frozenset:
    """The atoms the suffix of a principal group accounts for: its matches, less the oxygen on the nitrogen
    of a hydroximic or hydroxamic acid, which is the atom of the N-hydroxy prefix."""
    atoms = frozenset().union(*(set(m) for m in matches))
    shape = _SHAPE.get(fg_name or "")
    if shape is not None and shape[2] is not None:
        atoms -= {m[shape[2]] for m in matches if len(m) > shape[2]}
    return atoms


def _acid_centre_ok(mol, centre: int, others: frozenset) -> bool:
    """True when the acid carbon ``centre`` has, beside the group's own atoms, only carbon or hydrogen on it: a
    heteroatom there is a carbonic acid derivative ('carbamimidic acid',,:30758), named with the
    parent acid of its own. The S of a sulfonimidic acid is fixed by the perception pattern."""
    atom = mol.GetAtomWithIdx(centre)
    if atom.GetSymbol() != "C":
        return True
    return all(nb.GetSymbol() == "C" for nb in atom.GetNeighbors()
               if nb.GetIdx() not in others and nb.GetAtomicNum() > 1)


def _has_stereo(mol, atoms: FrozenSet[int]) -> bool:
    """True when a stereocentre or a configured double bond lies in or on ``atoms`` (its descriptor would be
    cited at the front of the name for the parent, where it does not belong:, the Blue Book)."""
    from rdkit import Chem
    for a in atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
                return True
    return False


def _carbon_substituent_atoms(mol, idx: int, n_idx: int, centre: int, group: FrozenSet[int],
                              functional_groups: Optional[Dict]) -> Optional[List[int]]:
    """The atoms (in the order the substituent is named from, its attachment atom first) of the hydrocarbon group
    at ``idx`` on the N, or None when it cannot be cited whole: it reaches an atom of the group (a ring through the
    N, a lactim), has an atom that is not carbon or has a functional group of its own (it would be cited twice, as
    this substituent and as a group of the parent), or states a configuration. A match of ``functional_groups``
    that holds an atom of the group is the group's own reading of its atoms (an imine, a thiol), not a group of the
    substituent. A test of the structure only: perception reads it (:func:`scope_n_substituted_imidic_acids`)."""
    from .amides import _bfs_substituent
    atoms = _bfs_substituent(mol, idx, {n_idx, centre})
    atom_set = frozenset(atoms)
    if atom_set & group or any(mol.GetAtomWithIdx(a).GetSymbol() != "C" for a in atoms) or _has_stereo(mol, atom_set):
        return None
    for found in (functional_groups or {}).values():
        if any(set(m) & atom_set and not set(m) & group for m in found):
            return None
    return atoms


def n_substituents(mol, fg_name: str, match, functional_groups: Optional[Dict] = None, *,
                   named: bool = True) -> Optional[List[Tuple[Optional[str], FrozenSet[int]]]]:
    """``[(name, atoms)]`` of the substituents on the N of the group ``match``; ```` for a group whose N has none
    ('methanesulfonimidic acid') and for a group that is not one of:data:`N_PREFIX_GROUPS`; None where the group
    is not built (see the module docstring).

    ``named=False`` reads the structure only: the name of a carbon group is None and the group is not asked of the
    naming pipeline, whose answer may depend on the naming context (the tier, the substituent cache), so that
    perception, which is cached as a pure function of the molecule (``assembly/memo.py``), stays one. A carbon group
    that the pipeline then cannot name makes the producer decline (:func:`n_plan`).

    The substituent is a terminal OH ('hydroxy') or SH, SeH, TeH ('sulfanyl'...: only the OH on a hydroxamic acid)
    or a hydrocarbon group named from its point of attachment by
    the pipeline that names every substituent (``rules.amides._name_n_substituent``, which the N-substituents
    of an amide are named by). A hydroxamic acid is the OH alone: ``C(=O)N(R)OH`` is a tertiary amide."""
    shape = _SHAPE.get(fg_name)
    if shape is None:
        return []
    n_idx, centre, group = match[shape[0]], match[0], frozenset(match)
    if not _acid_centre_ok(mol, centre, group):
        return None
    subs: List[Tuple[str, FrozenSet[int]]] = []
    for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
        idx = nb.GetIdx()
        if idx == centre or nb.GetAtomicNum() == 1:
            continue
        if (nb.GetSymbol() in _CHALCOGEN_H_PREFIX and nb.GetDegree() == 1 and nb.GetTotalNumHs() == 1
                and nb.GetFormalCharge() == 0 and (nb.GetSymbol() == "O" or fg_name != "hydroxamic_acid")):
            subs.append((_CHALCOGEN_H_PREFIX[nb.GetSymbol()], frozenset({idx})))
        elif nb.GetSymbol() == "C" and nb.GetFormalCharge() == 0 and fg_name != "hydroxamic_acid":
            atoms = _carbon_substituent_atoms(mol, idx, n_idx, centre, group, functional_groups)
            if atoms is None:
                return None
            name = None
            if named:
                from .amides import _name_n_substituent
                name = _name_n_substituent(mol, atoms, len(atoms), refusal_as_none=True)
                if not name:
                    return None
            subs.append((name, frozenset(atoms)))
        else:
            return None
    return subs


def _second_group_of_the_suffix(features: Any, fg_name: str, parent: Optional[str]) -> bool:
    """True when the molecule has another group of the class the suffix of ``fg_name`` is that of: the amides are one
    class, so a hydroxamic acid beside another amide is a diamide ('N-hydroxypropanediamide',,
    the Blue Book); two imidic acids of one parent acid are a diimidic acid. The principal group's own
    match is one of them, so a second is a count above one."""
    fgs = features.functional_groups or {}
    mol = features.mol
    if fg_name == "hydroxamic_acid":
        return any(fgs.get(k) for k in ("primary_amide", "secondary_amide", "tertiary_amide", "thioamide",
                                        "selenoamide", "telluroamide"))
    if fg_name in CN_GROUPS:
        return sum(1 for key in ("hydroximic_acid", "n_substituted_imidic_acid") for m in fgs.get(key) or ()
                   if suffix_group(mol, key, [m]) == parent) > 1
    return False


def n_plan(features: Any) -> Optional[List[Tuple[str, FrozenSet[int]]]]:
    """``[(name, atoms)]`` of the N-substituents ``features``' principal group cites as prefixes (````: none, or
    the group is not one of:data:`N_PREFIX_GROUPS`), or None where the group is not built.

    One group only: two of them would need the N locants to carry the numbers of their groups
    ('N1,N4-dihydroxybutanediimidic acid', the Blue Book). A sulfonimidic acid several times over with no
    substituent on any N is the plain disulfonimidic acid and unchanged."""
    group = getattr(features, "principal_group", None)
    if group not in _SHAPE:
        return []
    matches = features.principal_group_atoms or []
    mol = features.mol
    if group == "sulfonimidic_acid" and len(matches) > 1:
        found = [n_substituents(mol, group, m, features.functional_groups) for m in matches]
        return [] if all(f == [] for f in found) else None
    need = 1 + max(i for i in _SHAPE[group][:3] if i is not None)
    if len(matches) != 1 or len(matches[0]) < need:
        return None
    if _second_group_of_the_suffix(features, group, suffix_group(mol, group, matches)):
        return None
    return n_substituents(mol, group, matches[0], features.functional_groups)


def scope_n_substituted_imidic_acids(mol, functional_groups: Dict) -> Dict:
    """Keep the ``n_substituted_imidic_acid`` matches of ``functional_groups`` only where the group is built, else drop
    the key (the molecule is then read as before: the =N-R as an imine, the -XH as a thiol or alcohol).

    The group has a name only as the principal group: the book prints no prefix for a carbon group on the N of an
    imidic acid demoted by a senior group (the prefix for the unsubstituted group is 'C-hydroxycarbonimidoyl',
     (1), the Blue Book, 'hydroxy' with 'imino' at a chain end, (2),:30021), so where another
    group is senior to it, or of its rank, the imine and the thiol or alcohol keep their own prefixes (the same
    scoping ``_detect_functional_groups_impl`` gives the chalcogen anhydrides under a senior acid). Dropped too: more than one
    such group (:func:`n_plan` builds one), and a group whose N-substituent is not one the producers cite
    (:func:`n_substituents`). The test reads the structure only (``named=False``): perception is a pure function of
    the molecule; whether the substituent's name can be built is the producer's decision."""
    from .seniority import SENIORITY_ORDER
    key = "n_substituted_imidic_acid"
    matches = functional_groups.get(key) or []
    if not matches:
        return functional_groups
    kept = [m for m in matches if n_substituents(mol, key, m, functional_groups, named=False)]
    if len(kept) == 1 == len(matches):
        rank = SENIORITY_ORDER.index(rank_group(mol, key, kept))
        for name, found in functional_groups.items():
            if name == key or not found:
                continue
            place = (SENIORITY_ORDER.index(rank_group(mol, name, found)) if name in SENIORITY_ORDER else None)
            if place is not None and place <= rank:
                kept = []
                break
    else:
        kept = []
    if kept:
        functional_groups[key] = kept
    else:
        functional_groups.pop(key, None)
    return functional_groups


def group_at_rank(mol, functional_groups, fg_name: str):
    """``(group, matches)`` of the hydroximic or N-substituted imidic acids of ``functional_groups`` whose rank is
    ``fg_name``'s: those whose imidic acid it is; ``(None, )`` for a ``fg_name`` that is not one of the four.

     class 7a (the Blue Book): the 'Suffix' acids are "each followed in turn by the
    corresponding peroxy, imidic, and hydrazonic acids. Chalcogen analogues follow each of the
    corresponding oxygen acids". The N-substituted derivative of an imidic acid is that acid for the order
    of citation as principal group,:30129: the prefix forms are for the case "When another
    group is present that has seniority for citation as principal group"), so it ranks as the imidic acid of
    its acid chalcogen: with the imidic acid for the O acid, with the imidothioic, imidoselenoic or
    imidotelluroic acid for the S, Se or Te acid, and above the hydrazonic acids."""
    if fg_name not in IMIDIC_GROUP.values():
        return None, []
    for key in ("hydroximic_acid", "n_substituted_imidic_acid"):
        found = [m for m in (functional_groups.get(key) or ()) if suffix_group(mol, key, [m]) == fg_name]
        if found:
            return key, found
    return None, []


def rank_group(mol, fg_name: str, matches) -> str:
    """The group in ``SENIORITY_ORDER`` whose place ``fg_name`` has: the imidic acid of its chalcogen for a
    hydroximic or N-substituted imidic acid class 7a, the Blue Book), the group itself
    for every other."""
    if fg_name in CN_GROUPS:
        return suffix_group(mol, fg_name, matches) if mol is not None and matches else "imidic_acid"
    return fg_name


def render_n_prefix(name: str) -> str:
    """'N-methyl', 'N-(2-chloroethyl)', 'N-hydroxy': the substituent ``name`` on the N, as one block."""
    from .locanted_prefixes import locanted_prefix_block
    return locanted_prefix_block([(name, "N")]) or ""


def ring_anchored_atoms(mol, fg_name: Optional[str], matches, ring_atoms) -> Set[int]:
    """Atoms of the groups ``matches`` of ``fg_name`` (one of:data:`N_PREFIX_GROUPS`) whose acid atom is bonded to an
    atom of ``ring_atoms``: the suffix of the ring, with the N-substituent the group cites itself.

    ``composer.ring_anchored_pg_atoms`` counts a group anchored to a ring only by a CARBON bonded to it (the
    carboxylic acid, the carbaldehyde) and takes the atoms of the match, so the branch of a ring sulfonimidic acid
    (its S is the atom on the ring) and the group on the N of any of these groups was read as a substituent of
    the ring: a carbon group on the N came out as a stray 'methyl' prefix, and the ring suffix kept its locant
    beside it ('N-methylcyclohexane-1-carboximidic acid', where the book writes no locant for the monosubstituted
    ring, 'N-sulfanylcyclopentanecarboximidic acid (PIN)',:30271)."""
    if fg_name not in _SHAPE:
        return set()
    anchored: Set[int] = set()
    for match in matches or ():
        if any(nb.GetIdx() in ring_atoms for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors()):
            anchored.update(match)
            anchored.update(a for found in (n_substituents(mol, fg_name, match) or ()) for a in found[1])
    return anchored


def group_nitrogens(features: Any) -> Set[int]:
    """The N atoms of the principal group when it is one of:data:`N_PREFIX_GROUPS` that is built, else none: the
    branch that holds such an N is the group's own (its substituent is cited by the group as an italic-N prefix,
    :func:`with_n_prefix_fragments`), so the walk that names the parent's substituents must not name it again
    ('3-hydroxy-N-methyl-1-(methylamino)propanimidic acid')."""
    group = getattr(features, "principal_group", None)
    if group not in _SHAPE or n_plan(features) is None:
        return set()
    return {m[_SHAPE[group][0]] for m in features.principal_group_atoms or [] if len(m) > _SHAPE[group][0]}


def n_prefix_strings(features: Any) -> Tuple[Optional[str], List[str]]:
    """``(group the suffix is written with, rendered N prefixes)`` of a molecule's principal group.

    For a group of:data:`N_PREFIX_GROUPS` that is built: its parent acid and ['N-hydroxy'], ['N-methyl']...; for a
    group that is not built, and for every other principal group, the group itself and nothing: the caller
    finds no suffix for 'hydroximic_acid' and declines."""
    group = features.principal_group
    if group not in _SHAPE:
        return group, []
    plan = n_plan(features)
    if plan is None:
        return group, []
    return suffix_group(features.mol, group, features.principal_group_atoms), [render_n_prefix(n) for n, _ in plan]


def n_italic_locant_cited(features: Any, fragments) -> bool:
    """True when ``features`` is a group of:data:`N_PREFIX_GROUPS` whose name cites an italic N locant: the
    fragments hold a prefix that starts with it ('N-hydroxy', 'N,2-dimethyl', 'N-(2-chloroethyl)').

     "Citation of locants" (the Blue Book): when any locant is essential all are cited, and an
    italic N says which atom of the parent carries the group, so no position of that parent compound can be
    left without its locant. Every licence that empties a scope of its locants to
     declines on it: the isomer a group on N would make is not the one on the carbon, whatever the
    rest of the scope looks like. (The licences read the molecule's hydrogens, and the N of this group holds
    none: its substituent is a position of the parent compound, not a decoration of the group.)"""
    import re
    return getattr(features, "principal_group", None) in N_PREFIX_GROUPS and any(
        f.fragment_type == "prefix" and re.match(r"N(?:[,'\d]|-)", f.text or "") for f in (fragments or ()))


def with_n_prefix_fragments(features: Any, prefixes: list) -> Optional[list]:
    """``prefixes`` (name fragments) with the N-substituent of the principal group cited among them, or None.

    The fragment carries its own locant in its text, as ``_generate_alkyl_prefixes`` does ('2-methyl'), and
    ``text_without_locants`` is the same text: an N locant is essential and no licence withholds it. A prefix
    the chain or ring already carries under the substituent's name is one multiplied prefix with it
    ('N,2-dimethyl', 'N,2-dihydroxy', (b)); the existing fragment is re-rendered from its own locants,
    and only when that re-rendering reproduces what the assembler would have printed for it (a fragment this
    cannot read back is refused, never edited). Unchanged for a group with no N-substituent; None (the caller
    declines) for a group that is not built (see:func:`n_plan`)."""
    from ..assembly.composer import NameFragment
    from ..assembly.naming_utils import alpha_sort_key
    from .locanted_prefixes import locanted_prefix_block

    plan = n_plan(features)
    if plan is None:
        return None
    if not plan:
        return prefixes
    if len(plan) != 1:
        return None
    name, atoms = plan[0]
    same = [f for f in prefixes if str(alpha_sort_key(f.text)) == str(alpha_sort_key(name))]
    if len(same) > 1:
        return None
    if not same:
        text = render_n_prefix(name)
        if not text:
            return None
        return list(prefixes) + [NameFragment(
            text=text, fragment_type="prefix", text_without_locants=text, atoms=frozenset(atoms))]
    old = same[0]
    if not old.locants or not all(isinstance(loc, int) and not isinstance(loc, bool) for loc in old.locants):
        return None
    pieces = [(name, str(loc)) for loc in old.locants]
    printed = old.text if (old.text or "")[:1].isdigit() else f"{','.join(map(str, old.locants))}-{old.text}"
    if locanted_prefix_block(pieces) != printed:
        return None
    merged = locanted_prefix_block(pieces + [(name, "N")])
    if not merged:
        return None
    # The ring parent's prefixes carry no atoms (``composer._generate_ring_alkyl_prefixes``); the merged prefix then
    # carries none either, as the fragment it replaces, and the name is proved by its read-back as before.
    merged_atoms = None if old.atoms is None else old.atoms | frozenset(atoms)
    return [
        NameFragment(text=merged, fragment_type="prefix", text_without_locants=merged,
                     atoms=merged_atoms) if f is old else f
        for f in prefixes
    ]
