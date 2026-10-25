"""N-substitution of amines, carboxamides and sulfonamides by heteroatom groups.

BLUE BOOK AUTHORITY
-------------------
 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" (the Blue Book): "Traditionally,
substitution on the nitrogen atom of amines and imines was allowed for all characteristic groups
cited as prefixes (see Table 5.1). This approach is maintained in these recommendations, unless a
higher class is formed that must be named in accordance with the seniority of classes (see."
Its examples: 'N-methoxyethanamine (PIN)' (:26600), 'N-hydroxyethanamine (PIN)' (:26602); "Substitution
of amines is permitted by -OR, -SR, -SeR, and -TeR groups, where R is an alkyl or aryl substituent
group" (:26596). The groups of Table 5.1 "Characteristic groups always cited as prefixes" (:24980)
that are not the groups of an amide of a acid (nitro, nitroso, the halogens and their
oxides: ``rules/nitramide.py``) are azido, isocyanato, isocyano, alkoxy and alkylperoxy, the
alkanesulfinyl and alkanesulfonyl groups (the amides of sulfinic and sulfonic acids), and the
chalcogen analogues of alkoxy.

 "AZIDES" (:25989): "Compounds containing a -N3 (-N=N+=N-) group attached to a parent hydride,
are named using substitutive nomenclature and the prefix 'azido'." ASSUMED for the nitrogen of an
amine: the book prints no N-azido amine; 's general sentence (Table 5.1 lists 'azido',:24990)
gives the reading, and the amine (class 19) is senior to every class a nitrogen chain could name
, class 21 "Nitrogen compounds: heterocycles, polyazanes, hydrazines...",:18197), so
'N-azido-N-methylmethanamine' is the name that reading gives.

The class of the compound is that of the host the group is a prefix of, so the groups of
``rules/nitramide.py`` ARE prefixes here when the host is a carboxamide or a sulfonamide: ranks
the amide of a carboxylic or sulfonic acid (class 7a acids, class 11 amides "in the order of the
corresponding acids",:18184) above the amide of a nitrogen or halogen acid (class 7e,, so
CH3-CO-N(CH3)-Cl is 'N-chloro-N-methylacetamide' and CH3-SO2-N(CH3)-NO is
'N-methyl-N-nitrosomethanesulfonamide'. For the nitro and nitroso groups the book says so itself,
"Nitro attached to heteroatoms" (:25957): "In the presence of a characteristic group having seniority to be named
as a suffix or a parent structure, nitro and nitroso groups can be attached to any atom", with the example
'N-methyl-N-nitrosourea (PIN)' (:25965). A group on a nitrogen of the amine host that is not one of
the amide-forming groups (azido, alkoxy...) stays a prefix of the amine.

THE CONSTRUCTION (reuse of the trusted path, never a rewrite of an emitted name)
-------------------------------------------------------------------------------
The host is the molecule with the group taken off the nitrogen (``M0``: an amine, carboxamide or
sulfonamide the engine names). Its name is read, not edited: the name of ``M0`` must be EXACTLY the
name rebuilt from independently computed parts -- the N-prefix of the other substituent of the nitrogen
(``_name_n_substituent``) in front of the name of the primary host that the skeleton alone makes
(``name_compound`` of the amine, amide or sulfonamide without N-substituents). Where it is, the
skeleton is a plain hydrocarbon group (a chain or a ring that carries no prefix of its own), so the
N-prefixes are the only prefixes of the name and the group takes its place among them in the
alphanumerical order of (``N-azido-N-methylmethanamine``, ``N-chloro-N-methylacetamide``).
Where it is not (a prefix on the skeleton, a stereodescriptor, a different parent choice), the class
is not built and the engine's other producers keep the molecule.

SCOPE (fails closed on everything else)
---------------------------------------
One acyclic N with exactly one group, the groups named here: azido; the chalcogen groups -OR, -SR, -SeR,
-TeR of an organyl group (``composed_chalcogen_group_prefix``) and -SH, -SeH, -TeH; and, on a
carboxamide or sulfonamide, -OH and the groups of ``rules/nitramide.py``. The other substituents of the N
are a carbon group, an acyl group or a sulfonyl/sulfinyl group of a plain skeleton. Every atom of the
molecule is in the groups of the N; no stereo is specified, nothing is charged or an isotope outside the
group.
Left to the producers that own them: the amide of a acid on an amine (``rules/nitramide.py``), the
N-hydroxy amine (the hydroxylamine class, and the secondary R-NH-O-R' with one carbon on each
side, the hydroxylamine handler); a ring nitrogen ('1-azidopiperidine': the heterocycle
carries the group as a prefix at its position 1); a skeleton that carries a prefix of its own; a fragment
named inside a substituent's recursion (the prefix converter has no conversion for an N-substituted amine
name, so the group would be lost to the composed prefix the general engine builds).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem

_CHALCOGENS = {8: 'O', 16: 'S', 34: 'Se', 52: 'Te'}


# --------------------------------------------------------------------------- the group on N

def _azido_group(mol, n_atom, z_atom) -> Optional[frozenset]:
    """The atoms of an azido group -N=N+=N- bonded to ``n_atom`` through ``z_atom``, else None."""
    if z_atom.GetAtomicNum() != 7 or z_atom.GetFormalCharge() != 0 or z_atom.GetDegree() != 2:
        return None
    if z_atom.GetTotalNumHs() or z_atom.GetIsAromatic() or z_atom.IsInRing():
        return None
    beta = [nb for nb in z_atom.GetNeighbors() if nb.GetIdx() != n_atom.GetIdx()]
    if len(beta) != 1 or beta[0].GetAtomicNum() != 7 or beta[0].GetFormalCharge() != 1:
        return None
    gamma = [nb for nb in beta[0].GetNeighbors() if nb.GetIdx() != z_atom.GetIdx()]
    if len(gamma) != 1 or gamma[0].GetAtomicNum() != 7 or gamma[0].GetFormalCharge() != -1:
        return None
    if gamma[0].GetDegree() != 1 or gamma[0].GetTotalNumHs():
        return None
    return frozenset((z_atom.GetIdx(), beta[0].GetIdx(), gamma[0].GetIdx()))


def _chalcogen_group(mol, n_atom, y_atom) -> Optional[Tuple[str, frozenset, str]]:
    """``(prefix, atoms, kind)`` of a chalcogen group bonded to ``n_atom``: -Y-R with R a carbon group of
    carbon and hydrogen alone ('methoxy', 'methylsulfanyl'...), or -YH ('sulfanyl', 'selanyl', 'tellanyl'
    and, on an amide or sulfonamide only, 'hydroxy': "even -OH groups and chalcogen analogues"),
    else None. Y is O, S, Se or Te."""
    if y_atom.GetAtomicNum() not in _CHALCOGENS or y_atom.GetFormalCharge() or y_atom.IsInRing():
        return None
    if y_atom.GetNumRadicalElectrons() or y_atom.GetIsotope() or y_atom.GetIsAromatic():
        return None
    symbol = _CHALCOGENS[y_atom.GetAtomicNum()]
    if y_atom.GetDegree() == 1:
        if y_atom.GetTotalNumHs() != 1:
            return None
        if symbol == 'O':
            return 'hydroxy', frozenset([y_atom.GetIdx()]), 'hydroxy'
        return {'S': 'sulfanyl', 'Se': 'selanyl', 'Te': 'tellanyl'}[symbol], frozenset([y_atom.GetIdx()]), 'amine'
    if y_atom.GetDegree() != 2 or y_atom.GetTotalNumHs():
        return None
    onward = [nb for nb in y_atom.GetNeighbors() if nb.GetIdx() != n_atom.GetIdx()]
    if len(onward) != 1 or onward[0].GetAtomicNum() != 6:
        return None
    from .sulfonamides import _branch_atoms

    branch = _branch_atoms(mol, onward[0].GetIdx(), y_atom.GetIdx())
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in branch):
        return None
    atoms = frozenset([y_atom.GetIdx()] + list(branch))
    from ..assembly.substituent_enumerator import composed_chalcogen_group_prefix

    prefix = composed_chalcogen_group_prefix(mol, atoms, y_atom.GetIdx(), [n_atom.GetIdx()])
    if not prefix:
        return None
    return prefix, atoms, 'amine'


def _group_on(mol, n_atom, z_atom) -> Optional[Tuple[str, frozenset, str]]:
    """``(prefix, atoms, kind)`` of the group that ``z_atom`` roots on ``n_atom``: kind 'amide' for the
    groups of an amide of a acid (nitro, nitroso, halogens: ``rules/nitramide.py``), 'amine'
    for the others (azido, the chalcogen groups), 'hydroxy' for -OH (a prefix of an amide or sulfonamide;
    on an amine it makes the hydroxylamine class, which the hydroxylamine handler names). None when
    ``z_atom`` roots no group named here."""
    from .nitramide import _acid_group

    acid = _acid_group(mol, n_atom, z_atom)
    if acid is not None:
        return acid.prefix_name(), acid.atoms, 'amide'
    azido = _azido_group(mol, n_atom, z_atom)
    if azido is not None:
        return 'azido', azido, 'amine'
    return _chalcogen_group(mol, n_atom, z_atom)


# --------------------------------------------------------------------------- the host

def _root_kind(mol, atom) -> Optional[str]:
    """'acyl' (carbonyl carbon), 'sulfonyl' / 'sulfinyl' (S with two / one terminal oxygens),
    'carbon' (any other carbon), else None."""
    z = atom.GetAtomicNum()
    if z == 6:
        for nb in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            if nb.GetAtomicNum() != 6 and bond.GetBondType() != Chem.BondType.SINGLE:
                return 'acyl' if (nb.GetAtomicNum() == 8 and nb.GetDegree() == 1) else None
        return 'carbon'
    if z == 16 and not atom.GetFormalCharge():
        oxo = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1
               and mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
        if len(oxo) == 2 and atom.GetDegree() == 4:
            return 'sulfonyl'
        if len(oxo) == 1 and atom.GetDegree() == 3:
            return 'sulfinyl'
    return None


def _is_plain_skeleton(mol, branch: List[int], root: int, kind: str) -> bool:
    """True when the branch rooted at ``root`` is a hydrocarbon skeleton that carries no prefix of
    its own: a chain that is one path, or one carbocycle, with the oxygens of the acyl or sulfonyl
    group on its root and nothing else (so the name of its primary amide, amine or sulfonamide has no
    prefix: 'butan-2-amine', 'benzamide', 'cyclohexanesulfonamide')."""
    atoms = [mol.GetAtomWithIdx(i) for i in branch]
    carbons = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6}
    hetero = set(branch) - carbons
    if kind in ('acyl', 'sulfonyl', 'sulfinyl'):
        allowed = {nb.GetIdx() for nb in mol.GetAtomWithIdx(root).GetNeighbors()
                   if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1}
        if kind != 'acyl':
            allowed.add(root)
        if hetero != allowed:
            return False
    elif hetero:
        return False
    if any(a.GetIsotope() or a.GetFormalCharge() for a in atoms):
        return False
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings() if set(r) & set(branch)]
    skeleton_carbons = set(carbons)
    if kind == 'acyl':
        skeleton_carbons.discard(root)
    if rings:
        if len(rings) != 1 or not rings[0] <= carbons:
            return False
        return skeleton_carbons == rings[0]
    # acyclic: the carbons (acyl carbon included) form one path
    c_nbrs = {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in carbons]
              for i in carbons}
    if any(len(v) > 2 for v in c_nbrs.values()):
        return False
    if kind == 'acyl' and len(c_nbrs[root]) > 1:
        return False
    if kind in ('sulfonyl', 'sulfinyl'):
        s_nbrs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(root).GetNeighbors() if nb.GetIdx() in carbons]
        if len(s_nbrs) != 1:
            return False
    return True


def _remove_atoms_smiles(mol, remove: set, n_idx: int) -> Optional[str]:
    """The SMILES of ``mol`` with ``remove`` taken off and the nitrogen ``n_idx`` filled with hydrogen."""
    rw = Chem.RWMol(mol)
    for idx in sorted(remove, reverse=True):
        rw.RemoveAtom(idx)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    try:
        host = rw.GetMol()
        Chem.SanitizeMol(host)
    except Exception:
        return None
    if len(Chem.GetMolFrags(host)) != 1:
        return None
    return Chem.MolToSmiles(host)


def _pin_name(smiles: str) -> Optional[str]:
    """The name of ``smiles`` as the engine's default tier gives it, or None (a label, no name). Not
    remembered across calls: a host name depends on the run (the OPSIN in use, the gate), and a refusal may be
    the state of the run and not the molecule's."""
    from ..errors import is_failure_name
    from ..namer import name_compound

    try:
        name = name_compound(smiles)
    except Exception:  # noqa: BLE001 - a host the engine cannot name is not this class
        return None
    if not name or is_failure_name(name):
        return None
    return name


def _n_prefix(name: str) -> str:
    from ..assembly.naming_utils import _wrap_n_substituent, enclose_if_compound

    return f"N-{_wrap_n_substituent(enclose_if_compound(name))}"


def name_n_hetero_substituted(mol, style: str = 'pin') -> Optional[str]:
    """The PIN of an amine, carboxamide or sulfonamide whose nitrogen carries one heteroatom group
    of this module's scope: ``N-azido-N-methylmethanamine``, ``N-chloro-N-methylacetamide``,
    ``N-methyl-N-(methylsulfanyl)methanamine``, ``N-methoxy-N-methylmethanamine``. None outside it.

    The dispatch asks twice for one molecule (the predicate, then the handler); the answer is kept for the
    naming call (``perception.molcache.cached_by_key``: never an RWMol, gone with the call). The answer depends
    on the style and on whether this is the name of a whole molecule, so both are in the key."""
    if mol is None:
        return None
    from ..assembly.fragment_naming import is_top_level_naming
    from ..perception.molcache import cached_by_key

    return cached_by_key(mol, "n_hetero_amines", (style, is_top_level_naming()),
                         lambda: _name_n_hetero_substituted(mol, style))


def _name_n_hetero_substituted(mol, style: str) -> Optional[str]:
    """``name_n_hetero_substituted`` without the memo."""
    if mol is None or style != 'pin' or len(Chem.GetMolFrags(mol)) != 1:
        return None
    # Only the name of a whole molecule. A fragment named inside a substituent's recursion is
    # asked for a parent name the prefix converter can turn into '[azido(methyl)amino]'; an amine
    # named 'N-azidomethanamine' has no such conversion, and the group would be lost to the
    # composed prefix the general engine builds for it (measured: [azido(methyl)amino]acetic acid).
    from ..assembly.fragment_naming import is_top_level_naming

    if not is_top_level_naming():
        return None
    core = _core(mol)
    if core is None:
        return None
    if Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False):
        return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return None
    n_idx, group, others = core
    prefix, group_atoms, group_kind = group

    from ..assembly.naming_utils import alpha_sort_key
    from .amides import _name_n_substituent
    from .sulfonamides import _branch_atoms

    skeletons = [o for o in others if o[1] in ('acyl', 'sulfonyl', 'sulfinyl')]
    plain = [o for o in others if o[1] == 'carbon']
    if skeletons:
        if len(skeletons) != 1 or len(plain) > 1:
            return None
        candidates = [(skeletons[0], plain)]
    else:
        if group_kind != 'amine' or not plain:
            return None  # the amide of a acid: rules/nitramide.py; N-OH: the hydroxylamine handler
        if len(plain) == 1 and _is_hydroxylamine_core_secondary(mol, n_idx):
            return None  # R-NH-O-R': the hydroxylamine handler's
        candidates = [(skel, [o for o in plain if o is not skel]) for skel in plain]

    for skel, subs in candidates:
        root, kind = skel
        skel_branch = _branch_atoms(mol, root, n_idx)
        if not _is_plain_skeleton(mol, skel_branch, root, kind):
            continue
        sub_branches = [_branch_atoms(mol, s[0], n_idx) for s in subs]
        if any(not all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in b) for b in sub_branches):
            continue
        covered = {n_idx} | set(group_atoms) | set(skel_branch)
        for b in sub_branches:
            covered.update(b)
        if len(covered) != mol.GetNumAtoms():
            continue
        name = _assemble(mol, n_idx, prefix, group_atoms, skel_branch, sub_branches, alpha_sort_key,
                         _name_n_substituent)
        if name:
            return name
    return None


def _is_hydroxylamine_core_secondary(mol, n_idx: int) -> bool:
    from ..assembly.handlers.hydroxylamine import no_disub_hydroxylamine_core

    core = no_disub_hydroxylamine_core(mol)
    return core is not None and core[0] == n_idx


def _core(mol):
    """``(n_idx, (prefix, atoms, kind), [(root, root_kind),...])`` for the one acyclic nitrogen with
    exactly one group of this module's scope and neighbours that are carbon, acyl or sulfonyl roots;
    None when the molecule has not exactly one such nitrogen."""
    from .nitramide import _is_amide_nitrogen_candidate

    found = None
    for atom in mol.GetAtoms():
        if not _is_amide_nitrogen_candidate(atom):
            continue
        groups = []
        others = []
        for nb in atom.GetNeighbors():
            group = _group_on(mol, atom, nb)
            if group is not None:
                groups.append(group)
                continue
            kind = _root_kind(mol, nb)
            if kind is None:
                groups = []
                break
            others.append((nb.GetIdx(), kind))
        if len(groups) != 1:
            continue
        if found is not None:
            return None
        found = (atom.GetIdx(), groups[0], others)
    return found


def _assemble(mol, n_idx, prefix, group_atoms, skel_branch, sub_branches, alpha_sort_key,
              name_substituent) -> Optional[str]:
    """The name from the host's parts, or None when the host is not what the construction needs."""
    from ..errors import is_refusal_sentinel

    sub_names: List[str] = []
    for branch in sub_branches:
        name = name_substituent(mol, branch, len(branch))
        if not name or is_refusal_sentinel(name):
            return None
        sub_names.append(name)

    remove_group = set(group_atoms)
    primary = _remove_atoms_smiles(mol, remove_group | {i for b in sub_branches for i in b}, n_idx)
    host = _remove_atoms_smiles(mol, remove_group, n_idx)
    if primary is None or host is None:
        return None
    parent = _pin_name(primary)
    host_name = _pin_name(host)
    if not parent or not host_name:
        return None

    # the host's name, read: it is the name rebuilt from the independent parts
    expected = ''.join(_n_prefix(n) for n in sorted(sub_names, key=alpha_sort_key)) + parent
    if len(sub_names) > 1 or host_name != expected:
        return None
    if not parent[:1].isalpha():
        return None  # a leading locant: the prefixes of the skeleton would interleave

    items = [(alpha_sort_key(prefix), prefix)] + [(alpha_sort_key(n), n) for n in sub_names]
    items.sort(key=lambda it: it[0])
    return '-'.join(_n_prefix(name) for _key, name in items) + parent
