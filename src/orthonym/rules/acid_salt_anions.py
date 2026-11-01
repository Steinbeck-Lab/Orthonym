"""Partially ionized polycarboxylic acids on one monocyclic ring method (1))."""
from typing import Dict

from rdkit import Chem

from .ions import (
    _anion_name_keeps_hydron_sites,
    _carboxyl_ring_anchor,
    _charged_ring_locants,
    _is_isolated_benzene_atom,
    _join_prefix_stem,
    _order_ring_cycle,
    _p74_bare_ring_stem,
    _p74_ring_substituent_prefix,
)


def name_ring_acid_salt_anion(mol, anion_site: Dict) -> str:
    """ method (1) (heading **** "Acid salts of polybasic organic acids are
    named in two ways", the Blue Book) for a partially ionized polycarboxylic acid whose
    ionized carboxy groups stand on one monocyclic ring: "(1) by substitutive nomenclature in
    which the free acid is cited as a prefix to the name of the anion" (:31593), and "Method (1)
    generates preferred IUPAC names, except when the structure of the acid salt is unknown."
    (:31596). The anion is senior to the acid, '4 Anions' before '7 Acids',:18167),
    so the ionized groups are the suffix and each un-ionized group is the 'carboxy' prefix:
    '2-carboxybenzoate' for O=C([O-])c1ccccc1C(=O)O, '4-carboxybenzoate' for the terephthalate,
    '3-carboxypyridine-2-carboxylate'. The retained 'benzoic acid' is the parent of a benzene ring
    with one ionized group,:29717, 'C6H5-COOH benzoic acid (PIN)'); any other ring,
    or two or more ionized groups, takes the '-carboxylate' / '-dicarboxylate' suffix on the
    parent hydride.

    ONE numbering for the whole name, the one the zwitterion builder uses: the ring
    heteroatoms, the indicated hydrogen, then the ionized groups (the suffix, (c)), then the
    prefixes (:3219; ``_charged_ring_locants`` with the ionized attachment atoms as the centres);
    every other substituent is cited in it by ``_p74_ring_substituent_prefix`` and the stem is the
    ring with every substituent deleted (``_p74_bare_ring_stem``). Declines ('') for a fused or
    bridged ring (no simple cycle), a ring system that is not the only ring, any charge outside
    the ionized groups, stereochemistry, a substituent the authoritative namer cannot spell, and
    a molecule that is not partially ionized. The name ships only when OPSIN reads it back to
    ``mol`` with the hydrons on the same atoms."""
    from ..perception.rings import get_ring_systems
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    if mol.GetRingInfo().NumRings() != 1:
        return ''
    if (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        return ''
    from rdkit.Chem import MolFromSmarts
    free_pat = MolFromSmarts('[CX3](=O)[OX2H1]')
    if free_pat is None or not mol.HasSubstructMatch(free_pat):
        return ''
    charged = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]
    if not charged:
        return ''
    try:
        ring_system = None
        for rs in get_ring_systems(mol, include_spiro=True):
            ring_system = rs
            break
    except (RuntimeError, ValueError):
        return ''
    if ring_system is None or _order_ring_cycle(mol, ring_system) is None:
        return ''
    carboxyl_atoms = set()
    attach = []
    for idx in charged:
        o = mol.GetAtomWithIdx(idx)
        if o.GetAtomicNum() != 8 or o.GetFormalCharge() != -1:
            return ''
        anchor = _carboxyl_ring_anchor(mol, idx, ring_system)
        if anchor is None:
            return ''
        carboxyl_c, ring_attach = anchor
        attach.append(ring_attach)
        carboxyl_atoms.add(carboxyl_c)
        for nb in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
            if nb.GetAtomicNum() == 8:
                carboxyl_atoms.add(nb.GetIdx())
    if len(set(attach)) != len(attach):
        return ''
    locants, _indicated_h = _charged_ring_locants(mol, ring_system, tuple(attach))
    if not locants:
        return ''
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, carboxyl_atoms)
    if prefix is None:
        return ''
    k = len(attach)
    locs = sorted(locants[a] for a in attach)
    if k == 1 and _is_isolated_benzene_atom(mol, attach[0]):
        name = f"{prefix}benzoate"
    else:
        stem = _p74_bare_ring_stem(mol, ring_system, attach[0])
        if not stem:
            return ''
        from ..assembly.naming_utils import get_suffix_multiplier_prefix
        try:
            mult = get_suffix_multiplier_prefix(k, 'carboxylate') if k > 1 else ''
        except ValueError:
            return ''
        name = (f"{_join_prefix_stem(prefix, stem)}-{','.join(str(x) for x in locs)}"
                f"-{mult}carboxylate")
    return name if _anion_name_keeps_hydron_sites(mol, name) else ''


def name_acetate_acid_salt_anion(mol, anion_site: Dict) -> str:
    """ method (1) (the Blue Book) for the one carboxylate of a partially ionized acid that is
    the acetate of 'X-CH2-COO(-)' with X an oxygen, sulfur or nitrogen atom: the anion is the parent and
    every un-ionized acid group is the prefix 'carboxy' ('[bis(carboxymethyl)amino]acetate' for the
    nitrilotriacetic acid mono-anion, '(carboxymethoxy)acetate', '[(carboxymethyl)sulfanyl]acetate').

    "Method (1) generates preferred IUPAC names, except when the structure of the acid salt is unknown." (:31596);
    "(1) by substitutive nomenclature in which the free acid is cited as a prefix to the name of the anion"
    (:31593); the anion is senior to the acid, '4 Anions' before '7 Acids',:18167) and
    "Choice of an anionic parent structure" (:41261) takes the parent with the anionic centre first (:41265). The
    neutral acid names these molecules multiplicatively ('2,2',2''-nitrilotriacetic acid,, a name whose
    suffixes are all the acid's, so the name of the fully ionized anion has more charges than the input. Here the
    one anionic centre has no parent but the chain of two carbons, since the atom X ends the chain: acetic acid,
    retained as the PIN and substitutable,:29715), so 'acetate',:40955) with the group
    on X as its prefix, cited without a locant for the only position of the parent that can carry it. Declines ('') for
    any other shape: the group's namer is the authoritative substituent namer, and the name ships only when OPSIN reads
    it back to ``mol`` with the hydrons on the same atoms.

    The label is the PIN's (the maintainers' ruling of 2026-11-01): condition (3) (:23184) leaves multiplicative
    nomenclature out, since an 'ate' and an 'ic acid' are two parents, so method (1) (:31596) gives the PIN. The project's
    PIN vocabulary flags a 'carboxymethyl' on an acetic ACID only (``pin_vocabulary._CARBOXYMETHYL_ON_ACETIC``)."""
    from rdkit.Chem import MolFromSmarts
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != -1:
        return ''
    if not mol.HasSubstructMatch(MolFromSmarts('[CX3](=O)[OX2H1]')):
        return ''
    oxygen = mol.GetAtomWithIdx(anion_site['atom_idx'])
    carboxyl = next((n for n in oxygen.GetNeighbors() if n.GetSymbol() == 'C'), None)
    if carboxyl is None or carboxyl.IsInRing() or carboxyl.GetDegree() != 3:
        return ''
    alpha = next((n for n in carboxyl.GetNeighbors() if n.GetSymbol() == 'C' and n.GetIdx() != oxygen.GetIdx()), None)
    if alpha is None or alpha.IsInRing() or alpha.GetIsAromatic() or alpha.GetTotalNumHs() != 2 or alpha.GetDegree() != 2:
        return ''
    hetero = next((n for n in alpha.GetNeighbors() if n.GetIdx() != carboxyl.GetIdx()), None)
    if hetero is None or hetero.GetSymbol() not in ('O', 'S', 'N') or hetero.GetIsAromatic() or hetero.GetFormalCharge():
        return ''
    if hetero.IsInRing() or hetero.GetSymbol() == 'N' and hetero.GetTotalNumHs() + hetero.GetDegree() != 3:
        return ''
    from .amine_anion_label import arm_is_acyl
    if any(arm_is_acyl(n) for n in hetero.GetNeighbors() if n.GetIdx() != alpha.GetIdx()):
        return ''  # an amide, an ester or a thioester: the group on X is an acyl group, not an amine, ether or sulfide
    rw = Chem.RWMol(mol)
    rw.RemoveBond(alpha.GetIdx(), hetero.GetIdx())
    for arm in [n.GetIdx() for n in hetero.GetNeighbors() if n.GetIdx() != alpha.GetIdx()]:
        rw.RemoveBond(hetero.GetIdx(), arm)
    parts = Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
    home = {atom: k for k, part in enumerate(parts) for atom in part}
    arms = [n.GetIdx() for n in hetero.GetNeighbors() if n.GetIdx() != alpha.GetIdx()]
    if home[alpha.GetIdx()] in {home[a] for a in arms} or len({home[a] for a in arms}) != len(arms):
        return ''  # a group that closes a ring back to the acetate or to the atom X
    prefix = _acid_prefix_on_atom(mol, hetero, arms, parts, home)
    if not prefix:
        return ''
    from ..assembly.naming_utils import apply_enclosing_marks
    for text in ((prefix,) if _is_enclosed(prefix) else (apply_enclosing_marks(prefix, -1), prefix)):
        name = text + 'acetate'
        if _anion_name_keeps_hydron_sites(mol, name):
            return name
    return ''


def _is_enclosed(text: str) -> bool:
    """``text`` is wrapped by one pair of enclosing marks: its first mark closes at its last character."""
    closers = {'(': ')', '[': ']', '{': '}'}
    if not text or text[0] not in closers or text[-1] != closers[text[0]]:
        return False
    depth = 0
    for k, ch in enumerate(text):
        depth += ch in closers
        depth -= ch in closers.values()
        if depth == 0:
            return k == len(text) - 1
    return False


def _acid_prefix_on_atom(mol, hetero, arms, parts, home) -> str:
    """The prefix of the atom X of 'X-CH2-COO(-)' with its groups: 'carboxymethoxy' (O), '(carboxymethyl)sulfanyl' (S),
    '(carboxymethyl)amino' and 'bis(carboxymethyl)amino' (N: the groups are cited as those of a mononuclear centre,
    , the Blue Book, the first without enclosing marks, a repeated one with the multiplier), or ''."""
    from ..assembly.prefix_marks import mononuclear_substituent_block
    from .group_prefixes import combinations, group_prefix_candidates
    if hetero.GetSymbol() != 'N':
        from ..assembly.substituent_naming import name_substituent_fragment
        alpha = next(n.GetIdx() for n in hetero.GetNeighbors() if n.GetIdx() not in arms)
        rw = Chem.RWMol(mol)
        rw.RemoveBond(alpha, hetero.GetIdx())
        group = next(part for part in Chem.GetMolFrags(rw.GetMol(), asMols=False, sanitizeFrags=False)
                     if hetero.GetIdx() in part)
        return name_substituent_fragment(mol, sorted(group), hetero.GetIdx(), [alpha]) or ''
    candidates = []
    for arm in arms:
        # each group of the nitrogen is named on its own (its candidates are substitutive names: 'carboxymethyl'); the
        # retained residue name 'glycinyl' that the namer gives to the whole group '-NH-CH2-COOH' is never asked for here
        names = group_prefix_candidates(mol, list(parts[home[arm]]), arm, 'yl', [hetero.GetIdx()], ligand_first=False)
        if not names:
            return ''
        candidates.append(names)
    for choice in combinations(candidates):
        block = mononuclear_substituent_block(list(choice), max_count=8)
        if block is not None:
            return block + 'amino'
    return ''
