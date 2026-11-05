"""Detachable prefixes of an acyclic chain substituent that hang on a chain carbon by a multiple bond, and the
guards of the alkoxy prefix of a decorated chain (lane/l17 item 3).

``assembly.substituent_naming`` names a chain substituent from its bonds: it enumerates the groups on the chain,
numbers the chain from the free valence and cites each group as a prefix,. Three of its decisions
live here, each a function of the molecule and the atoms handed in:

*:func:`consume_substituted_imino` -- a substituted imino group =N-X on a chain carbon;
*:func:`unsaturated_chain_plan` -- the principal chain of a branched unsaturated substituent, and which C=C are
  prefixes of an off-chain atom (d),;
*:func:`alkoxy_decoration_tokens_are_real` -- whether the 'hydroxy' / 'oxo' prefixes of the carbon side of an R-oxy
  group can only be spelled for -OH / =O groups that fragment has;
*:func:`consume_azido` -- the pendant azido group, moved from the producer unchanged.
"""
from __future__ import annotations

from rdkit import Chem


def alkoxy_decoration_tokens_are_real(mol, rest, attach) -> bool:
    """True when the 'hydroxy' / 'oxo' prefixes of the name of ``rest`` (the carbon side of an -O-R group,
    ``attach`` its free-valence carbon) can only be spelled for -OH / =O groups the fragment has.

     (the Blue Book, "Alkoxy and aryloxy groups"): an R-oxy prefix is the name of R with 'oxy' for its
    free valence, so a decorated R keeps its own detachable prefixes: 'bis(2-hydroxyethoxy)acetic acid (PIN)'
    (the Blue Book). Substituent R is named by the recursion of ``name_substituent_fragment``; its
    polyfunctional chain producer (``_name_polyfunctional_acyclic_substituent``) writes '2-hydroxyethyl' /
    '2-oxobutyl' from the bonds themselves, one prefix per real terminal group. What the recursion cannot yet
    express is an oxygen INSIDE R (an ether, ester or peroxide link), which it spells as a spurious 'hydroxy' / 'oxo'.
    So the tokens are real when (1) every oxygen of ``rest`` is terminal (one heavy neighbour, uncharged, no
    isotope), (2) no carbon carries both a terminal -OH and a terminal =O (that is a carboxyl carbon, 'carboxy',
    , (3) ``attach`` bears no double-bonded heteroatom (-O-C(=O)R is an acyloxy group, whose PIN prefix
    cites the acyl 'R-oyl',, never '1-oxo...oxy'), and (4) no atom of the molecule's principal
    characteristic group is in ``rest``: such a group is cited as the suffix of the parent,, never
    as a prefix of an R-oxy group. Fails closed (False) when perception fails."""
    rest_set = set(rest)
    attach_atom = mol.GetAtomWithIdx(attach)
    for bond in attach_atom.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(attach_atom).GetSymbol() != 'C':
            return False
    for i in rest_set:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'O':
            continue
        if (atom.GetDegree() != 1 or atom.GetFormalCharge() != 0
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()):
            return False
    for i in rest_set:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() != 'C':
            continue
        terminal = [mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondType()
                    for nb in atom.GetNeighbors()
                    if nb.GetSymbol() == 'O' and nb.GetIdx() in rest_set]
        if Chem.BondType.DOUBLE in terminal and Chem.BondType.SINGLE in terminal:
            return False
    principal = _principal_group_atoms(mol)
    return principal is not None and not (principal & rest_set)


def _principal_group_atoms(mol):
    """The atoms of every match of the molecule's principal characteristic group (an empty set when it has none),
    or None when perception fails. The principal group is a function of the molecule alone, and the alkoxy namer asks
    for it once per R-oxy group of the molecule, so the answer is kept on the Mol object (the way
    ``rules.oxoacid_group_prefix.group_shape_ok`` keeps its shape cache); an RWMol, which can still change, is not
    cached."""
    cached = getattr(mol, '_decoration_principal_atoms', None)
    if cached is not None:
        return cached
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        pg, pg_matches = get_principal_group(mol, detect_functional_groups(mol))
        atoms = frozenset(i for match in (pg_matches or ()) for i in match) if pg else frozenset()
    except Exception:  # noqa: BLE001 -- no perception, no exemption
        return None
    if not isinstance(mol, Chem.RWMol):
        try:
            mol._decoration_principal_atoms = atoms
        except AttributeError:  # pragma: no cover -- a Mol that takes no attribute is perceived each time
            pass
    return atoms


def unsaturated_chain_plan(mol, longest, attach_idx, core_double_bonds):
    """The principal chains of a branched unsaturated substituent and the C=C each carries, or None.

    ``longest`` is every longest carbon chain from the free valence (b), the Blue Book, "THE
    PRINCIPAL SUBSTITUENT CHAIN"); ``core_double_bonds`` the C=C of the backbone as frozensets of atom indices.
    Criterion (d) of: "The principal chain has the greater number of multiple bonds regardless of
    type", so the chains kept are those with the most C=C on the chain. A C=C that a kept chain does not carry is a
    prefix of its off-chain atom with the free valence of its bond: '2-methylidene-4-oxopentyl'); it is
    admitted only when one of its atoms is on the chain and its geometry is undefined (a geometry has no locant to
    be cited at in such a prefix).

    Returns ``[(chain, C=C on the chain),...]`` for the kept chains; ```` when no longest chain carries a C=C
    (every C=C is such a prefix, and the caller names the substituent as a saturated one, criteria (k)-(m)); and
    None for a plan this producer does not build (a chain not starting at the free valence, a C=C with a geometry
    when no chain carries one, a C=C that is not adjacent in its chain)."""
    if any(not chain or chain[0] != attach_idx for chain in longest):
        return None

    def _defined_geometry(pair):
        return mol.GetBondBetweenAtoms(*tuple(pair)).GetStereo() != Chem.BondStereo.STEREONONE

    most = max(sum(1 for pair in core_double_bonds if pair <= set(chain)) for chain in longest)
    if most == 0:
        return None if any(_defined_geometry(pair) for pair in core_double_bonds) else []
    plan = []
    for chain in longest:
        chain_double_bonds = {pair for pair in core_double_bonds if pair <= set(chain)}
        if len(chain_double_bonds) != most:
            continue
        off_chain = core_double_bonds - chain_double_bonds
        if any(len(pair & set(chain)) != 1 or _defined_geometry(pair) for pair in off_chain):
            continue
        if all(abs(chain.index(a) - chain.index(b)) == 1 for a, b in (tuple(p) for p in chain_double_bonds)):
            plan.append((chain, chain_double_bonds))
    return plan or None  #  is the answer for a chain-free C=C only: a chain that carries one and is not admitted fails


def consume_substituted_imino(mol, sub_set, attach_idx, consumed, add_prefix) -> bool:
    """ / /: a SUBSTITUTED imino group =N-X (an oxime =N-OH, an oxime ether =N-OR, an
    N-alkyl or N-aryl imine =N-R, a hydrazone =N-NR2) on a carbon of the chain that is not its free valence is the
    detachable prefix '(Ximino)' / 'hydrazinylidene', consumed here with its N-branch (``consumed`` grows, the prefix
    goes to ``add_prefix(host carbon, text)``). Under the heading " Oximes" (the Blue Book) the
    last sentence of its paragraph (:38460) reads: "In the presence of a characteristic group having priority for
    citation as a suffix, in substitutive nomenclature oximes are designated by the prefix 'hydroxyimino'";
    '3-(hydroxyimino)butanal (PIN)' (:38487), '4-[(ethoxyimino)methyl]benzene-1-sulfonic acid (PIN)' (:38485);
    '3-(methylimino)butan-2-ylidene (PIN)' (:43352). The token is the one
    ``name_substituent`` builds for the branch, which reads the double bond of its attachment
    (``_nitrogen_ylidene_prefix``); the ring parents use the same namer. Without this the N reached the producer's
    imino branch, which takes =NH only, and the whole chain declined ('2-(hydroxyimino)butyl').

    Skipped (the N is left for the producer, which then declines) when the carbon is the free valence -- R-(C=NR')- is
    an imidoyl acyl group, -- or bears another heteroatom (an imidic acid or amidine carbon, named as an
    acyl group), when the C=N carries an E/Z geometry (it has no locant to be cited at here), or when the branch is
    not a pendant one. Returns False when the same compound prefix would repeat (it needs 'bis',, which the
    producer's simple multiplier cannot spell): the caller then declines the chain."""
    try:
        from ..errors import is_refusal_sentinel
        from .naming_utils import enclose_if_compound
        from .substituent_enumerator import name_substituent
    except Exception:  # noqa: BLE001 -- no namer, no pass
        return True
    ring_info = mol.GetRingInfo()
    seen_tokens: dict = {}
    for idx in sorted(sub_set):
        if idx in consumed or idx == attach_idx:
            continue
        a = mol.GetAtomWithIdx(idx)
        if (a.GetSymbol() != 'N' or a.GetFormalCharge() != 0 or a.GetIsotope()
                or a.GetNumRadicalElectrons() or ring_info.NumAtomRings(idx) > 0):
            continue
        double = [b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE]
        if len(double) != 1 or any(
                b.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE) for b in a.GetBonds()):
            continue
        host = double[0].GetOtherAtomIdx(idx)
        host_atom = mol.GetAtomWithIdx(host)
        if (host not in sub_set or host in consumed or host == attach_idx
                or host_atom.GetSymbol() != 'C' or host_atom.GetFormalCharge() != 0
                or ring_info.NumAtomRings(host) > 0
                or double[0].GetStereo() != Chem.BondStereo.STEREONONE):
            continue
        if any(nb.GetSymbol() != 'C' for nb in host_atom.GetNeighbors() if nb.GetIdx() != idx):
            continue
        seeds = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIdx() != host]
        if not seeds or any(sd not in sub_set or sd in consumed for sd in seeds):
            continue
        branch = {idx}
        stack = list(seeds)
        bridged = False
        while stack:
            cur = stack.pop()
            if cur in branch:
                continue
            if cur in (host, attach_idx) or cur not in sub_set or cur in consumed:
                bridged = True
                break
            branch.add(cur)
            stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors() if nb.GetIdx() not in branch)
        if bridged:
            continue
        token = name_substituent(mol, sorted(branch), idx)
        if (not token or is_refusal_sentinel(token) or ' ' in token
                or not token.endswith(('imino', 'hydrazinylidene'))):
            continue
        seen_tokens[token] = seen_tokens.get(token, 0) + 1
        consumed |= branch
        add_prefix(host, enclose_if_compound(token))
    return not any(count >= 2 for count in seen_tokens.values())


def consume_azido(mol, sub_set, consumed, add_prefix, azido_prefix) -> None:
    """ /: a pendant AZIDO group on a backbone carbon (-N=[N+]=[N-], or its -[N-]-[N+]#N resonance
    spelling) is the detachable 'azido' prefix -- the same charge-separated, net-neutral shape as nitro, with the
    same consequence when it is not consumed before the backbone is read: the producer's per-atom charge decline
    refuses the fragment and the recursive namer's string surgery turns the capped '1-azidopropane' into the
    wrong-end '1-azidopropyl' for -CH2CH2CH2N3 (TRIAGE g7 C14; the free valence takes locant 1, /
    -> '3-azidopropyl'). Structural match only: Na (neutral or -1, bonded to the unconsumed host carbon by a single
    bond and to Nb only), Nb (+1, bonded to Na and Nc only), Nc (terminal, charge making the three sum to 0) with
    the bond orders of one of the two resonance forms."""
    for _naidx in list(sub_set):
        if _naidx in consumed:
            continue
        _na = mol.GetAtomWithIdx(_naidx)
        if _na.GetSymbol() != 'N' or _na.GetDegree() != 2 or _na.IsInRing():
            continue
        _host_c = _nb_atom = None
        for _nb in _na.GetNeighbors():
            _bond = mol.GetBondBetweenAtoms(_naidx, _nb.GetIdx())
            if (_nb.GetSymbol() == 'C' and _nb.GetIdx() in sub_set
                    and _bond.GetBondType() == Chem.BondType.SINGLE):
                _host_c = _nb.GetIdx()
            elif _nb.GetSymbol() == 'N' and _nb.GetIdx() in sub_set:
                _nb_atom = _nb
        if _host_c is None or _nb_atom is None or _host_c in consumed:
            continue
        if _nb_atom.GetFormalCharge() != 1 or _nb_atom.GetDegree() != 2:
            continue
        _nc_atom = next((x for x in _nb_atom.GetNeighbors()
                         if x.GetIdx() != _naidx), None)
        if (_nc_atom is None or _nc_atom.GetSymbol() != 'N'
                or _nc_atom.GetDegree() != 1 or _nc_atom.GetIdx() not in sub_set):
            continue
        _b_ab = mol.GetBondBetweenAtoms(_naidx, _nb_atom.GetIdx()).GetBondType()
        _b_bc = mol.GetBondBetweenAtoms(_nb_atom.GetIdx(), _nc_atom.GetIdx()).GetBondType()
        _form_1 = (_na.GetFormalCharge() == 0 and _nc_atom.GetFormalCharge() == -1
                   and _b_ab == Chem.BondType.DOUBLE and _b_bc == Chem.BondType.DOUBLE)
        _form_2 = (_na.GetFormalCharge() == -1 and _nc_atom.GetFormalCharge() == 0
                   and _b_ab == Chem.BondType.SINGLE and _b_bc == Chem.BondType.TRIPLE)
        if not (_form_1 or _form_2):
            continue
        if any(mol.GetAtomWithIdx(_x).GetTotalNumHs() for _x in
               (_naidx, _nb_atom.GetIdx(), _nc_atom.GetIdx())):
            continue
        consumed.update((_naidx, _nb_atom.GetIdx(), _nc_atom.GetIdx()))
        add_prefix(_host_c, azido_prefix)


def hand_on_numbering(pos_out, chain, name):
    """``name``, the name of a substituent on ``chain`` (free valence = chain[0] = locant 1), with the numbering handed to the
    caller in ``pos_out`` (atom index -> locant): a stereocentre of the chain is cited at its locant "NAMING OF
    STEREOISOMERS": "When they relate to substituent groups, they are cited at the front of the corresponding prefix. They are
    preceded by a numerical or letter locant to describe the position of the stereogenic unit when such locants are present"),
    which the located stereo block of the caller needs; without it every chain stereodescriptor of the substituent is dropped
    and the whole name loses its stereo."""
    if pos_out is not None:
        pos_out.update({idx: i + 1 for i, idx in enumerate(chain)})
    return name
