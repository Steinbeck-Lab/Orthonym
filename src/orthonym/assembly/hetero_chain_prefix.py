"""The prefix of a carbon-attached fragment whose capped molecule is named on a hydride of another element.

A fragment is named by capping its free valence with hydrogen and naming that molecule. Where the capped
molecule's parent is the hydride of an element senior to carbon -- "Selection of a senior parent
hydride" (the Blue Book) "The senior parent structure, whether cyclic or acyclic, has the senior atom...
N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C" -- the capped
-CH2-CH2-SiH3 is 'ethylsilane', and the '-yl' form 'ethylsilyl' puts the free valence on Si, a different
group. (:15762): "A compound substituent group consists of a simple substituent group (the parent
substituent group) to which is attached one or more simple substituent groups": the parent substituent group
carries the free valence, so a carbon attachment cannot be read off such a name. The free-valence-numbered
chain namer builds the prefix from the structure ('2-silylethyl', '2-(chlorosilyl)ethyl') when the carbon
skeleton is one unbranched chain; otherwise the fragment declines.
"""
from typing import Optional

#: Parent hydrides whose skeleton is not carbon,: a capped fragment whose molecule name ends
#: in one of these has the heteroatom as its parent.
NONCARBON_PARENT_HYDRIDE_STEMS = (
    'silane', 'germane', 'stannane', 'plumbane',
    'oxidane', 'sulfane', 'selane', 'tellane', 'azane',
    'phosphane', 'arsane', 'stibane', 'bismuthane', 'borane',
    'alumane', 'gallane', 'indigane', 'thallane',
    'iodane', 'bromane', 'chlorane', 'fluorane',
)


def unbranched_hetero_chain_prefix(mol, sub_atoms, attach_idx) -> Optional[str]:
    """The prefix of an acyclic carbon-attached fragment whose carbon atoms form ONE unbranched chain,
    numbered from the free valence,: the chain through the attachment atom is then the
    whole carbon skeleton, so its choice cannot depend on a seniority rule, and every non-carbon branch
    ('silyl', 'chlorosilyl', 'trimethylgermyl', 'chloro') is a located prefix of it: '2-silylethyl',
    '2-(chlorosilyl)ethyl'. None for a ring atom, a branched carbon skeleton, or a fragment the located
    namer declines. The prefix carries its own stereodescriptors (the located namer cites them)."""
    from .substituent_naming import _located_acyclic_alkyl_name
    try:
        sub_set = set(sub_atoms)
        ring_info = mol.GetRingInfo()
        if any(ring_info.NumAtomRings(i) > 0 for i in sub_set):
            return None
        carbons = [i for i in sub_set if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
        for i in carbons:
            c_nbrs = sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                         if n.GetIdx() in sub_set and n.GetSymbol() == 'C')
            if c_nbrs > 2:
                return None
        located = _located_acyclic_alkyl_name(mol, sorted(sub_set), attach_idx,
                                              allow_functional=True)
        if not located or not located[0]:
            return None
        return located[0]
    except Exception:  # noqa: BLE001 -- a namer bug must never crash naming
        return None


def fragment_parent_to_prefix(mol, sub_atoms, attach_idx, parent_name: str, carbon_count: int) -> Optional[str]:
    """Step 5 of ``substituent_naming._name_substituent_fragment_uncached``: the prefix of the fragment from
    the name of its capped molecule ``parent_name``; None where the fragment declines."""
    attach_symbol = mol.GetAtomWithIdx(attach_idx).GetSymbol() if attach_idx is not None else None
    if attach_symbol is not None and attach_symbol not in ('C', 'N', 'O', 'S'):
        # (the four common attach elements skip the test: each is in ``_ORGANIC_NONMETALS``)
        from ..rules.ions import _ORGANIC_NONMETALS
        if attach_symbol not in _ORGANIC_NONMETALS and not parent_name.strip().endswith(NONCARBON_PARENT_HYDRIDE_STEMS):
            # A capped molecule whose attach atom is a metal and whose name is no parent hydride of it is named as a
            # coordination compound ('methylidenealuminium', 'dimethylmercury'): a coordination name has no '-yl' form,
            # and 'methylidenealuminiumyl' reads as no structure. INTRODUCTION (the Blue Book): "those of
            # the transition elements (Groups 3 through 12) and of Groups 1 and 2, that are named additively" are that
            # case; ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS OF GROUPS 13 THROUGH 16 (:39739) names a Group
            # 13-16 element "by prefixing the appropriate substituent names to the name of a parent hydride", which the
            # stems above recognise and this decline leaves alone. Declined, the fragment is read by the later tiers
            # ('1-aluminaeth-1-en-1-yl'), which the first reading would void.
            return None
    if (attach_idx is not None and mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'C'
            and parent_name.strip().endswith(NONCARBON_PARENT_HYDRIDE_STEMS)):
        prefix = unbranched_hetero_chain_prefix(mol, sub_atoms, attach_idx)
        if prefix is not None and not _has_principal_group(mol):
            # (the Blue Book): with no principal characteristic group the senior atom decides
            # the parent, so a name that cites a hydride of Si, Ge, Sn,... as a prefix is the PIN only where
            # a principal group governs. The molecules the senior-atom parent producer names are not here;
            # the name is kept, below the PIN.
            from ..metrics.provenance import record_uncertified_pin_name
            record_uncertified_pin_name(prefix)
        return prefix
    if (attach_idx is not None and mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'P'
            and _cites_phosphorus_group_as_prefix(parent_name)):
        # the converse of the case above: capped with hydrogen, the P(=O)(OCH3)2 of a phosphonate is named on a carbon
        # skeleton ('{[methoxy(oxo)-λ5-phosphanyl]oxy}methane', the P-H form), and its '-yl' form puts the free valence
        # on that carbon, a different group,:15762: the parent substituent group carries the free valence).
        # A phosphorus attachment is read off a name whose parent is the phosphorus (a hydride, 'phosphanone') only; the
        # oxoacid-group prefix namer ('dimethoxyphosphoryl', rules.oxoacid_group_prefix) builds the rest from the structure.
        return None
    if parent_name.strip() == 'dioxidane' and _is_bare_hydroperoxy(mol, sub_atoms, attach_idx):
        # Hydroperoxides (the Blue Book): "The prefix 'hydroperoxy' is formed by concatenation to
        # describe the group –OOH as a substituent in the presence of a characteristic group having priority for
        # citation as a suffix"; the book's PINs cite it ('2-hydroperoxyethan-1-ol (PIN)',:28013;
        # '2-hydroperoxy-1-phenylethan-1-one (PIN)',:27953). The hydride prefix 'dioxidanyl' is no PIN vocabulary.
        return 'hydroperoxy'
    if amide_joined_at_nitrogen(parent_name, mol, sub_atoms, attach_idx):
        # the string converter reads '2-aminoacetamide' as a group joined at the far chain carbon
        # ('2-amino-2-oxoethyl'); joined at the amide nitrogen it is an acylamino group,
        # ('2-aminoacetamido'), which a later tier names from the structure
        return None
    from .substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
    return parent_to_prefix(parent_name, chain_length=carbon_count, attach_locant=ATTACH_LOCANT_UNKNOWN)


def _cites_phosphorus_group_as_prefix(parent_name: str) -> bool:
    """True when ``parent_name``, the name of a capped fragment, spells a phosphorus group as a prefix of another parent
    ('{[methoxy(oxo)-λ5-phosphanyl]oxy}methane', '(dimethoxyphosphoryl)ethane'): the phosphorus is not the parent of the name."""
    name = parent_name.strip()
    return ('phosphanyl' in name or 'phosphoryl' in name) and not name.endswith(NONCARBON_PARENT_HYDRIDE_STEMS)


def amide_joined_at_nitrogen(parent_name: str, mol, sub_atoms, attach_idx) -> bool:
    """True when ``parent_name``, the name of the capped fragment, is an amide and the fragment is joined through a
    nitrogen bonded to a C=O carbon of the fragment: ``parent_to_prefix`` reads the name as a group joined at the far
    chain carbon, so its prefix would be the name of another structure."""
    if not parent_name or not parent_name.strip().endswith('amide'):
        return False
    if attach_idx is None or attach_idx not in set(sub_atoms):
        return False
    atom = mol.GetAtomWithIdx(attach_idx)
    if atom.GetSymbol() != 'N':
        return False
    return any(nb.GetIdx() in set(sub_atoms) and nb.GetSymbol() == 'C'
               and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(nb).GetSymbol() == 'O' for b in nb.GetBonds())
               for nb in atom.GetNeighbors())


def _is_bare_hydroperoxy(mol, sub_atoms, attach_idx) -> bool:
    """True when the fragment is exactly the group –O–OH (two neutral, unlabelled oxygens, the outer one with one
    hydrogen), attached through the inner oxygen."""
    try:
        sub = set(sub_atoms)
        if attach_idx is None or attach_idx not in sub or len(sub) != 2:
            return False
        inner = mol.GetAtomWithIdx(attach_idx)
        (outer_idx,) = sub - {attach_idx}
        outer = mol.GetAtomWithIdx(outer_idx)
        return (inner.GetSymbol() == outer.GetSymbol() == 'O' and outer.GetTotalNumHs() == 1
                and not any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons() for a in (inner, outer))
                and mol.GetBondBetweenAtoms(attach_idx, outer_idx) is not None)
    except Exception:  # noqa: BLE001 -- an unreadable fragment keeps the hydride prefix
        return False


def _has_principal_group(mol) -> bool:
    """True when ``mol`` has a principal characteristic group, Table 4.1), which decides the parent
    before the senior atom does, the Blue Book)."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        return get_principal_group(mol, detect_functional_groups(mol))[0] is not None
    except Exception:  # noqa: BLE001 -- an unreadable molecule has none that is known
        return False


def carbon_component(mol, sub_set, start: int) -> set:
    """The atoms of ``sub_set`` joined to ``start`` through no nitrogen (all of them when ``start`` is no carbon): a
    carboxy group behind a nitrogen (the -NH-CH2-COOH of an amide branch) is its branch's, whose own prefix names it
    ('2-[(carboxymethyl)amino]-2-oxoethyl'); one behind an ester or ether oxygen stays on the chain's carbon, as
    before that rule (the acyloxy branch of a glycerophospholipid)."""
    if mol.GetAtomWithIdx(start).GetSymbol() != 'C':
        return {i for i in sub_set if mol.GetAtomWithIdx(i).GetSymbol() == 'C'}
    seen, stack = {start}, [start]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            if nb.GetIdx() in sub_set and nb.GetIdx() not in seen and nb.GetSymbol() != 'N':
                seen.add(nb.GetIdx())
                stack.append(nb.GetIdx())
    return seen
