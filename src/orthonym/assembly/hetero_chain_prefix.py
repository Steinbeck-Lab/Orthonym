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
    from .substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
    return parent_to_prefix(parent_name, chain_length=carbon_count, attach_locant=ATTACH_LOCANT_UNKNOWN)


def _has_principal_group(mol) -> bool:
    """True when ``mol`` has a principal characteristic group, Table 4.1), which decides the parent
    before the senior atom does, the Blue Book)."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from ..rules.seniority import get_principal_group
        return get_principal_group(mol, detect_functional_groups(mol))[0] is not None
    except Exception:  # noqa: BLE001 -- an unreadable molecule has none that is known
        return False
