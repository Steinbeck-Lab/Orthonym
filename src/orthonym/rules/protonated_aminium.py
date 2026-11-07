"""The '-aminium' name of a protonated chain amine beside a senior neutral group ('1-carboxyethan-1-aminium').

 (the Blue Book-18170) ranks '6 Cations' before '7 Acids', so a cation is the principal
characteristic group and a carboxylic acid the 'carboxy' prefix:29910, '4-carboxy-1-methylpyridin-1-ium
chloride (PIN)':29914); the cation is named from the largest neutral parent, the amine, by Table 7.4 'amine ->
aminium':41415, method (1):41431; '3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide (PIN)'
:31709). The quaternary nitrogen has this name already (``ions.name_quaternary_aminium``); this is the same mechanism
for a nitrogen that carries hydrogen (-NH3+, -NH2R+, -NHR2+): the amine group is injected as the principal group, the
neutral amine name is assembled and 'ium' added. An -NH3+ and an aryl nitrogen ('4-carboxyanilinium') are named by the
pipeline with the amine forced as the principal group (``_reentry_aminium``). A ring nitrogen, a nitrogen on two aryl
carbons or a heteroatom on the nitrogen is left to the other producers ('').
"""
from typing import Dict

from rdkit import Chem

_AMINE_FG = {3: "primary_amine", 2: "secondary_amine", 1: "tertiary_amine"}
# A neutral phosphorus oxoacid OH (the zwitterion route below; a sulfonate zwitterion is not named through it).
_P_OXOACID_OH = Chem.MolFromSmarts("[OX2H1]-[PX4]=O")


def name_protonated_aminium(mol, cation_site: Dict) -> str:
    """'1-carboxyethan-1-aminium' for a protonated chain or aryl amine, else ''."""
    from rdkit import Chem
    try:
        atom = mol.GetAtomWithIdx(cation_site['atom_idx'])
    except (RuntimeError, IndexError, KeyError, OverflowError):
        return ''
    hs = atom.GetTotalNumHs()
    nbrs = list(atom.GetNeighbors())
    if (atom.GetSymbol() != 'N' or atom.GetFormalCharge() != 1 or atom.IsInRing() or hs not in _AMINE_FG
            or len(nbrs) != 4 - hs or not nbrs
            or any(nb.GetSymbol() != 'C' for nb in nbrs)
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())):
        return ''
    n_idx = atom.GetIdx()
    if mol.HasSubstructMatch(_P_OXOACID_OH):
        # the charged router names a zwitterion by naming this cation and ionizing its acid suffix
        # ('[(1S)-1-azaniumylethyl]phosphonate'); an aminium-principal name has no acid suffix left to ionize
        return ''
    if any(_has_hetero_double_bond(nb) for nb in nbrs):
        return ''   # an N-acyl cation is an amidium ('N-methylacetamidium',:41443), no aminium
    n_aryl = sum(1 for nb in nbrs if nb.GetIsAromatic())
    if n_aryl > 1:
        return ''
    if n_aryl:      # an anilinium: the benzene writer takes the forced amine as its suffix
        return _reentry_aminium(mol, n_idx, 'aromatic_amine' if hs == 3 else _AMINE_FG[hs])
    if hs == 3:
        return _reentry_aminium(mol, n_idx, 'primary_amine')
    carbon_neighbors = sorted((nb.GetIdx() for nb in nbrs), key=lambda c: (-_branch_size(mol, n_idx, c), c))
    fg = _AMINE_FG[hs]
    try:
        from ..assembly.composer import _assemble_amine_name
        from ..errors import UnnameableSubstituentError
        from ..namer import Orthonym
        from .ions import name_aminium_cation
        from .locants import build_atom_to_locant
    except ImportError:
        return ''
    try:
        namer = Orthonym(style='pin', _disable_opsin_validity_gate=True, _principal_group_override=fg)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, canonical, canonical)
        features.functional_groups = dict(features.functional_groups)
        features.functional_groups[fg] = [tuple([n_idx] + carbon_neighbors)]
        features.functional_groups.pop('ammonium', None)
        namer._classify(features)
        chain = list(features.principal_chain or [])
        if not chain or not set(chain) & set(carbon_neighbors):
            return ''
        on_chain = [i for i, c in enumerate(chain) if c in carbon_neighbors]
        if len(on_chain) == 1 and on_chain[0] == len(chain) - 1 and len(chain) >= 2:
            chain.reverse()          # (c): the N-bearing carbon is C1 (as in name_quaternary_aminium)
            features.principal_chain = chain
            features.atom_to_locant = build_atom_to_locant(chain)
        off_chain = [c for c in carbon_neighbors if c not in set(chain)]
        features.principal_group_atoms = [tuple([n_idx] + off_chain)]
        amine_name = _assemble_amine_name(features, 'pin')
        aminium = name_aminium_cation(amine_name) if amine_name else ''
        return aminium if aminium and 'aminium' in aminium else ''
    except (RecursionError, ValueError, RuntimeError, KeyError, AttributeError, UnnameableSubstituentError):
        return ''


def _reentry_aminium(mol, n_idx: int, fg: str) -> str:
    """'1-carboxyethan-1-aminium' for an -NH3+ on a chain carbon, '4-carboxyanilinium' for a protonated aryl amine,
    else ''.

    A primary amine has no N-substituent, so ``_assemble_amine_name`` has nothing to build, and that assembler builds
    no ring parent; the neutral amine is named instead by the pipeline with the amine ``fg`` forced as the principal
    group, the retained-name and amino-acid routes left out (they name the molecule on the acid: 'alanine', 'benzoic
    acid'), and 'ium' added ('aniline -> anilinium',:41415). Declined when the molecule holds a second amine
    nitrogen: the forced suffix would then be 'diamine', which names two cationic centres."""
    from rdkit import Chem
    try:
        from .charged_router import _hydron_scope, _reenter_forced_at_pin_tier
        from .ions import name_aminium_cation
    except ImportError:
        return ''
    neutral = Chem.RWMol(mol)
    n_atom = neutral.GetAtomWithIdx(n_idx)
    n_hs = n_atom.GetTotalNumHs()
    n_atom.SetFormalCharge(0)
    n_atom.SetNumExplicitHs(n_hs - 1)
    n_atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
        neutral_smi = Chem.MolToSmiles(neutral)
    except (ValueError, RuntimeError):
        return ''
    if sum(1 for a in neutral.GetAtoms() if _is_amine_n(a)) != 1:
        return ''
    try:
        with _hydron_scope(mol):
            amine_name = _reenter_forced_at_pin_tier(neutral_smi, 'pin', fg)
    except (RecursionError, ValueError, RuntimeError, KeyError, AttributeError):
        return ''
    if (not amine_name or not amine_name.endswith(('amine', 'aniline')) or 'diamine' in amine_name
            or 'unknown' in amine_name):
        return ''
    aminium = name_aminium_cation(amine_name)
    return aminium if aminium.endswith(('aminium', 'anilinium')) else ''


def _has_hetero_double_bond(carbon) -> bool:
    """True when ``carbon`` carries a double bond to a heteroatom (C=O, C=S, C=N)."""
    from rdkit import Chem
    return any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetSymbol() != 'C'
               for b in carbon.GetBonds())


def _is_amine_n(atom) -> bool:
    """A neutral acyclic nitrogen with single bonds to aryl carbons or to chain carbons that carry no C=X (an amine
    N, not an amide, imine or ring N)."""
    from rdkit import Chem
    if atom.GetSymbol() != 'N' or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return False
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
        return False
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() != 'C':
            return False
        if nb.GetIsAromatic():
            continue
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in nb.GetBonds()):
            return False
    return True


def _branch_size(mol, n_idx: int, start: int) -> int:
    seen, stack = {n_idx}, [start]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in seen)
    return len(seen) - 1
