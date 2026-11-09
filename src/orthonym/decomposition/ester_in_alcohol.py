"""The ester name 'R <acid>-ate' whose alcohol part R holds a senior ester of a ring-nitrogen carboxylic acid.

The acid on a ring nitrogen is a carboxylic acid (the Blue Book, 'tetraazane-1-carboxylic acid (PIN) [not
(triazan-1-yl)carbamic acid; the carboxylic acid is senior to the carbonic acid derivative]';
'pyrrolidine-1-carboxylic acid (PIN)'), and of the esters of a polyester the senior one is the parent of the PIN:
 applies "seniority of rings, ring systems, and chains... (see through " ('phenyl
3-(benzoyloxy)benzoate (PIN) [not 3-(phenoxycarbonyl)phenyl benzoate...]'). So '1-(tert-butoxycarbonyl)piperidin-4-yl
benzoate' names the right molecule but not by its PIN, 'tert-butyl 4-(benzoyloxy)piperidine-1-carboxylate' (piperidine
is senior to benzene,. The decomposition ester assembler builds the name on the ester it cut; when a
ring-nitrogen ester of the molecule outranks it (``rules.ester_seniority`` with the ring-nitrogen acid ranked by its
ring), the name is labelled below the PIN (label only, ``metrics.provenance.record_non_pin_label``)."""
from typing import Optional

from rdkit import Chem

_CARBOXYLIC_ESTER = Chem.MolFromSmarts('[#6][CX3](=O)!@[OX2]!@[#6]')
_RING_N_ESTER = Chem.MolFromSmarts('[#7;R;X3]!@[CX3](=O)!@[OX2]!@[#6]')
_ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
_RULE = "P-65.6.3.3.4.2"
_DETAIL = "a ring-nitrogen ester of the molecule is senior to the ester the name is built on"


def _key(mol, match, ring_n: bool):
    from ..rules.ester_seniority import _alcohol_component, _ring_component, ester_seniority_key
    if not ring_n:
        return ester_seniority_key(mol, match[1:])
    n, c, _o_dbl, o, r = match
    return (_ring_component(mol, n), _alcohol_component(mol, r, o))


def senior_ring_n_ester_outranks(parent_smiles: Optional[str], acid_smiles: Optional[str]) -> bool:
    """True when ``parent_smiles`` has a ring-nitrogen ester that outranks every ester whose acid half is the acid
    fragment ``acid_smiles`` of the name (the ester the name is built on)."""
    mol = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    acid = Chem.MolFromSmiles(acid_smiles) if acid_smiles else None
    if mol is None or acid is None:
        return False
    ring_n = list(mol.GetSubstructMatches(_RING_N_ESTER))
    if not ring_n:
        return False
    acid_c = {m[0] for m in acid.GetSubstructMatches(_ACID)}
    built_on = {hit[c] for hit in mol.GetSubstructMatches(acid) for c in acid_c}
    named = [m for m in mol.GetSubstructMatches(_CARBOXYLIC_ESTER) if m[1] in built_on]
    named += [m for m in ring_n if m[1] in built_on]
    if not named or any(m[1] in built_on for m in ring_n):
        return False
    best_named = min(_key(mol, m, False) for m in named if len(m) == 5)
    return any(_key(mol, m, True) < best_named for m in ring_n)


def ring_n_ester_is_senior(mol, c_idx: int) -> bool:
    """True when the ester of the ring-nitrogen carboxylic acid whose carbonyl carbon is ``c_idx`` outranks every other
    carboxylic ester of ``mol``, by:func:`_key`): its ester is then the parent of the PIN ('tert-butyl
    4-(acetyloxy)piperidine-1-carboxylate')."""
    ring_n = [m for m in mol.GetSubstructMatches(_RING_N_ESTER) if m[1] == c_idx]
    others = [m for m in mol.GetSubstructMatches(_CARBOXYLIC_ESTER) if m[1] != c_idx]
    if len(ring_n) != 1 or not others:
        return False
    # an acid carbon on the ring system of the ring nitrogen makes one polyacid of the two: its mixed diester is
    # named on that acid, 'ethyl methyl benzene-1,3-dicarboxylate (PIN)' (the Blue Book), no claim here
    from ..perception.rings import get_containing_ring_system
    system = set(get_containing_ring_system(mol, [ring_n[0][0]]))
    if any(m[0] in system for m in others):
        return False
    try:
        mine = _key(mol, ring_n[0], True)
        return all(mine < _key(mol, m, False) for m in others)
    except Exception:  # noqa: BLE001 - not decidable: no claim
        return False


_ANY_ESTER = Chem.MolFromSmarts('[CX3](=O)[OX2][#6]')


def ring_n_diester_matches(mol) -> Optional[list]:
    """The two ester matches (carbonyl C, =O, O, organyl C) of a molecule whose carboxylic esters are one ester of a
    ring-nitrogen carboxylic acid and one ester of a carboxylic acid on the same ring system, or None. The two acids are
    one diacid, 'pyrrolidine-1,2-dicarboxylic acid' 'pyrrolidine-1-carboxylic acid (PIN)'), and the
    molecule is its mixed diester, '1-tert-butyl 2-methyl pyrrolidine-1,2-dicarboxylate',
    the Blue Book, organyl locants as in '1,1-diethyl 3-methyl butane-1,1,3-tricarboxylate (PIN)':31783)."""
    ri = mol.GetRingInfo()
    matches = [m for m in mol.GetSubstructMatches(_ANY_ESTER)
               if not any(m[0] in r and m[2] in r for r in ri.AtomRings())]
    ring_n = [m for m in mol.GetSubstructMatches(_RING_N_ESTER)]
    if len(matches) != 2 or len(ring_n) not in (1, 2) or mol.HasSubstructMatch(_ACID):
        return None
    from ..perception.rings import get_containing_ring_system
    system = set(get_containing_ring_system(mol, [ring_n[0][0]]))
    # the other ester: a C ester on that ring system, or a second ring-nitrogen ester of it ('piperazine-1,4-...')
    if not all(any(nb.GetIdx() in system for nb in mol.GetAtomWithIdx(m[0]).GetNeighbors()) for m in matches):
        return None
    return matches


def label_ester_with_acid_ester_alcohol(name: str, alkyl_prefix: str, fragment_smiles, parent_smiles) -> str:
    """``name``, recorded below the PIN (by its alcohol word ``alkyl_prefix``) when a ring-nitrogen ester outranks the ester
    the name is built on (:func:`senior_ring_n_ester_outranks`)."""
    if fragment_smiles and senior_ring_n_ester_outranks(parent_smiles, fragment_smiles.get("acid")):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(alkyl_prefix, _RULE, _DETAIL)
    return name
