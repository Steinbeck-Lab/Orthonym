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
_RING_N_ESTER = Chem.MolFromSmarts('[N;R;X3]!@[CX3](=O)!@[OX2]!@[#6]')
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


def label_ester_with_acid_ester_alcohol(name: str, alkyl_prefix: str, fragment_smiles, parent_smiles) -> str:
    """``name``, recorded below the PIN (by its alcohol word ``alkyl_prefix``) when a ring-nitrogen ester outranks the ester
    the name is built on (:func:`senior_ring_n_ester_outranks`)."""
    if fragment_smiles and senior_ring_n_ester_outranks(parent_smiles, fragment_smiles.get("acid")):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(alkyl_prefix, _RULE, _DETAIL)
    return name
