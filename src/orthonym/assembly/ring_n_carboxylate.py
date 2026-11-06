"""The ester of a carboxylic acid on a ring nitrogen: 'ethyl piperidine-1-carboxylate', 'tert-butyl
morpholine-4-carboxylate'.

"Carboxy groups attached to cyclic parent hydrides or heteroacyclic parent hydrides are always named by using the
suffix 'carboxylic acid'", the Blue Book: 'pyrrolidine-1-carboxylic acid (PIN)'), and "All
preferred IUPAC names for esters are named by functional class nomenclature". The carbamate namer
cannot cite a ring through the nitrogen (``composer._named_n_substituents`` declines it), so this producer names the
acid of the ester by the pipeline (``namer.name_pipeline_only``) and writes the ester of that acid: the name of the
alkyl or aryl group R, then the acid name with 'carboxylic acid' changed to 'carboxylate'. Only when the acid name has
one acid and that acid is the ring-nitrogen carboxylic acid; the name passes the read-back gates of the tier.
Kept out of ``composer.py`` so that file stays within its size budget."""
from typing import Optional

from rdkit import Chem

_ESTER = Chem.MolFromSmarts('[N;R;X3]!@[CX3](=O)!@[OX2]!@[#6]')
#: the acid of the ester must be the only acid of the molecule ('ethyl 4-(6-carboxynaphthalen-2-yl)piperidine-1-
#: carboxylate' is no PIN: the carboxylic acid is the senior group,
_CARBOXY = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
_OTHER_ACID = Chem.MolFromSmarts('[S,P,Se,B,As](=O)[OX2H1,OX1-]')


def ring_n_carboxylate_ester_name(mol, style: str = "pin") -> Optional[str]:
    """'R <ring>-N-carboxylate' for the one ring-nitrogen carboxylic ester of ``mol``, or None."""
    matches = mol.GetSubstructMatches(_ESTER)
    if len(matches) != 1:
        return None
    _n_idx, c_idx, _o_dbl, o_idx, r_idx = matches[0]
    split = Chem.RWMol(mol)
    split.RemoveBond(o_idx, r_idx)
    frags = Chem.GetMolFrags(split.GetMol(), asMols=False, sanitizeFrags=False)
    acid_atoms = next(f for f in frags if c_idx in f)
    if r_idx in acid_atoms:
        return None                       # R is joined to the acid part elsewhere: a lactone-like ring, not this shape
    r_atoms = set(next(f for f in frags if r_idx in f))
    acid = Chem.RWMol(mol)
    acid.GetAtomWithIdx(o_idx).SetNumExplicitHs(1)
    acid.GetAtomWithIdx(o_idx).SetNoImplicit(True)
    for idx in sorted(r_atoms, reverse=True):
        acid.RemoveAtom(idx)
    try:
        acid_mol = acid.GetMol()
        Chem.SanitizeMol(acid_mol)
        acid_smiles = Chem.MolToSmiles(acid_mol)
    except (RuntimeError, ValueError):
        return None
    if len(acid_mol.GetSubstructMatches(_CARBOXY)) != 1 or acid_mol.HasSubstructMatch(_OTHER_ACID):
        return None                       # another acid is senior to the ester: not this producer's name
    from ..namer import name_pipeline_only
    acid_name = name_pipeline_only(acid_smiles, style=style)
    if not acid_name or acid_name.count('acid') != 1 or not acid_name.endswith('-carboxylic acid'):
        return None
    from .composer import _name_r_group
    r_name = _name_r_group(mol, r_idx, exclude_atoms=set(acid_atoms))
    if not r_name:
        return None
    return f"{r_name} {acid_name[:-len('carboxylic acid')]}carboxylate"
