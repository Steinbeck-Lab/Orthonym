"""Ring-attached acid suffixes of benzene that functional replacement derives from -C(=X)-OH, and the
thioacyl prefix of a ring branch.

``rules/benzene.py`` names a benzene ring through the suffix of its senior group. The carboxylic acid
is replaced by its imidic acid, -C(=NH)-OH, and by the acids whose -OH or =O is replaced by S, Se or Te.
These suffixes follow one pattern -- a SMARTS that matches (C,...) and a fixed suffix -- so they are one
table here, read by ``benzene._identify_suffix_fg_on_benzene``.

 "Substitutive nomenclature, suffix mode" (the Blue Book): 'benzenecarboximidic acid (PIN)'
(:30000); "Functional replacement in systematic names of carboxylic acids" (:30208):
'propanimidothioic acid (PIN)' (:30267), 'butanehydrazonoselenoic acid (PIN)' (:30269); "General
methodology of functional replacement" (:18472), Table 4.3 (:18499): '-C(=NH)-SH carboximidothioic acid'
(:18537), '-C(=NH)-SeH carboximidoselenoic acid' (:18538), '-CS-SH carbodithioic acid' (:18522-:18523);
 "Functional replacement in retained names of carboxylic acids" (:30277): 'benzene-1,2-dicarbodithioic
acid (PIN) (not tetrathiophthalic acid)' (:30313).
"""
from rdkit import Chem

from . import hydroximic_acids as _hydroximic

#: (functional group, suffix), in the order the groups are looked for. =[NX2H1] (an unsubstituted imino N,
#: as the perception FG requires) keeps an N-substituted imidate or imine out, and the -XH keeps the
#: imidate ester (O-alkyl) and the amidine out; the =N- and the -XH make every one of these disjoint from
#: the groups looked for before and after them.
ACID_SUFFIX_FGS = (
    ('imidic_acid', 'carboximidic acid'),
    ('imidothioic_acid', 'carboximidothioic acid'),
    ('imidoselenoic_acid', 'carboximidoselenoic acid'),
    ('imidotelluroic_acid', 'carboximidotelluroic acid'),
    ('hydrazonothioic_acid', 'carbohydrazonothioic acid'),
    ('hydrazonoselenoic_acid', 'carbohydrazonoselenoic acid'),
    ('hydrazonotelluroic_acid', 'carbohydrazonotelluroic acid'),
    ('dithio_acid', 'carbodithioic acid'),
    # the N-hydroxy derivatives, the Blue Book), after the plain acids: the =N-OH of
    # their SMARTS is not the =NH of those
    *_hydroximic.ACID_SUFFIX_FGS,
)

ACID_SUFFIX_SMARTS = {
    'imidic_acid': Chem.MolFromSmarts('[CX3](=[NX2H1])[OX2H1]'),
    'imidothioic_acid': Chem.MolFromSmarts('[CX3](=[NX2H1])[SX2H1]'),
    'imidoselenoic_acid': Chem.MolFromSmarts('[CX3](=[NX2H1])[SeX2H1]'),
    'imidotelluroic_acid': Chem.MolFromSmarts('[CX3](=[NX2H1])[TeX2H1]'),
    'hydrazonothioic_acid': Chem.MolFromSmarts('[CX3](=[NX2][NX3H2])[SX2H1]'),
    'hydrazonoselenoic_acid': Chem.MolFromSmarts('[CX3](=[NX2][NX3H2])[SeX2H1]'),
    'hydrazonotelluroic_acid': Chem.MolFromSmarts('[CX3](=[NX2][NX3H2])[TeX2H1]'),
    # -C(=S)-SH has no tautomer, so no italic letter locant
    'dithio_acid': Chem.MolFromSmarts('[CX3](=[SX1])[SX2H1]'),
    **_hydroximic.ACID_SUFFIX_SMARTS,
}


def identify_chalcogen_acyl_group(mol, start_idx, ring_atoms):
    """A ring-attached -C(=X)-R branch (X = S, Se, Te; R a carbon) as its acyl prefix,
    (the Blue Book): the acid R-C(=X)-OH is named by the strict PIN path and verified by OPSIN, and its
    '-ic O-acid' becomes '-yl' ('ethanethioic O-acid' -> 'ethanethioyl'). The prefix is always enclosed,
     (:7356) "Parentheses are used to clarify a modification made by functional replacement
    prefixes" ('4-(ethanethioyl)benzoic acid (PIN)',:7358), so it is its own locant scope and the
    acid is named as a whole molecule: 'sulfanyl(sulfanylidene)ethanethioyl (preferred prefix)' (:30604),
    '2-[amino(sulfanylidene)ethanethioyl]benzoic acid (PIN)' (:33215). The 'atoms' cover the whole branch.
    None = fail closed."""
    from .benzene import _bfs_substituent_atoms

    start_atom = mol.GetAtomWithIdx(start_idx)
    if (start_atom.GetSymbol() != 'C' or start_atom.IsInRing()
            or start_atom.GetFormalCharge() != 0):
        return None
    x_nbrs = [nb.GetIdx() for nb in start_atom.GetNeighbors()
              if nb.GetSymbol() in ('S', 'Se', 'Te')
              and mol.GetBondBetweenAtoms(
                  start_idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0]
    ring_nbrs = [nb.GetIdx() for nb in start_atom.GetNeighbors() if nb.GetIdx() in ring_atoms]
    r_nbrs = [nb for nb in start_atom.GetNeighbors()
              if nb.GetIdx() not in ring_atoms and nb.GetIdx() not in x_nbrs]
    if (len(x_nbrs) != 1 or len(ring_nbrs) != 1 or len(r_nbrs) != 1
            or r_nbrs[0].GetSymbol() != 'C'):
        return None
    sub_atoms = set(_bfs_substituent_atoms(mol, start_idx, ring_atoms))
    if any(mol.GetAtomWithIdx(a).IsInRing() for a in sub_atoms):
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
           for a in mol.GetAtoms() if a.GetIdx() in sub_atoms):
        return None
    try:
        rw = Chem.RWMol(mol)
        oh = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(start_idx, oh, Chem.BondType.SINGLE)
        acid_smiles = Chem.MolFragmentToSmiles(rw, sorted(sub_atoms | {oh}), canonical=True)
        acid = Chem.MolFromSmiles(acid_smiles)
        if acid is None:
            return None
        acid_smiles = Chem.MolToSmiles(acid)
    except Exception:
        return None
    from ..errors import is_failure_name
    from ..namer import name_compound
    from ..assembly.substituent_naming import (
        acid_name_to_acyl_prefix,
        fragment_acid_name_verified,
    )
    from ..assembly.naming_utils import apply_enclosing_marks
    acid_name = name_compound(acid_smiles)
    if not acid_name or is_failure_name(acid_name):
        return None
    acyl = acid_name_to_acyl_prefix(acid_name)
    if not acyl or not fragment_acid_name_verified(acid_name, acid_smiles):
        return None
    return {'name': apply_enclosing_marks(acyl, -1), 'atoms': sorted(sub_atoms)}
