"""The λ-convention name of a divalent or trivalent radical component: 'dimethyl-λ2-silane' for C[Si]C.

 "The λ-convention" (the Blue Book): "Divalent and trivalent radical centers in a parent hydride
formally derived by the removal of two or three hydrogen atoms from the same skeletal atom in its standard valence
state may be described by the λ-convention (see... This method is only for general nomenclature."
('dichloro-λ2-methane' beside the PIN 'dichloromethylidene', 'fluoro-λ1-methane', 'phenyl-λ1-azane'). An adduct
names each component as a compound, and OPSIN 2.9.0 reads no radical component of an adduct name in any
position, while it reads the λ name of the same structure as an ordinary component ('dimethyl-λ2-silane—water
(1/1)'). The spelling is offered only as another spelling of an adduct component (``rules.adducts``
``_component_spellings_once``), which is tried when the adduct name does not read back, and it ships only when it
reads back to the component and the whole adduct name to the input. Never a PIN.
"""
from typing import List, Optional

from rdkit import Chem

#: the mononuclear parent hydrides of the centres the λ-convention describes, Table 2.1)
_PARENT = {'C': 'methane', 'Si': 'silane', 'Ge': 'germane', 'Sn': 'stannane', 'Pb': 'plumbane', 'B': 'borane',
           'N': 'azane', 'P': 'phosphane', 'As': 'arsane', 'Sb': 'stibane', 'Bi': 'bismuthane'}
#: the prefix of a chalcogen atom double bonded to the centre
_CHALCOGEN_YLIDENE = {'O': 'oxo', 'S': 'sulfanylidene', 'Se': 'selanylidene', 'Te': 'tellanylidene'}
#: the prefix of a halogen atom on the centre
_HALO = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def lambda_component_name(frag_smi: str) -> Optional[str]:
    """'<prefixes>-λn-<parent hydride>' for a component with one acyclic radical centre the λ-convention describes
    (``lambda_convention.radical_lambda_bonding_number``) and no other unpaired electron, or None."""
    from ..assembly.composer import _assemble_decorated_amino_prefix
    from .group_prefixes import group_prefix_candidates
    from .lambda_convention import LAMBDA, radical_lambda_bonding_number
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        return None
    centres = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(centres) != 1 or centres[0].IsInRing() or centres[0].GetSymbol() not in _PARENT:
        return None
    centre = centres[0]
    n = radical_lambda_bonding_number(mol, centre.GetIdx())
    if n is None:
        return None
    names: List[str] = []
    for bond in centre.GetBonds():
        attach = bond.GetOtherAtomIdx(centre.GetIdx())
        atoms = _branch(mol, attach, centre.GetIdx())
        order = bond.GetBondType()
        if order == Chem.BondType.DOUBLE and len(atoms) == 1:
            prefix = _CHALCOGEN_YLIDENE.get(mol.GetAtomWithIdx(attach).GetSymbol())
        elif order == Chem.BondType.SINGLE and len(atoms) == 1 and mol.GetAtomWithIdx(attach).GetSymbol() in _HALO:
            prefix = _HALO[mol.GetAtomWithIdx(attach).GetSymbol()]  # 'dichloro-λ2-methane'
        elif order in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            suffix = 'yl' if order == Chem.BondType.SINGLE else 'ylidene'
            found = group_prefix_candidates(mol, atoms, attach, suffix, [centre.GetIdx()])
            prefix = found[0] if found else None
        else:
            prefix = None
        if not prefix:
            return None
        names.append(prefix)
    head = f"{LAMBDA}{n}-{_PARENT[centre.GetSymbol()]}"
    if not names:
        return head
    core = _assemble_decorated_amino_prefix([(nm, False) for nm in names], enclose=False, head="@")
    return core[:-1] + "-" + head if core and core.endswith("@") else None


def _branch(mol, start: int, centre: int) -> List[int]:
    """The atoms reached from ``start`` without passing ``centre``."""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen or cur == centre:
            continue
        seen.add(cur)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(cur).GetNeighbors())
    return sorted(seen)
