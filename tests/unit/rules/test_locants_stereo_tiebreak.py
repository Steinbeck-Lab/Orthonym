"""P-45.6.3: 'R' before 'S' orientation tie-break (Wave-2 P0c Task 5).

BB P-45.6.3 (BlueBookV2.md:22603): "When names based on alphanumerical order
and isotopic descriptors are the same, further choice depends on the
alphabetic order of the stereochemical descriptors 'R' and 'S'."
"""
import pytest
from rdkit import Chem

from orthonym.rules.locants import orient_chain, build_atom_to_locant


MESO = "C[C@H](Cl)[C@H](Cl)C"  # meso-2,3-dichlorobutane


def _chain_atoms(mol):
    # The 4 chain carbons in SMILES atom order: C0-C1-C3-C5 (Cl at 2 and 4).
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]


class TestOrientChainP4563:
    def test_meso_dichlorobutane_r_gets_low_locant(self):
        mol = Chem.MolFromSmiles(MESO)
        chain = _chain_atoms(mol)
        subs = {}
        for a in mol.GetAtoms():
            if a.GetSymbol() == "Cl":
                c = a.GetNeighbors()[0].GetIdx()
                subs.setdefault(c, []).append([a.GetIdx()])
        oriented = orient_chain(chain, mol, set(), [], [],
                                substituent_positions=subs)
        atl = build_atom_to_locant(oriented)
        from rdkit.Chem import rdCIPLabeler
        probe = Chem.Mol(mol)
        rdCIPLabeler.AssignCIPLabels(probe)
        codes = sorted(
            (atl[a.GetIdx()], a.GetProp("_CIPCode"))
            for a in probe.GetAtoms() if a.HasProp("_CIPCode")
        )
        # P-45.6.3: 'R' at the first point of difference -> locant 2 is R.
        assert codes == [(2, "R"), (3, "S")]

    def test_end_to_end_name(self):
        from orthonym.namer import name_compound
        result = name_compound(MESO)
        name = result if isinstance(result, str) else getattr(result, "name", result)
        assert "(2R,3S)" in str(name), f"got {name!r}"

    def test_achiral_chain_unchanged(self):
        # No stereocentres: criterion (f) must not perturb plain chains.
        mol = Chem.MolFromSmiles("CCCC")
        chain = [a.GetIdx() for a in mol.GetAtoms()]
        assert orient_chain(chain, mol, set(), [], []) == chain
