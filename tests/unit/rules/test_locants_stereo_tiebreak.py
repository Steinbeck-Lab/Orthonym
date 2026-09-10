"""P-14.4(j) / P-45.6.3: CIP-stereodescriptor orientation tie-break.

BB P-14.4(j) (the Blue Book): "When there is a choice for lower locants
related to the presence of stereogenic centers or stereoisomers, the lower
locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry)
that are preferred to E, S, P, and s, respectively...". BB P-45.6.3
(the Blue Book) states the R-before-S citation-order half.

The orient_chain criterion (f) reuses ``cip_descriptor_rank_key`` so this
numbering tie-break cannot disagree with the citation tie-break -- and it ranks
BOTH atom stereocentres AND stereogenic double bonds, so a Z-vs-E numbering
choice is decided (the old key saw only atoms and got 'Z senior to E' backwards
by falling through to input order).
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


class TestOrientChainP1444jDoubleBonds:
    """P-14.4(j): the Z double-bond descriptor takes the lower locant.

    The whole reason criterion (f) reuses ``cip_descriptor_rank_key`` instead of
    a plain string comparison: 'Z' is senior to 'E' although 'E' < 'Z'
    alphabetically. The previous key ignored double bonds entirely, so this
    numbering choice fell through to input order.
    """

    # Constitutionally symmetric deca-2,8-diene, one bond E and one Z. Both
    # numbering directions give the same locant SET {2, 8}; only the E/Z
    # assignment differs, so P-14.4(j) decides -> Z at the lower locant.
    EZ_DIENE = r"C/C=C/CCCC/C=C\C"

    def test_z_double_bond_gets_low_locant_direct(self):
        mol = Chem.MolFromSmiles(self.EZ_DIENE)
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
        chain = [a.GetIdx() for a in mol.GetAtoms()]  # 10 sp3/sp2 carbons, in order
        # chain double bonds as orient_chain receives them
        double_bonds = []
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        oriented = orient_chain(chain, mol, set(), double_bonds, [])
        atl = build_atom_to_locant(oriented)
        # locant -> CIP code for each stereogenic double bond
        loc_code = {}
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and b.HasProp("_CIPCode"):
                loc = min(atl[b.GetBeginAtomIdx()], atl[b.GetEndAtomIdx()])
                loc_code[loc] = b.GetProp("_CIPCode")
        # P-14.4(j): Z is preferred -> it must sit at the lower locant (2).
        assert loc_code == {2: "Z", 8: "E"}, loc_code

    def test_ez_diene_end_to_end_name(self):
        from orthonym.namer import name_compound
        result = name_compound(self.EZ_DIENE)
        name = result if isinstance(result, str) else getattr(result, "name", result)
        assert str(name) == "(2Z,8E)-deca-2,8-diene", f"got {name!r}"


class TestOrientChainP1444jCombined:
    """Combined atom + double-bond stereo tie (the task-7C3 row).

    ``C/C=C\\[C@@H](O)CCC[C@@H](O)/C=C\\C`` is a meso undeca-2,9-diene-4,8-diol:
    constitution is symmetric, both double bonds are Z, and the two stereocentres
    are R and S. P-14.4(j) prefers R at the first differing locant, so the PIN is
    ``(2Z,4R,8S,9Z)`` -- R at locant 4, NOT ``(2Z,4S,8R,9Z)``.
    """

    ROW = r"C/C=C\[C@@H](O)CCC[C@@H](O)/C=C\C"

    def test_task_7c3_row_name(self):
        from orthonym.namer import name_compound
        result = name_compound(self.ROW)
        name = result if isinstance(result, str) else getattr(result, "name", result)
        assert str(name) == "(2Z,4R,8S,9Z)-undeca-2,9-diene-4,8-diol", f"got {name!r}"

    def test_deterministic(self):
        from orthonym.namer import name_compound

        def _name(s):
            r = name_compound(s)
            return r if isinstance(r, str) else getattr(r, "name", r)

        first = _name(self.ROW)
        for _ in range(3):
            assert _name(self.ROW) == first
