"""Acyl halides on a substituted ring: the acyl name of the acid, then the halide word.

 'Acid halides' (the Blue Book): "Acyl halides are named by citing the name of the corresponding acyl
group followed by the class name(s) of the halide(s)"; '2-carbonocyanidoyl-5-methylbenzoyl chloride (PIN)' (:31543);
 (:30624): 'carboxylic acid' -> 'carbonyl'. The ring producer named only the bare ring, so a substituted
benzoyl chloride fell to '1-[chloro(oxo)methyl]-4-methylbenzene' (rules.acyl_halide_from_acid).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("Cc1ccc(C(Cl)=O)cc1", "4-methylbenzoyl chloride"),
    ("Clc1ccc(C(Cl)=O)cc1", "4-chlorobenzoyl chloride"),
    ("O=C(F)c1ccccc1F", "2-fluorobenzoyl fluoride"),
    ("O=C(Cl)c1ccc([N+](=O)[O-])cc1", "4-nitrobenzoyl chloride"),
    ("NS(=O)(=O)c1ccc(C(Cl)=O)cc1", "4-sulfamoylbenzoyl chloride"),
    ("NC(=O)c1ccc(C(Cl)=O)cc1", "4-carbamoylbenzoyl chloride"),     # the acid halide is senior to the amide
    ("CC(=O)Nc1ccc(C(Cl)=O)cc1", "4-acetamidobenzoyl chloride"),
    ("O=C(Cl)c1ccc2ccccc2c1", "naphthalene-2-carbonyl chloride"),
    ("O=C(Cl)c1cccnc1", "pyridine-3-carbonyl chloride"),
    ("O=C(Cl)c1cc2ccccc2o1", "1-benzofuran-2-carbonyl chloride"),
    ("O=C(Cl)c1cc(C(=O)Cl)cc(C(Cl)=O)c1", "benzene-1,3,5-tricarbonyl trichloride"),
    # method (2) (:31522): 'chlorosulfonyl (preselected prefix)' (:36472) beside a senior group
    ("O=C(Cl)c1ccc(S(=O)(=O)Cl)cc1", "4-(chlorosulfonyl)benzoyl chloride"),
    ("OC(=O)c1ccc(S(=O)(=O)Cl)cc1", "4-(chlorosulfonyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)(=O)F)cc1", "4-(fluorosulfonyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)Cl)cc1", "4-(chlorosulfinyl)benzoic acid"),
    ("CNC(=O)c1ccc(S(=O)(=O)Cl)cc1", "4-(methylcarbamoyl)benzene-1-sulfonyl chloride"),
    # the sulfonyl chloride outranks a sulfonamide: its suffix, never 'chlorosulfonyl' on a sulfonamide parent
    ("O=S(=O)(Cl)c1ccc(S(=O)(=O)Nc2cccc3c(Cl)c[nH]c23)cc1",
     "4-[(3-chloro-1H-indol-7-yl)sulfamoyl]benzene-1-sulfonyl chloride"),
    # unchanged
    ("ClS(=O)(=O)c1ccc(S(=O)(=O)Cl)cc1", "benzene-1,4-disulfonyl dichloride"),
    ("O=C(Cl)c1ccccc1", "benzoyl chloride"),
    ("O=C(Cl)C1CCCCC1", "cyclohexanecarbonyl chloride"),
    ("O=C(Cl)c1ccc(C(Cl)=O)cc1", "benzene-1,4-dicarbonyl dichloride"),
])
def test_ring_acyl_halide_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "O=C(Cl)c1ccc(C(=O)Br)cc1",          # two halogens: not built here
    "O=C(Cl)c1ccc(CCC(Cl)=O)cc1",        # a second acyl halide off the ring: the acid suffix does not count it
])
def test_other_shapes_are_not_built_here(smiles):
    from rdkit import Chem
    from orthonym.rules.acyl_halide_from_acid import acyl_halide_from_acid
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts("[CX3](=O)[Cl,Br]")
    combined = [(m, mol.GetAtomWithIdx(m[2]).GetSymbol()) for m in mol.GetSubstructMatches(pattern)]
    assert acyl_halide_from_acid(mol, combined) is None


def test_no_chlorosulfonyl_prefix_on_a_junior_parent_at_any_tier():
    row = tier_row("O=S(=O)(Cl)c1ccc(S(=O)(=O)Nc2cccc3c(Cl)c[nH]c23)cc1", "best-effort")
    assert "sulfonamide" not in row["name"].split("-")[-1], row
