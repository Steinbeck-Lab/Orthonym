"""An acyl halide or pseudohalide group on a ring or a heteroatom as a prefix: 'carbonochloridoyl'.

 'Acyl halides and pseudohalides as substituent groups' (the Blue Book): "(1) by a prefix formed from
the name of the acid, for example, 'carbonochloridoyl';... (3) at the end of an acyclic carbon chain by a prefix
denoting the halide or pseudohalide group and the prefix 'oxo'... Method (1) leads to preferred IUPAC names when the
suffix '-carboxylic acid' is used to name the corresponding acid; method (3) generates preferred IUPAC names for
acyclic carbon chains." The general substituent writer spelled the group 'chloro(oxo)methyl', which has no PIN
(``assembly.acyl_halide_prefix``).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    # the book's own PIN rows
    ("N#CC(=O)c1ccc(C)cc1C(Cl)=O", "2-carbonocyanidoyl-5-methylbenzoyl chloride"),
    ("OC(=O)c1ccccc1C(Cl)=O", "2-carbonochloridoylbenzoic acid"),
    ("N#CC(=S)c1ccccc1C(Cl)=O", "2-carbonocyanidothioylbenzoyl chloride"),
    ("OC(=O)CC(=O)OC(Br)=O", "3-(carbonobromidoyloxy)-3-oxopropanoic acid"),
    # beside an ester, an acid, on a heteroatom, each halogen
    ("COC(=O)c1ccc(C(Cl)=O)cc1", "methyl 4-carbonochloridoylbenzoate"),
    ("O=C(Br)c1ccc(C(=O)OC)cc1", "methyl 4-carbonobromidoylbenzoate"),
    ("OC(=O)c1ccc(C(I)=O)cc1", "4-carbonoiodidoylbenzoic acid"),
    ("OC(=O)C1CCCCC1C(F)=O", "2-carbonofluoridoylcyclohexane-1-carboxylic acid"),
    ("OC(=O)c1ccccc1C(Br)=S", "2-carbonobromidothioylbenzoic acid"),
    ("OC(=O)c1ccc(C(Cl)=O)cn1", "5-carbonochloridoylpyridine-2-carboxylic acid"),
    ("OC(=O)CCOC(Cl)=O", "3-(carbonochloridoyloxy)propanoic acid"),
    # on a chain: method (3), unchanged
    ("COC(=O)CCC(Cl)=O", "methyl 4-chloro-4-oxobutanoate"),
    ("OC(=O)C(C)C(Cl)=O", "3-chloro-2-methyl-3-oxopropanoic acid"),
    ("O=C(Cl)CCc1ccc(C(=O)O)cc1", "4-(3-chloro-3-oxopropyl)benzoic acid"),
])
def test_acyl_halide_prefix_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "OC(=O)CCC(Cl)=O",        # on a chain carbon: method (3)
    "OC(=O)c1ccccc1C(=O)CCl",  # a chloroacetyl group, not an acyl halide
    "OC(=O)c1ccccc1C([37Cl])=O",  # a nuclide label
])
def test_other_shapes_get_no_carbono_prefix(smiles):
    from rdkit import Chem
    from orthonym.assembly.acyl_halide_prefix import carbono_acyl_prefix
    mol = Chem.MolFromSmiles(smiles)
    acyl = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.IsInRing()
            and any(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds())
            and not any(n.GetSymbol() == "O" and n.GetTotalNumHs() for n in a.GetNeighbors())]
    for carbon in acyl:
        frag = {carbon} | {n.GetIdx() for n in mol.GetAtomWithIdx(carbon).GetNeighbors()
                           if not n.IsInRing() and n.GetSymbol() != "C" or n.GetDegree() == 1}
        assert carbono_acyl_prefix(mol, frag, carbon) is None
