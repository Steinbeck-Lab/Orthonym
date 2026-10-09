"""The zero-bridge fusion reading of a ring system WITH ring heteroatoms (lane N5b step 2).

 (the Blue Book, "Five-membered ring requirement"): "Fusion nomenclature gives
preferred IUPAC names only to compounds having at least two rings of at least five or more
members. [...] When fusion names are not allowed, unsaturated von Baeyer ring system names are
preferred IUPAC names", so a fused ring system with a ring heteroatom is never named by a von
Baeyer descriptor either. ``selection.fused_split`` used to return None for any ring atom that is
not carbon; it now declines only a ring heteroatom with a charge or a nonstandard bonding number
 :2744,:2756). Every name below is read back by a fresh OPSIN call to the
input's full InChIKey."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build_fused, build_fused_substituent, selection
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate

# (smiles, name, rule)
NAMED = [
    # indicated hydrogen on a ring heteroatom, (:3557), (:24635)
    ("C1Cc2ccccc2N1", "2,3-dihydro-1H-indole", "NH takes the hydro pair and the 1H"),
    ("CN1CCc2ccccc21", "1-methyl-2,3-dihydro-1H-indole", "N-substituent on the indicated-hydrogen atom"),
    ("C1CCC2NCCC2C1", "octahydro-1H-indole", "total hydrogenation, locants omitted"),
    ("c1ccc2OCOc2c1", "2H-1,3-benzodioxole", "O is not eligible: the CH2 carries the 2H (:14624)"),
    ("C1=Cc2ccccc2OC1", "2H-1-benzopyran", "2H, not 4H: lowest locant"),
    ("C1CCc2ccccc2O1", "3,4-dihydro-2H-1-benzopyran", "P-31.2 hydro prefixes, :24673"),
    ("C1CCc2c(C1)[nH]c1ccccc21", "2,3,4,9-tetrahydro-1H-carbazole", "retained carbazole parent"),
    # added hydrogen and pseudoketones next to a ring heteroatom, (:24687), (:32084)
    ("O=C1C=Cc2ccccc2O1", "2H-1-benzopyran-2-one", "lactone: method (1) gives the PIN"),
    ("O=C1CCc2ccccc2O1", "3,4-dihydro-2H-1-benzopyran-2-one", "lactone, hydro prefixes"),
    ("O=C1CCOc2ccccc12", "2,3-dihydro-4H-1-benzopyran-4-one", "ketone on the ring"),
    ("O=C1C=COc2ccccc12", "4H-1-benzopyran-4-one", "ketone, no hydro prefix"),
    ("O=C1CCc2ccccc2N1", "3,4-dihydroquinolin-2(1H)-one", "lactam, added hydrogen (:28334)"),
    ("O=C1Cc2ccccc2N1", "1,3-dihydro-2H-indol-2-one", "lactam in a five-membered ring"),
    ("O=C1CN=C(c2ccccc2)c2ccccc2N1", "5-phenyl-1,3-dihydro-2H-1,4-benzodiazepin-2-one", "two N, one NH"),
    ("O=C1CCC2CCCN12", "hexahydro-3H-pyrrolizin-3-one", "bridgehead N, ketone next to it"),
    ("OC1CCC2CCCCN2C1", "octahydro-2H-quinolizin-3-ol", "bridgehead N, suffix"),
    ("C1CC2CCCN2C1", "hexahydro-1H-pyrrolizine", "5-5 with a fusion nitrogen"),
    ("C1CC2CCOC2O1", "hexahydrofuro[2,3-b]furan", "two-component parent from the fusion tables"),
    ("C1COC2OCCC2O1", "hexahydrofuro[2,3-b][1,4]dioxine", "two-component parent, hetero partner"),
    ("OC12CCOC1COC2", "tetrahydrofuro[3,4-b]furan-3a(4H)-ol", "suffix on a fusion carbon"),
    ("O=C1OC=Cc2ccccc12", "1H-2-benzopyran-1-one", "2-benzopyran lactone"),
    ("O=C1OCc2ccccc12", "2-benzofuran-1(3H)-one", "five-membered lactone"),
    ("C1CCC2=C(C1)OC1=C2CCCC1", "1,2,3,4,6,7,8,9-octahydrodibenzo[b,d]furan", "retained dibenzofuran parent"),
]


@pytest.mark.parametrize("smiles,expected,why", NAMED)
def test_the_builder_names_a_hetero_fused_system(smiles, expected, why):
    got = build_fused(Chem.MolFromSmiles(smiles))
    assert got is not None and got[0] == expected, (why, got)
    assert_full_rt(expected, smiles)


# declines: a name here would be wrong, non-PIN, or beyond the two-component parents
DECLINED = [
    ("OC(=O)C1CS(=O)c2ccccc21", "ring S(=O): nonstandard bonding number, P-14.1.3 (:2756)"),
    ("OC(=O)C1CS(=O)(=O)c2ccccc21", "ring S(=O)(=O), the lambda convention is not spelled"),
    ("OC(=O)C1CP(=O)c2ccccc21", "ring P(=O)"),
    ("C[n+]1ccc2ccccc2c1", "ring cation"),
    ("OC(=O)c1ccc2cc[n+]([O-])cc2c1", "ring N-oxide (charged atoms)"),
    ("Cc1cc[c]([Bi]2([Cl])([c]3ccc(C)cc3)[O]C(C(F)(F)F)(C(F)(F)F)c3cccc[c]32)cc1",
     "ring Bi with bonding number 5: the book's name carries 2,1lambda5"),
    ("O=C1OC2=C(C=CC=C2)C(=O)c2ccccc12", "three components, no parent from the tables"),
    ("C1CCC2(C1)OCCc1ccccc12", "spiro"),
]


@pytest.mark.parametrize("smiles,why", DECLINED)
def test_the_builder_declines(smiles, why):
    assert build_fused(Chem.MolFromSmiles(smiles)) is None, why


def test_standard_bonding_is_the_only_new_guard():
    """fused_split took every non-carbon ring atom out; the bonding check puts back exactly the
    nonstandard ones: 3 for N, P, As, B; 2 for O, S, Se, Te)."""
    ok = Chem.MolFromSmiles("c1ccc2sccc2c1")
    bad = Chem.MolFromSmiles("O=S1CCc2ccccc21")
    cation = Chem.MolFromSmiles("c1ccc2[nH+]cccc2c1")
    assert selection._standard_bonding(ok, set(range(ok.GetNumAtoms())))
    assert not selection._standard_bonding(bad, {a.GetIdx() for a in bad.GetAtoms() if a.IsInRing()})
    assert not selection._standard_bonding(cation, {a.GetIdx() for a in cation.GetAtoms() if a.IsInRing()})


def test_the_bare_ring_as_a_prefix():
    sub = Chem.MolFromSmiles("C1Cc2ccccc2O1")           # 2,3-dihydro-1-benzofuran, free valence on C2
    name = build_fused_substituent(sub, 0)
    assert name == "2,3-dihydro-1-benzofuran-2-yl"
