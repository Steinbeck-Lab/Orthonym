"""Sulfur acid groups as the suffix of a fused ring parent: 'quinoline-8-sulfonyl chloride'.

 'Substitutive nomenclature, suffix mode, for sulfonic, sulfinic, etc., acids' (the Blue Book),
 sulfonamides (:32744), acyl halides from suffix acids (:31412); seniority
(:18162). The fused ring producer had no sulfur suffix, so these rings had no name at the PIN tier
(rules.fused_sulfur_suffix).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=S(=O)(Cl)c1cccc2cccnc12", "quinoline-8-sulfonyl chloride"),
    ("NS(=O)(=O)c1cccc2cccnc12", "quinoline-8-sulfonamide"),
    ("NS(=O)(=O)c1ccc2[nH]ccc2c1", "1H-indole-5-sulfonamide"),
    ("OS(=O)(=O)c1cc2ccccc2o1", "1-benzofuran-2-sulfonic acid"),
    ("NS(=O)(=O)c1nc2ccccc2s1", "1,3-benzothiazole-2-sulfonamide"),
    ("NS(=O)(=O)c1ccc2c(c1)OCO2", "2H-1,3-benzodioxole-5-sulfonamide"),
    ("O=S(=O)(Cl)c1ccc2c(c1)OCCO2", "2,3-dihydro-1,4-benzodioxine-6-sulfonyl chloride"),
    ("O=S(=O)(Cl)c1ccc2c(c1)CCC2", "2,3-dihydro-1H-indene-5-sulfonyl chloride"),
    ("O=S(=O)(Cl)c1ccc2c(c1)CCN2C(C)=O", "1-acetyl-2,3-dihydro-1H-indole-5-sulfonyl chloride"),
    ("O=C1CCc2cc(S(=O)(=O)Cl)ccc21", "1-oxo-2,3-dihydro-1H-indene-5-sulfonyl chloride"),
    ("O=C1Nc2ccc(S(N)(=O)=O)cc2C1", "2-oxo-2,3-dihydro-1H-indole-5-sulfonamide"),
    ("OS(=O)(=O)c1cc2cc(S(=O)(=O)O)ccc2o1", "1-benzofuran-2,5-disulfonic acid"),
    ("O=S(=O)(Cl)n1ccc2ccccc21", "1H-indole-1-sulfonyl chloride"),
    ("NS(=O)(=O)c1ccc2nc(N)sc2c1", "2-amino-1,3-benzothiazole-6-sulfonamide"),
    ("O=S(=O)(O)c1ccc2c(c1)C(=O)c1ccccc1C2=O", "9,10-dioxo-9,10-dihydroanthracene-2-sulfonic acid"),
    # an N-substituted sulfonamide: the N groups are italic-N prefixes,:32744)
    ("CNS(=O)(=O)c1ccc2c(c1)CCN2", "N-methyl-2,3-dihydro-1H-indole-5-sulfonamide"),
    ("CNS(=O)(=O)c1cccc2cccnc12", "N-methylquinoline-8-sulfonamide"),
    ("O=S(=O)(Nc1ccccc1)c1ccc2[nH]ccc2c1", "N-phenyl-1H-indole-5-sulfonamide"),
    ("CN(C)S(=O)(=O)c1ccc2c(c1)OCO2", "N,N-dimethyl-2H-1,3-benzodioxole-5-sulfonamide"),
    ("CC(=O)Nc1ccc(S(=O)(=O)Nc2ccc3c(c2)OCO3)cc1", "N-{4-[(2H-1,3-benzodioxol-5-yl)sulfamoyl]phenyl}acetamide"),
    # a senior group elsewhere keeps the suffix; the sulfur group is a prefix (unchanged)
    ("OC(=O)c1ccc2cc(S(=O)(=O)O)ccc2n1", "6-sulfoquinoline-2-carboxylic acid"),
    ("NC(=O)c1ccc2cc(S(N)(=O)=O)ccc2n1", "6-sulfamoylquinoline-2-carboxamide"),
])
def test_fused_sulfur_suffix_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # a carboxamide outranks a sulfonamide, N-hydroxy and N-alkoxy amides included, the Blue Book
    # "11 Amides [in the order of the corresponding acids]";:17568 N-hydroxy amides), though the perception ranks the
    # sulfonamide first (rules.fused_sulfur_suffix.amide_rival)
    ("NS(=O)(=O)c1ccc2[nH]ccc2c1C(=O)NOC", "N-methoxy-5-sulfamoyl-1H-indole-4-carboxamide"),
    ("NS(=O)(=O)c1ccc2occc2c1C(=O)NO", "N-hydroxy-5-sulfamoyl-1-benzofuran-4-carboxamide"),
    ("NS(=O)(=O)c1ccc2c(c1)CCCC2C(=O)NOC", "N-methoxy-6-sulfamoyl-1,2,3,4-tetrahydronaphthalene-1-carboxamide"),
    ("ONC(=O)CCc1c[nH]c2ccc(S(N)(=O)=O)cc12", "N-hydroxy-3-(5-sulfamoyl-1H-indol-3-yl)propanamide"),
])
def test_carboxamide_class_keeps_the_suffix(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles", [
    "NNS(=O)(=O)c1ccc2[nH]ccc2c1",        # a sulfonohydrazide,:33932), never 'N-amino...sulfonamide'
    "CC(=O)NS(=O)(=O)c1ccc2[nH]ccc2c1",   # an N-acyl sulfonamide: the acyl amide is senior
])
def test_no_n_amino_or_n_acyl_sulfonamide_pin(smiles):
    row = tier_row(smiles, "pin")
    assert row["tier"] != "pin_verified" or "sulfonamide" not in row["name"], row
