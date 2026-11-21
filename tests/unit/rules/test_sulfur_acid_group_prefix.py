"""A sulfur acid group as a prefix through the shared substituent namers: 'sulfamoyl', 'aminosulfinyl', 'chlorosulfonyl'.

Appendix 2 preferred prefixes: 'sulfamoyl*' (the Blue Book,, 'phenylsulfamoyl*' (:55494),
'aminosulfinyl* (not sulfinamoyl)' (:55485); 'chlorosulfonyl (preselected prefix)' (:36472). Acid
halides rank above amides, so the acyl halide is the suffix. The shared namers returned a sentinel for these
groups and the chain acid halide producer dropped them ('propanoyl chloride', rejected)
(assembly.sulfur_acid_group_prefix).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("NS(=O)(=O)CCC(Cl)=O", "3-sulfamoylpropanoyl chloride"),
    ("CNS(=O)(=O)CCC(Cl)=O", "3-(methylsulfamoyl)propanoyl chloride"),
    ("CN(C)S(=O)(=O)CCC(Cl)=O", "3-(dimethylsulfamoyl)propanoyl chloride"),
    ("NS(=O)CCC(Cl)=O", "3-(aminosulfinyl)propanoyl chloride"),
    ("O=S(=O)(Cl)CCC(Cl)=O", "3-(chlorosulfonyl)propanoyl chloride"),
    # unchanged
    ("NS(=O)(=O)CCC(=O)OC", "methyl 3-sulfamoylpropanoate"),
    ("O=C(O)CCS(=O)(=O)Nc1ccccc1", "3-(phenylsulfamoyl)propanoic acid"),
    ("OS(=O)CCC(Cl)=O", "3-chloro-3-oxopropane-1-sulfinic acid"),
])
def test_sulfur_acid_group_prefix_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,frag,attach,expected", [
    # a carboxylic acid is senior, so the sulfur group is a prefix
    ("OC(=O)CCS(=O)(=O)O", [5, 6, 7, 8], 5, "sulfo"),
    ("OC(=O)CCS(=O)O", [5, 6, 7], 5, "sulfino"),
    ("OC(=O)CCS(N)(=O)=O", [5, 6, 7, 8], 5, "sulfamoyl"),
    ("OC(=O)CCS(F)=O", [5, 6, 7], 5, "fluorosulfinyl"),
    ("OS(=O)(=O)CC", [0, 1, 2, 3], 1, None),         # the principal group itself: the suffix
    ("CS(=O)(=O)CC", [0, 1, 2, 3], 1, None),         # a sulfonyl with a carbon: another namer's case
    ("[15NH2]S(=O)(=O)CC", [0, 1, 2, 3], 1, None),  # a nuclide label
    ("NS(=O)(=O)CC", [0, 1, 2, 3], 0, None),        # attached through the nitrogen
])
def test_fragment_shapes(smiles, frag, attach, expected):
    from rdkit import Chem
    from orthonym.assembly.sulfur_acid_group_prefix import sulfur_acid_group_prefix
    assert sulfur_acid_group_prefix(Chem.MolFromSmiles(smiles), frag, attach) == expected


@pytest.mark.parametrize("smiles", [
    # the sulfonamide is the principal group, so the suffix: '4-ethoxy-N-[(pyridin-2-yl)methyl]naphthalene-1-
    # sulfonamide', never the prefix form '1-ethoxy-4-{[(pyridin-2-yl)methyl]sulfamoyl}naphthalene',
    # the Blue Book); the prefix writers decline it (assembly.acyl_halide_prefix.holds_principal_group)
    "CCOc1ccc(S(=O)(=O)NCc2ccccn2)c2ccccc12",
    "Cn1cnc(S(=O)(=O)Nc2ccc(N3CCCCC3)cc2)c1",
    "CCNS(=O)(=O)c1cc(Cl)sc1Cl",
    "COc1ccc(NS(=O)(=O)c2c(C)noc2C)c(OC)n1",
])
def test_principal_sulfonamide_is_never_a_pin_prefix(smiles):
    row = tier_row(smiles, "pin")
    assert row["tier"] != "pin_verified" or "sulfamoyl" not in row["name"], row


def test_unperceived_principal_group_is_not_a_prefix():
    # N-SO2Cl is not perceived as a principal group; it must not become '1-(chlorosulfonyl)-1H-indole' (PIN form
    # 1H-indole-1-sulfonyl chloride)
    row = tier_row("O=S(=O)(Cl)n1ccc2ccccc21", "pin")
    assert row["tier"] != "pin_verified" or "chlorosulfonyl" not in row["name"], row
