"""A component of one covalently bonded metal cites its oxidation number in an adduct name: 'oxomolybdenum(II)'.

OPSIN gives a neutral metal without an oxidation number a typical charge when the other components are ions, and
keeps a metal that cites one at that number less its bonds. The number is the sum of the bonds to ligands counted as
anions (oxo two, chloro and alkyl one; a carbonyl zero); a carbene, a silylidene or a metal-metal bond gets none
(``rules.adduct_stock_number``). An adduct name is no PIN, the Blue Book): best-effort only.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.parametrize("smiles,expected", [
    ("O=[Mo]", ("Mo", 2)),
    ("O=[Re](=O)=O", ("Re", 6)),
    ("CCC(C)N=[Nb]", ("Nb", 2)),
    ("[Cl][Ni]", ("Ni", 1)),
    ("C[Pt]Cl", ("Pt", 2)),
    ("O=C=[Fe](Cl)Cl", ("Fe", 2)),       # the carbonyl counts zero
    ("Cl[Zr](Cl)=C1CCCC1", None),         # a carbene: no number
    ("[SiH2]=[Zr]", None),                # a silylidene: no number
    ("[Fe]=[Mn]", None),                  # metal-metal
    ("O=[Mo+]", None),                    # a charged metal
    ("C(=O)=[Fe]=C=O", None),             # zero is no Roman numeral
    ("N#C[Cu]Cl", ("Cu", 2)),             # a cyanide counts as an anion, an isocyanide does not
    ("C[N+]#C[Fe](Cl)Cl", ("Fe", 2)),
    ("C12C3C4C5C1[Ti]2345(Cl)(Cl)Cl", None),  # an eta drawing: bonded neighbours
])
def test_the_oxidation_number_follows_the_ligand_rule(smiles, expected):
    from orthonym.rules.adduct_stock_number import stock_number
    assert stock_number(Chem.MolFromSmiles(smiles)) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C(=O)(N)[NH-].O=[Pt]", "carbamoylazanide—oxoplatinum(II) (1/1)"),
    ("COCC[O-].COCC[O-].COCC[O-].O=[V]", "2-methoxyethoxide—oxovanadium(II) (3/1)"),
    ("[CH2-]CCN1CCCCC1.O=[Re](=O)=O", "1-(propan-3-id-1-yl)piperidine—trioxorhenium(VI) (1/1)"),
    ("C[CH2-].C[CH2-].C[CH2-].CCC(C)N=[Nb].C[NH-]",
     "methanaminide—ethan-1-ide—[(butan-2-yl)imino]niobium(II) (1/3/1)"),
])
def test_a_covalent_metal_component_beside_ions_reads_back_as_drawn(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
def test_a_carbene_metal_component_gets_no_oxidation_number():
    r = name_best_effort("CCCCC1=C[CH-]C2=CC=CC=C21.CCCCC1=C[CH-]C2=CC=CC=C21.C1CCC(=[Zr](Cl)Cl)C1")
    assert not r["name"] or "zirconium(" not in r["name"], r
