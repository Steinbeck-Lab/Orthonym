"""Charged molecules of the PubChem left-out pool that now get a verified name (lane ION).

Every case is named at the best-effort tier and read back through OPSIN independently of the engine's own gate
(``tests.support.rt_assert``): the name must read back to the full InChIKey of the input and to the canonical
SMILES of it, which carries the charges. Each class is pinned with the rule that governs it:

* the substituted spiro ammonium, SKELETAL REPLACEMENT ('a') NOMENCLATURE FOR CATIONS (the Blue Book):
  the junction cation is the ylium of the lambda5 hydride, '5λ5-arsaspiro[4.4]nonan-5-ylium (PIN)' (:41802), the
  groups on the rings as prefixes;
* the -uide anion of a hypervalent chalcogen or phosphorus, 'ANIONS FORMED BY ADDITION OF HYDRIDE IONS'
  (:41089): 'hexafluoro-λ5-iodanuide (preselected name)' (:41112), 'difluoro(phenyl)sulfanuide (PIN)' (:41108);
* the zwitterion whose anion is the nitrogen of an amide or sulfonamide, INTRODUCTION (:42411): "anionic
  centers... become the parent structure, into which the cationic part is substituted";
* the anion of a ketene imine and of a diazene, 'Anions derived from amines and imines' (:41047,
  'butaniminide (PIN)') and;
* the anion of a phosphorus(III) acid, (:40955): the 'ous acid' ending of the acid name becomes 'ite';
* the zwitterion of a diazene, 'Ionic centers in the same parent structure' (:42415): the suffixes 'ium' and 'ide' of one parent hydride;
* the diazonium ion on a nitrogen, (:41607): the suffix 'diazonium' on the parent hydride azane;
* the azinate, / (:25969): 'ethylideneazinic acid (PIN)', 'azinic acid, H2N(O)-OH, a
  preselected name'.

The names of the classes the Blue Book gives no example of are not PINs: the tier of each is below the PIN's.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import _full_inchikey, _independent_parse, name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.roundtrip]


def _reads_back(smiles: str, name: str) -> bool:
    parsed = _independent_parse(name, smiles)
    if not parsed:
        return False
    read, want = Chem.MolFromSmiles(parsed), Chem.MolFromSmiles(smiles)
    return (read is not None and _full_inchikey(parsed) == _full_inchikey(smiles)
            and Chem.MolToSmiles(read) == Chem.MolToSmiles(want))


CASES = [
    # (smiles, name, rule)
    # -- the groups on the rings of a spiro ammonium, the ylium of the lambda5 hydride,:41802)
    ("CC1CC[N+]2(C1)CCCC2", "2-methyl-5λ5-azaspiro[4.4]nonan-5-ylium", "P-73.4"),
    ("CC1[N+]2(CCCC2)CCO1", "1-methyl-2-oxa-5λ5-azaspiro[4.4]nonan-5-ylium", "P-73.4"),
    ("CC(=O)OC1CCC[N+]12CCCCC2", "1-(acetyloxy)-5λ5-azaspiro[4.5]decan-5-ylium", "P-73.4"),
    # -- the -uide anion of a hypervalent centre,:41089)
    ("N=[S-](=O)N", "amino(imino)(oxo)-λ4-sulfanuide", "P-72.3"),
    ("O=[S-](=N)c1ccccc1", "imino(oxo)(phenyl)-λ4-sulfanuide", "P-72.3"),
    ("CN=[S-](=O)N", "amino(methylimino)(oxo)-λ4-sulfanuide", "P-72.3"),
    ("F[S-](F)(F)(F)c1ccccc1", "tetrafluoro(phenyl)-λ4-sulfanuide", "P-72.3"),
    ("C[P-](C)(C)(C)(C)C", "hexamethyl-λ5-phosphanuide", "P-72.3"),
    # -- a zwitterion whose anion is a sulfonamide nitrogen,:42411)
    ("C[NH2+]CC[N-]S(C)(=O)=O", "(methanesulfonyl)[2-(methylazaniumyl)ethyl]azanide", "P-74.0"),
    ("[NH3+]Cc1ccccc1S(=O)(=O)[N-]C", "[2-(azaniumylmethyl)benzene-1-sulfonyl](methyl)azanide", "P-74.0"),
    ("C[NH+](C)CC[N-]S(C)(=O)=O", "[2-(dimethylazaniumyl)ethyl](methanesulfonyl)azanide", "P-74.0"),
    # -- the anion of a ketene imine,:41047; ketenes,:29292)
    ("C(=C(C#N)S(=O)(=O)C(F)(F)F)=[N-]", "2-cyano-2-(trifluoromethanesulfonyl)ethen-1-iminide", "P-72.2.2.2.3"),
    ("CC(C)(C)C(=C=[N-])C#N", "2-tert-butyl-2-cyanoethen-1-iminide", "P-72.2.2.2.3"),
    ("N#CC(C#N)=C=[N-]", "2,2-dicyanoethen-1-iminide", "P-72.2.2.2.3"),
    ("C1CCC(CC1)=C=[N-]", "cyclohexylidenemethaniminide", "P-72.2.2.2.3"),
    ("C1C(C(=C=[N-])C(=S)NC1=O)C2=CC=CC=C2",
     "(6-oxo-4-phenyl-2-sulfanylidenepiperidin-3-ylidene)methaniminide", "P-72.2.2.2.3"),
    # -- the anion of a diazene
    ("CN=[N-]", "2-methyldiazen-1-ide", "P-72.2.2.1"),
    ("c1ccccc1N=[N-]", "2-phenyldiazen-1-ide", "P-72.2.2.1"),
    ("C=C(c1ccccc1)NN=[N-]", "2-[(1-phenylethenyl)amino]diazen-1-ide", "P-72.2.2.1"),
    ("COC1=CC=C(C=C1)NC(=S)C(=C=[N-])C(=O)OC2CCCCC2",
     "2-[(cyclohexyloxy)carbonyl]-2-{[(4-methoxyphenyl)amino](sulfanylidene)methyl}ethen-1-iminide",
     "P-72.2.2.2.3"),
    ("CC(=[N-])CC(=O)O", "(1-carboxypropan-2-ylidene)azanide", "P-72.2.2.2.4"),
    ("CN(C)[NH-]", "2,2-dimethylhydrazin-1-ide", "P-72.2.2.1"),
    # -- the zwitterion of a diazene
    ("c1ccccc1[NH+]=[N-]", "1-phenyldiazen-1-ium-2-ide", "P-74.1.1"),
    ("C[N+](C)=[N-]", "1,1-dimethyldiazen-1-ium-2-ide", "P-74.1.1"),
    # -- the anion of a phosphorus(III) acid: 'ous acid' -> 'ite',:40955)
    ("CCP([O-])[O-]", "ethylphosphonite", "P-72.2.2.2.1.1"),
    ("CP(C)[O-]", "dimethylphosphinite", "P-72.2.2.2.1.1"),
    # -- the diazonium ion on a nitrogen: the parent hydride of an amino group is azane,:41607)
    ("CN(C)[N+]#N", "dimethylazanediazonium", "P-73.2.2.3"),
    ("CC(=O)N(C)[N+]#N", "acetyl(methyl)azanediazonium", "P-73.2.2.3"),
    # -- the azinate,:25969): a nitrogen cation with two oxides
    ("CC=[N+]([O-])[O-]", "ethylideneazinate", "P-61.5.3"),
    ("C[NH+]([O-])[O-]", "methylazinate", "P-61.5.3"),
    ("c1ccccc1[NH+]([O-])[O-]", "phenylazinate", "P-61.5.3"),
    ("C[N+](C)([O-])[O-]", "dimethylazinate", "P-61.5.3"),
    ("C=C1C=CC(=[N+]([O-])[O-])C=C1", "(4-methylidenecyclohexa-2,5-dien-1-ylidene)azinate", "P-61.5.3"),
]


@pytest.mark.parametrize("smiles,expected,rule", CASES, ids=[c[0] for c in CASES])
def test_name_pinned_and_read_back(smiles, expected, rule):
    row = name_best_effort(smiles)
    assert row["name"] == expected, (rule, row["name"], row["tier"])
    assert _reads_back(smiles, row["name"])


@pytest.mark.parametrize("smiles", [c[0] for c in CASES if c[2] not in ("P-73.4", "P-72.2.2.2.1.1")])
def test_class_without_a_blue_book_example_is_no_pin(smiles):
    """The Blue Book gives no example of these anions (the lambda hydride of a substituted centre is a
    'preselected name' of the parent hydride only,:41112): the name is valid, not preferred."""
    assert name_best_effort(smiles)["tier"] != "pin_verified"


# -- the names that were given before keep their spelling
CONTROLS = [
    ("C[P-](C)(C)C", "tetramethylphosphanuide"),
    ("C[Si-](C)(C)(C)C", "pentamethylsilanuide"),
    ("C[S-]", "methanethiolate"),
    ("CC(=S)[S-]", "ethane(dithioate)"),
    ("C1CC[N+]2(C1)CCCC2", "5λ5-azaspiro[4.4]nonan-5-ylium"),
    ("CC=N(=O)O", "ethylideneazinic acid"),
    ("CC(=O)Nc1ccc([N-]S(C)(=O)=O)cc1", "(4-acetamidophenyl)(methanesulfonyl)azanide"),
    ("CC=[N-]", "ethaniminide"),
]


@pytest.mark.parametrize("smiles,expected", CONTROLS, ids=[c[0] for c in CONTROLS])
def test_controls_keep_their_names(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


# -- the λ-convention of the -uide anion, without a JVM
@pytest.mark.parametrize("symbol,bonds,lam", [
    ("S", 3, 2), ("S", 5, 4), ("S", 7, 6), ("S", 1, None), ("S", 2, None), ("S", 4, None),
    ("P", 4, 3), ("P", 6, 5), ("P", 5, None), ("I", 2, 1), ("I", 4, 3), ("I", 6, 5), ("I", 8, 7),
    ("Si", 5, 4), ("Si", 7, None), ("B", 4, 3), ("C", 4, None),
])
def test_uide_lambda(symbol, bonds, lam):
    from orthonym.rules.uide_lambda import uide_lambda
    assert uide_lambda(symbol, bonds) == lam


def test_a_chalcogen_with_three_or_more_bonds_is_no_thiolate():
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import classify_anion
    for smiles, cls in [("N=[S-](=O)N", "uide_anion"), ("C[S-]", "thiolate"), ("CC(=S)[S-]", "carbodithioate")]:
        mol = Chem.MolFromSmiles(smiles)
        assert classify_anion(mol, get_ion_sites(mol)["anions"][0]) == cls
