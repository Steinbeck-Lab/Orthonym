"""An amide-class group (amide, thioamide, amidine, sulfonamide) that is the senior class takes its suffix
class 11, the Blue Book;,:33196;,:32971; 'benzenecarbothioamide (PIN)',:33155;
'pyrrolidine-1-sulfonamide (PIN)',:32758). A single-word name that ends in a bare parent hydride and cites the group only as
a prefix is correct but no PIN (validation/spelling/checks_amide_class_prefix.py).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_amide_class_prefix import amide_class_as_prefix_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CN(C)C(=S)c1ccccc1", "(dimethylcarbamothioyl)benzene"),
    ("CNC(=S)c1ccccc1", "(methylcarbamothioyl)benzene"),
    ("NC(=N)N1CCOCC1", "4-carbamimidoylmorpholine"),
    ("NS(=O)(=O)Cc1ccc(CC)cc1", "1-ethyl-4-(sulfamoylmethyl)benzene"),
    ("CC(=O)NS(=O)(=O)Cc1ccc(C)cc1", "1-[(acetylsulfamoyl)methyl]-4-methylbenzene"),
    ("NC(=O)Cc1ccccc1", "(carbamoylmethyl)benzene"),                                  # a plain amide
    ("CNC(=S)c1ccc(cc1)[N+](=O)[O-]", "1-(methylcarbamothioyl)-4-nitrobenzene"),      # nitro: no ion
])
def test_a_prefix_only_amide_class_name_is_flagged(smiles, name):
    failure = amide_class_as_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-66.1.1.4.2", name


@pytest.mark.parametrize("smiles,name", [
    ("NC(=S)c1ccc(cc1)[N+](=O)[O-]", "4-nitrobenzene-1-carbothioamide"),
    ("NS(=O)(=O)N1CCCCC1", "piperidine-1-sulfonamide"),
    ("NC(=N)NC1CCCCC1", "N-cyclohexylguanidine"),
    ("CN(C)C(=S)SC", "methyl dimethylcarbamodithioate"),
    ("CN(C)C(=S)SSC(C)(C)C", "1-(tert-butyldisulfanyl)-N,N-dimethylmethanethioamide"),   # not a hydride ending
    ("CN(C)C(=S)SSC(C)(C)C", "2-tert-butyl-1-(dimethylcarbamothioyl)disulfane"),        # a thiuram: excluded
    ("NC(=O)c1ccc(cc1)S(Cl)(=O)=O", "4-carbamoylbenzene-1-sulfonyl chloride"),
    ("NC(=S)Cc1ccc(cc1)C(F)=O", "4-(2-amino-2-sulfanylideneethyl)benzoyl fluoride"),
    ("CC(=S)SCc1ccccc1", "(thiocarbamoylmethyl)benzene"),                                 # no amide N; 'thiocarbamoyl' not read
    ("CS(=O)(=O)N1CCCCC1", "1-(methanesulfonyl)piperidine"),         # a sulfonyl on a ring N: no 'sulfamoyl' word
    ("NC(=O)Cc1ccccc1.Cl", "(carbamoylmethyl)benzene hydrochloride"),   # a mixture
    ("C[N+](C)(C)CC(N)=O", "(carbamoylmethyl)trimethylammonium"),       # an ion
])
def test_other_names_pass(smiles, name):
    assert amide_class_as_prefix_check(Chem.MolFromSmiles(smiles), name) is None, name
