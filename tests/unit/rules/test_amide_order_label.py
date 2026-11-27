"""Inside class 11 the amides follow their acids, the Blue Book "in the order of the corresponding acids"):
carboxylic (7a) before sulfonic (7a,:18172) before carbonic (7b,:18174); Table 4.4 'carboxamide' (16,:18759) before
'sulfonamide' (19,:18769); 'N-carbamoylbenzenesulfonamide (PIN)' (:33362); 'N-carbamimidoylacetamide (PIN)... (not
N-acetylguanidine)' and 'N-carbamimidoylurea (PIN)... (not N-carbamoylguanidine)',:34296). A name on
the junior amide is correct but no PIN (validation/spelling/checks_parent.py, principal_class_check).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_parent import principal_class_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)N=C(N)N", "N-acetylguanidine"),
    ("NC(=O)NC(N)=N", "N-carbamoylguanidine"),
    ("Cc1ccc(S(=O)(=O)N=C(N(C)C)N(C)C)cc1", "N,N,N',N'-tetramethyl-N''-(4-methylbenzene-1-sulfonyl)guanidine"),
    ("O=C(NS(=O)(=O)c1ccccc1Cl)c1ccccc1", "N-benzoyl-2-chlorobenzene-1-sulfonamide"),
    ("CCNS(=O)(=O)c1ccc(C(N)=S)cc1", "4-carbamothioyl-N-ethylbenzene-1-sulfonamide"),
    ("CC(=O)N=C1SC(S(N)(=O)=O)=NN1C", "5-(acetylimino)-4-methyl-4,5-dihydro-1,3,4-thiadiazole-2-sulfonamide"),
    ("NC(=O)NC(=O)c1ccccc1", "N-benzoylurea"),
    ("ONC(=O)c1ccc(S(N)(=O)=O)cc1", "4-(hydroxycarbamoyl)benzene-1-sulfonamide"),   # an N-hydroxy amide
    ("CN(C)C(=S)N=C(N(C)C)N(C)C", "N''-(dimethylcarbamothioyl)-N,N,N',N'-tetramethylguanidine"),
    ("CC(C)(C)OC(=O)NCCNC(=O)Nc1ccccc1", "N-[2-(tert-butoxycarbonylamino)ethyl]-N'-phenylurea"),  # the ester is senior
])
def test_a_name_on_a_junior_amide_is_flagged(smiles, name):
    failure = principal_class_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-41", name


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)N=C(N)N", "N-carbamimidoylacetamide"),                          #:34296
    ("NC(=O)NS(=O)(=O)c1ccccc1", "N-carbamoylbenzenesulfonamide"),          #:33362
    ("NC(=O)NC(=O)c1ccccc1", "N-carbamoylbenzamide"),                        #:33360
    ("CNC(=N)NC", "N,N'-dimethylguanidine"),                                 #:34262
    ("CN(C)C(=N)N(C)C", "N,N,N',N'-tetramethylguanidine"),
    ("O=C(Nc1ccccc1)Nc1ccccc1", "N,N'-diphenylurea"),
    ("CN(N=O)C(N)=O", "N-methyl-N-nitrosourea"),                             # an N-nitroso urea is a urea
    ("CS(N)(=O)=O", "methanesulfonamide"),
    ("NC(=O)N1CCCCC1", "piperidine-1-carboxamide"),                          # a urea named as a carboxamide
    ("O=C(NS(=O)(=O)c1ccccc1Cl)c1ccccc1", "N-(2-chlorobenzene-1-sulfonyl)benzamide"),
    # a sulfamide is an amide of sulfuric acid (class 7d), junior to the carbonic amides (7b,:18178)
    ("NC(=O)NS(N)(=O)=O", "N-sulfamoylurea"),
    ("NC(=N)NS(=O)(=O)N(C)C", "N-(dimethylsulfamoyl)guanidine"),
])
def test_names_on_the_senior_amide_pass(smiles, name):
    assert principal_class_check(Chem.MolFromSmiles(smiles), name) is None, name
