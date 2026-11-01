"""The nucleoside producer names what its name states, and refuses everything else.

 "SUBSTITUTION ON NUCLEOSIDES" (the Blue Book): "Nucleosides having retained names can
be fully substituted on the purine or pyrimidine ring.... The ribofuranosyl component may be modified
as prescribed for carbohydrates"; example `2'-deoxy-1-methylguanosine` (:54967): '2'-deoxy' is a prefix
cited in alphanumerical order with the base prefixes, not part of the stem.

The producer reads the base from its exocyclic N/O markers and takes the stem from a table, so it
must check what the stem and the alkyl words state: the beta-D-ribofuranosyl configuration
 "RETAINED NAMES OF NUCLEOSIDES",:54943), a sugar with nothing but the hydroxy groups, a
mancude base, and unbranched saturated alkyl groups. Each refused row below used to get a retained
name of another compound (AZT was '5-methyl2'-deoxyuridine'); the OPSIN read-back blocked them, but
the producer must not build them.
"""
import pytest
from rdkit import Chem

from orthonym.rules.nucleosides import name_nucleoside

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles, expected", [
    # (:54967): '2'-deoxy' sorts with the base prefixes
    ("Nc1nc2c(ncn2[C@H]9C[C@H](O)[C@@H](CO)O9)c(=O)n1C", "2'-deoxy-1-methylguanosine"),
    ("CCNc1nc2c(ncn2[C@H]9C[C@H](O)[C@@H](CO)O9)c(=O)[nH]1", "2'-deoxy-N2-ethylguanosine"),
    ("COc1nc(N)nc2c1ncn2[C@H]9C[C@H](O)[C@@H](CO)O9", "2'-deoxy-O6-methylguanosine"),
    ("Cc1cn([C@H]9C[C@H](O)[C@@H](CO)O9)c(=O)nc1N", "2'-deoxy-5-methylcytidine"),
    # ribo series and phosphate / acyl esters are unchanged
    ("Cc1cn([C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O)c(=O)[nH]c1=O", "5-methyluridine"),
    ("CNc1ncnc2c1ncn2[C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O", "N6-methyladenosine"),
    ("CN(C)c1ncnc2c1ncn2[C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O", "N6,N6-dimethyladenosine"),
    ("CN1C=NC2=C(C1=N)N=CN2[C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O", "1-methyladenosine"),
    ("Nc1ncnc2c1ncn2[C@H]9C[C@H](O)[C@@H](COP(=O)(O)O)O9",
     "2'-deoxyadenosine 5'-(dihydrogen phosphate)"),
    ("CC(=O)OC[C@H]9O[C@@H](n1cnc2c(N)ncnc21)[C@H](OC(C)=O)[C@@H]9OC(C)=O",
     "adenosine 2',3',5'-triacetate"),
])
def test_names_what_it_states(smiles, expected):
    assert name_nucleoside(Chem.MolFromSmiles(smiles)) == expected


@pytest.mark.parametrize("label, smiles", [
    ("3'-azido (AZT)", "Cc1cn([C@H]9C[C@H](N=[N+]=[N-])[C@@H](CO)O9)c(=O)[nH]c1=O"),
    ("3'-fluoro", "Cc1cn([C@H]9C[C@H](F)[C@@H](CO)O9)c(=O)[nH]c1=O"),
    ("5'-deoxy", "Cc1cn([C@@H]9O[C@H](C)[C@@H](O)[C@H]9O)c(=O)[nH]c1=O"),
    ("3'-deoxy", "CNc1ncnc2c1ncn2[C@@H]9O[C@H](CO)C[C@H]9O"),
    ("alpha anomer", "Cc1cn([C@@H]9C[C@H](O)[C@@H](CO)O9)c(=O)[nH]c1=O"),
    ("no stereo", "Cc1cn(C9CC(O)C(CO)O9)c(=O)[nH]c1=O"),
    ("ethynyl, not ethyl", "C#Cc1cn([C@H]9C[C@H](O)[C@@H](CO)O9)c(=O)[nH]c1=O"),
    ("isopropyl, not propyl", "CC(C)c1cn([C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O)c(=O)[nH]c1=O"),
    ("5,6-dihydro base", "CC1CN([C@H]9C[C@H](O)[C@@H](CO)O9)C(=O)NC1=O"),
    # the old producer named this '2-methyladenosine', citing the methyl and dropping the N-hydroxy
    ("N6-hydroxy", "ONc1nc(C)nc2c1ncn2[C@@H]9O[C@H](CO)[C@@H](O)[C@H]9O"),
    ("methyl on the glycosidic N", "C[N+]1(C=NC2=C(N=CN=C21)N)[C@H]3C[C@@H]([C@H](O3)CO)O"),
])
def test_refuses_what_it_does_not_state(label, smiles):
    assert name_nucleoside(Chem.MolFromSmiles(smiles)) is None, label
