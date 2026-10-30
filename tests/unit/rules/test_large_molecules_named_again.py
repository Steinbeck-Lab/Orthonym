"""Large molecules the best-effort tier names again.

- A cage whose branch RDKit cannot write leaves the prefix criterion of the cage numbering unknown
  (``rules.cage_numbering.cage_branches`` returns None) instead of ending the analysis; the numbering is then not
  certified, so the name is no PIN and its locants are not pinned here.
- Eight ylium centres: '-octakis(ylium)', (the Blue Book).
- A molecule of many tert-butyl groups whose OPSIN parse is matched onto it in canonical atom order
  (``perception.substructure``).
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    pytest.param('CC(C)(C)C1=CC(=CC(=C1)[C+]2C3=CC=C(N3)[C+](C4=CC=C(N4)[C+](C5=CC6=C(N5)[C+](C7=CC=C2N7)C8=CC9=C(C=C8C6=O)[C+]1C2=CC=C(N2)[C+](C2=CC=C(N2)[C+](C2=CC=C(N2)[C+](C2=CC(=C1N2)C9=O)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C)C1=CC=CC=C1)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C)C1=CC=CC=C1)C(C)(C)C',
                 '11,21,35,45-tetrakis(3,5-di-tert-butylphenyl)-6,30-dioxo-16,40-diphenyl-9,33,49,50,51,53,54,55-octaazatridecacyclo[44.2.1.1^7,10.1^12,15.1^17,20.1^22,25.1^31,34.1^36,39.1^41,44.0^2,32.0^3,29.0^5,27.0^8,26]hexapentaconta-1(48),3(29),4,7,10(56),12,14,17,19,22,24,27,31,34(52),36,38,41,43,46-nonadecaen-2,11,16,21,26,35,40,45-octakis(ylium)',
                 id='octakis_ylium_cid_139042876'),
    pytest.param('B12C3=CC(=CC4=C3C(CC5=C6C7=CC=CC=C7C8=C(C6=CC(=C51)N9C1=C(C=C(C=C1)N(C1=CC=C(C=C1)C(C)(C)C)C1=CC=C(C=C1)C(C)(C)C)C1=C9C2=CC(=C1)N(C1=CC=C(C=C1)C(C)(C)C)C1=CC=C(C=C1)C(C)(C)C)C=CC(=C8)N1C2=C(C=C(C=C2)C(C)(C)C2=CC=CC=C2)C2=C1C=CC(=C2)C(C)(C)C1=CC=CC=C1)C1=C4C=C(C=C1)N(C1=CC=C(C=C1)C(C)(C)C)C1=CC=C(C=C1)C(C)(C)C)N(C1=CC=C(C=C1)C(C)(C)C)C1=CC=C(C=C1)C(C)(C)C',
                 '6,11,17,22-tetrakis[bis(4-tert-butylphenyl)amino]-38-[3,6-bis(2-phenylpropan-2-yl)-9H-carbazol-9-yl]-2-aza-14-boradodecacyclo[26.15.1.1^2,9.1^15,19.0^3,8.0^14,44.0^20,25.0^29,42.0^30,35.0^36,41.0^13,46.0^26,45]hexatetraconta-1(44),3,5,7,9,11,13(46),15(45),16,18,20(25),21,23,28,30,32,34,36,38,40,42-henicosaene',
                 id='opsin_order_match_cid_174094167'),
])
def test_the_molecule_is_named(smiles, expected):
    row = name_best_effort(smiles)
    assert row["name"] == expected, row
    assert name_is_rt_exact(row["name"], smiles), row


def test_a_cage_with_a_branch_rdkit_cannot_write_is_named_below_the_pin():
    smiles = 'COC1=C2C=C3CC4=CC(=C5C=C4CC6=CC(=C(C=C6CC3=C1)OCCOC7=C(C=C8CC9=CC(=C(C=C9CC1=CC(=C(C=C1CC8=C7)OCC(=O)C(N[C@@H](CCCNC(=N)N)C(=O)N([C@@H](CCCCN)C=O)N[C@H](CCCNC(=N)N)C=O)SC1CC(=O)CN(C1=O)CCC(=O)NCCOCCOCCNC(=O)CCCCCC1C3C(CS1)NC(=O)N3)OCCO5)OC)OCCO2)OC)OC)OC'
    row = name_best_effort(smiles)
    assert row["name"] and not row.get("is_pin"), row
    assert name_is_rt_exact(row["name"], smiles), row
