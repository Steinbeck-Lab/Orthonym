"""One name for every spelling where a choice fell to the order of the input atoms.

- Tied parent ring systems (``rules.ring_selection._most_prefixes_then_canonical``): the one with the most
  substituent groups cited as prefixes, (the Blue Book, "the maximum number of substituent groups
  cited as prefixes"), then the canonical ranks; the tie used to fall to the first system in atom order.
- The fused ring system a bridged fused name is built on (``rules.bridged_fused_pin.selection.ring_system``): of
  two largest systems of one size, the one senior by.
- The charge forms of a cation beside an azide (``namer._rdkit_charge_candidates``): RDKit stops at its cap, and
  which forms it gives then depends on the atom order; the forms are taken from ``perception.charge_shift``.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,other_spelling,expected", [
    pytest.param('CC1=C(N=C(C=C1CC2=CN=C(C=C2)/C(=C/N)/C=NC)C(=O)NC3CCOCC3)N4C=CC=N4',
                 'c1(/C(C=NC)=C/N)ncc(Cc2c(C)c(-n3cccn3)nc(C(=O)NC3CCOCC3)c2)cc1',
                 '4-[(6-{(1Z)-2-amino-1-[(methylimino)methyl]ethen-1-yl}pyridin-3-yl)methyl]-3-methyl-6-[(oxan-4-yl)carbamoyl]-2-(1H-pyrazol-1-yl)pyridine',
                 id='ring_tie_cid_162462495'),
    pytest.param('C1=[N+](NN=N1)N=[N+]=[N-]',
                 'N([n+]1[nH]nnc1)=[N+]=[N-]',
                 '1-azido-1H-tetrazol-2-ium',
                 id='charge_forms_cid_150949789'),
    pytest.param('C[C@@]1(COC2=C(N=C(C=C21)[C@@](CNC(=O)C3=CC4=CN(N=C4C(=C3)OC)C5CC5)(C6=NC=C(C=C6)F)O)C7=CC(=C(C=C7F)F)Cl)C(=O)N',
                 'c1(F)cc(F)c(-c2c3c(cc([C@](c4ccc(F)cn4)(CNC(=O)c4cc5c(nn(C6CC6)c5)c(OC)c4)O)n2)[C@](C(N)=O)(C)CO3)cc1Cl',
                 '(3R)-7-(5-chloro-2,4-difluorophenyl)-5-[(1R)-2-(2-cyclopropyl-7-methoxy-2H-indazole-5-carboxamido)-1-(5-fluoropyridin-2-yl)-1-hydroxyethyl]-3-methyl-2,3-dihydrofuro[2,3-c]pyridine-3-carboxamide',
                 id='fused_system_tie_cid_170411034'),
    pytest.param('C1=NN[N+](=C1)N=[N+]=[N-]',
                 'c1[n+](N=[N+]=[N-])[nH]nc1',
                 '1-azido-1H-1,2,3-triazol-2-ium',
                 id='charge_forms_cid_141621038'),
])
def test_both_spellings_give_one_name(smiles, other_spelling, expected):
    for s in (smiles, other_spelling):
        row = name_best_effort(s)
        assert row["name"] == expected, row
        assert name_is_rt_exact(row["name"], s), row
