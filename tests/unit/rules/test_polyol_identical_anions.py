"""Identical anions on one polyol: functional class multiplicative names.

 (the Blue Book): "When anions are identical functional class
multiplicative nomenclature is used... 'di', 'tri', etc. are used when anions are
unsubstituted; when substituted, prefixes 'bis', 'tris', etc. are used"; 'ethane-1,2-diyl
diacetate (PIN)' (:31823), 'propane-1,3-diyl bis(chloroacetate) (PIN)' (:31825), 'propane-1,2,3-
triyl triacetate (PIN)' (:31827). (:35918): esters of noncarbon acids "are named in
the same way as esters of organic acids". The diol diester producer read ClCH2-CO- as 'acetic'
(a different molecule, stopped only by the read-back) and rejected nothing about unsaturation;
the nitrite and nitrate producers named one anion and cited the other as a prefix
('2-(nitrosooxy)ethyl nitrite', pin_verified). Names read back by OPSIN 2.9.0 to the full
InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,pin", [
    ("O=NOCCON=O", "ethane-1,2-diyl dinitrite"),
    ("O=NOCCCON=O", "propane-1,3-diyl dinitrite"),
    ("[O-][N+](=O)OCCO[N+](=O)[O-]", "ethane-1,2-diyl dinitrate"),
    ("[O-][N+](=O)OCC(CO[N+](=O)[O-])O[N+](=O)[O-]", "propane-1,2,3-triyl trinitrate"),
    ("CC(CO[N+](=O)[O-])O[N+](=O)[O-]", "propane-1,2-diyl dinitrate"),
    ("ClCC(=O)OCCCOC(=O)CCl", "propane-1,3-diyl bis(chloroacetate)"),
    ("CC(=O)OCCOC(C)=O", "ethane-1,2-diyl diacetate"),
    ("CC(=O)OCC(COC(C)=O)OC(C)=O", "propane-1,2,3-triyl triacetate"),
    ("O=C(OCCOC(=O)c1ccccc1)c1ccccc1", "ethane-1,2-diyl dibenzoate"),
    # an acylal: both ester oxygens on one carbon,:32203)
    ("CCCC(=O)OC(C)OC(=O)CCC", "ethane-1,1-diyl dibutanoate"),
])
def test_polyol_identical_anions(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _esters(smiles):
    from rdkit import Chem
    from orthonym.perception.functional_groups import detect_functional_groups
    mol = Chem.MolFromSmiles(smiles)
    return mol, detect_functional_groups(mol)["ester"]


@pytest.mark.parametrize("smiles,expected", [
    # the acid keeps its substituents (was 'propane-1,3-diyl diacetate': Cl dropped)
    ("ClCC(=O)OCCCOC(=O)CCl", "propane-1,3-diyl bis(chloroacetate)"),
    # an unsaturated residue is not the saturated diyl (was 'butane-1,4-diyl diacetate'). Re-asserted 2026-11-05
    # (lane/l17 item 5, change-asserted-value): the producer declined (None) and now names the residue it sees, the
    # unsaturated group 'but-2-ene-1,4-diyl' (the retained prefix of, the Blue Book) with the descriptor
    # of its double bond in front style, '(2E)'), in front of the multiplied anion,
    #:31819). The name is read back by OPSIN to the full InChIKey, the E/Z layer included (asserted below).
    ("CC(=O)OC/C=C/COC(C)=O", "(2E)-but-2-ene-1,4-diyl diacetate"),
    ("CC(=O)OC/C=C\\COC(C)=O", "(2Z)-but-2-ene-1,4-diyl diacetate"),
])
def test_the_diol_diester_producer_names_only_what_it_sees(smiles, expected):
    from orthonym.rules.esters import _try_functional_class_diol_diester
    mol, esters = _esters(smiles)
    assert _try_functional_class_diol_diester(mol, esters) == expected
    if expected:
        assert_full_rt(expected, smiles)


def test_the_acyloxy_polyol_producer_declines_a_substituted_acid():
    # was '1,3-bis(acetyloxy)propane' (Cl dropped, a different molecule)
    from orthonym.rules.esters import name_polyol_polyester
    mol, esters = _esters("ClCC(=O)OCCCOC(=O)CCl")
    assert name_polyol_polyester(mol, esters) is None
