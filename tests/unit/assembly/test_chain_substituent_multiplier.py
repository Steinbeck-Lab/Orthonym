"""The multiplier of a prefix on a chain substituent: 'bis' where 'di' would name another group.

 (the Blue Book, "The prefixes 'bis', 'tris', 'tetrakis', etc. are also used to
avoid ambiguity:") (a) "before a mononuclear subset of a polynyclear acyclic structure" (:7136):
"bis(sulfanyl) (preferred prefix; defines two –SH groups, see; whereas disulfanyl
defines the –SSH group; see " (:7140), with the Note "And similarly for the analogous
Se, Te, N, P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, and Tl groups" (:7146); '3,4-bis
(sulfanyl)butanoic acid (PIN)' (:27601). (:7104) (a): 'bis' multiplies compound or
complex (substituted) prefixes, 'bis(dimethylamino) (preferred prefix)'. (###
GENERAL METHODOLOGY,:17954): "... to distinguish between two simple prefixes and those
including a basic multiplying term, for example disulfanyl, '–SSH', and bis(sulfanyl), two –SH
groups."

The four chain-substituent assemblers (unbranched, branched, internal free valence, branched
unsaturated) kept a private {2: 'di', 3: 'tri'} table: 'disulfanylmethyl' for –CH(SH)2 names
–CH2–S–SH (OPSIN reads it back to another molecule, so the final-name check threw the
name away and the PIN tier abstained), and '1,2-disulfanylethyl' for two –SH groups was
labelled a PIN. The multiplier now comes from naming_utils.multiplied_component, as in the
other producers. Names read back by OPSIN 2.9.0 to the input's full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import _name_polyfunctional_acyclic_substituent_impl
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import name_is_rt_exact


def _branch_name(smiles):
    """Name the branch of ``smiles`` hanging off atom 0: atom 1 is the free valence and every
    atom but atom 0 belongs to the branch."""
    mol = Chem.MolFromSmiles(smiles)
    return _name_polyfunctional_acyclic_substituent_impl(
        mol, list(range(1, mol.GetNumAtoms())), 1, {0})


BIS_ROWS = [
    # unbranched chain, free valence at locant 1
    ("CC(S)CS", "1,2-bis(sulfanyl)ethyl"),
    ("CC(P)CP", "1,2-bis(phosphanyl)ethyl"),
    ("CC(S)C(S)CS", "1,2,3-tris(sulfanyl)propyl"),
    ("CC(S)(S)C", "1,1-bis(sulfanyl)ethyl"),
    ("CC(N(C)C)CN(C)C", "1,2-bis(dimethylamino)ethyl"),
    # one-carbon core: the locant is elided ('disulfanylmethyl' is –CH2–S–SH)
    ("CC(S)S", "bis(sulfanyl)methyl"),
    ("CC(S)(S)S", "tris(sulfanyl)methyl"),
    ("CC(Cl)(S)S", "chlorobis(sulfanyl)methyl"),
    # branched chain, internal free valence, unsaturated chain
    ("CC(S)C(C)CS", "2-methyl-1,3-bis(sulfanyl)propyl"),
    ("CC(CS)CS", "1,3-bis(sulfanyl)propan-2-yl"),
    ("CC(S)C=CS", "1,3-bis(sulfanyl)prop-2-en-1-yl"),
]


@pytest.mark.parametrize("smiles,prefix", BIS_ROWS)
def test_a_mononuclear_subset_prefix_is_multiplied_by_bis(smiles, prefix):
    assert _branch_name(smiles) == prefix


@pytest.mark.parametrize("smiles,prefix", [
    ("CC(Cl)CCl", "1,2-dichloroethyl"),
    ("CC(O)CO", "1,2-dihydroxyethyl"),
    ("CC(N)CN", "1,2-diaminoethyl"),
    ("CC(O)O", "dihydroxymethyl"),
    ("CC(O)C(C)CO", "1,3-dihydroxy-2-methylpropyl"),
])
def test_simple_prefixes_keep_di(smiles, prefix):
    assert _branch_name(smiles) == prefix


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,prefix", BIS_ROWS)
def test_the_prefixes_read_back(smiles, prefix):
    # The branch on benzene: '[<prefix>]benzene'.
    assert name_is_rt_exact(f"[{prefix}]benzene", "c1ccccc1" + smiles[1:])


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)c1ccccc1C(S)CS", "2-[1,2-bis(sulfanyl)ethyl]benzoic acid"),
    ("OC(=O)c1ccccc1C(S)S", "2-[bis(sulfanyl)methyl]benzoic acid"),
    ("OC(=O)c1ccccc1C(CS)CS", "2-[1,3-bis(sulfanyl)propan-2-yl]benzoic acid"),
    ("OC(=O)CC(CCCCCC)C(P)CP", "3-[1,2-bis(phosphanyl)ethyl]nonanoic acid"),
])
def test_the_names_are_the_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
