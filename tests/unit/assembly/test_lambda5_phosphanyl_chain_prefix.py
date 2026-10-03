"""A λ5-phosphanyl group on a chain substituent, numbered from the free valence.

 (the Blue Book, "The preferred IUPAC name is based on the senior parent structure
that has the lower locant or set of locants for substituents cited as prefixes"), example (9) at
:21749: '5-bromo-3-[3-nitro-1-(λ5-phosphanyl)propyl]-4-(λ5-phosphanyl)hexanoic acid (PIN)';
(:21791) example (11) at:22152: '3-[2-bromo-1-(λ5-phosphanyl)propyl]-5-chloro-4-(λ5-phosphanyl)
hexanoic acid (PIN)'. The free valence of the chain substituent takes locant 1 and its
prefixes are numbered from it.

The chain-substituent namer placed nitro, halogen and 'phosphanyl' but returned None for the
λ5-hydride -PH4; the capped-name path built the first name, and since the string converter
declines a prefixed chain stem (its locants are the capped molecule's) the default tier abstained
on it. The namer places '(λ5-phosphanyl)' now where the numbering has no choice: an unbranched
chain whose free valence is on a terminal carbon. Where a chain or direction is chosen,
(:22770) first ranks the substituents of the highest bonding number, which the branched and
internal namers do not apply; two λ5-phosphanyl groups take 'bis' (:22784); those decline.
Names read back by OPSIN 2.9.0 to the input's full InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN,
    _name_polyfunctional_acyclic_substituent_impl,
    parent_to_prefix,
)
from tests.support.pin_tiers import assert_pin_at_both_tiers
from tests.support.rt_assert import name_is_rt_exact


def _branch_name(smiles):
    """Name the branch of ``smiles`` hanging off atom 0: atom 1 is the free valence and every
    atom but atom 0 belongs to the branch."""
    mol = Chem.MolFromSmiles(smiles)
    sub = list(range(1, mol.GetNumAtoms()))
    return _name_polyfunctional_acyclic_substituent_impl(mol, sub, 1, {0})


@pytest.mark.parametrize("smiles,prefix", [
    ("CC([PH4])CC[N+](=O)[O-]", "3-nitro-1-(λ5-phosphanyl)propyl"),
    ("CC([PH4])C(C)Br", "2-bromo-1-(λ5-phosphanyl)propyl"),
    ("CC([PH4])C", "1-(λ5-phosphanyl)ethyl"),
    ("CCCC[PH4]", "3-(λ5-phosphanyl)propyl"),
    ("CC[PH4]", "(λ5-phosphanyl)methyl"),
    ("CC([PH4])CCP", "1-(λ5-phosphanyl)-3-phosphanylpropyl"),
])
def test_lambda5_phosphanyl_is_numbered_from_the_free_valence(smiles, prefix):
    assert _branch_name(smiles) == prefix


@pytest.mark.parametrize("smiles,prefix", [
    ("CC(P)CC[PH4]", "3-(λ5-phosphanyl)-1-phosphanylpropyl"),
    ("CC(P)C[PH4]", "2-(λ5-phosphanyl)-1-phosphanylethyl"),
    ("CC(P)C(P)C[PH4]", "3-(λ5-phosphanyl)-1,2-bis(phosphanyl)propyl"),
    ("CC([PH4])C(P)CP", "1-(λ5-phosphanyl)-2,3-bis(phosphanyl)propyl"),
])
def test_lambda5_phosphanyl_is_cited_before_phosphanyl(smiles, prefix):
    """The letters of 'λ5-phosphanyl' and 'phosphanyl' are the same; the Blue Book cites the λ5
    group first even where it has the higher locant, so this is not the lower-locant-first order
    of. (h) (### NUMBERING, the Blue Book) example at:3334:
    '1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol (PIN) (λ5-phosphanyl is cited before phosphanyl
    and is given the lower locant)'; (## THE PRINCIPAL SUBSTITUENT CHAIN) at
    :22756: '5-(λ5-phosphanyl)-2,3-bis(phosphanyl)heptan-4-yl (preferred prefix)'; at
    :22798: '2-(λ5 phosphanyl)-3-phosphanyl-1-[2-(λ5 phosphanyl)-1 phosphanylpropyl]butyl'.
    Two phosphanyl groups take 'bis' (a),:7140 and:7146)."""
    assert _branch_name(smiles) == prefix


@pytest.mark.parametrize("smiles", [
    "CC([PH4])C[PH4]",      # two λ5 groups: 'bis(λ5-phosphanyl)', not spelled here
    "CC(C[PH4])CC",         # internal free valence: the direction is chosen
    "CCC(C)C[PH4]",         # branched: the principal chain is chosen
    "CC(O)[PH4]",           # one carbon, two prefixes: enclosing not built
    "CC(c1ccccc1)[PH4]",    # one carbon, a ring branch and the λ5 group
])
def test_a_choice_of_chain_or_numbering_declines(smiles):
    assert _branch_name(smiles) is None


@pytest.mark.parametrize("smiles,prefix", [
    ("CC(Cl)CC[N+](=O)[O-]", "1-chloro-3-nitropropyl"),
    ("CCCP", "2-phosphanylethyl"),
])
def test_controls_keep_their_prefix(smiles, prefix):
    assert _branch_name(smiles) == prefix


def test_the_string_converter_still_declines_the_capped_name():
    assert parent_to_prefix("3-nitro-1-(λ5-phosphanyl)propane", 3,
                            attach_locant=ATTACH_LOCANT_UNKNOWN) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("CC(Br)C([PH4])C(CC(=O)O)C([PH4])CC[N+](=O)[O-]",
     "5-bromo-3-[3-nitro-1-(λ5-phosphanyl)propyl]-4-(λ5-phosphanyl)hexanoic acid"),
    ("CC(Cl)C([PH4])C(CC(=O)O)C([PH4])C(C)Br",
     "3-[2-bromo-1-(λ5-phosphanyl)propyl]-5-chloro-4-(λ5-phosphanyl)hexanoic acid"),
])
def test_blue_book_rows_are_the_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("OC(=O)CC(CCC)C(P)C[PH4]", "3-[2-(λ5-phosphanyl)-1-phosphanylethyl]hexanoic acid"),
    ("OC(=O)CC(CCCCCC)C(P)C(P)C[PH4]",
     "3-[3-(λ5-phosphanyl)-1,2-bis(phosphanyl)propyl]nonanoic acid"),
])
def test_lambda5_and_phosphanyl_on_one_substituent_at_both_tiers(smiles, pin):
    # λ5 cited first (:3334,:22756,:22798) and 'bis(phosphanyl)' (a),:7140).
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,prefix", [
    ("CC([PH4])CC[N+](=O)[O-]", "3-nitro-1-(λ5-phosphanyl)propyl"),
    ("CC([PH4])C(C)Br", "2-bromo-1-(λ5-phosphanyl)propyl"),
])
def test_the_prefixes_read_back(smiles, prefix):
    # The branch on a methyl parent: '<prefix>' + 'methane' is not a name, so read back the
    # branch on benzene instead ('(<prefix>)benzene').
    ring_smiles = "c1ccccc1" + smiles[1:]
    assert name_is_rt_exact(f"({prefix})benzene", ring_smiles)
