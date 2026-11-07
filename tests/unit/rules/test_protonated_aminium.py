"""A protonated chain amine beside a senior neutral group is the '-aminium' parent (``rules.protonated_aminium``).

 (the Blue Book-18170): '6 Cations' before '7 Acids'; the acid is the 'carboxy' prefix
 :29910, '4-carboxy-1-methylpyridin-1-ium chloride (PIN)':29914); the cation is named from the amine by
Table 7.4 'amine -> aminium':41415; '3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide (PIN)':31709).
Was '3-(methylazaniumyl)propanoic acid', labelled below the PIN.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,expected", [
    ("C[NH2+]CCC(=O)O", "2-carboxy-N-methylethan-1-aminium"),
    ("C[NH+](C)CCC(=O)O", "2-carboxy-N,N-dimethylethan-1-aminium"),
    # -NH3+: no N-substituent, so the pipeline names the amine-principal neutral molecule (was
    # '2-azaniumylpropanoic acid' below the PIN, and 'glycinium', 'L-alaninium', names with no PIN)
    ("CC([NH3+])C(=O)O", "1-carboxyethan-1-aminium"),
    ("[NH3+]CCCC(=O)O", "3-carboxypropan-1-aminium"),
    ("C[C@H]([NH3+])C(=O)O", "(1S)-1-carboxyethan-1-aminium"),
    # '1-hydrazinylmethanamine (PIN)', the Blue Book): the C locant of methanamine is cited
    ("[NH3+]CC(=O)O", "1-carboxymethanaminium"),
    ("C[NH2+]CC(=O)O", "1-carboxy-N-methylmethanaminium"),
    # aryl: 'aniline -> anilinium':41415; 'N,N,N-trimethylanilinium (PIN)':41455)
    ("[NH3+]c1ccc(C(=O)O)cc1", "4-carboxyanilinium"),
    ("Oc1ccc([NH3+])cc1", "4-hydroxyanilinium"),
    ("[NH3+]c1ccc(S(=O)(=O)O)cc1", "4-sulfoanilinium"),
    ("C[NH2+]c1ccc(C(=O)O)cc1", "4-carboxy-N-methylanilinium"),
    ("C[NH+](C)c1ccc(C(=O)O)cc1", "4-carboxy-N,N-dimethylanilinium"),
])
def test_pin(smiles, expected):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_the_zwitterion_keeps_the_anion_parent():
    # (:42411): the anionic centre is the parent of a zwitterion
    assert tier_row("CC([NH3+])C(=O)[O-]", "pin")["name"] == "2-azaniumylpropanoate"


@pytest.mark.parametrize("smiles,expected", [
    ("[NH3+]c1ccc(C(=O)[O-])cc1", "4-azaniumylbenzoate"),
    ("[NH3+]CC(=O)[O-]", "azaniumylacetate"),
])
def test_aryl_and_glycine_zwitterions_keep_the_anion_parent(smiles, expected):
    assert tier_row(smiles, "pin")["name"] == expected


def test_a_second_amine_nitrogen_declines_the_forced_amine():
    # the forced suffix would be 'diamine', two cationic centres; the molecule is left to the other producers
    from rdkit import Chem
    from orthonym.rules.protonated_aminium import name_protonated_aminium
    mol = Chem.MolFromSmiles("NCCC([NH3+])C(=O)O")
    site = {"atom_idx": next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)}
    assert name_protonated_aminium(mol, site) == ""


@pytest.mark.parametrize("smiles,expected", [
    # an N-acyl cation is an amidium, not an aminium; '*N*,*N*,*N*-trimethylbenzamidium (PIN)'
    # the Blue Book)
    ("C[NH2+]C(C)=O", "N-methylacetamidium"),
    ("C[NH+](C)C(C)=O", "N,N-dimethylacetamidium"),
    ("C[NH2+]C(=S)C", "N-methylethanethioamidium"),
])
def test_an_n_acyl_cation_keeps_the_amidium_pin(smiles, expected):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (expected, "pin_verified"), row


def test_a_protonated_carbamate_claims_no_aminium_pin():
    # (:41415): the largest neutral parent with a nitrogen suffix; never '1-methoxy-1-oxomethanaminium'
    assert tier_row("C[NH2+]C(=O)OC", "pin")["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles,expected", [
    ("[NH3+][C@@H](Cc1ccccc1)C(=O)O", "L-phenylalaninium"),
    ("[NH3+]C(c1ccccc1)C(=O)O", "azaniumyl(phenyl)acetic acid"),
])
def test_an_aminium_name_that_does_not_read_back_leaves_the_older_name(smiles, expected):
    # the forced amine re-entry names another molecule here; the router reads it back and keeps the older name
    assert tier_row(smiles, "best-effort")["name"] == expected


def test_a_phosphonate_zwitterion_keeps_its_anion_name():
    # the zwitterion is named through its cation with the acid suffix kept; the aminium form would leave nothing
    # to ionize (tests/integration/test_stereo_benchmark.py row s00_HA7)
    row = tier_row("C[C@@H]([NH3+])P(=O)([O-])[O-]", "best-effort")
    assert row["name"] == "[(1S)-1-azaniumylethyl]phosphonate", row
