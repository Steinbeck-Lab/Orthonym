"""Hydrazinecarboxylic acids and their acyl prefixes take the locants 1 and 2 (program 14 / 17 leads, 2026-11-09).

 "Hydrazine and derivatives" (the Blue Book): "As a parent hydride, hydrazine is numbered by numerical
locants, 1 and 2, not N and N'", as in 'N,1-dimethylhydrazine-1-carboxamide (PIN)' (:38637). The acid was spelled
'N'-methylhydrazinecarboxylic acid' and labelled a PIN.

 "Substituent groups derived from hydrazides" (:33985-33990): -CO-NH-NH2 is named "(1) as an acyl group
derived from the corresponding acid, which gives the preferred prefix": 'hydrazinecarbonyl (preferred prefix)'. Ring
cations spelled it '(aminocarbamoyl)' and labelled the name a PIN.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CNNC(=O)O", "2-methylhydrazine-1-carboxylic acid"),
    ("CN(N)C(=O)O", "1-methylhydrazine-1-carboxylic acid"),
    ("CN(C)NC(=O)O", "2,2-dimethylhydrazine-1-carboxylic acid"),
    ("CCNN(C)C(=O)O", "2-ethyl-1-methylhydrazine-1-carboxylic acid"),
    ("NNC(=O)O", "hydrazinecarboxylic acid"),
    # a hydrazide keeps N and N', 'N'-methylacetohydrazide')
    ("CNNC(C)=O", "N'-methylacetohydrazide"),
])
def test_hydrazinecarboxylic_acid_is_numbered_1_and_2(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    ("NNC(=O)c1cc[n+](C)cc1", "4-(hydrazinecarbonyl)-1-methylpyridin-1-ium"),
    ("NNC(=O)C1CC[N+](C)(C)CC1", "4-(hydrazinecarbonyl)-1,1-dimethylpiperidin-1-ium"),
    ("CNNC(=O)c1cc[n+](C)cc1", "1-methyl-4-(2-methylhydrazine-1-carbonyl)pyridin-1-ium"),
    # unchanged: the benzene writer already spelled it (book example:34012 form)
    ("NNC(=O)c1ccc(C(=O)O)cc1", "4-(hydrazinecarbonyl)benzoic acid"),
])
def test_the_acyl_group_of_hydrazinecarboxylic_acid_is_hydrazinecarbonyl(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


def test_a_triazane_is_not_a_hydrazinecarbonyl():
    row = Orthonym().name_tiered("NNNC(=O)c1cc[n+](C)cc1")
    assert "hydrazinecarbonyl" not in row["name"] and "aminocarbamoyl" not in row["name"]


@pytest.mark.parametrize("smiles", [
    "NNC(=O)C1CC2CCC1C2",
    "C1[C@@H]2C[C@@H]([C@H]1C[C@@H]2C(=O)NN)C(=O)NN",
])
def test_a_hydrazide_cited_only_as_a_prefix_is_not_the_pin(smiles):
    """'-CO-NHNH2 carbohydrazide (preferred suffix)' (the Blue Book): a parent hydride that cites the principal
    group as 'hydrazinecarbonyl' and has no suffix is not the PIN (``pin_vocabulary._SUFFIX_GROUP_PREFIX``). The
    old '(aminocarbamoyl)' spelling was caught by 'carbamoyl'; the book spelling must be caught too."""
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


@pytest.mark.parametrize("smiles, name", [
    # an N=O or N=N on the amide N is no hydrazine: named on 'amino':36415), the carbamoyl form stays
    ("OC(=O)c1ccc(cc1)C(=O)NN=O", "4-(nitrosocarbamoyl)benzoic acid"),
    ("OC(=O)c1ccc(cc1)C(=O)N(C)N=O", "4-[methyl(nitroso)carbamoyl]benzoic acid"),
    ("OC(=O)c1ccc(cc1)C(=O)NN=NC", "4-[(methyldiazenyl)carbamoyl]benzoic acid"),
    # a hydrazone carbon on the second N is a ylidene prefix of the acid, as in
    # '2-(propan-2-ylidene)hydrazine-1-carboxamide (PIN)' (:5096)
    ("OC(=O)c1ccc(cc1)C(=O)NN=C(C)C", "4-[2-(propan-2-ylidene)hydrazine-1-carbonyl]benzoic acid"),
    ("OC(=O)c1ccc(cc1)C(=O)NN=Cc1ccccc1", "4-(2-benzylidenehydrazine-1-carbonyl)benzoic acid"),
])
def test_only_a_hydrazine_is_a_hydrazinecarbonyl(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


def test_a_ring_second_nitrogen_keeps_its_carbamoyl_name():
    row = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True).name_tiered("OC(=O)c1ccc(cc1)C(=O)NN1CCCCC1")
    assert row["name"] == "4-[(piperidin-1-yl)carbamoyl]benzene-1-carboxylic acid", row


@pytest.mark.parametrize("smiles", [
    "Oc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])C(=O)NN=Cc1ccc(o1)[N+](=O)[O-]",
    "Oc1ccc(Cl)cc1C(=O)NN=Cc1ccco1",
])
def test_a_phenol_name_over_a_hydrazide_is_not_the_pin(smiles):
    """ (the Blue Book): '12 Hydrazides' outrank '17 Hydroxy compounds', so '...-2-{2-[...]
    hydrazine-1-carbonyl}phenol' is not the PIN (``checks_parent.principal_class_check``; a nitro group drawn
    charge-separated no longer hides a molecule from that check)."""
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


def test_a_ketone_name_over_an_n_nitroso_amide_is_not_the_pin():
    """An N-nitroso amide is an amide:18186, '11 Amides' before '16 Ketones'), so
    '1-[4-(nitrosocarbamoyl)phenyl]ethan-1-one' is not the PIN, as '4-acetylbenzamide (PIN)' is named on the amide;
    it shipped pin_verified before because the class check skipped every N-N amide."""
    row = Orthonym().name_tiered("CC(=O)c1ccc(cc1)C(=O)NN=O")
    assert row["tier"] != "pin_verified", row


@pytest.mark.parametrize("smiles, name", [
    ("N(N)C(NN)=NN", "hydrazinecarbohydrazonohydrazide"),
    ("NNC(=N)NN", "hydrazinecarboximidohydrazide"),
    ("CNC(=N)NC", "N,N'-dimethylguanidine"),
])
def test_a_carbonic_hydrazide_is_no_guanidine(smiles, name):
    """The hydrazide class of the check must not see the carbon of a carbonic hydrazide as a guanidine (gold rows
    W2C-C2-, W2E-P1AM-02)."""
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
