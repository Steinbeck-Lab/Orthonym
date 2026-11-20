"""An -O-CH(aryl)2 ether whose aryl groups are not two plain phenyls is no 'benzyloxy' (C4 lead, 2026-11-20).

``assembly.substituent_naming._name_aryl_methyl_ether`` ended with ``return "benzyloxy"`` for every case it did not
handle: two aromatic neighbours that are not two plain phenyls (the ring carbon of
10,11-dihydro-5H-dibenzo[a,d][7]annulene, of fluorene, a di(pyridinyl)methyl). 'benzyloxy' names one phenyl
, so the prefix dropped atoms: deptropine (CHEBI:50189) got '3-(benzyloxy)-8-methyl-8-azabicyclo[3.2.1]octane',
which the gate refused, and the molecule got no name. Such a case is now no clean aryl-methyl ether (None), and the
other producers name it."""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import tier_row


@pytest.mark.parametrize("smiles,central,oxygen", [
    ("COC1c2ccccc2CCc2ccccc12", 2, 1),       # dibenzosuberyl ether
    ("COC1c2ccccc2-c2ccccc12", 2, 1),        # fluoren-9-yl ether
    ("COC(c1ccccn1)c1ccccn1", 2, 1),         # di(pyridin-2-yl)methyl ether
])
def test_two_aryl_neighbours_that_are_not_two_phenyls_are_no_benzyloxy(smiles, central, oxygen):
    from orthonym.assembly.substituent_naming import _name_aryl_methyl_ether
    assert _name_aryl_methyl_ether(Chem.MolFromSmiles(smiles), central, oxygen) is None


@pytest.mark.parametrize("smiles,central,oxygen,prefix", [
    ("COCc1ccccc1", 2, 1, "benzyloxy"),
    ("COC(c1ccccc1)c1ccccc1", 2, 1, "diphenylmethoxy"),
])
def test_the_clean_cases_keep_their_prefix(smiles, central, oxygen, prefix):
    from orthonym.assembly.substituent_naming import _name_aryl_methyl_ether
    assert _name_aryl_methyl_ether(Chem.MolFromSmiles(smiles), central, oxygen) == prefix


@pytest.mark.opsin_gate
def test_deptropine_is_named_at_best_effort():
    smiles = "CN1[C@@H]2CC[C@H]1C[C@@H](OC1c3ccccc3CCc3ccccc31)C2"
    row = tier_row(smiles, "best-effort")
    assert row["name"] and row["tier"] != "abstain" and "benzyloxy" not in row["name"], row
    # its '3r' is a block OPSIN cannot read; the stereo-free name reads back to deptropine's constitution
    from orthonym.rules.stereochemistry import strip_stereo
    from orthonym.validation.opsin_roundtrip import _extended_smiles
    read = Chem.MolFromSmiles(_extended_smiles(strip_stereo(row["name"])).split(" ")[0])
    flat = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(flat)
    assert Chem.MolToInchiKey(read) == Chem.MolToInchiKey(flat)
