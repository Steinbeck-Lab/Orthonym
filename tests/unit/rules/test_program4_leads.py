"""Program 4 leads (the four naming leads of program 3, 2026-11-03)."""
import pytest

from orthonym.cli import _tiered_namer

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier="pin"):
    return _tiered_namer("pin", tier).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book) "method (2) ['carbamoyl'] for rings and ring systems"; (2) (:29451)
    ("O=C(O)CCC(=O)N1CCCC1C(=O)NC", "4-[2-(methylcarbamoyl)pyrrolidin-1-yl]-4-oxobutanoic acid"),
    ("N[C@@H](CCC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CS)C(=O)O",
     "(4S)-4-amino-5-[(2S)-2-{[(1R)-1-carboxy-2-sulfanylethyl]carbamoyl}pyrrolidin-1-yl]-5-oxopentanoic acid"),
])
def test_an_n_substituted_carboxamide_on_a_ring_is_a_carbamoyl_prefix(smiles, name):
    row = _row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_carboxy_group_behind_an_ester_oxygen_keeps_the_phosphate_ester_name():
    # the carboxy rule of the amide branch (a carboxy group behind a nitrogen is the branch's) stops at N only; the
    # glycerophospholipid keeps its phosphate ester name at the wider tier (it lost it to a replacement name)
    smi = r"CCCCCCCC/C=C\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCC(=O)/C=C/C(=O)O"
    row = _row(smi, "best-effort")
    assert row["name"].endswith("propyl 2-(trimethylazaniumyl)ethyl phosphate"), row


def test_a_wider_tier_names_a_polyanhydride_with_an_anhydride_name_opsin_reads():
    # the PIN tier keeps the dianhydride PIN (read through its acids); a wider tier spells the anhydride of
    # one class word, which OPSIN reads, labelled below the PIN, not the general engine's '1,4-bis(acetyloxy)-...'
    smi = "CC(=O)OC(=O)CCC(=O)OC(C)=O"
    assert _row(smi)["name"] == "diacetic butanedioic dianhydride"
    row = _row(smi, "best-effort")
    assert (row["name"], row["tier"]) == ("acetic 4-(acetyloxy)-4-oxobutanoic anhydride", "systematic_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # (1) (the Blue Book): an alkyl prefix has its free valence at 1; (:44624) cites the
    # locant of a stereodescriptor ('[(1R)-1-chloropropyl]benzene (PIN)',:44668)
    ("CC(C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",
     "(4S)-5-{[(1S)-1-carboxy-2-methylpropyl]amino}-4-[(2S)-2,5-diamino-5-oxopentanamido]-5-oxopentanoic acid"),
])
def test_a_stereocentre_at_the_free_valence_of_an_alkyl_prefix_cites_locant_one(smiles, name):
    row = _row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # (the Blue Book) method (2) 'carbamoyl' for rings, N-substituents without the italic N:
    # '5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid (PIN)' (:32957); before, the benzene handler dropped them
    # ('2-carbamoylbenzoic acid', a different molecule, withdrawn by the read-back) and the PIN tier declined
    ("Cc1ccc(C(=O)N(C)c2ccccc2)c(C(=O)O)c1", "5-methyl-2-[methyl(phenyl)carbamoyl]benzoic acid"),
    ("OC(=O)c1ccccc1C(=O)NC", "2-(methylcarbamoyl)benzoic acid"),
    ("OC(=O)c1ccc(C(=O)N(C)C)cc1", "4-(dimethylcarbamoyl)benzoic acid"),
    # the sulfamoyl analogue through the same assembler; 'dimethylsulfamoyl',:32982)
    ("OS(=O)(=O)c1ccccc1S(=O)(=O)N(C)CC", "2-[ethyl(methyl)sulfamoyl]benzene-1-sulfonic acid"),
])
def test_a_demoted_n_substituted_amide_keeps_its_n_substituents(smiles, name):
    row = _row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    # the same carbamoyl prefix on a heteroarene acid,:32920); before, the heterocycle handler declined
    ("OC(=O)c1ccc(nc1)C(=O)NC", "6-(methylcarbamoyl)pyridine-3-carboxylic acid"),
    ("OC(=O)c1ccc(s1)C(=O)N(C)C", "5-(dimethylcarbamoyl)thiophene-2-carboxylic acid"),
    ("OC(=O)c1ccc(nc1)C(=O)NO", "6-(hydroxycarbamoyl)pyridine-3-carboxylic acid"),
])
def test_an_n_substituted_carbamoyl_on_a_heteroarene_acid(smiles, name):
    row = _row(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
