"""Peptides with acidic side chains at the PIN tier: the systematic substitutive name.

 (the Blue Book) gives the peptide names ('aspartylglycylglycine') no PIN, so the PIN is the substitutive
name: the chain with the most carboxylic acid groups, then the longest, is the parent,; an amide on
a terminal chain atom is 'amino' + 'oxo',:32920); an N-acyl amino group is an acylamino prefix
. A hydroxamic acid is an N-hydroxy amide,:30144) and counts with the amides."""
import pytest

from orthonym.cli import _tiered_namer

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("N[C@@H](CC(=O)O)C(=O)NCC(=O)NCC(=O)O",
     "(3S)-3-amino-4-({2-[(carboxymethyl)amino]-2-oxoethyl}amino)-4-oxobutanoic acid"),
    ("N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)O",
     "(4S)-4-amino-5-{[(1R)-1-carboxy-2-sulfanylethyl]amino}-5-oxopentanoic acid"),
    # a middle residue: its N-acyl group is an acylamino prefix
    ("NCC(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)O",
     "(3S)-3-(2-aminoacetamido)-4-[(carboxymethyl)amino]-4-oxobutanoic acid"),
    ("OC(=O)CCC(=O)NCC(=O)N", "4-[(2-amino-2-oxoethyl)amino]-4-oxobutanoic acid"),
    # hydroxamic acids beside an amide, and two of them
    ("ONC(=O)CCC(=O)NC", "N1-hydroxy-N4-methylbutanediamide"),
    ("ONC(=O)CCC(=O)NO", "N1,N4-dihydroxybutanediamide"),
    ("CON(C)C(C)=O", "N-methoxy-N-methylacetamide"),
    # the carbon of an N-substituent bears no amide group of the parent chain
    ("CNC(=O)C(C)NC(=O)CCC(N)=O", "N1-[1-(methylamino)-1-oxopropan-2-yl]butanediamide"),
])
def test_systematic_pins(smiles, name):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_stereo_diamide_keeps_its_pin():
    row = _tiered_namer("pin", "pin").name_tiered("NC(=O)[C@@H](O)[C@H](O)C(N)=O")
    assert (row["name"], row["tier"]) == ("(2S,3S)-2,3-dihydroxybutanediamide", "pin_verified"), row


@pytest.mark.parametrize("smiles,name", [
    ("O=C(CCCCCCNC(=O)c1cnc(N(c2ccccc2)c2ccccc2)nc1)NO",
     "2-(diphenylamino)-N-[7-(hydroxyamino)-7-oxoheptyl]pyrimidine-5-carboxamide"),
    # a carbamate ester is class 9 and ranks above every amide (class 11), the hydroxamic acid among them: the ester is
    # the parent, the amides of the chain are prefixes of its organyl group (the next test)
    ("CC(C)C[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)OCc1ccccc1)C(=O)NO",
     "benzyl [(2S)-1-{[(2S)-1-(hydroxyamino)-4-methyl-1-oxopentan-2-yl]amino}-3-(4-hydroxyphenyl)-1-oxopropan-2-yl]"
     "carbamate"),
])
def test_a_hydroxamic_acid_behind_an_amide_nitrogen_stays_in_its_prefix(smiles, name):
    """ (the Blue Book): a hydroxamic acid that shares no carbon parent with the amide stands in the
    amide's N-substituent, and that prefix names it; the amide keeps its suffix,:32920) where no group of a
    senior class is present, and with a carbamate ester the ester is the parent,:18182 and:18184).

    change-asserted-value (lane/l17 fix a performance pass), second row: '(2S)-2-{[(benzyloxy)carbonyl]amino}-N-[(2S)-1-(hydroxyamino)-4-methyl-
    1-oxopentan-2-yl]-3-(4-hydroxyphenyl)propanamide' -> 'benzyl [(2S)-1-{[(2S)-1-(hydroxyamino)-4-methyl-1-oxopentan-2-yl]amino}-3-
    (4-hydroxyphenyl)-1-oxopropan-2-yl]carbamate'. Quote: SENIORITY ORDER FOR CLASSES, Table 4.1 (:18158): "9 Esters
    (functional class names are given to noncyclic esters;...)" ranks above "11 Amides" (:18184), and the carbamate is an ester
    of carbamic acid,:30766: '2-hydroxypropyl (2-aminoethyl)carbamate (PIN)', the ester named by its organyl group
    with the junior amine in a prefix); (:31698): an ester group is cited by a prefix only "when... another group
    is present that has priority for citation as the principal group", which an amide has not. The hydroxamic acid stays in
    its prefix, as '(hydroxyamino)...oxo' of the organyl group,:32920). Independent check: a fresh OPSIN call
    reads the new name to the input's full InChIKey (``test_the_carbamate_ester_name_of_a_z_dipeptide_hydroxamate``). Mutation:
    with the carbamate ester not ranked above the amides (``seniority.ESTER_ACID_ORDER`` without 7b) the amide parent is named
    and this row fails."""
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == name, row


def test_the_carbamate_ester_name_of_a_z_dipeptide_hydroxamate_reads_back():
    from tests.support.rt_assert import assert_full_rt
    smiles = "CC(C)C[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)OCc1ccccc1)C(=O)NO"
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["tier"] == "pin_unverified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)
