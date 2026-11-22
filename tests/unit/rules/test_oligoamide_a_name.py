"""The skeletal replacement ('a') PIN of an acid or ester on a chain of four or more skeletal amide nitrogens.

## **** Skeletal replacement ('a') nomenclature in acyclic chains (the Blue Book), (:23405,:23428):
'methyl 7,14,21,28-tetraacetamido-2,9,16,23-tetraoxo-3,10,17,24-tetraazatriacontan-30-oate (PIN; an ester is senior to an
amide or a ketone)'. The numbering is fixed by the heteroatoms first,:6476), then the suffix
,:6496); the acid suffix keeps its locant on such a chain,:29860). User decision 2026-11-22: the
'a' name of a peptide of standard amino acids is its PIN too over. rules/skeletal_chain.name_acyl_a_chain.
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("COC(=O)CC(CCCNC(=O)CC(CCCNC(=O)CC(CCCNC(=O)CC(CCCNC(C)=O)NC(C)=O)NC(C)=O)NC(C)=O)NC(C)=O",
     "methyl 7,14,21,28-tetraacetamido-2,9,16,23-tetraoxo-3,10,17,24-tetraazatriacontan-30-oate"),     # the book row
    ("NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O", "14-amino-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O", "4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("NCCCOCCOCCOCCCNC(=O)CCC(=O)O", "1-amino-15-oxo-4,7,10-trioxa-14-azaoctadecan-18-oic acid"),
    ("OC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CC(=O)O", "4,7,10,13-tetraoxo-3,6,9,12-tetraazapentadecane-1,15-dioic acid"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_oligoamide_a_pin(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_three_amide_nitrogens_keep_the_substitutive_pin():
    row = tier_row("NCC(=O)NCC(=O)NCC(=O)NCC(=O)O", "pin")
    assert (row["name"], row["tier"]) == ("{2-[2-(2-aminoacetamido)acetamido]acetamido}acetic acid", "pin_verified")


@pytest.mark.parametrize("smiles,name", [
    # a diester multiplies its alkyl word as the book does: 'di-tert-butyl' (:25717), 'di(propan-2-yl)' (:36921)
    ("CC(C)OC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CC(=O)OC(C)C",
     "di(propan-2-yl) 4,7,10,13-tetraoxo-3,6,9,12-tetraazapentadecane-1,15-dioate"),
    ("CC(C)(C)OC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CC(=O)OC(C)(C)C",
     "di-tert-butyl 4,7,10,13-tetraoxo-3,6,9,12-tetraazapentadecane-1,15-dioate"),
    # a methyl on a chain carbon is a prefix
    ("NC(C)C(=O)NCC(=O)NCC(=O)NCC(=O)NC(C)C(=O)O", "14-amino-2-methyl-4,7,10,13-tetraoxo-3,6,9,12-tetraazapentadecan-1-oic acid"),
])
def test_oligoamide_a_pin_prefixes_and_esters(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_unsaturated_or_urea_chain_is_not_written_here():
    from rdkit import Chem
    from orthonym.rules.skeletal_chain import name_acyl_a_chain
    # ene/yne terms are not built,:6502); a urea nitrogen is no plain amide one; neither raises
    assert name_acyl_a_chain(Chem.MolFromSmiles("OC(=O)C=CCNC(=O)CNC(=O)CNC(=O)CNC(C)=O")) is None
    assert name_acyl_a_chain(Chem.MolFromSmiles("COCCNC(=O)NCC(=O)O")) is None


@pytest.mark.parametrize("smiles,name", [
    # the CIP descriptors with their locants in front of the name, the Blue Book), of the anion of an ester
    ("C[C@H](NC(=O)CN)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N[C@H](C)C(=O)O",
     "(2R,5R,8S,11S)-14-amino-2,5,8,11-tetramethyl-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("CC(C)[C@@H](C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)OC)NC(=O)C",
     "methyl (2S,5S,8S,11S)-4,7,10,13-tetraoxo-2,5,8,11-tetra(propan-2-yl)-3,6,9,12-tetraazatetradecan-1-oate"),
])
def test_oligoamide_a_pin_with_stereo(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_stereo_tie_takes_the_lower_locant_for_r():
    # both orientations tie on every other criterion: (j) (the Blue Book) gives the lower locant to R
    smiles = "OC(=O)[C@H](C)NC(=O)CNCCNCC(=O)N[C@H](C)C(=O)O"
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (
        "(2R,13S)-2,13-dimethyl-4,11-dioxo-3,6,9,12-tetraazatetradecane-1,14-dioic acid", "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # side chains named by the shared substituent namer, a ring among them (a ring as a prefix keeps the 'a' PIN of the
    # chain: '4-cyclohexyl-...-3,5-dioxa-8-thia-4λ5-phosphatetradecan-8-ium iodide (PIN)',, the Blue Book)
    ("NCC(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-benzyl-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("NCC(=O)N[C@@H](CO)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-(hydroxymethyl)-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
    ("NCC(=O)N[C@@H](CCCCN)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(11S)-14-amino-11-(4-aminobutyl)-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid"),
])
def test_oligoamide_a_pin_with_side_chains(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_ring_parts_are_not_alkyl_and_ring_e_z_declines():
    from rdkit import Chem
    from orthonym.rules.skeletal_chain import name_acyl_a_chain
    # a ring is no alkyl group: 'cyclopropyl', never 'propan-2-yl'
    assert name_acyl_a_chain(Chem.MolFromSmiles("NCC(=O)NC(C1CC1)C(=O)NCC(=O)NCC(=O)NCC(=O)O")) == (
        "14-amino-11-cyclopropyl-4,7,10,13-tetraoxo-3,6,9,12-tetraazatetradecan-1-oic acid")
    assert name_acyl_a_chain(Chem.MolFromSmiles("NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)OC1CC1")) is None
    # an E/Z double bond, in a ring prefix too, is not cited by this writer
    assert name_acyl_a_chain(Chem.MolFromSmiles("NCC(=O)NC(C1CCC/C=C/CCC1)C(=O)NCC(=O)NCC(=O)NCC(=O)O")) is None
