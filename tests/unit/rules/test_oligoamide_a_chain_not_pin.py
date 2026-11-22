"""An amide nitrogen is a skeletal atom of an 'a' chain when a group senior to amides is the principal group.

## **** Skeletal replacement ('a') nomenclature in acyclic chains (the Blue Book), (:23348):
"Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to generate
preferred IUPAC names for acyclic structures when four or more heterounits are present in a unbranched chain
containing at least one carbon atom and when none of the heteroatoms constitute all or part of the principal
characteristic group of the compound." (:23405,:23428): 'methyl 7,14,21,28-tetraacetamido-2,9,16,23-
tetraoxo-3,10,17,24-tetraazatriacontan-30-oate (PIN; an ester is senior to an amide or a ketone)'. Orthonym builds no
such 'a' name; the substitutive name was labelled pin_verified and is now below the PIN (the reader,
validation/spelling/checks_skeletal_chain.py). Three amide nitrogens, or amides that are the principal group, or a
urea or carbamate nitrogen, leave the rule out.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_skeletal_chain import a_chain_applies
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit]

BOOK_ESTER = "COC(=O)CC(CCCNC(=O)CC(CCCNC(=O)CC(CCCNC(=O)CC(CCCNC(C)=O)NC(C)=O)NC(C)=O)NC(C)=O)NC(C)=O"


@pytest.mark.parametrize("smiles", [
    BOOK_ESTER,                                                              # the book's own molecule
    "CC(C)[C@@H](C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)OC)NC(=O)C",   # 4 amide N, ester
    "NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O",                                 # 4 amide N, acid
    "NCCCOCCOCCOCCCNC(=O)CCC(=O)O",                                          # 3 ethers + 1 amide N, acid
])
def test_amide_chain_applies(smiles):
    assert a_chain_applies(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("smiles", [
    "NCC(=O)NCC(=O)NCC(=O)NCC(=O)O",              # three amide nitrogens only
    "CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(N)=O",        # amides are the principal group
    "CNC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)O",       # the first N is a urea nitrogen, three amide N remain
    "CC(C)(C)OC(=O)NCC(=O)NCC(=O)NCC(=O)OC",      # a carbamate nitrogen is not a plain amide nitrogen
    # two acid groups at the chain end: the diacid is the parent, the Blue Book) and its junction N
    # or O is no chain atom,:23430), so three units remain
    "CC(=O)NCC(=O)NCC(=O)NCC(=O)NC(CC(=O)O)C(=O)O",
    "CC(=O)NCC(=O)NCC(=O)NCC(=O)NC(CCC(=O)O)C(=O)O",
    "CC(=O)NCC(=O)NCC(=O)NCC(=O)NC(C(=O)O)C(=O)O",
    "COCCOCCOCCC(=O)NC(CC(=O)O)C(=O)O",
    "COCCOCCOCCOC(CC(=O)O)C(=O)O",
])
def test_amide_chain_does_not_apply(smiles):
    assert not a_chain_applies(Chem.MolFromSmiles(smiles))


@pytest.mark.opsin_gate
def test_n_acyl_aspartic_acid_keeps_the_pin():
    row = tier_row("CC(=O)NCC(=O)NCC(=O)NCC(=O)NC(CC(=O)O)C(=O)O", "pin")
    assert (row["name"], row["tier"]) == ("2-{2-[2-(2-acetamidoacetamido)acetamido]acetamido}butanedioic acid",
                                          "pin_verified"), row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    # a side chain outside the vocabulary of the 'a' writer (butan-2-yl): no PIN is built (the rows it builds are in
    # tests/unit/rules/test_oligoamide_a_name.py)
    "CC(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)OC",
])
def test_oligoamide_substitutive_name_is_below_the_pin(smiles):
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    row = tier_row(smiles, "best-effort")
    assert row["tier"] == "systematic_verified", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.opsin_gate
def test_three_amide_nitrogens_keep_the_pin():
    row = tier_row("NCC(=O)NCC(=O)NCC(=O)NCC(=O)O", "pin")
    assert (row["name"], row["tier"]) == ("{2-[2-(2-aminoacetamido)acetamido]acetamido}acetic acid",
                                          "pin_verified"), row


@pytest.mark.parametrize("smiles,applies", [
    # a ring as a prefix leaves the 'a' PIN of the chain ('4-cyclohexyl-8-methyl-4-oxo-3,5-dioxa-8-thia-
    # 4λ5-phosphatetradecan-8-ium iodide (PIN)',, the Blue Book); a ring nitrogen (proline) ends a chain
    ("CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)Oc1ccccc1", True),
    ("C[C@H]([C@@H](C(=O)N[C@@H](CO)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H]([C@@H](C)O)C(=O)O)"
     "NC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(C)C)N)O", True),
    ("CC(C)CC(C(=O)O)NC(=O)C(CCCCN)NC(=O)C(CCCCN)NC(=O)C(CCSC)N", True),     # -S- and three amide N: four units
    # the acid on a ring, or an acid that no chain of four units can end with, leaves the rule out
    ("O=C(O)c1ccc(NC(=O)CNC(=O)CNC(=O)CNC(=O)C)cc1", False),
    ("CC(C)C(C(=O)NC(CCSC)C(=O)O)NC(=O)C(CCCCN)NC(=O)C(C)N", False),
    ("OC(=O)CCOCCOCCOCCOc1ccccc1", False),        # a ring beside a chain without amide nitrogens is not read
])
def test_ring_bearing_chain(smiles, applies):
    assert a_chain_applies(Chem.MolFromSmiles(smiles)) is applies


@pytest.mark.parametrize("smiles", [
    # a lactone is a pseudoketone below the amides class 9, the Blue Book): the amide stays principal
    "CC(=O)NCCOCCOCCOCC1CCOC1=O",
    "CC(=O)NCCOCCOCCOCCC1CCC(=O)O1",
    # an acid on a ring branch is not read as a chain-end group may make the ring the parent)
    "CC(=O)NCC(=O)NCC(=O)NCC(=O)NCc1ccc(CC(=O)O)cc1",
])
def test_lactone_or_ring_acid_is_not_read(smiles):
    assert not a_chain_applies(Chem.MolFromSmiles(smiles))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    # no name of these is the PIN; (:50943) leaves the choice between the two others open. A chain
    # of retained residues takes the peptide name,:54698), a capped or non-standard one the substitutive
    # name,:54251) -- the Blue Book reading the user asked for, 2026-11-22
    # (side chains outside the vocabulary of the 'a' writer, so no PIN is built: a butan-2-yl and a carbamoylethyl)
    ("CC[C@H](C)[C@@H](C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCC(=O)N)C(=O)O)NC(=O)CN", "glycylisoleucylalanylglycylglutamine"),
    ("CC(=O)N[C@@H]([C@@H](C)CC)C(=O)NCC(=O)NCC(=O)NCC(=O)O",
     "(2-{2-[(2S,3S)-2-acetamido-3-methylpentanamido]acetamido}acetamido)acetic acid"),
])
def test_which_non_pin_name_a_long_peptide_takes(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
    assert_full_rt(row["name"], smiles)
