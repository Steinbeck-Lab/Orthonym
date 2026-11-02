"""A chain substituent with four or more heterounits is named by skeletal replacement ('a') nomenclature.

 (the Blue Book) "Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative
names must be used to generate preferred IUPAC names for acyclic structures when four or more heterounits are
present in a unbranched chain"; the heteroatoms fix the numbering and the free valence follows it,
'2,5,8,11-tetraoxatridec-12-en-13-yl (preferred prefix) (not 3,6,9,12-tetraoxatridec-1-en-1-yl...)' (:24543).
"""
import pytest
from rdkit import Chem

from orthonym.rules.replacement_chain_prefix import replacement_chain_prefix
from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.unit


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("COCCOCCOCCOCCOc1ccc(C(=O)O)cc1", "4-[(2,5,8,11-tetraoxatridecan-13-yl)oxy]benzoic acid"),
    ("COCCOCCOCCOCCOC(=O)c1ccccc1", "2,5,8,11-tetraoxatridecan-13-yl benzoate"),
    # three heterounits: substitutive
    ("COCCOCCOCCOC(=O)c1ccccc1", "2-[2-(2-methoxyethoxy)ethoxy]ethyl benzoate"),
])
def test_names(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


def test_the_heteroatoms_are_numbered_first_then_the_free_valence():
    m = Chem.MolFromSmiles("COCCSCCOCCOCCc1ccncc1")      # a pyridine N outranks the chain O
    assert replacement_chain_prefix(m, list(range(13)), 12) == "2,8,11-trioxa-5-thiatridecan-13-yl"
    m = Chem.MolFromSmiles("COCCOCCOOCCOCCc1ccncc1")      # a peroxy (two adjacent heteroatoms): not built
    assert replacement_chain_prefix(m, list(range(14)), 13) is None


@pytest.mark.opsin_gate
def test_a_chain_with_a_senior_atom_is_not_a_prefix_of_a_carbocycle():
    """Review fix: (the Blue Book) "The senior parent structure, whether cyclic or acyclic, has the
    senior atom": with no principal group, a cyclohexane is not the parent of a selenium chain, so no
    '(2,5,8,11-tetraselenatridecan-13-yl)cyclohexane' is labelled a PIN."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered("C[Se]CC[Se]CC[Se]CC[Se]CCC1CCCCC1")
    assert "tetraselenatridecan-13-yl" not in (row["name"] or ""), row
