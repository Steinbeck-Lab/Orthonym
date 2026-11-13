"""Three defects of the names the decomposition engine builds from its fragments.

- A glycoside name cites its sugars as 'glycosyloxy' prefixes: a fragment that is no sugar ('water',
  'methanol') is no such prefix.
- A split name with a part that is never a PIN (a demoted N-acyl float, 'N-acetyl-3-amino...oxane') does not replace
  the pipeline's acceptable name that has none ('N-[...oxan-3-yl]acetamide', the amide as the suffix,.
- A fragment of a lipid acyl group is recorded below the PIN only when the general engine named it, not when the
  outer naming's source was the general engine ('acetic acid', and with it every 'acetyloxy').
"""
import pytest

from orthonym.decomposition import engine
from orthonym.decomposition import fragment_assembly as fa
from orthonym.metrics import provenance as pv

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_provenance():
    pv.clear_provenance()
    yield
    pv.clear_provenance()


def test_a_glycoside_cites_no_fragment_that_is_no_sugar():
    core = ({"smiles": "OC1CCCCC1", "side": "middle"}, "cyclohexanol")
    sugar = ({"smiles": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "side": "acid"}, "β-D-glucopyranosyloxy")
    water = ({"smiles": "O", "side": "alkyl"}, "water")
    assert fa._assemble_multi_glycoside([sugar, core, water]) is None
    assert fa._assemble_multi_glycoside([sugar, core, sugar]) is not None


def test_a_split_name_with_a_non_pin_part_does_not_replace_a_clean_pipeline_name(clean_provenance):
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC(=O)NC1CCOCC1")
    existing, split = "N-(oxan-4-yl)acetamide", "N-acetyloxan-4-amine"
    assert not engine._existing_is_preferred_to(existing, split, mol)
    pv.record_non_pin_fragment("N-acetyl")
    assert engine._existing_is_preferred_to(existing, split, mol)


def test_an_acid_named_without_a_producer_record_is_no_non_pin_fragment(clean_provenance):
    from orthonym.rules import lipids
    pv.record_source("general_engine")
    assert lipids._name_fragment("CC(=O)O") == "acetic acid"
    assert "acetic acid" not in pv.get_provenance().get("non_pin_fragments", ())
    assert pv.get_provenance().get("source") == "general_engine"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O[C@@H]2O[C@@H](C)[C@H](O)"
     "[C@@H](O)[C@H]2O)[C@H]1NC(C)=O",
     "N-[(2R,3R,4R,5S,6R)-6-(hydroxymethyl)-2-methoxy-5-{[(2S,3R,4S,5R,6R)-3,4,5-trihydroxy-6-(hydroxymethyl)oxan-2-yl]"
     "oxy}-4-{[(2S,3R,4R,5R,6S)-3,4,5-trihydroxy-6-methyloxan-2-yl]oxy}oxan-3-yl]acetamide"),
    ("CC(=O)OCCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C[C@H](OC(C)=O)[C@@H]4C[C@H](OC(C)=O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "(3R,5R,6S,8S,9S,10R,13R,14S,17R)-6-(acetyloxy)-17-[(2R)-5-(acetyloxy)pentan-2-yl]-10,13-dimethylhexadecahydro-"
     "1H-cyclopenta[a]phenanthren-3-yl acetate"),
])
def test_the_best_effort_tier_names_them_with_a_pin(smiles, name):
    from tests.support.pin_tiers import tier_row
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_verified")
