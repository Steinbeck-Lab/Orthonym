"""A chlorine or bromine with a nonstandard bonding number, named through its iodine drawing
(``assembly/halogen_drawing.py``).

RDKit reads no SMILES with a λ3-chlorane or a two-bond chloranuide, so these inputs used to raise
'Invalid SMILES'. "Nonstandard bonding numbers" (the Blue Book) names the atom with its λ;
 (:6446) keeps the halogens together in the seniority order, so the iodine drawing has the parent and
the numbering of the input. The name is systematic_verified (another drawing never gives the PIN) and reads back
to the input's full standard InChIKey.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.assembly.halogen_drawing import (
    halogen_drawing,
    relaxed_mol,
    stem_may_change_the_spelling,
)
from orthonym.errors import OrthonymLimitError
from orthonym.jvm_bridge import opsin_stdout
from orthonym.jvm_budget import jvm_slots
from tests.support.jars import jar_or_skip

_BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


def _reads_back(name, smiles):
    """OPSIN's parse of ``name`` has the full standard InChIKey of ``smiles`` (both read without the valence check)."""
    txt, _ = opsin_stdout(name, allow_radicals=False, jar_path=jar_or_skip())
    parsed = relaxed_mol((txt or "").strip().split("\n")[0].strip())
    return parsed is not None and Chem.MolToInchiKey(parsed) == Chem.MolToInchiKey(relaxed_mol(smiles))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)Cl=C", "tert-butyl(methylidene)-λ3-chlorane"),
    ("C=Clc1ccc(F)cc1", "1-fluoro-4-(methylidene-λ3-chloranyl)benzene"),
    ("FC1=CC(=CC=C1Br=C)C", "2-fluoro-4-methyl-1-(methylidene-λ3-bromanyl)benzene"),
])
def test_a_hypervalent_halogen_is_named_through_its_iodine_drawing(smiles, expected):
    assert Chem.MolFromSmiles(smiles) is None          # RDKit reads none of them
    with jvm_slots(1, purpose="test-halogen-drawing"):
        row = Orthonym(**_BEST_EFFORT).name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert _reads_back(expected, smiles)


@pytest.mark.opsin_gate
def test_the_default_tier_declines_the_name_of_another_drawing():
    with jvm_slots(1, purpose="test-halogen-drawing"):
        row = Orthonym().name_tiered("CC(C)(C)Cl=C")
        assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN"
        assert name_compound("CC(C)(C)Cl=C", emit_tier="pin") == ""
        assert name_compound("CC(C)(C)Cl=C") == "tert-butyl(methylidene)-λ3-chlorane"


@pytest.mark.parametrize("smiles", [
    "CCO",                          # RDKit reads it: named as before
    "C=Ic1ccccc1",                  # RDKit reads the iodane itself
    "C=Clc1ccc(I)cc1",              # an iodine of its own: an iodine stem could not be told apart
    "C=Clc1ccc(Br=C)cc1",           # a chlorine and a bromine: one stem each, not told apart
])
def test_no_iodine_drawing(smiles):
    assert halogen_drawing(smiles) is None


def test_the_iodine_drawing_keeps_the_hydrogens_of_the_input():
    drawing = halogen_drawing("CC(C)(C)Cl=C")
    iodine = Chem.MolFromSmiles(drawing.iodine_smiles)
    assert next(a for a in iodine.GetAtoms() if a.GetSymbol() == "I").GetTotalNumHs() == 0
    assert drawing.stem == "chlor"


@pytest.mark.parametrize("iodine_name,changes", [
    ("1-fluoro-4-(methylidene-λ3-iodanyl)benzene", False),   # no other part begins with 'methylidene'
    ("tert-butyl(methylidene)-λ3-iodane", False),             # the parent
    ("ethyl(4-fluorophenyl)iodanuide", False),                # the parent, after the 'yl' of a substituent
    ("7λ3-ioda-2-azatricyclo[8.4.0.0^3,8]tetradeca-1(14),2,4,6,8,10,12-heptaene", False),  # parent 'a' prefix
    ("2λ3,3λ3-heptaiodatetracyclo[7.6.1]hexadecane", False),  # after a multiplying prefix, in the parent
    #: 'chloryl' and 'bromyl' sort before 'chloro', 'fluoro', 'methyl' (and (g) then numbers them)
    ("2,4-diiodylbenzoic acid", True),
    ("1-chloro-2-iodyl-3-methylbenzene", True),
    ("N-ethyl-N-iodylmethanamine", True),
    ("5-bromo-2-(1λ3-iodacyclohex-2-en-1-yl)pyridine", True),
    # 'difluoro-λ3-chloranyl' sorts before 'difluoroethyl'; 'difluoro-λ3-iodanyl' after it
    ("1-(2,2-difluoroethyl)-3-(difluoro-λ3-iodanyl)benzene", True),
    #: 'benzo' keeps its 'o' before 'chlor' ('benzochloroxole'), not before 'iod' ('benziodoxole')
    ("1-hydroxy-1λ3,2-benziodoxol-3(1H)-one", True),
])
def test_stem_may_change_the_spelling(iodine_name, changes):
    assert stem_may_change_the_spelling(iodine_name) is changes


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "Cc1cccc(Cl)c1Br(=O)=O",          # 1-bromyl-2-chloro-3-methylbenzene, not the iodine drawing's order
    "FCl(F)c1cccc(CC(F)F)c1",         # the difluoro-λ3-chloranyl prefix sorts before difluoroethyl
    "O=C1OCl(O)c2ccccc21",            # benzochloroxol, not benzchloroxol
])
def test_a_name_whose_spelling_the_element_may_change_is_declined(smiles):
    with jvm_slots(1, purpose="test-halogen-drawing"):
        row = Orthonym(**_BEST_EFFORT).name_tiered(smiles)
        assert row["tier"] == "abstain" and row["limit_code"] == "UNNAMEABLE", row
        assert Orthonym(**_BEST_EFFORT).name(smiles) == ""
        with pytest.raises(OrthonymLimitError):
            Orthonym(**_BEST_EFFORT).name(smiles, raise_on_limit=True)


@pytest.mark.parametrize("smiles", ["Cl1CC1", "C[Cl+16](C)C", "C(C"])
def test_an_input_no_iodine_drawing_holds_still_raises(smiles):
    # a ring chlorine with two bonds: iodine would carry a radical electron the input does not
    assert halogen_drawing(smiles) is None
    with pytest.raises(ValueError):
        Orthonym(**_BEST_EFFORT).name(smiles)


@pytest.mark.opsin_gate
def test_every_entry_point_names_it():
    with jvm_slots(1, purpose="test-halogen-drawing"):
        namer = Orthonym(**_BEST_EFFORT)
        assert namer.name("C=Clc1ccc(F)cc1") == "1-fluoro-4-(methylidene-λ3-chloranyl)benzene"
        record = namer.name_with_confidence("C=Clc1ccc(F)cc1")
        assert record["name"] == "1-fluoro-4-(methylidene-λ3-chloranyl)benzene"
        assert record["tier"] == "systematic_verified" and record["handler"] == "iodine_drawing"
        row = namer.name_tiered("C=Clc1ccc(F)cc1")
        assert row["iodine_name"] == "1-fluoro-4-(methylidene-λ3-iodanyl)benzene"
        with pytest.raises(OrthonymLimitError):
            Orthonym().name("CC(C)(C)Cl=C", raise_on_limit=True)
