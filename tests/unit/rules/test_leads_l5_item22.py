"""Leads program item 22 (lane L5): a ring that bears the principal ester keeps its numbering,
its hydro locants and its other ring groups.

Four writers each gave a name the validity gate rejected, so the PIN tier declined; the gate was
right every time (the names described other molecules).

1. ``data/retained_names.py`` carried fixed-locant rows for the two mono-ene dihydropyrans. The
   hydro locants of a ring depend on its numbering, and 'NUMBERING' (the Blue Book)
   numbers from the principal characteristic group (c,:3256) before hydro prefixes and 'ene'
   endings (e)(i,:3289): '5,6-dihydro-2H-pyran-3-carboxylic acid', not '3,6-dihydro-2H-pyran-3-
   carboxylic acid' (which OPSIN reads as O=C(O)C1C=CCOC1, a different molecule).
2. ``heterocycles.name_substituted_heterocycle`` dropped the ring group that is not the cited
   suffix when the ring had another prefix, and glued it to a hydro-locant parent without a hyphen.
    'SENIORITY ORDER FOR CLASSES' (:18158): one class is the suffix, the others are prefixes.
3. ``esters.get_ring_acid_name`` spelled every all-carbon ring 'cyclo<alk>anecarboxylic'; a ring
   with a double bond is not that parent. (:31663): all PIN esters are functional class
   names; 'methyl cyclohexanecarboxylate (PIN)' (:31671) is the saturated case and stays.
4. ``esters._build_ester_acid_word`` excluded the carbonyl oxygen of EVERY group on the acid side from
   the substituent walk, so a ring carbamoyl read as 'aminomethyl' and a formyl as 'methyl'.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate

PIN = Orthonym(style="pin")
BEST = Orthonym(style="pin", **_emit_tier_flags("best-effort"))

# (SMILES, PIN) -- every name is verified against OPSIN 2.9.0 by assert_full_rt (full InChIKey)
ROWS = [
    ("COC(=O)C1=CCCN(C)C1C(N)=O",
     "methyl 2-carbamoyl-1-methyl-1,2,5,6-tetrahydropyridine-3-carboxylate"),
    ("COC(=O)C1=CCCOC1C(N)=O",
     "methyl 2-carbamoyl-5,6-dihydro-2H-pyran-3-carboxylate"),
    ("COC(=O)C1=CCCOC1", "methyl 5,6-dihydro-2H-pyran-3-carboxylate"),
    ("OC(=O)C1=CCCOC1", "5,6-dihydro-2H-pyran-3-carboxylic acid"),
    ("COC(=O)C1=CCCCC1", "methyl cyclohex-1-ene-1-carboxylate"),
    ("COC(=O)C1=CCCC1", "methyl cyclopent-1-ene-1-carboxylate"),
    ("COC(=O)[C@@H]1C=CCCC1", "methyl (1S)-cyclohex-2-ene-1-carboxylate"),
    ("COC(=O)c1ccccc1C(N)=O", "methyl 2-carbamoylbenzoate"),
    ("COC(=O)c1ccccc1C=O", "methyl 2-formylbenzoate"),
    # the acid twins of the heterocycle rows: the writer that dropped the amide
    ("CN1CCC=C(C(=O)O)C1C(N)=O", "2-carbamoyl-1-methyl-1,2,5,6-tetrahydropyridine-3-carboxylic acid"),
    ("OC(=O)C1=CCCOC1C(N)=O", "2-carbamoyl-5,6-dihydro-2H-pyran-3-carboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_pin_tier_names_the_ring_ester(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_best_effort_agrees_with_the_pin_tier(smiles, expected):
    # the wider tier used to ship '(methoxycarbonyl)' / 'carboxamide' spellings for these rows
    row = BEST.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row


@pytest.mark.parametrize("smiles,expected", [
    # the unsubstituted rings are still named, now by the generic hydro path
    ("C1=CCOCC1", "3,6-dihydro-2H-pyran"),
    ("C1=COCCC1", "3,4-dihydro-2H-pyran"),
    # the thiopyran twin always took that path
    ("OC(=O)C1=CCCSC1", "5,6-dihydro-2H-thiopyran-3-carboxylic acid"),
    # the saturated carbocycle keeps the licensed omission, (the PIN example at:31671)
    ("COC(=O)C1CCCCC1", "methyl cyclohexanecarboxylate"),
    ("COC(=O)C1CCCCC1C", "methyl 2-methylcyclohexane-1-carboxylate"),
    ("COC(=O)C1CCCCC1C(N)=O", "methyl 2-carbamoylcyclohexane-1-carboxylate"),
    ("OC(=O)C1CCCOC1C(N)=O", "2-carbamoyloxane-3-carboxylic acid"),
    ("Cc1ncccc1C(=O)O", "2-methylpyridine-3-carboxylic acid"),
    ("COC(=O)c1ccccc1C#N", "methyl 2-cyanobenzoate"),
    ("COC(=O)c1ccccc1C(C)=O", "methyl 2-acetylbenzoate"),
])
def test_controls_keep_their_names(smiles, expected):
    row = PIN.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert_full_rt(expected, smiles)


def test_a_demoted_ring_group_is_alphabetised_with_the_ring_prefixes():
    # 'carbamoyl' (from the non-cited carboxamide) sorts before 'methyl', the Blue Book)
    # and both are cited in front of the hydro-locant parent with a hyphen
    name = PIN.name_tiered("CN1CCC=C(C(=O)O)C1C(N)=O")["name"]
    assert name.startswith("2-carbamoyl-1-methyl-1,2,5,6-tetrahydro"), name


COMPLETE = Orthonym(style="pin", **_emit_tier_flags("complete"))


def test_general_engine_ring_lactone_takes_its_hydro_locants_from_the_suffix_numbering():
    # With the fixed-locant dihydropyran rows gone the general engine's lone-monocycle path had no
    # parent for a ring lactone that holds a C=C ('monocycle parent name underivable') and the
    # complete tier fell to the suffix-free '6-oxo-3,6-dihydro-2H-pyran' name. The path now takes the
    # hydrogen of a ring '-one' from the shared rule, the Blue Book;,:24689)
    # under the numbering the suffix selects: '5,6-dihydro-2H-pyran-2-one' (the retained row had given
    # '3,6-dihydro-2H-pyran-2-one', the isomer with the double bond at C4=C5).
    smiles = "CC1C=CC(=O)OC1C=CC(C)(O)C(CC(O)C=CC=CC1CCCCC1)OP(=O)(O)O"
    expected = ("6-[10-cyclohexyl-3,6-dihydroxy-3-methyl-4-(phosphonooxy)deca-1,7,9-trien-1-yl]-"
                "5-methyl-5,6-dihydro-2H-pyran-2-one")
    row = COMPLETE.name_tiered(smiles)
    assert row["name"] == expected, row
    assert_full_rt(expected, smiles)
