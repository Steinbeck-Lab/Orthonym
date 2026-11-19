"""Phosphatidylinositols and inositol phosphate esters (ChEBI C3, 2026-11-19).

 "Phosphatidylinositols" (the Blue Book) names them substitutively, the inositol as a
'2,3,4,5,6-pentahydroxycyclohexyl' group inside the phosphoryloxy prefix of the glycerol diester:
'(2R)-3-[(hydroxy{[(1s,2R,3R,4s,5S,6S)-2,3,4,5,6-pentahydroxycyclohexyl]oxy}phosphoryl)oxy]propane-1,2-diyl
dihexadecanoate'. OPSIN 2.9.0 cannot assign the CIP stereo of the inositol ring, so the name reads back through the
stereo-layer check (constitution by OPSIN, every descriptor against the input's CIP labels)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

pytestmark = pytest.mark.opsin_gate

_INOSITYL = "O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O"   # 1D-myo-inositol 1-yl, as ChEBI draws it


@pytest.mark.parametrize("smiles,name", [
    #: the diacyl phosphatidylinositol, both acyl groups the same
    ("CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)" + _INOSITYL[0] + _INOSITYL[1:] + ")OC(=O)CCCCCCCCCCCCCCC",
     "(2R)-3-[(hydroxy{[(1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl]oxy}phosphoryl)oxy]propane-1,2-diyl "
     "dihexadecanoate"),
    # the glycerophosphoinositol anion: the functional-class diester anion
    ("O=P([O-])(OC[C@H](O)CO)" + _INOSITYL,
     "(2R)-2,3-dihydroxypropyl (1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl phosphate"),
])
def test_inositol_phosphate_esters_are_named(smiles, name):
    with jvm_slots(1, purpose="inositol-phospholipids"):
        for tier in ("pin", "best-effort"):
            row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
            assert row["name"] == name, (tier, row["name"])
            assert row["tier"] == "pin_unverified"


def test_only_the_word_mode_strips_a_block_after_a_space():
    """The second organyl word of an ester carries its own block. The stereo-layer carve-out strips it (``words=True``);
    every other caller keeps the old stripper, which leaves it (review 2026-11-19: '(2RS)-butan-2-yl (2S)-2-methylbutanoate'
    lost its '(2RS)' at best-effort when every caller stripped it)."""
    from orthonym.rules.stereochemistry import locate_stereo_blocks, strip_stereo_blocks
    name = "(2R)-2,3-dihydroxypropyl (1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl phosphate"
    assert strip_stereo_blocks(name) == (
        "2,3-dihydroxypropyl (1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl phosphate", ["(2R)"])
    assert strip_stereo_blocks(name, words=True) == (
        "2,3-dihydroxypropyl 2,3,4,5,6-pentahydroxycyclohexyl phosphate", ["(2R)", "(1S,2R,3R,4S,5S,6R)"])
    assert [name[s:e] for s, e, _t in locate_stereo_blocks(name, words=True)] == ["(2R)-", "(1S,2R,3R,4S,5S,6R)-"]


@pytest.mark.parametrize("name,smiles", [
    # review 2026-11-19: locants restart in every word, so a block read by its locants alone could land in the other
    # word: R is on the butan-2-yl carbon, the acid C2 is open
    ("butan-2-yl (2R)-2-methylbutanoate", "C[C@H](CC)OC(=O)C(C)CC"),
    ("2,3-dihydroxypropyl (2S)-2-methylbutanoate", "CC(C(=O)OC[C@@H](O)CO)CC"),
    # the glycerol is R and the ring C2 is open; the wrong ring block must not borrow the glycerol C2
    ("2,3-dihydroxypropyl (1S,2R,3S,4S,5R,6R)-2,3,4,5,6-pentahydroxycyclohexyl phosphate",
     "O=P([O-])(OC[C@H](O)CO)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)C1O"),
])
def test_a_word_block_is_not_read_in_another_word(name, smiles):
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.stereo_descriptors import checked_block_units
    with jvm_slots(1, purpose="inositol-phospholipids"):
        assert checked_block_units(name, smiles, strip_stereo_blocks(name, words=True)[0]) is None


def test_a_retained_inositol_name_has_no_group_form():
    """'myo-inositol' names a hexol; no parser reads 'myo-inosityl' gives no group form)."""
    from orthonym.decomposition.fragment_assembly import _alcohol_to_alkyl
    assert _alcohol_to_alkyl("myo-inositol") == "myo-inositol"
    assert _alcohol_to_alkyl("ethanol") == "ethyl"


def test_the_anion_of_a_phosphatidylinositol_is_named_on_its_phosphate():
    """ ranks anions (class 4) above esters (class 9): the phosphate anion is the principal group, not an
    'oxido' prefix on the glycerol diester. The substituent namer cites the glycerol centre without its locant here
    ('(R)-2,3-bis(...)propyl', a documented limit of assembly/substituent_naming._add_substituent_stereo_text), so the
    name ships below the PIN and the PIN tier declines."""
    from orthonym.rules.lipids import name_lipid
    smiles = "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])" + _INOSITYL + ")OC(=O)CCCCCCCCCCCCCCC"
    assert name_lipid(Chem.MolFromSmiles(smiles)) is None
    with jvm_slots(1, purpose="inositol-phospholipids"):
        be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
        pin = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
    assert be["name"] == ("(R)-2,3-bis(hexadecanoyloxy)propyl (1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl "
                          "phosphate")
    assert be["tier"] == "best_effort" and pin["tier"] == "abstain"


def test_the_gate_keeps_a_word_block_the_check_cannot_place():
    """The final gate strips a block that starts a later word only when the descriptor check places it (end check
    2026-11-19): '(6R)' of the benzoate word cites a ring carbon that is no stereocentre, so it stays in the stripped name
    and OPSIN refuses that name, as before the word mode."""
    from orthonym.validation.stereo_descriptors import carve_out_reduced
    smiles = "COC(=O)c1ccc([C@](C)(O)CCCC(C)(C)O)c(O)c1"
    name = "methyl (6R)-4-[(2R)-2,6-dihydroxy-6-methylheptan-2-yl]-3-hydroxybenzoate"
    with jvm_slots(1, purpose="inositol-phospholipids"):
        assert carve_out_reduced(name, lambda n: False, smiles) == (
            "methyl (6R)-4-[2,6-dihydroxy-6-methylheptan-2-yl]-3-hydroxybenzoate")
        glycerophosphoinositol = ("(2R)-2,3-dihydroxypropyl (1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl "
                                  "phosphate")
        assert carve_out_reduced(glycerophosphoinositol, lambda n: False, "O=P([O-])(OC[C@H](O)CO)" + _INOSITYL) == (
            "2,3-dihydroxypropyl 2,3,4,5,6-pentahydroxycyclohexyl phosphate")
