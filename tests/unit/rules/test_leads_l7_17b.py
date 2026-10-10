"""Leads program L7 / 17b: per-unit isotope descriptors in multiplied units.

 (heading ' NAMES', the Blue Book),:43718: the nuclide symbols,
"preceded by any necessary locant(s)", are inserted before the part of the compound that is
isotopically substituted; for a multiplied substituent that is the substituent word, inside
the multiplier's marks: '1,2-di[(13C)methyl]benzene (PIN)' (:43734,.
(:43770): 'cyclohexane-1,1-di[(14C)carboxylic acid] (PIN)' (:43774). The descriptor is that of
ONE unit: a hexadecanoyl has one C-1, so two [1-14C]hexadecanoyl groups are
'bis[(1-14C)hexadecanoyloxy]', not the sum over both ('(1,1-14C2)hexadecanoyloxy').

 'Omission of locants' (:44178): a locant is omitted only under (:44190,
"only one atom of a given element") or (:44196, "all positions... completely
isotopically substituted or modified in the same way"); (:44202): "Locants are not
omitted when there is a possibility of isomers". So '(14C)methyl' (one carbon) and '(2H3)methyl'
(fully labelled) cite no locant; '(1-13C)ethyl' and '(1-14C)hexadecanoyloxy' do.

Every name below is read back by a fresh OPSIN 2.9.0 parse (not the engine's own validity
oracle) to the input's full InChIKey.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules.isotopes import (
    _decorate_uniform_multiplier,
    _escalate_for_unit_descriptor,
    _multiplied_unit_slots,
)
from tests.support.rt_assert import _independent_parse

DPPC_14C2 = (
    "CCCCCCCCCCCCCCC[14C](=O)OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)O[14C](=O)CCCCCCCCCCCCCCC"
)

# (smiles, name, expected full InChIKey of the input): the per-unit descriptor on every copy.
UNIFORM = [
    ("[14CH3]OS(=O)(=O)O[14CH3]", "di[(14C)methyl] sulfate", "VAYGXNSJCAHWJZ-XPULMUKRSA-N"),
    ("C[13CH2]OC(=O)C(=O)O[13CH2]C", "di[(1-13C)ethyl] oxalate", None),
    ("c1ccc(C[13CH3])c(C[13CH3])c1", "1,2-di[(2-13C)ethyl]benzene", None),
    #: the label at C-1 of the ethyl needs its locant ('(13C)ethyl' reads the same
    # to OPSIN, which places an unlocanted label by default, and was the shipped spelling)
    ("c1ccc([13CH2]C)c([13CH2]C)c1", "1,2-di[(1-13C)ethyl]benzene", None),
    (
        "[2H]C([2H])([2H])[Si](C([2H])([2H])[2H])(C([2H])([2H])[2H])C([2H])([2H])[2H]",
        "tetra[(2H3)methyl]silane",
        None,
    ),
    (
        "[2H]C([2H])([2H])n1cnc2c1c(=O)[nH]c(=O)n2C([2H])([2H])[2H]",
        "3,7-di[(2H3)methyl]-3,7-dihydro-1H-purine-2,6-dione",
        None,
    ),
]

# Names that must not change: the Blue Book's own example and the caffeine row of ChEBI.
CONTROLS = [
    ("[13CH3]c1ccccc1[13CH3]", "1,2-di[(13C)methyl]benzene"),        # BB:43734 (PIN)
    (
        "[13CH3]n1c(=O)c2c(ncn2[13CH3])n([13CH3])c1=O",
        "1,3,7-tri[(13C)methyl]-3,7-dihydro-1H-purine-2,6-dione",
    ),
]


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.mark.opsin_gate
class TestPerUnitDescriptors:
    @pytest.mark.parametrize("smiles,name,key", UNIFORM)
    def test_pin_tier_name(self, smiles, name, key):
        r = Orthonym(style="pin").name_tiered(smiles)
        assert (r["name"], r["tier"]) == (name, "pin_verified"), r

    @pytest.mark.parametrize("smiles,name,key", UNIFORM)
    def test_name_reads_back_to_the_input(self, smiles, name, key):
        parsed = _independent_parse(name)
        assert parsed is not None, f"OPSIN rejects {name!r}"
        assert _key(parsed) == _key(smiles), (name, parsed)
        if key:
            assert _key(smiles) == key

    @pytest.mark.parametrize("smiles,name", CONTROLS)
    def test_controls_unchanged(self, smiles, name):
        assert Orthonym(style="pin").name(smiles) == name

    def test_dppc_14c2_best_effort(self):
        # ChEBI idx 90065. Lost at 220a9d4f1, which removed the sum spelling
        # '(1,1-14C2)hexadecanoyloxy' (a hexadecanoyl has one C-1) and left no per-unit
        # producer. The descriptor is that of one unit; the skeleton is not certified as
        # the PIN (the default tier declines it unlabelled too), so the name is below it.
        expected = (
            "2-[({(R)-2,3-bis[(1-14C)hexadecanoyloxy]propoxy}hydroxyphosphoryl)oxy]-"
            "N,N,N-trimethylethan-1-aminium"
        )
        r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(DPPC_14C2)
        assert (r["name"], r["tier"]) == (expected, "systematic_verified"), r
        parsed = _independent_parse(expected)
        assert parsed is not None
        assert _key(parsed) == _key(DPPC_14C2) == "KILNVBDSWZSGLL-WYTZSWIOSA-O"
        assert Orthonym(style="pin").name_tiered(DPPC_14C2)["tier"] == "abstain"

    def test_dmso_d6_never_a_pin_label(self):
        # 'dimethyl sulfoxide' is the functional-class name; the PIN is
        # '(methanesulfinyl)methane', so the name built on it is below the PIN.
        smi = "[2H]C([2H])([2H])S(=O)C([2H])([2H])[2H]"
        r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smi)
        assert r["name"] == "di[(2H3)methyl] sulfoxide" and r["tier"] == "systematic_verified"
        assert Orthonym(style="pin").name_tiered(smi)["tier"] == "abstain"

    def test_dmf_d7_stays_abstained(self):
        # OPSIN cannot read 'N,N-di[(2H3)methyl](2H)formamide': no name that reads back
        smi = "[2H]C(=O)N(C([2H])([2H])[2H])C([2H])([2H])[2H]"
        assert Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smi)[
            "tier"] == "abstain"


@pytest.mark.unit
class TestUniformProducer:
    def test_unit_count_is_the_total_divided_by_the_copies(self):
        # 2 x 14C over 2 methyls is ONE per methyl: '(14C)', never the sum '(14C2)'
        original = Chem.MolFromSmiles("[14CH3]OS(=O)(=O)O[14CH3]")
        got = _decorate_uniform_multiplier(
            "dimethyl sulfate", [((14, "C"), 2, 1)], original, None)
        assert got == "di[(14C)methyl] sulfate"

    def test_an_uneven_split_places_nothing(self):
        # one 14C over two copies: the single-labelled-copy shape (_decorate_demultiplied)
        original = Chem.MolFromSmiles("[14CH3]OS(=O)(=O)OC")
        assert _decorate_uniform_multiplier(
            "dimethyl sulfate", [((14, "C"), 1, 1)], original, None) is None

    def test_locant_search_over_the_unit_numbering(self):
        original = Chem.MolFromSmiles("C[13CH2]OC(=O)C(=O)O[13CH2]C")
        got = _decorate_uniform_multiplier(
            "diethyl oxalate", [((13, "C"), 2, 1)], original, None)
        assert got == "di[(1-13C)ethyl] oxalate"

    def test_a_multiplier_inside_an_outer_group_is_found(self):
        slots = [(c, b) for c, _h, b, _t, _e in _multiplied_unit_slots(
            "2-({[(R)-2,3-bis(hexadecanoyloxy)propoxy]hydroxyphosphoryl}oxy)-"
            "N,N,N-trimethylethan-1-aminium")]
        assert (2, "hexadecanoyloxy") in slots
        assert (3, "methyl") in slots        # N,N,N-trimethyl (a surplus slot is harmless)

    @pytest.mark.parametrize("skeleton,expected", [
        ("tetramethylsilane", [(4, "methyl")]),
        ("dimethyl sulfate", [(2, "methyl")]),
        ("1,2-diethylbenzene", [(2, "ethyl")]),
    ])
    def test_slots_of_simple_skeletons(self, skeleton, expected):
        assert [(c, b) for c, _h, b, _t, _e in _multiplied_unit_slots(skeleton)] == expected

    def test_a_multiplier_like_run_inside_a_word_is_not_a_slot(self):
        assert [s for s in _multiplied_unit_slots("hexadecanoyl chloride")] == []

    def test_hydrogen_nuclide_marks_step_up(self):
        # apply_enclosing_marks reads a leading '(2H)' as an indicated hydrogen and leaves
        # 'di((2H)methyl)'; the stand-in descriptor makes it a plain parenthesised unit
        assert _escalate_for_unit_descriptor("di((2H)methyl)x", 3, "(2H)") == "di[(2H)methyl]x"
        assert _escalate_for_unit_descriptor(
            "tetra((2H3)methyl)silane", 6, "(2H3)") == "tetra[(2H3)methyl]silane"
