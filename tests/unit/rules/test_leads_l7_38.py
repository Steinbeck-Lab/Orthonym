"""Leads program L7 / 38: the connecting 'a' before a multiplied 'ene' in a macrolactone name.

 (heading 'General methodology', the Blue Book),:16497: "For
euphonic reasons, when the endings 'ene' and 'yne' are preceded by a multiplying prefix and
a locant the letter 'a' is inserted." Examples::8498 '1-oxacycloundeca-
2,4,6,8,10-pentaene (PIN)' (heading 'Numbering':8492); with a suffix
:26555 '4-iminocyclohexa-2,5-dien-1-one (PIN)'.

The macrolactone namer wrote '1-oxacyclotridec-3,5-dien-2-one' (no 'a'). A single ene has no
multiplying prefix and takes no 'a': '1-oxacyclotridec-3-en-2-one'.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.rules.lactones import name_lactone_ring
from tests.support.rt_assert import _independent_parse


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# (smiles, name): a multiplied ene takes the connecting 'a'.
MULTIPLIED_ENE = [
    ("O=C1OCCCCCCCC=CC=C1", "1-oxacyclotrideca-3,5-dien-2-one"),
    ("O=C1OCCCCCC=CCC=CC1", "1-oxacyclotrideca-4,7-dien-2-one"),
    ("O=C1OCCCCCCCCC=CC=C1", "1-oxacyclotetradeca-3,5-dien-2-one"),
    ("O=C1OCCCCCCCC=CC=CC=CC=C1", "1-oxacycloheptadeca-3,5,7,9-tetraen-2-one"),
    (
        "C[C@H]1CCC/C=C/CC/C=C/C=C/C(=O)O1",
        "(3E,5E,9E,14S)-14-methyl-1-oxacyclotetradeca-3,5,9-trien-2-one",
    ),
    (
        "CC1CCC/C=C\\C=C\\C(O)CC(O)C/C=C\\C=C\\C(O)C/C=C/C=C\\C(=O)O1",
        "(3Z,5E,9E,11Z,17E,19Z)-8,14,16-trihydroxy-24-methyl-1-oxacyclotetracosa-3,5,9,"
        "11,17,19-hexaen-2-one",
    ),
]

# (smiles, name): no multiplying prefix (or no ene): no 'a'. Must not change.
CONTROLS = [
    ("O=C1CCCCCCC/C=C/CCO1", "(10E)-1-oxacyclotridec-10-en-2-one"),
    ("O=C1OCCCCCCCCCC=C1", "1-oxacyclotridec-3-en-2-one"),
    ("O=C1CCCCCCCCCCCO1", "1-oxacyclotridecan-2-one"),
    ("C1=CCCCC=CCCCCC1", "cyclododeca-1,6-diene"),   # the bare carbocycle already has it
]


@pytest.mark.opsin_gate
class TestMacrolactoneEneConnectingA:
    @pytest.mark.parametrize("smiles,expected", MULTIPLIED_ENE + CONTROLS)
    def test_name(self, smiles, expected):
        assert Orthonym(style="pin").name(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", MULTIPLIED_ENE + CONTROLS)
    def test_name_reads_back_to_the_input(self, smiles, expected):
        # a fresh OPSIN 2.9.0 parse, not the engine's validity oracle
        parsed = _independent_parse(expected)
        assert parsed is not None, f"OPSIN rejects {expected!r}"
        assert _key(parsed) == _key(smiles), (expected, parsed)

    @pytest.mark.parametrize("smiles,expected", MULTIPLIED_ENE)
    def test_pin_tier_label(self, smiles, expected):
        r = Orthonym(style="pin").name_tiered(smiles)
        assert (r["name"], r["tier"]) == (expected, "pin_verified"), r


@pytest.mark.unit
class TestNameLactoneRing:
    @pytest.mark.parametrize(
        "ring,locants,expected",
        [
            (13, [3, 5], "1-oxacyclotrideca-3,5-dien-2-one"),
            (14, [3, 5, 9], "1-oxacyclotetradeca-3,5,9-trien-2-one"),
            (24, [3, 5, 9, 11, 17, 19], "1-oxacyclotetracosa-3,5,9,11,17,19-hexaen-2-one"),
            (13, [10], "1-oxacyclotridec-10-en-2-one"),  # one ene: no multiplier, no 'a'
        ],
    )
    def test_ene_stem(self, ring, locants, expected):
        assert name_lactone_ring(ring, ene_locants=locants) == expected

    def test_thione_keeps_the_final_e(self):
        # 'thione' does not begin with an elision vowel: '...diene-2-thione'
        assert (
            name_lactone_ring(13, chalcogen="S", ene_locants=[3, 5])
            == "1-oxacyclotrideca-3,5-diene-2-thione"
        )
