import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
def test_nitrophenoxymethyl_oxirane_names(namer):
    # NOTE: the task brief predicted "2-[(4-nitrophenoxy)methyl]oxirane". That
    # string contradicts this codebase's OWN Blue-Book-cited precedent (see
    # tests/unit/rules/test_p14_3_4_task3b_baked_locants.py, BB P-13.1 table
    # row 7, verbatim PIN `phenyloxirane`, plus `chlorooxirane`/`methyloxirane`):
    # a MONOsubstituted oxirane omits its ring locant because the two ring CH2
    # positions are one orbit (`l3_one_kind_of_substitutable_h`), independent of
    # what decorates the substituent. `get_alkoxy_prefix` (called directly,
    # bypassing the front-filter bug) already returns bare "4-nitrophenoxy" --
    # structurally identical in shape to "4-methylphenoxy"/"4-chlorophenoxy" --
    # so nitro must take the SAME omission as the sibling controls below, not a
    # one-off bracket/locant escalation with no Blue Book basis distinguishing
    # it from methyl/chloro.
    smi = "O=[N+]([O-])c1ccc(OCC2CO2)cc1"
    name = namer.name(smi)
    assert name == "(4-nitrophenoxymethyl)oxirane", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("c1ccc(OCC2CO2)cc1", "(phenoxymethyl)oxirane"),
    ("Cc1ccc(OCC2CO2)cc1", "(4-methylphenoxymethyl)oxirane"),
])
def test_plain_aryloxymethyl_unchanged(namer, smi, expected):
    assert namer.name(smi) == expected
