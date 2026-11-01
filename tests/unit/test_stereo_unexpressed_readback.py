"""LO10 -- the provenance flag ``stereo_unexpressed`` for a name of any producer.

``stereo_unexpressed`` says that a stereo element of the input (a stereocentre or a double bond
the SMILES configures) is not stated in the name. It used to be set only for a name of the
general engine under ``ORTHONYM_BE_STRIP_STEREO`` (namer ``_stereo_emit_decision``), so a name
of the strict path or of the composer that dropped the input's E/Z carried ``False``:

    C[N+](C)=C/C=C/C=C/N1CCCC1 -> 1-[5-(dimethylamino)penta-2,4-dien-1-ylidene]pyrrolidin-1-ium
                                     (pin_verified; the input configures two double bonds, the name none)

The bonds that carry the configuration are single bonds in the form that gives the PIN (the
double bonds of the drawing shift to the nitrogen end), so the standard InChIKey of the name
equals the input's and the round trip passes: the name is the PIN, the configuration is
simply not stated. ``name_tiered`` now reports it (no name, tier or label changes).

Meaning of the flag, pinned here: the OPSIN read-back of the shipped name configures FEWER
stereo elements than the input does (``validation.stereo_stated``; the read-back is the parse the
gate already made, a cache hit). A name OPSIN cannot read (lowercase r/s, a grammar carve-out)
has no read-back and is not flagged: the flag never guesses from the string.

Blue Book (no PIN decision here; the name and its tier stay as they were):
   "Omission of stereodescriptors" (the Blue Book),:44637: omission of the
  descriptors of double bonds is recommended only for three- to seven-membered unsaturated
  alicyclic rings; "From nine-membered rings onwards, stereodescriptors are required."
   "Stereodescriptors 'Z' and 'E', and 'cis' and 'trans'" (:46740),:46742: they "are
  recommended to specify the configuration of double bonds as preferred stereodescriptors in
  preferred IUPAC names."
   "Compounds containing more than one double bond" (:47102),:47104: "The
  stereodescriptors 'E' and 'Z' related to the parent are cited... before the complete name of the
  compound".
  The book gives no PIN for the input's configuration in a form whose double bonds are single
  bonds, so the tier is not moved (ASSUMED: keep the label; the user ruling of 2026-10-13 is that a
  name that had a PIN keeps it).
"""
import json

import pytest

from orthonym.cli import _emit_tier_flags
from orthonym.namer import Orthonym

# the OPSIN validity gate is off suite-wide by default (tests/conftest.py); these rows are about what the gate
# accepted and the read-back it made, so the module runs with it on
pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _namer(tier):
    f = _emit_tier_flags(tier)
    return Orthonym(general_fallback=f["general_fallback"],
                    general_fallback_unverified=f["general_fallback_unverified"],
                    allow_aromatic_general=f["allow_aromatic_general"],
                    full_coverage=f.get("full_coverage", False))


# The four pin_verified rows of the finding (internal notes): the input
# configures its double bonds, the PIN form is a form in which they are single bonds. The name and the
# tier are the ones shipped before this flag existed.
UNSTATED = [
    ("C[N+](C)=C/C=C/C=C/N1CCCC1",
     "1-[5-(dimethylamino)penta-2,4-dien-1-ylidene]pyrrolidin-1-ium"),
    ("CC\\1(C2=CC=CC=C2N(/C1=C/C=[N+]3CCCC3)C)C",
     "1,3,3-trimethyl-2-[2-(pyrrolidin-1-yl)ethenyl]-3H-indol-1-ium"),
    ("CC\\1(C2=CC=CC=C2N(/C1=C/C=[N+]3CCOCC3)CCOC4=CC=CC=C4)C",
     "3,3-dimethyl-2-[2-(morpholin-4-yl)ethenyl]-1-(2-phenoxyethyl)-3H-indol-1-ium"),
    ("CCC(=O)N/C(=[NH+]\\C1=NC(=CC(=N1)C)C)/NC2=CC=C(C=C2)OCC",
     "N-(4,6-dimethylpyrimidin-2-yl)-N'-(4-ethoxyphenyl)-N''-propanoylguanidinium"),
]

# Names that state every configured element: the flag stays False. L-phenylalanine states its centre by
# the retained 'L-' (no descriptor token in the string), which a token count would have missed.
STATED = [
    ("C/C=C/C=C/C(=O)[O-]", "(2E,4E)-hexa-2,4-dienoate"),
    ("C[C@H](N)C(=O)[O-]", "(2S)-2-aminopropanoate"),
    ("CCO", "ethanol"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", UNSTATED)
def test_a_pin_name_that_leaves_the_configuration_out_is_flagged(smiles, name, tier):
    row = _namer(tier).name_tiered(smiles)
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "pin_verified", True)
    assert row["stereo_unexpressed"] is True


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", STATED)
def test_a_name_that_states_every_element_is_not_flagged(smiles, name, tier):
    row = _namer(tier).name_tiered(smiles)
    assert row["name"] == name
    assert row["stereo_unexpressed"] is False


def test_an_amino_acid_zwitterion_name_states_its_centre():
    # (the Blue Book, example:54561): the zwitterion is named by the name, which cites the
    # configuration with its descriptor; the retained 'L-phenylalanine' names the neutral acid and has no PIN
    #:50943), so it no longer names the zwitterion (ruling R6, RULINGS-ANION.md)
    row = _namer("best-effort").name_tiered("[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]")
    assert row["name"] == "(2S)-2-azaniumyl-3-phenylpropanoate"
    assert row["stereo_unexpressed"] is False


def test_the_flag_is_in_the_json_row_and_changes_no_other_key():
    smiles, name = UNSTATED[0]
    row = _namer("pin").name_tiered(smiles)
    back = json.loads(json.dumps(row))
    assert back["stereo_unexpressed"] is True
    assert back["name"] == name and back["tier"] == "pin_verified"
    assert back["verified"] == "opsin" and back["opsin"] == "verified"
    assert set(back) == {"name", "tier", "is_pin", "source", "opsin", "gates_passed", "gate_outcome", "formula",
                         "limit_code", "stereo_unexpressed", "suffix_free_prefix_name", "prefix_order_fallback",
                         "verified", "spelling_failures", "producer"}


def test_a_decline_is_not_flagged():
    # the default tier declines this input (NO_VERIFIED_PIN); the row of a decline names nothing.
    row = _namer("pin").name_tiered("CC[N+](CC)=C/C=C/N(CC)CC")
    assert row["tier"] == "abstain"
    assert row["stereo_unexpressed"] is False


# --- the helper, with a stubbed read-back (no OPSIN): what it counts and when it answers ----------------------

class _Stub:
    """Replaces ``namer._validity_gate_name_to_smiles``: records the names asked, answers from a table."""

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def __call__(self, name):
        self.asked.append(name)
        return self.answers.get(name)


@pytest.fixture
def stub(monkeypatch):
    s = _Stub({})
    monkeypatch.setattr("orthonym.namer._validity_gate_name_to_smiles", s)
    return s


def test_helper_counts_the_elements_of_the_read_back_against_the_input(stub):
    from orthonym.validation.stereo_stated import name_states_fewer_stereo as fewer
    two_bonds = "C[N+](C)=C/C=C/C=C/N1CCCC1"
    stub.answers = {"none": "CN(C)C=CC=CC=[N+]1CCCC1",                 # configures 0 of 2
                    "one": "CN(C)/C=C/C=CC=[N+]1CCCC1",                  # configures 1 of 2
                    "both": "CN(C)/C=C/C=C/C=[N+]1CCCC1",                # configures 2 of 2
                    "more": "CN(C)/C=C/C=C/C=[N+]1CC[C@H](C)C1"}         # configures 3 (a centre more): not fewer
    assert fewer(two_bonds, "none") is True
    assert fewer(two_bonds, "one") is True
    assert fewer(two_bonds, "both") is False
    assert fewer(two_bonds, "more") is False


def test_helper_answers_false_without_a_read_back(stub):
    from orthonym.validation.stereo_stated import name_states_fewer_stereo as fewer
    # OPSIN rejected the name (or could not be asked): nothing was read back, so nothing is claimed
    assert fewer("C[C@H](N)C(=O)O", "(1s,4s)-whatever") is False
    assert stub.asked == ["(1s,4s)-whatever"]


def test_helper_does_not_ask_for_an_input_without_a_stereo_mark_or_a_name_of_a_decline(stub):
    from orthonym.validation.stereo_stated import name_states_fewer_stereo as fewer
    assert fewer("CCO", "ethanol") is False                             # no '/', '\\' or '@' in the SMILES
    assert fewer("C[C@H](N)C(=O)O", None) is False
    assert fewer("C[C@H](N)C(=O)O", "unknown organic compound") is False
    assert stub.asked == []


def test_helper_reuses_the_cached_parse_and_never_runs_opsin_for_a_name_the_gate_read(monkeypatch):
    """The read-back is ``_validity_gate_name_to_smiles``: the oracle's cache of the gate's own parse. For a name
    already read it costs no OPSIN call (the parse a name was given once is the parse every later question gets)."""
    from orthonym import namer
    from orthonym.assembly.retained_substitution import OpsinOracle
    from orthonym.validation.stereo_stated import name_states_fewer_stereo as fewer
    oracle = namer._validity_oracle()
    name = "1-[5-(dimethylamino)penta-2,4-dien-1-ylidene]pyrrolidin-1-ium"
    assert oracle.name_to_smiles(name)                                   # the gate's parse (one OPSIN call, cached)

    def no_opsin(self, _name):
        raise AssertionError("OPSIN was asked again for a name the gate had read")
    monkeypatch.setattr(OpsinOracle, "_invoke_opsin", no_opsin)
    assert fewer("C[N+](C)=C/C=C/C=C/N1CCCC1", name) is True
