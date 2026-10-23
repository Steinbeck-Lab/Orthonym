"""Lone-pair stereocentres of the elements ``centres`` (CDK) gives no label (As, Sb, Bi, Te, Po), and the
centres RDKit drops at parsing (the N of an N-aryl aziridine, Sb, Bi, Te, Po).

Measured on the base of PubChem 1M lane S2 (the review of the lane):

* ``centres`` labels N (in a three-membered ring), P, S and Se and nothing at As, Sb, Bi, Te, Po. The label the
  engine gave an As centre was the one RDKit's legacy labeller set at parsing, the other configuration than
  ``rdCIPLabeler``'s and than OPSIN's: 'C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1' is (1R,2S) (OPSIN's own
  StdInChIKey of '(1R,2S)-1,2-diphenylarsolane' is RDKit's key of the input, which RDKit reads the standard way)
  and was named (1S,2S) at pin_verified. The exit check passed it: no label on either side.
* RDKit drops the chirality of Sb, Bi, Te, Po and of the N of 'C[C@H]1C[N@]1c1ccccc1' when it parses; CDK and the
  standard InChI keep it. A name without that configuration ('(2S)-2-methyl-1-phenylaziridine') was pin_verified
  though the input defines it.

The parity rule of a SMILES string does not depend on the element, so such an atom is read through a stand-in of the
same valence (P for As, Sb, Bi; S for Te, Po), both labellers reading the stand-in; the label of the element is
``rdCIPLabeler``'s on the Mol; and the exit check fails closed where a compared centre has no label or the name
leaves a centre of the input out. Every expectation below rests on OPSIN's own StdInChIKey of the name and on the
independent reader of ``tests/support/indep_oracle`` (CDK's SMILES parser into the InChI library), through
``tests.support.rt_assert.name_is_rt_exact``.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.perception import lone_pair_reading as lpr
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ARSOLANE = "C1C[C@H]([As@@](C1)c1ccccc1)c1ccccc1"
ARSOLANE_OTHER = "C1C[C@H]([As@](C1)c1ccccc1)c1ccccc1"
ARSOLANE_FIRST = "[As@]1(c2ccccc2)CCC[C@@H]1c1ccccc1"
ARSOLANE_METHYL = "C[C@@H]1CCC[As@]1c1ccccc1"
ARSANE_FIRST = "[As@@](C)(CC)c1ccccc1"
STIBOLANE = "C1C[C@H]([Sb@@](C1)c1ccccc1)c1ccccc1"
TELLURANE_OXIDE = "C[C@@H]1CC[Te@@](=O)C1"
ARYL_AZIRIDINE = "C[C@H]1C[N@]1c1ccccc1"
BENZYL_AZIRIDINE = "C[C@H]1C[N@]1Cc1ccccc1"


def _row(smiles, tier):
    namer_ = (Orthonym(style="pin") if tier == "pin"
              else Orthonym(style="pin", **_emit_tier_flags(tier)))
    return namer_.name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    (ARSOLANE, "(1R,2S)-1,2-diphenylarsolane"),
    (ARSOLANE_OTHER, "(1S,2S)-1,2-diphenylarsolane"),
    (ARSOLANE_FIRST, "(1S,2R)-1,2-diphenylarsolane"),
    (ARSOLANE_METHYL, "(1R,2R)-2-methyl-1-phenylarsolane"),
    (ARSANE_FIRST, "(R)-ethyl(methyl)(phenyl)arsane"),
])
@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_an_arsenic_centre_is_named_by_the_configuration_of_the_input(smiles, name, tier):
    res = _row(smiles, tier)
    assert res.get("name") == name, res.get("name")
    assert name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("smiles", [ARSOLANE, ARSOLANE_OTHER, ARSOLANE_FIRST, ARSOLANE_METHYL, ARSANE_FIRST])
def test_the_names_of_the_other_configurations_of_an_arsenic_centre_are_not_read_back(smiles):
    # the independent reading of the input is not the reading of the name that differs at the As only
    name = _row(smiles, "best-effort")["name"]
    other = name.replace("(1R,", "(1X,").replace("(1S,", "(1R,").replace("(1X,", "(1S,")
    if "(R)-" in name or "(S)-" in name:
        other = name.replace("(R)-", "(X)-").replace("(S)-", "(R)-").replace("(X)-", "(S)-")
    assert other != name
    assert not name_is_rt_exact(other, smiles)


PHOSPHOLANYL = "C[C@@H]1CC[P@@](=O)C1"
STIBOLANE_NAME = "(2S)-1,2-diphenylstibolane"
TELLURANE_NAME = "(3R)-3-methyl-1-oxo-1λ4-tellurolane"


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_a_name_that_leaves_out_a_configuration_inchi_sees_is_not_shipped(tier):
    # the N of the N-aryl aziridine: RDKit drops its tag, the standard InChI keeps it (the key of
    # '(1R,2S)-2-methyl-1-phenylaziridine' is the input's, that of '(2S)-...' is not), so the name that leaves it
    # out is not the input: declined at every tier
    res = _row(ARYL_AZIRIDINE, tier)
    assert res.get("tier") == "abstain", res
    assert not lpr.dropped_centres_invisible_to_inchi(ARYL_AZIRIDINE)


@pytest.mark.parametrize("smiles,name", [(STIBOLANE, STIBOLANE_NAME), (TELLURANE_OXIDE, TELLURANE_NAME)])
def test_a_name_that_leaves_out_a_configuration_no_inchi_sees_is_shipped_below_the_pin(smiles, name):
    # Sb and Te: RDKit drops the tag and the standard InChI of the structure is the same with or without it, so no
    # round trip refutes the name; it is not the PIN cites each stereogenic unit): best-effort ships
    # it flagged (the label it had before the check, systematic_verified or below), the default tier declines
    assert lpr.dropped_centres_invisible_to_inchi(smiles)
    row = _row(smiles, "best-effort")
    assert row.get("name") == name, row
    assert row.get("is_pin") is False and row.get("stereo_unexpressed") is True
    assert row.get("tier") == "systematic_verified"
    assert namer._lone_pair_configuration_verified(name, smiles) is False
    assert namer._lone_pair_configuration_verified(name, smiles, held_only=True) is True
    assert _row(smiles, "pin").get("tier") == "abstain"


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_a_radical_phosphorus_is_no_lone_pair_centre(tier):
    # 'C[C@@H]1CC[P@@](=O)C1' is a phospholanyl radical: the fourth position of the P holds the unpaired electron
    # and the standard InChI does not see the P. The name of the base is read back (OPSIN with radicals allowed,
    # CDK into the InChI library) and stays pin_verified.
    name = "(3R)-3-methyl-1-oxo-1λ5-phospholan-1-yl"
    row = _row(PHOSPHOLANYL, tier)
    assert row.get("name") == name and row.get("tier") == "pin_verified", row
    assert name_is_rt_exact(name, PHOSPHOLANYL)
    assert not name_is_rt_exact("(3S)-3-methyl-1-oxo-1λ5-phospholan-1-yl", PHOSPHOLANYL)
    assert namer._has_lone_pair_stereocentre(PHOSPHOLANYL) is False
    assert lpr.lone_pair_centres(lpr.raw_reading(PHOSPHOLANYL)) == []
    assert lpr.lone_pair_centres(lpr.raw_reading("C[C@@H]1CC[P@@](C1)c1ccccc1")) != []       # the phosphine is one


def test_the_exit_check_reads_what_the_string_defines():
    verified = namer._lone_pair_configuration_verified
    # the N of the aziridine: the name with the N descriptor holds it, the name without does not
    assert verified("(1R,2S)-2-methyl-1-phenylaziridine", ARYL_AZIRIDINE) is True
    assert verified("(2S)-2-methyl-1-phenylaziridine", ARYL_AZIRIDINE) is False
    # Sb: OPSIN reads no configuration at Sb in the name
    assert verified("(2S)-1,2-diphenylstibolane", STIBOLANE) is False
    assert verified("(2S)-1,2-diphenylstibolane", STIBOLANE, held_only=True) is True
    # As, through its stand-in
    assert verified("(1R,2S)-1,2-diphenylarsolane", ARSOLANE) is True
    assert verified("(1S,2S)-1,2-diphenylarsolane", ARSOLANE) is False


def test_the_exit_check_fails_closed_where_a_compared_centre_has_no_label(monkeypatch):
    # without the stand-in no label reads the As centre of either string: not confirmed, as it was passed
    # before (a missing label on both sides was an agreement)
    import re
    monkeypatch.setattr(lpr, "_STAND_IN_RE", re.compile(r"(\[\d*)(NOSUCHELEMENT)(?![a-z])"))
    lpr.stand_in_string.cache_clear()
    try:
        assert namer._lone_pair_configuration_verified("(1R,2S)-1,2-diphenylarsolane", ARSOLANE) is False
        assert namer._lone_pair_configuration_verified("(1S,2S)-1,2-diphenylarsolane", ARSOLANE) is False
    finally:
        monkeypatch.undo()
        lpr.stand_in_string.cache_clear()
    assert namer._lone_pair_configuration_verified("(1R,2S)-1,2-diphenylarsolane", ARSOLANE) is True


def test_a_stand_in_is_written_in_place_and_audited():
    assert lpr.stand_in_string(ARSOLANE) == "C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1"
    assert lpr.stand_in_string("[75As@]1(c2ccccc2)CCC[C@@H]1c1ccccc1") == "[75P@]1(c2ccccc2)CCC[C@@H]1c1ccccc1"
    assert lpr.stand_in_string(TELLURANE_OXIDE) == "C[C@@H]1CC[S@@](=O)C1"
    assert lpr.stand_in_string("C[Sb@](CC)c1ccccc1") == "C[P@](CC)c1ccccc1"
    # nothing to write: the string itself; an atom that is no stand-in element is left alone
    assert lpr.stand_in_string("[As+](C)(C)(C)C") == "[P+](C)(C)(C)C"
    for smiles in ("CCO", "C[S@@](=O)c1ccccc1", "C[Se@@](=O)c1ccccc1", "c1ccc[as]c1"):
        assert lpr.stand_in_string(smiles) == smiles


def test_centres_labels_an_as_centre_through_its_stand_in():
    from orthonym.perception.centres_bridge import centres_label_batch
    assert centres_label_batch([ARSOLANE])[ARSOLANE] == {3: "S"}              # no label at the As
    assert lpr.standard_centres_labels(ARSOLANE) == {3: "S", 4: "R"}
    assert lpr.standard_centres_labels("C1C[C@H]([P@@](C1)c1ccccc1)c1ccccc1") == {3: "S", 4: "R"}


def test_the_engine_label_of_an_unlabelled_centre_is_rdcip_not_the_legacy_one():
    # RDKit's legacy labeller set the other label on the As at parsing; the Mol the engine labels holds
    # rdCIPLabeler's
    from rdkit.Chem import rdCIPLabeler

    from orthonym.perception.stereo import assign_stereochemistry
    mol = Chem.MolFromSmiles(ARSOLANE)
    legacy = {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    held = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(held)
    new = {a.GetIdx(): a.GetProp("_CIPCode") for a in held.GetAtoms() if a.HasProp("_CIPCode")}
    assert legacy[3] != new[3]
    assign_stereochemistry(mol)
    engine = {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}
    assert engine[3] == new[3] == "R"


def test_a_string_with_a_dropped_centre_is_read_by_what_it_writes():
    assert namer._has_lone_pair_stereocentre(ARYL_AZIRIDINE) is True        # RDKit's own reading has none
    assert not any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and a.GetSymbol() == "N"
                   for a in Chem.MolFromSmiles(ARYL_AZIRIDINE).GetAtoms())
    assert namer._has_lone_pair_stereocentre(STIBOLANE) is True
    assert namer._has_lone_pair_stereocentre("CC[N@](C)c1ccccc1") is False  # an amine N outside a ring: not compared
    assert namer._has_lone_pair_stereocentre("C[C@H](O)CC") is False
    assert namer._has_lone_pair_stereocentre(BENZYL_AZIRIDINE) is True
