"""Lone-pair stereocentres read by RDKit (TRIAGE.md 'Lone-pair stereocentres -- RDKit
reading').

Every round trip in the engine compares the RDKit InChIKey of the input with the RDKit
InChIKey of OPSIN's SMILES of the name. RDKit does not read a lone pair the way it reads an
implicit hydrogen everywhere in a SMILES string: '[C@@H](C)(O)c1ccccc1' is read as
'C[C@H](O)c1ccccc1' (the implicit H of a first atom is the 'from' atom), but
'[S@@](C)(=O)c1ccccc1' as 'C[S@@](=O)c1ccccc1', the other enantiomer; some ring-closure
positions at P(III) and at an aziridine N likewise. CDK (the ``centres`` labeller) and
OPSIN's SMILES writer treat the lone pair as an implicit hydrogen. When RDKit misreads the
input and OPSIN's SMILES alike, the round trip passes for a name of the other stereoisomer:
'[S@@](C)(=O)c1ccccc1' shipped '(R)-(methanesulfinyl)benzene' (pin_verified) although
the input, read with the lone pair as an implicit hydrogen, is the (S) enantiomer --
'(S)-(methanesulfinyl)benzene (PIN)' (the Blue Book,. The exit check now
withdraws such a name (``namer._lone_pair_configuration_verified``): CDK's labels of the
input as written and of OPSIN's own SMILES of the name must agree on every compared centre.

None of these SMILES is in eval/splits/a holdout split.json (full InChIKey and connectivity).
"""
import subprocess

import pytest

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from tests.support.rt_assert import _independent_parse, assert_full_rt
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C[S@@](=O)c1ccc(CNC(=NCCc2cccc3c2OCCO3)NC2CC2)cc1",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)


pytestmark = pytest.mark.opsin_gate

# The lone-pair centre written first: RDKit reads it as C[S@@](=O)c1ccccc1 (R); CDK as
# C[S@](=O)c1ccccc1 (S).
FIRST_ATOM_SULFOXIDE = "[S@@](C)(=O)c1ccccc1"
WRONG_NAME = "(R)-(methanesulfinyl)benzene"
RIGHT_NAME = "(S)-(methanesulfinyl)benzene"
# The same (S) molecule written so that RDKit and CDK read it alike.
CANONICAL_SULFOXIDE = "C[S@](=O)c1ccccc1"
# P(III) with a ring-closure digit: RDKit misreads the input and OPSIN's SMILES of the
# name alike; the name is right (PubChem 500k).
P_RING_ROW = "CO[P@@]1OC[C@H](C)O1"
P_RING_NAME = "(2R,4S)-2-methoxy-4-methyl-1,3,2-dioxaphospholane"
# A guanidine whose name OPSIN writes as another tautomer (ZINC 500k): the check pairs the
# centres through the shared standard InChI.
TAUTOMER_ROW = "C[S@@](=O)c1ccc(CNC(=NCCc2cccc3c2OCCO3)NC2CC2)cc1"
TAUTOMER_NAME = ("N-cyclopropyl-N'-[2-(2,3-dihydro-1,4-benzodioxin-5-yl)ethyl]"
                 "-N''-({4-[(R)-methanesulfinyl]phenyl}methyl)guanidine")
# An aziridine N (ZINC 500k), and the same string with the N written first (the other
# N configuration for CDK, the same one for RDKit).
AZIRIDINE_ROW = "COC(=O)[C@H]1C[N@]1Cc1cc(-c2cccc3c2OC(F)(F)O3)ccc1F"
AZIRIDINE_FIRST = "[N@@]1(Cc2cc(-c3cccc4c3OC(F)(F)O4)ccc2F)[C@@H](C(OC)=O)C1"
AZIRIDINE_NAME = ("methyl (1R,2R)-1-{[5-(8,8-difluoro-7,9-dioxabicyclo[4.3.0]"
                  "nona-1,3,5-trien-2-yl)-2-fluorophenyl]methyl}aziridine-2-carboxylate")
# A sulfinate ester (PubChem 500k) whose name is built from a nested name of the acid,
# a SMILES RDKit wrote: only the caller's string is checked.
SULFINATE_ROW = "CCCCCCC#CCO[S@](=O)c1ccc(C)cc1"
SULFINATE_NAME = "non-2-yn-1-yl (S)-4-methylbenzene-1-sulfinate"
# A bridgehead N that RDKit perceives beyond the InChI stereo rules; OPSIN writes no
# configuration for it and nothing is compared there (PubChem 500k / ChEBI cinchona class).
CINCHONA_ROW = "C=C[C@H]1C[N@]2CC[C@H]1CC2Cc1ccnc2ccccc12"


def _opsin_own_stdinchikey(name: str) -> str:
    """OPSIN 2.9.0's own StdInChIKey of ``name`` (-ostdinchikey): no SMILES, no RDKit."""
    from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
    from orthonym.validation.opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar("2.9.0")
    proc = subprocess.run(["java", *JVM_HYGIENE_FLAGS, "-jar", jar, "-ostdinchikey"],
                          input=name + "\n", capture_output=True, text=True, timeout=120)
    return proc.stdout.strip()


def _rdkit_key(smiles: str) -> str:
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _centres(smiles: str) -> dict:
    from orthonym.perception.centres_bridge import centres_label_batch
    return centres_label_batch([smiles])[smiles]


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_first_atom_sulfoxide_ships_no_name_of_the_other_enantiomer(tier):
    # Until the standard spelling (TRIAGE.md 'Lone-pair centre written first -- naming
    # and the gate probe') this row abstained: the exit check withdrew the (R) name. The
    # engine now names the standard reading of the string, the (S) enantiomer; the
    # expectation rests on CDK's reading of the input (test below: {1: 'S'}), OPSIN's own
    # StdInChIKey of the name (no RDKit) and '(S)-(methanesulfinyl)benzene
    # (PIN)' (the Blue Book); a mutation that skips the rewrite makes it abstain again.
    namer_ = (Orthonym(style="pin") if tier == "pin"
              else Orthonym(style="pin", **_emit_tier_flags(tier)))
    res = namer_.name_tiered(FIRST_ATOM_SULFOXIDE)
    name = res.get("name")
    assert name != WRONG_NAME
    assert name == RIGHT_NAME, f"shipped {name!r}"
    assert res.get("tier") == "pin_verified"
    assert _opsin_own_stdinchikey(name) == _rdkit_key(CANONICAL_SULFOXIDE)
    assert namer._lone_pair_configuration_verified(name, FIRST_ATOM_SULFOXIDE) is True


def test_the_withdrawn_name_is_the_other_enantiomer():
    # CDK (centres) reads the input as S at the sulfur (atom 1), as it reads the
    # canonical spelling (atom 2).
    assert _centres(FIRST_ATOM_SULFOXIDE) == {1: "S"}
    assert _centres(CANONICAL_SULFOXIDE) == {2: "S"}
    # OPSIN's own keys of the two names differ; RDKit's key of the input is the (R) one,
    # which is why the RDKit round trip passed.
    wrong_key, right_key = _opsin_own_stdinchikey(WRONG_NAME), _opsin_own_stdinchikey(RIGHT_NAME)
    assert wrong_key and right_key and wrong_key != right_key
    assert _rdkit_key(FIRST_ATOM_SULFOXIDE) == wrong_key
    assert _rdkit_key(CANONICAL_SULFOXIDE) == right_key
    # OPSIN's SMILES of the wrong name, read by CDK, is R at the sulfur.
    wrong_smiles = _independent_parse(WRONG_NAME)
    assert "R" in _centres(wrong_smiles).values()
    assert namer._lone_pair_configuration_verified(WRONG_NAME, FIRST_ATOM_SULFOXIDE) is False
    assert namer._lone_pair_configuration_verified(RIGHT_NAME, FIRST_ATOM_SULFOXIDE) is True


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
def test_canonical_sulfoxide_keeps_its_name(tier):
    namer_ = (Orthonym(style="pin") if tier == "pin"
              else Orthonym(style="pin", **_emit_tier_flags(tier)))
    res = namer_.name_tiered(CANONICAL_SULFOXIDE)
    assert res.get("name") == RIGHT_NAME
    assert res.get("tier") == "pin_verified"
    assert_full_rt(res.get("name"), CANONICAL_SULFOXIDE)
    assert _opsin_own_stdinchikey(res.get("name")) == _rdkit_key(CANONICAL_SULFOXIDE)


def test_p_ring_row_misread_on_both_sides_keeps_its_name():
    res = _dt_row(P_RING_ROW)
    assert res.get("name") == P_RING_NAME
    assert_full_rt(res.get("name"), P_RING_ROW)
    # RDKit misreads the input (its key is not OPSIN's own key of the right name)...
    assert _rdkit_key(P_RING_ROW) != _opsin_own_stdinchikey(P_RING_NAME)
    #... and OPSIN's SMILES of it alike, so the round trip passes; CDK agrees the name
    # is the input's configuration.
    assert namer._lone_pair_configuration_verified(P_RING_NAME, P_RING_ROW) is True


def test_tautomer_name_pairs_through_the_standard_inchi():
    res = _dt_row(TAUTOMER_ROW)
    assert res.get("name") == TAUTOMER_NAME
    assert_full_rt(res.get("name"), TAUTOMER_ROW)
    assert namer._lone_pair_configuration_verified(TAUTOMER_NAME, TAUTOMER_ROW) is True
    flipped = TAUTOMER_NAME.replace("(R)-methanesulfinyl", "(S)-methanesulfinyl")
    assert namer._lone_pair_configuration_verified(flipped, TAUTOMER_ROW) is False


def test_aziridine_nitrogen():
    res = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(AZIRIDINE_ROW)
    assert res.get("name") == AZIRIDINE_NAME
    assert_full_rt(res.get("name"), AZIRIDINE_ROW)
    # The N written first is the other N configuration for CDK; RDKit reads the two
    # strings alike, so the round trip of the same name passes for it too.
    assert _rdkit_key(AZIRIDINE_FIRST) == _rdkit_key(AZIRIDINE_ROW)
    assert namer._lone_pair_configuration_verified(AZIRIDINE_NAME, AZIRIDINE_FIRST) is False
    first = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(AZIRIDINE_FIRST)
    assert first.get("name") is None or is_failure_name(first.get("name")), first.get("name")


def test_sulfinate_ester_is_checked_on_the_callers_string_only():
    res = _dt_row(SULFINATE_ROW)
    assert res.get("name") == SULFINATE_NAME
    assert res.get("tier") == "pin_verified"
    assert_full_rt(res.get("name"), SULFINATE_ROW)
    assert namer._lone_pair_configuration_verified(SULFINATE_NAME, SULFINATE_ROW) is True


# The D3 witness (PubChem 500k row 493267) and its name in both prefix orders
# (tests/unit/rules/test_prefix_order_fallback.py): OPSIN's own StdInChIKey is the same for
# both, but only the P-first spelling's SMILES carries a ring-closure digit at the P.
D3_P_ROW = ("Cc1cn([C@H]2CC(O[P@]3O[C@](C)(c4ccccc4)[C@@H]4CCCN43)[C@@H](CO)O2)"
            "c(=O)[nH]c1=O")
D3_P_FIRST = (
    "(2S,4R,5S)-2-{1-[(2R,4R)-4-(5-methyl-2,4-dioxo-1,3-diazacyclohex-5-en-1-yl)"
    "-2-(2-oxaethan-1-yl)-3-oxacyclopentan-1-yl]-1-oxamethan-1-yl}"
    "-4-(cyclohexa-1,3,5-trien-1-yl)-4-methyl"
    "-3-oxa-1-aza-2-phosphabicyclo[3.3.0]octane")
D3_ORDERED = (
    "(2S,4R,5S)-4-(cyclohexa-1,3,5-trien-1-yl)-4-methyl"
    "-2-{1-[(2R,4R)-4-(5-methyl-2,4-dioxo-1,3-diazacyclohex-5-en-1-yl)"
    "-2-(2-oxaethan-1-yl)-3-oxacyclopentan-1-yl]-1-oxamethan-1-yl}"
    "-3-oxa-1-aza-2-phosphabicyclo[3.3.0]octane")


def test_reads_opsins_own_smiles_not_rdkits_rewrite():
    # Both spellings name the input for CDK and for OPSIN's own key; RDKit's canonical
    # rewrite of the ordered spelling's SMILES carries RDKit's reading and would say no.
    assert _opsin_own_stdinchikey(D3_P_FIRST) == _opsin_own_stdinchikey(D3_ORDERED)
    assert namer._lone_pair_configuration_verified(D3_P_FIRST, D3_P_ROW) is True
    assert namer._lone_pair_configuration_verified(D3_ORDERED, D3_P_ROW) is True


def test_bridgehead_nitrogen_is_not_compared(monkeypatch):
    assert namer._has_lone_pair_stereocentre(CINCHONA_ROW) is False
    calls = []
    monkeypatch.setattr(namer, "_validity_gate_name_to_opsin_smiles",
                        lambda n: calls.append(n))
    assert namer._lone_pair_configuration_verified("anything", CINCHONA_ROW) is True
    assert namer._lone_pair_configuration_verified("anything", "C[C@H](O)CC") is True
    assert calls == []


def test_fails_closed_without_opsin_smiles(monkeypatch):
    monkeypatch.setattr(namer, "_validity_gate_name_to_opsin_smiles", lambda n: None)
    assert namer._lone_pair_configuration_verified(RIGHT_NAME, CANONICAL_SULFOXIDE) is False


def test_fails_closed_without_centres(monkeypatch):
    import orthonym.perception.centres_bridge as cb
    monkeypatch.setattr(cb, "centres_label_batch", lambda smiles, timeout=120.0: None)
    assert namer._lone_pair_configuration_verified(RIGHT_NAME, CANONICAL_SULFOXIDE) is False
