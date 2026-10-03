"""The default tier's emission rule (user decision 2026-09-30, "Ship it in 1.0.2").

The submitted paper, Methods, "Tiers" (L73): "The default configuration emits a name
only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
declines." "The label pin_verified means the strict PIN path built the name and
verified it." "A decline carries a reason". Exceptions (L75): "names derived from the
natural-product and metal-complex lists... are emitted based solely on their exact
structural match. At the default tier, ten further name formats absent from OPSIN's
grammar (e.g., inositols, phanes, and thioperoxols) are emitted without parse-back
validation"; checks (L74): "At the default tier, it compares only the constitutional
part of the key".

So at ``--emit-tier pin`` a name is emitted only when it is pin_verified or one of
those exceptions; every other name is declined with the reason code NO_VERIFIED_PIN,
on every public entry point, and the wider tiers keep it. Every emitted or kept name
here is read back by a fresh OPSIN call that does not go through the engine
(``tests.support.rt_assert``), by full InChIKey, except the list and grammar-gap
names OPSIN cannot read. Dev-set rows, Blue Book rows and minimal analogues only.
"""
import os
import subprocess
import sys

import pytest

from orthonym import Orthonym, OrthonymLimitError, classify_limit, name_compound
from orthonym.errors import is_failure_name
from tests.support.default_tier import (
    assert_best_effort_gives,
    assert_default_tier_declines,
    strict_path_row,
)
from tests.support.rt_assert import assert_full_rt, name_best_effort
from tests.unit.rules.test_d1_coordination_v36 import HEME_B

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

# '(trimethylazaniumyl)acetate': the code records a carbon-substituted N+ cited as
# 'azaniumyl' as not the PIN, the Blue Book,
# '(N,N-dimethylmethanaminiumyl)acetate (PIN) (trimethylammoniumyl)acetate').
BETAINE = "C[N+](C)(C)CC(=O)[O-]"
BETAINE_NAME = "(trimethylazaniumyl)acetate"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_a_pin_verified_name_is_emitted():
    row = Orthonym().name_tiered(CAFFEINE)
    assert row["name"] == "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione", row
    assert row["tier"] == "pin_verified" and row["is_pin"] and row["limit_code"] is None
    assert_full_rt(row["name"], CAFFEINE)
    assert name_compound(CAFFEINE) == row["name"]


def test_a_name_the_code_records_as_not_the_pin_is_declined_everywhere():
    row = assert_default_tier_declines(BETAINE)
    assert row["source"] == "abstain" and row["gate_outcome"] == "suppressed", row
    assert row["formula"] == "C5H11NO2", row
    # the plain call, name_compound, raise_on_limit, classify_limit, confidence
    assert is_failure_name(Orthonym().name(BETAINE))
    assert name_compound(BETAINE) == row["name"]
    with pytest.raises(OrthonymLimitError) as err:
        Orthonym().name(BETAINE, raise_on_limit=True)
    assert err.value.code == "NO_VERIFIED_PIN"
    assert classify_limit(BETAINE).code == "NO_VERIFIED_PIN"
    conf = Orthonym().name_with_confidence(BETAINE)
    assert is_failure_name(conf["name"]) and conf["limit"]["code"] == "NO_VERIFIED_PIN"
    # the strict path still builds it, labelled below the PIN, and best-effort keeps it
    strict = strict_path_row(BETAINE)
    assert strict["name"] == BETAINE_NAME and strict["tier"] == "systematic_verified"
    be = assert_best_effort_gives(BETAINE, BETAINE_NAME)
    assert be["tier"] == "systematic_verified" and not be["is_pin"], be


def test_a_name_only_a_breadth_producer_built_is_declined():
    # the PIN tier's promotion re-run (pin_unverified: "a name in PIN form that only a
    # breadth producer built"); (the Blue Book)
    smi = "NCc1csc(-c2cccs2)n1"
    assert_default_tier_declines(smi)
    assert strict_path_row(smi)["tier"] == "pin_unverified"
    assert_best_effort_gives(smi, "1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine")


@pytest.mark.parametrize("smiles,expected", [
    ("CSO", "methane-SO-thioperoxol"),                                 # grammar gap
    ("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "scyllo-inositol"),
    ("C1Cc2ccc(cc2)CCc2ccc1cc2", "1,4(1,4)-dibenzenacyclohexaphane"),
    ("CCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](N)CO", "(4E)-sphing-4-enine"),  # NP list
])
def test_the_exact_construction_exceptions_are_emitted(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert row["name"] == expected, row
    assert row["gate_outcome"].startswith("carveout:"), row
    assert row["limit_code"] is None, row
    # a grammar-gap name is labelled below the PIN; a natural-product list name takes
    # the paper run's label, pin_verified (verified 'identity')
    if row["gate_outcome"] == "carveout:np_stereoparent":
        assert row["tier"] == "pin_verified" and row["verified"] == "identity", row
    else:
        assert row["tier"] != "pin_verified", row


def test_the_metal_complex_list_name_is_emitted():
    row = Orthonym().name_tiered(HEME_B["smiles"])
    assert row["name"] == HEME_B["name"] and row["verified"] == "identity", row
    assert row["tier"] == "pin_verified", row


def test_a_strict_path_pin_verified_by_its_constitution_is_emitted():
    # lowercase pseudoasymmetry descriptors, which OPSIN cannot read
    smi = "O[C@H]1CC[C@@H](O)CC1"
    row = Orthonym().name_tiered(smi)
    assert row["name"] == "(1s,4s)-cyclohexane-1,4-diol", row
    assert row["gate_outcome"] == "self_consistency_constitution_only", row
    assert row["verified"] == "opsin_constitution" and row["limit_code"] is None, row


def test_a_retained_trivial_name_is_emitted_only_with_trivial_fallback():
    # 'diethylstilbestrol': no Blue Book hit, and the strict path builds no PIN for it.
    # This was 'bisphenol a' until PIN class program batch 2 (Task 10) named that one
    # '4,4'-(propane-2,2-diyl)diphenol' at the default tier, the Blue Book).
    smi = "CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1"
    assert_default_tier_declines(smi)
    row = Orthonym(trivial_fallback=True).name_tiered(smi)
    assert row["name"] == "diethylstilbestrol" and row["source"] == "trivial_retained", row
    assert_full_rt(row["name"], smi)


def test_the_wider_tiers_keep_every_name_and_label():
    be = name_best_effort(CAFFEINE)
    assert be["tier"] == "pin_verified", be          # the strict twin still agrees
    be = name_best_effort(BETAINE)
    assert be["name"] == BETAINE_NAME, be


def test_the_rule_is_off_when_the_gate_is_disabled(monkeypatch):
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_DISABLE_VALIDITY_GATE", True)
    assert Orthonym().name(BETAINE) == BETAINE_NAME


def test_the_cli_default_tier_prints_the_decline():
    env = dict(os.environ)
    out = subprocess.run([sys.executable, "-m", "orthonym", BETAINE, "--provenance"],
                         capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert '"limit_code": "NO_VERIFIED_PIN"' in out.stdout, out.stdout


# The --help text of --emit-tier and --trivial, pinned word for word: the guide and the
# README quote it (the paper's words; guide_sentences Part B).
EMIT_TIER_HELP = (
    "Which names to return. pin (the default): a name only when the pipeline can "
    "build the preferred IUPAC name (PIN), that is when the strict PIN path built the "
    "name and verified it (tier pin_verified). The exceptions are names from the "
    "natural-product and metal-complex lists, the name formats absent from OPSIN's "
    "grammar (for example inositols, phanes and thioperoxols), PINs whose "
    "stereodescriptors OPSIN cannot read, for which the default tier compares the "
    "constitution, and, with --trivial, a retained trivial name. Otherwise it "
    "declines, with the reason code NO_VERIFIED_PIN when it built a name that is not "
    "a verified PIN. The rule applies to the default --style pin. valid: also names "
    "from the general "
    "engine. complete: also general names for aromatic and heterocyclic ring systems. "
    "best-effort: also the last-resort producers, von Baeyer and spiro names for "
    "ring systems of up to 100 skeletal atoms and 11 rings (the other tiers build "
    "them up to 40 atoms and 8 rings), and adducts with a one-atom ion. full-coverage: "
    "also the "
    "coordination-name builder for metal tetrapyrrole and corrin complexes, which "
    "builds a name or declines. The wider tiers also return the names the default "
    "tier declines, each labelled with its tier; there every name must pass a "
    "full-InChIKey OPSIN round trip, except a name from the natural-product and "
    "metal-complex lists, so the name formats absent from OPSIN's grammar are "
    "declined there. --provenance shows each name's tier.")
TRIVIAL_HELP = (
    "When no preferred name can be built, also allow a retained trivial name that is "
    "not a preferred name. A preferred name that can be built is never replaced "
    "(glycerol stays propane-1,2,3-triol). Without this option, two kinds of retained "
    "trivial name are still used at the wider tiers, and the default tier declines "
    "them: names from a small last-resort table, and the trivial natural-product names "
    "of molecules whose preferred bridged fused name the engine does not build yet "
    "(diamorphine). --provenance labels both systematic_verified, source "
    "trivial_retained. A natural-product name built on a parent, such as "
    "'(9R,13S,14S)-3-methoxy-17-methylmorphinan', is not a trivial name, and "
    "--trivial does not return it at the default tier.")


def test_the_help_text_says_what_the_default_tier_does(capsys, monkeypatch):
    from orthonym.cli import main
    monkeypatch.setenv("COLUMNS", "100000")   # argparse wraps at the terminal width
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert EMIT_TIER_HELP in text
    assert TRIVIAL_HELP in text


# The decline text of NO_VERIFIED_PIN (the paper, L73: "A decline carries a reason"):
# the engine built a name for the structure, so its elements are supported; the label
# is the NO_VERIFIED_PIN text 'unknown organic compound' at every entry point, never
# the UNSUPPORTED_ELEMENT text '<metal> compound (not supported)'. 'ethenylsodium' is a
# Group 1 organometallic name, the Blue Book: no PIN noted), declined.
ETHENYLSODIUM = "C=C[Na]"


def test_a_declined_metal_compound_does_not_say_its_element_is_unsupported(capsys):
    from orthonym import name_with_tree
    from orthonym.cli import main
    label = "unknown organic compound"
    row = assert_default_tier_declines(ETHENYLSODIUM)
    assert row["name"] == label, row
    assert Orthonym().name(ETHENYLSODIUM) == label
    assert name_compound(ETHENYLSODIUM) == label
    assert name_compound(ETHENYLSODIUM, include_confidence=True)["name"] == label
    assert name_with_tree(ETHENYLSODIUM).name == label
    with pytest.raises(OrthonymLimitError) as err:
        Orthonym().name(ETHENYLSODIUM, raise_on_limit=True)
    assert (err.value.code, err.value.message) == ("NO_VERIFIED_PIN", label)
    lim = classify_limit(ETHENYLSODIUM)
    assert (lim.code, lim.message) == ("NO_VERIFIED_PIN", label)
    assert main([ETHENYLSODIUM]) == 0
    assert capsys.readouterr().out.strip() == label
    # the wider tiers keep the name
    assert_best_effort_gives(ETHENYLSODIUM, "ethenylsodium")


def test_a_true_unsupported_element_decline_keeps_its_own_text():
    # the covalent drawing [Na]Cl: no name is built, the reason is UNSUPPORTED_ELEMENT
    row = Orthonym().name_tiered("[Na]Cl")
    assert row["limit_code"] == "UNSUPPORTED_ELEMENT", row
    assert row["name"] == "sodium compound (not supported)", row
    assert name_compound("[Na]Cl") == "sodium compound (not supported)"


# --- review fixes (delta review of the default-tier rule) -------------------------

def test_the_rule_applies_to_the_pin_style_only():
    # the paper's default configuration is the PIN style; '--style general' keeps its
    # general forms
    assert Orthonym(style="general").name(ETHENYLSODIUM) == "ethenylsodium"
    assert is_failure_name(Orthonym(style="pin").name(ETHENYLSODIUM))


_REDUCED = r"""
import os, sys, json
os.environ["ORTHONYM_ALLOW_REDUCED"] = "1"
os.environ["ORTHONYM_NO_DOWNLOAD"] = "1"
os.environ["ORTHONYM_JAR_DIR"] = sys.argv[1]
for k in ("ORTHONYM_OPSIN_JAR", "ORTHONYM_CENTRES_JAR"):
    os.environ.pop(k, None)
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import _find_opsin_jar
out = {"jar": _find_opsin_jar()}
for s in sys.argv[2:]:
    r = Orthonym().name_tiered(s)
    out[s] = [r["name"], r["tier"], r["gate_outcome"], r["limit_code"]]
print(json.dumps(out))
"""


def test_the_rule_applies_in_the_reduced_mode_without_a_jar(tmp_path):
    # the user's answer (a) does not depend on the jar: a name the code records as
    # not the PIN is declined; a strict-path name in PIN form ships unverified
    import json
    env = {k: v for k, v in os.environ.items()
           if k not in ("ORTHONYM_REQUIRE_JARS", "ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE")}
    r = subprocess.run([sys.executable, "-c", _REDUCED, str(tmp_path), ETHENYLSODIUM,
                        BETAINE, "CCO"], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["jar"] is None, out
    for smi in (ETHENYLSODIUM, BETAINE):
        name, tier, _gate, code = out[smi]
        assert tier == "abstain" and code == "NO_VERIFIED_PIN", (smi, out[smi])
        assert name == "unknown organic compound", (smi, out[smi])
    assert out["CCO"] == ["ethanol", "pin_unverified", "unavailable", None], out


GERMACRANE = "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1"
GRAYANOTOXANE_ROW = "C[C@H]1C[C@@]23CC[C@H]4[C@@H](CCC4(C)C)[C@H]([C@@H]2CC[C@@H]1C3)C"


def test_a_natural_product_list_name_is_labelled_and_emitted_as_in_the_paper_run():
    from orthonym.cli import _emit_tier_flags
    from tests.support.rt_assert import _independent_parse
    # the paper run (ChEBI, germacrane) labelled it pin_verified; L74: the exact-match
    # list names are the exceptions of the best-effort tier
    for tier in ("pin", "valid", "best-effort"):
        n = Orthonym() if tier == "pin" else Orthonym(**_emit_tier_flags(tier))
        row = n.name_tiered(GERMACRANE)
        assert row["name"] == "germacrane", (tier, row)
        assert row["tier"] == "pin_verified" and row["verified"] == "identity", (tier, row)
        assert row["gate_outcome"] == "carveout:np_stereoparent", (tier, row)
    assert _independent_parse("germacrane") is None      # OPSIN cannot read it
    # at the wider tiers the list name is the last resort only: a verified name wins
    be = name_best_effort(GRAYANOTOXANE_ROW)
    assert be["name"] != "grayanotoxane" and be["verified"] == "opsin", be
    assert_full_rt(be["name"], GRAYANOTOXANE_ROW)
    assert Orthonym().name_tiered(GRAYANOTOXANE_ROW)["tier"] == "pin_verified"


def test_a_polyacid_carve_out_does_not_pre_empt_a_verifying_round_trip():
    smi = "OS(=O)(=O)SS(=O)(=O)O"
    row = Orthonym().name_tiered(smi)
    assert row["name"] == "2-thiodisulfuric acid", row
    assert row["gate_outcome"] == "self_consistency_verified", row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert_full_rt(row["name"], smi)
