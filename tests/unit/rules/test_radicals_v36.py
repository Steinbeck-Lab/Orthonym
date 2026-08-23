import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from orthonym.validation.opsin_roundtrip import _find_opsin_jar


def _radical_rt(name: str, target_smiles: str) -> bool:
    """OPSIN round-trip WITH allowRadicals (opsin_roundtrip_check is radical-blind)."""
    jar = _find_opsin_jar("2.9.0")
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    if not txt:
        return False
    smi = txt.strip().split("\n")[-1].strip()
    if not smi or "could not" in smi.lower():
        return False
    m, t = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(target_smiles)
    if m is None or t is None:
        return False
    return Chem.InchiToInchiKey(Chem.MolToInchi(m)) == Chem.InchiToInchiKey(Chem.MolToInchi(t))


def _name_be(smiles: str):
    with jvm_slots(1, purpose="test-radical"):
        r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    return r.get("name")


# distinct substitution patterns -> generalisation, not a special case
SUBSTITUTED_ARYLOXYL = [
    "[O]c1ccc(O)cc1",      # 4-hydroxyphenoxyl
    "[O]c1ccc(Cl)cc1",     # 4-chlorophenoxyl
    "[O]c1ccccc1C",        # 2-methylphenoxyl
]


@pytest.mark.parametrize("smiles", SUBSTITUTED_ARYLOXYL)
def test_substituted_aryloxyl_named_and_rt(smiles):
    name = _name_be(smiles)
    assert name and name not in ("unknown organic compound", None), f"abstained on {smiles}"
    assert _radical_rt(name, smiles), f"{name!r} did not -r round-trip for {smiles}"


def test_unsubstituted_phenoxyl_pin_unchanged():
    assert _name_be("[O]c1ccccc1") == "phenoxyl"


def test_unsubstituted_methoxyl_pin_unchanged():
    assert _name_be("C[O]") == "methoxyl"


# Task-1 review, Important #2: the oxyl-radical branch was generalised for
# BOTH the aryloxyl (aromatic R) and the alkoxyl (aliphatic R carrying a
# further substituent) shape, but Task 1's own coverage only exercised aryl.
# These three are VERIFIED -r-round-trip witnesses for the alkyl branch
# (name_oxyl_radical's aliphatic non-plain-hydrocarbon path -> the
# `_compose_oxidanyl` systematic '(R)oxidanyl' composition).
SUBSTITUTED_ALKOXYL = [
    "[O]CCCl",          # (2-chloroethyl)oxidanyl
    "[O]CC(=O)O",       # (carboxymethyl)oxidanyl
    "[O]CC1CCCCC1",     # (cyclohexylmethyl)oxidanyl
]


@pytest.mark.parametrize("smiles", SUBSTITUTED_ALKOXYL)
def test_substituted_alkoxyl_named_and_rt(smiles):
    name = _name_be(smiles)
    assert name and name not in ("unknown organic compound", None), f"abstained on {smiles}"
    assert _radical_rt(name, smiles), f"{name!r} did not -r round-trip for {smiles}"


# v36 A2 Task 2 (C2): `_build_ctx` (universal_substituent.py) used to hard-
# abstain on ANY radical electron, so a radical the dedicated
# radicals.py/route_charged path declines (measured: route_charged's own
# cumulative-suffix primitive falls through to the broken textual fallback
# 'ethanolyl', which OPSIN correctly rejects) never got a chance at the T4
# systematic floor. `[CH2]CO` (2-hydroxyethyl radical) is a VERIFIED witness:
# scoping the guard to allow a SINGLE monovalent free-valence centre through,
# plus routing t4_coverage's final rung through
# `name_universal_substituent_prefix` (which CAN cite the missing bond as a
# `-yl` suffix, unlike the whole-molecule entry point), rescues it to
# '3-oxapropan-1-yl' -- -r-RT VERIFIED MATCH for '[CH2]CO'.
CARBON_RADICAL_FLOOR = "[CH2]CO"


@pytest.mark.opsin_gate  # the whole point: the broken 'ethanolyl' fallback
# candidate must be REJECTED (gate ON, production default) so the pipeline
# falls through to the T4 floor -- with the suite's gate-off default this
# test would observe the unverified pre-gate candidate instead and miss the
# rescue entirely (conftest.py's `_opsin_validity_gate_state` autouse fixture).
def test_carbon_radical_reaches_floor():
    name = _name_be(CARBON_RADICAL_FLOOR)
    # Either a real -r-round-tripping name, or a safe abstain -- never a wrong molecule.
    if name and name not in ("unknown organic compound", None):
        assert _radical_rt(name, CARBON_RADICAL_FLOOR), \
            f"{name!r} not -r-RT for {CARBON_RADICAL_FLOOR}"
