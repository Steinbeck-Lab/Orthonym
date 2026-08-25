"""v37: general acyloxy-ester substituent naming (generalize the SP2.1' fix).

STEP-1 spy (invariant 8) findings that scope these tests:
  * The oxa-replacement chain form ('...-2-oxo-1-oxabutyl') is OPSIN-VALID and
    round-trips EXACT -- the SP2.1' 'OPSIN-grammar-invalid' premise is refuted.
  * On the BEST-EFFORT tier a systematic acyloxy substituent already emits that
    RT-valid oxa-chain (ugly but correct) -- a spelling gap, not a breadth gap.
  * On the DEFAULT/PIN tier it ABSTAINS ('unknown organic compound') because
    Tier-1.95 only handles RETAINED acyls (a static cache); systematic acyls fall
    through and the fragment is dropped -- a genuine breadth gap.

The fix routes a plain acyloxy ester '-O-C(=O)-R' (systematic, non-retained)
through the SAME recognizer SP2.1' used (composer._acyloxy_prefix_for_frag ->
rules.lipids._acyloxy_for_site), producing the P-65.6.3.2.3 '<acyl>oxy' prefix on
ALL tiers. 0-wrong is preserved by the downstream RT/SELF-01 gate + the B2 stereo
guard; retained acyls and pure-ether oxa chains are byte-identical (fire-only-when-
differs + fail-closed recognizer).
"""
import re
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check


# The SELF-01 OPSIN validity gate is DISABLED by default in the test harness
# (conftest autouse `_opsin_validity_gate_state`); real deployment runs it ON.
# 0-wrong is a property of the gate-ON config, so every test here opts in -- it
# skips (never silently passes) if the OPSIN jar is absent.
pytestmark = pytest.mark.opsin_gate


# oxa-replacement-chain signature that the OLD path emitted for a systematic acyloxy
_OXA_CHAIN = re.compile(r"oxo-\d+-oxa|dioxa\w*yl\b|oxa\w*yl\)")


def _rt_exact(smiles: str, name: str) -> bool:
    return bool(name) and "unknown" not in name and \
        opsin_roundtrip_check(smiles, name)["passed"]


def _name_default(smi: str) -> str:
    return Orthonym().name(smi)


def _name_besteffort(smi: str) -> str:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True).name(smi)


# --------------------------------------------------------------------------- #
# DELIVERABLE 1 — default-tier breadth rescue: abstain -> preferred acyloxy PIN.
# RING parents (benzene / saturated) correctly attribute the acyl's own OH inside
# the acyloxy prefix, so these emit on the strict DEFAULT tier. (The acyl here --
# 3-hydroxy-3-methylbutanoic -- is genuinely SYSTEMATIC: not in FRAGMENT_NAME_CACHE,
# so it abstained on HEAD; my fix supplies the '<acyl>oxy' PIN prefix.)
# --------------------------------------------------------------------------- #
# (smiles, acyl-oxy substring that MUST appear in the preferred name)
DEFAULT_WITNESSES = [
    # a benzene-ring parent with a systematic acyloxy on the ring
    ("OC(=O)c1ccc(OC(=O)CC(C)(C)O)cc1", "3-hydroxy-3-methylbutanoyloxy"),
    # a saturated ring parent
    ("O=C(O)C1CCC(OC(=O)CC(C)(C)O)CC1", "3-hydroxy-3-methylbutanoyloxy"),
]


@pytest.mark.parametrize("smi,acyloxy", DEFAULT_WITNESSES)
def test_default_tier_systematic_acyloxy_is_named_and_rt_exact(smi, acyloxy):
    name = _name_default(smi)
    assert _rt_exact(smi, name), f"{smi} -> {name!r} not RT-exact"
    assert acyloxy in name, f"{smi} -> {name!r} lacks preferred acyloxy {acyloxy!r}"
    assert not _OXA_CHAIN.search(name), f"{smi} -> {name!r} still an oxa-chain"


# --------------------------------------------------------------------------- #
# DELIVERABLE 2 — best-effort spelling improvement (the breadth metric the harness
# uses): a systematic acyloxy substituent used to ship the ugly-but-RT-valid
# oxa-replacement chain ('...-2-oxo-1-oxabutyl'); now the preferred '<acyl>oxy'.
# Covers CHAIN parents (whose strict-default polyfunctional path has a SEPARATE
# pre-existing acyl-FG double-count bug -- out of scope, 0-wrong holds) + a STEREO
# acyl (descriptor must survive) + a halo acyl.
# --------------------------------------------------------------------------- #
BEST_EFFORT_WITNESSES = [
    ("OC(=O)CCCCCOC(=O)CC(C)(C)O", "3-hydroxy-3-methylbutanoyloxy"),  # chain
    ("OC(=O)CCCCCOC(=O)[C@](C)(O)CCl", "chloro"),   # stereo acyl
    ("OC(=O)CCCCOC(=O)CCl", "chloro"),              # chloroacetyloxy
]


@pytest.mark.parametrize("smi,frag", BEST_EFFORT_WITNESSES)
def test_besteffort_systematic_acyloxy_prefers_acyloxy_form(smi, frag):
    name = _name_besteffort(smi)
    assert _rt_exact(smi, name), f"{smi} -> {name!r} not RT-exact"
    assert frag in name, f"{smi} -> {name!r} lacks expected acyloxy marker {frag!r}"
    assert "oyloxy" in name or "acetyloxy" in name, \
        f"{smi} -> {name!r} not in acyloxy form"
    assert not _OXA_CHAIN.search(name), f"{smi} -> {name!r} still an oxa-chain"


# --------------------------------------------------------------------------- #
# NEVER-WRONG — every new emit must round-trip (0-wrong absolute)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smi", [w[0] for w in DEFAULT_WITNESSES + BEST_EFFORT_WITNESSES])
def test_never_wrong_default_and_besteffort(smi):
    for name in (_name_default(smi), _name_besteffort(smi)):
        if name and "unknown" not in name:
            assert opsin_roundtrip_check(smi, name)["passed"], \
                f"{smi} -> {name!r} is a WRONG molecule"


# --------------------------------------------------------------------------- #
# CONTROLS — retained acyls + pure-ether a-replacement stay BYTE-IDENTICAL
# --------------------------------------------------------------------------- #
BYTE_IDENTICAL = [
    ("COCCOCCOC", "2,5,8-trioxanonane"),                 # a-replacement PIN (no carbonyl)
    ("CC(=O)OCC", "ethyl acetate"),                      # retained acyl, functional class
    ("CC(=O)Oc1ccccc1", "phenyl acetate"),
    ("O=C(Oc1ccccc1)c1ccccc1", "phenyl benzoate"),
    ("OC(=O)CCCCCOC(=O)C", "6-(acetyloxy)hexanoic acid"),  # retained acyloxy prefix
]


@pytest.mark.parametrize("smi,expected", BYTE_IDENTICAL)
def test_controls_byte_identical(smi, expected):
    assert _name_default(smi) == expected


# --------------------------------------------------------------------------- #
# NO REGRESSION — the SP2.1' spiro-VB witnesses still name (full-InChIKey RT)
# --------------------------------------------------------------------------- #
SPIRO_VB_WITNESSES = [
    "C=C1C(=O)O[C@H]2[C@H]1[C@@H](OC(=O)[C@](C)(O)CCl)CC(=C)[C@@H]1C[C@H](O)[C@@]3(CO3)[C@H]21",
    "CC(=O)OCC12CC(OC(=O)CC(C)(C)O)C(C)=CC1OC1C(O)C(OC(C)=O)C2(C)C12CO2",
]


@pytest.mark.parametrize("smi", SPIRO_VB_WITNESSES)
def test_sp21prime_spiro_vb_still_named(smi):
    name = _name_besteffort(smi)
    assert _rt_exact(smi, name), f"SP2.1' spiro-VB witness regressed: {smi} -> {name!r}"
