"""v30 — an added-carbon multi-suffix parent (>=3 -carboxylic acid on an acyclic core,
`propane-1,2,3-tricarboxylic acid`) must also carry SIMPLE substituents on the core as
prefixes, so citric-acid-family metabolites name at PIN. Previously `name_added_carbon_parent`
failed closed on ANY extra substituent (`return None`), and the molecule fell to a wrong
pentanedioic-chain candidate → abstain.

Targets VERIFIED RT-exact in OPSIN (2-hydroxypropane-1,2,3-tricarboxylic acid == citric acid).
Un-nameable substituents still fail closed (0-wrong).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _pin():
    return Orthonym(style="pin")


@pytest.mark.parametrize("smi,expected", [
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O", "2-hydroxypropane-1,2,3-tricarboxylic acid"),   # citric
    ("OC(=O)C(O)C(CC(=O)O)C(=O)O", "1-hydroxypropane-1,2,3-tricarboxylic acid"),   # isocitric
    ("OC(=O)C(O)C(O)(CC(=O)O)C(=O)O", "1,2-dihydroxypropane-1,2,3-tricarboxylic acid"),  # hydroxycitric (2 subs)
    ("OC(=O)CC(N)(CC(=O)O)C(=O)O", "2-aminopropane-1,2,3-tricarboxylic acid"),
    ("OC(=O)CC(Cl)(CC(=O)O)C(=O)O", "2-chloropropane-1,2,3-tricarboxylic acid"),
])
def test_substituted_added_carbon_tricarboxylic_acid(smi, expected):
    assert _pin().name(smi) == expected


def test_citrate_ion_composes_with_substituent():
    # the charge layer composes with the new substituted added-carbon parent
    assert _pin().name("[O-]C(=O)CC(O)(CC(=O)[O-])C(=O)[O-]") == "2-hydroxypropane-1,2,3-tricarboxylate"


def test_unsubstituted_added_carbon_unchanged():
    # bare added-carbon parents must stay byte-identical (no regression)
    assert _pin().name("OC(=O)CC(CC(=O)O)C(=O)O") == "propane-1,2,3-tricarboxylic acid"
    assert _pin().name("OC(=O)C(C(=O)O)C(=O)O") == "methanetricarboxylic acid"
    assert _pin().name("NC(=O)C(C(=O)N)C(=O)N") == "methanetricarboxamide"


def test_added_carbon_substituent_path_declines_complex_substituents():
    """The added-carbon substituent path names ONLY hetero substituents name_substituent
    handles (hydroxy/amino/halo); it DECLINES a substituent with carbons (rejected by the
    skeleton==chain check) or one name_substituent cannot name (phosphonooxy). At PIN these
    abstain (my path declines; SELF-01 suppresses the pre-existing pentane-path candidate) —
    production 0-wrong. (The pentane path's gate-off `...pentanetrioic acid` is the pre-existing
    #37-class defect, A/B-identical with/without this change, tracked separately.)"""
    # Isolate MY path: it must NOT emit a `...tricarboxylic acid` name for these (it
    # declined). The pytest env runs SELF-01 off, so the FULL pipeline may still show the
    # pre-existing pentane-path `...pentanetrioic acid` — that is NOT this path and is
    # A/B-identical with/without this change (verified). In a real process (SELF-01 on)
    # both abstain.
    for smi in ("OC(=O)CC(OP(=O)(O)O)(CC(=O)O)C(=O)O",
                "OC(=O)CC(NS(=O)(=O)c1ccc(N)cc1)(CC(=O)O)C(=O)O"):
        n = _pin().name(smi) or ""
        assert "tricarboxylic" not in n, f"my added-carbon path must decline, got {n}"


# ---- fable review 7daf8b68 findings, now fixed ----

def test_fable_b1_isomer_constitution_guard():
    """BLOCKER 1: name_substituent's symbols-only fallback mis-named a nitrite
    -O-N=O as 'nitro' (same {N,O,O} count). The gate-independent constitution
    re-anchor must reject it -> the added-carbon path does NOT emit a nitro name."""
    for smi in ("OC(=O)CC(ON=O)(CC(=O)O)C(=O)O", "OC(=O)CC(N(O)O)(CC(=O)O)C(=O)O"):
        n = _pin().name(smi) or ""
        assert "nitropropane" not in n and "tricarboxylic" not in n, n


def test_fable_b2_p14_4_g_tiebreak_deterministic():
    """BLOCKER 2: P-14.4(g) — lowest locant to the alphabetically-first substituent;
    the same molecule must get ONE name regardless of SMILES atom order."""
    a = _pin().name("NC(C(=O)O)C(C(=O)O)C(O)C(=O)O")
    b = _pin().name("OC(C(=O)O)C(C(=O)O)C(N)C(=O)O")
    assert a == b == "1-amino-3-hydroxypropane-1,2,3-tricarboxylic acid", (a, b)


def test_fable_b3_stereo_defined_declined_by_added_carbon_path():
    """BLOCKER 3: a flat added-carbon name cannot express stereo, so the added-carbon
    path must DECLINE a stereo-defined input (isolate it: no `...tricarboxylic acid`
    output). At PIN (SELF-01 on) the molecule abstains; the pytest env runs SELF-01
    off, so the pre-existing pentane path may still surface its (stereo) pentanetrioic
    candidate — that is NOT this path and abstains under SELF-01."""
    n = _pin().name("OC(=O)[C@H](O)[C@@H](CC(=O)O)C(=O)O") or ""
    assert "tricarboxylic" not in n, n


def test_fable_r6_mononuclear_locant_omitted():
    """RISK 6 / P-14.3.4.2(a): locant '1' omitted on a substituted mononuclear core."""
    assert _pin().name("OC(C(=O)O)(C(=O)O)C(=O)O") == "hydroxymethanetricarboxylic acid"
