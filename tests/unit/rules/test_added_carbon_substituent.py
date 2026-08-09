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


@pytest.mark.parametrize("smi,expected", [
    # BLOCKER 3 resolved into EXPRESSION: chain R/S stereocentres are now emitted with
    # locants in the chosen numbering. Natural chiral TCA metabolites name RT-exact.
    ("OC(=O)[C@H](O)[C@@H](CC(=O)O)C(=O)O", "(1R,2R)-1-hydroxypropane-1,2,3-tricarboxylic acid"),
    ("O[C@@H]([C@@H](CC(O)=O)C(O)=O)C(O)=O", "(1S,2R)-1-hydroxypropane-1,2,3-tricarboxylic acid"),  # D-isocitric
    ("OC(=O)[C@@H](O)[C@](O)(CC(=O)O)C(=O)O", "(1S,2R)-1,2-dihydroxypropane-1,2,3-tricarboxylic acid"),
])
def test_chain_stereocentres_expressed(smi, expected):
    assert _pin().name(smi) == expected


def test_off_chain_stereo_fails_closed():
    """A stereocentre OFF the parent chain (inside a substituent) can't be mapped to a
    chain locant -> the added-carbon path declines (no `...tricarboxylic acid` output);
    at PIN the molecule abstains rather than shipping unexpressed stereo."""
    n = _pin().name("OC(=O)CC(O[C@@H](C)CC)(CC(=O)O)C(=O)O") or ""
    assert "tricarboxylic" not in n, n


def test_fable_r6_mononuclear_locant_omitted():
    """RISK 6 / P-14.3.4.2(a): locant '1' omitted on a substituted mononuclear core."""
    assert _pin().name("OC(C(=O)O)(C(=O)O)C(=O)O") == "hydroxymethanetricarboxylic acid"


# ---- stereo fable review (a4240802) findings, now fixed ----

def test_stereo_fable_b1_p14_4_j_tiebreak():
    """BLOCKER: P-14.4(j) — when suffix/substituent locants tie, the lower locant goes
    to the preferred CIP descriptor (R over S). A meso molecule must get ONE PIN name
    regardless of input atom order (was nondeterministic (2S,3R) vs (2R,3S))."""
    a = _pin().name("OC(=O)C[C@H](C(O)=O)[C@H](C(O)=O)CC(O)=O")
    b = _pin().name("OC(=O)C[C@@H](C(O)=O)[C@@H](C(O)=O)CC(O)=O")
    assert a == b == "(2R,3S)-butane-1,2,3,4-tetracarboxylic acid", (a, b)


def test_stereo_pseudoasymmetric_fails_closed():
    """RISK: a pseudo-asymmetric r/s centre yields an OPSIN-unparseable name that would
    ship stereo-unvalidated via the carve-out; fail closed (no `...tricarboxylic` output)."""
    n = _pin().name("OC(=O)C[C@H](O)[C@H](C(O)=O)[C@H](O)CC(O)=O") or ""
    assert "carboxylic" not in n, n


# ---- aconitic family: UNSATURATED added-carbon polycarboxylic core (v30 breadth) ----

@pytest.mark.parametrize("smi,expected", [
    # flat (no defined geometry) -> constitutional name, no descriptor
    ("OC(=O)CC(=CC(=O)O)C(=O)O",       "prop-1-ene-1,2,3-tricarboxylic acid"),
    # trans-aconitic: core C=C is CIP-E (bond _CIPCode 'E') -> (1E)
    ("OC(=O)C/C(=C\\C(=O)O)C(=O)O",    "(1E)-prop-1-ene-1,2,3-tricarboxylic acid"),
    # cis-aconitic: core C=C is CIP-Z -> (1Z)
    ("OC(=O)C/C(=C/C(=O)O)C(=O)O",     "(1Z)-prop-1-ene-1,2,3-tricarboxylic acid"),
])
def test_aconitic_unsaturated_added_carbon(smi, expected):
    # BB: fumaric acid = (2E)-but-2-enedioic acid (PIN) -> the E/Z descriptor carries
    # the ene locant even for a single double bond. Aconitic mirrors that.
    assert _pin().name(smi) == expected


def test_aconitic_numbering_deterministic_by_atom_order():
    # ene gets the lowest locant (P-31.1.4, after the tied 1,2,3 suffixes); the name
    # must not depend on SMILES atom order.
    a = _pin().name("OC(=O)C/C(=C\\C(=O)O)C(=O)O")
    b = _pin().name("OC(=O)/C=C(\\CC(=O)O)C(=O)O")
    assert a == b == "(1E)-prop-1-ene-1,2,3-tricarboxylic acid", (a, b)


def test_saturated_added_carbon_unchanged_by_unsaturated_branch():
    # the saturated core path must stay byte-identical (no regression)
    assert _pin().name("OC(=O)CC(O)(CC(=O)O)C(=O)O") == "2-hydroxypropane-1,2,3-tricarboxylic acid"
    assert _pin().name("OC(=O)CC(C(=O)O)CC(=O)O") == "propane-1,2,3-tricarboxylic acid"


def test_dinuclear_ethene_core_omits_ene_locant():
    """Fable BLOCKER: a 2-carbon (dinuclear) unsaturated core must be `ethene-...`,
    NOT `eth-1-ene-...`. For a dinuclear chain the double-bond locant is structurally
    redundant (P-14.3.3 deny-by-default; BB `ethene-1,1,2-triyl`, `eth-1-ene` 0x).
    The OPSIN re-anchor cannot catch it (both spellings parse to one InChIKey), so it
    is fixed at the emitter."""
    assert _pin().name("OC(=O)C(C(=O)O)=CC(=O)O") == "ethene-1,1,2-tricarboxylic acid"
