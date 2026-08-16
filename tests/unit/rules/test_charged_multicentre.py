"""Charged multi-centre subsystem (root R11; P-71..75) — W8-P5.

Task 0: guard-characterization test locking the ~44 already-built charged
golds so later work in this subsystem can never regress them (these already
pass at HEAD; this is a characterization lock, not a new-behaviour test).

Task 1 (P-73.2.3.1, BB 41623 PIN): acylium detection + neutralize-as-acid
emitter — ``classify_cation`` must return 'acylium' (not the generic
carbenium 'ylium') for a C+ double-bonded to O, and the name must be the
reconstructed-acid PIN ('acetylium'/'cyclohexanecarbonylium'), never the
generic hydride-loss aldehyde form ('acetaldehydylium').
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.ions import get_ion_sites
from orthonym.rules.ions import classify_cation


GUARD = {
    "C[N-][N+](C)(C)C": "1,2,2,2-tetramethylhydrazin-2-ium-1-ide",   # P-74.1.1
    "CC(C)=[O+][O-]":   "2-(propan-2-ylidene)dioxidan-2-ium-1-ide",  # P-74.1.1
    "C[P+](C)(C)[C-](C)C": "2-(trimethylphosphaniumyl)propan-2-ide", # P-74.2.1.1
    "[C-]#[C-]": "ethynediide",                                       # P-72.2.2.1
    "[O-]CC[O-]": "ethane-1,2-bis(olate)",                            # P-72.2.2.2.2
    "[NH-]CC[NH-]": "ethane-1,2-bis(aminide)",                        # P-72.2.2.2.3
    "[NH3+]CC[NH3+]": "ethane-1,2-bis(aminium)",                      # P-73.5 poly-aminium
    "C[N+](C)(C)C": "N,N,N-trimethylmethanaminium",                   # P-73.1.2.1
    "[CH-]1CCCCC1": "cyclohexan-1-ide",                               # P-72.2.2.1 ring
    "C[B-](C)(C)C": "tetramethylboranuide",                           # P-72.3
    "C[P-](C)(C)C": "tetramethylphosphanuide",                        # P-72.3
    "NC(=[OH+])N": "uronium",                                         # P-73.1.2.2
    "CCC=[S+][O-]": "propylidene-λ4-sulfanone",                       # P-74.2.2.1.8
    "CC(C)[O-]": "propan-2-olate",                                    # P-72.2.2.2.2
}


@pytest.mark.parametrize("smi,expected", GUARD.items())
def test_charged_guard_no_regression(smi, expected):
    assert Orthonym().name(smi) == expected


# === Task 1: acylium (P-73.2.3.1) =============================================

def test_classify_acylium():
    mol = Chem.MolFromSmiles("C[C+]=O")  # CH3-C(+)=O
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "acylium"


def test_classify_cyclohexanecarbonylium():
    mol = Chem.MolFromSmiles("[C+](=O)C1CCCCC1")
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "acylium"


def test_classify_plain_carbenium_unaffected():
    """A plain carbenium (no double-bonded O) must stay 'ylium' (no regression
    on the pre-existing carbenium path)."""
    mol = Chem.MolFromSmiles("C[C+](C)C")  # tert-butyl cation
    site = get_ion_sites(mol)["cations"][0]
    assert classify_cation(mol, site) == "ylium"


def test_acetylium_pin():
    assert Orthonym().name("C[C+]=O") == "acetylium"


def test_cyclohexanecarbonylium_pin():
    assert Orthonym().name("[C+](=O)C1CCCCC1") == "cyclohexanecarbonylium"


def test_acylium_gated_equals_raw():
    """Gated == raw (gate-off) confirms the emitter is source-level correct,
    not merely OPSIN-lucky (Global Constraint: verify with the gate-off
    namer since the RT-gate fails OPEN without Java)."""
    gated = Orthonym()
    raw = Orthonym(_disable_opsin_validity_gate=True)
    for smi in ("C[C+]=O", "[C+](=O)C1CCCCC1"):
        assert gated.name(smi) == raw.name(smi)


# === Task 2: Group-14/halogen uide extension (P-72.3 / P-72.8.1) ============
#
# NOTE on evidence SMILES: the plan doc's Task 2 fixture ``C[SiH3-]`` is a
# 4-coordinate Si anion (degree 1 + 3 H = 4 bonds), which is NEITHER the
# 'ide' pattern (v-1 = 3 bonds, P-72.2.2.1) NOR the 'uide' pattern (v+1 = 5
# bonds, P-72.3) for standard-valence-4 silicon -- it is a typo. Verified via
# OPSIN reverse-parse of the literal PIN string "methylsilanuide", which
# gives the canonical SMILES ``C[SiH4-]`` (5-coordinate: 1 C + 4 H, charge
# -1) -- BYTE-IDENTICAL to RDKit's own canonicalization of ``C[SiH4-]``. The
# corrected 5-coordinate SMILES is used here.
#
# Reproduce-first (2026-07-18, at HEAD 98070e30, before this task's changes)
# found the Group-14/halogen element/valence tables were ALREADY generalized
# by an earlier wave (commit 55377897, "feat(w4-i2): generalize -uide anion
# beyond boron") -- ``methylsilanuide`` already ships correctly, gated and
# raw. The two things NOT yet built: (1) OPSIN 2.9 rejects ANY SUBSTITUTED
# halogen-uide token ("unphysical valency state") even though it happily
# parses the unsubstituted 'iodanuide' -> [IH2-] AND the BB-cited explicit-
# lambda alternative 'diphenyl-lambda3-iodanide' -> the IDENTICAL structure
# (BB 41110 states this equivalence verbatim) -- a genuine OPSIN grammar
# limitation, not a naming defect, needing a correct-by-construction carve-
# out; (2) an atom-drop leak in ``_emit_group13_uide``: classify_substituent
# names an 'alkyl'-typed branch by CARBON COUNT ONLY, silently dropping any
# heteroatom in that branch (independently reproduced: ``[I-](CCO)c1ccccc1``
# -> 'ethylphenyliodanuide', dropping -OH; ``[B-](CCO)(C)(C)C`` ->
# 'ethyltrimethylboranuide', same drop) -- needs a source-level veto since
# the RT-gate fails OPEN for the halogen-uide carve-out family.

def test_methylsilanuide_pin():
    assert Orthonym().name("C[SiH4-]") == "methylsilanuide"


def test_methylsilanuide_gated_equals_raw():
    gated = Orthonym()
    raw = Orthonym(_disable_opsin_validity_gate=True)
    assert gated.name("C[SiH4-]") == raw.name("C[SiH4-]") == "methylsilanuide"


def test_trimethylsilanide_unaffected():
    """The sibling 'ide' (not 'uide') 3-coordinate Si anion — regression
    guard for the classify_anion valence-gate discriminator."""
    assert Orthonym().name("C[Si-](C)C") == "trimethylsilanide"


def test_diphenyliodanuide_pin():
    """BB 41110 PIN. OPSIN 2.9 rejects ANY substituted 'iodanuide'/
    'bromanuide'/'chloranuide' token outright ("unphysical valency state")
    even though it happily parses the unsubstituted 'iodanuide' -> [IH2-]
    AND the BB-cited explicit-lambda alternative 'diphenyl-lambda3-iodanide'
    -> the IDENTICAL 2-coordinate structure (BB 41110 states this
    equivalence verbatim) -- a genuine OPSIN substituted-halogen-uide grammar
    limitation, not a naming defect. Carved out via
    namer._HALOGEN_UIDE_PIN_RE (correct-by-construction, same precedent as
    the inositol/dianhydride/phane carve-outs)."""
    assert Orthonym().name("[I-](c1ccccc1)c1ccccc1") == "diphenyliodanuide"


def test_diphenyliodanuide_gated_equals_raw():
    gated = Orthonym()
    raw = Orthonym(_disable_opsin_validity_gate=True)
    smi = "[I-](c1ccccc1)c1ccccc1"
    assert gated.name(smi) == raw.name(smi) == "diphenyliodanuide"


def test_uide_atom_drop_veto_iodanuide():
    """Source-level atom-conservation veto (MANDATORY per Global Constraints:
    the RT-gate fails OPEN for the halogen-uide carve-out, since OPSIN cannot
    parse ANY substituted halogen-uide name to run the SELF-01 check).
    classify_substituent, invoked on a bare interior atom subset, silently
    degrades a hetero-substituted branch to a plain hydrocarbon name
    ('CCO' -> 'ethyl', dropping -OH). ``_emit_group13_uide`` must decline
    (never ship the drop) -- verified with the gate-off raw namer per the
    Global Constraints."""
    raw = Orthonym(_disable_opsin_validity_gate=True)
    assert raw.name("[I-](CCO)c1ccccc1") != "ethylphenyliodanuide"


def test_uide_atom_drop_veto_boranuide():
    """Same veto, defense-in-depth on the already-shipped boranuide family
    (this case was already fail-closed via the SELF-01 OPSIN gate before this
    task's change; the new source-level veto adds a Java-independent second
    guard)."""
    raw = Orthonym(_disable_opsin_validity_gate=True)
    assert raw.name("[B-](CCO)(C)(C)C") != "ethyltrimethylboranuide"


# === v28 Cluster A Fix 1: azido/diazo are NOT cumulative zwitterions =========
# P-61.7: the -N=[N+]=[N-] azide group is a NEUTRAL internal-charge (P-59
# Table 5.1) prefix group named 'azido' by substitutive nomenclature — NOT a
# P-74.1.1 same-parent '-ium…-ide' zwitterion. emit_cumulative_ium_ide must
# decline (return None) for any azide/diazo so dispatch cascades to the neutral
# azido-prefix / acyl-azide handlers, while genuine cumulative zwitterions
# (single-bond hydrazinium/triazenium, dioxidane) still name.

from orthonym.rules.ions import emit_cumulative_ium_ide  # noqa: E402


class TestAzidoNotCumulativeZwitterion:
    @pytest.mark.parametrize("smi", [
        "[N-]=[N+]=Nc1ccccc1",  # azidobenzene
        "CN=[N+]=[N-]",          # azidomethane
        "CCCC(=O)N=[N+]=[N-]",   # butanoyl azide (acyl azide)
        "C=[N+]=[N-]",           # diazomethane (diazo, also internal-charge)
    ])
    def test_azide_declines_emitter(self, smi):
        # The +/- pair belongs to a neutral internal-charge group -> decline.
        assert emit_cumulative_ium_ide(Chem.MolFromSmiles(smi)) is None

    @pytest.mark.parametrize("smi", [
        "C[N-][N+](C)(C)C",   # 1,2,2,2-tetramethylhydrazin-2-ium-1-ide (N-N single)
        "C[N+]([N-]C)=NC",    # 1,2,3-trimethyltriaz-2-en-2-ium-1-ide
        "C[N-][N+](=C)C",     # 1,2-dimethyl-2-methylidenehydrazin-2-ium-1-ide
        "CC(C)=[O+][O-]",     # 2-(propan-2-ylidene)dioxidan-2-ium-1-ide (O chain)
    ])
    def test_genuine_zwitterion_still_emitted(self, smi):
        # These are NOT internal-charge groups -> the emitter still fires.
        assert emit_cumulative_ium_ide(Chem.MolFromSmiles(smi)) is not None

    @pytest.mark.parametrize("smi,expected", [
        ("[N-]=[N+]=Nc1ccccc1", "azidobenzene"),   # P-61.7 (PIN)
        ("CN=[N+]=[N-]", "azidomethane"),           # P-66.4.1
        ("CCCC(=O)N=[N+]=[N-]", "butanoyl azide"),  # P-65.5.2.1 acyl azide
    ])
    def test_azide_full_name(self, smi, expected):
        assert Orthonym().name(smi) == expected


class TestAlkylammoniumAminiumPIN:
    """v28 Cluster C (P-73.1.2.1): protonated/alkylated amine cations take the
    substitutive '-aminium' PIN, NOT the general 'alkylammonium' retained name.
    BB: 'methanaminium chloride (PIN)' (26672), 'N,N,N-trimethylmethanaminium
    (PIN)' over 'tetramethylammonium' (41354). NH4+ stays the retained PIN
    'ammonium'."""

    @pytest.mark.parametrize("smi,expected", [
        ("C[NH3+]", "methanaminium"),
        ("CC[NH3+]", "ethanaminium"),
        ("CCC[NH3+]", "propan-1-aminium"),
        ("C[NH2+]C", "N-methylmethanaminium"),
        ("C[NH+](C)C", "N,N-dimethylmethanaminium"),
        ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),
        ("[NH4+]", "ammonium"),  # genuine retained PIN, not denied
    ])
    def test_aminium_pin(self, smi, expected):
        assert Orthonym().name(smi) == expected
