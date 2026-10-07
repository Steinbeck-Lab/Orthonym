"""
Batch regression tests for medium molecule (21-40 HA) parent selection fixes.

a phase Plans 01-02: Tests grouped by compound class to verify parent selection
improvements for medium-sized molecules. Each test verifies that the generated
name contains expected structural features (substring matching for robustness).

Test groups (Plan 01):
  1. Steroid parent selection (NP scaffold + methyl/halogen decoration)
  2. Fused heterocycle with chain substituents
  3. Polycyclic VB naming
  4. Polycyclic aromatic routing
  5. Charged species / salt naming
  6. Alkaloid parent selection

Test groups (Plan 02):
  7. VB polycyclic format verification (VB names are complete)
  8. Macrocyclic compound naming
  9. small-molecule stereo compounds (baseline documentation)
  10. Steroid decoration completeness
"""

import pytest
from orthonym import name_compound
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
    "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
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


# Suite fix j6-breadth (TRIAGE g4 C4/C5): rows whose PIN the PIN tier cannot
# build. Each keeps a strict xfail naming the missing producer; what ships
# (best-effort RT-exact, PIN tier sentinel or RT-exact) is asserted in
# tests/unit/rules/test_j6_breadth.py::test_pin_not_built_rows_keep_the_tier_contract.
_J6_TODO = "TODO in TRIAGE.md 'Suite fix -- j6-breadth'"
_XF_MACROLIDE = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs macrolide parent selection -- the 16-membered "
    "lactone ring carries the principal characteristic groups (P-44.1.1, "
    "BlueBookV2.md:18875) and the thiazole is a substituent; best-effort names "
    "the thiazole as parent (non-PIN) -- " + _J6_TODO))
_XF_FUSED = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs the fusion PIN of this ortho-fused ring system "
    "(P-52.2.4.1, BlueBookV2.md:23710: fusion names are PINs with two rings of "
    "five or more members; von Baeyer names are not); a von Baeyer token in the "
    "expected value is itself non-PIN and must be corrected with the build -- "
    + _J6_TODO))
_XF_STEROID = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs the hydro-cyclopenta[a]phenanthrene PIN for a "
    "steroid whose input leaves ring stereocentres undefined -- a stereoparent "
    "name implies the configuration of all of them (P-101.2.6, BlueBookV2.md:"
    "51047) -- " + _J6_TODO))
_XF_POLYCYCLE = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs a PIN-tier producer for this spiro/bridged "
    "polycycle with ylidene and lactone substituents; best-effort names it "
    "RT-exact with non-PIN parts ('methan-1-ylidene', '2-oxaethan-1-yl') -- "
    + _J6_TODO))
_XF_INDOLONE = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs the indol-2-one / piperazine-2,5-dione parent "
    "choice with the decorated 2,3-dihydro-1H-indol-2-one substituent; "
    "'indoline' is not a PIN stem (P-54.4.3.2, BlueBookV2.md:24256), so the "
    "expected token is 'indol' -- " + _J6_TODO))
_XF_DIHYDRO_OXO = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs the '4-methyl-2,5-dioxo-2,5-dihydrofuran-3-yl' "
    "prefix (oxo and hydro prefixes on a mancude heteromonocycle, P-31.1.4.2.4) "
    "at the PIN tier; best-effort ships the non-PIN '1-oxacyclopent-3-en' form. "
    "PIN '(2E)-3-(methoxycarbonyl)-2-[16-(4-methyl-2,5-dioxo-2,5-dihydrofuran-3-"
    "yl)hexadecyl]pent-2-enedioic acid' (OPSIN exact) -- " + _J6_TODO))


# ---------------------------------------------------------------------------
# Group 1: Steroid parent selection
# Verifies that steroid NP scaffolds are detected and decorated with
# explicit methyl groups (IUPAC retained names with substitution)
# ---------------------------------------------------------------------------
STEROID_PARENT_FIXES = [
    pytest.param(
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)"
        "[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C",
        "trimethylergost",
        id="ergost-dien-ol-trimethyl",
    ),
    pytest.param(
        "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)"
        "C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O",
        "trimethylcholest",
        id="cholest-en-trione-trimethyl",
        marks=pytest.mark.xfail(strict=True, reason=(
            "j7 (TRIAGE g7 C12): C-10/13/17/20 are undefined in the input and the "
            "cholestane stereoparent implies them (P-101.2.6, BlueBookV2.md:51047), so "
            "the steroid name is declined; a stereoparent name would need 'xi' for "
            "those centres, which OPSIN 2.9.0 cannot verify. Best-effort names the "
            "molecule RT-exact (test_stereo_mismatch_fixes::TestSM40).")),
    ),
    pytest.param(
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)"
        "C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)"
        "[C@@]1(C)CC3",
        "trimethyl",
        id="cholest-dien-yl-acetate-trimethyl",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", STEROID_PARENT_FIXES)
def test_steroid_parent_selection(smiles, expected_substr):
    """Steroid NP scaffolds must include explicit methyl group decoration."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 2: Fused heterocycle with chain substituents
# Verifies that ring systems are preferred as parent over chains, and
# that chain substituents/stereo are enumerated correctly
# ---------------------------------------------------------------------------
FUSED_HETERO_CHAIN_FIXES = [
    pytest.param(
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
        "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "pentamethyl",  # P86: now also finds 6-oxo, so substring can't span methyl→oxacyclo
        id="macrolide-pentamethyl-oxacyclohexadecanone",
marks=_XF_MACROLIDE,
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)"
        "O[C@@H](C)C/C=C\\C(=O)O1",
        "trimethyl",  # P86: now also finds 6,12-dioxo, so substring can't span methyl→oxacyclo
        id="macrolide-trimethyl-oxacyclohexadecanone",
    ),
    pytest.param(
        "C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
        "trienoic acid",
        id="prenylated-phenol-trienoic-acid",
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/COC[C@H]1O[C@@H]"
        "(N2CCC(=O)NC2=O)[C@H](O)[C@@H]1O",
        # the ring that carries the principal characteristic group is the parent
        #, the Blue Book); the cyclic amide is a pseudoketone
        #,:29314, "Cyclic anhydrides, esters and amides are named as
        # pseudoketones"; '1,3-diazinane-2,4,6-trione (PIN)':29344), so the
        # oxolane is a substituent
        "oxolan-2-yl]-1,3-diazinane-2,4-dione",
        id="geranyl-nucleoside-thf-parent",
    ),
    pytest.param(
        "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)"
        "C(=O)OC1=O)C(=O)O",
        "2,5-dioxo-2,5-dihydrofuran-3-yl",
        id="long-chain-dicarboxylic-acid",
marks=_XF_DIHYDRO_OXO,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", FUSED_HETERO_CHAIN_FIXES)
def test_fused_heterocycle_chain_parent(smiles, expected_substr):
    """Fused heterocycles with chains must select ring as parent."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 3: Polycyclic VB naming
# Verifies that polycyclic systems get appropriate VB or retained names
# instead of being reduced to small fragment names
# ---------------------------------------------------------------------------
POLYCYCLIC_VB_FIXES = [
    pytest.param(
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        # 'chromane' is not a PIN: (the Blue Book), "Names
        # listed in Table 3.1 are retained names that are not used as preferred
        # IUPAC names"; 'chromane 3,4-dihydro-2H-1-benzopyran (PIN)' (:17004)
        "3,4-dihydro-2H-1-benzopyran-3,4,7-triol",
        id="catechin-trihydroxychromane",
    ),
    pytest.param(
        "O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)"
        "[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        # (:16980,:17004), as above
        "2,3-dihydro-4H-1-benzopyran-4-one",
        id="xylopyranoside-trihydroxychromanone",
    ),
    pytest.param(
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)"
        "c3ccccc32)NC1=O",
        "indol",
        id="indolinone-derivative",
marks=_XF_INDOLONE,
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2"
        "CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O",
        "tricyclo",
        id="prenylated-tetracyclic-vb",
        marks=pytest.mark.xfail(
            reason="v22 G0/DD7 S1: this is a benzo-fused ring system; the prior "
            "von-Baeyer 'tricyclo[...]' name DROPPED the aromaticity (a wrong, "
            "de-aromatised cage). G0 fails it closed; the correct bridged-fused "
            "PIN is a Phase-G1 build.",
            strict=False,
        ),
    ),
    pytest.param(
        "O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4"
        "CCCCN4[C@@H]32)C(=O)N[C@H]1Cc1ccccc1",
        "tetracyclo",
        id="peptide-tetracyclic-vb",
        marks=pytest.mark.xfail(
            reason="v22 G0/DD7 S1: benzo-fused ring system; the prior von-Baeyer "
            "'tetracyclo[...]' name dropped the aromaticity. G0 fails it closed "
            "(or routes to a non-VB name); correct bridged-fused PIN is Phase-G1.",
            strict=False,
        ),
    ),
    pytest.param(
        "C=C1CC[C@@H](C/C=C2/CC[C@]3(OC2)O[C@@]2(O)"
        "CC[C@]3(C)OC2(C)C)C(C)(C)[C@H]1[C@@H](O)"
        "C=C1CCOC1=O",
        "trioxa",
        id="trioxa-tricyclic-terpene",
marks=_XF_POLYCYCLE,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", POLYCYCLIC_VB_FIXES)
def test_polycyclic_vb_naming(smiles, expected_substr):
    """Polycyclic compounds must get VB or retained ring names, not fragments."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 4: Polycyclic aromatic routing
# Verifies that polycyclic aromatic systems are correctly identified
# ---------------------------------------------------------------------------
POLYCYCLIC_AROMATIC_FIXES = [
    pytest.param(
        "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1",
        "piperazine",
        id="indole-imidazole-piperazinedione",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", POLYCYCLIC_AROMATIC_FIXES)
def test_polycyclic_aromatic_routing(smiles, expected_substr):
    """Polycyclic aromatic compounds must route to correct naming path."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 5: Charged species / salt naming
# Verifies that charged compounds produce meaningful structural names
# instead of generic "ammonium <acid>" fragments
# ---------------------------------------------------------------------------
CHARGED_SPECIES_FIXES = [
    pytest.param(
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])"
        "OCC[N+](C)(C)C)OC(C)=O",
        # 'palmitic acid' is retained for general nomenclature only:
        # (the Blue Book, "The following names are retained for general
        # nomenclature"), 'palmitic acid hexadecanoic acid (PIN)' (:29787)
        "hexadecanoyloxy",
        id="phospholipid-palmitate-ester",
    ),
    pytest.param(
        "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]",
        "hydrochloride",
        id="methoxyphenyl-amine-hydrochloride",
    ),
    pytest.param(
        "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])"
        "C[N+](C)(C)C",
        # The inner salt is named as the '-ate' zwitterion and the acyl is
        # 3-hydroxyhexadecanoyl, so the old substring was wrong for this SMILES.
        # Expected spelling: substituent prefixes in alphanumerical order,
        # (the Blue Book); OPSIN 2.9.0 full-InChIKey EXACT (TRIAGE g4
        # C7). The shipped name is RT-exact and labelled below pin_verified
        # (test_carnitine_ester_zwitterion_tier_contract).
        "3-[(3-hydroxyhexadecanoyl)oxy]-4-(trimethylazaniumyl)butanoate",
        id="carnitine-palmitoyl-acid",
        marks=pytest.mark.xfail(strict=True, reason=(
            "the inner-salt name is general nomenclature (its P-74.1.3 PIN cites '(N,N-dimethylmethanaminiumyl)', which OPSIN 2.9.0 cannot verify; decision A part 2) and _route_zwitterion glues the cation prefix in front of the anion parent's own prefixes: '4-(trimethylazaniumyl)3-[(3-hydroxyhexadecanoyl)oxy]butanoate' (no hyphen, P-14.5.2 order broken; charged_router needs a structured composition) -- TODO in TRIAGE.md 'Suite fix -- j5-pin-labels-b'")),
    ),
    pytest.param(
        "CC(C)[C@@]1(C)N=C(c2nc3ccccc3cc2C(=O)[O-])"
        "NC1=O.[NH4+]",
        "quinoline",
        id="ammonium-quinoline-carboxylate",
        marks=pytest.mark.xfail(strict=True, reason=(
            "PIN tier abstains (as it did in production at 4e0e5c29b); the old "
            "'ammonium quinoline-3-carboxylate' dropped the imidazolone ring "
            "(OPSIN: a different molecule; 704facd16). Needs the "
            "4,5-dihydro-1H-imidazol-2-yl substituent at the PIN tier -- TODO in "
            "TRIAGE.md 'Suite fix -- j1-regressions'")),
    ),
]

# Imazaquin ammonium: what ships in production (gate on) is asserted below --
# the PIN tier fails closed or ships an RT-exact name, and best-effort names it
# RT-exact with the quinoline carboxylate as the parent.
IMAZAQUIN_NH4 = "CC(C)[C@@]1(C)N=C(c2nc3ccccc3cc2C(=O)[O-])NC1=O.[NH4+]"


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", CHARGED_SPECIES_FIXES)
def test_charged_species_naming(smiles, expected_substr):
    """Charged species must produce structural names, not generic salts."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


@pytest.mark.integration
@pytest.mark.opsin_gate
def test_carnitine_ester_zwitterion_tier_contract():
    """The strict-xfail carnitine row above, in production (gate on): a name ships,
    RT-exact, and below pin_verified PIN not verifiable; the composed
    spelling is not the order yet -- TRIAGE g4 C7)."""
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    smi = "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C"
    r = _dt_row(smi)
    assert r["tier"] != "pin_verified", r
    assert name_is_rt_exact(r["name"], smi), r


@pytest.mark.integration
@pytest.mark.opsin_gate
def test_imazaquin_ammonium_tier_contract():
    """The strict-xfail row above, in production: the tier contract holds and the
    best-effort name keeps the quinoline-3-carboxylate parent."""
    from tests.support.rt_assert import assert_tier_contract
    _pin, be = assert_tier_contract(IMAZAQUIN_NH4)
    assert "quinoline-3-carboxylate" in be and be.startswith("ammonium "), be


# ---------------------------------------------------------------------------
# Group 6: Alkaloid parent selection
# Verifies that alkaloid scaffolds are recognized with correct naming
# ---------------------------------------------------------------------------
ALKALOID_FIXES = [
    pytest.param(
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",
        "tropan",  # a phase-02: tropane NP naming correctly identifies tropane scaffold
        id="tropane-indole-ester",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", ALKALOID_FIXES)
def test_alkaloid_parent_selection(smiles, expected_substr):
    """Alkaloid compounds must identify correct scaffold parent."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ===========================================================================
# Plan 02 Groups (a phase-02)
# ===========================================================================

# ---------------------------------------------------------------------------
# Group 7: VB polycyclic format verification
# Verifies that VB-named polycyclic compounds produce complete structural
# names with VB descriptors (these are correct names but OPSIN interprets
# differently -- classified as unfixable_vb_interpretation)
# ---------------------------------------------------------------------------
VB_FORMAT_VERIFICATION = [
    pytest.param(
        "O=C(O)c1cc2cc3c4c(c2oc1=O)CCCN4CCC3",
        "tetracyclo",
        id="vb-aza-tetracyclic-acid",
marks=_XF_FUSED,
    ),
    pytest.param(
        "COc1cccc2c1C(=O)c1ccc3c(c1C2=O)C(=O)C[C@@H](C)[C@H]3O",
        "tetracyclo",
        id="vb-methoxy-tetracyclic-trione",
marks=_XF_FUSED,
    ),
    pytest.param(
        "COc1cc(O)c2c(c1O)C(=O)c1c(C(C)=O)c(O)cc(O)c1C2=O",
        "tricyclo",
        id="vb-polyhydroxy-tricyclic-dione",
marks=_XF_FUSED,
    ),
    pytest.param(
        "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)"
        "C15CC2)[C@@H]3[C@@H]4O",
        "-2,12b:5a,7a-diethanoazepino[4,3-c]carbazole-",
        id="vb-hexacyclic-diaza-alkaloid",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", VB_FORMAT_VERIFICATION)
def test_vb_format_completeness(smiles, expected_substr):
    """VB-named compounds must produce names containing VB ring descriptors."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 8: Macrocyclic compound naming
# Verifies that macrocyclic lactones/lactams produce correct ring-size
# prefix and substituent enumeration
# ---------------------------------------------------------------------------
MACROCYCLIC_FIXES = [
    pytest.param(
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
        "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "oxacyclohexadecan",
        id="macrolide-16-ring-oxa",
marks=_XF_MACROLIDE,
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)"
        "O[C@@H](C)C/C=C\\C(=O)O1",
        # the ring has two C=C, so its name is not the saturated
        # '...oxacyclohexadecane': (the Blue Book), "In rings
        # modified by skeletal replacement ('a') nomenclature, low locants are
        # assigned first to heteroatoms and then to unsaturated sites"
        # ('1,4,7,10-tetraoxacyclododec-2-ene (PIN)':16562; the 'a' before
        # 'diene' as in 'cycloocta-1,3,5,7-tetraene (PIN)':16554)
        "1,5,11-trioxacyclohexadeca-7,13-diene-2,6,12-trione",
        id="macrolide-trilactone-16ring",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", MACROCYCLIC_FIXES)
def test_macrocyclic_naming(smiles, expected_substr):
    """Macrocyclic compounds must include correct ring-size prefix."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 9: small-molecule wrong-parent stereo baseline
# Documents current naming status of 14 small stereo compounds from a phase.
# All are blocked by wrong parent selection, not stereo labeling errors.
# Tests verify the name is non-None (structural description produced).
# ---------------------------------------------------------------------------
SMALL_STEREO_PARENT_BASELINE = [
    pytest.param(
        "C=C(C)[C@@H]1CC[C@@H](C)[C@@]12CC=C(C)CC2",
        "spiro",
        id="ster04-spiro-isopropenyl-cyclohexene",
    ),
    pytest.param(
        "CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21",
        "naphthalen",
        id="ster04-decalin-ketone",
marks=_XF_FUSED,
    ),
    pytest.param(
        "CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O",
        "spiro",
        id="ster04-spirocyclopentanone",
    ),
    pytest.param(
        "CC(C)=CCc1ccc(O)c2c1[C@H](CC(=O)O)OC2=O",
        "acid",
        id="ster04-isocoumarinone-acid",
    ),
    pytest.param(
        "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",
        # a phase-01: decomposition now produces acyl prefix form
        "heptanoyl",
        id="ster04-pyrrolizinone-amide",
    ),
    # Breadth job 1: with the suite's gate off this row passed on a wrong molecule,
    # 'hydroxy-oxocyclopentane-1-carboxamide' (name_amide spelled the pyrrolone as a
    # cyclopentane; it now declines a ring it cannot spell). With the production gate
    # the PIN tier abstains, before and after; best-effort names it RT-exact
    # ('4-hydroxy-2-oxo-3-[(2E,4E)-1-oxohexa-2,4-dien-1-yl]-5-(propan-2-ylidene)-2,5-
    # dihydro-1H-pyrrole').
    pytest.param(
        "C/C=C/C=C/C(=O)C1=C(O)C(=C(C)C)NC1=O",
        "oxo",
        id="ster04-dienoyl-pyrrole",
        marks=[pytest.mark.opsin_gate, pytest.mark.xfail(strict=True, reason=(
            "PIN tier abstains: needs a PIN-tier producer for 3-acyl tetramic acids "
            "(1,5-dihydro-2H-pyrrol-2-one with an exocyclic ylidene); best-effort "
            "names it RT-exact"))],
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", SMALL_STEREO_PARENT_BASELINE)
def test_small_stereo_parent_baseline(smiles, expected_substr):
    """ small stereo compounds must produce non-None names with structural content."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"


# ---------------------------------------------------------------------------
# Group 10: Steroid decoration completeness
# Verifies that steroid NP names include hydroxy, ketone, and unsaturation
# decorations (these steroids already have reasonable names but may be
# missing methyls or other substituents)
# ---------------------------------------------------------------------------
STEROID_DECORATION_COMPLETENESS = [
    pytest.param(
        "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
        "cyclopenta[a]phenanthren",
        id="stigmastane-diol-retained-name",
marks=_XF_STEROID,
    ),
    pytest.param(
        "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
        "cyclopenta[a]phenanthren",
        id="ergostane-dienol-retained-name",
marks=_XF_STEROID,
    ),
    pytest.param(
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1"
        "[C@@H](O)[C@@H](O)[C@@H]2O",
        "estra",
        id="estrane-tetraol-decoration",
    ),
    pytest.param(
        "CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3"
        "[C@H](O)C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C",
        "cyclopenta[a]phenanthren",
        id="cholestane-tetraol-sulfonate",
marks=_XF_STEROID,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substr", STEROID_DECORATION_COMPLETENESS)
def test_steroid_decoration_completeness(smiles, expected_substr):
    """Steroid NP names must include correct scaffold stem."""
    name = _dt_name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected_substr in name, f"Expected '{expected_substr}' in name: {name}"
