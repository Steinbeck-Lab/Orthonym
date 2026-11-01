"""
Parent Mismatch Benchmark: Measures improvement in parent structure selection.

Originally created in a phase, targeting parent_mismatch RT failures where OPSIN
parses the generated name but InChI doesn't match because the wrong parent structure
was selected (wrong ring system, wrong ring-vs-chain decision, or wrong principal
group location).

a phase audit (55 compounds from v13 benchmark) found:
  - 28 hit cascade (correct parent selection, name wrong for other reasons)
  - 9 no ring systems (acyclic -- parent selection not applicable)
  - 7 "unclear" fallback (PG separated from ring/chain by 2+ hops -- a phase+)
  - 5 (a) chain only
  - 3 (a) ring only
  - 2 hydrocarbon rules
  - 1 (b) PG count comparison

a phase changes:
  - Fixed is_principal_group_on_ring self-check (symmetry with chain counterpart)
  - Fixed _count_pg_on_ring self-check (symmetry with chain counterpart)
  - Moved 5 xpassed compounds from EXPECTED_UNFIXED to EXPECTED_FIXED
  - 7 "unclear" fallback compounds deferred (PG detection range limitation)

Success criterion: At least 35 of 43 benchmark compounds produce correct parent names.
"""

import pytest

from orthonym import name_compound
from tests.support.np_producer import np_producer_only  # noqa: F401

# NPL 2026-11-01: the subject is the natural-product producer's own name; the route now offers the systematic
# name first INTRODUCTION, the Blue Book), so the producer is named with that offer off.
pytestmark = pytest.mark.usefixtures("np_producer_only")


# ---------------------------------------------------------------------------
# Parent mismatch compounds: (SMILES, expected_parent_substring, description)
# ---------------------------------------------------------------------------

# Compounds where a phase is expected to produce correct parent selection.
# These use firm assertions (no xfail).
EXPECTED_FIXED = [
    # --- STEROID / TERPENOID ---
    # a phase: ring classification + retained NP names.
    # This molecule carries the 16,22-epoxy O ring, so its parent is FUROSTANE, not
    # cholestane: (b) furostan (gold row P14-NP-FUROSTAN, "was
    # non-retained epoxy-name"); the old 'cholest' substring named a different
    # skeleton. OPSIN 2.9.0 full-InChIKey: 'furostan' EXACT. (the Blue Book:
    # 50943) identifies no PIN for Chapter; the stereoparent name is kept at the
    # PIN tier by the controller ruling (TRIAGE.md 'Controller rulings'; TRIAGE g5
    # C14).
    (
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3(C)[C@H]2[C@@H]1C",
        "furostan",
        "Steroid: cholestane retained NP name",
    ),
    # The next two inputs define no stereo at all. 'Stereochemical configuration
    # of parent structures' (the Blue Book): "The name of a fundamental parent
    # structure usually implies the absolute configuration of all chirality centers [...]
    # All chirality must be defined", so a stereo-free input cannot take the stereoparent
    # name ('stigmast'/'ergost'). Commit fa7a9a109 (2026-09-27, 'an undefined implied centre
    # declines the stereoparent') made the producer decline it on purpose; the name that
    # ships is the fully systematic one on the cyclopenta[a]phenanthrene fusion parent
    #; OPSIN 2.9.0 full InChIKey exact). The stereoparent path is still
    # asserted by the stereo-defined steroid rows of this list.
    (
        "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
        "cyclopenta[a]phenanthren",
        "Steroid: stigmastane diol, stereo-free input -> systematic fusion parent",
    ),
    (
        "C[C@]12CC[C@@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12",
        "androstan",
        "Steroid: androstane-3,17-diol (canary compound)",
    ),
    (
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CC(=O)CC[C@]34C)[C@@H]1CC(=O)[C@@H]2O",
        "androstan",
        "Steroid: androstanedione-ol",
    ),
    (
        "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
        "cyclopenta[a]phenanthren",
        "Steroid: ergostadienol, stereo-free input -> systematic fusion parent",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
        "chol",
        "Steroid: cholanoid acid",
    ),
    (
        "C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc43)[C@@H]1CC[C@@H]2O",
        "estra",
        "Steroid: estrane-3,17-diol",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1C[C@@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C",
        "chol",
        "Steroid: dihydroxycholadienone acid",
    ),

    # --- PAH / FUSED AROMATIC ---
    # a phase: identifies fused aromatic systems correctly
    # NOTE: 40-atom PAH moved to EXPECTED_UNFIXED -- a phase PAH size guard
    # correctly prevents naphthalene over-matching. Extended PAH naming needed.
    (
        "CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21",
        "naphthal",
        "PAH: tetrahydronaphthalene derivative",
    ),
    (
        "O=Cc1ccc2ccccc2c1O",
        "naphthal",
        "PAH: hydroxynaphthalenecarbaldehyde (canary)",
    ),
    (
        "C=C(C)C1C=C2C(C)=CCCC2(C)CC1",
        "decahydro",
        "PAH: decahydronaphthalene",
    ),

    # --- MONOCYCLIC ---
    # a phase: ring-over-chain + correct ring selection
    (
        "CC1CC=C(N2CCCC2)C1=O",
        "cyclopent",
        "Monocyclic: methylcyclopentenone",
    ),
    (
        "CCC1CC=C(N2CCCC2)C1=O",
        "cyclopent",
        "Monocyclic: ethylcyclopentenone",
    ),
    (
        "CC1=CC(=O)CC(C)(C)C1",
        "cyclohex",
        "Monocyclic: trimethylcyclohexenone (canary)",
    ),
    (
        "CC1=C(O)C(=O)C([C@@]2(C)CCCC2(C)C)=C(O)C1=O",
        "cyclohexa",
        "Monocyclic: cyclohexadienedione with cyclopentyl",
    ),

    # --- FUSED HETEROCYCLE ---
    # a phase: ring_system_score prefers heterocyclic over carbocyclic
    # quick-wins F-R8: (the Blue Book) keeps 'chromane' and 'isochromane'
    # out of PINs; the parent is the benzopyran, '3,4-dihydro-2H-1-benzopyran' and
    # '3,4-dihydro-1H-2-benzopyran',:11815). Was 'chroman' / 'isochroman'.
    (
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        "1-benzopyran",
        "FusedHet: trihydroxychromane",
    ),
    (
        "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O",
        "2-benzopyran",
        "FusedHet: dihydroxy-methoxy isochromanone",
    ),
    (
        "Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1",
        "imidazo",
        "FusedHet: imidazo[1,2-a]pyridine",
    ),
    (
        "CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
        "quinolin",
        "FusedHet: chloroquinoline ester",
    ),
    (
        "Oc1ccnc2ccccc12",
        "quinol",
        "FusedHet: hydroxyquinoline (canary)",
    ),
    (
        "CCCCCCCCCc1cc(=O)c2ccccc2n1C",
        "quinol",
        "FusedHet: N-methyl-nonylquinolinone (canary)",
    ),

    # --- BRIDGED POLYCYCLIC ---
    # a phase: ring_system_score selects correct polycyclic parent
    # quick-wins F-R8: the bridged fused name is the PIN, the Blue Book;
    #:19532, fused > bridged fused > von Baeyer), built by bridged fused S2:
    # '...decahydro-3a,7-methanocyclopenta[8]annulene-3,6-diol'. Was 'tricyclo'.
    (
        "CC1(C)C[C@H](O)[C@]23CC[C@@H](O)[C@](C)(CC[C@@H]12)C3",
        "methanocyclopenta[8]annulene",
        "Bridged: tricyclic diol (canary)",
    ),
    (
        "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C",
        "tricyclo",
        "Bridged: macrocyclic lactone (canary)",
    ),

    # --- ACYCLIC / CHAIN ---
    # a phase: cascade preserves correct chain selection
    (
        "CCCCCCCCCCC(C)C(=O)O",
        "dodecanoic",
        "Acyclic: 2-methyldodecanoic acid (canary)",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OCC",
        "docosanoate",
        "Acyclic: ethyl docosanoate (canary)",
    ),
    (
        "CC(C)CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
        "heptacosanoic",
        "Acyclic: 26-methylheptacosanoic acid (canary)",
    ),

    # --- SUGAR / GLYCOSIDE ---
    # a phase: ring selection preserves pyranose parent
    (
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        "pyran",
        "Sugar: xylopyranosyl galactose",
    ),

    # --- SIMPLE RETAINED NAMES ---
    (
        "Oc1ccccc1",
        "phenol",
        "Simple: phenol",
    ),
    (
        "c1ccncc1",
        "pyridin",
        "Simple: pyridine",
    ),
    (
        "Cl.c1ccc2sc(C3(N4CCCCC4)CCCCC3)cc2c1",
        "benzothioph",
        "FusedHet: benzothiophene salt",
    ),

    # --- a phase-02: DKP compound with indole (piperazine-2,5-dione retained name added) ---
    (
        "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1",
        "piperazin",
        "Diketopiperazine with indole - fixed by Phase 118-02 (piperazine-2,5-dione retained name)",
    ),

    # --- a phase-02: Tropane NP naming (numbering map added) ---
    (
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",
        "tropan",
        "Tropane ester HCl salt - fixed by Phase 118-02 (tropane numbering map)",
    ),
    # --- a phase-02: Ergostene derivative (exact SMILES lookup) ---
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@@H](O)[C@@]1(C)CC3)C(C)C",
        "ergost",
        "Steroid: ergostene - fixed by Phase 118-02 (NP derivative lookup)",
    ),

    # --- a phase audit: compounds confirmed fixed by earlier phases (moved from EXPECTED_UNFIXED) ---
    (
        "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12",
        "indol",
        "Peptide with indole - fixed by decomposition improvements",
    ),
    (
        "COc1ccc(C(=O)N2CCCC2=O)cc1",
        "methoxy",
        "Benzoylpyrrolidinone - fixed by decomposition improvements",
    ),
    (
        "CC(=O)[C@@H](C)Nc1ccccc1C(=O)O",
        "amino",
        "Aminobenzoic acid derivative - fixed by FG detection improvements",
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC[C@@H]2SC[C@@H]3NC(=O)"
        "N[C@@H]32)[C@@H](O)[C@H]1O",
        "adenosin",
        "Nucleotide conjugate - fixed by multi-fragment assembly improvements",
    ),
    (
        "CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",
        "cyclohex",
        "Cyclohexanone with chain FG - fixed by FG detection improvements",
    ),
    # --- Phospholipid: passed while it sat in EXPECTED_UNFIXED (an XPASS), so it is a fixed row ---
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)"
        "OC[C@H](N)C(=O)O",
        "phospho",
        "Phospholipid - multi-fragment naming",
    ),
]

# Rows whose old passing name described a DIFFERENT molecule; the default tier now
# fails closed (as production did with the gate on at 4e0e5c29b) and the
# best-effort tier names them RT-exact (test_parent_mismatch_tier_contract below).
# Strict: the substring may only come back through a real, complete name.
_BIOTINYL_AMP = (
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC[C@@H]2SC[C@@H]3NC(=O)"
    "N[C@@H]32)[C@@H](O)[C@H]1O")
_PIN_TIER_TARGETS = {
    # Suite fix j6-breadth (TRIAGE g5 C11): three more rows whose PIN the PIN
    # tier cannot build; best-effort names them RT-exact (test_j6_breadth tier
    # contract).
    "Cyclohexanone with chain FG - fixed by FG detection improvements": (
        "PIN tier abstains: needs the acid-parent PIN (the carboxylic acid is "
        "the suffix, P-44.1.1, BlueBookV2.md:18875) with a cycloalkylidene "
        "substituent and an amide carbon in the principal chain; best-effort's "
        "'...-2-oxocyclohexane' ring-parent name is not the PIN -- TODO in "
        "TRIAGE.md 'Suite fix -- j6-breadth'"),
    "FusedHet: benzothiophene salt": (
        "PIN tier abstains: the free base 1-[1-(1-benzothiophen-2-yl)cyclohexyl]"
        "piperidine is not named at the PIN tier (a 1-substituted cycloalkyl on "
        "a ring N; the raw generator counts its ring carbons as a chain), and "
        "the drawn neutral HCl adduct has no PIN (P-77.1.3; P-14.8.2 mixed "
        "adducts); best-effort '...piperidine—hydrogen chloride (1/1)' is "
        "RT-exact -- TODO in TRIAGE.md 'Suite fix -- j6-breadth'"),
    "PAH: decahydronaphthalene": (
        "PIN tier abstains: needs the hydro-naphthalene fusion PIN (P-52.2.4.1, "
        "BlueBookV2.md:23710; P-31.2.3.3.2); the molecule has two ring C=C, so "
        "it is a hexahydronaphthalene and the expected 'decahydro' substring is "
        "itself wrong -- TODO in TRIAGE.md 'Suite fix -- j6-breadth'"),
    "Nucleotide conjugate - fixed by multi-fragment assembly improvements": (
        "the old 'adenosine (1R,5S,6S)-6-(7-hydroxy-5,7-dioxo-6,8-dioxa-7-"
        "phosphaoctyl)-...' spliced two loose fragments (OPSIN: a 2-component "
        "mixture; 16f45443a stopped it). 'adenosine' is a P-105.1 retained name, "
        "not a PIN (P-100, BlueBookV2.md:50942); the systematic PIN is not built "
        "-- TODO in TRIAGE.md 'Suite fix -- j1-regressions'"),
}


# ---------------------------------------------------------------------------
# Tests for compounds expected to be fixed by a phase
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "smiles,expected_parent_substring,description",
    EXPECTED_FIXED,
    ids=[
        d.split(":")[0].strip().replace(" ", "_")
        + f"_{i:02d}"
        for i, (_, _, d) in enumerate(EXPECTED_FIXED)
    ],
)
def test_parent_mismatch_fixed(smiles, expected_parent_substring, description, request):
    """Should produce correct parent structure for these compounds.

    Verifies that the generated name contains the expected parent structure
    substring, indicating the correct ring system or chain was selected.
    """
    if description in _PIN_TIER_TARGETS:
        request.applymarker(pytest.mark.xfail(
            strict=True, reason=_PIN_TIER_TARGETS[description]))
    name = name_compound(smiles)
    assert name, f"Should produce a name for: {description}"
    assert expected_parent_substring.lower() in name.lower(), (
        f"PARENT MISMATCH: {description}\n"
        f"  SMILES: {smiles}\n"
        f"  Generated: {name}\n"
        f"  Expected parent substring: '{expected_parent_substring}'"
    )


@pytest.mark.integration
@pytest.mark.opsin_gate
def test_parent_mismatch_tier_contract():
    """Production contract for the _PIN_TIER_TARGETS rows: the default tier ships
    the failure sentinel or an RT-exact name, and best-effort names the molecule
    RT-exact (breadth never drops)."""
    from tests.support.rt_assert import assert_tier_contract
    assert_tier_contract(_BIOTINYL_AMP)


# ---------------------------------------------------------------------------
# Summary test: count total improvements
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# a phase Integration Tests: Chain Selection Fixes
# ---------------------------------------------------------------------------

class TestPhase88FGInstanceCounting:
    """: FG instance counting uses distinct instances, not atom overlap."""

    @pytest.mark.integration
    def test_diacid_both_groups_on_chain(self):
        """Glutaric acid: both COOHs on the 5-carbon chain (2 instances)."""
        name = name_compound("OC(=O)CCCC(=O)O")
        assert name, "Should produce a name for glutaric acid"
        assert "pentanedioic" in name.lower() or "glutar" in name.lower(), (
            f"Glutaric acid: expected pentanedioic acid. Got: {name}"
        )


class TestPhase88RingScoring:
    """: Ring scoring tuple reorder before."""

    @pytest.mark.integration
    def test_pyridine_is_principal_ring(self):
        """Pyridine retained name should be produced."""
        name = name_compound("c1ccncc1")
        assert name, "Should name pyridine"
        assert "pyridin" in name.lower(), f"Expected pyridine. Got: {name}"

    @pytest.mark.integration
    def test_phenol_retained(self):
        """Phenol retained name."""
        name = name_compound("Oc1ccccc1")
        assert name, "Should name phenol"
        assert "phenol" in name.lower(), f"Expected phenol. Got: {name}"


class TestPhase88HydrocarbonGuard:
    """: Parent selection runs for hydrocarbons."""

    @pytest.mark.integration
    def test_butylcyclohexane_ring_parent(self):
        """Butylcyclohexane: ring should be parent."""
        name = name_compound("CCCCC1CCCCC1")
        assert name, "Should produce a name"
        assert "cyclohex" in name.lower(), (
            f"Butylcyclohexane: ring should be parent. Got: {name}"
        )

    @pytest.mark.integration
    def test_propylcyclopentane_ring_parent(self):
        """Propylcyclopentane: ring should be parent."""
        name = name_compound("CCCC1CCCC1")
        assert name, "Should produce a name"
        assert "cyclopent" in name.lower(), (
            f"Propylcyclopentane: ring should be parent. Got: {name}"
        )


class TestPhase88CascadeCriteria:
    """: Full cascade criteria 5-9."""

    @pytest.mark.integration
    def test_cyclohexanone_ring_parent(self):
        """Cyclohexanone: ring with ketone is parent."""
        name = name_compound("O=C1CCCCC1")
        assert name, "Should name cyclohexanone"
        assert "cyclohex" in name.lower(), f"Expected cyclohexanone. Got: {name}"

    @pytest.mark.integration
    def test_benzoic_acid_ring_parent(self):
        """Benzoic acid: ring is parent with -COOH suffix."""
        name = name_compound("OC(=O)c1ccccc1")
        assert name, "Should name benzoic acid"
        assert "benz" in name.lower(), f"Expected benzoic acid. Got: {name}"
