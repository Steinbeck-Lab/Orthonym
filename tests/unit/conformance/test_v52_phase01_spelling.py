"""v52 a phase — Blue Book PIN spelling/numbering conformance (TDD red step).

Pins 13 representative rows drawn from the v52 audit of `the Blue Book Blue Book`
(benchmarks/bb_conformance/). Each assertion is the verbatim `(PIN)` string the
Blue Book gives for that row; later tasks in -spelling make these pass.

Grouped by sub-pattern (SP1-SP4), one class per pattern, each with a docstring
citing the governing rule id + section heading.
"""

from orthonym import name_compound
from orthonym.namer import Orthonym


def _name_or_tiered(smiles: str) -> str:
    """Name a SMILES, falling back to the tiered pipeline if name_compound abstains.

    SP2/SP4 rows are RT-gated (best-effort/tiered); if the public API returns
    None, fall back to Orthonym.name_tiered(smiles)["name"] and assert on
    whatever name string the pipeline actually emits for that row.
    """
    name = name_compound(smiles)
    if name is None:
        return Orthonym().name_tiered(smiles)["name"]
    return name


class TestSP1LocantOneOmission:
    """SP1 - "The locant '1' is omitted:" (c) "in monosubstituted
    homogeneous monocyclic rings" (the Blue Book Blue Book,:2913).
    """

    def test_sp1_cyclohexanecarboximidic_acid(self):
        assert name_compound("N=C(O)C1CCCCC1") == "cyclohexanecarboximidic acid"

    def test_sp1_cyclohexanecarbohydrazonic_acid(self):
        assert name_compound("NN=C(O)C1CCCCC1") == "cyclohexanecarbohydrazonic acid"

    def test_sp1_benzenesulfinyl_chloride(self):
        assert name_compound("O=S(Cl)c1ccccc1") == "benzenesulfinyl chloride"


class TestSP2VonBaeyerMinimumCompoundLocants:
    """SP2 - "If there is a choice of names and numbering..." (1)
    "a minimum number of compound locants" (the Blue Book Blue Book,
    :16635); (3) "compound locants are kept to a minimum" (:16689)
    for the ene+yne row.
    """

    def test_sp2_bicyclo420oct6ene(self):
        assert _name_or_tiered("C1=C2CCCCC2C1") == "bicyclo[4.2.0]oct-6-ene"

    def test_sp2_bicyclo651tetradec8ene(self):
        assert _name_or_tiered("C1=C2CCCCCCC(CCCC1)C2") == "bicyclo[6.5.1]tetradec-8-ene"

    def test_sp2_bicyclo831tetradeca4610trien2yne(self):
        assert _name_or_tiered("C1#CC2CCC=C(CCC=CC=C1)C2") == \
            "bicyclo[8.3.1]tetradeca-4,6,10-trien-2-yne"

    def test_sp2_tetracyclo_nonadecadiene(self):
        # Corrected locants with our engine's caret spelling (see brief's
        # dagger caveat - this row will not flip in bb_diff because the gold
        # oracle row lost its superscript carets on extraction).
        assert _name_or_tiered("C1=C2CC(CC1)CC1CC3=CC(CCC3)CC(C2)C1") == \
            "tetracyclo[7.7.1.1^3,7.1^11,15]nonadeca-3,11(18)-diene"

    def test_sp2_tricyclo_polycyclic_triple_branch_compound_aware(self):
        """(1)/(2) compound-aware ranking, exercised on the
        tricyclo (``polycyclic.py::VonBaeyerAnalyzer``) engine rather than
        bicyclo.py -- a fused-ring analogue of the bicyclo[8.3.1] ene+yne row
        above (a one-bond ring fusion added onto the same skeleton, chosen
        so the two orientation candidates STILL tie on 's
        secondary-bridge tiers, exactly like the bicyclo case). Before the
        Fix a performance pass port, this engine used a naive cited-locant compare for
        (1)/(2) and picked the WRONG orientation here too:
        'tricyclo[8.3.1.0^4,7]tetradeca-1(13),4,6-trien-8-yne' (1 compound
        locant) instead of this row's 0-compound-locant PIN form. Both
        strings round-trip via OPSIN 2.9.0 to the identical InChIKey
        (an InChIKey) -- pure locant-choice, not a
        wrong-molecule defect.
        """
        assert _name_or_tiered("C1#CC2CCC=C(CCC3=CC=C13)C2") == \
            "tricyclo[8.3.1.0^4,7]tetradeca-4,6,10-trien-2-yne"


class TestSP3PhenoFusedPnictogenChalcogenIndicatedH:
    """SP3 - "Pheno...ine components": the P/As/Sb 10H-isomer cites its
    indicated hydrogen. the Blue Book Blue Book list
    "phenoxaphosphinine (PIN, 10H-isomer shown)" beside "the PIN is 10H-phenoxazine"
    (:11781) for the N analogue, and a PIN cites it:3721 "in a preferred
    IUPAC name a locant and the symbol 'H' must be cited";:24639). The bare
    spelling is a listing spelling, by analogy with Note 1 (:14685, the note of
    the seniority lists: "Indicated hydrogen atoms are not shown in this kind of
    listing").
    """

    def test_sp3_phenoxaphosphinine(self):
        assert name_compound("c1ccc2c(c1)Oc1ccccc1P2") == "10H-phenoxaphosphinine"

    def test_sp3_phenoxarsinine(self):
        assert name_compound("c1ccc2c(c1)Oc1ccccc1[AsH]2") == "10H-phenoxarsinine"

    def test_sp3_phenoxastibinine(self):
        assert name_compound("c1cc[c]2c(c1)Oc1cccc[c]1[SbH]2") == "10H-phenoxastibinine"

    def test_sp3_phenothiarsinine(self):
        assert name_compound("c1ccc2c(c1)Sc1ccccc1[AsH]2") == "10H-phenothiarsinine"


class TestSP4ChalcogenAcidLocantOmission:
    """SP4 - "Functional replacement in systematic names of
    carboxylic acids": "Normally, these [italic O/S] locants are omitted,
    because the exact position of chalcogen atoms is not known or important
    in acids; such letter locants are used mainly in naming esters."
    (the Blue Book Blue Book,:30215, ex.:30235).

    2026-09-25 (pre-existing-failures plan, Task 5, R13): the omission applies
    where the position is NOT known (ex.:30235 is drawn C{O/Se}H;:31081
    "the location of the sulfur atoms is unknown"). A SMILES fixes it, so the
    determined C(=Se)-OH keeps the designator, like "hexanethioic O-acid (PIN)"
    (:30225). The SP4 rows now pin the designator.
    """

    def test_sp4_hexaneselenoic_acid(self):
    # PIN per R13::30215 designates the tautomer with an italic element
    # symbol; the designator is dropped only where "the location of the sulfur atoms is
    # unknown" (:31081 vs "1,3-dithiodicarbonic S1,S3-acid (PIN)":31083); the
    # determined -CS-OH is "hexanethioic O-acid (PIN)" (:30225), and "hexaneselenoic acid
    # (PIN)" (:30235) is drawn C{O/Se}H (undetermined). C(=Se)-OH is determined, so the
    # PIN keeps "O-acid". OPSIN RT exact (constitution; the tautomer is fixed by the FG
    # class, not by RT).
        assert _name_or_tiered("CCCCCC(O)=[Se]") == "hexaneselenoic O-acid"

    def test_sp4_amino_dioxopropanethioic_acid(self):
        # v52 a review-fix a performance pass D5: the polyfunctional strip that used to drop
        # the O/S designator for ALL three chalcogens is now Se/Te-only (see
        # TestD5PolyfunctionalThioAcidDesignatorKept below) -- this row's own
        # bare-form BB PIN (:30297) derives from an unspecified {O/S} drawing
        # tautomer that a SMILES cannot recover, so the SPECIFIED =S,-OH
        # tautomer here now correctly keeps the O-acid designator.
        assert _name_or_tiered("NC(=O)C(=O)C(O)=S") == "3-amino-2,3-dioxopropanethioic O-acid"


class TestD1RingSuffixLocantEssentialWhenRingUnsaturated:
    """D1 (v52 a review-fix a performance pass, REGRESSION) - "Citation of locants"
    (the Blue Book Blue Book): deny-by-default -- if ANY locant in a
    scope is essential, ALL locants in that scope must be cited. A ring
    double bond (ene) makes the ring numbering essential, so the appended
    ring-suffix locant '1' must be cited too, even though it would be omitted
    on the saturated ring (c)). BB verbatim: "...cyclohexa-2,5-
    diene-1-carboxylic acid (PIN)" (the Blue Book).
    """

    def test_d1_cyclohex3ene1carboximidic_acid(self):
        assert name_compound("N=C(O)C1CCC=CC1") == "cyclohex-3-ene-1-carboximidic acid"

    def test_d1_cyclohex3ene1carbohydrazonic_acid(self):
        assert name_compound("NN=C(O)C1CCC=CC1") == "cyclohex-3-ene-1-carbohydrazonic acid"

    def test_d1_cyclohex1ene1carboximidic_acid(self):
        assert name_compound("N=C(O)C1=CCCCC1") == "cyclohex-1-ene-1-carboximidic acid"

    def test_d1_cyclohex3ene1carboxylic_acid(self):
        # Pre-existing instance of the same root cause (not a v52 regression).
        assert name_compound("OC(=O)C1CCC=CC1") == "cyclohex-3-ene-1-carboxylic acid"

    def test_d1_guard_saturated_ring_still_omits_locant_one(self):
        # SP1 guard: a SATURATED ring has no essential locant, so the
        # trivial mono-suffix locant '1' must still be omitted.
        assert name_compound("N=C(O)C1CCCCC1") == "cyclohexanecarboximidic acid"
        assert name_compound("NN=C(O)C1CCCCC1") == "cyclohexanecarbohydrazonic acid"
        assert name_compound("O=S(Cl)c1ccccc1") == "benzenesulfinyl chloride"


class TestD5PolyfunctionalThioAcidDesignatorKept:
    """D5 (v52 a review-fix a performance pass, REGRESSION) -: the Task-5
    "polyfunctional thio acid -> drop O-acid designator" rule is REFUTED by
    BB counter-examples that KEEP the designator on a specified =S,-OH
    tautomer: "[(thiocarboxy)oxy]methanethioic O-acid (PIN)" (:31093),
    "2-(thiocarboxy)benzene-1-carbothioic S-acid (PIN)" (:30309),
    "carbonobromidothioic O-acid (PIN)" (:30846). The safe rule is to KEEP
    the thio designator in the polyfunctional path (matching the simple-acid
    path's existing behavior); only the unrefuted Se/Te strip stays.
    """

    def test_d5_hydroxypropanethioic_o_acid(self):
        assert _name_or_tiered("OCCC(O)=S") == "3-hydroxypropanethioic O-acid"

    def test_d5_oxobutanethioic_o_acid(self):
        assert _name_or_tiered("CC(=O)CC(O)=S") == "3-oxobutanethioic O-acid"

    def test_d5_guard_selenoic_o_acid_keeps_designator(self):
        # The Se/Te strip is refuted by the same specified-tautomer reading as thio.
    # PIN per R13::30215 designates the tautomer with an italic element
    # symbol; the designator is dropped only where "the location of the sulfur atoms is
    # unknown" (:31081 vs "1,3-dithiodicarbonic S1,S3-acid (PIN)":31083); the
    # determined -CS-OH is "hexanethioic O-acid (PIN)" (:30225), and "hexaneselenoic acid
    # (PIN)" (:30235) is drawn C{O/Se}H (undetermined). C(=Se)-OH is determined, so the
    # PIN keeps "O-acid". OPSIN RT exact (constitution; the tautomer is fixed by the FG
    # class, not by RT).
        assert _name_or_tiered("CCCCCC(O)=[Se]") == "hexaneselenoic O-acid"


class TestD3SulfurRingSuffixNumberingOutranksEne:
    """D3 (v52 a review-fix a performance pass) - "Numbering" (the Blue Book/
    the Blue Book): low locants go to (c) "principal characteristic
    groups and free valences (suffixes)" BEFORE (d) "saturation/unsaturation
    (ene/yne endings)". A ring -sulfonyl/-sulfinyl halide or -sulfonic/
    -sulfinic acid suffix must therefore get locant 1, and the ring ene
    locant follows from there -- mirroring the carboxylic-acid analogue
    'cyclohex-3-ene-1-carboxylic acid' (TestD1 above), which already got
    this right because its anchor is carbon-rooted; the sulfur-rooted
    anchor (S, not C) was skipped by the namer.py carbon-only loop.
    """

    def test_d3_cyclohex3ene1sulfinyl_chloride(self):
        assert name_compound("O=S(Cl)C1CCC=CC1") == "cyclohex-3-ene-1-sulfinyl chloride"

    def test_d3_cyclohex3ene1sulfinic_acid(self):
        assert name_compound("O=S(O)C1CCC=CC1") == "cyclohex-3-ene-1-sulfinic acid"

    def test_d3_cyclohex3ene1sulfonic_acid(self):
        assert name_compound("O=S(=O)(O)C1CCC=CC1") == "cyclohex-3-ene-1-sulfonic acid"

    def test_d3_cyclohex3ene1sulfonyl_chloride(self):
        assert name_compound("O=S(=O)(Cl)C1CCC=CC1") == "cyclohex-3-ene-1-sulfonyl chloride"


class TestD2SulfurRingSuffixLocantOneOmission:
    """D2 (v52 a review-fix a performance pass) - (c) (the Blue Book/
    the Blue Book): "the locant '1' is omitted in monosubstituted
    homogeneous monocyclic rings" (worked example 'cyclohexanethiol'). BB
    verbatim unlocanted sulfur stem: '4-(cyclohexanesulfinyl)morpholine-2-
    carboxylic acid (PIN)' (:31394). Requires D3's numbering fix (the
    suffix must already be at locant 1 before its omission is decided).
    """

    def test_d2_cyclohexanesulfonyl_chloride(self):
        assert name_compound("O=S(=O)(Cl)C1CCCCC1") == "cyclohexanesulfonyl chloride"

    def test_d2_cyclohexanesulfinyl_chloride(self):
        assert name_compound("O=S(Cl)C1CCCCC1") == "cyclohexanesulfinyl chloride"

    def test_d2_cyclohexanesulfonic_acid(self):
        assert name_compound("O=S(=O)(O)C1CCCCC1") == "cyclohexanesulfonic acid"

    def test_d2_cyclohexanesulfinic_acid(self):
        assert name_compound("O=S(O)C1CCCCC1") == "cyclohexanesulfinic acid"

    def test_d2_cyclopropanesulfinyl_chloride(self):
        assert name_compound("O=S(Cl)C1CC1") == "cyclopropanesulfinyl chloride"

    def test_d2_guard_unsaturated_ring_keeps_locant_one(self):
        # D3 guard: an ene ring makes the suffix locant essential
        # deny-default), so D2's omission licence must NOT fire here.
        assert name_compound("O=S(=O)(Cl)C1CCC=CC1") == "cyclohex-3-ene-1-sulfonyl chloride"

    def test_d2_guard_benzene_unaffected(self):
        assert name_compound("O=S(Cl)c1ccccc1") == "benzenesulfinyl chloride"

    def test_d2_guard_disubstituted_ring_keeps_locants(self):
        assert name_compound("OC1CCC(CC1)S(=O)(=O)O") == "4-hydroxycyclohexane-1-sulfonic acid"

    def test_d2_guard_heteroaromatic_keeps_locant(self):
        assert name_compound("O=S(Cl)c1ccncc1") == "pyridine-4-sulfinyl chloride"


class TestD4PhenoSeTeAzineIndicatedH:
    """D4 (v52 a review-fix a performance pass, pre-existing data gap) - table
    (the Blue Book,:11787): "X = Se phenoselenazine (10H-isomer shown;
    the PIN is 10H-phenoselenazine)", "X = Te... the PIN is
    10H-phenotellurazine". These are N-cases (like phenoxazine/phenothiazine)
    where 10H- IS part of the PIN, unlike the P/As/Sb O-cases where "10H-isomer
    shown" is descriptive only.
    """

    def test_d4_10h_phenoselenazine(self):
        assert name_compound("c1ccc2c(c1)Nc1ccccc1[Se]2") == "10H-phenoselenazine"

    def test_d4_10h_phenotellurazine(self):
        assert name_compound("c1ccc2c(c1)Nc1ccccc1[Te]2") == "10H-phenotellurazine"
