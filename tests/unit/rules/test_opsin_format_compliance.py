"""
Tests for OPSIN format compliance fixes (a phase, Plan 01).

Covers:
  : Acylamino bracket format -- OPSIN requires brackets around acylamino prefixes
  : phenylamino -> anilino -- OPSIN recognizes "anilino" as a simple substituent

These are root-cause fixes at the point of name generation, not postprocessors.
"""

import os
import subprocess

import pytest

from orthonym.namer import name_compound
from tests.support.jars import jar_or_none

# Check OPSIN availability for round-trip tests
OPSIN_JAR = jar_or_none()
OPSIN_AVAILABLE = OPSIN_JAR is not None


def opsin_parses(name: str) -> bool:
    """Return True if OPSIN can parse the given IUPAC name to a SMILES."""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Section 1: Anilino prefix tests
# ---------------------------------------------------------------------------


class TestAnilinoPrefix:
    """Verify phenylamino -> anilino substitution in all contexts."""

    @pytest.mark.unit
    def test_anilino_prefix_on_chain(self):
        """Nc1ccc(Nc2ccccc2)cc1 is the PIN N-phenylbenzene-1,4-diamine.

        SUPERSEDED (C4b, 2026-07-03): this test previously required the OLD
        non-PIN general name '...anilinobenzene' for this molecule. Under the
        C4/C4b aromatic-diamine fix both amino N's are the principal group, so
        the PIN is the benzene-1,4-diamine parent with the aryl group cited as
        an italic-N prefix -> 'N-phenylbenzene-1,4-diamine' (OPSIN round-trip
        verified). No 'anilino' fragment appears; that is correct PIN behaviour,
        not a regression. 'phenylamino' must still never appear.
        """
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # (the Blue Book): "Superscript arabic numbers, which are the locants of the parent structure, are used to differentiate the nitrogen atoms of di- and polyamines"; "N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)" (:26404) -- 'N1-', not a bare 'N-'. OPSIN RT exact.
        name = name_compound("Nc1ccc(Nc2ccccc2)cc1")
        assert name == "N1-phenylbenzene-1,4-diamine", (
            f"Expected PIN 'N1-phenylbenzene-1,4-diamine', got '{name}'"
        )
        assert "phenylamino" not in name, f"'phenylamino' should not appear in '{name}'"

    @pytest.mark.unit
    def test_anilino_prefix_on_ring(self):
        """Compound with -NHPh on a ring parent produces 'anilino'."""
        # 2-anilinopyridine
        name = name_compound("c1ccc(Nc2ccccn2)cc1")
        assert name is not None, "name_compound returned None"
        # The compound may name differently through decomposition,
        # but if anilino appears it should never be phenylamino
        assert "phenylamino" not in name, f"'phenylamino' should not appear in '{name}'"

    @pytest.mark.unit
    def test_anilino_no_outer_brackets(self):
        """anilino should NOT have outer parentheses (OPSIN simple substituent).

        Uses 4-anilinobenzoic acid: the senior CO2H keeps the -NHPh amine
        demoted to the 'anilino' PREFIX seniority), so this still
        exercises the anilino-prefix path. (The bare NH2/NHPh diamine now names
        as the PIN N-phenylbenzene-1,4-diamine — see test_anilino_prefix_on_chain.)
        """
        name = name_compound("OC(=O)c1ccc(Nc2ccccc2)cc1")
        assert "anilino" in name, f"Expected 'anilino' prefix in '{name}'"
        # Should be "...anilinobenzoic acid" not "...(anilino)benzoic acid"
        assert "(anilino)" not in name, (
            f"'(anilino)' with brackets found in '{name}' -- anilino is a simple substituent"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_anilino_opsin_parses(self):
        """OPSIN should parse names containing 'anilino'."""
        name = name_compound("OC(=O)c1ccc(Nc2ccccc2)cc1")
        if "anilino" in name:
            assert opsin_parses(name), f"OPSIN failed to parse '{name}'"


# ---------------------------------------------------------------------------
# Section 2: Acylamino bracket format tests
# ---------------------------------------------------------------------------


class TestAcylaminoBrackets:
    """N-acyl prefixes use the amido form method (1) = PIN).

    Wave2 T1c: the method-(2) '(pentanoylamino)' bracketed forms were
    replaced by the preferred amido family — formamido/acetamido/
    {stem}anamido — which are simple prefixes and take NO enclosing marks
    (Blue Book: 4-formamidobenzoic acid, 4-acetamidobenzoic acid).
    """

    @pytest.mark.unit
    def test_amido_short_chain(self):
        """C2 acyl on glycine -> acetamido, unbracketed."""
        name = name_compound("CC(=O)NCC(=O)O")
        # PIN per R3 (analogy): "only acetic acid, benzoic acid, and oxamic acid
        # can be substituted" the Blue Book, "acetic acid (PIN) ethanoic acid":29725;
        # "All locants are omitted for parent compounds when all substitutable
        # hydrogen atoms have the same locant.":3031; "acetamido* = acetylamino" (the '*'
        # "designates the preferred prefix"):55416/:55422, unenclosed as in
        # "(4-acetamido-3-methylphenyl)arsonic acid (PIN)":33010. OPSIN RT exact (TRIAGE.csv;
        # re-checked in Task 7/8).
        assert name == "acetamidoacetic acid", (
            f"Expected 'acetamidoacetic acid', got '{name}'"
        )

    @pytest.mark.unit
    def test_amido_long_chain(self):
        """C5 acyl on glutamic acid -> pentanamido, unbracketed."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert name == "2-pentanamidopentanedioic acid", (
            f"Expected '2-pentanamidopentanedioic acid', got '{name}'"
        )

    @pytest.mark.unit
    def test_amido_medium_chain(self):
        """C3 acyl on GABA -> propanamido, unbracketed."""
        name = name_compound("CCC(=O)NCCCC(=O)O")
        assert name == "4-propanamidobutanoic acid", (
            f"Expected '4-propanamidobutanoic acid', got '{name}'"
        )

    @pytest.mark.unit
    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not found")
    def test_amido_opsin_parses(self):
        """OPSIN should parse names with amido prefixes."""
        name = name_compound("CCCCC(=O)NC(CCC(=O)O)C(=O)O")
        assert "pentanamido" in name, f"Expected 'pentanamido' in '{name}'"
        assert opsin_parses(name), f"OPSIN failed to parse '{name}'"


# ---------------------------------------------------------------------------
# Section 3: No bare phenylamino regression (parametrized)
# ---------------------------------------------------------------------------


class TestNoBarePhenylamino:
    """Parametrized test ensuring phenylamino never appears in any context."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "smiles,description",
        [
            ("Nc1ccc(Nc2ccccc2)cc1", "aminodiphenylamine"),
            ("c1ccc(Nc2ccccc2)cc1", "diphenylamine"),
            ("c1ccc(Nc2ccccn2)cc1", "phenyl-aminopyridine"),
            ("c1ccc(NCCCC(=O)O)cc1", "phenyl-amino-acid"),
            ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
            ("c1ccc(Nc2ccc3ccccc3c2)cc1", "phenyl-aminonaphthalene"),
        ],
        ids=[
            "aminodiphenylamine",
            "diphenylamine",
            "phenyl-aminopyridine",
            "phenyl-amino-acid",
            "N-phenylacetamide",
            "phenyl-aminonaphthalene",
        ],
    )
    def test_no_bare_phenylamino_in_any_context(self, smiles, description):
        """No compound should produce 'phenylamino' in its IUPAC name."""
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {description}"
        assert "phenylamino" not in name, (
            f"{description}: 'phenylamino' found in '{name}' -- should be 'anilino'"
        )


# ---------------------------------------------------------------------------
# Section 4: Canary regression guard
# ---------------------------------------------------------------------------


# Task 11 (pre-existing-failures plan, 2026-09-26): canary rows whose current PIN-tier name
# (name_compound, gate off) is NOT a verified new baseline. 31 are not OPSIN full-InChIKey exact
# (wrong molecule, unparseable, connectivity-only, or the engine declines; routed to T3/ in
# TRIAGE_calls.csv); 189 are RT-exact but not the PIN (peptide names, von Baeyer names of
# ortho-fused systems, method (2) polyol esters, acids written as hydroxy + oxo,...). They are
# NOT re-baselined: the fixture keeps their old value, and the test pins the CURRENT name
# below, so it stays green only for the verified rows and turns red as soon as a defect row
# changes (then re-verify it: OPSIN full InChIKey + the Blue Book, and move it to the fixture).
# The Task 12 fix round owns them (TRIAGE.md ' outcome').
# fix a performance pass (wp2-numbering): calls 113 (DK-OXANE), 301 and 665 (DK-P14G) now give the
# names those classes asked for (c):3256 / (g):3307) and moved to the fixture; the
# two classes are gone with them.
# fix a performance pass (wp7-verify-fixes): call 296 (DK-ALPHA) now gives the order (the shared
# key compares letters first,:3442/:3477) and moved to the fixture; the class is gone with it.

# Class -> reason (Task 11, 2026-09-26). The reasons, the Blue Book lines and the best-effort
# evidence per row are in internal notes (disposition 'known-defect').
_CANARY_DEFECT_REASONS = {
    "DK-ACIDPFX": (
        "a carboxylic acid written as 'hydroxy' + 'oxo'/'-one' on the stereoparent's "
        "terminal carbon; the acid is the principal characteristic group ('-oic acid'; "
        "P-41 Table 4.1, acids > esters > ketones > alcohols; P-63.1.4 :27257)"
    ),
    "DK-ADAM": (
        "tricyclo[3.3.1.1^3,7]decane for adamantane; P-23.7 :9881 retains adamantane as "
        "the PIN (large-polycycle Task 12)"
    ),
    "DK-ETHANAMINE": (
        "'ethan-1-amine' with only N-substituents: P-14.3.4.2 (b) omits the locant "
        "('N,N-diethylethanamine (PIN)' :26235); the old fixture value was right"
    ),
    "DK-GLYM2": (
        "method (2) polyol ester name; P-65.6.3.3.3.2 :31836 makes method (1) the PIN, "
        "which OPSIN 2.9.0 cannot parse (verification-blocked, T5 rows 10/11)"
    ),
    "DK-GLYSUB": (
        "acyloxy prefixes on a propanol ('...propan-3-ol'): the ester outranks the "
        "alcohol (P-41 Table 4.1, R5) and the -ol takes the lowest locant; PIN is the "
        "method (1) ester (P-65.6.3.3.3.2 :31836)"
    ),
    "DK-GUAN": (
        "'guanidino' is not acceptable in PINs; the preferred prefix is "
        "'carbamimidoylamino' (P-66.4.1.2.1.3 :34266-34268; :1700)"
    ),
    "DK-INDH": (
        # fix a performance pass (wp6-tests; whole-branch review nit 15): the old reason
        # ('indicated hydrogen at the ketone', cannot apply -- no
        # 4H-2-benzopyran isomer exists (with C4 sp3, C3 has no double-bond partner).
        "hydro prefixes instead of added indicated hydrogen: the parent's indicated "
        "hydrogen cannot sit at the ketone, so P-58.2.3.1.4 (:24841) sends it to "
        "P-58.2.3.1.3 (:24806): indicated hydrogen at the lowest locant (1H), the ketone "
        "by added indicated hydrogen; P-58.2.2 (:24687) prefers added hydrogen to hydro "
        "prefixes; cf. 'hexahydro-1H-2-benzopyran-1,3(4H)-dithione (PIN)' (:32549), "
        "'3,4-dihydronaphthalen-1(2H)-one (PIN)' (:3276). PIN "
        "'(3S)-6,7-dihydroxy-8-methoxy-3-methyl-1H-2-benzopyran-4(3H)-one' (OPSIN exact)"
    ),
    "DK-NACYL": (
        "N-acyl float onto an amino-acid name ('N-[...]pentanoyl](2S)-2-amino...acid'); "
        "not a PIN spelling (P-62.2.2.1 :26225; PIN form as T9 row 18)"
    ),
    "DK-NOTEXACT": (
        "new name is not OPSIN full-InChIKey exact (wrong molecule, unparseable, "
        "connectivity-only, or the engine declines); routed in TRIAGE.md 'Canary calls "
        "routed per call' and 'T4 outcome'"
    ),
    "DK-P101CIP": (
        "CIP descriptors on the stereoparent's implied centres instead of the P-101 "
        "alpha/beta form (P-101.2.6 :51047 'the name of a fundamental parent structure "
        "usually implies the absolute configuration'; P-101.2.6.1.1 alpha/beta at "
        "chirality centres; the T8 row 104 re-baseline dropped them); the C-26/C-27 swap "
        "in 362/566 is P-14.4 (c) with no BB ruling"
    ),
    "DK-PEP": (
        "peptide name at the PIN tier; peptides are not PINs, the PIN is the substitutive "
        "name (controller ruling, CHEBI:141425)"
    ),
    "DK-RINGNLOC": (
        "'N-' locant for a ring nitrogen; ring atoms take numeric locants "
        "(1-(4-methoxybenzoyl)...; cf. '1-methylpyrrolidin-2-yl' :4679); also the N-acyl "
        "lactam PIN form (pseudoketone vs hidden amide, P-66.1.3) needs a ruling"
    ),
    "DK-SFXNUM": (
        "ring numbering puts 'ene' before the suffixes: the two -OH are 1,3 apart, so the "
        "diol must be -1,3-diol (P-31.1.4 / P-14.4 (c) :3256 suffixes before hydro/ene)"
    ),
    "DK-VBFUSED": (
        "von Baeyer name for a system with ortho-fused rings of 5+ members; the "
        "(hydro/bridged) fusion name is the PIN (P-52.2.4.1 :23710; P-44.2.2.2 :19532 "
        "fused > bridged fused > von Baeyer); R22 class, large-polycycle Task 6a"
    ),
    "DK-VBNUM": (
        # fix a performance pass (wp6-tests; whole-branch review F5, F15b): rows 536 and 599
        # ('RT-exact, no BB ruling' re-baselines) and 661 (was DK-VBFUSED).
        "von Baeyer numbering: the secondary-bridge superscripts are not the lowest "
        "(P-23.2.6.2.4 :9685 'The superscript locants for the secondary bridges must be as "
        "low as possible'; 599 {14,17} for {5,8}, 661 0^3,7 for 0^1,5), or every ring "
        "criterion ties and P-14.4 (g) (:3307) is skipped (536: ethyl 32, not 5). Producer: "
        "polycyclic.VonBaeyerAnalyzer._choose_lowest_locant_numbering drops the "
        "lower-superscript candidate by descriptor-string equality; _locant_criteria_key "
        "has no (g) tier. The fixture holds the BB-derived name for 536/599. Open: whether "
        "661 (a 4-ring shared by a 5- and a 6-ring meeting in one atom) is fusion-named is "
        "not settled by any Blue Book example (P-25.3.1.1.2 :11871; :23747 and :23767 do not "
        "cover it), and 599 may be a phane (P-52.2.5.1 (1) :23828, ASSUMED)"
    ),
}

# Suite fix j6-breadth (2026-09-27): canary calls 135/142/323 (the prenylated-phenol trienoic
# acid) and 171 (the tetramethyl-1,3-dioxolanyl terpene) left this table -- their raw PIN-tier
# names are RT-exact now (strict XPASS below) and they are verified fixture rows
# (tests/integration/test_canary_rt75.py, tags 'j6 RB-RINGBR' / 'j6 RB-HWSTEM').
# Breadth job 3 review fixes (2026-09-29): canary calls 155 (DK-NOTEXACT, the fused
# ring ketone that raised TypeError on the '4a'/7 locant sort, fixed by 2c79719e3) and
# 684 (DK-SFXNUM, fixed by 7c39f2243: every ring -OH of the alcohol class is a suffix)
# left this table: both raw PIN-tier names are RT-exact now and are verified fixture
# rows (tests/integration/test_canary_rt75.py, tags 'b3 RB-...').
# Bridged fused S0 follow-up (2026-10-01): canary call 175 (DK-NOTEXACT, the 'cyclohexa[b]pyridine'
# acid with no hydro prefixes) left this table: the hydro prefixes now go on the mancude parent
#, the Blue Book), its raw PIN-tier name '4,7,8-trihydroxy-7,8-dihydroquinoline-
# 2-carboxylic acid' is RT-exact (OPSIN 2.9.0 full key and FixedH InChI) and is a verified fixture
# row (tests/integration/test_canary_rt75.py, tag 'bf RB-HYDRO').
# SMILES -> (current PIN-tier name with the gate off, class, canary call numbers).
CANARY_KNOWN_DEFECTS = {
    "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O": (
        "2-(acetyloxy)-4-hydroxyheptadec-16-yne-1,4-diol",
        "DK-NOTEXACT", (6,),
    ),
    "CC1=CC[C@]23O[C@@]2(C)CC[C@@H]2[C@H](OC(=O)[C@H]2C)[C@@H]13": (
        "(1R,3S,6S,7S,10S,11R)-3,7,12-trimethyl-2,9-dioxatetracyclo[9.3.0.0^1,3.0^6,10]tetradec-12-en-8-one",
        "DK-VBFUSED", (12,),
    ),
    "N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N1CCC[C@@H]1C(=O)O": (
        "serylseryl-D-proline",
        "DK-PEP", (14,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCC(=O)O)C(=O)O": (
        "aspartyltryptophylglutamic acid",
        "DK-PEP", (17,),
    ),
    "C[C@H]1C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@@]32OC(=O)/C=C\\[C@@](C)(O)C1=O": (
        "(1S,2Z,5S,7R,8Z,12R,15S,16S,17S,18R,20S)-15-benzyl-7-hydroxy-5,7,17,18-tetramethyl-11,19-dioxa-14-azatetracyclo[10.8.0.0^12,16.0^18,20]icosa-2,8-diene-6,10,13-trione",
        "DK-VBFUSED", (22, 318, 325),
    ),
    "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C": (
        "(1S,2E,7S,8E,12R,15S,16S,17S)-7-hydroxy-3,17,18-trimethyl-15-(2-methylpropyl)-11-oxa-14-azatricyclo[10.7.0.0^12,16]nonadeca-2,8,18-triene-10,13-dione",
        "DK-VBFUSED", (23, 319, 327),
    ),
    "C[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)O": (
        "aspartylalanyllysine",
        "DK-PEP", (29,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(=O)O)C(=O)O": (
        "glutamylglutamylisoleucine",
        "DK-PEP", (35,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)O": (
        "aspartylserylserine",
        "DK-PEP", (36,),
    ),
    "CC1(C)C[C@H](O)[C@]23CC[C@@H](O)[C@](C)(CC[C@@H]12)C3": (
        "(1S,2S,5S,8R,9R)-4,4,8-trimethyltricyclo[6.3.1.0^1,5]dodecane-2,9-diol",
        "DK-VBFUSED", (39,),
    ),
    "NCCCC[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O": (
        "glutaminyllysylcysteine",
        "DK-PEP", (44,),
    ),
    "C[C@H]1C(=O)O[C@@H]2CCN3CC=C(COC(=O)[C@](C)(O)[C@]1(C)O)[C@H]23": (
        "(1R,4R,5R,6R,16R)-5,6-dihydroxy-4,5,6-trimethyl-2,8-dioxa-13-azatricyclo[8.5.1.0^13,16]hexadec-10-ene-3,7-dione",
        "DK-VBFUSED", (48,),
    ),
    "NC(=O)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "asparaginylaspartylaspartic acid",
        "DK-PEP", (50,),
    ),
    "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CC(N)=O)C(=O)O": (
        "leucylhistidylasparagine",
        "DK-PEP", (65,),
    ),
    "CC1=C(O)C(=O)[C@]2(O)C[C@H]3C[C@](C)(C(=O)O)C[C@H]3[C@]12C": (
        "unknown organic compound",
        "DK-NOTEXACT", (68,),
    ),
    "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O": (
        "(3S)-6,7-dihydroxy-8-methoxy-3-methyl-3,4-dihydro-1H-2-benzopyran-4-one",
        "DK-INDH", (133,),
    ),
    "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3": (
        "(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-trihydroxy-4,4,14-trimethyl-21-oxocholesta-8,23-dien-3-yl acetate",
        "DK-ACIDPFX", (143, 317),
    ),
    "CCN(CC)Cc1ccccc1": (
        "N-benzyl-N-ethylethan-1-amine",
        "DK-ETHANAMINE", (149,),
    ),
    "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC": (
        # j7 (TRIAGE g5 C12): now the Blue Book method (1) spelling with the single
        # stereo-bearing anion enclosed:31846); still DK-NOTEXACT
        # because OPSIN 2.9.0 parses no multi-anion locant ester name.
        "propane-1,2,3-triyl 2-[(11Z,14Z)-icosa-11,14-dienoate] 1,3-di[(9Z,12Z)-octadeca-9,12-dienoate]",
        "DK-NOTEXACT", (151, 315, 326),
    ),
    "COC1CC(=O)C23C(=O)NC(CC(C)C)C2C(C)C(C)=CC3/C=C(\\C)CCCC1O": (
        "(9E)-5-hydroxy-4-methoxy-9,13,14-trimethyl-16-(2-methylpropyl)-17-azatricyclo[9.7.0.0^1,15]octadeca-9,12-diene-2,18-dione",
        "DK-VBFUSED", (154, 320, 333),
    ),
    "CC(C)=CCOc1ccc(C2=C(CC(C)C)C(=O)NC2=O)cc1": (
        # fix a performance pass (wp5): the ring now takes the form (was
        # '...-2,5-dihydro-1H-pyrrole-2,5-dione'); the row stays a known defect: the
        # gate-off raw name still says 'pentyloxy' for the prenyloxy group (wrong
        # molecule). With the gate on the PIN tier abstains and best-effort is exact,
        # as at the task start.
        "3-(2-methylpropyl)-4-[4-(pentyloxy)phenyl]-1H-pyrrole-2,5-dione",
        "DK-NOTEXACT", (172,),
    ),
    "COc1ccc(C(=O)N2CCCC2=O)cc1": (
        "N-(4-methoxybenzoyl)pyrrolidin-2-one",
        "DK-RINGNLOC", (173, 293),
    ),
    "CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1": (
        "unknown organic compound",
        "DK-NOTEXACT", (174,),
    ),
    "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12": (
        "N-heptanoyl(7aS)-3-amino-5,6,7,7a-tetrahydropyrrolizin-1(3aH)-one",
        "DK-NOTEXACT", (179,),
    ),
    "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C": (
        "unknown organic compound",
        "DK-NOTEXACT", (180,),
    ),
    "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C": (
        "unknown organic compound",
        "DK-NOTEXACT", (181,),
    ),
    "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl": (
        "(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate—hydrogen chloride (1/1)",
        "DK-NOTEXACT", (182,),
    ),
    "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12": (
        "unknown organic compound",
        "DK-NOTEXACT", (202, 316, 338),
    ),
    "NC(=O)CC[C@H](NC(=O)CNC(=O)[C@@H](N)CO)C(=O)O": (
        "serylglycylglutamine",
        "DK-PEP", (230,),
    ),
    "N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": (
        "aspartylglycyltryptophan",
        "DK-PEP", (239,),
    ),
    "C[C@@H](O)[C@H](NC(=O)CN)C(=O)N[C@@H](CC(N)=O)C(=O)O": (
        "glycylthreonylasparagine",
        "DK-PEP", (242,),
    ),
    "C[C@H](NC(=O)[C@@H](N)CS)C(=O)N[C@H](C(=O)O)[C@@H](C)O": (
        "cysteinylalanylthreonine",
        "DK-PEP", (247,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](CS)C(=O)O": (
        "glutamylthreonylcysteine",
        "DK-PEP", (251,),
    ),
    "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O": (
        "aspartylvalylphenylalanine",
        "DK-PEP", (255,),
    ),
    "CC(C)[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O": (
        "glutamylvalylphenylalanine",
        "DK-PEP", (258,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": (
        "glutaminylthreonyltryptophan",
        "DK-PEP", (263,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(N)=O)C(=O)O": (
        "tryptophylthreonylasparagine",
        "DK-PEP", (264,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CC(N)=O)[C@@H](C)O)C(=O)O": (
        "asparaginylthreonylisoleucine",
        "DK-PEP", (265,),
    ),
    "C[C@@H](O)[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "threonylglutaminylaspartic acid",
        "DK-PEP", (273,),
    ),
    "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "(2S)-2-(phenylalanylglutamylamino)butanedioic acid",
        "DK-PEP", (277,),
    ),
    "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@@H](CO)COC(=O)CCC": (
        "(2S)-1-(butanoyloxy)-3-hydroxypropan-2-yl (9Z)-octadec-9-enoate",
        "DK-GLYM2", (281,),
    ),
    "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC": (
        "(2S)-3-(decanoyloxy)-2-hydroxypropyl docosanoate",
        "DK-GLYM2", (284,),
    ),
    "OC1C2CC3CC1CC(O)(C3)C2": (
        "tricyclo[3.3.1.1^3,7]decane-1,4-diol",
        "DK-ADAM", (344,),
    ),
    "CC1(C)CC[C@]2(C(=O)O)CC[C@]3(C)C(=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@]43C)[C@@H]2C1": (
        "unknown organic compound",
        "DK-NOTEXACT", (353,),
    ),
    "C[C@@H]1CC[C@@H]2C=C(C(=O)O)[C@H]3C[C@](C)(C(=O)O)C[C@]132": (
        "(1S,4R,7R,8R,10S)-7,10-dimethyltricyclo[6.3.0.0^4,8]undec-2-ene-2,10-dicarboxylic acid",
        "DK-VBFUSED", (356,),
    ),
    "C[C@H](CCC1OCC1CO)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C": (
        "(3R,5S,7R,8R,9S,10S,12S,13R,14S,17R,20R)-24,27-epoxycholestane-3,7,12,26-tetrol",
        "DK-P101CIP", (362,),
    ),
    "NCCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O": (
        "(2S)-6-amino-2-(tyrosylaspartylamino)hexanoic acid",
        "DK-PEP", (364,),
    ),
    "CC(C)C[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "aspartylleucylaspartic acid",
        "DK-PEP", (368,),
    ),
    "NC(=O)CC[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "glutamylglutaminylhistidine",
        "DK-PEP", (379,),
    ),
    "CC(C)C[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](N)CCCCN)C(=O)O": (
        "lysylasparaginylleucine",
        "DK-PEP", (382,),
    ),
    "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCC(N)=O)C(=O)O": (
        "leucylhistidylglutamine",
        "DK-PEP", (383,),
    ),
    "CC1=C2C(=O)C[C@@]2(C)[C@@H]2C[C@@H]3[C@H](O)C[C@@H](C)[C@@]2(CC1)C3(C)C": (
        "(1S,3S,4S,11R,12R,14R)-14-hydroxy-4,8,12,15,15-pentamethyltetracyclo[9.3.1.0^3,11.0^4,7]pentadec-7-en-6-one",
        "DK-VBFUSED", (385,),
    ),
    "NC(=O)CC[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylphenylalanylglutamine",
        "DK-PEP", (400,),
    ),
    "NC(=O)C[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O": (
        "glutamylasparaginylphenylalanine",
        "DK-PEP", (406,),
    ),
    "C[C@@H]1CC[C@H]2C(C=O)=C[C@@H]3CC(C)(C)CC132": (
        "(1R,4S,9R)-6,6,9-trimethyltricyclo[6.3.0.0^4,8]undec-2-ene-2-carbaldehyde",
        "DK-VBFUSED", (413,),
    ),
    "C/C(=C\\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@@H](O)C(C)(C)[C@@H]1C[C@H]3O)C(=O)O": (
        # j7: the acid is the '-26-oic acid' suffix now:52548); the row keeps
        # its whole-graph R/S block on implied centres (was DK-ACIDPFX).
        "(3R,5R,7R,10S,13R,14R,17R,20R,24E)-3,7-dihydroxy-4,4,14-trimethylcholesta-8,24-dien-26-oic acid",
        "DK-P101CIP", (416,),
    ),
    "C/C(=C\\[C@@H](O)C[C@@H](C)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)C[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1C[C@@H]3O)C(=O)O": (
        # j7: the acid is the '-26-oic acid' suffix now:52548); the row keeps
        # its whole-graph R/S block on implied centres (was DK-ACIDPFX).
        "(3S,5R,7S,10S,13R,14R,17R,20R,23S,24E)-3,7,23-trihydroxy-4,4,14-trimethyl-11,15-dioxocholesta-8,24-dien-26-oic acid",
        "DK-P101CIP", (417,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylglutamylisoleucine",
        "DK-PEP", (419,),
    ),
    "C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(Cl)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO": (
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-chloro-11,17,21-trihydroxy-16-methylpregna-1,4-diene-3,20-dione",
        "DK-P101CIP", (421,),
    ),
    "C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O": (
        "glycylaspartylalanine",
        "DK-PEP", (428,),
    ),
    "CC1CCC2C(C(=O)O)=CC3CC(C)(C)CC132": (
        "6,6,9-trimethyltricyclo[6.3.0.0^4,8]undec-2-ene-2-carboxylic acid",
        "DK-VBFUSED", (429,),
    ),
    "N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "glutamyltryptophylaspartic acid",
        "DK-PEP", (432,),
    ),
    "Oc1ccc(CCC(O)CC/C=C/c2ccccc2)cc1": (
        "unknown organic compound",
        "DK-NOTEXACT", (435,),
    ),
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)CC(=O)O)C(=O)O": (
        "aspartylaspartylleucine",
        "DK-PEP", (437,),
    ),
    "CC(C)C1=C2[C@H]3CC=C(C=O)CC(=O)[C@]3(C)CC[C@@]2(C)CC1": (
        "unknown organic compound",
        "DK-NOTEXACT", (440,),
    ),
    "C[C@@H](O)[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(N)=O)C(=O)O": (
        "threonyltryptophylasparagine",
        "DK-PEP", (442,),
    ),
    "CC(C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "valylasparaginylaspartic acid",
        "DK-PEP", (443,),
    ),
    "NCCCC[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O": (
        "lysylglutamylglutamic acid",
        "DK-PEP", (454,),
    ),
    "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)O": (
        "isoleucylglutaminylleucine",
        "DK-PEP", (459,),
    ),
    "CC(C)[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "N-[(2S)-2,5-diamino-5-oxopentanoyl](2S)-2-amino-3-methylbutanoic acid",
        "DK-NACYL", (460,),
    ),
    "C[C@@H]1C=C[C@H]2C3C1CC[C@@](C)(O)O[C@@H]3OC(=O)[C@@H]2C": (
        "(1R,4R,5S,8R,12S)-12-hydroxy-4,8,12-trimethyl-2,13-dioxatricyclo[7.4.1.0^5,14]tetradec-6-en-3-one",
        "DK-VBFUSED", (472,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "aspartyltyrosylhistidine",
        "DK-PEP", (476,),
    ),
    "CC1(C)CC23[C@@H]4CC(=O)[C@@H]2COC(=O)[C@@H]3CC[C@H]41": (
        "(2R,5S,9R,12R)-13,13-dimethyl-7-oxatetracyclo[7.5.0.0^1,5.0^2,12]tetradecane-4,8-dione",
        "DK-VBFUSED", (486,),
    ),
    "C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylglutamylalanine",
        "DK-PEP", (494,),
    ),
    "N[C@@H](CO)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "serylaspartylhistidine",
        "DK-PEP", (495,),
    ),
    "CC1=CCC(=O)CC(=O)[C@@]23C(=O)N[C@@H](CC(C)C)[C@@H]2[C@H](C)C(C)=C[C@@H]3C1": (
        "(1S,9S,12S,13R,14S)-7,11,12-trimethyl-14-(2-methylpropyl)-15-azatricyclo[7.7.0.0^1,13]hexadeca-6,10-diene-2,4,16-trione",
        "DK-VBFUSED", (500,),
    ),
    "N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "glutamylcysteinylhistidine",
        "DK-PEP", (501,),
    ),
    "CC(=O)OC[C@]12CC[C@H](O)C(C)(C)[C@@H]1CCC1=C2CC[C@]2(C)[C@@H]([C@H](C)[C@H](C/C=C(/C)C(=O)O)OC(C)=O)CC[C@@]12C": (
        "(3S,5R,10R,13R,14R,17R,20S,22S,24Z)-3,26-dihydroxy-4,4,14-trimethyl-26-oxocholesta-8,24-dien-19,22-diyl diacetate",
        "DK-ACIDPFX", (510,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@@H](N)CC(N)=O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "asparaginylisoleucylhistidine",
        "DK-PEP", (518,),
    ),
    "C[C@H](CC[C@H](O)C(C)(C)O)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3": (
        "(3S,5R,10S,13R,14R,17R,20R,24S)-4,4,14-trimethylcholest-8-ene-3,24,25-triol",
        "DK-P101CIP", (522,),
    ),
    "CC(C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylglutamylvaline",
        "DK-PEP", (524,),
    ),
    "NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O": (
        "N-glycylglutamylglycine",
        "DK-PEP", (538,),
    ),
    "CC1=CC2/C=C(\\C)CCC3OC3/C=C/C(=O)C23C(=O)NC(CC(C)C)C3C1C": (
        "(2E,9E)-3,17,18-trimethyl-15-(2-methylpropyl)-7-oxa-14-azatetracyclo[10.7.0.0^6,8.0^12,16]nonadeca-2,9,18-triene-11,13-dione",
        "DK-VBFUSED", (540,),
    ),
    "CCC(=O)OCC(=O)[C@@]1(OC(=O)CC)[C@@H](C)C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C": (
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-fluoro-11-hydroxy-16-methyl-3,20-dioxopregna-1,4-dien-17,21-diyl dipropanoate",
        "DK-P101CIP", (550,),
    ),
    "NC(=O)C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O": (
        "histidylglutamylasparagine",
        "DK-PEP", (562,),
    ),
    "N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O": (
        "histidylglutamyltyrosine",
        "DK-PEP", (565,),
    ),
    "C/C(=C\\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3=O)CO": (
        "(3S,5R,10S,13R,14R,17R,20R,24E)-3,26-dihydroxy-4,4,14-trimethylcholesta-8,24-dien-7-one",
        "DK-P101CIP", (566,),
    ),
    "C#CCCCCCCCCCCCC(CC(O)CO)OC(C)=O": (
        "4-(acetyloxy)-2-hydroxyheptadec-16-yne-1,2-diol",
        "DK-NOTEXACT", (567,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)CCC(=O)O)C(=O)O": (
        "glutamyllysylthreonine",
        "DK-PEP", (572,),
    ),
    "C[C@H](O)[C@H]([NH3+])C(=O)[O-]": (
        "unknown organic compound",
        "DK-NOTEXACT", (582,),
    ),
    "CC(C)C[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)O": (
        "leucylglutaminylasparagine",
        "DK-PEP", (591,),
    ),
    "CC(C)C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CC(N)=O)C(=O)O": (
        "asparaginylglutamylleucine",
        "DK-PEP", (596,),
    ),
    "CC(C)CC1C(=O)N2CC(C)CC2C(=O)NC(C(C)C)C(=O)OC(C(C)C)C(=O)N2NCCCC2C(=O)N2NCC(O)CC2C(=O)N1C": (
        "unknown organic compound",
        "DK-NOTEXACT", (597,),
    ),
    "COC1OC2(OC)CC3CCC(O)C(C)C3(C)C(OC)C2=C1C": (
        "unknown organic compound",
        "DK-NOTEXACT", (600,),
    ),
    "C[C@@]12C=C[C@]3(C1)[C@@H](O)C[C@H]1[C@@](C)(CCC[C@@]1(C)C(=O)O)[C@@H]3CC2": (
        "(1S,4S,5S,9R,10S,12S,13S)-12-hydroxy-1,5,9-trimethyltetracyclo[11.2.1.0^4,13.0^5,10]hexadec-14-ene-9-carboxylic acid",
        "DK-VBFUSED", (601,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](CS)C(=O)O": (
        "aspartylcysteine",
        "DK-PEP", (603,),
    ),
    "[NH2+]=C(C[C@H](O)[C@H](O)CO)C(=O)[O-]": (
        "unknown organic compound",
        "DK-NOTEXACT", (608,),
    ),
    "CC1(C)CCC(=O)[C@@]2(C)O[C@]3(O)CC[C@@]12C[C@H]3O": (
        "(1R,3S,8R,10R)-1,10-dihydroxy-3,7,7-trimethyl-2-oxatricyclo[6.2.2.0^3,8]dodecan-4-one",
        "DK-VBFUSED", (611,),
    ),
    "N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": (
        "glutamylcysteinyltryptophan",
        "DK-PEP", (616,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "aspartyltryptophylhistidine",
        "DK-PEP", (624,),
    ),
    "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO)COC(=O)CCCCCCCCCCCCCCCCC": (
        "(2R)-1-(octadecanoyloxy)-2-[(9Z)-octadec-9-enoyloxy]propan-3-ol",
        "DK-GLYSUB", (633,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CCCNC(=N)N)[C@@H](C)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "arginylthreonylthreonylaspartic acid",
        "DK-PEP", (647,),
    ),
    "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](O)COC(=O)CCC/C=C\\C/C=C\\C/C=C\\CCCCCCCC": (
        "(2S)-2-hydroxy-3-[(5Z,8Z,11Z)-icosa-5,8,11-trienoyloxy]propyl (7Z,10Z,13Z,16Z)-docosa-7,10,13,16-tetraenoate",
        "DK-GLYM2", (648,),
    ),
    "CCC(C)CCCCCCCCCCCCC(=O)OC[C@H](O)COC(=O)CCCCCCCCCCC(C)CC": (
        "(2S)-2-hydroxy-3-[(12-methyltetradecanoyl)oxy]propyl 14-methylhexadecanoate",
        "DK-GLYM2", (649,),
    ),
    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "(2S)-2-(tryptophylglutamylamino)butanedioic acid",
        "DK-PEP", (654,),
    ),
    "CC(C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O": (
        "valylglutamylglutamic acid",
        "DK-PEP", (656,),
    ),
    "CC1=C[C@@H]2C(C)(C)[C@H]3CC[C@H](C)[C@@]23CC1": (
        "(1R,3R,6S,7R)-2,2,6,10-tetramethyltricyclo[5.4.0.0^3,7]undec-10-ene",
        "DK-VBNUM", (661,),
    ),
    # fix a performance pass (wp6-tests): 536 and 599 moved here from the re-baseline.
    "CCC1CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)O1": (
        "32-ethyl-2,5,11,14,20,23,29-heptamethyl-4,13,22,31,37,38,39,40-octaoxapentacyclo"
        "[32.2.1.1^7,10.1^16,19.1^25,28]tetracontane-3,12,21,30-tetrone",
        "DK-VBNUM", (536,),
    ),
    "CCC12C=C/C(C)=C\\C(C)(O)CCC(OC)C(C)C(O)C(C)C(O)C3OC(=CC3=O)CC(=O)OC(C1)C(C)C(=O)O2": (
        "(4Z)-1-ethyl-6,11,13-trihydroxy-9-methoxy-4,6,10,12,22-pentamethyl-20,24,26-"
        "trioxatricyclo[19.3.1.1^14,17]hexacosa-2,4,16-triene-15,19,23-trione",
        "DK-VBNUM", (599,),
    ),
    "CC(C)C[C@H](NC(=O)[C@H](C)N)C(=O)N[C@H](C(=O)N[C@@H](CCC(N)=O)C(=O)O)[C@@H](C)O": (
        "alanylleucylthreonylglutamine",
        "DK-PEP", (664,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "cysteinylisoleucylaspartic acid",
        "DK-PEP", (666,),
    ),
    "NCCCC[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](N)CC(=O)O)C(=O)O": (
        "aspartylasparaginyllysine",
        "DK-PEP", (669,),
    ),
    "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](O)COC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC": (
        "(2R)-2-hydroxy-3-[(11Z,14Z)-icosa-11,14-dienoyloxy]propyl (7Z,10Z,13Z,16Z,19Z)-docosa-7,10,13,16,19-pentaenoate",
        "DK-GLYM2", (672,),
    ),
    "CCCCOc1ccc(CC(=O)NO)cc1": (
        "N-hydroxyacetamide",
        "DK-NOTEXACT", (678,),
    ),
    "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCCCCCC": (
        "(2S)-2-hydroxy-3-(tetradecanoyloxy)propyl octadecanoate",
        "DK-GLYM2", (683,),
    ),
    "CC(C)[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)[C@@H](C)O)C(=O)O": (
        "threonylglutaminylvaline",
        "DK-PEP", (685,),
    ),
    "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O": (
        "aspartylhistidyltyrosine",
        "DK-PEP", (686,),
    ),
    "CC1(C)C(=O)CC(O)C23C(=O)OC4OCC(=CCC12)C43": (
        "2-hydroxy-5,5-dimethyl-11,13-dioxatetracyclo[7.5.1.0^1,6.0^12,15]pentadec-8-ene-4,14-dione",
        "DK-VBFUSED", (688,),
    ),
    "CC(=O)OCC(=O)[C@@]1(O)[C@@H](C)C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C": (
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-fluoro-11,17-dihydroxy-16-methyl-3,20-dioxopregna-1,4-dien-21-yl acetate",
        "DK-P101CIP", (691,),
    ),
    "C[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": (
        "glutamylalanyltryptophan",
        "DK-PEP", (700,),
    ),
}


# ---------------------------------------------------------------------------
# Task 12 fix a performance pass (wp6-tests; whole-branch review F12, t12-research canary-and-labels
# item 4). DK-NOTEXACT rows are NOT pinned by '==' any more: pinning required the engine to
# keep emitting a wrong-molecule or unparseable string (gate off) for the test to stay green.
# What ships is checked instead, gate ON (the tier contract: best-effort names the molecule
# RT-exact, and the PIN tier ships nothing that is not RT-exact), and a strict xfail per row
# asks for the raw producer to be fixed (it XPASSes then, forcing a re-baseline into the
# fixture). Rows whose PIN is derived get a strict xfail on the exact string.
# ---------------------------------------------------------------------------
_NOT_RT_EXACT = sorted(smi for smi, (_r, cls, _c) in CANARY_KNOWN_DEFECTS.items()
                       if cls == "DK-NOTEXACT")

_NOT_RT_EXACT_SHIP_DEFECTS = {
    # call 182 (tropisetron hydrochloride)
    "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl": (
        "DEFECT (label): the PIN tier (and best-effort) ships '(1R,3r,5S)-tropan-3-yl "
        "1H-indole-3-carboxylate—hydrogen chloride (1/1)' as pin_verified on a "
        "constitution-only check "
        "(gate_outcome self_consistency_constitution_only); OPSIN 2.9.0 cannot parse it (the "
        "'r'), so it is not RT-verifiable, and P-21.1.1.2 (BlueBookV2.md:7954) gives no PIN "
        "label to names including hydrogen chloride. .planning/TODO-2026-09-24.md 'Open "
        "from T12 fix round 2 (wp6)'"),
}

_LABEL_DEFECTS = {
    # calls 151/315/326 (a triglyceride). FIXED in suite fix j4 (TRIAGE g3 C10b):
    # rules.esters.name_polyol_polyester and the decomposition weave record their
    # acyloxy-on-hydride names as non-PIN, so the row ships best_effort; the value
    # None below turns its strict xfail into an ordinary assertion.
    "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC": None,
}

# (the reason the row carried while it was open, kept for the record:)
_LABEL_DEFECTS_HISTORY = {
    "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC": (
        "DEFECT (label): the gate-on PIN tier ships the RT-exact acyloxy-on-propane name "
        "'2-{[(11Z,14Z)-icosa-11,14-dienoyl]oxy}-1,3-bis{[(9Z,12Z)-octadeca-9,12-dienoyl]oxy}"
        "propane' at pin_verified; esters outrank the propane parent (P-41 :18158, 9 :18182), "
        "method (1) is the PIN (P-65.6.3.3.3.2 :31836) and OPSIN 2.9.0 cannot verify it -- the "
        "DK-GLYM2/GLYSUB class that wp5 demotes elsewhere. .planning/TODO-2026-09-24.md "
        "'Open from T12 fix round 2 (wp6)'"),
}

# SMILES -> the Blue-Book-derived PIN (OPSIN 2.9.0 full-InChIKey exact, fresh run outside
# the engine) for known-defect rows whose current name is RT-exact but not the PIN.
_KNOWN_DEFECT_PIN_TARGETS = {
    # 133, DK-INDH (reason above)
    "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O":
        "(3S)-6,7-dihydroxy-8-methoxy-3-methyl-1H-2-benzopyran-4(3H)-one",
    # 149, DK-ETHANAMINE: (b) (:2891) omits the locant; '(1)
    # N,N-diethylethanamine (PIN)' (:26235). The row predates the branch (Task 1 name,
    # CANARY.csv changed_since_task1 = no).
    "CCN(CC)Cc1ccccc1": "N-benzyl-N-ethylethanamine",
    # 536, 599, 661: DK-VBNUM (reason above). 661's string is the correct von Baeyer
    # spelling; whether its PIN is a fusion name awaits a ruling.
    "CCC1CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)O1":
        "5-ethyl-2,11,14,20,23,29,32-heptamethyl-4,13,22,31,37,38,39,40-octaoxapentacyclo"
        "[32.2.1.1^7,10.1^16,19.1^25,28]tetracontane-3,12,21,30-tetrone",
    "CCC12C=C/C(C)=C\\C(C)(O)CCC(OC)C(C)C(O)C(C)C(O)C3OC(=CC3=O)CC(=O)OC(C1)C(C)C(=O)O2":
        "(17Z)-21-ethyl-9,11,16-trihydroxy-13-methoxy-10,12,16,18,24-pentamethyl-2,22,26-"
        "trioxatricyclo[19.3.1.1^5,8]hexacosa-5,17,19-triene-3,7,23-trione",
    "CC1=C[C@@H]2C(C)(C)[C@H]3CC[C@H](C)[C@@]23CC1":
        "(1R,2S,5R,7R)-2,6,6,9-tetramethyltricyclo[5.4.0.0^1,5]undec-8-ene",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    pytest.param(smi, marks=pytest.mark.xfail(strict=True, reason=_NOT_RT_EXACT_SHIP_DEFECTS[smi]))
    if smi in _NOT_RT_EXACT_SHIP_DEFECTS else smi
    for smi in _NOT_RT_EXACT])
def test_canary_not_rt_exact_rows_ship_no_wrong_name(smiles):
    """Gate ON: a DK-NOTEXACT row never ships its wrong or unparseable raw name. The
    best-effort tier names it RT-exact (breadth), and whatever the PIN tier ships is a
    failure signal or RT-exact -- never pin_verified on a name OPSIN cannot verify."""
    from orthonym import Orthonym
    from tests.support.rt_assert import assert_tier_contract, name_is_rt_exact
    recorded = CANARY_KNOWN_DEFECTS[smiles][0]
    assert_tier_contract(smiles)
    r = Orthonym(style="pin").name_tiered(smiles)
    assert not (r["tier"] == "pin_verified" and r["name"] == recorded), r
    if r["tier"] == "pin_verified":
        assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.parametrize("smiles", [
    pytest.param(smi, marks=pytest.mark.xfail(strict=True, reason=(
        "DK-NOTEXACT, canary call(s) %s: the raw (gate-off) PIN-tier producer gives a name "
        "that is not OPSIN full-InChIKey exact (wrong molecule, unparseable, "
        "connectivity-only, or the whole sentinel); routed in TRIAGE.md 'Canary calls "
        "routed per call' / 'T4 outcome'. XPASS = fixed: re-verify and re-baseline the row."
        % (CANARY_KNOWN_DEFECTS[smi][2],))))
    for smi in _NOT_RT_EXACT])
def test_canary_not_rt_exact_raw_name_is_fixed(smiles):
    from tests.support.rt_assert import name_is_rt_exact
    assert name_is_rt_exact(name_compound(smiles), smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    pytest.param(smi, marks=pytest.mark.xfail(strict=True, reason=reason))
    if reason is not None else smi
    for smi, reason in _LABEL_DEFECTS.items()])
def test_canary_known_non_pin_is_not_labelled_pin(smiles):
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["tier"] != "pin_verified", r


@pytest.mark.parametrize("smiles,target", [
    pytest.param(smi, target, marks=pytest.mark.xfail(strict=True, reason=(
        "%s, canary call(s) %s: %s" % (CANARY_KNOWN_DEFECTS[smi][1], CANARY_KNOWN_DEFECTS[smi][2],
                                       _CANARY_DEFECT_REASONS[CANARY_KNOWN_DEFECTS[smi][1]]))))
    for smi, target in _KNOWN_DEFECT_PIN_TARGETS.items()])
def test_canary_known_defect_takes_its_pin(smiles, target):
    """Strict xfail on the exact Blue-Book-derived string: XPASS once the producer is
    fixed (then move the row out of CANARY_KNOWN_DEFECTS)."""
    assert name_compound(smiles) == target


class TestCanaryRegression:
    """The 703 canary compounds (tests/integration/test_canary_rt75.py) name as recorded.

    Re-baselined in Task 11 (2026-09-26): verified rows assert their fixture name; the rows in
    CANARY_KNOWN_DEFECTS assert their current (defective) name, so any change is flagged --
    except DK-NOTEXACT rows (Task 12 fix a performance pass), which are checked by the tests above and
    never pinned to a wrong or unparseable string.
    """

    @pytest.mark.unit
    def test_canary_compounds_no_regressions(self):
        """Verified rows keep their name; a known-defect row that changes is flagged too."""
        from tests.integration.test_canary_rt75 import CANARY_COMPOUNDS

        fixture_smiles = {smi for smi, _ in CANARY_COMPOUNDS}
        stale = sorted(set(CANARY_KNOWN_DEFECTS) - fixture_smiles)
        assert not stale, f"CANARY_KNOWN_DEFECTS entries not in the fixture: {stale}"
        assert {cls for _, cls, _ in CANARY_KNOWN_DEFECTS.values()} <= set(_CANARY_DEFECT_REASONS)

        failures = []
        for smi, expected_name in CANARY_COMPOUNDS:
            try:
                result = name_compound(smi)
                if smi in CANARY_KNOWN_DEFECTS:
                    recorded, cls, calls = CANARY_KNOWN_DEFECTS[smi]
                    if cls == "DK-NOTEXACT":
                        continue  # not RT-exact: see test_canary_not_rt_exact_rows_*
                    if result != recorded:
                        failures.append(
                            f"KNOWN-DEFECT ROW CHANGED ({cls}, canary call(s) {calls}): {smi}: "
                            f"recorded '{recorded}', got '{result}' -- re-verify it (OPSIN full "
                            f"InChIKey + Blue Book) and re-baseline it in the fixture")
                elif result is None:
                    failures.append(f"None for {smi}")
                elif result != expected_name:
                    failures.append(f"CHANGED: {smi}: expected '{expected_name}', got '{result}'")
            except Exception as e:
                failures.append(f"CRASH for {smi}: {e}")

        assert len(failures) == 0, (
            f"{len(failures)} canary regressions:\n" + "\n".join(failures[:10])
        )
