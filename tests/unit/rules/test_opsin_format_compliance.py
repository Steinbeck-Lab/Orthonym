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
# Leads program 2026-10-09, item 43a (change-asserted-value): an acylated nitrogen that is not a
# suffix nitrogen of its amine parent is cited as the method-(1) acylamino prefix, not as an
# 'N-<acyl>' float 'Substituents of the types -NH-CO-R and -NH-SO2-R', the Blue Book
#:32991,:32995,:32998 "Method (1) generates preferred IUPAC names."). Canary call 460
# (DK-NACYL, the class is gone with it) and the two peptide rows 263 and 518 (DK-PEP) now give the
# substitutive names the classes asked for, each OPSIN 2.9.0 full-InChIKey exact (fresh java run
# outside the engine); they moved to the fixture (tests/support/canary_rt75_data.py, tag
# 'l6 RB-ACYLAMINO').

# PB lane S, S1 (2026-10-16, change-asserted-value): the 19 peptide rows 44, 65, 230, 242, 247, 264, 265, 273,
# 382, 383, 400, 442, 443, 459, 591, 647, 664, 666 and 685 (DK-PEP) now give the substitutive names the class
# asked for (prefix_claims.reconcile_prefix_claims: an atom is spelled once, a chain amide is 'amino' + 'oxo';
#, the Blue Book-32998;,:30373), each OPSIN 2.9.0 full-InChIKey exact (19/19, fresh
# java run outside the engine); they moved to the fixture (tests/support/canary_rt75_data.py, tag 'PB-S S1').

# PB lane S, S3 (2026-10-16): the two peptide rows 368 and 669 (DK-PEP; an N-terminal Asp) now give the 'carboxy' amido names
# (acyl_amido_polyacid;, the Blue Book), each OPSIN 2.9.0 full-InChIKey exact (fresh java run), tag 'PB-S S3'.

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
    "DK-BRIDGEDFUSED": (
        # re-baselined 2026-10-09 (change-asserted-value): these four rows (canary calls 39, 385,
        # 601, 611) used to be DK-VBFUSED von Baeyer names; fusion routing (likely sources
        # 7ea3918e4 all-carbon and cb3bf9850 with ring heteroatoms; not bisected) now names them
        # as bridged fused systems, as "Five-membered ring requirement" (the Blue Book) asks ('When fusion
        # names are not allowed, unsaturated von Baeyer ring system names are preferred IUPAC
        # names'). Each name is OPSIN 2.9.0 full-InChIKey exact. They stay pinned here, not in the
        # fixture, because the choice of the bridge excision, the ring numbering and the
        # hydro/indicated-hydrogen spelling of each, were not checked
        # against the Blue Book row by row.
        # DT4-B (2026-11-01, change-asserted-value): canary calls 22/318/325, 23/319/327, 154/320/333 and
        # 500 were DK-VBFUSED von Baeyer names; a fused ring system with a ring of eight or more members
        # now has its fusion name "Prefixes for attached components", the Blue Book
        # "There is no upper ring-size limit to this criterion"; "Heteromonocycles":11984
        # "Heteromonocycles having more than ten members... are used as parent components in preferred
        # IUPAC fusion names";:23831 fusion names of rings attached to a large ring are PINs).
        # Each name is OPSIN 2.9.0 full-InChIKey exact (fresh java run); the component choice and numbering
        # were not checked against the book row by row, so the rows are change detectors.
        "fused name, RT-exact; the PIN status of the bridge selection and numbering is "
        "unverified (P-25.4.3.4.2 / P-25.4.3.3), so the row is a change detector, not a fixture"
    ),
    "DK-LARGERINGFUSED": (
        # DT4-B (2026-11-01): fused names with a ring of eight or more members, the Blue Book;
        #,:23831), plain fusion names with no bridge prefix; OPSIN 2.9.0 full-InChIKey exact. Pinned here
        # as change detectors: their component choice and numbering were checked against the book's own large-ring
        # PINs (tests/unit/rules/test_fused_large_ring_pin.py), not row by row; call 22 is a three-component name.
        "fusion name with a large ring, numbering checked by read-back only"
    ),
    "DK-SMALLRINGFUSED": (
        # DT5 (2026-11-01, change-asserted-value): canary call 540 was a DK-VBFUSED von Baeyer name; a three-membered
        # ring is a second-order fusion component now "Fusion names composed of first- and higher-order
        # attached components", the Blue Book; "Five-membered ring requirement":23706 asks for two rings
        # of five or more members, the isoindole and the cycloundecene ring), so the row has its fusion name. OPSIN 2.9.0
        # and main read it back to the full InChIKey (fresh parse outside the engine). The component choice and numbering
        # were not checked against a book row, so the row is a change detector.
        "fusion name with a three-membered second-order component, numbering checked by read-back only"
    ),
    "DK-STEROIDYL": (
        # DT5 (2026-11-01, change-asserted-value): canary call 522 was a DK-P101CIP stereoparent name INTRODUCTION,
        # the Blue Book "PINs are not identified for the compounds in this Chapter"); the heptane-2,3-diol parent
        # (two hydroxy groups,:18875) now has its steroid '-yl' prefix, whose own hydroxy group is the prefix
        # 'hydroxy':15800). OPSIN 2.9.0 and main read it back to the full InChIKey; the component choice and numbering
        # were not checked against a book row, so the row is a change detector.
        "systematic name with a decorated fused '-yl' prefix, numbering checked by read-back only"
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
        # d6f633508 (2026-10-02) re-numbered the secondary-bridge superscripts of these cages
        #, the Blue Book "The superscript locants for the secondary bridges
        # must be as low as possible when considered as a set in ascending numerical order"):
        # e.g. 0^4,8 -> 0^1,5; the recorded names below follow it (all OPSIN-exact).
        "von Baeyer name for a system with ortho-fused rings of 5+ members; the "
        "(hydro/bridged) fusion name is the PIN (P-52.2.4.1 :23710; P-44.2.2.2 :19532 "
        "fused > bridged fused > von Baeyer); R22 class, large-polycycle Task 6a"
    ),
}

# Suite fix j6-breadth (2026-09-27): canary calls 135/142/323 (the prenylated-phenol trienoic
# acid) and 171 (the tetramethyl-1,3-dioxolanyl terpene) left this table -- their raw PIN-tier
# names are RT-exact now (strict XPASS below) and they are verified fixture rows
# (tests/support/canary_rt75_data.py, tags 'j6 RB-RINGBR' / 'j6 RB-HWSTEM').
# Breadth job 3 review fixes (2026-09-29): canary calls 155 (DK-NOTEXACT, the fused
# ring ketone that raised TypeError on the '4a'/7 locant sort, fixed by 2c79719e3) and
# 684 (DK-SFXNUM, fixed by 7c39f2243: every ring -OH of the alcohol class is a suffix)
# left this table: both raw PIN-tier names are RT-exact now and are verified fixture
# rows (tests/support/canary_rt75_data.py, tags 'b3 RB-...').
# Bridged fused S0 follow-up (2026-10-01): canary call 175 (DK-NOTEXACT, the 'cyclohexa[b]pyridine'
# acid with no hydro prefixes) left this table: the hydro prefixes now go on the mancude parent
#, the Blue Book), its raw PIN-tier name '4,7,8-trihydroxy-7,8-dihydroquinoline-
# 2-carboxylic acid' is RT-exact (OPSIN 2.9.0 full key and FixedH InChI) and is a verified fixture
# row (tests/support/canary_rt75_data.py, tag 'bf RB-HYDRO').
# N5b (2026-10-08) change-asserted-value: canary calls 180, 181 (the stereo-free ergostadienol and
# stigmastadienediol) and 353 (the pentacyclic triterpenoid acid) left this table: a carbon-only
# ortho-fused system with two rings of five or more members is named by fusion nomenclature, not
# von Baeyer "Five-membered ring requirement", the Blue Book-23710), so their raw
# PIN-tier names are real names now (previously the sentinel 'unknown organic compound'); each is
# OPSIN 2.9.0 full-InChIKey exact (fresh java run outside the engine) and re-baselined in
# tests/support/canary_rt75_data.py (tag 'n5b RB-FUSION').
# SMILES -> (current PIN-tier name with the gate off, class, canary call numbers).
CANARY_KNOWN_DEFECTS = {
    "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O": (
        "2-(acetyloxy)heptadec-16-yne-1,4-diol",
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
        "(1Z,4S,6R,7Z,10aR,13S,13aS,14S,14aR,15aS,15bS)-13-benzyl-6-hydroxy-4,6,14,14a-tetramethyl-13,13a,14,14a,15a,15b-hexahydro-3H-[1]oxacyclododecino[2,3-d]oxireno[2,3-f]isoindole-5,9,11(4H,6H,12H)-trione",
        "DK-LARGERINGFUSED", (22, 318, 325),
    ),
    "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C": (
        "(3E,5S,9E,10aS,13S,13aS,14S,16aR)-5-hydroxy-9,12,13-trimethyl-14-(2-methylpropyl)-6,7,8,10a,13,13a,14,15-octahydro-2H-[1]oxacyclododecino[2,3-d]isoindole-2,16(5H)-dione",
        "DK-LARGERINGFUSED", (23, 319, 327),
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
        "(3S,3aS,6R,7R,9aS)-1,1,7-trimethyldecahydro-3a,7-methanocyclopenta[8]annulene-3,6-diol",
        "DK-BRIDGEDFUSED", (39,),
    ),
    "C[C@H]1C(=O)O[C@@H]2CCN3CC=C(COC(=O)[C@](C)(O)[C@]1(C)O)[C@H]23": (
        "(1R,4R,5R,6R,16R)-5,6-dihydroxy-4,5,6-trimethyl-2,8-dioxa-13-azatricyclo[8.5.1.0^13,16]hexadec-10-ene-3,7-dione",
        "DK-VBFUSED", (48,),
    ),
    "NC(=O)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O": (
        "asparaginylaspartylaspartic acid",
        "DK-PEP", (50,),
    ),
    "CC1=C(O)C(=O)[C@]2(O)C[C@H]3C[C@](C)(C(=O)O)C[C@H]3[C@]12C": (
        "unknown organic compound",
        "DK-NOTEXACT", (68,),
    ),
    # (canary call 133, DK-INDH, left this table in LO3: the PIN builder's name
    # '(3S)-6,7-dihydroxy-8-methoxy-3-methyl-1H-2-benzopyran-4(3H)-one' is offered in place of the
    # fused namer's hydro-prefix name, which that namer records as not a PIN, and it is the
    # strict-xfail target the row carried; the fixture holds it now)
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
        "(7E)-12-hydroxy-13-methoxy-4,5,8-trimethyl-3-(2-methylpropyl)-3,3a,4,6a,9,10,11,12,13,14-decahydro-1H-cycloundeca[d]isoindole-1,15(2H)-dione",
        "DK-LARGERINGFUSED", (154, 320, 333),
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
    "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl": (
        "(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate—hydrogen chloride (1/1)",
        "DK-NOTEXACT", (182,),
    ),
    "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12": (
        "unknown organic compound",
        "DK-NOTEXACT", (202, 316, 338),
    ),
    "N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": (
        "aspartylglycyltryptophan",
        "DK-PEP", (239,),
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
    "C[C@@H]1CC[C@@H]2C=C(C(=O)O)[C@H]3C[C@](C)(C(=O)O)C[C@]132": (
        "(1R,3S,5S,8R,11R)-3,11-dimethyltricyclo[6.3.0.0^1,5]undec-6-ene-3,6-dicarboxylic acid",
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
    "NC(=O)CC[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "glutamylglutaminylhistidine",
        "DK-PEP", (379,),
    ),
    "CC1=C2C(=O)C[C@@]2(C)[C@@H]2C[C@@H]3[C@H](O)C[C@@H](C)[C@@]2(CC1)C3(C)C": (
        "(5aR,6R,8R,9S,10aS,10bS)-8-hydroxy-3,6,10b,11,11-pentamethyl-1,4,5,6,7,8,9,10,10a,10b-decahydro-2H-5a,9-methanocyclobuta[a]heptalen-2-one",
        "DK-BRIDGEDFUSED", (385,),
    ),
    "NC(=O)C[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O": (
        "glutamylasparaginylphenylalanine",
        "DK-PEP", (406,),
    ),
    "C[C@@H]1CC[C@H]2C(C=O)=C[C@@H]3CC(C)(C)CC132": (
        "(2R,5R,8S)-2,10,10-trimethyltricyclo[6.3.0.0^1,5]undec-6-ene-6-carbaldehyde",
        "DK-VBFUSED", (413,),
    ),
    "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylglutamylisoleucine",
        "DK-PEP", (419,),
    ),
    "C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O": (
        "glycylaspartylalanine",
        "DK-PEP", (428,),
    ),
    "CC1CCC2C(C(=O)O)=CC3CC(C)(C)CC132": (
        "2,10,10-trimethyltricyclo[6.3.0.0^1,5]undec-6-ene-6-carboxylic acid",
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
    "NCCCC[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O": (
        "lysylglutamylglutamic acid",
        "DK-PEP", (454,),
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
        "(3S,3aR,4S,6aS,13aS)-4,5,8-trimethyl-3-(2-methylpropyl)-3,3a,4,6a,7,10-hexahydro-1H-cyclonona[d]isoindole-1,11,13(2H,12H)-trione",
        "DK-LARGERINGFUSED", (500,),
    ),
    "N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O": (
        "glutamylcysteinylhistidine",
        "DK-PEP", (501,),
    ),
    "C[C@H](CC[C@H](O)C(C)(C)O)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3": (
        "(3S,6R)-6-[(3S,5R,10S,13R,14R,17R)-3-hydroxy-4,4,10,13,14-pentamethyl-2,3,4,5,6,7,10,11,12,13,14,15,16,17-tetradecahydro-1H-cyclopenta[a]phenanthren-17-yl]-2-methylheptane-2,3-diol",
        "DK-STEROIDYL", (522,),
    ),
    "CC(C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O": (
        "glutaminylglutamylvaline",
        "DK-PEP", (524,),
    ),
    "NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O": (
        # (the Blue Book) encloses the compound acyl group; (:54480);
        # OPSIN 2.9.0 full-InChIKey exact
        "N-(glycylglutamyl)glycine",
        "DK-PEP", (538,),
    ),
    "CC1=CC2/C=C(\\C)CCC3OC3/C=C/C(=O)C23C(=O)NC(CC(C)C)C3C1C": (
        "(7E,12E)-4,5,8-trimethyl-3-(2-methylpropyl)-3,3a,4,6a,9,10,10a,11a-octahydro-1H-oxireno[2',3':7,8]cycloundeca[1,2-d]isoindole-1,14(2H)-dione",
        "DK-SMALLRINGFUSED", (540,),
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
        "4-(acetyloxy)heptadec-16-yne-1,2-diol",
        "DK-NOTEXACT", (567,),
    ),
    "C[C@@H](O)[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)CCC(=O)O)C(=O)O": (
        "glutamyllysylthreonine",
        "DK-PEP", (572,),
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
        "(4R,4aS,6S,6aS,9S,11aS,11bS)-6-hydroxy-4,9,11b-trimethyl-1,2,3,4,4a,5,6,9,10,11,11a,11b-dodecahydro-6a,9-methanocyclohepta[a]naphthalene-4-carboxylic acid",
        "DK-BRIDGEDFUSED", (601,),
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
        "(2R,3R,4aR,8aS)-2,3-dihydroxy-5,5,8a-trimethyltetrahydro-2H,5H-2,4a-ethano-1-benzopyran-8(8aH)-one",
        "DK-BRIDGEDFUSED", (611,),
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
    # 661 left (quick-wins F-Q1: the lowest superscripts,:9685, give the
    # fixture's '(1R,2S,5R,7R)-2,6,6,9-tetramethyltricyclo[5.4.0.0^1,5]undec-8-ene').
    # fix a performance pass (wp6-tests): 536 and 599 moved here from the re-baseline.
    # 536 left (large-polycycle fix a performance pass): the cage numbering ranks (g) (:3307, "lowest
    # locants for the substituent cited first as a prefix in the name"): the ethyl is cited first and
    # takes locant 5 where it took 32, the fixture's name (both read back exactly; the locant sets of
    # the prefixes tie, {2,5,11,14,20,23,29,32}).
    # 599 left (quick-wins F-Q1, as 661 above): the fixture's '...trioxatricyclo[19.3.1.1^5,8]...'.
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
    # 133, DK-INDH: fixed in LO3 (rules/fused_pin_offer.py); the target is the fixture's name now
    # 149, DK-ETHANAMINE: (b) (:2891) omits the locant; '(1)
    # N,N-diethylethanamine (PIN)' (:26235). The row predates the branch (Task 1 name,
    # CANARY.csv changed_since_task1 = no).
    "CCN(CC)Cc1ccccc1": "N-benzyl-N-ethylethanamine",
    # 536, 599 and 661 left the table (quick-wins F-Q1; large-polycycle fix a performance pass).
}


# One strict-xfail row takes 55 s: it runs in the nightly run only.
_SLOW_NOT_RT_EXACT = {"CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl"}


def _not_rt_exact_param(smi):
    marks = []
    if smi in _NOT_RT_EXACT_SHIP_DEFECTS:
        marks.append(pytest.mark.xfail(strict=True, reason=_NOT_RT_EXACT_SHIP_DEFECTS[smi]))
    if smi in _SLOW_NOT_RT_EXACT:
        marks.append(pytest.mark.slow)
    return pytest.param(smi, marks=marks) if marks else smi


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [_not_rt_exact_param(smi) for smi in _NOT_RT_EXACT])
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


#: DK-NOTEXACT rows whose raw name is OPSIN full-InChIKey exact since the zero-bridge reading of the
#: bridged fused builder spells a ring system with heteroatoms (N5b step 2,,
#: the Blue Book): the xfail is lifted for them. Re-baselined: 'COC1OC2(OC)CC3CCC(O)C(C)C3(C)
#: C(OC)C2=C1C' is '2,4,9a-trimethoxy-3,4a,5-trimethyl-2,4,4a,5,6,7,8,8a,9,9a-decahydronaphtho[2,3-b]
#: furan-6-ol' (canary call 600).
#: L5 (the fixall round, 2026-10-09): the two heptadecyne acetates of a diol (canary calls 6 and 567)
#: cited the secondary OH twice, as 'hydroxy' and inside '-1,4-diol' / '-1,2-diol' (a gem-diol to OPSIN),
#: because the ester-demotion path of name_polyfunctional re-selected the principal group without
#: dropping its same-class alcohol subtypes from the prefix loop. (the Blue Book): "The
#: senior parent structure has the maximum number of substituents corresponding to the principal
#: characteristic group (suffix)". The raw names are now '2-(acetyloxy)heptadec-16-yne-1,4-diol' and
#: '4-(acetyloxy)heptadec-16-yne-1,2-diol', OPSIN 2.9.0 full-InChIKey exact (fresh java run outside the
#: engine); valid but not the PIN (the PIN is the functional-class ester,:18158 class 9 over class
#: 17, which name_substituent cannot spell for an unsaturated hydroxylated alkyl).
#: HXL: bufexamac (canary call 678), a hydroxamic acid, is named by the fragment assembly now
#: "Hydroxamic acids", the Blue Book; the substituent on the N is a prefix,,:3436): the raw name is
#: '2-(4-butoxyphenyl)-N-hydroxyacetamide' (was 'N-hydroxyacetamide', the carrier dropped), OPSIN 2.9.0
#: full-InChIKey exact and isomeric-SMILES exact (fresh java run outside the engine).
#: DT4 F1 (2026-10-31, change-asserted-value): canary calls 68 and 440 (DK-NOTEXACT). A ring =O (or =S, =NH) that is
#: not the principal group is the prefix 'oxo' ('sulfanylidene', 'imino') of the fused parent hydride
#: "Ketones", the Blue Book, '9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid (PIN)':29471;
#::24862), which the fused builder did not spell: the raw names are now '(2S,3aR,3bR,6aS,7aR)-5,6a-
#: dihydroxy-2,3b,4-trimethyl-6-oxo-2,3,3a,3b,6,6a,7,7a-octahydro-1H-cyclopenta[a]pentalene-2-carboxylic acid'
#: and '(3aR,5aR,10aR)-3a,5a-dimethyl-6-oxo-1-(propan-2-yl)-2,3,3a,4,5,5a,6,7,10,10a-decahydrocyclohepta[e]indene-
#: 8-carbaldehyde'; the locants of the second were checked atom by atom against the input, and both are OPSIN
#: 2.9.0 full-InChIKey exact (``test_canary_not_rt_exact_raw_name_is_fixed`` reads each back with a fresh call
#: outside the engine on every run; with the ylidene prefixes off, ``test_bf_f1_ylidene_prefixes.py``, they
#: decline again).
_NOW_RT_EXACT = frozenset({
    "CCCCOc1ccc(CC(=O)NO)cc1",
    "COC1OC2(OC)CC3CCC(O)C(C)C3(C)C(OC)C2=C1C",
    "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O",
    "C#CCCCCCCCCCCCC(CC(O)CO)OC(C)=O",
    "CC1=C(O)C(=O)[C@]2(O)C[C@H]3C[C@](C)(C(=O)O)C[C@H]3[C@]12C",
    "CC(C)C1=C2[C@H]3CC=C(C=O)CC(=O)[C@]3(C)CC[C@@]2(C)CC1",
    # DT4-B: an ortho-fused system with an 11-membered ring is a fusion name now ('...tetradecahydrocycloundeca[e]
    # indene', the Blue Book "no upper ring-size limit";:23831); OPSIN-exact.
    "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12",
    # DT5 (2026-11-01, change-asserted-value): the diarylheptanoid names on its phenol ring now, '4-[(6E)-3-hydroxy-7-
    # phenylhept-6-en-1-yl]phenol': the ring bearing the most hydroxy groups is the parent, the Blue Book;
    # one each, so a ring is senior to a chain,:19336), and the writer takes the chain with its ring
    # (``ring_nitrogen_yl.ring_yl_branch_in_scope``). OPSIN 2.9.0 and main read it back to the full InChIKey.
    "Oc1ccc(CCC(O)CC/C=C/c2ccccc2)cc1",
})


@pytest.mark.parametrize("smiles", [
    smi if smi in _NOW_RT_EXACT else
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
    """The 703 canary compounds (tests/support/canary_rt75_data.py) name as recorded.

    Re-baselined in Task 11 (2026-09-26): verified rows assert their fixture name; the rows in
    CANARY_KNOWN_DEFECTS assert their current (defective) name, so any change is flagged --
    except DK-NOTEXACT rows (Task 12 fix a performance pass), which are checked by the tests above and
    never pinned to a wrong or unparseable string.
    """

    @pytest.mark.unit
    def test_canary_compounds_no_regressions(self):
        """Verified rows keep their name; a known-defect row that changes is flagged too."""
        from tests.support.canary_rt75_data import CANARY_COMPOUNDS

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
