"""Wave-2 completion batch B1 — skeletal-replacement engine extensions.

Covers (all expected names OPSIN-RT verified against the evidence SMILES):
  * P-21.2.4.1/2  λ-convention on heterochains and homogeneous polysulfanes
                  (embedded SH2/SH4 are skeletal atoms, not thiols; λ numbering
                  tie-breaks: λ set low, then higher bonding number low).
  * P-15.4.3.2.2  substituent prefixes on the fixed heterochain numbering
                  (5,5-dimethyl-2,5λ4,8,11-tetrathiadodecane).
  * P-15.4.3.1    'a'-prefix citation in ELEMENT SENIORITY order
                  (3-phospha-2,5,7-trisilaoctane, 8-thia-2,4,6-trisiladecane).
  * P-51.4.1.3    -oic/-dioic acid suffixes on the fixed heterochain numbering
                  (3,6,9,12-tetraoxatetradecanedioic acid BB verbatim;
                  3,6,9,12-tetraoxapentadecan-15-oic acid P-59.2.2 — the acid
                  carbon takes the HIGH locant).
  * P-51.4.1.4    heteroatom chain terminators (2-oxa-4-thia-1,5-disilapentane).
  * P-51.4.1.1    strict >=4-heterounit gate for all NEW classes (fail-closed
                  witnesses stay unknown).
  * P-22.2.3      cyclic replacement: senior element takes the low locants on a
                  positional tie (Kryptofix-22 PIN).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name


def _skel(smiles):
    return try_skeletal_replacement_name(Chem.MolFromSmiles(smiles))


@pytest.mark.unit
class TestLambdaHeterochain:
    def test_tetrathiadodecane_parent_bb_verbatim(self):
        # P-21.2.4.1: the embedded SH2 (2 heavy neighbours) is a λ4 skeletal
        # atom, not a terminal thiol.
        assert _skel("CSCC[SH2]CCSCCSC") == "2,5lambda4,8,11-tetrathiadodecane"

    def test_lambda_numbering_low_locant(self):
        # λ4-S must take locant 5, not 8 (P-21.2.4.1 "lower locants to a
        # higher nonstandard bonding number"), independent of input order.
        rev = Chem.MolFromSmiles("CSCCSCC[SH2]CCSC")
        assert (try_skeletal_replacement_name(rev)
                == "2,5lambda4,8,11-tetrathiadodecane")

    def test_substituted_lambda_chain(self):
        # P-15.4.3.2.2 substituent prefixes on the fixed numbering.
        assert (_skel("CSCCS(C)(C)CCSCCSC")
                == "5,5-dimethyl-2,5lambda4,8,11-tetrathiadodecane")

    def test_genuine_terminal_thiol_still_blocked(self):
        # A real -SH terminus keeps the substitutive path (Gate 2 [SX2H]).
        assert _skel("SCCOCCOCCOCC") is None


@pytest.mark.unit
class TestCitationSeniorityOrder:
    def test_phospha_cited_before_sila(self):
        # BB verbatim P-51.4.1.2 — was '2,5,7-trisila-3-phosphaoctane'.
        assert _skel("C[SiH2]PC[SiH2]C[SiH2]C") == "3-phospha-2,5,7-trisilaoctane"

    def test_thia_cited_before_sila(self):
        # BB verbatim: seniority order even though sila holds locant 2.
        assert _skel("C[SiH2]C[SiH2]C[SiH2]CSCC") == "8-thia-2,4,6-trisiladecane"

    def test_oxa_thia_order_unchanged(self):
        # Established behaviour where seniority and locant order coincide.
        assert _skel("COCSCOC") == "2,6-dioxa-4-thiaheptane"


@pytest.mark.unit
class TestHeterochainAcidSuffix:
    def test_diacid_bb_verbatim(self):
        assert (name_compound("OC(=O)COCCOCCOCCOCC(=O)O")
                == "3,6,9,12-tetraoxatetradecanedioic acid")

    def test_monoacid_high_locant_p59_2_2(self):
        # The fixed heteroatom numbering owns the LOW locants; the acid
        # carbon takes 15 (P-51.4.1.2 / P-59.2.2).
        assert (name_compound("OC(=O)CCOCCOCCOCCOCC")
                == "3,6,9,12-tetraoxapentadecan-15-oic acid")

    def test_three_unit_diacid_fails_closed(self):
        # 11-atom homolog carries only 3 heterounits — P-51.4.1.1 requires 4;
        # the replacement name is NOT the PIN, so the engine must decline.
        assert _skel("OC(=O)COCCOCCOCC(=O)O") is None

    def test_plain_acids_untouched(self):
        # No heterochain: the ordinary acid namers keep ownership.
        assert _skel("CC(=O)O") is None
        assert _skel("OC(=O)CCC(=O)O") is None

    def test_acid_with_ene_fails_closed(self):
        # Suffix + ene composite numbering not built — never a lossy name.
        assert _skel("OC(=O)COCCOCCOCCOCC=C") is None


@pytest.mark.unit
class TestHeteroatomTerminators:
    def test_disilapentane_bb_verbatim(self):
        assert _skel("[SiH3]OCS[SiH3]") == "2-oxa-4-thia-1,5-disilapentane"

    def test_three_unit_si_chain_fails_closed(self):
        # Si-O-C-Si: 3 heterounits < 4 (P-51.4.1.1) — not the replacement PIN.
        assert _skel("[SiH3]OC[SiH3]") is None

    def test_two_unit_ether_ene_fails_closed(self):
        # C=CCOCCOC: 2 heterounits — '2,5-dioxaoct-7-ene' would be non-PIN.
        assert _skel("C=CCOCCOC") is None

    def test_branched_phosphane_fails_closed(self):
        # Branched ledger mis-transcription: its PIN is the substitutive
        # bis(methylsilyl)[(methylsilyl)methyl]phosphane, not a chain name.
        assert _skel("C[SiH2]P([SiH2]C)C[SiH2]C") is None

    def test_carbonless_siloxane_stays_catenated(self):
        # No chain carbon -> catenated-hydride territory (trisiloxane).
        assert _skel("[SiH3]O[SiH2]O[SiH3]") is None
        assert name_compound("[SiH3]O[SiH2]O[SiH3]") == "trisiloxane"


@pytest.mark.unit
class TestLambdaPolysulfane:
    def test_hexasulfane_bb_verbatim(self):
        # P-21.2.4.2: λ6 takes the lower locant over λ4 on the positional tie.
        assert name_compound("S[SH4]SS[SH2]S") == "2lambda6,5lambda4-hexasulfane"

    def test_lambda4_tetrasulfane_low_locant(self):
        assert name_compound("SS[SH2]S") == "2lambda4-tetrasulfane"

    def test_plain_polysulfanes_unchanged(self):
        assert name_compound("SSS") == "trisulfane"
        # Corrected 2026-07-30 (v29 Phase C Task 11): BB 39339 prints
        # `CH3-S-S-S-CH3 dimethyltrisulfane (PIN)` verbatim -- P-14.3.4.4
        # (BB 2953) omits the locants because trisulfane's middle sulfur bears no
        # hydrogen, so S1/S3 are the only placements and they are one orbit.
        assert name_compound("CSSSC") == "dimethyltrisulfane"


@pytest.mark.unit
class TestCyclicSeniorityNumbering:
    def test_kryptofix22_pin(self):
        # P-22.2.3: O (senior) takes the low locants on the positional tie;
        # oxa cited before aza. Was '1,10-diaza-4,7,13,16-tetraoxa...'.
        assert (_skel("C1COCCOCCNCCOCCOCCN1")
                == "1,4,10,13-tetraoxa-7,16-diazacyclooctadecane")

    def test_crown_ether_unchanged(self):
        assert (_skel("C1COCCOCCOCCOCCO1")
                == "1,4,7,10,13-pentaoxacyclopentadecane")


@pytest.mark.unit
class TestClassicPathsByteIdentical:
    @pytest.mark.parametrize("smiles,expected", [
        ("COCCOCCOC", "2,5,8-trioxanonane"),
        ("OCCOCCOCC", "3,6-dioxaoctan-1-ol"),
        ("C[SiH2]C[SiH2]C[SiH2]C[SiH2]C=C", "2,4,6,8-tetrasiladec-9-ene"),
        ("COCCOCCOCCOC", "2,5,8,11-tetraoxadodecane"),
    ])
    def test_established_names(self, smiles, expected):
        assert _skel(smiles) == expected

    @pytest.mark.parametrize("smiles", [
        "COCSC",     # carbon-parent PIN methoxy(methylsulfanyl)methane
        "COCCOC",    # 1,2-dimethoxyethane
        "CCNCCC",    # amine Gate 2c
        "CCCCOO",    # peroxol
    ])
    def test_blocked_classes_stay_blocked(self, smiles):
        assert _skel(smiles) is None
