"""Tests for W8-P4: P-65.7 (organic-acid anhydrides — thio/seleno/peroxy/
mixed/poly) and P-67 (mono-/poly-nuclear noncarbon oxoacid) coverage.

Two families of assertion:
  1. Tractable BUILDS (real names, gated + raw): mixed cyanic anhydride
     (P-65.7.2), cyclic thio-anhydride dione (P-65.7.7.3), chalcogen
     di-anhydride bis(thioanhydride) (P-65.7.6.4.1).
  2. Atom-drop SAFETY-FLOOR vetoes (P4-1): structural motifs the RAW (no-Java,
     gate-OFF) namer used to silently drop atoms or mis-perceive the
     constitution for — verified fail-closed (never the old wrong string)
     under BOTH the gated AND the raw namer.
"""
import pytest

from orthonym.namer import Orthonym, name_compound

RAW = Orthonym(_disable_opsin_validity_gate=True)


@pytest.mark.unit
class TestMixedCyanicAnhydride:
    """P-65.7.2 / P-65.2.2: BB 30979 'CH3-CO-O-CN acetic cyanic anhydride (PIN)'."""

    def test_acetic_cyanic_anhydride_gated(self):
        assert name_compound("CC(=O)OC#N") == "acetic cyanic anhydride"

    def test_acetic_cyanic_anhydride_raw(self):
        assert RAW.name("CC(=O)OC#N") == "acetic cyanic anhydride"


@pytest.mark.unit
class TestCyclicThioAnhydrideDione:
    """P-65.7.7.3 method 1 (PIN): thio analogue of phthalic anhydride.

    BB 32546 verbatim gives the SATURATED benzo form 'hexahydro-2-
    benzothiophene-1,3-dione (PIN)'; the mancude/aromatic form here drops
    'hexahydro' (direct BB-sanctioned extension, OPSIN-RT confirmed).
    """

    def test_benzothiophene_dione_gated(self):
        assert name_compound("O=C1SC(=O)c2ccccc12") == "2-benzothiophene-1,3-dione"

    def test_benzothiophene_dione_raw(self):
        assert RAW.name("O=C1SC(=O)c2ccccc12") == "2-benzothiophene-1,3-dione"

    def test_acyclic_thioanhydride_still_correct(self):
        # Regression guard: the cyclic-guard added alongside this build must
        # not touch the pre-existing ACYCLIC chalcogen-anhydride path.
        assert name_compound("c1ccccc1C(=O)SC(=O)c1ccccc1") == "benzoic thioanhydride"
        assert name_compound("CC(=O)[Se]C(C)=O") == "acetic selenoanhydride"
        assert name_compound("CC(=O)OOC(C)=O") == "acetic peroxyanhydride"


@pytest.mark.unit
class TestChalcogenDianhydride:
    """P-65.7.6.4.1: BB 32446 'diacetic butanedioic bis(thioanhydride) (PIN)'."""

    def test_bis_thioanhydride_gated(self):
        result = name_compound("CC(=O)SC(=O)CCC(=O)SC(C)=O")
        assert result == "diacetic butanedioic bis(thioanhydride)"

    def test_bis_thioanhydride_raw(self):
        result = RAW.name("CC(=O)SC(=O)CCC(=O)SC(C)=O")
        assert result == "diacetic butanedioic bis(thioanhydride)"


@pytest.mark.unit
class TestLinearPolyanhydrideVerify:
    """P-65.7.6.2: the general-nomenclature method-2 name IS BB-verbatim for
    this exact structure (BB 32410-32412), so it must NOT be vetoed (it
    describes the correct molecule; only the substitutive method-1 PIN is
    unbuilt this cycle)."""

    def test_trianhydride_ships_bb_verbatim_general_name(self):
        smi = "CC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)CC"
        assert name_compound(smi) == "acetic dibutanedioic propanoic trianhydride"


@pytest.mark.unit
class TestAtomDropSafetyFloor:
    """P4-1: verified WRONG/DROP leaks in the RAW (gate-OFF) path must ship
    NEITHER the old wrong string NOR any new wrong string -- fail-closed
    ('unknown organic compound') under both the raw and the gated namer."""

    @pytest.mark.parametrize("smi,old_wrong", [
        ("CC(=O)OS", "ethane"),                                    # P-65.1.5.3 thioperoxy acid, drops S+O2
        ("O=[N+]([O-])NCC(=O)O", "ethanoic acid"),                 # P-67.1.4.3.2 nitramido, drops -NH-NO2
        ("OC(=O)CCOP(=O)(O)OP(=O)(O)O", "propanoic acid"),         # P-67.2.6 diphosphate substituent, drops diphosphate
        ("CC(=O)OC(=O)OC(=O)O", "1-(propanoyloxy)methanoic acid"),  # P-67.3.1 wrong constitution
        ("COS(=O)(=O)OS(=O)(=O)SCC", "1-methanoic anhydridylethanoic anhydride"),  # P-67.2.5.2 partial ester
    ])
    def test_leak_fails_closed_raw(self, smi, old_wrong):
        result = RAW.name(smi)
        assert result != old_wrong
        assert result is None or result.startswith("unknown")

    @pytest.mark.parametrize("smi", [
        "CC(=O)OS",
        "O=[N+]([O-])NCC(=O)O",
        "OC(=O)CCOP(=O)(O)OP(=O)(O)O",
        "CC(=O)OC(=O)OC(=O)O",
        "COS(=O)(=O)OS(=O)(=O)SCC",
    ])
    def test_leak_fails_closed_gated(self, smi):
        result = name_compound(smi)
        assert result is None or result.startswith("unknown")

    def test_gated_matches_raw_for_gated_vetoes(self):
        """Every veto must fire identically gated vs raw (source-level, not
        an OPSIN-dependent suppression)."""
        for smi in [
            "CC(=O)OS", "O=[N+]([O-])NCC(=O)O",
            "OC(=O)CCOP(=O)(O)OP(=O)(O)O", "CC(=O)OC(=O)OC(=O)O",
            "COS(=O)(=O)OS(=O)(=O)SCC",
        ]:
            assert RAW.name(smi) == name_compound(smi)


@pytest.mark.unit
class TestAtomDropVetoFalsePositiveProtection:
    """The P4-1 veto is gated on class_id == GENERAL + tight structural
    motifs; it must NEVER fire on molecules a dedicated handler already
    correctly owns."""

    def test_nucleoside_diphosphate_unaffected(self):
        # ADP -- claimed by the NUCLEOSIDE class before ever reaching GENERAL.
        smi = ("Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)"
               "[C@@H](O)[C@H]1O")
        assert name_compound(smi) == "adenosine 5'-(trihydrogen diphosphate)"

    def test_benzenesulfonic_anhydride_unaffected(self):
        smi = "O=S(=O)(OS(=O)(=O)c1ccccc1)c1ccccc1"
        assert name_compound(smi) == "benzenesulfonic anhydride"

    def test_cyclic_carbonate_unaffected(self):
        # A cyclic 'naked carbonate' carbon (1,3-dioxan-2-one) is ring-excluded
        # from the naked-carbonate motif -- must keep its ring name.
        assert name_compound("O=C1OCCCO1") == "1,3-dioxan-2-one"

    def test_free_carbonic_acid_family_unaffected(self):
        assert name_compound("OC(=O)O") == "carbonic acid"
        assert name_compound("OC(=O)OC(=O)O") == "dicarbonic acid"

    def test_diphosphoric_acid_family_unaffected(self):
        assert name_compound("OP(=O)(O)OP(=O)(O)O") == "diphosphoric acid"
        assert name_compound("O=P(O)(O)P(=O)(O)O") == "hypodiphosphoric acid"
