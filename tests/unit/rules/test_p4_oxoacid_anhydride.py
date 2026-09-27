"""Tests for W8-P4: (organic-acid anhydrides — thio/seleno/peroxy/
mixed/poly) and (mono-/poly-nuclear noncarbon oxoacid) coverage.

Two families of assertion:
  1. Tractable BUILDS (real names, gated + raw): mixed cyanic anhydride
     , cyclic thio-anhydride dione, chalcogen
     di-anhydride bis(thioanhydride).
  2. Atom-drop SAFETY-FLOOR vetoes (P4-1): structural motifs the RAW (no-Java,
     gate-OFF) namer used to silently drop atoms or mis-perceive the
     constitution for — verified fail-closed (never the old wrong string)
     under BOTH the gated AND the raw namer.
"""
import pytest

from orthonym.namer import Orthonym, name_compound
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check

RAW = Orthonym(_disable_opsin_validity_gate=True)


@pytest.mark.unit
class TestMixedCyanicAnhydride:
    """ /: BB 30979 'CH3-CO-O-CN acetic cyanic anhydride (PIN)'."""

    def test_acetic_cyanic_anhydride_gated(self):
        assert name_compound("CC(=O)OC#N") == "acetic cyanic anhydride"

    def test_acetic_cyanic_anhydride_raw(self):
        assert RAW.name("CC(=O)OC#N") == "acetic cyanic anhydride"


@pytest.mark.unit
class TestCyclicThioAnhydrideDione:
    """ method 1 (PIN): thio analogue of phthalic anhydride.

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
    """: BB 32446 'diacetic butanedioic bis(thioanhydride) (PIN)'."""

    def test_bis_thioanhydride_gated(self):
        result = name_compound("CC(=O)SC(=O)CCC(=O)SC(C)=O")
        assert result == "diacetic butanedioic bis(thioanhydride)"

    def test_bis_thioanhydride_raw(self):
        result = RAW.name("CC(=O)SC(=O)CCC(=O)SC(C)=O")
        assert result == "diacetic butanedioic bis(thioanhydride)"


@pytest.mark.unit
class TestLinearPolyanhydrideVerify:
    """: the general-nomenclature method-2 name IS BB-verbatim for
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
    ('unknown organic compound') under both the raw and the gated namer.

    NOTE (2026-09-19): two original floor rows became obsolete because the
    engine's behaviour on them changed -- the change PREDATES v52 (measured
    identical at pre-v52 cb2f3807f, Phase-2-end 35bd65ec9 and HEAD; both rows
    were still fail-closed at the floor's birth a469b6b80). They are handled in
    dedicated methods below and removed from the source-level veto lists:
      * O=[N+]([O-])NCC(=O)O -> now correctly named '(carboxymethyl)nitramide'
        (RT-verified, no atom drop) -> see test_nitramido_acetic_acid_named.
      * OC(=O)CCOP(=O)(O)OP(=O)(O)O -> GATED namer abstains (0-wrong holds);
        only the RAW gate-OFF path emits a benign mixture-split the gate rejects
        -> see test_carboxypropyl_diphosphate_gated_abstains.
    The three genuine source-level fail-closed rows remain below.
    """

    @pytest.mark.parametrize("smi,old_wrong", [
        ("CC(=O)OS", "ethane"),                                    # thioperoxy acid, drops S+O2
        ("CC(=O)OC(=O)OC(=O)O", "1-(propanoyloxy)methanoic acid"),  # wrong constitution
        ("COS(=O)(=O)OS(=O)(=O)SCC", "1-methanoic anhydridylethanoic anhydride"),  # partial ester
    ])
    def test_leak_fails_closed_raw(self, smi, old_wrong):
        result = RAW.name(smi)
        assert result != old_wrong
        assert result is None or result.startswith("unknown")

    @pytest.mark.parametrize("smi", [
        # These fail closed at the SOURCE level (the producer itself returns
        # None/unknown), so they hold under the suite-default gate-OFF too.
        # B2 (OC(=O)CCOP...) is NOT here: it only fails closed with the gate ON
        # -- see the gate-marked test_carboxypropyl_diphosphate_gated_abstains.
        "CC(=O)OS",
        "CC(=O)OC(=O)OC(=O)O",
        "COS(=O)(=O)OS(=O)(=O)SCC",
    ])
    def test_leak_fails_closed_gated(self, smi):
        result = name_compound(smi)
        assert result is None or result.startswith("unknown")

    def test_gated_matches_raw_for_gated_vetoes(self):
        """Every veto in THIS list must fire identically gated vs raw
        (source-level, not an OPSIN-dependent suppression). B2 is excluded: its
        gated abstention is OPSIN-dependent (rejects the raw mixture-
        split), so raw != gated by design -- see the dedicated B2 method."""
        for smi in [
            "CC(=O)OS", "CC(=O)OC(=O)OC(=O)O",
            "COS(=O)(=O)OS(=O)(=O)SCC",
        ]:
            assert RAW.name(smi) == name_compound(smi)

    @pytest.mark.opsin_gate
    def test_nitramido_acetic_acid_named(self):
        """B1 (was a fail-closed floor row, raw+gated): the substitutive
        nitramide producer now names O2N-NH-CH2-COOH in FULL. The floor no
        longer applies -- it existed for when this could ONLY leak the atom-
        dropped 'ethanoic acid'. The producer change predates v52 (routing was
        already live by; the N- locant was dropped at caa7fd7b1 under
        , giving the current bare '(carboxymethyl)nitramide').

        Runs under the PRODUCTION gate (opsin_gate): the gate ACCEPTS this valid
        name, so production ships it. Load-bearing safety property: RT-match
        proves the name denotes the EXACT input structure -- no atom drop. The
        exact string is the current gate-approved emission (also emitted raw)."""
        smi = "O=[N+]([O-])NCC(=O)O"
        result = name_compound(smi)
        assert result != "ethanoic acid"          # never the old atom-dropped leak
        assert result == "(carboxymethyl)nitramide"
        assert RAW.name(smi) == "(carboxymethyl)nitramide"
        rt = opsin_roundtrip_check(smi, result)
        assert rt["passed"], f"round-trip failed (atom-drop guard): {rt}"

    @pytest.mark.opsin_gate
    def test_carboxypropyl_diphosphate_gated_abstains(self):
        """B2 (was a fail-closed floor row, raw+gated): under the PRODUCTION
        gate (opsin_gate) the namer correctly ABSTAINS -- 0-wrong holds, the
        load-bearing guard. (The suite default is gate-OFF, under which
        name_compound == raw and would ship the split; hence the marker.)

        The RAW gate-OFF namer used to emit a mixture-split
        ('3-hydroxypropanoic acid diphosphoric acid') for this CONNECTED input;
        OPSIN parses that name to a DISCONNECTED 2-fragment structure, so the
        production gate's SELF-01 rejected it as a different molecule. Since the
        pre-existing-failures plan the raw namer fails closed too (TRIAGE 'Suite
        fix -- j1-regressions'; 0-wrong: a shipped name must denote the input), so
        the raw check now asserts that: the failure sentinel or an RT-exact name,
        never the split. Best-effort names it RT-exact ('3-[(1,3,3-trihydroxy-
        1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]propanoic acid')."""
        from orthonym.errors import is_failure_name
        from tests.support.rt_assert import name_is_rt_exact
        smi = "OC(=O)CCOP(=O)(O)OP(=O)(O)O"
        gated = name_compound(smi)
        assert gated is None or gated.startswith("unknown")   # production: fail-closed
        raw = RAW.name(smi)
        assert raw != "propanoic acid"            # never the old atom-dropped leak
        assert raw != "3-hydroxypropanoic acid diphosphoric acid"   # nor the split
        assert is_failure_name(raw) or name_is_rt_exact(raw, smi), raw


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
