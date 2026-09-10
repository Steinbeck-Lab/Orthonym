"""W2F-P1 Tasks 4-5-7 — P-29.5.2 benzylic-ether classes.

Symmetric class (Tasks 4-5): the PIN is MULTIPLICATIVE (P-51.3.1, BB 23180;
research §2.A corrects the brief) — new composite-bridge recognizer
_try_chalcogenbis_methylene_bridge in rules/multiplicative.py:
-CH2-O-CH2- -> 'oxybis(methylene)' (P-15.3.1.2.2.1, BB 5242 verbatim),
-CH2-S-CH2- -> 'sulfanediylbis(methylene)' (P-29.5.2 family, BB 16256).

Asymmetric sub-class (Task 7): P-51.3.1(3) fails -> substitutive PIN with the
phenol (PCG) ring as parent; retained 'benzyloxy' is forbidden on a
substituted ring in PINs (P-29.6.1, BB 16274).

All expected names OPSIN-2.9-verified in
internal notes §2.D.
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound

UNKNOWN = "unknown organic compound"


def _ring_atoms(mol):
    out = set()
    for r in mol.GetRingInfo().AtomRings():
        out.update(r)
    return out


@pytest.mark.unit
class TestOxybisMethyleneBridge:
    """Task 4: symmetric -CH2-X-CH2- bridge, end-to-end + direct."""

    def test_curated_target_diphenol(self):
        assert name_compound("Oc1ccc(COCc2ccc(O)cc2)cc1") == \
            "4,4'-[oxybis(methylene)]diphenol"

    def test_sulfur_analog(self):
        assert name_compound("Oc1ccc(CSCc2ccc(O)cc2)cc1") == \
            "4,4'-[sulfanediylbis(methylene)]diphenol"

    def test_substituted_identical_units_bis(self):
        # P-15.3.2.1/P-15.3.2.4.1: substituted identical units take bis(...)
        assert name_compound("Cc1cc(COCc2cc(C)c(O)cc2)ccc1O") == \
            "4,4'-[oxybis(methylene)]bis(2-methylphenol)"

    def test_recognizer_direct(self):
        from orthonym.rules.multiplicative import (
            _try_chalcogenbis_methylene_bridge,
        )
        mol = Chem.MolFromSmiles("Oc1ccc(COCc2ccc(O)cc2)cc1")
        assert _try_chalcogenbis_methylene_bridge(mol, _ring_atoms(mol)) == \
            "4,4'-[oxybis(methylene)]diphenol"

    # --- regression anchors: shipped bridge family stays byte-identical ---

    def test_methylenebis_oxy_untouched(self):
        # gold W2C-C-MA-02 (recognizer _try_methylenebis_oxy_bridge:862)
        assert name_compound("Oc1ccc(OCOc2ccc(O)cc2)cc1") == \
            "4,4'-[methylenebis(oxy)]diphenol"

    def test_single_atom_methylene_untouched(self):
        assert name_compound("Oc1ccc(Cc2ccc(O)cc2)cc1") == \
            "4,4'-methylenediphenol"


@pytest.mark.unit
class TestBridgeFailClosed:
    """Task 5: research §2.D fail-closed rows."""

    def test_unnameable_units_refuse(self):
        # boronic-acid units: the unit namer cannot produce them -> the
        # recognizer (and the whole pipeline) must refuse, never emit a
        # wrong unit name.
        assert name_compound("OB(O)c1ccc(COCc2ccc(B(O)O)cc2)cc1") == UNKNOWN

    def test_p51_3_1_locant_guard_direct(self):
        # 3-OH vs 4-OH: the split FRAGMENTS are both 'phenol'
        # (canon-identical) so the identity test passes — the shared
        # _resolve_unit_and_assemble per-connection locant equality check
        # (multiplicative.py:1617-1620) is the LOAD-BEARING rejection
        # (P-51.3.1 condition (3): suffix locants must be identical).
        from orthonym.rules.multiplicative import (
            _try_chalcogenbis_methylene_bridge,
        )
        mol = Chem.MolFromSmiles("Oc1ccc(COCc2cc(O)ccc2)cc1")
        assert _try_chalcogenbis_methylene_bridge(mol, _ring_atoms(mol)) is None

    def test_locant_mismatch_never_emits_44(self):
        # end-to-end: whatever the pipeline does with the 3-OH/4-OH probe,
        # it must NEVER be a false 4,4'-multiplicative name.
        out = name_compound("Oc1ccc(COCc2cc(O)ccc2)cc1")
        assert "oxybis(methylene)" not in out
        assert not out.startswith("4,4'")


@pytest.mark.unit
class TestBridgeDeterminism:
    """Task 5: research §2.E protocol — 10+ random spellings, byte-identical.

    The recognizer sidesteps the symmetric parent choice entirely (units are
    canonicalized fragments; fixed try-chain order), so any spread here is a
    REAL bug — root-cause, never quarantine."""

    @pytest.mark.parametrize("smi,expected", [
        ("Oc1ccc(COCc2ccc(O)cc2)cc1", "4,4'-[oxybis(methylene)]diphenol"),
        ("Cc1cc(COCc2cc(C)c(O)cc2)ccc1O",
         "4,4'-[oxybis(methylene)]bis(2-methylphenol)"),
    ])
    def test_random_spellings_byte_identical(self, smi, expected):
        mol = Chem.MolFromSmiles(smi)
        outs = {
            name_compound(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            for _ in range(12)
        }
        assert outs == {expected}


@pytest.mark.unit
class TestAsymmetricSubstitutedBenzyloxy:
    """Task 7: C2 — P-51.3.1(3) fails (methoxy vs hydroxy) -> substitutive
    PIN, phenol parent (only ring with the -ol PCG); P-29.6.1 kills retained
    benzyloxy on the substituted ring."""

    def test_chokepoint_substituted_ring(self):
        from orthonym.assembly.substituent_naming import _name_aryl_methyl_ether
        mol = Chem.MolFromSmiles("COc1ccc(COCc2ccc(O)cc2)cc1")
        # atom 6 = benzylic CH2 of the METHOXY ring; atom 7 = ether O
        assert _name_aryl_methyl_ether(mol, 6, 7) == "(4-methoxyphenyl)methoxy"

    def test_chokepoint_plain_phenyl_byte_identical(self):
        from orthonym.assembly.substituent_naming import _name_aryl_methyl_ether
        mol = Chem.MolFromSmiles("c1ccccc1COC")
        # atom 6 = benzylic CH2, atom 7 = O -> retained preferred prefix
        assert _name_aryl_methyl_ether(mol, 6, 7) == "benzyloxy"

    @pytest.mark.xfail(
        reason="w2f p1 Task 7 DEFERRAL: the producer chokepoint is fixed "
        "(test_chokepoint_substituted_ring + name_substituent_fragment now "
        "emit '[(4-methoxyphenyl)methoxy]methyl' correctly), but the "
        "benzene HANDLER declines PARENT SELECTION for this molecule — two "
        "benzene rings each bearing an O-substituent (phenol-OH + anisole-OMe) "
        "make it split at the central ether O instead of naming the phenol "
        "ring with the full -CH2-O-CH2-anisole substituent (traced: no ESC-CALL "
        "n>=3; general_acyclic 'empty parent stem'). The methyl-far-ring "
        "variant (single ring-substituent) DOES route through and names OK, "
        "proving the residual is benzene-handler ring-disambiguation — OUT of "
        "this plan's scope (Task 8: parent-selection edits are out of scope; "
        "Out-of-scope #6 acceptable fail-closed residual). Stays fail-closed "
        "(never-wrong). Gold W2F-P1-10 dropped; un-xfail + add gold when the "
        "benzene-handler two-O-ring parent selection lands.",
        strict=True,
    )
    def test_asymmetric_target_braces_pin(self):
        # strict P-16.5.4 escalation parens->brackets->braces (research §0)
        assert name_compound("COc1ccc(COCc2ccc(O)cc2)cc1") == \
            "4-{[(4-methoxyphenyl)methoxy]methyl}phenol"

    def test_asymmetric_target_fails_closed_residual(self):
        # DOCUMENTS the current never-wrong end state: the benzene-handler
        # parent-selection residual keeps this fail-closed (unknown), NEVER a
        # decoration-dropped wrong name. Paired with the xfail target above.
        assert name_compound("COc1ccc(COCc2ccc(O)cc2)cc1") == UNKNOWN

    def test_benzyloxy_sibling_regression(self):
        # HEAD-OK anchor: retained benzyloxy VALID while ring+CH2 are bare
        # (P-29.6.1; P-35.4.2 BB 18116 '(benzyloxy)carbonyl' precedent)
        assert name_compound("Oc1ccc(COCc2ccccc2)cc1") == \
            "4-(benzyloxymethyl)phenol"
