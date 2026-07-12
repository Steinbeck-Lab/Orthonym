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
 §2.D.
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
        # gold W2C-C-MA-02 (recognizer _try_methylenebis_oxy_bridge :862)
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
