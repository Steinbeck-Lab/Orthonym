"""v30 — crash fix (Fable review RISK 6): a PRIMED-locant tuple must not crash the
stereo-descriptor sort.

`_composite_locant_sort_key` handled int locants and str composite locants ('3a') but
raised `TypeError: int() argument ... not 'tuple'` on a primed locant `(5, "'")` — the
tuple form multi-component-ring `atom_to_locant` returns (5', 3', ...). A crash where an
abstention is correct. Reproducer: a spiro benzofuran-oxane glycoside.
"""
import pytest
from rdkit import Chem

from orthonym.rules.stereochemistry import _composite_locant_sort_key


@pytest.mark.unit
class TestPrimedLocantSortKey:
    def test_primed_tuple_locant_does_not_crash(self):
        # (5, "'") is a primed locant 5' from a multi-component ring
        assert _composite_locant_sort_key(((5, "'"), "R")) is not None

    def test_mixed_int_str_primed_sort_stable_and_ordered(self):
        # unprimed 5 sorts before primed 5'; composite '3a' between 3 and 4; no crash
        items = [((5, "'"), "R"), (5, "S"), ("3a", "S"), (3, "R"), (4, "S")]
        ordered = sorted(items, key=_composite_locant_sort_key)
        locants = [it[0] for it in ordered]
        assert locants.index(3) < locants.index("3a") < locants.index(4)
        assert locants.index(5) < locants.index((5, "'"))

    def test_regression_int_only_unchanged(self):
        # int-only path byte-identical to the historical (x, '') key
        assert _composite_locant_sort_key((7, "R")) == (7, "")


@pytest.mark.unit
class TestPrimedLocantMoleculeNoCrash:
    def test_spiro_glycoside_names_without_crashing(self):
        """Best-effort naming must not raise TypeError; it may abstain (fail closed)."""
        from orthonym.cli import _emit_tier_flags
        from orthonym import Orthonym
        f = _emit_tier_flags("best-effort")
        namer = Orthonym(style="pin", general_fallback=f["general_fallback"],
                          general_fallback_unverified=f["general_fallback_unverified"],
                          allow_aromatic_general=f["allow_aromatic_general"])
        smi = Chem.CanonSmiles("CCCCC[C@H]1O[C@]2(OCc3c(O)cccc32)[C@H](O)[C@@H]1O")
        res = namer.name_tiered(smi)  # must not raise
        assert res.get("tier") is not None  # a verdict, not a crash
