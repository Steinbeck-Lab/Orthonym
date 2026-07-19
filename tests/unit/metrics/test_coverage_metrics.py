"""v25 P0 Task 0.3 — three-metric coverage instrument (dual matcher).

No Java: OPSIN is a dict-backed fake injected as ``name_to_smiles``. Every
branch of the classifier and both matchers are exercised on hand-built
structures with known relationships (identical / tautomer / charge-only /
stereo-only / different-constitution / unparseable).
"""

import pytest

from orthonym.metrics.coverage_metrics import (
    Matcher,
    RowClassification,
    aggregate,
    classify_row,
    parity_match,
    strict_match,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

class TestParityMatcher:
    def test_identical(self):
        assert parity_match("CCO", "CCO") is True

    def test_charge_only_difference_is_match(self):
        # acetate anion vs acetic acid: parity uncharges -> same
        assert parity_match("CC(=O)[O-]", "CC(=O)O") is True

    def test_keto_enol_tautomer_is_match(self):
        # parity uses RDKit canonical tautomer -> merges keto/enol of acetone
        assert parity_match("CC(C)=O", "CC(=C)O") is True

    def test_amide_tautomer_is_match(self):
        # 2-pyridone / 2-hydroxypyridine
        assert parity_match("O=c1cccc[nH]1", "Oc1ccccn1") is True

    def test_stereo_difference_is_forgiven(self):
        # DOCUMENTED: CanonicalTautomer strips stereo, so parity forgives a
        # stereo difference (credits omission; an inversion is a C.1 residual,
        # not caught here). L-ala vs D-ala -> match.
        assert parity_match("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O") is True

    def test_different_constitution_is_mismatch(self):
        assert parity_match("CCO", "CCC") is False

    def test_unparseable_side_is_mismatch(self):
        assert parity_match("CCO", "not a smiles!!!") is False


class TestStrictMatcher:
    def test_identical(self):
        assert strict_match("CCO", "CCO") is True

    def test_stereo_difference_is_match(self):
        # strict skeleton is stereo-INSENSITIVE
        assert strict_match("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O") is True

    def test_charge_difference_is_mismatch(self):
        # net charge differs -> strict mismatch (the [O-]O -> dioxidane leak)
        assert strict_match("[O-]O", "OO") is False

    def test_mobile_h_tautomer_is_match(self):
        # InChIKey skeleton block IS mobile-H normalized: 2-pyridone /
        # 2-hydroxypyridine (an amide mobile-H shift) merge.
        assert strict_match("O=c1cccc[nH]1", "Oc1ccccn1") is True

    def test_keto_enol_is_mismatch(self):
        # BOUNDARY: standard InChI does NOT merge keto-enol, so the strict
        # (skeleton-block) matcher flags acetone vs its enol — unlike parity.
        assert strict_match("CC(C)=O", "CC(=C)O") is False

    def test_different_constitution_is_mismatch(self):
        assert strict_match("CCO", "CCC") is False


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------

def _fake_opsin(mapping):
    def _f(name):
        return mapping.get(name)
    return _f


class TestClassifyRow:
    def test_abstention(self):
        r = classify_row("CCO", "unknown organic compound", _fake_opsin({}))
        assert r.emitted is False
        assert r.parity is RowClassification.ABSTAINED
        assert r.strict is RowClassification.ABSTAINED

    def test_empty_name_is_abstention(self):
        r = classify_row("CCO", "", _fake_opsin({}))
        assert r.emitted is False

    def test_rt_valid_both_matchers(self):
        r = classify_row("CCO", "ethanol", _fake_opsin({"ethanol": "CCO"}))
        assert r.emitted is True and r.opsin_parsed is True
        assert r.parity is RowClassification.RT_VALID
        assert r.strict is RowClassification.RT_VALID

    def test_unparseable_bucket_not_wrong(self):
        # emitted, but OPSIN returns None -> UNPARSEABLE under BOTH, never wrong
        r = classify_row("CCO", "ethanol-but-opsin-cant-parse",
                         _fake_opsin({}))
        assert r.opsin_parsed is False
        assert r.parity is RowClassification.UNPARSEABLE
        assert r.strict is RowClassification.UNPARSEABLE

    def test_confidently_wrong(self):
        # OPSIN parses to a DIFFERENT molecule -> wrong under both
        r = classify_row("CCO", "propane", _fake_opsin({"propane": "CCC"}))
        assert r.opsin_parsed is True
        assert r.parity is RowClassification.CONFIDENTLY_WRONG
        assert r.strict is RowClassification.CONFIDENTLY_WRONG

    def test_matchers_can_disagree_on_charge(self):
        # charge-only difference: parity credits it (RT_VALID), strict flags
        # it (CONFIDENTLY_WRONG) — the exact reason both are reported.
        r = classify_row("[O-]O", "dioxidane", _fake_opsin({"dioxidane": "OO"}))
        assert r.opsin_parsed is True
        assert r.parity is RowClassification.RT_VALID
        assert r.strict is RowClassification.CONFIDENTLY_WRONG

    def test_matchers_can_disagree_on_tautomer(self):
        # keto-enol difference: parity (broad tautomer) credits it, strict
        # (InChI mobile-H only) flags it. Input acetone, name parses to enol.
        r = classify_row("CC(C)=O", "prop-1-en-2-ol",
                         _fake_opsin({"prop-1-en-2-ol": "CC(=C)O"}))
        assert r.opsin_parsed is True
        assert r.parity is RowClassification.RT_VALID
        assert r.strict is RowClassification.CONFIDENTLY_WRONG


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregate:
    def _rows(self):
        opsin = _fake_opsin({
            "ethanol": "CCO",             # rt valid both
            "propane": "CCC",             # wrong both (input methanol)
            "dioxidane": "OO",            # parity-ok, strict-wrong (input [O-]O)
            "weird": None,                # unparseable
        })
        data = [
            ("CCO", "ethanol"),
            ("CO", "propane"),            # confidently wrong
            ("[O-]O", "dioxidane"),       # matcher-disagreement
            ("CCCO", "weird"),            # unparseable emission
            ("c1ccccc1C2CCCCC2", "unknown organic compound"),  # abstain
        ]
        return [classify_row(s, n, opsin) for s, n in data]

    def test_counts_and_rates(self):
        agg = aggregate(self._rows(), pin_pct=0.42)
        assert agg["pin_pct"] == 0.42

        p = agg["parity"]
        assert p["total"] == 5
        assert p["covered"] == 4                 # 4 emitted, 1 abstained
        assert abs(p["coverage"] - 0.8) < 1e-9
        # parity RT-valid: ethanol + dioxidane = 2
        assert p["rt_valid"] == 2
        assert abs(p["of_emitted_rt"] - 0.5) < 1e-9
        # confidently wrong (parity): propane only = 1 (of total)
        assert p["confidently_wrong_count"] == 1
        assert abs(p["confidently_wrong"] - 0.2) < 1e-9
        # unparseable: weird = 1 (matcher-invariant), NOT counted as wrong
        assert p["unparseable_count"] == 1
        # abs_rt_valid = coverage * of_emitted_rt = 0.8 * 0.5 = 0.4 = 2/5
        assert abs(p["abs_rt_valid"] - 0.4) < 1e-9

        s = agg["strict"]
        # strict RT-valid: ethanol only (dioxidane is charge-wrong) = 1
        assert s["rt_valid"] == 1
        assert s["confidently_wrong_count"] == 2   # propane + dioxidane
        assert s["unparseable_count"] == 1
        assert abs(s["abs_rt_valid"] - 0.2) < 1e-9  # 1/5

    def test_unparseable_never_inflates_wrong(self):
        # A run that is all-unparseable emissions has confidently_wrong == 0.
        opsin = _fake_opsin({})
        rows = [classify_row("CCO", "x", opsin) for _ in range(3)]
        agg = aggregate(rows)
        assert agg["parity"]["confidently_wrong_count"] == 0
        assert agg["parity"]["unparseable_count"] == 3

    def test_empty(self):
        agg = aggregate([])
        assert agg["parity"]["coverage"] == 0.0
        assert agg["parity"]["abs_rt_valid"] == 0.0
