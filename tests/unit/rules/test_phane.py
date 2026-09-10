"""a phase.A unit tests for src/orthonym/rules/phane.py.

Per internal notes + internal notes-A.md: OPSIN 2.9.0 does NOT parse
[m.n]paracyclophane semi-systematic names, so the unit tier asserts
direct string equality against Blue Book reference; the integration
tier (test_cyclophane_corpus.py) skip-quarantines the OPSIN-RT step.

Test classes:
- TestIsCyclophane: topology gate (corrected SSSR criterion)
- TestClassifyPhaneTopology: sub-class enum dispatch
- TestBuildCompositeLocant: composite-locant emitter
- TestNameCyclophane: top-level handler emission

Source: 155-internal notes ////; internal notes-A.md
Critical Finding 0 (corrected SSSR-based criterion); internal notes.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.phane import (
    PhaneTopology,
    _build_composite_locant,
    _classify_phane_topology,
    is_cyclophane,
    name_cyclophane,
)


def test_phane_module_exports_public_api() -> None:
    """Smoke test: phane.py exports the six public symbols.

    Source: 155-internal notes //;; internal notes.
    """
    assert callable(is_cyclophane)
    assert callable(name_cyclophane)
    assert callable(_classify_phane_topology)
    assert callable(_build_composite_locant)
    assert PhaneTopology.PARACYCLOPHANE.value == "paracyclophane"
    assert PhaneTopology.METACYCLOPHANE.value == "metacyclophane"
    assert PhaneTopology.ORTHOCYCLOPHANE.value == "orthocyclophane"
    assert PhaneTopology.GENERIC_CYCLOPHANE.value == "generic"


# ---------------------------------------------------------------------------
# TestIsCyclophane (corrected SSSR criterion)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIsCyclophane:
    """Topology gate per 155-internal notes + internal notes-A.md Critical Finding 0.

    Cyclophane criterion (audit-corrected SSSR-based):
    1. >= 2 disjoint SSSR rings of size <= 8 (small rings).
    2. >= 1 macrocyclic SSSR ring of size > 8.
    3. Shortest atom-disjoint chain between two small rings has
       >= 2 intermediate atoms minimum bridge length).
    4. Chain shares atoms with at least one macrocyclic SSSR ring.
    5. Mutually exclusive with ring-assembly / spiro / fused / multiplicative.
    """

    @pytest.mark.parametrize(
        "smiles,expected,label",
        [
            # Positive cases: cyclophanes per Blue Book
            ("c1cc2ccc1CCc1ccc(cc1)CC2", True, "[2.2]paracyclophane"),
            ("c1cc2cc(c1)CCc1cccc(c1)CC2", True, "[2.2]metacyclophane"),
            ("c1cc2ccc1CCCc1ccc(cc1)CCC2", True, "[3.3]paracyclophane"),
            ("c1cc2ccc1CNc1ccc(cc1)CN2", True, "aza-paracyclophane (R3)"),
            ("c1cc2ccc1COc1ccc(cc1)CO2", True, "oxa-paracyclophane (R3)"),
            ("c1cc2ncc1CCc1ccc(cn1)CC2", True, "[2.2](2,5)pyridinophane"),
            # Negative cases: NOT cyclophanes (mutual-exclusion controls)
            ("c1ccc(-c2ccccc2)cc1", False, "biphenyl (ring_assembly)"),
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", False, "4,4'-methylenedianiline (multiplicative)"),
            ("c1ccc2ccccc2c1", False, "naphthalene (fused)"),
            ("c1ccc(CCc2ccc(CCc3ccccc3)cc2)cc1", False, "1,4-bis(2-phenylethyl)benzene (open-chain)"),
            ("c1ccccc1CCc1ccccc1", False, "two benzenes via -CH2-CH2- (acyclic)"),
            ("CCO", False, "ethanol (no rings)"),
        ],
    )
    def test_is_cyclophane_classification(self, smiles, expected, label) -> None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES for {label}: {smiles!r}"
        result = is_cyclophane(mol)
        assert result is expected, (
            f"is_cyclophane({label}!r) returned {result}; expected {expected}. "
            f"SMILES={smiles!r}"
        )

    def test_is_cyclophane_rejects_none(self) -> None:
        assert is_cyclophane(None) is False


# ---------------------------------------------------------------------------
# TestClassifyPhaneTopology
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyPhaneTopology:
    """Sub-class enum dispatch per 155-CONTEXT.md D-04 + 155-AUDIT-A.md §4."""

    @pytest.mark.parametrize(
        "smiles,expected_topology,label",
        [
            ("c1cc2ccc1CCc1ccc(cc1)CC2", PhaneTopology.PARACYCLOPHANE, "[2.2]paracyclophane"),
            ("c1cc2cc(c1)CCc1cccc(c1)CC2", PhaneTopology.METACYCLOPHANE, "[2.2]metacyclophane"),
            ("c1cc2ccc1CCCc1ccc(cc1)CCC2", PhaneTopology.PARACYCLOPHANE, "[3.3]paracyclophane"),
            ("c1cc2ncc1CCc1ccc(cn1)CC2", PhaneTopology.GENERIC_CYCLOPHANE, "pyridinophane"),
        ],
    )
    def test_classify_phane_topology(
        self, smiles, expected_topology, label
    ) -> None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        result = _classify_phane_topology(mol)
        assert result is expected_topology, (
            f"_classify_phane_topology({label}!r) returned {result.name}; "
            f"expected {expected_topology.name}"
        )


# ---------------------------------------------------------------------------
# TestBuildCompositeLocant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCompositeLocant:
    """Composite-locant emitter per 155-CONTEXT.md D-05 + 155-AUDIT-A.md §5."""

    def test_ascii_style(self) -> None:
        assert _build_composite_locant(1, 4, style="ascii") == "1(4)"
        assert _build_composite_locant(2, 5, style="ascii") == "2(5)"

    def test_ascii_default(self) -> None:
        # Default style is ASCII
        assert _build_composite_locant(1, 4) == "1(4)"

    def test_superscript_style(self) -> None:
        assert _build_composite_locant(1, 4, style="superscript") == "1⁴"
        assert _build_composite_locant(2, 14, style="superscript") == "2¹⁴"

    def test_unknown_style_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _build_composite_locant(1, 4, style="latex")


# ---------------------------------------------------------------------------
# TestNameCyclophane (+)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNameCyclophane:
    """Top-level handler per 155-internal notes + + internal notes-A.md

    Wave-8 P8: `name_cyclophane` now delegates to `build_phane_pin` (the
    /.3 simplified-skeletal PIN engine) for the monocyclic
    all-benzene-homophane class, RETIRING the semi-systematic bracket-prefix
    form for these cases (`[2.2]paracyclophane` -> PIN
    `1,4(1,4)-dibenzenacyclohexaphane`). See
    docs/superpowers/plans/2026-07-16-wave8-p8-phane.md Task 8.7 +
    tests/unit/rules/test_phane_pin.py for the full engine test suite;
    these 4 cases stay here (pre-existing fixture SMILES) purely so this
    file's own coverage of `name_cyclophane`'s public contract doesn't rot.
    """

    @pytest.mark.parametrize(
        "smiles,expected_name,label",
        [
            ("c1cc2ccc1CCc1ccc(cc1)CC2", "1,4(1,4)-dibenzenacyclohexaphane",
             "[2.2]paracyclophane -> P-26 PIN (BB P-26.3.2.1 :14947 (PIN) verbatim)"),
            ("c1cc2cc(c1)CCc1cccc(c1)CC2", "1,4(1,3)-dibenzenacyclohexaphane",
             "[2.2]metacyclophane -> P-26 PIN (BB P-26.4.1.4 :15024 (PIN) verbatim)"),
            ("c1cc2ccc1CCCc1ccc(cc1)CCC2", "1,5(1,4)-dibenzenacyclooctaphane",
             "[3.3]paracyclophane -> P-26 PIN (rule-derived homolog)"),
            ("c1cc2ccc1CCCc1ccc(cc1)CC2", "1,4(1,4)-dibenzenacycloheptaphane",
             "[3.2]paracyclophane -> P-26 PIN (rule-derived homolog, asymmetric bridge)"),
        ],
    )
    def test_name_cyclophane_emits_blue_book_form(
        self, smiles, expected_name, label
    ) -> None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        result = name_cyclophane(mol)
        assert result == expected_name, (
            f"name_cyclophane({label}!r) returned {result!r}; expected {expected_name!r}"
        )

    @pytest.mark.parametrize(
        "smiles,label",
        [
            ("c1ccc(-c2ccccc2)cc1", "biphenyl (ring_assembly)"),
            ("Nc1ccc(Cc2ccc(N)cc2)cc1", "methylenedianiline (multiplicative)"),
            ("c1ccc2ccccc2c1", "naphthalene (fused)"),
            ("CCO", "ethanol (no rings)"),
        ],
    )
    def test_name_cyclophane_returns_none_for_non_cyclophane(
        self, smiles, label
    ) -> None:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert name_cyclophane(mol) is None, (
            f"name_cyclophane({label}!r) should be None for non-cyclophane "
            f"input but returned a name."
        )

    def test_name_cyclophane_returns_none_on_none_input(self) -> None:
        assert name_cyclophane(None) is None
