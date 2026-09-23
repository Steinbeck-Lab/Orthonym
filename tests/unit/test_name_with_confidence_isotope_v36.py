"""-A3: name_with_confidence must honour the isotope decorator hook.

a review cross-model review #1 found a 0-wrong hole: name / name_tiered are
protected by an isotope hook (namer.py:2938) that routes an isotope-labeled
mol through the fail-closed decorator BEFORE _name_impl strips the label, but
name_with_confidence (namer.py) called _name_impl directly and skipped it.
Result: name_with_confidence('[2H]C(Cl)(Cl)Cl') -> 'trichloromethane', silently
dropping the D3 label and shipping the UNLABELED skeleton = a wrong molecule.

 cannot catch this: the isotope layer is InChIKey block 2, which the
skeleton compare ignores; the OPSIN validity gate passes because the label-
dropped name parses to the (real, unlabeled) structure. The decorator's own RT
gate is the only protection, and name_with_confidence bypassed it.

These tests pin the mirrored hook: decorator success returns the DECORATED name
(RT-verified), decorator fail-closed abstains to the descriptive fallback (never
the unlabeled skeleton), and a non-isotope input is byte-identical (the
has_isotopes gate makes the hook a no-op).
"""
import pytest

from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def test_isotope_label_not_dropped_and_roundtrips(namer):
    """RED->GREEN: the D3-chloroform label survives, and the name round-trips.

    Before the fix this returned 'trichloromethane' (label silently dropped).
    Isotopes do not use OPSIN -r, so the round-trip is a PLAIN opsin check.
    """
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation import opsin_roundtrip_check

    smiles = "[2H]C(Cl)(Cl)Cl"
    with jvm_slots(1, purpose="test-v36-a3-isotope"):
        result = namer.name_with_confidence(smiles)
        name = result["name"]

        # The label-dropped skeleton name is the exact bug — must NOT appear.
        assert name != "trichloromethane", (
            "isotope label was silently dropped (0-wrong hole reopened)"
        )
        # It must be a D3-labelled name that OPSIN round-trips to the input.
        assert "2H" in name, f"expected an isotope-labelled name, got {name!r}"
        rt = opsin_roundtrip_check(smiles, name)
        assert rt and rt.get("passed"), (
            f"decorated isotope name {name!r} did not round-trip to {smiles}"
        )
        # It comes from the isotope hook, not the label-dropping producer.
        assert result["handler"] == "isotope"


def test_decorator_never_emits_unlabeled_skeleton(namer):
    """Invariant: the isotope decorator must NEVER fall through to the bare
    unlabeled skeleton name for a labelled molecule. It either emits a correct
    LABELLED name (round-trip-verified) or abstains to the descriptive fallback.

    [2H]O[2H] (deuterated water): the decorator now names it correctly as the
    LABELLED '(2H2)water' via the isotope hook; OPSIN-RT-verified to the
    input). Earlier this molecule abstained; the durable invariant tested here is
    the anti-skeleton guard (never bare 'water'/'oxidane'), which holds either way.
    """
    from orthonym.jvm_budget import jvm_slots

    smiles = "[2H]O[2H]"
    with jvm_slots(1, purpose="test-v36-a3-failclosed"):
        result = namer.name_with_confidence(smiles)
        name = result["name"]

        # Core invariant: NEVER the bare unlabeled skeleton (the label-drop failure mode).
        assert name not in ("oxidane", "water", "dihydridooxygen"), (
            f"decorator emitted the unlabeled skeleton {name!r}"
        )
        if result["handler"] == "isotope":
            # Named correctly as the labelled form (the isotope hook only emits a
            # name that passed its internal OPSIN round-trip gate -> 0-wrong).
            assert name == "(2H2)water", f"unexpected D2O name {name!r}"
        else:
            # If it ever abstains again, it must be a descriptive fallback, never a skeleton.
            assert result["abstention"] is not None
            from orthonym.errors import is_failure_name
            assert is_failure_name(name), (
                f"expected a descriptive-fallback name, got {name!r}"
            )


def test_non_isotope_input_byte_identical(namer):
    """Regression: the has_isotopes gate makes the hook a no-op for a plain
    (non-isotope) molecule, so its name is byte-identical to today's output.

    Captured 2026-08-23 at HEAD before the hook: name_with_confidence('CCO')
    ['name'] == 'ethanol'.
    """
    from orthonym.jvm_budget import jvm_slots

    with jvm_slots(1, purpose="test-v36-a3-regression"):
        result = namer.name_with_confidence("CCO")
        assert result["name"] == "ethanol"
