""" a phase Task 6 (C4): NP keying invariant — the class C3 cannot see.

C3 (namer.py::_self_consistency_verdict, "nb>na -> mismatch") is a GATE-time check:
it re-parses the emitted name via OPSIN and compares specified-stereocentre counts.
It can only fire for names OPSIN can parse.

NAME_EXACT_NP_PARENTS (data/natural_products.py) is a frozenset of retained parent
names that are individually whitelisted to BYPASS the OPSIN-validity gate entirely
(namer.py's `_final_opsin_validity_gate` carve-out) because OPSIN cannot parse them
at all. For those names, C3 never runs, so the ONLY thing preventing a stereo
fabrication is that the SMILES key which maps to one of those names in
NATURAL_PRODUCT_DERIVATIVES is itself stereo-DEFINED -- a flat (stereo-stripped)
input can therefore never dict-match that key, so the gate-bypassing name is never
reached from stereo-undefined input.

This test locks that invariant in data, independent of any gate/producer code path:
every NATURAL_PRODUCT_DERIVATIVES key whose value is a NAME_EXACT_NP_PARENTS name
must carry stereo markup (@, /, or \\) in its SMILES key. If this test ever fails,
it means a new NAME_EXACT_NP_PARENTS entry was added with a flat (stereo-undefined)
SMILES key -- a real latent 0-wrong hole (a flat input could dict-match and emit a
config-implying, OPSIN-unparseable name that C3 can never catch).

Reference:.the workflow tooling/sdd/2026-08-16--phase1-stereo-honesty/task-6-brief.md
"""
from orthonym.data.natural_products import (
    NATURAL_PRODUCT_DERIVATIVES,
    NAME_EXACT_NP_PARENTS,
)


def test_gate_bypassing_np_parent_keys_are_stereo_defined():
    """Every NATURAL_PRODUCT_DERIVATIVES key whose value is in
    NAME_EXACT_NP_PARENTS (the OPSIN-unparseable, gate-bypassing NP parents) must
    itself carry stereo markup, so a flat/stereo-undefined input can never dict-match
    it and fabricate an unverifiable stereo-config-implying name."""
    offenders = [
        smi
        for smi, name in NATURAL_PRODUCT_DERIVATIVES.items()
        if name in NAME_EXACT_NP_PARENTS
        and not ("@" in smi or "/" in smi or "\\" in smi)
    ]
    assert offenders == [], (
        "flat (stereo-undefined) SMILES keys map to gate-bypassing NAME_EXACT_NP_PARENTS "
        f"names -- these would fabricate stereo with no gate able to catch it: {offenders}"
    )


def test_name_exact_np_parents_is_nonempty_sanity():
    """Sanity control: the frozenset this invariant protects must actually be
    populated, otherwise the first assertion passes vacuously."""
    assert len(NAME_EXACT_NP_PARENTS) > 0


def test_some_derivative_keys_map_to_name_exact_parents_sanity():
    """Sanity control: at least one NATURAL_PRODUCT_DERIVATIVES value must actually
    hit NAME_EXACT_NP_PARENTS, otherwise the main test's list comprehension is
    checking an empty selection and would pass even if the invariant were wrong."""
    matches = [
        smi
        for smi, name in NATURAL_PRODUCT_DERIVATIVES.items()
        if name in NAME_EXACT_NP_PARENTS
    ]
    assert len(matches) > 0
