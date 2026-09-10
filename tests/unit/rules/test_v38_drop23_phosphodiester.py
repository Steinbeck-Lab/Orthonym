""" substituent_no_prefix_form phosphodiester-bridge — regression guard + refuted-premise record.

The substituent_no_prefix_form brief asked to render an asymmetric phosphodiester bridge
``R-O-P(=O)(OH)-O-R'`` as a substituent prefix instead of dropping it, on the
premise that the class ABSTAINS today (trace: ``COP(=O)(O)OC[C@H](N)C(=O)O`` →
``unknown``).

That premise is REFUTED on HEAD 8773c400 for the entry point the eval harness
(``eval/harness.py`` → ``name_tiered``) and ``scripts/measure_breadth.py``
(→ ``name_tiered``) actually use: the whole addressable class already EMITS an
RT-verified (full-InChIKey) name at best-effort tier T3 via the general engine.
The trace's "abstains today" was an artifact of probing through
``name_with_confidence()`` — an entry point that DIVERGES from
``name()`` / ``name_tiered()`` (it abstains where they emit) and is used by
neither harness (only ``name_compound(..., include_confidence=True)`` reaches it).

Full analysis: `internal notes`.

These tests therefore PIN the already-correct behaviour so it cannot silently
regress:

  * the symmetric-diester passers stay byte-identical (the invariant-14 risk the
    brief flagged) — at BOTH default and best-effort tiers,
  * the witness emits the PIN-correct, *alphabetically ordered* prefix
    ``[hydroxy(methoxy)phosphoryl]oxy`` (P-14.5.2) and round-trips 0-wrong,
  * that spelling is deterministic across input SMILES orderings,
  * the class members (lyso-PE, guanidinoethyl phosphate) emit a non-failure,
    RT-verified name via ``name_tiered`` at best-effort.

It also documents (``test_weave_renderer_is_head_first_not_pin``) why the sited
mechanism (reuse ``weave._phosphoryloxy_name`` at ``polyfunctional.py`` substituent_no_prefix_form)
must NOT be shipped as written: that renderer emits the *head-first* order
``[(methoxy)hydroxyphosphoryl]oxy``, which — running earlier in the cascade than
the general engine — would OVERRIDE and regress the current PIN-correct T3
spelling. 0-wrong / PIN-first / no-regression forbid it.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name
from orthonym.validation.reconstruct import verify_or_none


WITNESS = "COP(=O)(O)OC[C@H](N)C(=O)O"  # O-phospho-L-serine methyl ester
WITNESS_PIN = "(2S)-2-amino-3-{[hydroxy(methoxy)phosphoryl]oxy}propanoic acid"

# --- invariant-14 controls: symmetric / small phosphate (di)esters that ALREADY
# name and MUST stay byte-identical (they never reach the substituent_no_prefix_form site —
# polyfunctional declines them, substituent_no_prefix_form does not fire). ---
SYMMETRIC_CONTROLS = {
    "COP(=O)(O)OC": "dimethyl hydrogen phosphate",
    "CCOP(=O)(O)OCC": "diethyl hydrogen phosphate",
    "COP(=O)(O)OCC": "ethyl methyl hydrogen phosphate",
    "COP(=O)(O)O": "methyl dihydrogen phosphate",
}
TRIVIAL_CONTROLS = {"CCO": "ethanol", "c1ccccc1": "benzene"}


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected",
                         list(SYMMETRIC_CONTROLS.items()) + list(TRIVIAL_CONTROLS.items()))
def test_symmetric_diester_controls_byte_identical_default(smiles, expected):
    """Invariant-14 risk: the substituent_no_prefix_form signature also fires on passing rows.
    The symmetric diesters must stay byte-identical at the DEFAULT (PIN) tier."""
    assert name_compound(smiles) == expected


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected",
                         list(SYMMETRIC_CONTROLS.items()) + list(TRIVIAL_CONTROLS.items()))
def test_symmetric_diester_controls_byte_identical_besteffort(smiles, expected):
    """Same controls at BEST-EFFORT tier (general_fallback=True)."""
    assert name_compound(smiles, general_fallback=True) == expected


@pytest.mark.roundtrip
@pytest.mark.opsin_gate  # production runs gate-ON; the rescue depends on it
def test_witness_emits_pin_ordered_prefix_at_besteffort_tiered():
    """The witness already names via the eval/breadth entry point
    (``name_tiered``) at best-effort tier T3, with the P-14.5.2 alphabetical
    prefix order ``hydroxy(methoxy)phosphoryl`` — NOT the head-first order."""
    res = Orthonym(general_fallback=True).name_tiered(WITNESS)
    assert res.get("name") == WITNESS_PIN
    # the ordering is the PIN one, not the renderer's head-first form
    assert "hydroxy(methoxy)phosphoryl" in res["name"]
    assert "(methoxy)hydroxyphosphoryl" not in res["name"]


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_witness_is_zero_wrong_full_inchikey():
    """0-wrong: the emitted witness name round-trips on the FULL InChIKey."""
    res = Orthonym(general_fallback=True).name_tiered(WITNESS)
    name = res.get("name")
    assert not is_failure_name(name)
    assert verify_or_none(name, WITNESS) == name


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_witness_spelling_is_deterministic_across_orderings():
    """Determinism (a project rule): different input SMILES orderings of the same
    structure must yield the identical name."""
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(WITNESS))
    reorder = "OC(=O)[C@@H](N)COP(=O)(O)OC"
    names = {
        Orthonym(general_fallback=True).name_tiered(s)["name"]
        for s in (WITNESS, canon, reorder)
    }
    assert names == {WITNESS_PIN}


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
def test_witness_default_tier_still_abstains_documents_residual_gap():
    """DOCUMENTS the only genuine residual: at the DEFAULT (PIN) tier the witness
    still abstains — the class emits at best-effort (T3) but not at T1. Promoting
    it is a separate tier-policy / PIN-determination task (see the finding doc),
    NOT the sited best-effort offer-not-return wiring fix. Guards against an
    accidental default-tier emission that ships the non-PIN head-first order."""
    got = name_compound(WITNESS)  # default tier
    if not is_failure_name(got):
        # If a future change DOES emit at default, it must be the PIN-ordered
        # form and round-trip 0-wrong — never the head-first regression.
        assert got == WITNESS_PIN
        assert verify_or_none(got, WITNESS) == got


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)(O)OCCN",  # lyso-PE
    "COP(=O)(O)OCCNC(=N)N",                                  # guanidinoethyl Me phosphate
])
def test_addressable_class_members_emit_rt_verified_at_besteffort(smiles):
    """The rest of the addressable class also already emits a non-failure,
    RT-verified name via ``name_tiered`` at best-effort. (The exact strings are
    not pinned — some are ugly replacement names whose PIN quality is a separate
    lever — only that they emit 0-wrong, refuting the abstention premise.)"""
    res = Orthonym(general_fallback=True).name_tiered(smiles)
    name = res.get("name")
    assert not is_failure_name(name)
    assert verify_or_none(name, smiles) == name


def test_weave_renderer_is_head_first_not_pin():
    """Documents WHY the sited mechanism must not be shipped: the existing
    renderer ``weave._phosphoryloxy_name`` emits the head-first substituent order
    ``[(methoxy)hydroxyphosphoryl]oxy``, which is NOT P-14.5.2 alphabetical.
    Wired at substituent_no_prefix_form (earlier in the cascade than the general engine) it would
    override and REGRESS the current PIN-correct T3 spelling."""
    from orthonym.decomposition import weave
    mol = Chem.MolFromSmiles(WITNESS)
    p_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P")
    p = mol.GetAtomWithIdx(p_idx)
    rendered = None
    for o in p.GetNeighbors():
        b = mol.GetBondBetweenAtoms(p_idx, o.GetIdx())
        if (o.GetSymbol() == "O" and b.GetBondType() == Chem.BondType.SINGLE
                and o.GetTotalNumHs() == 0):
            c = [x for x in o.GetNeighbors() if x.GetIdx() != p_idx]
            if not c:
                continue
            core = set(weave._arm_atoms(mol, {p_idx, o.GetIdx()}, c[0].GetIdx())) | {o.GetIdx()}
            nm = weave._phosphoryloxy_name(mol, core, p_idx, o.GetIdx())
            if nm and "methoxy" in nm:
                rendered = nm
    assert rendered == "[(methoxy)hydroxyphosphoryl]oxy"  # head-first, non-PIN order
