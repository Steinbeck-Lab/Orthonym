"""a phase: parity tests for _classify cyclic branch refactor.

Per V18_MILESTONE_PLAN §6 a phase + Appendix A.5: when the cyclic
branch finds a meaningful chain (chain_len >= 2), select_parent() runs
end-to-end and features.parent_selection_result is populated. The
is_known_fused_heterocycle short-circuit at namer.py:1126-1141
(pre-148) is collapsed; cascade entry is the only gate per.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1 cascade
Source: V18_MILESTONE_PLAN §6 a phase + Appendix A.5
Source: a phase CONTEXT,,.
Source: AUTONOM 1990 §4 (Wisniewski J. Chem. Inf. Comput. Sci. 30, 324-332)
        — full seniority cascade on ALL structures, no bypass.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, compute_features


def _classify_full(smiles: str):
    """Helper: run perceive + classify (mirrors what name_compound does
    internally before assembly). compute_features() alone runs only
    perception; parent_selection_result is set inside _classify.

    Source: namer.py:894-897 (name_compound's perceive→classify chain).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit failed to parse SMILES: {smiles!r}"
    canonical = Chem.CanonSmiles(smiles)
    namer = Orthonym()
    feats = namer._perceive(mol, smiles, canonical)
    namer._classify(feats)
    return feats


@pytest.mark.unit
def test_parent_selection_result_wired():
    """a phase / V18 Appendix A.5: features.parent_selection_result
    is populated when _classify cyclic branch fires select_parent().

    Indole + C11 acid is the canonical cascade-fires case (chain_len=11
    well above the 2-atom minimum). Pre-148 the bypass guard would have
    short-circuited this case; post-148 the cascade runs end-to-end and
    populates parent_selection_result.

    Source: V18_MILESTONE_PLAN §6 a phase Appendix A.5
    Source: a phase CONTEXT,.
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: AUTONOM 1990 §4.
    """
    # indole + 11C acid → cascade fires (chain_len >= 2)
    feats = _classify_full('c1ccc2[nH]ccc2c1CCCCCCCCCCC(=O)O')
    assert feats.parent_selection_result is not None, (
        "features.parent_selection_result should be populated by _classify "
        "when select_parent() runs (Phase 148 D-02 wiring)."
    )
    # ParentSelectionResult is a NamedTuple/dataclass with.parent_type
    assert feats.parent_selection_result.parent_type in ('ring', 'chain'), (
        f"parent_type must be 'ring' or 'chain'; got "
        f"{feats.parent_selection_result.parent_type!r}"
    )


@pytest.mark.unit
def test_classify_no_known_fused_heterocycle_short_circuit():
    """a phase SC-2 /: cascade entry condition is the only gate.

    The is_known_fused_heterocycle short-circuit at namer.py:1126-1141
    (pre-148) is collapsed; cascade fires whenever (is_cyclic AND
    chain_len >= 2). This test verifies that an indole + chain compound
    now flows through select_parent() (and thus produces a populated
    parent_selection_result), where it would have been bypassed pre-148.

    If this assertion fails, the cyclic branch refactor regressed and
    re-introduced the band-aid memory/root-cause-fixes.md identifies
    as the canonical pattern this triple is deleting.

    Source: a phase CONTEXT; V18 Appendix A.5.
    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: AUTONOM 1990 §4 — full seniority cascade on ALL structures.
    Source: memory/root-cause-fixes.md (band-aid pattern reference).
    """
    feats = _classify_full('c1ccc2[nH]ccc2c1CCCCCCCCCCC(=O)O')
    # Pre-148: features.parent_selection_result would be None (bypass fired
    # via is_known_fused_heterocycle == True, skipping the cascade body).
    # Post-148: bypass is deleted; cascade fires; result is populated.
    assert feats.parent_selection_result is not None, (
        "Pre-148 the is_known_fused_heterocycle short-circuit would have "
        "skipped select_parent() for indole + C11 acid; if this assertion "
        "fails the refactor reintroduced the band-aid (D-03 violation)."
    )
