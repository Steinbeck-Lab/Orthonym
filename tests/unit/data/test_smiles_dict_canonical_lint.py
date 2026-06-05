"""Phase 169.7 BBR-HYG / D-11 — anti-drift lint for hard-coded SMILES dicts.

The data layer keys several lookups by SMILES. If a key is NOT the RDKit canonical
SMILES, the canonical-SMILES lookup silently MISSES it — a dead/mis-routed entry
(audit Dim-02 §4; the D-10 mislabel class). This lint re-canonicalizes every
hard-coded SMILES-keyed dict against live RDKit so the drift surfaces on every CI
run and NEW non-canonical keys are rejected.

RATCHET, not a big-bang fix: the fused-heterocycle + ion dicts are HARD-gated at
zero non-canonical keys. ``RETAINED_NAMES`` carries 16 pre-existing non-canonical
keys (``_KNOWN_NONCANONICAL_RETAINED``) — a MIXED bag of redundant duplicates
(acetamide/imidazole/thiazole have correct canonical twins) and genuine bugs
(``oxalic acid``'s correct-name key is dead while the canonical key carries the
wrong ``dihydroxalate`` value). Re-canonicalizing them is behaviour-changing
(activates dead retained names) and needs per-entry canary validation, so it is a
TRACKED FOLLOW-ON (see  + 169.7-VERIFICATION.md).
This test pins the known set so the count can only SHRINK, never grow.
"""
import pytest
from rdkit import Chem


def _noncanonical(dictobj):
    """Return the list of non-canonical SMILES keys in a SMILES-keyed dict.
    Composite ``||`` keys (Phase 136) are split and each part checked; keys RDKit
    cannot parse (e.g. OPSIN radical-prefix forms) are skipped (not the lint target)."""
    bad = []
    for k in dictobj:
        if not isinstance(k, str):
            continue
        for part in (k.split("||") if "||" in k else [k]):
            m = Chem.MolFromSmiles(part)
            if m is None:
                continue
            if Chem.MolToSmiles(m) != part:
                bad.append(part)
    return sorted(set(bad))


# Pre-existing non-canonical RETAINED_NAMES keys (BBR-HYG/D-11 baseline, 2026-06-05).
# Re-canonicalizing these is a TRACKED FOLLOW-ON (behaviour-changing). New keys must
# be canonical; this set may only shrink.
_KNOWN_NONCANONICAL_RETAINED = frozenset({
    "CC(=O)N", "OC(=O)C(N)Cc1ccccc1", "OC(=O)C(O)=O", "OC(=O)CC(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", "OC(=O)CCC(=O)O", "OC(=O)CCCC(=O)O",
    "OC(=O)CCCCC(=O)O", "c1cc[nH]n1", "c1ccno1", "c1ccsn1", "c1cnc2ccccc2n1",
    "c1cnc[nH]1", "c1cnco1", "c1cncs1", "c1ncnc2[nH]cnc12",
})


@pytest.mark.unit
def test_fused_heterocycle_keys_canonical():
    """HARD: every FUSED_HETEROCYCLE_DATA key is the RDKit canonical SMILES."""
    from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    assert _noncanonical(FUSED_HETEROCYCLE_DATA) == []


@pytest.mark.unit
def test_ion_retained_keys_canonical():
    """HARD: the ion retained-name dicts have only canonical SMILES keys."""
    from orthonym.data.ion_retained_names import RETAINED_ANIONS, RETAINED_CATIONS
    assert _noncanonical(RETAINED_ANIONS) == []
    assert _noncanonical(RETAINED_CATIONS) == []


@pytest.mark.unit
def test_retained_names_noncanonical_ratchet():
    """RATCHET: RETAINED_NAMES non-canonical keys must be a SUBSET of the known
    baseline — no NEW non-canonical key may be introduced (the count may only shrink
    as the tracked follow-on re-canonicalizes them)."""
    from orthonym.data import RETAINED_NAMES
    current = set(_noncanonical(RETAINED_NAMES))
    new = current - _KNOWN_NONCANONICAL_RETAINED
    assert new == set(), (
        f"NEW non-canonical RETAINED_NAMES keys introduced: {sorted(new)}. "
        f"All new SMILES dict keys MUST be RDKit-canonical (use Chem.MolToSmiles). "
        f"If a correction shrinks the baseline, update _KNOWN_NONCANONICAL_RETAINED."
    )


@pytest.mark.unit
def test_lint_detects_a_planted_bad_key():
    """The lint logic itself flags a non-canonical key (self-check)."""
    assert _noncanonical({"OC(=O)C(O)=O": "x"}) == ["OC(=O)C(O)=O"]
    assert _noncanonical({"O=C(O)C(=O)O": "x"}) == []  # canonical -> clean
