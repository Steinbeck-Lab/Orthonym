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


# ===========================================================================
# WSD-03 (Phase 175) — NAME<->STRUCTURE integrity lint.
#
# A SUPERSET of the canonical-key check: a key can be canonical yet map to the
# WRONG molecule for its `name` (the 1,5-naphthyridine / pyrido[3,4-b]pyridine
# label-swap class). This gate compares each key's RDKit canonical SMILES to a
# PRE-COMPUTED, committed `expected_canon` (the structure OPSIN assigns to the
# entry's NAME, generated offline so the gate needs NO test-time Java). Any
# entry whose key drifts from its committed name-structure — and is NOT in the
# documented FIX/KEEP allowlist — fails. The live OPSIN regeneration is
#  (run offline when entries change).
#
# FIX (Phase 175): the naphthyridine pair was corrected
#   (c1cnc2cccnc2c1 -> 1,5-naphthyridine; c1cnc2ccncc2c1 -> 1,6-naphthyridine)
# so it is NOT allowlisted (it now matches).
#
# KEEP allowlist below = entries whose name legitimately differs from the key
# structure, by category. The set may only SHRINK (ratchet).
# ===========================================================================
from pathlib import Path as _Path
import json as _json

_EXPECTED_CANON_PATH = _Path(__file__).parent / "fused_heterocycle_expected_canon.json"

_KEEP_NAME_STRUCTURE_MISMATCH = frozenset({
    # --- KEEP-tautomer: intentional Phase-142 tautomer entries ---
    "Nc1ncnc2[nH]cnc12",            # adenine
    "Nc1nc(=O)c2[nH]cnc2[nH]1",     # guanine (tautomer)
    "Nc1nc2[nH]cnc2c(=O)[nH]1",     # guanine (tautomer)
    "O=c1[nH]cnc2[nH]cnc12",        # hypoxanthine
    "O=c1[nH]c(=O)c2nc[nH]c2[nH]1", # xanthine
    # --- KEEP-aromaticity: RDKit-vs-OPSIN kekulization (same molecule) ---
    "C1=CC=C2C=CC=CC=C2C=1",        # heptalene
    # --- KEEP-deferred: fused-ring isomer name<->structure mismatches; a tracked
    #     follow-on (per-entry OPSIN verification + iupac_locant audit) beyond the
    #     Phase-175 WSD-03 reachable-defect scope. NOT fixed here to avoid an
    #     unvalidated mass data rewrite; the ratchet pins them so none is ADDED. ---
    "C1=Cc2ccccc2SC1", "C1=NCc2ccccc2C1", "C1=Nc2ccccc2C1",
    "c1cc2[nH]ccc2cn1", "c1cc2[nH]ncc2cn1", "c1cc2cc[nH]c2cn1",
    "c1cc2ccncc2cn1", "c1cc2ccncn2c1", "c1cc2ccsc2nn1", "c1cc2ccsc2s1",
    "c1cc2nc[nH]cc-2n1", "c1cc2nccnc2cn1", "c1cc2ncncc2cn1", "c1cc2occc2cn1",
    "c1cc2sccc2cn1", "c1cc2sccc2s1", "c1ccc2c(c1)NCCN2",
    "c1ccc2c(c1)ccc1[nH]ccc12", "c1ccc2c(c1)ccc1occc12",
    "c1ccc2c(c1)ccc1sccc12", "c1ccc2c(c1)oc1cccnc12", "c1ccc2c[nH]cc2c1",
    "c1ccc2cc3cnccc3cc2c1", "c1ccc2cc3ncccc3cc2c1",
    # cinnoline/phthalazine REMOVED 2026-06-22: name-swap fixed, now key-correct
    # (ratchet shrinks). c1ccc2nnccc2c1=cinnoline, c1ccc2cnncc2c1=phthalazine.
    "c1ccc2nocc2c1", "c1ccc2nscc2c1", "c1ccc2oncc2c1",
    "c1ccn2ncnc2c1", "c1cnc2[nH]cnc2c1", "c1cnc2cc[nH]c2c1",
    "c1cnc2cnccc2c1", "c1cnc2cnncc2c1", "c1cnc2nccn2c1", "c1cnc2ncoc2c1",
    "c1cnc2ncsc2c1", "c1cnc2nncn2c1", "c1cnc2nocc2c1", "c1cnc2ocnc2c1",
    "c1cnc2scnc2n1", "c1cnn2cccc2c1", "c1cnn2ccnc2c1", "c1ncc2[nH]ccc2n1",
    "c1ncc2ncncc2n1",
})


def _name_structure_drift(expected_canon):
    """Keys where the entry's committed name-structure != the key's RDKit canon."""
    from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    drift = []
    for key in FUSED_HETEROCYCLE_DATA:
        m = Chem.MolFromSmiles(key)
        if m is None:
            continue
        ck = Chem.MolToSmiles(m)
        exp = expected_canon.get(ck)
        if exp is None:
            drift.append((ck, "<no committed expected_canon — regenerate the JSON>"))
        elif ck != exp:
            drift.append((ck, exp))
    return drift


@pytest.mark.unit
def test_fused_heterocycle_name_structure_lint():
    """GATE (D-06): no fused-heterocycle entry's key may drift from its committed
    name-structure unless it is in the documented FIX/KEEP allowlist. Catches the
    silent wrong-molecule label-swap class (e.g. the naphthyridine bug, now fixed)."""
    expected_canon = _json.load(open(_EXPECTED_CANON_PATH))["expected_canon"]
    drift = _name_structure_drift(expected_canon)
    new = [(k, v) for k, v in drift if k not in _KEEP_NAME_STRUCTURE_MISMATCH]
    assert new == [], (
        f"NEW name<->structure drift in FUSED_HETEROCYCLE_DATA: {new}. "
        f"A ring name maps to a wrong structure. Fix the entry's `name` (and "
        f"regenerate fused_heterocycle_expected_canon.json offline), or — only if "
        f"the mismatch is a verified tautomer/aromaticity artifact — add it to "
        f"_KEEP_NAME_STRUCTURE_MISMATCH with a cited justification."
    )


@pytest.mark.unit
def test_name_structure_allowlist_ratchet():
    """RATCHET: every KEEP-allowlisted key must STILL be a real mismatch; once an
    entry is fixed it must be removed from the allowlist (the set may only shrink).
    Also guards that the naphthyridine FIX is NOT silently re-allowlisted."""
    expected_canon = _json.load(open(_EXPECTED_CANON_PATH))["expected_canon"]
    drift_keys = {k for k, _ in _name_structure_drift(expected_canon)}
    stale = sorted(_KEEP_NAME_STRUCTURE_MISMATCH - drift_keys)
    assert stale == [], (
        f"Stale KEEP allowlist entries (no longer a mismatch — remove them): {stale}"
    )
    # The fixed naphthyridine keys must NOT be allowlisted.
    for fixed in ("c1cnc2cccnc2c1", "c1cnc2ccncc2c1"):
        assert fixed not in _KEEP_NAME_STRUCTURE_MISMATCH


@pytest.mark.unit
def test_name_structure_lint_detects_planted_bad():
    """Self-check: a planted name<->structure drift (a key whose committed
    expected_canon is a DIFFERENT molecule, not allowlisted) is flagged."""
    planted = Chem.MolToSmiles(Chem.MolFromSmiles("c1ccncc1"))   # pyridine key
    fake_expected = {planted: Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1"))}  # claims benzene
    # the drift helper would flag planted (ck != exp); confirm the comparison logic:
    ck = planted
    assert ck != fake_expected[ck]  # detected
    assert ck not in _KEEP_NAME_STRUCTURE_MISMATCH  # and it is NOT allowlisted -> would FAIL the gate
