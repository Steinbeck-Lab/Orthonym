"""Pinned witness set for M2.3 — best-effort relative cis/trans on a
ring-as-substituent.

These six molecules all ABSTAIN at best-effort today (2026-09-01, HEAD
``): the general engine reaches a valid *constitution* but the
stereo-composition emits an OPSIN-unparseable PSEUDOASYMMETRIC descriptor
(e.g. ``(1r,4R)-`` / ``(1s,3R)-``) on the 1,3-cyclobutane or 1,4-cyclohexane
ring substituent, so the whole candidate fails round-trip and is suppressed.
M2.3 will instead emit an OPSIN-parseable *relative* ``cis``/``trans`` on that
ring, reclaiming each at 0-wrong.

Each ``ref_name`` below is a reference name (it
is the acceptance ORACLE, not shipped output). Every ref_name was pre-verified
in a fresh process:

  (a) Orthonym abstains at best-effort today
      (``-m orthonym <smi> --emit-tier best-effort --provenance`` -> tier
      ``abstain``, name ``None``);
  (b) the ref_name contains a ``cis``/``trans`` WORD and FULL-InChIKey
      round-trips through OPSIN 2.9.0 to the input (stereo layer included);
  (c) with ``ORTHONYM_BE_STRIP_STEREO=1`` the constitution renders to a
      round-trip-verified systematic name -> **stereo is the SOLE blocker**,
      i.e. M2.3's descriptor fix is sufficient to reclaim it (rejected two
      otherwise-valid candidates whose stripped form was still ``UNNAMEABLE``,
      needing composition-reach beyond stereo).

Full method + the liveness call-counts proving the wiring sites are on-path:
`internal notes`.

Provenance of the SMILES: best-effort-abstain cohort; the six are
inlined as literals here (NIT 8 — never import the volatile /tmp reclaim file).
This is a FIXTURE module (imported by M2.3 tests); it asserts only that every
SMILES parses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Witness:
    wid: str
    smiles: str
    ref_name: str          # reference cis/trans name (acceptance oracle)
    ring: str              # ring carrying the relative descriptor
    descriptor_kind: str   # descriptors present in ref_name
    note: str              # one-line provenance / role


WITNESSES: List[Witness] = [
    # --- 3 x monocyclic-1,3 (cyclobutane, cis/trans-3-…) ---------------------
    Witness(
        wid="W1",
        smiles="C[C@H](CNC(=O)[C@H]1CCC(=O)N1C)CC(=O)N[C@H]1C[C@@H](F)C1",
        ref_name="(2R)-N-((2S)-4-((cis-3-fluorocyclobutyl)amino)-2-methyl-4-oxobutyl)-1-methyl-5-oxopyrrolidine-2-carboxamide",
        ring="cyclobutane-1,3",
        descriptor_kind="cis + abs(2R,2S)",
        note="amino-linked cyclobutyl; strip->systematic_verified (stereo sole blocker). "
             "Reaches decorated_ring_substituent_name but NOT _prepend_ring_substituent_stereo.",
    ),
    Witness(
        wid="W3",
        smiles="Cn1cc(Br)c(CN[C@H]2C[C@H](NC(=O)c3ccc4[nH][nH]c(=N)c4c3)C2)n1",
        ref_name="3-imino-N-(trans-3-(((4-bromo-1-methyl-1H-pyrazol-3-yl)methyl)amino)cyclobutyl)-1H,2H-indazole-5-carboxamide",
        ring="cyclobutane-1,3",
        descriptor_kind="trans (relative only — no absolute centre)",
        note="pure-relative cyclobutane; strip->systematic_verified (stereo sole blocker).",
    ),
    Witness(
        wid="CB",
        smiles="CN(CCC(=O)NCCN1CCCN(C(=O)[C@H]2C[C@H](F)C2)CC1)S(C)(=O)=O",
        ref_name="3-((methyl)(methylsulfonyl)amino)-N-(2-(4-((trans-3-fluorocyclobutyl)carbonyl)-1,4-diazepan-1-yl)ethyl)propanamide",
        ring="cyclobutane-1,3",
        descriptor_kind="trans (relative only — ring is the ONLY stereo in the molecule)",
        note="cleanest relative-only case; strip->systematic_verified (stereo sole blocker). "
             "Replaces an earlier candidate whose stripped form was still UNNAMEABLE.",
    ),
    # --- 1 x monocyclic-1,4 (cyclohexane, trans-4-…) -------------------------
    Witness(
        wid="W4",
        smiles="C[C@@H]1CN(C(=O)[C@H]2CC[C@H](C)CC2)C[C@@H]1CNC(=O)CCNc1ccccc1",
        ref_name="N-(((3S,4S)-4-methyl-1-((trans-4-methylcyclohexyl)carbonyl)pyrrolidin-3-yl)methyl)-3-(phenylamino)propanamide",
        ring="cyclohexane-1,4",
        descriptor_kind="trans + abs(3S,4S)",
        note="1,4-cyclohexyl ring substituent; strip->systematic_verified (stereo sole blocker).",
    ),
    # --- 2 x "both": relative ring + an independent absolute centre ----------
    Witness(
        wid="W5",
        smiles="C=CCO[C@H](C)C(=O)N[C@H]1CC[C@H](CN(C)C(=O)c2ccccc2C2CC2)CC1",
        ref_name="(2R)-2-((prop-2-en-1-yl)oxy)-N-(trans-4-((N-methyl-2-cyclopropylbenzamido)methyl)cyclohexyl)propanamide",
        ring="cyclohexane-1,4",
        descriptor_kind="abs(2R) + trans  (needs BOTH R/S and cis/trans in one name)",
        note="independent (2R) chain centre + trans-4-cyclohexyl ring; strip->systematic_verified.",
    ),
    Witness(
        wid="W6",
        smiles="COCC(=O)NCCC(=O)N[C@@H]1CCCN(C(=O)[C@H]2C[C@H](S(C)(=O)=O)C2)[C@@H]1C",
        ref_name="3-(2-methoxyacetamido)-N-((2R,3R)-2-methyl-1-((trans-3-(methylsulfonyl)cyclobutyl)carbonyl)piperidin-3-yl)propanamide",
        ring="cyclobutane-1,3",
        descriptor_kind="abs(2R,3R) + trans  (needs BOTH R/S and cis/trans in one name)",
        note="independent (2R,3R) piperidine centres + trans-3-cyclobutyl ring; strip->systematic_verified.",
    ),
]

# Verified SPARE (pre-verify PASSED but strip->UNNAMEABLE, so NOT a clean M2.3
# target — recorded for the audit trail, not part of the six):
# CC[C@@H](CC(=O)N[C@H]1C[C@H](CNC(=O)[C@@H]2C[C@H](F)CN2C)C1)CC(F)(F)F
# -> (3S)-3-ethyl-5,5,5-trifluoro-N-(trans-3-(((2S,4S)-4-fluoro-1-methylpyrrolidine-2-carboxamido)methyl)cyclobutyl)pentanamide


def test_witnesses_parse():
    """Every pinned witness SMILES must parse (real assertion; guards the literals)."""
    from rdkit import Chem

    assert len(WITNESSES) == 6
    assert len({w.wid for w in WITNESSES}) == 6, "duplicate witness ids"
    for w in WITNESSES:
        mol = Chem.MolFromSmiles(w.smiles)
        assert mol is not None, f"{w.wid}: SMILES does not parse: {w.smiles}"
        assert mol.GetNumAtoms() > 0, f"{w.wid}: empty molecule"
        assert ("cis" in w.ref_name) or ("trans" in w.ref_name), \
            f"{w.wid}: ref_name lacks a cis/trans word"
