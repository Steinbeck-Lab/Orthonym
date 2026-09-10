"""The atom-coverage validator must prove STRUCTURE, not compare counts.

 residue, Task X. ``validation/atom_coverage.py`` scored a name by
``min(parsed_heavy / total_heavy, 1.0)`` and called it complete at >= 0.80.
Three independent failures followed, and each has a test here:

  1. the clamp made atom GAIN structurally invisible -- a name that invents
     atoms could never score below 1.0;
  2. the 0.80 threshold passed a genuine atom DROP (0.833 -> ``is_complete``);
  3. a count cannot distinguish *the same atoms* from *the same number of
     atoms*, so ``CNCC(=O)O`` (sarcosine) named ``2-aminopropanoic acid``
     (alanine) passed on an identical element multiset.

Every expectation below is grounded in observed OPSIN 2.9.0 output, recorded
in the case docstrings -- none is a guess. Asserted at the producer
(``validate_atom_coverage``), never through a consumer.
"""

import shutil

import pytest
from rdkit import Chem

from orthonym.validation.atom_coverage import (
    find_opsin_jar,
    validate_atom_coverage,
)

_OPSIN_JAR = find_opsin_jar()
_OPSIN_AVAILABLE = _OPSIN_JAR is not None and shutil.which("java") is not None

needs_opsin = pytest.mark.skipif(
    not _OPSIN_AVAILABLE, reason="OPSIN JAR or Java not available"
)


def _cov(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    return validate_atom_coverage(mol, name, opsin_jar=_OPSIN_JAR)


# ---------------------------------------------------------------------------
# 1. atom DROP must not be called complete
# ---------------------------------------------------------------------------


@needs_opsin
def test_dropped_carbon_is_not_complete():
    """``CS(=O)(=O)NC`` -> ``methanesulfonamide`` drops the N-methyl carbon.

    OPSIN parses the name to ``CS(N)(=O)=O`` -- 5 heavy atoms against the
    input's 6, i.e. ratio 0.833. The old 0.80 threshold shipped this as
    ``is_complete=True``.
    """
    r = _cov("CS(=O)(=O)NC", "methanesulfonamide")
    assert r.method == "parse_back"
    assert r.total_heavy_atoms == 6
    assert r.parsed_heavy_atoms == 5
    assert r.unclaimed_atoms == 1, "the N-methyl carbon is unaccounted for"
    assert r.is_complete is False, "a dropped atom is never complete coverage"


# ---------------------------------------------------------------------------
# 2. atom GAIN must be visible at all
# ---------------------------------------------------------------------------


@needs_opsin
def test_invented_nitrogen_is_visible_and_not_complete():
    """``CNCC(=O)N`` -> ``2-amino-2-(methylamino)acetamide`` invents an N.

    OPSIN parses to ``CNC(N)C(N)=O``: C3 N3 O1 (7 heavy) against the input's
    C3 N2 O1 (6 heavy). Every input atom has a counterpart, so any
    coverage ratio is 1.0 -- the excess is only visible as ``extra_atoms``.
    Under the clamp this case scored a clean 1.000 / ``is_complete=True``.
    """
    r = _cov("CNCC(=O)N", "2-amino-2-(methylamino)acetamide")
    assert r.method == "parse_back"
    assert r.total_heavy_atoms == 6
    assert r.parsed_heavy_atoms == 7
    assert r.extra_atoms == 1, "the fabricated nitrogen must be counted"
    assert r.is_complete is False, "a name that invents an atom is not complete"


@needs_opsin
def test_gain_is_recoverable_from_the_result_not_hidden_by_a_clamp():
    """``parsed_heavy_atoms`` records the raw parse-back count, unclamped.

    The defect was that ``min(..., 1.0)`` destroyed the information before
    the caller ever saw it. A caller must be able to reconstruct the
    unclamped count ratio from the result.
    """
    r = _cov("CNCC(=O)N", "2-amino-2-(methylamino)acetamide")
    assert r.parsed_heavy_atoms > r.total_heavy_atoms
    assert r.parsed_heavy_atoms / r.total_heavy_atoms > 1.0


# ---------------------------------------------------------------------------
# 3. the core claim -- a count is not a structure
# ---------------------------------------------------------------------------


@needs_opsin
def test_same_formula_different_constitution_is_rejected():
    """Sarcosine ``CNCC(=O)O`` named as alanine ``2-aminopropanoic acid``.

    These are different molecules with the SAME molecular formula and the
    SAME heavy-atom element multiset (C3 N1 O2, 6 heavy atoms):
      input InChIKey FSYKKLYZXJSNPZ-UHFFFAOYSA-N
      parsed InChIKey QNAYBMKLOCPYGJ-UHFFFAOYSA-N
    No count-based and no formula-based check can separate them; only
    constitution can. This is the test that a formula comparison fails.
    """
    r = _cov("CNCC(=O)O", "2-aminopropanoic acid")
    assert r.method == "parse_back"
    assert r.total_heavy_atoms == r.parsed_heavy_atoms == 6
    assert r.extra_atoms == 0 and r.unclaimed_atoms == 0, (
        "counts agree exactly -- that is precisely the trap"
    )
    assert r.coverage_ratio == 1.0, "the ratio cannot see this defect"
    assert r.constitution_match is False
    assert r.is_complete is False, "different molecule, however the counts fall"


@needs_opsin
def test_completeness_is_not_a_function_of_the_ratio():
    """No threshold on ``coverage_ratio`` can reproduce ``is_complete``.

    Two rows, both with ``coverage_ratio == 1.0``, disagree on completeness.
    Any predicate of the form ``ratio >= t`` must therefore be wrong for one
    of them -- which is the structural argument against the 0.80 rule.
    """
    good = _cov("CCO", "ethanol")
    bad = _cov("CNCC(=O)O", "2-aminopropanoic acid")
    assert good.coverage_ratio == bad.coverage_ratio == 1.0
    assert good.is_complete is True
    assert bad.is_complete is False


# ---------------------------------------------------------------------------
# 3b. coverage is counted PER ELEMENT, and the ratio is not a bare count
# ---------------------------------------------------------------------------


@needs_opsin
def test_element_substitution_is_not_covered():
    """``CCO`` named ``ethylamine``: 3 heavy atoms on both sides, 2 shared.

    OPSIN parses ``ethylamine`` to ``CCN``. A bare count says 3 of 3 and
    calls the oxygen covered by the nitrogen; only a per-ELEMENT multiset
    intersection sees that the O is unaccounted for and the N is invented.
    Added after mutation testing: mutants that restored the bare count, and
    that restored the clamped ``min(parsed/total, 1.0)`` ratio, both survived
    the suite without this case.
    """
    r = _cov("CCO", "ethylamine")
    assert r.method == "parse_back"
    assert r.total_heavy_atoms == 3 and r.parsed_heavy_atoms == 3
    assert r.claimed_atoms == 2, "only the two carbons are shared"
    assert r.unclaimed_atoms == 1, "the oxygen has no counterpart"
    assert r.extra_atoms == 1, "the nitrogen is invented"
    assert r.coverage_ratio == pytest.approx(2 / 3), (
        "a bare count, clamped or not, would report 1.0 here"
    )
    assert r.is_complete is False


# ---------------------------------------------------------------------------
# 3c. no constitution evidence => no completeness claim
# ---------------------------------------------------------------------------


@needs_opsin
def test_inchi_failure_is_fail_closed(monkeypatch):
    """If InChI cannot be generated, the verdict must be 'not complete'.

    The hazard is specific: a helper that returned some CONSTANT on failure
    would make both sides compare equal and pass everything. Added after
    mutation testing -- a mutant doing exactly that survived, as did one
    that reported the no-InChI exit as complete.
    """
    from orthonym.validation import atom_coverage as mod

    def boom(*a, **kw):
        raise RuntimeError("InChI unavailable")

    monkeypatch.setattr(mod.Chem, "MolToInchiKey", boom)

    assert mod._inchikey(Chem.MolFromSmiles("CCO")) == "", (
        "a failed InChI must yield the empty string, never a sentinel key"
    )
    r = _cov("CCO", "ethanol")
    assert r.method == "parse_back_no_inchi"
    assert r.is_complete is False, "no evidence is not evidence of completeness"
    assert r.constitution_match is False


# ---------------------------------------------------------------------------
# 4. the exact case must keep working (CLAUDE.md #9 -- do not unmask worse)
# ---------------------------------------------------------------------------


@needs_opsin
def test_exact_match_is_complete():
    """``CCO`` -> ``ethanol`` round-trips to the identical InChIKey."""
    r = _cov("CCO", "ethanol")
    assert r.method == "parse_back"
    assert (r.total_heavy_atoms, r.claimed_atoms, r.unclaimed_atoms) == (3, 3, 0)
    assert r.extra_atoms == 0
    assert r.coverage_ratio == 1.0
    assert r.constitution_match is True
    assert r.is_complete is True


@needs_opsin
def test_inchikeys_are_recorded_for_both_sides():
    """The evidence for the verdict must be inspectable by the caller."""
    r = _cov("CCO", "ethanol")
    assert r.input_inchikey.startswith("LFQSCWFLJHTTHZ")
    assert r.parsed_inchikey.startswith("LFQSCWFLJHTTHZ")


# ---------------------------------------------------------------------------
# 5. declared scope: constitution, NOT stereochemistry
# ---------------------------------------------------------------------------


@needs_opsin
def test_stereo_omission_is_out_of_scope_and_still_complete():
    """A stereo-blind name still covers every atom.

    ``C[C@H](N)C(=O)O`` -> ``2-aminopropanoic acid`` parses to the same
    constitution (QNAYBMKLOCPYGJ) and differs only in the stereo block
    (REOHCLBHSA vs UHFFFAOYSA). This validator answers "are these the same
    atoms, bonded the same way"; stereo is a different channel and is
    deliberately not judged here. Both full keys are exposed so a stricter
    caller can compare them.
    """
    r = _cov("C[C@H](N)C(=O)O", "2-aminopropanoic acid")
    assert r.constitution_match is True
    assert r.is_complete is True
    assert r.input_inchikey != r.parsed_inchikey, "stereo does differ"


# ---------------------------------------------------------------------------
# 6. the fallback stays fail-closed (explicitly NOT to be "fixed")
# ---------------------------------------------------------------------------


def test_unparseable_name_is_fail_closed():
    """OPSIN unavailable / parse failure -> 0.0, incomplete, 'unavailable'."""
    mol = Chem.MolFromSmiles("CCO")
    r = validate_atom_coverage(mol, "ethanol", opsin_jar="/nonexistent/path.jar")
    assert r.method == "unavailable"
    assert r.coverage_ratio == 0.0
    assert r.is_complete is False
    assert r.constitution_match is False


@needs_opsin
def test_garbage_name_is_fail_closed():
    """A name OPSIN cannot parse must never be reported as complete."""
    mol = Chem.MolFromSmiles("CCO")
    r = validate_atom_coverage(mol, "qqzzxx-not-a-name", opsin_jar=_OPSIN_JAR)
    assert r.method == "unavailable"
    assert r.is_complete is False


# ---------------------------------------------------------------------------
# 7. severe truncation (the pre-existing behaviour, kept)
# ---------------------------------------------------------------------------


@needs_opsin
def test_severe_truncation_is_incomplete():
    """C16 chain named ``methanol``: 1 of 16 atoms matched by element."""
    r = _cov("CCCCCCCCCCCCCCCC", "methanol")
    assert r.total_heavy_atoms == 16
    assert r.coverage_ratio < 0.80
    assert r.is_complete is False
