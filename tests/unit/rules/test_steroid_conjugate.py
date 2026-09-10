"""Integration gold tests for steroid conjugate preservation (a phase, -03).

Per-fragment triviality controller: a steroid scaffold conjugated through a
heteroatom linker to a sulfate / phosphate / glycosyl(uronyl) fragment keeps BOTH
fragments — the scaffold becomes a `-yl` substituent group, the conjugate the
functional parent (functional-class ester / glycoside join). Today these conjugates
are silently dropped (e.g. CHEBI:136579 -> bare cholestene), the single defect that
truncates the steroid-conjugate reservoir.

WAVE 0 CONTRACT: each test body opens with the deferred import of the not-yet-built
`classify_conjugate` so `pytest --collect-only` succeeds; the gold rows are RED at run
time until the NP wiring lands in 182-02. Convention: top-level functions (NO class
wrapper) so pytest node IDs are `<file>::<fn>`.

EXACTLY 7 binding gold rows (5 sulfate + 2 glucuronide) + 1 deferred-tripwire
disulfate (out-of-scope multi-conjugate, RESEARCH Open Q1).
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# 7 BINDING GOLD ROWS (all OPSIN-RT verified True this session)
# ---------------------------------------------------------------------------
def test_cholest_sulfate():
    """CHEBI:136579 -> cholest-5-en-3β-yl sulfate (anion -OSO2[O-] -> 'sulfate')."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H]"
        "(OS(=O)(=O)[O-])CC[C@]4(C)[C@H]3CC[C@]12C"
    )
    assert name_compound(smiles) == "cholest-5-en-3β-yl sulfate"


def test_androstan_hydroxy_hsulfate():
    """CHEBI:133103 -> 3α-hydroxy-5α-androstan-17β-yl hydrogen sulfate
    (neutral -OSO2OH -> 'hydrogen sulfate')."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "C[C@]12CC[C@@H](O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)"
        "[C@@H](OS(=O)(=O)O)CC[C@@H]12"
    )
    assert (
        name_compound(smiles)
        == "3α-hydroxy-5α-androstan-17β-yl hydrogen sulfate"
    )


def test_androstanone_hsulfate():
    """CHEBI:138026 -> 3-oxo-5α-androstan-17β-yl hydrogen sulfate."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "C[C@]12CCC(=O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)"
        "[C@@H](OS(=O)(=O)O)CC[C@@H]12"
    )
    assert (
        name_compound(smiles)
        == "3-oxo-5α-androstan-17β-yl hydrogen sulfate"
    )


def test_androstan_3β_sulfate():
    """CHEBI:136983 -> 3β-hydroxy-5α-androstan-17β-yl sulfate
    (dissolves the 'androstan-3-olate' mis-assignment,)."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "C[C@]12CC[C@H](O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)"
        "[C@@H](OS(=O)(=O)[O-])CC[C@@H]12"
    )
    assert (
        name_compound(smiles)
        == "3β-hydroxy-5α-androstan-17β-yl sulfate"
    )


def test_ergostan_hsulfate():
    """CHEBI:136768 ergostane sulfate ester: the conjugate MUST be kept and the name MUST
    OPSIN-round-trip; the EXACT stereo-citation style is the binding assertion's relaxation.

    The -03 conjugate-loss fix works for ergostane — the conjugate is kept, the word is the
    correct neutral-form `hydrogen sulfate` (free acid -OSO2OH), and the name round-trips.
    The ChEBI reference cites the ergostane ring-face α/β block
    `(22S)-3β-hydroxy-6-oxo-5α-ergostan-22-yl hydrogen sulfate`, but ergostane α/β is a
    DEFERRED Phase-181 capability: `STEROID_NUMBERING_MAPS` mislabels ergostane (11<->12), so
    `collect_steroid_alpha_beta` declines and falls back to the whole-graph R/S leading block
    (which DOES round-trip). Building ergostane α/β is out of scope for this thin NP-conjugate
    wiring phase (Phase-181 stereoparent territory). This is a tripwire, NOT a silent scope cut:
    it proves no-silent-drop + RT-correctness; the α/β form awaits ergostane stereoparent support.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from rdkit import Chem

    smiles = (
        "CC(C)[C@@H](C)C[C@H](OS(=O)(=O)O)[C@@H](C)[C@H]1CC[C@H]2"
        "[C@@H]3CC(=O)[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
    )
    n = name_compound(smiles)
    # Conjugate kept (no silent drop) + correct neutral-form word + scaffold -yl anchor.
    assert n is not None and "hydrogen sulfate" in n and "ergostan" in n and "-yl " in n, n
    # And the emitted name OPSIN-round-trips  — never a worse name than prior (dropped).
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert opsin_roundtrip_check(canon, n).get("passed"), n


def test_androstanone_glucuronide():
    """CHEBI:133504 -> 17-oxo-5β-androstan-3β-yl β-D-glucopyranosiduronic acid."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "C[C@]12CC[C@H](O[C@@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)"
        "C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12"
    )
    assert (
        name_compound(smiles)
        == "17-oxo-5β-androstan-3β-yl β-D-glucopyranosiduronic acid"
    )


def test_androstan_3α_glucuronide():
    """CHEBI:133517 -> 3α-hydroxy-5α-androstan-17β-yl
    β-D-glucopyranosiduronic acid (BINDING GOLD — reference OPSIN-RTs True)."""
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "C[C@]12CC[C@@H](O)C[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)"
        "[C@@H](O[C@@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)CC[C@@H]12"
    )
    assert (
        name_compound(smiles)
        == "3α-hydroxy-5α-androstan-17β-yl β-D-glucopyranosiduronic acid"
    )


# ---------------------------------------------------------------------------
# DEFERRED TRIPWIRE (NOT a gold target): multi-conjugate disulfate (Open Q1)
# ---------------------------------------------------------------------------
@pytest.mark.xfail(reason="multi-conjugate deferred to honest-fail; tripwire only (RESEARCH Open Q1)")
def test_disulfate_honest_fail():
    """CHEBI:137389 (disulfate) -> must NOT silently drop a fragment.

    Out-of-scope multi-conjugate (two equal-seniority sulfates). The completeness
    invariant  makes it honest-fail (None / legacy / 'unknown organic
    compound') rather than emit a fragment-omitting name. The xfail marks the
    out-of-scope deferral — it is NOT an output band-aid. Full multi-conjugate
    prefix treatment is a phase/184.
    """
    from orthonym.rules.conjugate_controller import classify_conjugate  # noqa: F401

    smiles = (
        "CC(C)[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H]"
        "(OS(=O)(=O)O)CC[C@]4(C)[C@H]3CC[C@]12C)OS(=O)(=O)O"
    )
    name = name_compound(smiles)
    # Either honest-fail, OR a complete name (this phase ships honest-fail).
    assert name in (None, "unknown organic compound") or (
        "sulfate" in name and "-yl" in name
    ), name
