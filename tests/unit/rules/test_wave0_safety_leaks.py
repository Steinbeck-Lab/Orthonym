"""Wave 0 — accuracy-first safety: the constitutional self-consistency
gate must not let a name for the WRONG structure through.

Empirical verification (2026-07-01) of the GAP-REPORT b/c leak list found that
almost all listed "leaks" were either already fail-closed at production HEAD, or
audit structure-reading errors (the emitted name was actually correct). The one
genuine remaining leak: the hydroperoxide anion [O-]O was named 'dioxidane'
(neutral OO) — 's InChIKey-skeleton comparison EXCLUDES the charge layer,
so a charge-dropping name shared the skeleton and passed. Fix: the verdict also
compares net formal charge.

These tests target the pure verdict function directly (deterministic, no OPSIN
subprocess / gate-config dependence — the end-to-end fail-close is covered by the
 PIN-conformance gate which runs with all gates enabled).
"""
from orthonym.namer import (
    _self_consistency_verdict,
    _self_consistency_net_charge,
    name_compound,
)


# --- The fix: charge-drop is a constitutional mismatch --------------------------

def test_charge_drop_is_a_mismatch():
    # [O-]O (HOO-, charge -1) vs OO ('dioxidane', charge 0): same InChIKey skeleton,
    # different net charge -> must be flagged mismatch (was 'ok' before the fix).
    assert _self_consistency_net_charge("[O-]O") == -1
    assert _self_consistency_net_charge("OO") == 0
    assert _self_consistency_verdict("[O-]O", "OO") == "mismatch"


def test_skeleton_mismatch_still_caught():
    # heptalene (C10) vs benzene (C6) — different skeleton, unchanged behavior.
    assert _self_consistency_verdict("C1=CC=CC2=CC=CC=CC=C12", "c1ccccc1") == "mismatch"


def test_neutral_input_exempt_from_charge_check():
    # Policy reversed by baeb8d1ab ("a name must reproduce the input's charge form...
    # compares the net charge and the protonation flag for every input"): a neutral input
    # is NOT exempt from the charge check. OPSIN reads 'methyl phosphate' as the dianion
    # (OP(=O)(O)OC is the neutral ester, whose name is 'methyl dihydrogen phosphate',
    # "Esters of mononuclear noncarbon oxoacids", the Blue Book
    # 'P(O)(O-CH3)(OH)2 methyl dihydrogen phosphate (PIN)'), so a name that parses to the
    # dianion denotes a different species and the gate must reject it.
    assert _self_consistency_verdict("OP(=O)(O)OC", "P(=O)(OC)([O-])[O-]") == "mismatch"


# --- No over-suppression: same constitution+charge is still "ok" ---------------

def test_charge_check_does_not_oversuppress():
    # Legitimate charged species (charge preserved by the name) stay ok.
    assert _self_consistency_verdict("CC(=O)[O-]", "CC(=O)[O-]") == "ok"
    assert _self_consistency_verdict("[NH4+]", "[NH4+]") == "ok"
    assert _self_consistency_verdict("C[N+](C)(C)C", "C[N+](C)(C)C") == "ok"


def test_stereo_only_difference_is_ok():
    # Policy: since 0c4d4a2d3 ("reject stereo OMISSION as a mismatch (0-wrong; 20/500 default
    # stereo-drops)") a parse that carries LESS stereo than the input defines is a
    # mismatch -- the name describes a less specific, different molecule. Stereo that is
    # identical on both sides is still fine, and the stereo carve-out (ignore_stereo=True,
    # a constitution-only comparison) stays tolerant.
    assert _self_consistency_verdict("C[C@H](O)CC", "CC(O)CC") == "mismatch"
    assert _self_consistency_verdict("C[C@H](O)CC", "C[C@H](O)CC") == "ok"
    assert _self_consistency_verdict("C[C@H](O)CC", "CC(O)CC", ignore_stereo=True) == "ok"


# --- Audit mis-reads: these are CORRECT names (must NOT be fail-closed) ---------
# These ship regardless of gate config (skeleton+charge match / ungated path).

def test_indene_substituent_is_correct_not_a_leak():
    # GAP-REPORT b called this a dihydronaphthalene leak, but the ring is [5,6]
    # (indene) and the name round-trips exactly to the input — it is correct.
    assert name_compound("OC(=O)CCC1=CCc2ccccc21", style="pin") == "3-(1H-inden-3-yl)propanoic acid"


def test_methylestrane_is_correct_not_a_leak():
    # Ring sizes [5,6,6,6] (estrane skeleton); audit mislabeled it "D-homo steroid".
    # The input defines no stereochemistry, so the stereoparent name 'estrane' is not
    # emitted: "Stereochemical configuration of parent structures"
    # (the Blue Book) "The name of a fundamental parent structure usually implies the
    # absolute configuration of all chirality centers", and OPSIN reads '5-methylestrane' as
    # a molecule with four defined stereocentres (full an InChIKey,
    # the input's is an InChIKey), so that name fabricates configuration
    # (rejects it since be9f7cabc). The stereo-free input gets the fusion name, which
    # OPSIN parses to the input's full InChIKey.
    assert name_compound("CC12CCC3C(CCC4(CCCCC34)C)C1CCC2", style="pin") == (
        "5,13-dimethylhexadecahydro-1H-cyclopenta[a]phenanthrene")
