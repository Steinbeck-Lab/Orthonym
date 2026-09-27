"""a phase + a phase inheritance: PYTHONHASHSEED
determinism gate for aromatic-NP-scaffold canary rows.

This harness asserts that name_compound produces byte-identical output
across PYTHONHASHSEED ∈ {0, 1, 2, 3, 42} for the canary rows whose
scaffold_class is in _NP_AROMATIC_RING_LOCANTS (i.e., the aromatic-NP-
scaffold subset). Pre- fix, Chem.Kekulize-based detection produced
different Kekulé forms across processes; post- fix, the canonical-
locant table produces deterministic output.

Per internal notes: budget < 2 minutes for 5 seeds × ≥ 11 fixtures = 55
parameterized tests. Any cross-seed drift is a HARD pytest.fail.

For a phase Plan-03-00b shipping, the harness covers ONLY the
estradiol-derivative canary fixture (name_stability_265) because that is
the only aromatic-NP-scaffold row in the canary set whose scaffold_class
is in _NP_AROMATIC_RING_LOCANTS. a phase+ additions to the table will
expand the fixture list (morphinan, ergoline, etc.).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


# Aromatic-NP-scaffold canary fixtures whose scaffold_class IS in
# _NP_AROMATIC_RING_LOCANTS at this commit. Sourced from
# tests/canary/canary_post_blockers_fix_160.1.csv at HEAD.
#
# Each tuple: (fixture_id, smiles, expected_name_substring)
# The expected_name_substring is the IUPAC-canonical form that ALL
# PYTHONHASHSEED values MUST produce (the determinism contract per).
AROMATIC_NP_FIXTURES = [
    # Estradiol-derivative — the canary row at which Blocker B surfaced
    (
        "name_stability_265",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O",
        "1,3,5(10)-trien",
    ),
    # Estrone (aromatic A-ring; ketone at C17)
    (
        "estrone",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CCC2=O",
        "1,3,5(10)-trien",
    ),
    # Estradiol (aromatic A-ring; 3,17-diol)
    (
        "estradiol",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
        "1,3,5(10)-trien",
    ),
    # 17-alpha-estradiol
    (
        "alpha_estradiol",
        "C[C@]12CC[C@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@H]2O",
        "1,3,5(10)-trien",
    ),
    # 17-deoxyestradiol (aromatic A-ring; 3-ol only)
    (
        "deoxyestradiol",
        "C[C@]12CCC3=C(CC[C@@H]4[C@@H]3CCC[C@@H]42)C=CC=C1",
        # Skip exact expected for compounds where scaffold detection is uncertain;
        # this is included for diversity; the assertion checks determinism across
        # seeds rather than specific output.
        None,
    ),
    # Ethinyl estradiol (17-ethinyl-estradiol)
    (
        "ethinyl_estradiol",
        "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C",
        "1,3,5(10)-trien",
    ),
    # 16-alpha-hydroxyestrone
    (
        "16a_hydroxyestrone",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)CC2=O",
        "1,3,5(10)-trien",
    ),
    # 2-methoxyestradiol
    (
        "2_methoxyestradiol",
        "COc1cc2CC[C@H]3[C@@H]4CC[C@@H](O)[C@@]4(C)CC[C@H]3c2cc1O",
        "1,3,5(10)-trien",
    ),
    # 4-hydroxyestradiol
    (
        "4_hydroxyestradiol",
        "Oc1ccc2c(c1O)CC[C@@H]1[C@H]2CC[C@@]2(C)[C@@H]1CC[C@@H]2O",
        "1,3,5(10)-trien",
    ),
    # Mestranol (3-methoxy-17-ethinyl-estradiol)
    (
        "mestranol",
        "COc1ccc2c(c1)CC[C@@H]1[C@@H]2CC[C@@]2(C)[C@@]1(C#C)CC[C@@H]2O",
        "1,3,5(10)-trien",
    ),
    # 2-hydroxyestradiol
    (
        "2_hydroxyestradiol",
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCc1cc(O)c(O)cc13",
        "1,3,5(10)-trien",
    ),
]


# Fixtures whose '1,3,5(10)-trien' steroid name the engine cannot build yet.
# The 17-ethynyl is a decoration the natural-product scaffold producer has no
# finder for: the old outputs '(8R,9S,13S,14S,17R)-estra-1,3,5(10)-triene-3,17-
# diol' / '...-3-methoxyestra-1,3,5(10)-trien-17-ol' DROPPED it (OPSIN 2.9.0 reads
# them as estradiol / 3-O-methylestradiol, C18/C19 for the C20/C21 inputs: a
# different molecule), "SUBSTITUTIVE NOMENCLATURE" (the Blue Book,
# every substituent is cited as a prefix or a suffix). 286a491c8 made the producer
# decline instead; they passed before only when the subprocess inherited a
# gate-OFF environment (TRIAGE C1) or the OPSIN gate timed out and failed open.
# Production abstains at the PIN tier (as it already did at 4e0e5c29b with the
# gate on); best-effort names both RT-exact (von Baeyer). The xfail is strict:
# the substring may only come back as a real, decoration-complete name.
_NEEDS_ETHYNYL_DECORATION = {
    "ethinyl_estradiol": "17-ethynyl estradiol",
    "mestranol": "17-ethynyl 3-methoxyestradiol",
}
_ETHYNYL_XFAIL_REASON = (
    "needs a 17-ethynyl (alkynyl) decoration finder in the NP steroid producer "
    "(rules/natural_products.py; P-15.1 :4720): the old '1,3,5(10)-trien' name "
    "dropped the ethynyl (286a491c8); TODO in TRIAGE.md 'Suite fix -- "
    "j1-regressions'")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
@pytest.mark.parametrize("fixture_id,smi,expected_substring", AROMATIC_NP_FIXTURES)
def test_aromatic_np_scaffold_deterministic_across_pythonhashseed(
    seed, fixture_id, smi, expected_substring, request,
):
    """Per internal notes: ALL aromatic-NP-scaffold canary rows produce
    byte-identical output across PYTHONHASHSEED ∈ {0, 1, 2, 3, 42}.

    The harness runs name_compound in a subprocess with the given
    PYTHONHASHSEED, then compares the output against the expected
    canonical substring (1,3,5(10)-trien for estradiol-class). If the
    substring is None, only cross-seed determinism is asserted
    (output is invariant across seeds even if specific form is uncertain).
    """
    if fixture_id in _NEEDS_ETHYNYL_DECORATION:
        request.applymarker(pytest.mark.xfail(strict=True,
                                              reason=_ETHYNYL_XFAIL_REASON))
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    # Use the worktree src path so we test the amended code.
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["PYTHONPATH"] = os.path.join(cwd, "src")

    cmd = [
        sys.executable, "-c",
        f"from orthonym import name_compound; print(name_compound({smi!r}))",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0, (
        f"name_compound subprocess failed for {fixture_id} at "
        f"PYTHONHASHSEED={seed}:\n"
        f"  stderr: {result.stderr[:500]}"
    )
    name = result.stdout.strip()

    if expected_substring is not None:
        assert expected_substring in name, (
            f"Determinism contract violated for {fixture_id} at "
            f"PYTHONHASHSEED={seed}: expected substring "
            f"{expected_substring!r} in output, got: {name!r}"
        )


@pytest.mark.parametrize("fixture_id,smi,expected_substring", AROMATIC_NP_FIXTURES)
def test_aromatic_np_scaffold_byte_identical_across_seeds(
    fixture_id, smi, expected_substring,
):
    """Per internal notes strict determinism: across all 5 PYTHONHASHSEED
    values {0, 1, 2, 3, 42}, the name_compound output for an aromatic-NP-
    scaffold fixture MUST be byte-identical.

    This is the strongest determinism assertion. Pre- fix, RDKit's
    Chem.Kekulize varied based on PYTHONHASHSEED-amplified atom-iteration
    order, producing different equally-valid Kekulé forms. Post- fix,
    the canonical-locant table bypasses Kekulize for scaffold-aromatic
    atoms, producing deterministic output.
    """
    seeds = [0, 1, 2, 3, 42]
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outputs = {}
    for seed in seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["PYTHONPATH"] = os.path.join(cwd, "src")
        cmd = [
            sys.executable, "-c",
            f"from orthonym import name_compound; print(name_compound({smi!r}))",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=60,
        )
        assert result.returncode == 0, (
            f"name_compound subprocess failed for {fixture_id} at seed {seed}"
        )
        outputs[seed] = result.stdout.strip()

    # All 5 outputs MUST be byte-identical
    distinct = set(outputs.values())
    assert len(distinct) == 1, (
        f"Determinism drift across PYTHONHASHSEED for {fixture_id}: "
        f"{len(distinct)} distinct outputs across {len(seeds)} seeds:\n"
        + "\n".join(
            f"  seed={s}: {o!r}" for s, o in sorted(outputs.items())
        )
    )


@pytest.mark.parametrize("fixture_id", sorted(_NEEDS_ETHYNYL_DECORATION))
def test_ethynyl_steroids_never_ship_a_decoration_dropping_name(fixture_id):
    """0-wrong for the two strict-xfail fixtures above, in production (a fresh
    subprocess, OPSIN gate on): whatever the default tier ships is either the
    failure sentinel or a name that round-trips to the input's full InChIKey --
    never the ethynyl-dropping steroid name."""
    from orthonym.errors import is_failure_name
    from tests.support.rt_assert import name_is_rt_exact
    smi = next(s for f, s, _ in AROMATIC_NP_FIXTURES if f == fixture_id)
    env = os.environ.copy()
    cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env["PYTHONPATH"] = os.path.join(cwd, "src")
    for key in ("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "ORTHONYM_SELF_CONSISTENCY_GATE"):
        env.pop(key, None)   # production gates, whatever the suite exported
    result = subprocess.run(
        [sys.executable, "-c",
         f"from orthonym import name_compound; print(name_compound({smi!r}))"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]
    name = result.stdout.strip()
    assert is_failure_name(name) or name_is_rt_exact(name, smi), (
        f"{fixture_id}: the default tier shipped {name!r}, which is not the input")
