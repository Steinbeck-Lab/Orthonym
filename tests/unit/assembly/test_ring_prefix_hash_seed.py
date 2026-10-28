"""One name for one molecule, whatever the hash seed: a ring prefix whose two groups differ only in CIP descriptors.

The imidazolidin-2-ylidene of ``C[C@@H]1CC(=CC(=C1N2CCN([C]2)C3=C(C=C(C[C@@H]3C)C)C)C)C`` carries two
2,4,6-trimethylcyclohexa-1,3-dien-1-yl groups on its two nitrogens, one ``(6R)`` and one ``(6S)``. The ring runs
either way round from its free valence (C-2), so the numbering is decided by "NUMBERING" (the Blue Book),
criterion (g) "lowest locants for the substituent cited first as a prefix in the name" (:3307), and the groups
are cited R before S: (:22603, under ``### **** CRITERIA RELATED ONLY TO CONFIGURATION``), "When names
based on alphanumerical order and isotopic descriptors are the same, further choice depends on the alphabetic order of
the stereochemical descriptors 'R' and 'S'" (``1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromoethyl]cyclopentane (PIN)``,
:22609), and criterion (j) (:3346) gives the lower locant to the descriptor R: '1-[(6R)-...]-3-[(6S)-...]'.

 (:3446) leaves the descriptors out of the alphanumerical order, so the two names tie on it, and
``ring_prefix_numbering.g_key`` sorted the SET of the names by that order alone: the order of a set of strings is the
order of their hashes, so the name was '1-[(6R)-...]-3-[(6S)-...]' or '3-[(6R)-...]-1-[(6S)-...]' with the hash seed
(5 of 6 runs of the default seed gave the second). Principle 4 (the contributor guide, Determinism): the same input always yields
the same output.
"""
import os
import subprocess
import sys

import pytest

from tests.support.jars import jar_or_skip

ROW = "C[C@@H]1CC(=CC(=C1N2CCN([C]2)C3=C(C=C(C[C@@H]3C)C)C)C)C"
NAME = ("1-[(6R)-2,4,6-trimethylcyclohexa-1,3-dien-1-yl]-3-[(6S)-2,4,6-trimethylcyclohexa-1,3-dien-1-yl]"
        "imidazolidin-2-ylidene")
#: seeds that gave '3-[(6R)...]-1-[(6S)...]' before the fix (1, 2, 4, 7) and seeds that gave the other one (3, 5, 6, 8)
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


def _run(code, seed, *args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONHASHSEED"] = str(seed)
    return subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True, text=True,
                          timeout=600)


_G_KEY = r"""
import sys
from orthonym.assembly.ring_prefix_numbering import g_key
named = [(10, '(6S)-2,4,6-trimethylcyclohexa-1,3-dien-1-yl'), (11, '(6R)-2,4,6-trimethylcyclohexa-1,3-dien-1-yl')]
print(g_key({10: 3, 11: 1}, named), g_key({10: 1, 11: 3}, named))
"""


@pytest.mark.unit
def test_g_key_cites_r_before_s_under_every_hash_seed():
    """The groups R and S tie on, so the order of citation is the descriptors',:22603): R first,
    and the numbering that puts R at the lower locant has the lower key (g), (j))."""
    outs = {seed: _run(_G_KEY, seed) for seed in SEEDS}
    assert {s: r.returncode for s, r in outs.items()} == {s: 0 for s in SEEDS}, {s: r.stderr[-400:] for s, r in outs.items()}
    assert {r.stdout.strip() for r in outs.values()} == {"((1,), (3,)) ((3,), (1,))"}, {s: r.stdout for s, r in outs.items()}


_NAME_ROW = r"""
import sys
from orthonym.cli import _tiered_namer
print(_tiered_namer("pin", sys.argv[2]).name_tiered(sys.argv[1]).get("name"))
"""

#: (:48163, "Relative configuration: the stereodescriptors 'cis' and 'trans'"): "'cis' arrangements of
#: ligands are cited before 'trans' isomers, when a choice is possible"; the two groups tie on and on (j),
#: which ranks 'cis' and 'trans' alike as non-CIP descriptors, so the order of the groups is the one of that rule.
#: the same two groups with one atom order and the other: R is cited first, at locant 1, in the name of either tier
#:,:22603; the citation order of ``_recursive_fragment_substituent_name`` followed the order of the atoms)
CHLOROETHYL_ROW = "OCC1N([C@H](C)Cl)CCN1[C@@H](C)Cl"
CHLOROETHYL_NAME = "{1-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]imidazolidin-2-yl}methanol"
CIS_TRANS_ROW = "OCC1N([C@H]2CC[C@H](Cl)CC2)CCN1[C@H]1CC[C@@H](Cl)CC1"
CIS_TRANS_NAME = "[1-(cis-4-chlorocyclohexyl)-3-(trans-4-chlorocyclohexyl)imidazolidin-2-yl]methanol"


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("row, tier, name", [
    (ROW, "best-effort", NAME),
    (ROW, "pin", NAME),
    (CIS_TRANS_ROW, "best-effort", CIS_TRANS_NAME),
    (CHLOROETHYL_ROW, "pin", CHLOROETHYL_NAME),
    (CHLOROETHYL_ROW, "best-effort", CHLOROETHYL_NAME),
], ids=["imidazolidinylidene-best-effort", "imidazolidinylidene-pin", "cis-trans-best-effort",
        "chloroethyl-pin", "chloroethyl-best-effort"])
def test_the_ring_prefixes_get_the_one_name_under_every_hash_seed(row, tier, name):
    """ (g), (j): the R group (the cis group) takes locant 1, whatever the hash seed (a fresh process per seed)."""
    jar_or_skip("opsin")
    outs = {seed: _run(_NAME_ROW, seed, row, tier) for seed in SEEDS}
    names = {s: r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:] for s, r in outs.items()}
    assert set(names.values()) == {name}, names


#: The numbering of the ring is fixed here by the methyl group (C-4, so N-3 is the nitrogen next to it), so the two groups
#: that tie on do not choose it: R is cited first at the HIGHER locant 3, before S at 1. ``**** General
#: principles`` (:22272, under ``### **** CRITERIA RELATED ONLY TO CONFIGURATION``) has the same order in
#: ``4-{3,4-bis[(1R)-1-chloroethyl]phenoxy}-1,2-bis[(1S)-1-chloroethyl]benzene (PIN)`` (:22597), and ``****``
#: (:3517, under ``### **** ALPHANUMERICAL ORDER``) compares the locants inside the prefix names, not those of the
#: ring. Main cited S first (the order of the atoms of the input) at both tiers, for every hash seed.
FIXED_NUMBERING_ROW = "OCC1N([C@H](C)Cl)CC(C)N1[C@@H](C)Cl"
FIXED_NUMBERING_NAME = "{3-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl]-4-methylimidazolidin-2-yl}methanol"


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_r_group_is_cited_first_at_the_higher_locant_under_every_hash_seed(tier):
    """ (:22597), (:3517): the one name under every seed, and OPSIN reads it back to the input."""
    from tests.support.rt_assert import name_is_rt_exact
    jar_or_skip("opsin")
    outs = {seed: _run(_NAME_ROW, seed, FIXED_NUMBERING_ROW, tier) for seed in SEEDS}
    names = {s: r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:] for s, r in outs.items()}
    assert set(names.values()) == {FIXED_NUMBERING_NAME}, names
    assert name_is_rt_exact(FIXED_NUMBERING_NAME, FIXED_NUMBERING_ROW)
