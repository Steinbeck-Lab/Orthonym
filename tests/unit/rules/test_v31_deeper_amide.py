""" lever A — deeper amide: a polyfunctional secondary amide whose ACYL
chain is branched / unsaturated and whose junior FG lives inside the
N-substituent now EMITS (was abstaining), via the acyl-aware composer namer
``_assemble_amide_name`` /. Also covers two honesty/spelling
fixes surfaced by the gate-OFF honesty sweep:

* the ``_enrich_ring_n_substituent`` aromatic double-count
  (``CC(C)C(=O)NCc1ccc(O)cc1`` used to emit ``...[4-hydroxy(4-hydroxyphenyl)
  methyl]...`` — a parseable WRONG molecule), and
* the unsaturated-branch N-prefix glue that dropped the hyphen before a leading
  acyl locant (``N-(4-hydroxyphenyl)3-methylbut-2-enamide``).

Root cause (was): the polyfunctional amide delegation handed the amide only to
the COUNT-based ``rules.amides.name_amide`` (``_acyl_is_simple_saturated_chain``
guard); a branched / unsaturated acyl fell through to the chain machinery, which
mis-rooted the N-aryl and abstained. Now such an on-chain acyl is routed to the
acyl-aware ``_assemble_amide_name`` under the SAME coverage guards.

NOTE on prefix ORDER: these names follow Orthonym's existing uniform amide
convention (C-substituent before N-substituent for the saturated branch). The
 / alphanumerical N/C merge-and-order is a SEPARATE conformance
lever, tracked by
``test_amide_merged_prefix_p16_3_3::test_chain_amide_merges_too`` (still xfail).
Every name below is RT-exact (correct constitution), which is what this lever
delivers.
"""
from __future__ import annotations

import pytest

from orthonym.namer import name_compound


@pytest.mark.parametrize("smiles,expected", [
    # --- the stated targets: paracetamol's isobutyryl / acryloyl cousins ---
    ("CC(C)C(=O)Nc1ccc(O)cc1", "2-methyl-N-(4-hydroxyphenyl)propanamide"),
    ("C=CC(=O)Nc1ccc(O)cc1",   "N-(4-hydroxyphenyl)prop-2-enamide"),
    # branched saturated, other acyls
    ("CC(C)(C)C(=O)Nc1ccc(O)cc1", "2,2-dimethyl-N-(4-hydroxyphenyl)propanamide"),
    ("CC(C)CC(=O)Nc1ccc(O)cc1",   "3-methyl-N-(4-hydroxyphenyl)butanamide"),
    ("CCC(CC)C(=O)Nc1ccc(O)cc1",  "2-ethyl-N-(4-hydroxyphenyl)butanamide"),
    # unsaturated acyls
    ("C#CC(=O)Nc1ccc(O)cc1",   "N-(4-hydroxyphenyl)prop-2-ynamide"),
    ("CC=CC(=O)Nc1ccc(O)cc1",  "N-(4-hydroxyphenyl)but-2-enamide"),
    # branched AND unsaturated -> the hyphen must separate the leading acyl locant
    ("CC(C)=CC(=O)Nc1ccc(O)cc1", "N-(4-hydroxyphenyl)-3-methylbut-2-enamide"),
    ("C=C(C)C(=O)Nc1ccc(O)cc1",  "N-(4-hydroxyphenyl)-2-methylprop-2-enamide"),
    # carbocyclic substituent on the acyl (phenylacetyl / cyclohexylacetyl)
    ("c1ccccc1CC(=O)Nc1ccc(O)cc1",  "2-phenyl-N-(4-hydroxyphenyl)acetamide"),
    # junior FG = amine (also junior to amide,, and other ring positions
    ("CC(C)C(=O)Nc1ccc(N)cc1",  "2-methyl-N-(4-aminophenyl)propanamide"),
    ("CC(C)C(=O)Nc1ccccc1O",    "2-methyl-N-(2-hydroxyphenyl)propanamide"),
    # N-alkyl carrying the junior FG (hydroxyethyl)
    ("CC(C)C(=O)NCCO",  "2-methyl-N-(2-hydroxyethyl)propanamide"),
])
def test_polyfunctional_branched_or_unsaturated_amide_now_names(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # _enrich_ring_n_substituent no longer double-counts an aromatic ring-FG.
    # (branched acyl -> new route; acetyl -> pre-existing name_amide route: BOTH
    # used to double-count the hydroxy and abstain.)
    ("CC(C)C(=O)NCc1ccc(O)cc1", "2-methyl-N-[(4-hydroxyphenyl)methyl]propanamide"),
    ("CC(=O)NCc1ccc(O)cc1",     "N-[(4-hydroxyphenyl)methyl]acetamide"),
])
def test_aromatic_n_substituent_ring_fg_not_double_counted(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # controls: simple-saturated acyl still goes through the count path unchanged
    ("CC(=O)Nc1ccc(O)cc1",   "N-(4-hydroxyphenyl)acetamide"),   # paracetamol
    ("CCCCC(=O)Nc1ccc(O)cc1", "N-(4-hydroxyphenyl)pentanamide"),
])
def test_simple_saturated_controls_unchanged(smiles, expected):
    assert name_compound(smiles) == expected
