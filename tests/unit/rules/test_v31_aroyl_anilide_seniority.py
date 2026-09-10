""" lever C — aroyl/acyl anilide seniority /.

A ring-attached secondary/tertiary amide whose acyl is a simple unsubstituted
carbocycle (benzoyl, cyclohexanecarbonyl,...) and whose only junior FGs live on
the N-aryl ring is a ring-attached AMIDE: the amide -senior to phenol /
amine / ether) is the parent (benzamide / cyclohexanecarboxamide), and the N-aryl
is a located N-substituent. Tier-A ring competition otherwise picked the junior
phenol/aniline ring as parent and demoted the amide to an 'N-benzoyl' prefix:

    O=C(Nc1ccc(O)cc1)c1ccccc1 -> 'N-benzoyl-4-aminophenol' violation)
    O=C(Nc1ccc(N)cc1)c1ccccc1 -> abstained

Fixed by a tightly-scoped, fail-closed preempt at the top of
``handlers/tier_a_ring.name_tier_a_ring`` that delegates to
``rules.polyfunctional._name_ring_attached_anilide`` (which uses
``rules.amides.name_amide``'s ring-attached path). Substituted / fused /
heteroaromatic acyl rings are NOT simple carbocycles and are NOT intercepted
(name_amide would drop a ring substituent or misname the ring), so they keep
their previous behaviour. Every name below is RT-exact.
"""
from __future__ import annotations

import pytest

from orthonym.namer import name_compound
from orthonym.errors import is_failure_name


@pytest.mark.parametrize("smiles,expected", [
    ("O=C(Nc1ccc(O)cc1)c1ccccc1", "N-(4-hydroxyphenyl)benzamide"),
    ("O=C(Nc1ccc(N)cc1)c1ccccc1", "N-(4-aminophenyl)benzamide"),
    ("O=C(Nc1ccc(OC)cc1)c1ccccc1", "N-(4-methoxyphenyl)benzamide"),
    ("O=C(Nc1ccccc1O)c1ccccc1", "N-(2-hydroxyphenyl)benzamide"),
    # lever B + C together: the N-aryl carries a compound-alkyl decoration
    ("O=C(Nc1ccc(CO)cc1)c1ccccc1", "N-[4-(hydroxymethyl)phenyl]benzamide"),
    # cycloalkanecarbonyl acyl rings
    ("O=C(Nc1ccc(O)cc1)C1CCCCC1", "N-(4-hydroxyphenyl)cyclohexanecarboxamide"),
    ("O=C(Nc1ccc(O)cc1)C1CCCC1", "N-(4-hydroxyphenyl)cyclopentanecarboxamide"),
    ("O=C(Nc1ccc(O)cc1)C1CC1", "N-(4-hydroxyphenyl)cyclopropanecarboxamide"),
    # tertiary N
    ("O=C(N(C)c1ccc(O)cc1)c1ccccc1", "N-(4-hydroxyphenyl)-N-methylbenzamide"),
])
def test_ring_attached_anilide_names_the_senior_amide(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # controls that must NOT change (no junior ring-FG / not an anilide)
    ("O=C(Nc1ccccc1)c1ccccc1", "N-phenylbenzamide"),
    ("O=C(N)c1ccccc1", "benzamide"),
    ("Oc1ccccc1", "phenol"),
    ("Nc1ccccc1", "aniline"),
    ("Cc1ccc(O)cc1", "4-methylphenol"),
])
def test_controls_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,fires", [
    ("O=C(Nc1ccc(O)cc1)c1ccccc1", True),    # benzene acyl -> intercept fires
    ("O=C(Nc1ccc(O)cc1)C1CCCCC1", True),    # cyclohexane acyl -> fires
    ("O=C(Nc1ccc(O)cc1)c1ccncc1", False),   # pyridine acyl -> declined (hetero)
    ("O=C(Nc1ccc(O)cc1)c1ccco1", False),    # furan acyl -> declined (hetero)
    ("O=C(Nc1ccc(O)cc1)c1ccc(Cl)cc1", False),  # substituted acyl ring -> declined
    ("O=C(Nc1ccc(O)cc1)c1cccc2ccccc12", False),  # fused (naphthoyl) -> declined
    ("O=C(Nc1ccc(O)cc1)C1CCC(O)CC1", False),  # substituted cyclohexyl -> declined
])
def test_intercept_is_fail_closed_on_non_simple_carbocyclic_acyl(smiles, fires):
    """The lever-C intercept fires ONLY for a simple unsubstituted carbocyclic
    acyl ring — name_amide would drop a ring substituent or misname a hetero /
    fused ring, so those are declined (fail-closed, producer-honest gate-OFF)."""
    from rdkit import Chem
    from orthonym.namer import Orthonym
    from orthonym.rules.polyfunctional import _name_ring_attached_anilide
    n = Orthonym()
    m = Chem.MolFromSmiles(smiles)
    f = n._perceive(m, smiles, Chem.MolToSmiles(m))
    n._classify(f)
    result = _name_ring_attached_anilide(f)
    assert (result is not None) is fires, f"{smiles} -> {result!r}"
