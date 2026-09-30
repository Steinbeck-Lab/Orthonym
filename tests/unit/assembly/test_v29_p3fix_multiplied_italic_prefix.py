"""-FIX Item 4 — `di-tert-butyl`, decided from the Blue Book.

The coordinator's V6 said neither `ditert-butyl` nor `di-tert-butyl` appears in the
Blue Book, making this open research. **V6 is REFUTED, and it is a grep artefact:**
the Blue Book text marks italics with asterisks, so the form is stored as
`di-*tert*-butyl`. `grep -c 'di-\\*tert\\*-butyl' the Blue Book Blue Book` = **6**,
three of them `(PIN)`. A plain `grep di-tert-butyl` returns 0 and means nothing.
(Chapter 's body is separately OCR-mangled -- hyphens as `G`, spaces as `!`,
rule numbers as `P"16.x` -- which is why the governing rule also looked absent.)

So the Blue Book settles all three candidate spellings outright:

* the HYPHEN: `**** Hyphens` -> `**** Hyphens are used in
  substitutive names:` -> clause `(d) to separate italic letters from Roman
  letters` (``the Blue Book``), whose verbatim example is
  ``di-tert-butyl `` (``:6964``). So not `ditert-butyl`.
* the MULTIPLIER: `**** General methodology` clause (a)
  (``the Blue Book``): "*Simple components are unsubstituted parent hydrides,
  such as naphthalene; unsubstituted prefixes, such as ethyl or tert-butyl;
  ... All of these are multiplied by the multiplicative prefixes 'di', 'tri',
  etc.*", with clause (c) reserving bis/tris for a component "*which is
  substituted*". `tert-butyl` is unsubstituted, so `di`, not `bis`.
* the ENCLOSING MARKS: none. `### **** Cyclic hydrocarbons` gives
  ``1,2-di-tert-butylbenzene (PIN)`` (``:25717``) and ``:37495`` gives
  ``2,4,6-tri-tert-butylphenyl`` -- so not `di(tert-butyl)` either, and `tri`
  rather than `tris` at count 3.

Nothing is ASSUMED here; every candidate is excluded by a cited example.

The defect was that three composers open-coded around the shared
`multiplier_needs_hyphen` primitive, so ONE fragment got TWO spellings depending on
which producer claimed the molecule -- `(ditert-butylmethylsilyl)acetic acid` from
the migrated Group-14 path versus `di-tert-butyldimethylsilane` from its sibling.
Both round-trip in OPSIN, so the phase gate could not see the disagreement.

Root-cause fix: the hyphen is now produced by `get_multiplier_prefix`, the ONE
place that already receives both the count and the name, so all ~40 call sites get
it without any of them knowing about it. `format_substituent_prefix`'s second copy
is replaced by an assertion that the primitive did its job.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.naming_utils import (format_substituent_prefix,
                                             get_multiplier_prefix)


# --------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------

@pytest.mark.parametrize("count,name,expected", [
    (2, 'tert-butyl', 'di-'),
    (3, 'tert-butyl', 'tri-'),
    (4, 'tert-butyl', 'tetra-'),
    (2, 'sec-butyl',  'di-'),
    # -CLOSEOUT Item A: the claim that an italicized-prefix-led name "can
    # never take a derived bis/tris multiplier" is FALSE -- `tert-butylsulfanyl`
    # reduces to the compound `butylsulfanyl` and takes `bis` (a)), with
    # NO hyphen because it is enclosed, BB 6968). The carve-out is
    # about the italicized PREFIX, not about everything it leads.
    (1, 'tert-butyl', ''),
    (2, 'tert-butylsulfanyl', 'bis'),
    # unchanged for everything else
    (2, 'methyl', 'di'),
    (3, 'phenyl', 'tri'),
    # -CLOSEOUT Item A re-baseline: was 'bis'. `propan-2-yl` is an
    # UNSUBSTITUTED simple prefix that merely carries a locant, so (a)
    # parenthesises it while (a) multiplies it with the SIMPLE 'di' --
    # BB's own example is `di(propanG2Gyl)!(preferred!prefix)` at:7087, and
    # `1,4-di(propan-2-yl)cyclohexane (PIN)` at:25721. `bis` is reserved by
    # (a) for a SUBSTITUTED prefix: contrast
    # `1,4-bis(2-chloropropan-2-yl)benzene (PIN)` (:25793), one chloro apart.
    (2, 'propan-2-yl', 'di'),
    (3, 'cyclohexylmethyl', 'tris'),
])
def test_the_multiplier_primitive_carries_the_hyphen(count, name, expected):
    assert get_multiplier_prefix(count, name) == expected


def test_the_hyphen_is_never_doubled():
    """`format_substituent_prefix` used to add a SECOND hyphen of its own."""
    assert format_substituent_prefix('tert-butyl', [1, 2], 2) == \
        '1,2-di-tert-butyl'
    assert '--' not in format_substituent_prefix('tert-butyl', [1, 2], 2)
    assert format_substituent_prefix('tert-butyl', [], 3) == 'tri-tert-butyl'


def test_a_simple_prefix_is_never_multiplied_with_bis():
    """(a)/(c): unsubstituted -> di/tri; substituted -> bis/tris."""
    assert get_multiplier_prefix(2, 'tert-butyl').rstrip('-') == 'di'
    assert not get_multiplier_prefix(2, 'tert-butyl').startswith('bis')


# --------------------------------------------------------------------------
# end to end: every producer now spells the SAME fragment the SAME way
# --------------------------------------------------------------------------

@pytest.fixture
def namer():
    from orthonym.namer import Orthonym
    return Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    # BB 25717 verbatim PIN.
    ('CC(C)(C)c1ccccc1C(C)(C)C', '1,2-di-tert-butylbenzene'),
    # The migrated Group-14 path -- was `(ditert-butylmethylsilyl)acetic acid`.
    ('CC(C)(C)[Si](C(C)(C)C)(C)CC(=O)O',
     '(di-tert-butylmethylsilyl)acetic acid'),
    #...and its SIBLING producer on the same fragment: these two disagreeing was
    # the defect, so they are asserted together.
    # (the Blue Book): the second cited prefix of a mononuclear
    # parent hydride is enclosed, its multiplier outside ('ethyldi(methyl)
    # phosphane (PIN)', 'tert-butyldi(methyl)phosphane (PIN)':16286).
    ('CC(C)(C)[Si](C(C)(C)C)(C)C', 'di-tert-butyldi(methyl)silane'),
    # count 3 (BB 37495's `tri-tert-butyl` shape)
    ('CC(C)(C)[Si](C(C)(C)C)(C(C)(C)C)CC(=O)O',
     '(tri-tert-butylsilyl)acetic acid'),
    # the third site, an amide N -- was `N,N-ditert-butylformamide`
    ('CC(C)(C)N(C(C)(C)C)C=O', 'N,N-di-tert-butylformamide'),
    # a mononuclear pnictogen hub
    ('CC(C)(C)[As](C(C)(C)C)C', 'di-tert-butyl(methyl)arsane'),
    # the -oxy form, where the italic prefix is interior to the token
    ('CC(C)(C)Oc1ccc(OC(C)(C)C)cc1', '1,4-di-tert-butoxybenzene'),
])
def test_multiplied_tert_butyl_is_spelled_the_blue_book_way(namer, smiles,
                                                           expected):
    assert namer.name(Chem.CanonSmiles(smiles)) == expected


def test_the_two_group14_paths_agree_on_one_fragment():
    """V3's defect stated as a property rather than as two strings: whichever
    producer claims the molecule, the di-tert-butyl fragment is spelled once."""
    from orthonym.namer import Orthonym
    n = Orthonym()
    a = n.name(Chem.CanonSmiles('CC(C)(C)[Si](C(C)(C)C)(C)CC(=O)O'))
    b = n.name(Chem.CanonSmiles('CC(C)(C)[Si](C(C)(C)C)(C)C'))
    assert 'di-tert-butyl' in a, a
    assert 'di-tert-butyl' in b, b
    assert 'ditert' not in a and 'ditert' not in b, (a, b)
