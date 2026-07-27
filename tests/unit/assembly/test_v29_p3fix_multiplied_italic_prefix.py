"""v29 P3-FIX Item 4 — `di-tert-butyl`, decided from the Blue Book.

The coordinator's V6 said neither `ditert-butyl` nor `di-tert-butyl` appears in the
Blue Book, making this open research. **V6 is REFUTED, and it is a grep artefact:**
the Blue Book text marks italics with asterisks, so the form is stored as
`di-*tert*-butyl`. `grep -c 'di-\\*tert\\*-butyl' BlueBookV2/BlueBookV2.md` = **6**,
three of them `(PIN)`. A plain `grep di-tert-butyl` returns 0 and means nothing.
(Chapter P-16's body is separately OCR-mangled -- hyphens as `G`, spaces as `!`,
rule numbers as `P"16.x` -- which is why the governing rule also looked absent.)

So the Blue Book settles all three candidate spellings outright:

* the HYPHEN: `**P-16.2.4** Hyphens` -> `**P-16.2.4.1** Hyphens are used in
  substitutive names:` -> clause `(d) to separate italic letters from Roman
  letters` (``BlueBookV2.md:6957``), whose verbatim example is
  ``di-tert-butyl (P-61.2.3)`` (``:6964``). So not `ditert-butyl`.
* the MULTIPLIER: `**P-16.3.2** General methodology` clause (a)
  (``BlueBookV2.md:7033``): "*Simple components are unsubstituted parent hydrides,
  such as naphthalene; unsubstituted prefixes, such as ethyl or tert-butyl;
  ... All of these are multiplied by the multiplicative prefixes 'di', 'tri',
  etc.*", with clause (c) reserving bis/tris for a component "*which is
  substituted*". `tert-butyl` is unsubstituted, so `di`, not `bis`.
* the ENCLOSING MARKS: none. `### **P-61.2.2** Cyclic hydrocarbons` gives
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
    # An italicized-prefix-led name is SIMPLE (P-16.3.2(a)), so it can never take
    # a derived bis/tris multiplier -- the hyphen and `bis` are mutually exclusive.
    (1, 'tert-butyl', ''),
    # unchanged for everything else
    (2, 'methyl', 'di'),
    (3, 'phenyl', 'tri'),
    (2, 'propan-2-yl', 'bis'),
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
    """P-16.3.2(a)/(c): unsubstituted -> di/tri; substituted -> bis/tris."""
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
    # ...and its SIBLING producer on the same fragment: these two disagreeing was
    # the defect, so they are asserted together.
    ('CC(C)(C)[Si](C(C)(C)C)(C)C', 'di-tert-butyldimethylsilane'),
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
