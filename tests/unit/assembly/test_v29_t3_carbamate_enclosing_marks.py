"""A COMPOUND N-substituent on a carbamate takes enclosing marks.

**This is NOT the _name_r_group caller defect** (see
``test_v29_t3_r_group_caller_refusals.py``). Attributed on its own evidence:
``_name_r_group`` returns the correct ``'2-methoxyethyl'`` for this molecule,
and the emission is byte-identical with the caller fix reverted. The two
defects merely surfaced in the same molecule family.

``_name_carbamate`` hand-rolled its N-prefix as ``f"N-{_wrap_n_substituent(n)}"``.
``_wrap_n_substituent`` only ESCALATES an enclosure that is already present
(parentheses -> square brackets); it never adds the first-level marks. So a
compound substituent came out bare and ``COC(=O)NCCOC`` shipped
``methyl N-2-methoxyethylcarbamate`` on the DEFAULT path -- the substituent's
locant abutting the N-locant, which is what the enclosure exists to prevent.

Blue Book, opened with ``sed`` at write time:

* ** "ENCLOSING MARKS"** (``the Blue Book``), ****
  (``:7232``): *"Parentheses are used around compound (see and complex
  (see prefixes; after the multiplicative prefixes 'bis', 'tris',
  etc.;..."*
* **** (``:15762``): *"A compound substituent group consists of a simple
  substituent group (the parent substituent group) to which is attached one or
  more simple substituent groups."* ``2-methoxyethyl`` is ``ethyl`` bearing
  ``methoxy``, so it is compound and applies.
* PIN exemplar (``:33336``):
  ``N-[1-cyano-3-(methylsulfanyl)propyl]-N'-methylurea (PIN)`` -- the compound
  N-substituent is enclosed, the simple ``methyl`` is not.

The carbamic-acid sibling never had the defect because it uses the shared
``_build_n_substituted_name``; the fix routes carbamate through that same
builder instead of re-deriving the rule at a second site.

Both now share ``_carbamic_n_substituted_name`` (same enclosing marks), which
cites the N-substituents WITHOUT the italic-N locant:
(the Blue Book) '(CH3)2N-COOH dimethylcarbamic acid (PIN)' (:30762),
'2-hydroxypropyl (2-aminoethyl)carbamate (PIN)' (:30766). OPSIN 2.9.0 full-key
exact for every expected name below (TRIAGE 'Suite fix -- j5-pin-labels-b').
"""

import pytest


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # The malformed shipper, and its ethyl ester twin.
        ("COC(=O)NCCOC", "methyl (2-methoxyethyl)carbamate"),
        ("CCOC(=O)NCCOC", "ethyl (2-methoxyethyl)carbamate"),
        # A located branched prefix is compound too via its locant).
        ("COC(=O)NC(C)C", "methyl (propan-2-yl)carbamate"),
        # SIMPLE substituents must stay bare -- byte-identical to before.
        ("COC(=O)NC", "methyl methylcarbamate"),
        ("CCOC(=O)NC", "ethyl methylcarbamate"),
        ("COC(=O)N(C)C", "methyl dimethylcarbamate"),
        ("CCOC(=O)N(C)C", "ethyl dimethylcarbamate"),
        ("COC(=O)Nc1ccccc1", "methyl phenylcarbamate"),
        ("COC(=O)N", "methyl carbamate"),
    ],
)
def test_carbamate_n_substituent_enclosing_marks(smiles, expected):
    from orthonym import name_compound

    res = name_compound(smiles)
    assert getattr(res, "name", res) == expected


def test_the_malformed_form_is_gone():
    """The exact string that used to ship."""
    from orthonym import name_compound

    res = name_compound("COC(=O)NCCOC")
    assert getattr(res, "name", res) != "methyl N-2-methoxyethylcarbamate"


def test_carbamic_acid_sibling_was_always_well_formed():
    """Corroboration that the enclosed form is the tree's own convention:
    the carbamic-acid path already used the shared builder."""
    from orthonym import name_compound

    res = name_compound("OC(=O)NCCOC")
    assert getattr(res, "name", res) == "(2-methoxyethyl)carbamic acid"
