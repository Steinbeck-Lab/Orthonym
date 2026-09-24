"""An N,N-dialkylamino prefix must be assembled, never concatenated.

`assembly/substituent_enumerator.py::_name_amino_branch` built its prefix from a
local `Counter` + `SIMPLE_MULTIPLIERS` copy and a raw `sorted`, joined with no
enclosing marks. For a SYMMETRIC pair that happens to be right
(`dimethylamino`), so the defect stayed invisible; for an ASYMMETRIC pair it
emitted `ethylmethylamino`, which OPSIN reads as the single substituent
*2-ethylmethyl*. `CCN(C)CC(=O)N` was therefore named
`2-ethylmethylaminoacetamide` -> `CCC(NC)C(N)=O`, a DIFFERENT constitution.

Governing rule, (the Blue Book, under
`### **P"16.5** ENCLOSING MARKS`):

    "For mononuclear parent hydrides with two or more substituents the first
    cited substituent never has enclosing marks unless it includes a locant.
    The second and further substituents are each enclosed with parentheses even
    for simple substituents. When the simple substituent groups are accompanied
    by multiplicative prefixes such as 'di' and 'tri', the multiplicative
    prefixes are not included in the parentheses."

The nitrogen is that mononuclear parent -- which is exactly why the symmetric
`dimethylamino` keeps its multiplier outside and needs no inner pair, while the
asymmetric `ethyl(methyl)amino` does. The whole prefix then takes the outer
bracket: POLYFUNCTIONAL COMPOUNDS, the Blue Book
`2-[di(butan-2-yl)amino]butan-2-ol (PIN)`.

Second, independent defect in the same function: each branch was named by
`get_alkyl_name(carbon_count)`. A COUNT names butyl, 2-methylpropyl, butan-2-yl
and tert-butyl all `butyl`, so `CC(C)NCC(=O)N` (isopropyl) emitted
`2-(propylamino)acetamide` -- again a different constitution. Both are cured by
routing through the primitives the ACID path already used
(`composed_prefix_organyl_name` + `composer._assemble_decorated_amino_prefix`),
so an amide and its acid analogue can no longer disagree.

Every expectation below is pinned by an OPSIN round-trip + InChIKey identity
test, not by the spelling alone.
"""

import pytest
from rdkit import Chem

import orthonym.namer as _namer
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    # Gate OFF on purpose: with the gate on, merely REFUSES the wrong
    # name ('unknown organic compound'). That hides the generator defect and
    # turns a wrong molecule into a coverage loss. Judge the generator.
    _namer._DISABLE_VALIDITY_GATE = True
    return Orthonym(style="pin")


def _name(namer, smiles):
    r = namer.name_tiered(smiles)
    return r.get("name") if isinstance(r, dict) else r


# (smiles, expected name) -- the amide/nitrile/ester rows are the ones that were
# wrong; the acid rows are the reference behaviour they must now match.
CASES = [
    # --- asymmetric N,N-dialkyl: the headline defect ---
    ("CCN(C)CC(=O)N", "2-[ethyl(methyl)amino]acetamide"),
    ("CN(CCC)CC(=O)N", "2-[methyl(propyl)amino]acetamide"),
    ("CCCCN(C)CC(=O)N", "2-[butyl(methyl)amino]acetamide"),
    ("CC(C)N(C)CC(=O)N", "2-[methyl(propan-2-yl)amino]acetamide"),
    ("CCN(C)CCC(=O)N", "3-[ethyl(methyl)amino]propanamide"),
    ("CCN(C)CC(=O)NC", "2-[ethyl(methyl)amino]-N-methylacetamide"),
    # --- branched SINGLE alkyl: the carbon-count defect ---
    ("CC(C)NCC(=O)N", "2-[(propan-2-yl)amino]acetamide"),
    ("CCC(C)NCC(=O)N", "2-[(butan-2-yl)amino]acetamide"),
    ("CC(C)(C)NCC(=O)N", "2-(tert-butylamino)acetamide"),
    # --- symmetric dialkyl and unbranched single alkyl: must NOT change ---
    ("CN(C)CC(=O)N", "2-(dimethylamino)acetamide"),
    ("CCN(CC)CC(=O)N", "2-(diethylamino)acetamide"),
    ("CNCC(=O)N", "2-(methylamino)acetamide"),
    ("CCCCNCC(=O)N", "2-(butylamino)acetamide"),
    # --- the acid analogues, which were already correct ---
    ("CCN(C)CC(=O)O", "[ethyl(methyl)amino]acetic acid"),
    ("CC(C)N(C)CC(=O)O", "[methyl(propan-2-yl)amino]acetic acid"),
    ("CC(C)NCC(=O)O", "[(propan-2-yl)amino]acetic acid"),
    ("CC(C)(C)NCC(=O)O", "(tert-butylamino)acetic acid"),
    # --- esters and nitriles ---
    ("CCN(C)CC(=O)OC", "methyl [ethyl(methyl)amino]acetate"),
    ("CC(C)NCC(=O)OC", "methyl [(propan-2-yl)amino]acetate"),
    ("CCN(C)CC#N", "2-[ethyl(methyl)amino]ethanenitrile"),
    ("CC(C)NCC#N", "2-[(propan-2-yl)amino]ethanenitrile"),
    # --- cyclic amine on the amide: a separate producer, anchored here ---
    ("C1CCN(CC1)CC(=O)N", "2-(piperidin-1-yl)acetamide"),
    ("CN(c1ccccc1)CC(=O)N", "2-(N-methylanilino)acetamide"),
]


@pytest.mark.parametrize("smiles,expected", CASES)
def test_amino_prefix_spelling(namer, smiles, expected):
    assert _name(namer, smiles) == expected


@pytest.mark.parametrize("smiles,expected", CASES)
def test_asymmetric_dialkylamino_is_never_a_bare_concatenation(namer, smiles, expected):
    """The structural invariant, independent of the exact spelling above.

    Two DISTINCT branch names on one nitrogen may never be concatenated bare --
    that is what made `ethylmethylamino` read as `2-ethylmethyl`. Whenever the
    emitted prefix carries two different organyl tokens, the second must be
    parenthesized.
    """
    name = _name(namer, smiles)
    assert name is not None
    if "amino" not in name:
        return
    core = name.split("amino")[0]
    # A symmetric pair (dimethyl-, diethyl-) legitimately has no inner pair;
    # an asymmetric pair always does.
    stems = ("methyl", "ethyl", "propyl", "butyl", "propan-2-yl", "butan-2-yl")
    distinct = {s for s in stems if s in core}
    # 'ethyl' is a substring of 'methyl'; drop the containment overlaps.
    distinct = {s for s in distinct if not any(s in o and s != o for o in distinct)}
    if len(distinct) >= 2 and not core.lstrip("([{").startswith("di"):
        assert "(" in core, (
            f"two organyls concatenated bare on one N -- reads as a single "
            f"substituent: {name!r}"
        )


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", CASES)
def test_expected_names_denote_the_input_molecule(smiles, expected):
    """The expectations are not self-certifying: OPSIN must parse each back to
    the SAME constitution as the input. A formula match is NOT sufficient --
    the headline defect preserved the formula exactly."""
    from orthonym.assembly.retained_substitution import OpsinOracle
    from tests.support.jars import jar_or_skip

    jar = jar_or_skip()  # OPSIN jar absent -- round-trip cannot be verified
    parsed = OpsinOracle(opsin_jar=jar).name_to_smiles(expected)
    assert parsed, f"OPSIN could not parse {expected!r}"
    got = Chem.MolToInchiKey(Chem.MolFromSmiles(parsed)).split("-")[0]
    want = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)).split("-")[0]
    assert got == want, f"{expected!r} denotes a DIFFERENT molecule than {smiles}"
