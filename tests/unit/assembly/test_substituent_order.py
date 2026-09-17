""" substituent citation order on heterocyclic-ring prefixes.

Family A / Task A5 (v50): names the RIGHT molecule but cited the substituents
in the WRONG order. Round-trip and both pass on these, so no gate sees
them -- this is the "spelling-layer blind spot".

Root cause fixed here: the heterocyclic-ring prefix sort
(`rules/heterocycles.py`) used raw `alpha_sort_key`, whose flat-string key KEEPS
a compound substituent's internal locants. A prefix whose name begins with a
locant digit (`(1,3-oxazol-5-yl)methyl` -> key `1,3-oxazol-5-ylmethyl`) then
sorted ahead of every letter-initial prefix (`chloro`), because ASCII `'1'` (49)
precedes `'c'` (99). (the Blue Book) orders **"Nonitalic Roman
letters... first"**; a locant is only a tie-break. The site now sorts
via `prefix_citation_sort_key`, whose tier-1 key is letters-only.

Each end-to-end case asserts the verbatim Blue-Book PIN string AND an OPSIN
round-trip proving the reordering did not change the molecule (a citation-order
fix moves tokens only).
"""
import pytest

from orthonym import name_compound

try:
    from rdkit import Chem
    from orthonym.validation.opsin_roundtrip import opsin_parse, _java_available
    _RT_OK = _java_available()
except Exception:  # pragma: no cover - RDKit/JVM absent
    _RT_OK = False


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"RDKit could not parse {smiles!r}"
    return Chem.MolToInchiKey(m)


# (def_id, input SMILES, expected Blue-Book PIN string)
A5_HETEROCYCLE_CASES = [
    # BB def 58.1: two ring substituents on 1,3-oxazole. 'chloro' (c) precedes
    # '(1,3-oxazol-5-yl)methyl' (o, letters-only) -- the compound prefix's
    # leading locant must NOT pull it in front of 'chloro'.
    ("58.1", "Clc1coc(Cc2cnco2)n1",
     "4-chloro-2-[(1,3-oxazol-5-yl)methyl]-1,3-oxazole"),
]


@pytest.mark.parametrize("def_id,smiles,expected", A5_HETEROCYCLE_CASES)
def test_a5_citation_order_matches_pin(def_id, smiles, expected):
    got = name_compound(smiles)
    assert got == expected, f"[{def_id}] expected {expected!r}, got {got!r}"


@pytest.mark.parametrize("def_id,smiles,expected", A5_HETEROCYCLE_CASES)
@pytest.mark.skipif(not _RT_OK, reason="OPSIN/JVM unavailable")
def test_a5_reorder_preserves_structure(def_id, smiles, expected):
    """The fix only reorders prefixes; OPSIN must parse our name back to the
    SAME constitution as the input (0-wrong: identical InChIKey skeleton)."""
    got = name_compound(smiles)
    parsed = opsin_parse(got)
    assert parsed, f"[{def_id}] OPSIN could not parse our name {got!r}"
    # Compare connectivity (skeleton) -- the fix never touches stereo.
    assert _inchikey(parsed).split("-")[0] == _inchikey(smiles).split("-")[0], (
        f"[{def_id}] {got!r} does not round-trip to the input structure")


# Regression guards: ordinary heterocyclic prefixes that were already correct
# must NOT reorder. 'chloro' < 'methyl' < 'ethyl'-comparisons, single prefixes.
UNCHANGED_CASES = [
    ("Cc1ccncc1", "4-methylpyridine"),
    ("Clc1ccncc1", "4-chloropyridine"),
    ("Cc1cc(C)nc(C)c1", "2,4,6-trimethylpyridine"),
    ("Cc1ccc(Cl)nc1", "2-chloro-5-methylpyridine"),
    ("CCc1ccc(C)cn1", "2-ethyl-5-methylpyridine"),
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED_CASES)
def test_ordinary_heterocycle_order_unchanged(smiles, expected):
    assert name_compound(smiles) == expected
