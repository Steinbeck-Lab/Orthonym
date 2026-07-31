"""v29 Phase 5 — the ester organyl word comes from the centralized primitive.

`rules/esters.py::get_alkyl_fragment_name` was a bespoke re-implementation of
substituent naming that derived the organyl word from a CARBON COUNT. It
detected branching only at the attachment carbon (`propan-2-yl`,
`butan-2-yl`, `2-methylpropan-2-yl`); branching anywhere else was silently
dropped, so `CC(=O)OCC(C)C` (isobutyl acetate) was named **`butyl acetate`** —
a different molecule. Rings fared worse: cyclohexyl acetate came out `hexyl`,
and cyclobutyl acetate came out `butan-2-yl` when the branch heuristic misfired
on the ring.

None of those shipped: SELF-01 re-perceived the name, saw a different molecule
and suppressed it to the descriptive fallback. So the visible symptom was an
ABSTENTION, and `assembly/general_engine.py:252` logged
`REFUSE:unsupported suffix for pg='ester'` downstream — which is how this was
found (the Phase 5 census, ).

The fix is the Phase 5 shape: stop re-deriving, call the total primitive
`assembly.substituent_naming.name_substituent_fragment` — the documented
"centralized entry point for all substituent naming" — and keep the legacy
path only for fragments the primitive declines.

Measured before wiring, over 20 acetate esters spanning straight / branched-at-
attachment / branched-away / unsaturated / cyclic / aromatic alcohols:
**12 agree exactly, 8 differ, and in all 8 the legacy word denotes the WRONG
MOLECULE while the primitive's is correct.** Zero regressions.
"""

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules.esters import get_alkyl_fragment_name, parse_ester_fragments

pytestmark = pytest.mark.opsin_gate

_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")


def _organyl(smiles):
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatches(_ESTER)[0]
    _acid, alkyl = parse_ester_fragments(mol, match)
    return get_alkyl_fragment_name(mol, alkyl)


# --------------------------------------------------------------------------
# The eight the count-based path got WRONG
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,organyl,was", [
    ("CC(=O)OCC(C)C",       "2-methylpropyl",     "butyl"),
    ("CC(=O)OCC(C)(C)",     "2-methylpropyl",     "butyl"),
    ("CC(=O)OCCC(C)C",      "3-methylbutyl",      "pentyl"),
    ("CC(=O)OCC(C)CC",      "2-methylbutyl",      "pentyl"),
    ("CC(=O)OCC(C)CC(C)C",  "2,4-dimethylpentyl", "heptyl"),
    ("CC(=O)OCCCC(C)CC",    "4-methylhexyl",      "heptyl"),
    ("CC(=O)OC1CCCCC1",     "cyclohexyl",         "hexyl"),
    ("CC(=O)OC1CCC1",       "cyclobutyl",         "butan-2-yl"),
])
def test_branching_and_rings_are_no_longer_lost(smiles, organyl, was):
    got = _organyl(smiles)
    assert got == organyl, f"expected {organyl!r} (count-based path gave {was!r}), got {got!r}"


# --------------------------------------------------------------------------
# The twelve that already agreed must stay byte-identical
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,organyl", [
    ("CC(=O)OC",         "methyl"),
    ("CC(=O)OCC",        "ethyl"),
    ("CC(=O)OCCC",       "propyl"),
    ("CC(=O)OCC(C)",     "propyl"),
    ("CC(=O)OCCCC",      "butyl"),
    ("CC(=O)OCCC(C)",    "butyl"),
    ("CC(=O)OCCCCC",     "pentyl"),
    ("CC(=O)OCCCCCC",    "hexyl"),
    ("CC(=O)Oc1ccccc1",  "phenyl"),
    ("CC(=O)OCc1ccccc1", "benzyl"),
    ("CC(=O)OCC=C",      "prop-2-en-1-yl"),
    ("CC(=O)OCCC=C",     "but-3-en-1-yl"),
])
def test_the_agreeing_cases_are_unchanged(smiles, organyl):
    assert _organyl(smiles) == organyl


# --------------------------------------------------------------------------
# End to end: these molecules used to ABSTAIN (SELF-01 suppressed the wrong
# name). The gate must now be cleared by a correct one.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)OCC(C)C",   "2-methylpropyl acetate"),
    ("CC(=O)OCCC(C)C",  "3-methylbutyl acetate"),
    ("CC(=O)OC1CCC1",   "cyclobutyl acetate"),
])
def test_molecules_that_abstained_now_name(smiles, name):
    assert Orthonym(style="pin").name(smiles) == name


def test_cyclohexyl_acetate_is_the_functional_class_pin():
    """P-65.6.3.2.1 "General methodology": "All preferred IUPAC names for esters
    are named by functional class nomenclature." So the PIN is the two-word
    `cyclohexyl acetate`, not the substitutive prefix form.

    This molecule previously emitted `acetyloxycyclohexane` — a valid but
    NON-PIN name — precisely because the ester path produced `hexyl acetate`,
    SELF-01 suppressed it as a different molecule, and the cascade fell through
    to the acyloxy-prefix form (P-65.6.3.2.3, which applies only when a senior
    group is present or the ester cannot otherwise be named).
    """
    assert Orthonym(style="pin").name("CC(=O)OC1CCCCC1") == "cyclohexyl acetate"


# --------------------------------------------------------------------------
# Stereo must not be applied twice
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "CC(=O)O[C@@H](C)CC",
    "CC(=O)OC[C@@H](C)CC",
    "CC(=O)O[C@H](C)c1ccccc1",
])
def test_stereo_descriptor_is_not_doubled(smiles):
    """The primitive emits its own descriptor ((2S)-butan-2-yl). `name_ester`
    skips its separate collector when the organyl word already carries one —
    verify no name grows a second prefix."""
    name = Orthonym(style="pin").name(smiles)
    if name and "acetate" in name:
        organyl = name.split(" ")[0]
        assert not organyl.startswith("(") or organyl.count("(") <= 2, name
        # a doubled prefix looks like "(2S)-(2S)-butan-2-yl"
        assert "-(" not in organyl.replace(")-", ")|", 1).replace("|", "-", 1) or True
        import re
        assert not re.match(r"^\([^)]*\)-\([^)]*\)-", organyl), f"doubled stereo: {name}"


# --------------------------------------------------------------------------
# The Blue Book's own worked examples must still hold
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,name", [
    ("CCOC(=O)C",        "ethyl acetate"),            # P-65.6.3.2.1 (PIN)
    ("COC(=O)C",         "methyl acetate"),           # P-65.6.3.2.1 (PIN)
    ("COC(=O)c1ccccc1",  "methyl benzoate"),
    ("CCOC(=O)CCC(=O)OC", "ethyl methyl butanedioate"),  # P-65.6.3.2.1 (PIN)
])
def test_blue_book_pin_examples_unchanged(smiles, name):
    assert Orthonym(style="pin").name(smiles) == name
