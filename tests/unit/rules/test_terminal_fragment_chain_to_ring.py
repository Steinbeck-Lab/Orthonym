"""terminal_fragment: a chain leading to a ring system (off-ring attachment) now
names, instead of the documented `_composite_fragment_name` refusal
("attachment is acyclic but the fragment carries a ring; out of implemented
scope"). v30 ring-branch breadth lever.

The acyl-with-ring branch-abort class -- benzoyl `-C(=O)Ph`, ring-carbonyl
`-C(=O)-N(ring)`, and plain chain-to-ring alkyls -- is named by building the
ACYCLIC backbone from the off-ring attach (ring-stopping) and recursing each ring
system as a decoration through the existing `_composite_fragment_name` path. The
carbonyl O becomes an `oxo`, so no acyl vocabulary is needed (the reference
"oxo-substituted chain" form).

The tokens are the module's REPLACEMENT-nomenclature spelling (a ring via
`terminal_ring` -> `cyclohexa-1,3,5-trien-1-yl` for benzene, the carbonyl O as a
backbone `oxa`) -- ugly but RT-CORRECT, exactly the best-effort contract
(the contributor guide invariant 1: a table miss degrades to an uglier name, never a
refusal). Prettier RETAINED ring names (phenyl / piperidin-1-yl) would come from
recursing the ring decoration through the full retained namer -- a follow-on.

Every expected token OPSIN-round-trip-verified IN A PARENT (a bare token does not
parse), e.g. `[1-(cyclohexa-1,3,5-trien-1-yl)-2-oxaeth-1-en-1-yl]cyclohexane`
-> O=C(C1CCCCC1)c1ccccc1 (RT-exact).

Best-effort tier only (reached under `allow_mancude`) -> PIN byte-identical.
"""
import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import terminal_fragment_name


def _tf(smiles, attach):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, set(range(mol.GetNumAtoms())), attach


# ---- NEW: off-ring attach + ring system in the fragment ----------------------

@pytest.mark.parametrize("smiles,attach,expected", [
    # benzoyl: -C(=O)-Ph, attach = carbonyl C (idx 1 in O=Cc1ccccc1)
    ("O=Cc1ccccc1", 1, "1-(cyclohexa-1,3,5-trien-1-yl)-2-oxaeth-1-en-1-yl"),
    # piperidin-1-yl-carbonyl: -C(=O)-N(ring), attach = carbonyl C
    ("O=CN1CCCCC1", 1, "1-(1-azacyclohexan-1-yl)-2-oxaeth-1-en-1-yl"),
    # plain chain-to-ring alkyl: -CH2-CH2-cyclohexyl, attach = terminal CH2 (idx 0)
    ("CCC1CCCCC1", 0, "2-(cyclohexan-1-yl)ethyl"),
])
def test_chain_to_ring_now_names(smiles, attach, expected):
    mol, frag, at = _tf(smiles, attach)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None, f"{smiles} refused (chain-to-ring not handled)"
    assert got.name == expected
    assert got.atoms == frozenset(frag), "completeness: every atom accounted"


# ---- REGRESSION: ring-ATTACH and pure-acyclic paths unchanged ----------------

def test_ring_attachment_still_composite_unchanged():
    # attach ON the ring -> ring is the parent (composite path), byte-identical
    mol, frag, at = _tf("C1CCCCC1", 0)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None
    assert got.name == "cyclohexan-1-yl"  # current composite-path output, unchanged


def test_pure_acyclic_unchanged():
    mol, frag, at = _tf("CCCC", 0)
    got = terminal_fragment_name(mol, frag, at)
    assert got is not None
    assert got.name == "butyl"


# ---- 0-WRONG: a branch joined by a NON-single bond (ylidene) fails closed -----
# Recursing it yields a `-yl` token asserting a SINGLE bond -> wrong constitution
# (fable ring-review R1). No -ylidene constructor here, so refuse.

@pytest.mark.parametrize("smiles,attach", [
    ("CC=C1CCCCC1", 0),   # ring-ylidene: -CH2-CH=C<cyclohexylidene (this lever's surface)
    ("CC(=C)CC", 0),      # acyclic methylidene: -CH2-C(=CH2)- (pre-existing sibling)
])
def test_ylidene_branch_fails_closed(smiles, attach):
    mol, frag, at = _tf(smiles, attach)
    assert terminal_fragment_name(mol, frag, at) is None, (
        "a non-single attach bond needs -ylidene, not -yl; must refuse not "
        "assert a single bond (wrong constitution)"
    )
