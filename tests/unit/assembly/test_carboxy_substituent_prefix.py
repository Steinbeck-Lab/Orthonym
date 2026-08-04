"""A `-C(=O)OH` substituent is `carboxy`, not `formyl` (P-65.1.1.2).

Measured defect, found while pre-validating the v30 assembly target: composing
`terminal_ring_name` parents with `name_substituent` prefixes produced
`5,7-diformyl-…` for an input whose only such group is `C(=O)O`, i.e. a name denoting
a molecule two oxygens short. 34 of 71 composed rows came out as a wrong molecule and
this is one confirmed cause.

The mechanism: Tier 4 of the substituent cascade names the fragment recursively as a
whole compound — `O=CO` names as *formic acid* — and then
`substituent_naming.parent_to_prefix` converts that NAME through
`_ACID_TO_ACYL_PREFIX = {'formic acid': 'formyl', 'acetic acid': 'acetyl', …}`.

That mapping is right for an **acyl group** and wrong here, and the reasoning is
decisive rather than a matter of taste: `formyl` is `HC(=O)–`, whose fragment is two
atoms and names as *formaldehyde*. If the recursive namer said *formic acid*, the
fragment necessarily still carries its hydroxyl — so `carboxy` is correct and `formyl`
is never correct for it. `parent_to_prefix` receives only a name, so it cannot tell;
the caller has the graph and must decide.

⚠ `parent_to_prefix` is deliberately NOT changed. Three things depend on the acyl
mapping as it stands: `rules/acid_halides.py:16` documents *"formic acid -> formyl
(retained, preferred for C1)"* for acid halides, where it is correct;
`decomposition/fragment_assembly.py:136` holds its own copy; and
`tests/unit/rules/test_v29_phase7_prefix_vocabulary.py:51` asserts the current return
value directly. So the fix is a graph-shape guard at the substituent caller, exactly as
`rules/ring_assemblies.py:1526` already does for the ring-assembly path — *"P-65.1.1.2:
the prefix for -COOH is 'carboxy'"*.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _tok(smiles, frag, attach):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return name_substituent(mol, list(frag), attach, allow_mancude=True)


def _tok_parent_side(smiles, frag, parent_atom):
    """Call with the PARENT-side attachment atom, which is the convention
    `discover_substituents` uses (`SubstituentInfo.attach_mol_idx`).

    This exists because the first version of the fix was green with a fragment-side
    index and never fired in production, which passes the parent-side one. Both
    conventions circulate in this tree -- `rules/ring_substituents.py:1869-1879`
    normalizes between them -- so both must be covered or the test proves nothing
    about the code path that actually runs.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert parent_atom not in set(frag), (
        "this helper is for the PARENT-side convention; the atom must be OUTSIDE "
        "the fragment or the test is not exercising the normalization"
    )
    return name_substituent(mol, list(frag), parent_atom, allow_mancude=True)


def test_bare_carboxy_is_carboxy_not_formyl():
    """The defect itself. `CC(=O)O` atoms {1,2,3} attached via atom 1 is -C(=O)OH."""
    assert _tok("CC(=O)O", [1, 2, 3], 1) == "carboxy"


def test_bare_carboxy_via_parent_side_attachment():
    """THE test that matters -- this is the convention production uses.

    Measured on a real dev500 row
    (`C[C@@H]1C[C@H](O)[C@H]2C(C)(C)C[C@](C)(C(=O)O)[C@@]2(O)[C@@H]1O`):
    `discover_substituents` yields `frag_atoms={12,13,14}` with
    `attach_mol_idx=10`, the RING carbon -- outside the fragment. With the raw index
    the guard cannot fire, and the composed name came out `5,7-diformyl-...` for a
    molecule whose only such group is `C(=O)O`.
    """
    assert _tok_parent_side("CC(=O)O", [1, 2, 3], 0) == "carboxy"


def test_formyl_is_still_formyl():
    """The control that stops the fix over-reaching. -CH=O really is formyl, and a
    guard that returned `carboxy` here would trade one wrong molecule for another."""
    assert _tok("CC=O", [1, 2], 1) == "formyl"


def test_carboxymethyl_unchanged():
    """Already correct before the fix; pinned so the guard cannot regress it.
    `CCC(=O)O` atoms {1,2,3,4} via atom 1 is -CH2-COOH."""
    assert _tok("CCC(=O)O", [1, 2, 3, 4], 1) == "carboxymethyl"


@pytest.mark.parametrize("smiles,frag,attach", [
    ("OC(=O)c1ccccc1", [0, 1, 2], 1),        # benzoic acid's -COOH on the ring
    ("OC(=O)CCC", [0, 1, 2], 1),             # butanoic acid's -COOH on the chain
])
def test_carboxy_on_other_parents(smiles, frag, attach):
    """The same shape wherever it attaches — the guard is on the fragment, not the
    parent, so it must hold for a ring and a chain alike."""
    assert _tok(smiles, frag, attach) == "carboxy"


def test_charged_carboxylate_is_not_called_carboxy():
    """A deprotonated -C(=O)O- is a P-73 anion, not the neutral `carboxy` prefix.
    Naming it `carboxy` would assert a proton the input does not have."""
    tok = _tok("CC(=O)[O-]", [1, 2, 3], 1)
    assert tok != "carboxy", (
        f"charged carboxylate must not be spelled as the neutral prefix; got {tok!r}"
    )


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_the_fix_makes_a_real_name_round_trip(opsin_gate):
    """Invariant 9: verify what is EMITTED afterwards, not just that the bad path
    stopped. A carboxy-bearing ring substituent must now denote the right molecule.
    """
    from rdkit.Chem import inchi

    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse

    smiles = "OC(=O)C1CCCCC1"          # cyclohexanecarboxylic acid
    emitted = Orthonym().name(smiles)
    parsed = opsin_parse(emitted)
    if not parsed:
        pytest.skip(f"OPSIN could not parse {emitted!r}")

    def skel(s):
        m = Chem.MolFromSmiles(s)
        Chem.RemoveStereochemistry(m)
        return inchi.MolToInchiKey(m).split("-")[0]

    assert skel(parsed) == skel(smiles), (
        f"{emitted!r} does not denote the input constitution"
    )
