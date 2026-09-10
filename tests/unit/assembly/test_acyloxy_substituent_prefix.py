"""An `-O-C(=O)-R` substituent is `<acyl>oxy`, not `<acyl>`.

Third member of one defect family, and the family is the point:
`substituent_naming.parent_to_prefix` maps an acid NAME to an ACYL prefix through
`_ACID_TO_ACYL_PREFIX = {'formic acid': 'formyl', 'acetic acid': 'acetyl', …}`, and that
mapping is wrong whenever the fragment retains an oxygen the acyl group does not have.
Which oxygen is lost depends only on where the fragment attaches:

    attached via the carbonyl C, retaining -OH -> `carboxy` (was `formyl`, 9e2648a2)
    attached via the ester O, retaining -O- -> `<acyl>oxy` (was `acetyl`, this file)

`parent_to_prefix` receives only a name, so it cannot tell the two apart — the caller has
the graph and must decide. Measured before the fix:

    -OC(=O)CH3 -> 'acetyl' which is -C(=O)CH3, an oxygen SHORT -> wrong molecule
    -C(=O)CH3 -> None (the real acetyl case is refused, separately)

Target verified by round-trip before being asserted: `(acetyloxy)benzene` parses to
`C(C)(=O)OC1=CC=CC=C1`, the same constitution as phenyl acetate, while `(acetyl)benzene`
parses to `C(C)(=O)C1=CC=CC=C1` — a different compound.

⚠ `parent_to_prefix` is deliberately NOT changed, for the third time: `rules/acid_halides.py:16`
documents `formic acid -> formyl` as correct for acid halides,
`decomposition/fragment_assembly.py:136` holds its own copy, and
`tests/unit/rules/test_v29_phase7_prefix_vocabulary.py:51` asserts its current return value.

⚠ Note `-OC(=O)CH2CH3` was already fixed by the `oxo`-branch commit (`78981848`): it now
spells `2-oxo-1-oxabutyl`, which round-trips exactly. So this file covers the case the
acid→acyl TABLE intercepts before the chain path can reach it, and
`test_propanoyloxy_still_uses_the_chain_form` pins that the two fixes do not fight.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _frag_on_benzene(smiles):
    """(mol, exocyclic fragment atoms, PARENT-side attachment atom).

    Parent-side is the convention `discover_substituents` supplies
    (`SubstituentInfo.attach_mol_idx`), and using the fragment-side index instead is how
    the first version of the `carboxy` guard passed its tests while being dead in
    production.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring]
    attach = next(n.GetIdx() for i in frag
                  for n in mol.GetAtomWithIdx(i).GetNeighbors()
                  if n.GetIdx() in ring)
    assert attach in ring, "helper must pass the PARENT-side atom"
    return mol, frag, attach


def _tok(smiles):
    mol, frag, attach = _frag_on_benzene(smiles)
    return name_substituent(mol, frag, attach, allow_mancude=True)


def test_acetyloxy_is_not_acetyl():
    """The defect: an oxygen dropped, i.e. a different molecule."""
    got = _tok("c1ccccc1OC(C)=O")
    assert got is not None, "the fragment must still be nameable"
    assert got != "acetyl", (
        "an -O-C(=O)CH3 fragment spelled 'acetyl' denotes -C(=O)CH3, one oxygen short"
    )


def test_acetyl_control_is_untouched():
    """The genuine acetyl case must not be turned into acetyloxy by an over-broad
    guard -- that would trade one wrong molecule for another. It currently refuses,
    which is a separate gap and NOT this fix's business."""
    got = _tok("c1ccccc1C(C)=O")
    assert got != "acetyloxy", f"real -C(=O)CH3 must never be acetyloxy; got {got!r}"


def test_propanoyloxy_still_uses_the_chain_form():
    """Pins that this fix and the oxo-branch fix (78981848) do not fight.
    `-OC(=O)CH2CH3` is handled by the chain path as `2-oxo-1-oxabutyl`, verified
    RT-exact, and must not regress to a refusal or to a bare acyl."""
    got = _tok("c1ccccc1OC(=O)CC")
    assert got is not None, "propanoyloxy regressed to a refusal"
    assert got != "propanoyl", "propanoyloxy spelled as the bare acyl drops an oxygen"


@pytest.mark.opsin_gate
@pytest.mark.slow
@pytest.mark.parametrize("smiles", [
    "c1ccccc1OC(C)=O",     # phenyl acetate
    "c1ccccc1OC(=O)CC",    # phenyl propanoate
])
def test_acyloxy_round_trips(opsin_gate, smiles):
    """Invariant 9 -- verify what is EMITTED, not just that the bad token stopped."""
    from rdkit.Chem import inchi

    from orthonym.validation.opsin_roundtrip import opsin_parse

    def skel(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None
        Chem.RemoveStereochemistry(m)
        return inchi.MolToInchiKey(m).split("-")[0]

    got = _tok(smiles)
    assert got, f"{smiles} became unnameable"
    parsed = opsin_parse(f"({got})benzene")
    if not parsed:
        pytest.skip(f"OPSIN could not parse ({got})benzene")
    assert skel(parsed) == skel(smiles), (
        f"({got})benzene denotes {parsed!r}, not the input constitution"
    )


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_the_premise_bare_acyl_really_is_a_different_molecule(opsin_gate):
    """Pins WHY this is a defect rather than a preference, so the file fails loudly
    instead of vacuously if OPSIN's reading ever changes."""
    from rdkit.Chem import inchi

    from orthonym.validation.opsin_roundtrip import opsin_parse

    def skel(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None
        Chem.RemoveStereochemistry(m)
        return inchi.MolToInchiKey(m).split("-")[0]

    target = skel("c1ccccc1OC(C)=O")
    bare = opsin_parse("(acetyl)benzene")
    oxy = opsin_parse("(acetyloxy)benzene")
    if not bare or not oxy:
        pytest.skip("OPSIN could not parse the premise probes")
    assert skel(bare) != target, "premise broken: (acetyl)benzene now IS phenyl acetate"
    assert skel(oxy) == target, "premise broken: (acetyloxy)benzene is not phenyl acetate"
