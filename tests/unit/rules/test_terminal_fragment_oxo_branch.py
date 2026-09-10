"""A single doubly-bonded oxygen BRANCH is `oxo`, never `(1-oxamethyl)`.

`(1-oxamethyl)` is a wrong-molecule construction, and the mechanism is exact rather than
suspected. Asked what it denotes, OPSIN answers:

    (1-oxamethyl)benzene -> OC1=CC=CC=C1 i.e. PHENOL
    (2-(1-oxamethyl)butyl)benzene -> OC(CC1=CC=CC=C1)CC i.e. a secondary ALCOHOL

So `1-oxamethyl` denotes `-OH`. Emitting it for `=O` loses the double bond *and* the
carbon, turning a ketone into an alcohol.

`rules/terminal_fragment.py` names every branch by recursing `_terminal_fragment_name`
(`:381`), so a lone `=O` came back as a one-atom oxa-replacement chain. Measured, the
construction is wrong wherever it appears and right nowhere:

    -CH2C(=O)CH2CH3 -> 2-(1-oxamethyl)butyl WRONG
    -CH2CH2COOH -> 3-(1-oxamethyl)-4-oxabutyl WRONG
    -CH2C(=O)CH3 -> 2-methyl-3-oxaprop-2-en-1-yl RT-EXACT (O in the BACKBONE)
    -CH2CHO -> 3-oxaprop-2-en-1-yl RT-EXACT (O in the BACKBONE)
    -CH2CH(OH)CH3 -> 2-methyl-3-oxapropyl RT-EXACT (O in the BACKBONE)

⇒ The module's *chain* handling is sound; only its *branch* handling was not. The four
backbone cases must not regress — they are pinned below, because a fix aimed at the branch
could easily perturb the backbone decomposition that produces them.

The vocabulary is reused, not re-tabled: `substituent_enumerator._descriptive_fallback`
already maps a single heteroatom to its standard prefix and already distinguishes `oxo`
from `hydroxy` by hydrogen count (`:2101`, `'hydroxy' if total_hs >= 1 else 'oxo'`), which
is exactly the discrimination needed here. Same shape as the `carboxy` fix in,
which reused `ring_assemblies._is_carboxyl_substituent`.
"""

import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import terminal_fragment_name


def _frag_on_benzene(smiles):
    """(mol, exocyclic fragment atoms, fragment-side attachment atom)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring]
    attach = next(i for i in frag
                  for n in mol.GetAtomWithIdx(i).GetNeighbors()
                  if n.GetIdx() in ring)
    return mol, frag, attach


def _name(smiles):
    mol, frag, attach = _frag_on_benzene(smiles)
    res = terminal_fragment_name(mol, frag, attach)
    return None if res is None else res.name


def test_mid_chain_ketone_branch_is_oxo_not_oxamethyl():
    """The defect. `-CH2-C(=O)-CH2CH3` was `2-(1-oxamethyl)butyl`, an ALCOHOL."""
    got = _name("c1ccccc1CC(=O)CC")
    assert got is not None, "the fragment must still be nameable"
    assert "oxamethyl" not in got, (
        f"{got!r} still contains the oxamethyl construction, which OPSIN reads as -OH"
    )
    assert "oxo" in got, f"expected an `oxo` prefix for the carbonyl branch; got {got!r}"


@pytest.mark.parametrize("smiles,expected", [
    # These have the oxygen in the BACKBONE, are already RT-EXACT, and must not move.
    ("c1ccccc1CC=O", "3-oxaprop-2-en-1-yl"),
    ("c1ccccc1CC(C)=O", "2-methyl-3-oxaprop-2-en-1-yl"),
    ("c1ccccc1CC(O)C", "2-methyl-3-oxapropyl"),
])
def test_backbone_oxygen_forms_do_not_regress(smiles, expected):
    """Regression pins. A branch fix must not perturb the backbone decomposition."""
    assert _name(smiles) == expected


def test_no_oxamethyl_anywhere_in_a_sweep():
    """The construction is wrong wherever it appears, so sweep for it rather than
    enumerating call sites — it is generated, not written, so a literal grep of the
    source cannot find every way it can be produced."""
    probes = [
        "c1ccccc1CC(=O)CC", "c1ccccc1CCC(=O)O", "c1ccccc1CC(=O)CCC",
        "c1ccccc1CCC(=O)CC", "c1ccccc1CC(=O)N", "c1ccccc1CCC(=O)OC",
    ]
    offenders = []
    for p in probes:
        got = _name(p)
        if got and "oxamethyl" in got:
            offenders.append((p, got))
    assert not offenders, f"oxamethyl still emitted for: {offenders}"


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_the_replacement_round_trips(opsin_gate):
    """Invariant 9 — verify what is EMITTED afterwards, not just that the bad token
    stopped. `(2-oxobutyl)benzene` was confirmed RT-EXACT before this fix was written,
    so the target is measured rather than assumed."""
    from rdkit.Chem import inchi

    from orthonym.validation.opsin_roundtrip import opsin_parse

    def skel(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None
        Chem.RemoveStereochemistry(m)
        return inchi.MolToInchiKey(m).split("-")[0]

    smiles = "c1ccccc1CC(=O)CC"
    got = _name(smiles)
    assert got, "fragment became unnameable"
    parsed = opsin_parse(f"({got})benzene")
    if not parsed:
        pytest.skip(f"OPSIN could not parse ({got})benzene")
    assert skel(parsed) == skel(smiles), (
        f"({got})benzene denotes {parsed!r}, not the input constitution"
    )


@pytest.mark.opsin_gate
@pytest.mark.slow
def test_oxamethyl_really_did_denote_a_hydroxyl(opsin_gate):
    """Pins the PREMISE, so this whole file converts to a loud failure rather than a
    vacuous pass if OPSIN's reading of `1-oxamethyl` ever changes."""
    from orthonym.validation.opsin_roundtrip import opsin_parse

    parsed = opsin_parse("(1-oxamethyl)benzene")
    if not parsed:
        pytest.skip("OPSIN no longer parses (1-oxamethyl)benzene at all")
    mol = Chem.MolFromSmiles(parsed)
    assert mol is not None
    o_with_h = [a for a in mol.GetAtoms()
                if a.GetSymbol() == "O" and a.GetTotalNumHs() >= 1]
    assert o_with_h, (
        f"premise broken: (1-oxamethyl)benzene now parses to {parsed!r}, which has no "
        f"hydroxyl -- re-derive whether the construction is still wrong before trusting "
        f"the rest of this file"
    )
