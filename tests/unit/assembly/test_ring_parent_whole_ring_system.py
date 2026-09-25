"""A 'cycloalkane' parent is built only for a WHOLE monocyclic ring system.

 "Saturated monocyclic hydrocarbons" (the Blue Book): "The names of
saturated monocyclic hydrocarbons are formed by attaching the nondetachable prefix
'cyclo' to the name of the acyclic saturated unbranched hydrocarbon with the same
number of carbon atoms." A ring that shares atoms with other rings is part of a
polycyclic ring system, which is named as a whole von Baeyer, spiro,
 fusion; for ortho-fused systems the PIN is the hydro-fusion name,
"Fused ring systems and mancude ring assemblies composed of fused ring systems",
:24221, "decahydronaphthalene (PIN) bicyclo[4.4.0]decane").

Before the fix, the general_acyclic catch-all took ONE ring of a fused polycycle as
a 'cyclo' parent and named the other rings' atoms as open-chain substituents, so
their ring bonds vanished: '1-henicosylhydroxy-4,4-dimethyloxocyclohexane-1-
carboxylic acid' (C30H56O4) for a C30H46O4 pentacyclic triterpenoid, and
'(1E,5Z,7S)-1,5-dimethyl-6-nonylcyclonona-1,5-diene' for a tricyclic
sesquiterpene. With the OPSIN validity gate off (the test default) the PIN tier
shipped them.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.handlers._handler_shared import _is_monocyclic_ring_system
from tests.support.rt_assert import assert_tier_contract

pytestmark = pytest.mark.unit


def _ring_of(smiles, size):
    mol = Chem.MolFromSmiles(smiles)
    ring = next(r for r in mol.GetRingInfo().AtomRings() if len(r) == size)
    return mol, ring


@pytest.mark.parametrize("smiles,size,expected", [
    ("C1CCCCC1", 6, True),               # cyclohexane
    ("c1ccc(cc1)C1CCCCC1", 6, True),     # ring assembly: joined by a non-ring bond
    ("C1CCC2CCCCC2C1", 6, False),        # decalin: fused
    ("C1CCC2(CC1)CCCC2", 6, False),      # spiro[4.5]decane
    ("C1CC2CCC1C2", 5, False),           # norbornane: bridged
    ("C1CC2CC1C2", 4, False),            # bicyclo[2.1.1]hexane
])
def test_is_monocyclic_ring_system(smiles, size, expected):
    mol, ring = _ring_of(smiles, size)
    assert _is_monocyclic_ring_system(mol, ring) is expected


# Rows of the pre-existing-failures triage (TRIAGE.csv / TRIAGE_calls.csv) whose
# gate-off PIN name was a monocycle + open chain. Each is a fused polycycle; the
# best-effort tier names it RT-exact (a von Baeyer name, not the PIN:,
# and the PIN tier may only fail closed or ship an RT-exact name.
_FUSED_POLYCYCLES = [
    # row 19, CHEBI:177888 (was '...-6-nonylcyclonona-1,5-diene')
    "C/C1=C2\\C[C@@]3(C)CCC(C)(C)[C@H]3[C@@H]2CC/C(C)=C/CC1",
    # row 103, rt-25 (was '1-henicosylhydroxy-4,4-dimethyloxocyclohexane-...')
    "CC1(C)CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CC(O)C(=O)C(C)(C)C5CCC43C)C2C1",
    # canary 68 (was '...-3-octyloxocyclopentane-1-carboxylic acid')
    "CC1=C(O)C(=O)[C@]2(O)C[C@H]3C[C@](C)(C(=O)O)C[C@H]3[C@]12C",
    # canary 180 (was '5-henicosyl-4-methylcyclohexan-1-ol')
    "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",
    # canary 181 (was '...-1-tridecylcyclopentanediol')
    "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",
    # canary 202/316/338 (was '(1R,2R,3S)-2-hexadecyl-1-methyl-...cyclopentane')
    "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12",
    # canary 353 (was '(1S,6S)-1-henicosylhydroxy-...cyclohexane-1-carboxylic acid')
    "CC1(C)CC[C@]2(C(=O)O)CC[C@]3(C)C(=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)"
    "[C@@H]5CC[C@]43C)[C@@H]2C1",
    # canary 440 (was '...-4-undecylcyclohept-1-ene-1-carbaldehyde')
    "CC(C)C1=C2[C@H]3CC=C(C=O)CC(=O)[C@]3(C)CC[C@@]2(C)CC1",
    # canary 600 (was '4-[...-2,5-dihydrofuranyl]-2,3-dimethylcyclohexan-1-ol')
    "COC1OC2(OC)CC3CCC(O)C(C)C3(C)C(OC)C2=C1C",
]


@pytest.mark.parametrize("smiles", _FUSED_POLYCYCLES)
def test_fused_polycycle_never_named_as_monocycle(smiles):
    assert_tier_contract(smiles)


# The same defect in the ester alcohol word: `rules/esters.get_alkyl_fragment_name`
# fell back to a CARBON COUNT for a ring fragment the organyl primitive declined,
# and its 'n-phenylalkyl' form counted every non-phenyl carbon as chain. With the
# gate off the PIN tier shipped these (gate-off names recorded before the fix):
_RING_ESTERS = [
    # canary call 354: was '12-phenyldodecyl acetate' (C20H32O2 for C20H22O3)
    "CC(=O)OC1C(c2ccccc2)CCC(c2ccccc2)C1O",
    # was 'octyl acetate'; the PIN tier now ships the RT-exact
    # '1-methylbicyclo[2.2.1]heptan-2-yl acetate'
    "CC(=O)OC1CC2CCC1(C)C2",
]


@pytest.mark.parametrize("smiles", _RING_ESTERS)
def test_ring_ester_alcohol_never_named_as_open_chain(smiles):
    assert_tier_contract(smiles)


@pytest.mark.parametrize("smiles,atoms_from,expected", [
    # the fragment the count form mis-named, and controls it must keep
    ("CC(=O)OC1C(c2ccccc2)CCC(c2ccccc2)C1O", 4, ""),
    # bornyl acetate: the word was 'decyl'. (With the gate off the PIN tier
    # then reaches a different, OPSIN-unparseable general_acyclic string; that
    # producer is recorded in TRIAGE.md, Task 4 residuals.)
    ("CC1(C)C2CCC1(C)C(OC(C)=O)C2", None, ""),
    ("CC(=O)OCc1ccccc1", 4, "benzyl"),
    ("CC(=O)OCCCc1ccccc1", 4, "3-phenylpropyl"),
])
def test_ester_alkyl_word_ring_shapes(smiles, atoms_from, expected):
    from rdkit import Chem
    from orthonym.rules.esters import get_alkyl_fragment_name
    mol = Chem.MolFromSmiles(smiles)
    if atoms_from is None:  # bornyl: every atom but the acetyl C-C(=O)-O
        acetyl = {a.GetIdx() for a in mol.GetAtoms()
                  if a.GetIdx() in mol.GetSubstructMatch(Chem.MolFromSmarts("[CH3]C(=O)O"))}
        alkyl = [i for i in range(mol.GetNumAtoms()) if i not in acetyl]
    else:
        alkyl = list(range(atoms_from, mol.GetNumAtoms()))
    word = get_alkyl_fragment_name(mol, alkyl)
    if expected:
        assert word == expected
    else:
        # never a carbon-count chain word ('decyl', '12-phenyldodecyl')
        assert not word.endswith("decyl"), word
