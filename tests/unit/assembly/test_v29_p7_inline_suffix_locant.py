# tests/unit/assembly/test_v29_p7_inline_suffix_locant.py
"""v29 P7 Task 5: an inline suffix locant designates the atom it CONVERTS.

P-64.2.2.2 "Cyclic ketones" (BlueBookV2/BlueBookV2.md:28384 heading; sentence
at :28386) -- "As the formation of ketones is achieved by the conversion of a
methylene, >CH2, group into a >C=O group, the suffix 'one' with appropriate
locants can be added to the name of parent hydrides having such groups."

The shipped ketone SMARTS is ``[#6][CX3](=O)[#6]``
(perception/functional_groups.py:342), so a match carries BOTH flanking
carbons.  The general engine used to take ``min()`` over every match atom on
the parent, which cited a NEIGHBOUR of the carbonyl carbon.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import (
    _INLINE_SUFFIX_CORES, _inline_suffix_locant, name_general_ring,
)

pytestmark = pytest.mark.unit


def _features(smiles):
    from orthonym.namer import Orthonym
    nm = Orthonym(_disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    feats = nm._perceive(mol, smiles, canonical)
    nm._classify(feats)
    return mol, feats


def _carbonyl_carbon(mol):
    """The single ring carbon bearing an exocyclic =O."""
    hits = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetSymbol() == "C" and a.IsInRing()
            and any(b.GetBondTypeAsDouble() == 2.0
                    and b.GetOtherAtom(a).GetSymbol() == "O"
                    for b in a.GetBonds())]
    assert len(hits) == 1, f"test fixture expects exactly one ring C=O, got {hits}"
    return hits[0]


class TestInlineSuffixLocantHelper:
    def test_picks_carbonyl_carbon_not_flanking_carbon(self):
        """The ketone SMARTS leads with a FLANKING carbon; index 1 is the one."""
        mol = Chem.MolFromSmiles("CC(=O)C")          # propan-2-one
        match = mol.GetSubstructMatch(Chem.MolFromSmarts("[#6][CX3](=O)[#6]"))
        assert len(match) == 4
        parent = {0, 1, 3}                            # the three carbons
        a2l = {0: 1, 1: 2, 3: 3}
        # min() over the whole match would give 1 (a flanking carbon).
        assert _inline_suffix_locant("ketone", match, parent, a2l) == 2

    def test_returns_none_when_characteristic_atom_is_off_parent(self):
        """An acetyl hanging off the chosen chain has NO locant on that chain."""
        mol = Chem.MolFromSmiles("CCC(=O)C")
        match = mol.GetSubstructMatch(Chem.MolFromSmarts("[#6][CX3](=O)[#6]"))
        carbonyl = [i for i in match
                    if mol.GetAtomWithIdx(i).GetSymbol() == "C"
                    and any(b.GetBondTypeAsDouble() == 2.0
                            and b.GetOtherAtom(mol.GetAtomWithIdx(i)).GetSymbol() == "O"
                            for b in mol.GetAtomWithIdx(i).GetBonds())][0]
        parent = {i for i in match
                  if i != carbonyl and mol.GetAtomWithIdx(i).GetSymbol() == "C"}
        a2l = {i: n + 1 for n, i in enumerate(sorted(parent))}
        assert _inline_suffix_locant("ketone", match, parent, a2l) is None

    def test_one_is_registered_as_an_inline_suffix(self):
        assert "one" in _INLINE_SUFFIX_CORES


class TestRingSuffixLocant:
    @pytest.mark.parametrize("smiles", [
        "CC12CCCCC1CCCC2=O",     # 1-methylbicyclo[4.4.0]decan-?-one
        "CC12CCCC1CCCC2=O",      # 1-methylbicyclo[4.3.0]nonan-?-one
        "CC1CCC2CCCCC2C1=O",     # 4-methylbicyclo[4.4.0]decan-?-one
    ])
    def test_suffix_locant_equals_the_carbonyl_carbons_own_locant(self, smiles):
        """Regression: the emitted '-N-one' must be the carbonyl carbon's N.

        Pre-fix these emitted a flanking atom's locant, producing a carbon with
        five bonds (e.g. '1-methylbicyclo[4.4.0]decan-1-one').
        """
        mol, feats = _features(smiles)
        res = name_general_ring(mol, feats, allow_aromatic_general=True)
        assert res is not None, "general ring engine refused; fixture no longer valid"

        from orthonym.rules.vonbaeyer_universal import analyze_cage_universal
        cage = analyze_cage_universal(mol, allow_mancude=True)
        expected = cage.atom_to_locant[_carbonyl_carbon(mol)]
        assert f"-{expected}-one" in res.name, (
            f"{res.name!r} does not cite the carbonyl carbon's locant {expected}")

    def test_suffix_locant_is_never_a_substituted_bridgehead(self):
        """The failure mode this class produced: substituent and =O collide."""
        mol, feats = _features("CC12CCCCC1CCCC2=O")
        res = name_general_ring(mol, feats, allow_aromatic_general=True)
        assert res is not None
        assert "1-methyl" in res.name
        assert "-1-one" not in res.name, (
            f"{res.name!r} puts the ketone on the methyl-bearing bridgehead")


class TestOffChainCarbonylFailsClosed:
    def test_general_chain_engine_refuses_an_acyl_off_the_parent_chain(self):
        """C2-2: an acetyl off the chosen chain must not become a '-one' suffix.

        Swallowing it would put the acyl carbon and its methyl into
        ``suffix_atoms``, where they are never spelled -- the C19H38O name
        ``5-butyl-2,4-dimethyltridecan-4-one`` for a C21H42O molecule.

        SCOPE: this pins the GENERAL CHAIN ENGINE only.  Measured 2026-07-31,
        the shipped name for this molecule is produced by ``composer.py``
        (``DROP-09 substituent_skip: reason=universal_pipeline_unnameable
        locant=4``), a different producer that still drops the acetyl.  That
        remains open -- see 
        residue R1.  Do not widen this test to the whole namer until DROP-09 is
        fixed; it would then be asserting someone else's bug.
        """
        from orthonym.assembly.general_engine import name_general_chain
        mol, feats = _features("CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C")
        assert name_general_chain(mol, feats) is None
