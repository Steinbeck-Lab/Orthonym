"""
Unit tests for orient_chain alphabetical tiebreaker (criterion e).

IUPAC 2013 (g): when substituent locant sets are identical in both
directions, the orientation giving the lowest locant to the alphabetically
first substituent prefix is preferred.

Reference: IUPAC 2013 Blue Book,,,
"""

import pytest
from rdkit import Chem

from orthonym.rules.locants import orient_chain, build_atom_to_locant
from orthonym.perception.chains import find_principal_chain


def _get_subs_by_atom(mol, chain):
    """Helper: get substituent_positions dict keyed by atom idx."""
    from orthonym.perception.chains import _bfs_substituent

    chain_set = set(chain)
    subs = {}
    for chain_idx in chain:
        chain_atom = mol.GetAtomWithIdx(chain_idx)
        position_subs = []
        for neighbor in chain_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in chain_set:
                continue
            sub_atoms = _bfs_substituent(mol, nbr_idx, chain_set)
            position_subs.append(sub_atoms)
        if position_subs:
            subs[chain_idx] = position_subs
    return subs


def _get_sub_carbon_map(mol, chain, subs, result):
    """Build {locant: carbon_count} map for substituents on the oriented chain."""
    locant_map = build_atom_to_locant(result)
    info = {}
    for atom_idx, sub_list in subs.items():
        locant = locant_map.get(atom_idx)
        if locant is not None:
            carbons = sum(
                1 for a in sub_list[0]
                if mol.GetAtomWithIdx(a).GetSymbol() == "C"
            )
            info[locant] = carbons
    return info


class TestOrientChainAlphabeticalTiebreaker:
    """Tests for criterion (e) in orient_chain -- alphabetical tiebreaker."""

    def test_symmetric_heptane_ethyl_before_methyl(self):
        """3-ethyl-5-methylheptane: criterion (d) ties at {3,5}/{3,5}.

        Criterion (e) should orient so ethyl (alphabetically first) gets
        the lower locant 3 (not methyl at 3).
        """
        # CCC(CC)CC(C)CC: heptane with ethyl at one end, methyl at other
        mol = Chem.MolFromSmiles("CCC(CC)CC(C)CC")
        chain = find_principal_chain(mol, set(), {})
        assert len(chain) == 7, f"Expected 7-carbon chain, got {len(chain)}"

        subs = _get_subs_by_atom(mol, chain)
        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=subs,
        )

        sub_info = _get_sub_carbon_map(mol, chain, subs, result)
        # Ethyl (2 carbons) should have locant 3, methyl (1 carbon) locant 5
        assert sub_info.get(3) == 2, (
            f"Ethyl (2C) should be at locant 3, got sub_info={sub_info}"
        )
        assert sub_info.get(5) == 1, (
            f"Methyl (1C) should be at locant 5, got sub_info={sub_info}"
        )

    def test_symmetric_nonane_ethyl_before_methyl(self):
        """4-ethyl-6-methylnonane: criterion (d) ties at {4,6}/{4,6}.

        Criterion (e) should orient so ethyl (alphabetically first) gets
        the lower locant 4 (not methyl at 4).
        """
        # CCCC(CC)CC(C)CCC: nonane with ethyl at position 4, methyl at 6
        mol = Chem.MolFromSmiles("CCCC(CC)CC(C)CCC")
        chain = find_principal_chain(mol, set(), {})
        assert len(chain) == 9, f"Expected 9-carbon chain, got {len(chain)}"

        subs = _get_subs_by_atom(mol, chain)
        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=subs,
        )

        sub_info = _get_sub_carbon_map(mol, chain, subs, result)
        # Ethyl (2 carbons) should get lower locant, methyl gets higher
        ethyl_locant = None
        methyl_locant = None
        for loc, carbons in sub_info.items():
            if carbons == 2:
                ethyl_locant = loc
            elif carbons == 1:
                methyl_locant = loc

        assert ethyl_locant is not None and methyl_locant is not None
        assert ethyl_locant < methyl_locant, (
            f"Ethyl should have lower locant than methyl: "
            f"ethyl={ethyl_locant}, methyl={methyl_locant}"
        )

    def test_criterion_d_resolves_before_e(self):
        """When criterion (d) resolves (different locant sets), (e) is NOT applied.

        CCC(CC)CC(C)C is a hexane: forward={3,5}, reverse={2,4}.
        Criterion (d) picks reverse {2,4}. No need for criterion (e).
        """
        mol = Chem.MolFromSmiles("CCC(CC)CC(C)C")
        chain = find_principal_chain(mol, set(), {})
        assert len(chain) == 6

        subs = _get_subs_by_atom(mol, chain)
        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=subs,
        )

        locant_map = build_atom_to_locant(result)
        sub_locants = sorted(locant_map[a] for a in subs if a in set(result))
        # Criterion (d) picks {2, 4} over {3, 5}
        assert sub_locants == [2, 4], (
            f"Criterion (d) should give locant set [2, 4], got {sub_locants}"
        )

    def test_same_substituent_type_ties_returns_deterministic(self):
        """Two methyl groups at symmetric positions -- criterion (e) ties.

        Should return consistent result (deterministic).
        2,4-dimethylpentane: CC(C)CC(C)C
        """
        mol = Chem.MolFromSmiles("CC(C)CC(C)C")
        chain = find_principal_chain(mol, set(), {})
        subs = _get_subs_by_atom(mol, chain)

        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=subs,
        )

        locant_map = build_atom_to_locant(result)
        sub_locants = sorted(locant_map[a] for a in subs if a in set(result))
        # Both directions give {2, 4} for this 5-carbon chain
        assert sub_locants == [2, 4], f"Expected [2, 4], got {sub_locants}"

    def test_integration_symmetric_heptane_naming(self):
        """Integration: name_compound on symmetric heptane with ethyl+methyl.

        Should produce '3-ethyl-5-methylheptane' (ethyl at lower locant).
        """
        from orthonym import name_compound

        name = name_compound("CCC(CC)CC(C)CC")
        assert "3-ethyl" in name, f"Expected '3-ethyl' in name, got: {name}"
        assert "5-methyl" in name, f"Expected '5-methyl' in name, got: {name}"

    def test_reversed_input_still_picks_alpha_order(self):
        """Even when input chain is reversed (methyl-first), criterion (e) corrects.

        This test explicitly passes the chain in the "wrong" order to verify
        that criterion (e) actively picks the alphabetical orientation rather
        than relying on which direction the chain was discovered.
        """
        mol = Chem.MolFromSmiles("CCC(CC)CC(C)CC")
        chain = find_principal_chain(mol, set(), {})
        assert len(chain) == 7

        # Reverse the chain so methyl gets the lower locant initially
        reversed_chain = list(reversed(chain))
        subs = _get_subs_by_atom(mol, reversed_chain)

        result = orient_chain(
            chain=reversed_chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=subs,
        )

        sub_info = _get_sub_carbon_map(mol, reversed_chain, subs, result)
        # Even with reversed input, ethyl should still get the lower locant
        ethyl_locant = None
        methyl_locant = None
        for loc, carbons in sub_info.items():
            if carbons == 2:
                ethyl_locant = loc
            elif carbons == 1:
                methyl_locant = loc

        assert ethyl_locant is not None and methyl_locant is not None
        assert ethyl_locant < methyl_locant, (
            f"Ethyl should have lower locant than methyl even with reversed input: "
            f"ethyl={ethyl_locant}, methyl={methyl_locant}"
        )

    def test_no_substituents_returns_forward(self):
        """Chain with no substituents should return forward unchanged."""
        mol = Chem.MolFromSmiles("CCCCC")
        chain = find_principal_chain(mol, set(), {})

        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
            substituent_positions=None,
        )
        assert result == chain
