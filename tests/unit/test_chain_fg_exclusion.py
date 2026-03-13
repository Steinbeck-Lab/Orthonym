"""
Tests for find_principal_chain() behavior with FG carbons.

IUPAC P-44.1: The principal chain must contain the principal
characteristic group and be the longest chain meeting all criteria.

Non-principal FG terminal carbons whose prefix form includes the carbon
(e.g., carbamoyl = -C(=O)NH2) are excluded from chain enumeration.
This prevents chain inflation and ensures correct round-trip naming.

FG terminal carbons whose prefix does NOT include the carbon (e.g.,
cyano = #N on chain C) are NOT excluded.

Ref: IUPAC 2013 P-66.1(c) (carbamoyl), P-66.6.3 (cyano).
"""

import pytest
from rdkit import Chem

from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups


class TestChainFGBehavior:
    """Test principal chain behavior with functional group carbons."""

    def test_linear_amide_acid_excludes_amide_carbon(self):
        """NC(=O)CCCC(=O)O -- 4-carbamoylbutanoic acid.

        The amide C is excluded from the chain because carbamoyl prefix
        includes the C.  Chain = 4 carbons (butanoic acid).
        Round-trip: OPSIN parses "4-carbamoylbutanoic acid" correctly.
        """
        mol = Chem.MolFromSmiles("NC(=O)CCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        cooh_carbon = fgs["carboxylic_acid"][0][0]
        amide_carbon = fgs["primary_amide"][0][0]
        assert cooh_carbon in chain, "COOH carbon must be in chain"
        assert amide_carbon not in chain, (
            f"Amide C (atom {amide_carbon}) should be excluded from chain "
            f"(carbamoyl prefix includes it). Chain: {chain}"
        )
        assert len(chain) == 4, (
            f"Chain should be 4 carbons (butanoic acid), got {len(chain)}"
        )

    def test_linear_shorter_amide_acids(self):
        """NC(=O)CCC(=O)O and NC(=O)CC(=O)O -- shorter linear amide acids.

        Same principle: amide C excluded because carbamoyl prefix includes it.
        """
        for smiles, expected_len, name in [
            ("NC(=O)CCC(=O)O", 3, "3-carbamoylpropanoic acid"),
            ("NC(=O)CC(=O)O", 2, "2-carbamoylethanoic acid"),
        ]:
            mol = Chem.MolFromSmiles(smiles)
            fgs = detect_functional_groups(mol)
            chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")
            assert len(chain) == expected_len, (
                f"{name}: chain should be {expected_len}C, got {len(chain)}"
            )

    def test_secondary_amide_excluded_naturally(self):
        """CC(=O)NCCCCC(=O)O -- 6-acetamidohexanoic acid.

        The secondary amide C (in CH3-C(=O)-N-) connects to the chain
        through N, NOT through a C-C bond. The chain finder naturally
        excludes it because it only follows carbon-carbon paths.
        """
        mol = Chem.MolFromSmiles("CC(=O)NCCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        amide_carbon = 1
        assert amide_carbon not in chain, (
            f"Secondary amide C (atom {amide_carbon}) connects through N, "
            f"should not be in carbon chain. Chain: {chain}"
        )
        assert len(chain) == 5, (
            f"Chain should be 5 carbons (pentanoic acid backbone), got {len(chain)}"
        )

    def test_glutamine_like_chain_excludes_amide_carbon(self):
        """NC(=O)CCCC(N)C(=O)O -- glutamine-like compound.

        The amide C is excluded from the chain (carbamoyl prefix includes it).
        Chain = 5 carbons: COOH-C(NH2)-C-C-C = pentanoic acid backbone.
        The amide becomes a carbamoyl substituent at position 5.
        """
        mol = Chem.MolFromSmiles("NC(=O)CCCC(N)C(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        assert len(chain) == 5, (
            f"Chain should be 5 carbons (pentanoic acid), "
            f"got {len(chain)}. Chain: {chain}"
        )
        cooh_carbon = fgs["carboxylic_acid"][0][0]
        amide_carbon = fgs["primary_amide"][0][0]
        assert cooh_carbon in chain, "COOH carbon must be in chain"
        assert amide_carbon not in chain, (
            "Amide C excluded (carbamoyl prefix includes it)"
        )

    def test_nitrile_amino_acid_includes_nitrile_carbon(self):
        """N#CCCCC(N)C(=O)O -- amino acid with nitrile side chain.

        The nitrile C IS in the chain because "cyano" prefix does NOT
        include the C (it represents only the #N).
        """
        mol = Chem.MolFromSmiles("N#CCCCC(N)C(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        nitrile_matches = fgs.get("nitrile", [])
        assert nitrile_matches, "nitrile should be detected"
        nitrile_carbon = nitrile_matches[0][0]
        assert nitrile_carbon in chain, (
            f"Nitrile C (atom {nitrile_carbon}) IS a chain member "
            f"(cyano prefix = just #N). Chain: {chain}"
        )
        assert len(chain) == 6, (
            f"Chain should be 6 carbons (includes nitrile C), got {len(chain)}"
        )

    def test_diacid_keeps_both_principal_fg_carbons(self):
        """OC(=O)CCCCC(=O)O -- adipic acid (hexanedioic acid).

        Both COOH groups ARE the principal group. Both carbonyl carbons
        must stay in the chain. Expected chain = 6 carbons.
        """
        mol = Chem.MolFromSmiles("OC(=O)CCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        assert len(chain) == 6, (
            f"Chain should be 6 carbons (hexanedioic acid), got {len(chain)}. "
            f"Chain: {chain}"
        )

    def test_hexane_no_fg_unchanged(self):
        """CCCCCC -- hexane (no functional groups).

        No FG carbons to consider. Chain = 6 unchanged.
        """
        mol = Chem.MolFromSmiles("CCCCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group=None)

        assert len(chain) == 6, (
            f"Hexane chain should be 6 carbons, got {len(chain)}"
        )

    def test_propanoic_acid_principal_fg_carbon_stays(self):
        """CCC(=O)O -- propanoic acid.

        The COOH carbon IS the principal group. It stays in the chain.
        Expected chain = 3 carbons.
        """
        mol = Chem.MolFromSmiles("CCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        assert len(chain) == 3, (
            f"Propanoic acid chain should be 3 carbons, got {len(chain)}"
        )

    def test_amide_as_principal_group_keeps_amide_carbon(self):
        """NC(=O)CCCC -- pentanamide.

        When amide IS the principal group, the amide carbon stays in
        the chain. Expected chain = 5 carbons.
        """
        mol = Chem.MolFromSmiles("NC(=O)CCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="primary_amide")

        amide_matches = fgs.get("primary_amide", [])
        assert amide_matches, "primary_amide should be detected"
        amide_carbon = amide_matches[0][0]
        assert amide_carbon in chain, (
            f"Amide carbon (atom {amide_carbon}) should be IN the chain "
            f"when amide is the principal group. Chain: {chain}"
        )
        assert len(chain) == 5, (
            f"Pentanamide chain should be 5 carbons, got {len(chain)}"
        )

    def test_chain_contains_principal_group(self):
        """Verify that find_principal_chain always contains the PG carbon."""
        test_cases = [
            ("CCC(=O)O", "carboxylic_acid"),
            ("CCCCCC(=O)O", "carboxylic_acid"),
            ("NC(=O)CCCC(=O)O", "carboxylic_acid"),
            ("OC(=O)CCCCC(=O)O", "carboxylic_acid"),
        ]
        for smiles, pg in test_cases:
            mol = Chem.MolFromSmiles(smiles)
            fgs = detect_functional_groups(mol)
            chain = find_principal_chain(mol, fgs, principal_group=pg)
            pg_carbon = fgs[pg][0][0]
            assert pg_carbon in chain, (
                f"{smiles}: PG carbon (atom {pg_carbon}) must be in chain. "
                f"Chain: {chain}"
            )
