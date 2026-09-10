"""
Tests for find_principal_chain behavior with FG carbons.

IUPAC: The principal chain must contain the principal
characteristic group and be the longest chain meeting all criteria.

Non-principal FG terminal carbons whose prefix form includes the carbon
AND has no in-chain oxo+heteroatom expansion (e.g., a non-principal nitrile
-> 'cyano') are excluded from chain enumeration. This prevents chain inflation
and ensures correct round-trip naming.

A NON-PRINCIPAL nitrile is the 'cyano' prefix whose carbon belongs to the
prefix (-C#N), NOT the parent chain, so it is EXCLUDED (Phase B / DD1
Fix 1,. When the nitrile IS the principal group its carbon
stays in the chain (the -nitrile suffix counts it).

W3-P03-5, BB 30384): a non-principal PRIMARY amide -CO-NH2 at a
chain end is DIFFERENT — per the Blue Book its carbon STAYS in the chain and
the group is expressed as 'oxo' (=O) + 'amino' (-NH2): "4-amino-4-oxobutanoic
acid" (PIN), NOT the non-PIN "3-carbamoylpropanoic acid". So primary_amide is
NOT excluded from chain enumeration (mirroring the acid-halide oxo+halo and
amidine amino+imino decisions). The 'carbamoyl' prefix remains the PIN only
for a RING-attached amide, where the carbon cannot join the ring.

Ref: IUPAC 2013 (amide amino+oxo, chain end),
 (cyano-prefix exclusion).
"""

import pytest
from rdkit import Chem

from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups


class TestChainFGBehavior:
    """Test principal chain behavior with functional group carbons."""

    def test_linear_amide_acid_includes_amide_carbon(self):
        """NC(=O)CCCC(=O)O -- 5-amino-5-oxopentanoic acid (PIN).

        W3-P03-5, BB 30384): the chain-end primary amide C STAYS in
        the chain, expressed as oxo+amino. Chain = 5 carbons (pentanoic acid).
        Round-trip: OPSIN parses "5-amino-5-oxopentanoic acid" correctly.
        """
        mol = Chem.MolFromSmiles("NC(=O)CCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        cooh_carbon = fgs["carboxylic_acid"][0][0]
        amide_carbon = fgs["primary_amide"][0][0]
        assert cooh_carbon in chain, "COOH carbon must be in chain"
        assert amide_carbon in chain, (
            f"Amide C (atom {amide_carbon}) must STAY in the chain "
            f"(P-65.1.6.1: expressed as oxo+amino, not carbamoyl). Chain: {chain}"
        )
        assert len(chain) == 5, (
            f"Chain should be 5 carbons (pentanoic acid), got {len(chain)}"
        )

    def test_linear_shorter_amide_acids(self):
        """NC(=O)CCC(=O)O and NC(=O)CC(=O)O -- shorter linear amide acids.

        W3-P03-5: the chain-end amide C stays in the chain
        (oxo+amino), so the chain is one carbon LONGER than the old carbamoyl
        form -> 4-amino-4-oxobutanoic acid / 3-amino-3-oxopropanoic acid (PINs).
        """
        for smiles, expected_len, name in [
            ("NC(=O)CCC(=O)O", 4, "4-amino-4-oxobutanoic acid"),
            ("NC(=O)CC(=O)O", 3, "3-amino-3-oxopropanoic acid"),
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

    def test_glutamine_like_chain_includes_amide_carbon(self):
        """NC(=O)CCCC(N)C(=O)O -- glutamine-like compound.

        W3-P03-5: the chain-end amide C STAYS in the chain
        (oxo+amino). Chain = 6 carbons: COOH-C(NH2)-C-C-C-C(=O amide) ->
        2,6-diamino-6-oxohexanoic acid (PIN).
        """
        mol = Chem.MolFromSmiles("NC(=O)CCCC(N)C(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        assert len(chain) == 6, (
            f"Chain should be 6 carbons (hexanoic acid backbone), "
            f"got {len(chain)}. Chain: {chain}"
        )
        cooh_carbon = fgs["carboxylic_acid"][0][0]
        amide_carbon = fgs["primary_amide"][0][0]
        assert cooh_carbon in chain, "COOH carbon must be in chain"
        assert amide_carbon in chain, (
            "Amide C stays in chain (P-65.1.6.1: oxo+amino, not carbamoyl)"
        )

    def test_nitrile_amino_acid_excludes_nitrile_carbon(self):
        """N#CCCCC(N)C(=O)O -- amino acid with a nitrile side chain.

         Phase B (DD1 Fix 1,: the carboxylic acid is the
        principal group, so the nitrile is a non-principal 'cyano' PREFIX whose
        carbon (-C#N) is EXCLUDED from the parent chain. The backbone is the
        5-carbon pentanoic acid (-> 2-amino-5-cyanopentanoic acid), NOT a 6-carbon
        chain. (Previously the nitrile C was wrongly counted, inflating the chain
        and round-tripping to a different molecule.)
        """
        mol = Chem.MolFromSmiles("N#CCCCC(N)C(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, principal_group="carboxylic_acid")

        nitrile_matches = fgs.get("nitrile", [])
        assert nitrile_matches, "nitrile should be detected"
        nitrile_carbon = nitrile_matches[0][0]
        assert nitrile_carbon not in chain, (
            f"Non-principal nitrile C (atom {nitrile_carbon}) must be EXCLUDED "
            f"(cyano prefix includes its carbon, P-66.5.1.1.4). Chain: {chain}"
        )
        assert len(chain) == 5, (
            f"Chain should be 5 carbons (nitrile C excluded), got {len(chain)}"
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
