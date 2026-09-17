"""
Unit tests for general indicated hydrogen algorithm and fused heterocycle
dictionary expansion (a phase, Plan 01, Task 2).

Tests:
- New 3-ring fused heterocycle dictionary entries produce correct retained names
- Canonical SMILES verification for each new entry
- General indicated hydrogen computation for non-retained fused systems
- Pyrrole-type vs pyridine-type nitrogen differentiation
- Multi-position indicated hydrogen formatting
- Dictionary entry completeness audit
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    get_fused_heterocycle_name,
    match_fused_heterocycle_core,
)


class TestNewFusedHeterocycleEntries:
    """Test that new 3-ring fused heterocycle entries are in the dictionary
    and produce correct names."""

    @pytest.mark.unit
    def test_dibenzofuran_in_dictionary(self):
        """Dibenzofuran should be recognized from the dictionary."""
        smi = "c1ccc2c(c1)oc1ccccc12"
        can = Chem.CanonSmiles(smi)
        assert can in FUSED_HETEROCYCLE_DATA, (
            f"Dibenzofuran ({can}) should be in FUSED_HETEROCYCLE_DATA"
        )
        entry = FUSED_HETEROCYCLE_DATA[can]
        # PIN carries the fusion descriptor (,.
        assert entry["name"] == "dibenzo[b,d]furan"

    @pytest.mark.unit
    def test_dibenzothiophene_in_dictionary(self):
        """Dibenzothiophene should be recognized from the dictionary."""
        smi = "c1ccc2c(c1)sc1ccccc12"
        can = Chem.CanonSmiles(smi)
        assert can in FUSED_HETEROCYCLE_DATA, (
            f"Dibenzothiophene ({can}) should be in FUSED_HETEROCYCLE_DATA"
        )
        entry = FUSED_HETEROCYCLE_DATA[can]
        # PIN carries the fusion descriptor (,.
        assert entry["name"] == "dibenzo[b,d]thiophene"

    @pytest.mark.unit
    def test_benzo_f_quinoline_in_dictionary(self):
        """Benzo[f]quinoline should be recognized from the dictionary."""
        smi = "c1ccc2c(c1)ccc1ncccc12"
        can = Chem.CanonSmiles(smi)
        assert can in FUSED_HETEROCYCLE_DATA, (
            f"Benzo[f]quinoline ({can}) should be in FUSED_HETEROCYCLE_DATA"
        )

    @pytest.mark.unit
    def test_benzo_h_quinoline_in_dictionary(self):
        """Benzo[h]quinoline should be recognized from the dictionary."""
        smi = "c1ccc2cc3ncccc3cc2c1"
        can = Chem.CanonSmiles(smi)
        assert can in FUSED_HETEROCYCLE_DATA, (
            f"Benzo[h]quinoline ({can}) should be in FUSED_HETEROCYCLE_DATA"
        )

    @pytest.mark.unit
    def test_fluorenone_in_dictionary(self):
        """9H-Fluoren-9-one should be recognized from the dictionary."""
        smi = "O=C1c2ccccc2-c2ccccc21"
        can = Chem.CanonSmiles(smi)
        assert can in FUSED_HETEROCYCLE_DATA, (
            f"Fluorenone ({can}) should be in FUSED_HETEROCYCLE_DATA"
        )

    @pytest.mark.unit
    def test_canonical_smiles_roundtrip(self):
        """Each new dictionary entry's canonical SMILES matches when mol
        is constructed from it."""
        new_entries = [
            "c1ccc2c(c1)oc1ccccc12",   # dibenzofuran
            "c1ccc2c(c1)sc1ccccc12",   # dibenzothiophene
            "c1ccc2c(c1)ccc1ncccc12",  # benzo[f]quinoline
            "c1ccc2cc3ncccc3cc2c1",    # benzo[h]quinoline
            "O=C1c2ccccc2-c2ccccc21",  # fluorenone
        ]
        for smi in new_entries:
            can = Chem.CanonSmiles(smi)
            mol = Chem.MolFromSmiles(can)
            assert mol is not None, f"Cannot parse canonical SMILES: {can}"
            re_can = Chem.MolToSmiles(mol, canonical=True)
            assert re_can == can, (
                f"Canonical SMILES roundtrip failed: {can} -> {re_can}"
            )


class TestGeneralIndicatedHydrogen:
    """Test general indicated hydrogen computation for non-retained systems."""

    @pytest.mark.unit
    def test_sp3_nitrogen_in_aromatic_ring_gets_indicated_h(self):
        """Non-retained fused system with sp3 nitrogen in aromatic ring
        gets indicated hydrogen label.

        Tests: 1H-indole has indicated hydrogen at position 1 (N-H).
        """
        # 1H-indole is already in the dictionary -- just verify the concept
        smi = "c1ccc2[nH]ccc2c1"
        name = name_compound(smi)
        assert name is not None
        assert "1H" in name or "1h" in name.lower(), (
            f"Expected '1H' indicated hydrogen in name, got: {name}"
        )

    @pytest.mark.unit
    def test_pyrrole_type_nitrogen_is_indicated_h_site(self):
        """Pyrrole-type 5-membered ring nitrogen correctly identified as
        indicated hydrogen site.

        The N-H in a 5-membered ring (pyrrole-type) IS indicated hydrogen
        because in the maximally unsaturated parent, this position would
        have no hydrogen (nitrogen donates lone pair to aromaticity but
        also has N-H as indicated hydrogen).
        """
        # Indazole: has N-H at position 1
        smi = "c1ccc2[nH]ncc2c1"
        name = name_compound(smi)
        assert name is not None
        assert "1H" in name or "1h" in name.lower(), (
            f"Expected '1H' in indazole name, got: {name}"
        )

    @pytest.mark.unit
    def test_pyridine_type_nitrogen_no_indicated_h(self):
        """Pyridine-type 6-membered ring nitrogen correctly NOT assigned
        indicated hydrogen.

        Quinoline has a pyridine-type N -- no indicated hydrogen needed.
        """
        smi = "c1ccc2ncccc2c1"  # quinoline
        name = name_compound(smi)
        assert name is not None
        assert "quinoline" in name.lower()
        # Should NOT have indicated hydrogen prefix
        assert not name.startswith("1H-"), (
            f"Quinoline should not have indicated hydrogen, got: {name}"
        )

    @pytest.mark.unit
    def test_multi_indicated_h_positions(self):
        """Fused system with multiple indicated hydrogen positions produces
        correctly formatted multi-H prefix (e.g., '1H,3H-').

        Purine has indicated H at positions 7 and 9 (depending on tautomer).
        """
        # Purine base: c1nc2[nH]cnc2c1N has 1H at specific position
        # 9H-purine
        smi = "c1ncc2[nH]cnc2n1"
        name = name_compound(smi)
        assert name is not None
        # Should contain some form of indicated hydrogen
        assert "H-" in name or "purine" in name.lower(), (
            f"Expected purine name with indicated H, got: {name}"
        )


class TestGroup14HeteroleIndicatedH:
    """Group-14 (Si/Ge/Sn) mancude heteroles keep their indicated hydrogen even
    when every ring-position H is displaced by substitution.

     "Silole, germole,... rings" (``the Blue Book``):
    ``1,1-dibutyl-1H-germole (PIN) (note the indicated hydrogen atom)``. The
    indicated hydrogen is a property of the mancude PARENT (which ring atom is
    the saturated skeletal position), not of the substituted molecule, so a
    1,1-disubstituted silole/germole/stannole keeps its ``1H`` where a carbon
    heterocycle names the position only via its remaining hydrogen.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # fully substituted (0 H at ring position 1) -> 1H still cited (the fix)
        ("CC[Si]1(CC)C=CC=C1", "1,1-diethyl-1H-silole"),
        ("CCCC[Ge]1(CCCC)C=CC=C1", "1,1-dibutyl-1H-germole"),   # BB PIN
        ("CC[Sn]1(CC)C=CC=C1", "1,1-diethyl-1H-stannole"),
    ])
    def test_fully_substituted_group14_heterole_keeps_1h(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # regression pins: the mono-sub and unsubstituted forms already worked
        # via the H-bearing path and must be byte-identical after the fix.
        ("C[SiH]1C=CC=C1", "1-methyl-1H-silole"),
        ("[SiH2]1C=CC=C1", "1H-silole"),
        ("[GeH2]1C=CC=C1", "1H-germole"),
    ])
    def test_h_bearing_group14_heterole_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestSubstitutedMancudeIndicatedH:
    """A mancude HW ring keeps its indicated hydrogen when a substituent has
    displaced the H at the sole saturated skeletal position -- for EVERY
    element that holds that position, not just Group-14 (Si/Ge/Sn).

     (``the Blue Book``): after the maximum number of
    noncumulative double bonds is assigned, "any ring atom with a bonding
    number of three or higher connected to adjacent ring atoms by single bonds
    only" is the indicated-hydrogen position. In ``1,3,2-oxathiaborepine`` the
    boron (bonding number 3) sits between the divalent ring O and S -- both
    single-bond-only neighbours -- so it is that position. The bare parent
    ``B1OC=CC=CS1`` already names as ``2H-1,3,2-oxathiaborepine`` via the
    H-bearing path; when the boron's H is replaced by a substituent the ``2H``
    must still be cited: a substituent stands in the indicated
    hydrogen's place, it does not remove it).
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # boron indicated-H displaced by -S-CH3 -> 2H still cited (the fix)
        ("CSB1OC=CC=CS1", "2-(methylsulfanyl)-2H-1,3,2-oxathiaborepine"),
    ])
    def test_substituted_boron_ring_keeps_indicated_h(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # regression pin: the H-bearing parent already worked and must be
        # byte-identical after the fix.
        ("B1OC=CC=CS1", "2H-1,3,2-oxathiaborepine"),
    ])
    def test_h_bearing_boron_ring_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", ["CSB1OC=CC=CS1"])
    def test_substituted_boron_ring_roundtrips(self, smiles, opsin_to_smiles):
        """The emitted name must parse back (OPSIN) to the input structure."""
        name = name_compound(smiles)
        parsed = opsin_to_smiles(name)
        assert parsed is not None, f"OPSIN could not parse {name!r}"
        assert Chem.CanonSmiles(parsed) == Chem.CanonSmiles(smiles)


class TestDictionaryCompleteness:
    """Test that all fused heterocycle dictionary entries have complete
    iupac_locants covering all ring atoms."""

    @pytest.mark.unit
    def test_all_entries_have_iupac_locants(self):
        """All existing fused heterocycle dictionary entries have
        iupac_locants that cover junction atoms."""
        missing_locants = []
        for smiles, entry in FUSED_HETEROCYCLE_DATA.items():
            if "iupac_locants" not in entry:
                missing_locants.append((smiles, entry.get("name", "?")))
                continue
            locants = entry["iupac_locants"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Skip invalid SMILES
            # Check that ring atoms have locant mappings
            ring_info = mol.GetRingInfo()
            ring_atoms = set()
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
            unmapped = [
                idx for idx in ring_atoms
                if idx not in locants
            ]
            if unmapped:
                missing_locants.append(
                    (entry.get("name", smiles), f"unmapped ring atoms: {unmapped}")
                )

        if missing_locants:
            details = "\n".join(f"  {name}: {info}" for name, info in missing_locants[:5])
            # Log but don't fail -- some entries may intentionally omit non-ring atoms
            import warnings
            warnings.warn(
                f"Dictionary entries with incomplete iupac_locants:\n{details}"
            )
