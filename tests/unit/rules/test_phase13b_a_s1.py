"""v23 Phase 13B(a) S1 — fusion-numbering engine wired into the PAH consumer.

Covers:
* ``get_polycyclic_iupac_locants`` derives numbering from the deterministic
  fusion engine for cataloged PAH whose ``iupac_numbering`` was never tabulated
  (tetracene/chrysene/triphenylene/benz[a]anthracene/pentacene/picene), and
  returns None (fail-closed) for systems the engine declines (fluorene: sp3 + a
  5-membered ring);
* the fused-catalog coverage guard rejects subset-hallucinations
  (pentacene contains a naphthacene substructure, picene contains chrysene) while
  every real catalog entry still self-matches;
* end-to-end substituted-PAH naming for the angular/branched/5-ring wins.
"""

import pytest
from rdkit import Chem
from rdkit import RDLogger

from orthonym.data.fused_heterocycles import (
    match_fused_heterocycle_core,
    _match_fused_heterocycle_core_impl,
)
from orthonym.data.polycyclic_data import POLYCYCLIC_DATA
from orthonym.rules.polycyclics import get_polycyclic_iupac_locants

RDLogger.DisableLog("rdApp.*")


@pytest.mark.unit
class TestEngineWiringIntoConsumer:
    """get_polycyclic_iupac_locants derives numbering via the engine when the
    tabulated map is empty; fail-closed when the engine declines."""

    @pytest.mark.parametrize("name", [
        "tetracene", "chrysene", "triphenylene",
        "benz[a]anthracene", "benzo[c]phenanthrene", "pentacene", "picene",
    ])
    def test_empty_catalog_numbering_supplied_by_engine(self, name):
        entry = POLYCYCLIC_DATA[name]
        assert not entry.get("iupac_numbering"), f"{name} unexpectedly tabulated"
        mol = Chem.MolFromSmiles(entry["canonical_smiles"])
        locants = get_polycyclic_iupac_locants(mol, name)
        assert locants is not None, f"{name}: engine numbering not wired"
        # covers every ring atom; peripheral atoms get int locants, fusion (n,'a')
        ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        assert set(locants) == ring_atoms
        assert any(isinstance(v, int) for v in locants.values())
        assert any(isinstance(v, tuple) for v in locants.values())

    def test_fluorene_engine_declines_returns_none(self):
        # fluorene has an sp3 C9 + a 5-membered ring -> not all-6 cata-fused;
        # the engine declines, so the consumer returns None (caller falls back).
        entry = POLYCYCLIC_DATA["fluorene"]
        mol = Chem.MolFromSmiles(entry["canonical_smiles"])
        assert get_polycyclic_iupac_locants(mol, "fluorene") is None


@pytest.mark.unit
class TestFusedCatalogCoverageGuard:
    """The coverage guard stops a smaller catalog pattern from matching a larger
    fused ring system as a substructure (structure-loss hallucination)."""

    def test_pentacene_subset_match_rejected(self):
        # bare pentacene contains a naphthacene substructure; the raw impl matches
        # it, but the coverage guard rejects (does not cover the 5th ring).
        mol = Chem.MolFromSmiles("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1")
        raw = _match_fused_heterocycle_core_impl(mol)
        assert raw is not None and raw[0] == "naphthacene"
        assert match_fused_heterocycle_core(mol) is None

    def test_picene_subset_match_rejected(self):
        mol = Chem.MolFromSmiles("c1ccc2c(c1)ccc1c2ccc2c3ccccc3ccc21")
        raw = _match_fused_heterocycle_core_impl(mol)
        assert raw is not None and raw[0] == "chrysene"
        assert match_fused_heterocycle_core(mol) is None

    @pytest.mark.parametrize("smi,expected", [
        ("c1ccc2cc3cc4ccccc4cc3cc2c1", "naphthacene"),       # exact -> still matches
        ("c1ccc2c(c1)c1ccccc1c1ccccc21", "triphenylene"),    # branched PAH entry
        ("c1ccc2c(c1)ccc1c3ccccc3ccc21", "chrysene"),        # 18-atom chrysene entry
        ("c1ccc2[nH]c3ccccc3c2c1", "9H-carbazole"),          # heterocycle unaffected
        ("c1ccc2[nH]ccc2c1", "1H-indole"),                   # heterocycle unaffected
    ])
    def test_full_coverage_entries_still_match(self, smi, expected):
        mol = Chem.MolFromSmiles(smi)
        result = match_fused_heterocycle_core(mol)
        assert result is not None and result[0] == expected


@pytest.mark.unit
class TestSubstitutedPahNaming:
    """End-to-end PINs for the S1 wins (SELF-01 gate disabled in the suite)."""

    @pytest.mark.parametrize("smi,expected", [
        # angular / branched cata-fused (numbering engine)
        ("Cc1ccc2ccc3cc4ccccc4cc3c2c1", "2-methylbenz[a]anthracene"),
        ("Cc1ccc2c3ccccc3c3ccccc3c2c1", "2-methyltriphenylene"),
        ("Cc1ccc2ccc3c4ccccc4ccc3c2c1", "3-methylchrysene"),
        # 5-ring (coverage guard + picene data)
        ("Cc1c2cc3ccccc3cc2cc2cc3ccccc3cc12", "6-methylpentacene"),
        ("Cc1cc2c(ccc3c4ccccc4ccc32)c2ccccc12", "5-methylpicene"),
    ])
    def test_substituted_pah_pin(self, smi, expected):
        from orthonym.namer import Orthonym
        name = Orthonym().name(smi)
        assert name == expected, f"got {name!r}"
