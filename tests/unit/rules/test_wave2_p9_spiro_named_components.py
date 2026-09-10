import pytest

from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.rules import spiro


# BB target (the Blue Book, verbatim PIN). Single hexavalent
# (lambda6) S shared by three distinct named ring components in three rings.
TARGET_SMILES = "C1=CC=CC=2S34(C5=C(C21)C=CC=C5)(OC5=C(S3)C=CC=C5)OC5=C(O4)C=CC=C5"
TARGET_PIN = (
    "2λ6-spiro[[1,3,2]benzodioxathiole-2,2'-([1,2,3]benzoxadithiole)-"
    "2,5''-dibenzo[b,d]thiophene]"
)

# 3 identical benzodioxathiole spiroter) — the new branch must decline.
SPIROTER_SMILES = "O1S23(OC4=C1C=CC=C4)(OC4=C(O2)C=CC=C4)OC4=C(O3)C=CC=C4"
SPIROTER_PIN = "2λ6,2',2''-spiroter[[1,3,2]benzodioxathiole]"

# 2 identical benzodioxathiole + 1 benzoxadithiole 'bis' form).
# OPSIN 2.9 cannot parse the 'bis' name -> permanently fail-closed.
BIS_SMILES = "O1S23(OC4=C1C=CC=C4)(OC4=C(O2)C=CC=C4)OC4=C(S3)C=CC=C4"


@pytest.mark.unit
class TestP24842SpiroNamedComponents:
    def test_target_pin(self):
        assert name_compound(TARGET_SMILES) == TARGET_PIN

    def test_target_deterministic_across_respellings(self):
        mol = Chem.MolFromSmiles(TARGET_SMILES)
        names = set()
        for _ in range(15):
            names.add(name_compound(Chem.MolToSmiles(mol, doRandom=True)))
        assert names == {TARGET_PIN}

    def test_three_component_predicate(self):
        assert spiro.is_spiro_named_components(Chem.MolFromSmiles(TARGET_SMILES))

    def test_spiroter_declines_and_stays_spiroter(self):
        mol = Chem.MolFromSmiles(SPIROTER_SMILES)
        # New branch declines (all-identical is not...
        assert spiro.name_spiro_named_components(mol) is None
        #...and the spiroter name is unchanged.
        assert name_compound(SPIROTER_SMILES) == SPIROTER_PIN

    def test_bis_fail_closed(self):
        # 2-identical 'bis': the three-component branch requires all
        # three DISTINCT, so it declines. No RT-able name exists in OPSIN for the
        # 'bis' form, so this is permanently fail-closed; the production
        # OPSIN validity gate suppresses the (pre-existing) heterocycle
        # fall-through to 'unknown' (verified with OPSIN via the W2F-P9-P04
        # gold). The invariant this unit owns: the new branch never claims it,
        # and never emits a (wrong) spiro name.
        m = Chem.MolFromSmiles(BIS_SMILES)
        assert spiro.name_spiro_named_components(m) is None
        assert spiro.is_spiro_named_components(m) is False
        assert "spiro" not in name_compound(BIS_SMILES).lower()

    def test_component_templates_registered(self):
        names = {t[0] for t in spiro._FUSED_HET_SPIRO_TEMPLATES}
        assert "[1,2,3]benzoxadithiole" in names
        assert "dibenzo[b,d]thiophene" in names
