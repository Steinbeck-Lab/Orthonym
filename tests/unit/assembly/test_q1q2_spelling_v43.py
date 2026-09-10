"""Two spelling-layer PIN fixes handed over from the 2026-09-03 engine-hygiene
pass (internal notes).

Q1: a substituted alkyl whose substituent is itself a '-yl' group
   ('cyclohexylmethyl' = cyclohexyl + methyl) is a compound substituent and takes
   enclosing marks. `needs_brackets` used to miss it (no digit/hyphen), so the
   fallback path shipped it unenclosed. NOTE: the main chain path builds
   '<organyl>amino' as one atomic string via a separate producer that is NOT yet
   fixed -- this test pins the PREDICATE + the fallback consumer only.

Q2: the plain-ring branch of the N-attached fallback dropped the
   heterocycle attach locant ('oxanylamino' for oxan-4-yl); oxan-2-yl != oxan-4-yl,
   so the locant is essential. Now passes the attachment point -> 'oxan-4-yl...'.
"""
import pytest

pytestmark = pytest.mark.unit


class TestQ1NeedsBrackets:
    @pytest.mark.parametrize("name", [
        "cyclohexylmethyl", "phenylmethyl", "cyclopropylmethyl",
    ])
    def test_cyclic_on_alkyl_needs_brackets(self, name):
        from orthonym.assembly.naming_utils import needs_brackets
        assert needs_brackets(name) is True

    @pytest.mark.parametrize("name", [
        "methyl", "ethyl", "cyclohexyl", "phenyl", "tert-butyl", "methoxy",
    ])
    def test_simple_substituents_stay_bare(self, name):
        from orthonym.assembly.naming_utils import needs_brackets
        assert needs_brackets(name) is False

    def test_enclose_flows_through(self):
        from orthonym.assembly.naming_utils import enclose_if_compound
        assert enclose_if_compound("cyclohexylmethyl") == "(cyclohexylmethyl)"


class TestQ2HeterocycleAttachLocant:
    @pytest.mark.opsin_gate
    def test_oxan_4_yl_amino_carries_locant_and_round_trips(self):
        from orthonym import Orthonym
        from orthonym.validation.opsin_roundtrip import opsin_parse
        from rdkit import Chem
        smi = "N#CCNC1CCOCC1"  # 2-((oxan-4-yl)amino)ethanenitrile; abstained before
        namer = Orthonym(general_fallback=True,
                          general_fallback_unverified=True,
                          allow_aromatic_general=True)
        name = namer.name_tiered(smi).get("name")
        assert name is not None
        assert "oxan-4-yl" in name  # the essential attach locant is present
        assert "oxanyl" not in name.replace("oxan-4-yl", "")
        osmi = opsin_parse(name)
        assert osmi is not None
        assert (Chem.MolToInchiKey(Chem.MolFromSmiles(osmi))
                == Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
