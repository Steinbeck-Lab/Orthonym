"""Leads program L7 / 43c: aryl isocyanates and isothiocyanates are named substitutively.

 ISOCYANATES (the Blue Book),:26001: "Preferred IUPAC names are generated
substitutively using the prefix 'isocyanato' attached directly to a parent hydride.
Previously, functional class names were recommended for this class." Examples:26007
'isocyanatocyclohexane (PIN) cyclohexyl isocyanate' and:26009 'isothiocyanatobenzene
(PIN) phenyl isothiocyanate'.

The handlers used to keep the functional-class form for an AROMATIC attachment and
labelled 'phenyl isocyanate' pin_verified (a non-PIN name labelled PIN). Every name
below is checked by an independent, fresh OPSIN 2.9.0 parse (not the engine's own
validity oracle) against the input's full InChIKey.

An acyl isocyanate is a different class, the Blue Book; example
:23138 'CH3-CO-NCO acetyl isocyanate (PIN; '): functional class, unchanged.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from tests.support.rt_assert import _independent_parse

# (smiles, PIN) -- each used to be named 'R isocyanate' / 'R isothiocyanate'.
ARYL = [
    ("O=C=Nc1ccccc1", "isocyanatobenzene"),
    ("O=C=Nc1ccc(Cl)cc1", "1-chloro-4-isocyanatobenzene"),
    ("O=C=Nc1ccc(OC)cc1", "1-isocyanato-4-methoxybenzene"),
    ("O=C=Nc1ccc([N+](=O)[O-])cc1", "1-isocyanato-4-nitrobenzene"),
    ("O=C=Nc1cccc2ccccc12", "1-isocyanatonaphthalene"),
    ("O=C=Nc1ccncc1", "4-isocyanatopyridine"),
    ("O=C=Nc1cccs1", "2-isocyanatothiophene"),
    ("O=C=Nc1ccc(N=C=O)cc1", "1,4-diisocyanatobenzene"),
    ("S=C=Nc1ccccc1", "isothiocyanatobenzene"),
    ("S=C=Nc1ccc(C)cc1", "1-isothiocyanato-4-methylbenzene"),
    ("Cc1ccc(N=C=O)cc1N=C=O", "2,4-diisocyanato-1-methylbenzene"),
]

# (smiles, name) -- already substitutive, or a different class: must not change.
CONTROLS = [
    ("O=C=NC", "isocyanatomethane"),
    ("O=C=NC1CCCCC1", "isocyanatocyclohexane"),          # BB:26007 verbatim
    ("O=C=NCc1ccccc1", "(isocyanatomethyl)benzene"),
    ("S=C=NCc1ccccc1", "(isothiocyanatomethyl)benzene"),
    ("CC(=O)N=C=O", "acetyl isocyanate"),                  # acyl: (BB:23138)
]


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.mark.opsin_gate
class TestAromaticIsocyanatePin:
    @pytest.mark.parametrize("smiles,expected", ARYL)
    def test_aryl_name_is_the_substitutive_pin(self, smiles, expected):
        r = Orthonym(style="pin").name_tiered(smiles)
        assert r["name"] == expected, r
        assert r["tier"] == "pin_verified", r

    @pytest.mark.parametrize("smiles,expected", ARYL)
    def test_aryl_name_reads_back_to_the_input(self, smiles, expected):
        parsed = _independent_parse(expected)
        assert parsed is not None, f"OPSIN rejects {expected!r}"
        assert _key(parsed) == _key(smiles), (expected, parsed)

    @pytest.mark.parametrize("smiles,expected", CONTROLS)
    def test_controls_unchanged(self, smiles, expected):
        assert Orthonym(style="pin").name(smiles) == expected

    @pytest.mark.parametrize("smiles", [s for s, _ in ARYL])
    def test_no_functional_class_form_at_any_tier(self, smiles):
        # the non-PIN functional-class word must not reach any emitting tier
        from orthonym.cli import _emit_tier_flags
        for tier in ("valid", "complete", "best-effort"):
            name = Orthonym(style="pin", **_emit_tier_flags(tier)).name(smiles)
            assert not name.endswith(("isocyanate", "isothiocyanate")), (tier, name)
