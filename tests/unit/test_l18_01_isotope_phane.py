"""lane/l18 item 1: a structure whose PIN is a linear phane name and that carries isotope labels.

The isotope decorator names the isotope-stripped skeleton and writes the descriptors into that name. The
phane producers (``rules.phane_pin``) cite no isotopic modification, so with an OPSIN that reads phane names the
skeleton came back as a phane name that no descriptor can be written into, and the labelled molecule had no name at
any tier (UNNAMEABLE), where OPSIN 2.9.0, which reads no phane name, named it by the substitutive name of the skeleton.

 (2) "linear phanes consist of four or more rings or ring systems, two of which must be terminal, and
together with acyclic atoms or chains must consist of at least seven nodes": the phane name is the PIN of the
unlabelled structure. (isotopically modified compounds, the descriptor cited in front of the name part it
modifies): the labelled structure is named by the descriptor on a name of its skeleton. The phane-form PIN with the
descriptor is not built (left open), so the substitutive name ships below the PIN gives the phane PIN).
"""
import pytest

from orthonym import opsin_capabilities as caps
from orthonym import name_compound

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# 2-(3-aryl...)pyridine with a trideuteriomethyl group and a deuterium: six ring systems on a chain of seven nodes
LABELLED = ("[2H]C1=C(C(=CN=C1C2=CC(=CC=C2)OC3=CC(=CC(=C3C)C4=CC=C(C=C4)C5=CC=CC=C5)C6=NC7=CC=CC=C7N6"
            "C8=C(C=CC=C8C(C)C)C(C)C)C([2H])([2H])[2H])C9=CC=CC=C9")
PLAIN = ("C1=C(C(=CN=C1C2=CC(=CC=C2)OC3=CC(=CC(=C3C)C4=CC=C(C=C4)C5=CC=CC=C5)C6=NC7=CC=CC=C7N6"
         "C8=C(C=CC=C8C(C)C)C(C)C)C)C9=CC=CC=C9")


def test_withheld_scope_switches_phane_off(monkeypatch):
    """The scope of the decorator withholds the phane producers whatever the OPSIN in use reads, and only inside."""
    monkeypatch.setattr(caps, "opsin_capabilities", lambda: {"phane": True, "halogen_uide": False})
    assert caps.phane_enabled()
    with caps.phane_names_withheld():
        assert not caps.phane_enabled()
    assert caps.phane_enabled()


@pytest.mark.skipif(not caps.phane_enabled(), reason="needs an OPSIN that reads phane names (OPSIN main)")
def test_unlabelled_structure_keeps_its_phane_name():
    """ (2) with: the unlabelled structure is still named by its linear phane name."""
    assert "heptaphan" in name_compound(PLAIN, emit_tier="best-effort")


@pytest.mark.skipif(not caps.phane_enabled(), reason="needs an OPSIN that reads phane names (OPSIN main)")
def test_labelled_structure_is_named_at_best_effort():
    """: the labelled structure gets a name with its descriptors (never the empty name)."""
    from orthonym import Orthonym
    r = Orthonym().name_tiered(LABELLED)
    name = name_compound(LABELLED, emit_tier="best-effort")
    assert name and "(2H3)methyl" in name
    assert r.get("tier") != "pin_verified"      # the phane PIN with the descriptor is not built (left open)


# (1): a cyclic phane with its descriptor stays the PIN; the phane skeleton name takes the descriptor,
# so withholding phane names (for a linear phane that takes none) happens only when the first pass finds none.
CYCLOPHANES = [
    ("[2H]C([2H])([2H])c1cc2ccc1CCc1ccc(cc1)CC2", "1^2-(2H3)methyl-1,4(1,4)-dibenzenacyclohexaphane"),
    ("[2H]C([2H])([2H])Oc1cc2ccc1CCc1ccc(cc1)CC2", "1^2-(2H3)methoxy-1,4(1,4)-dibenzenacyclohexaphane"),
    ("OC(=O)c1cc2ccc1CCc1ccc(cc1)C[13CH2]2", "(6-13C)-1,4(1,4)-dibenzenacyclohexaphane-1^2-carboxylic acid"),
]


@pytest.mark.skipif(not caps.phane_enabled(), reason="needs an OPSIN that reads phane names (OPSIN main)")
@pytest.mark.parametrize("smiles,expected", CYCLOPHANES)
def test_labelled_cyclophane_keeps_its_phane_pin(smiles, expected):
    from orthonym import Orthonym
    r = Orthonym().name_tiered(smiles)
    assert r.get("name") == expected
    assert r.get("tier") == "pin_verified"
