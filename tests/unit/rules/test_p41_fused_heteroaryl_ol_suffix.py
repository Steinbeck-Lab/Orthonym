"""A hydroxy group on a fused heteroaromatic parent takes the ``-ol`` SUFFIX.

``rules/fused_rings.py``'s substituent collector routed a bare ``hydroxy`` into the
generic prefix set (``c_substituents``) and ``_build_fused_suffix`` only ever built a
suffix for ``oxo`` (-one) or ``amino`` (-amine). A fused heterocycle whose ONLY
principal characteristic group is a hydroxy therefore shipped the parent hydride plus
a ``hydroxy-`` prefix -- a name with NO suffix for the senior characteristic group::

    Oc1cccc2cccnc12 -> 8-hydroxyquinoline (no suffix; non-PIN)
                      => quinolin-8-ol (PIN, the Blue Book)

RULE
----
 (the Blue Book) ranks hydroxy compounds (class 17) BELOW
ketones (class 16, -one) and above amines (class 19, -amine).
(the Blue Book) selects the parent with the maximum number of the principal
characteristic group expressed as a SUFFIX. So when a hydroxy is the senior group on
the fused parent -- no carbon-acid/aldehyde/amide/nitrile suffix group, no ketone --
it must be the ``-ol`` suffix, not a ``hydroxy-`` prefix.

Direct Blue Book PINs: ``quinolin-8-ol`` (the Blue Book,:3944;:8224 notes
``oxine`` as its trivial name) and ``1H-phenalen-4-ol`` (the Blue Book).

WHY THE PREFIX FORM IS NOT AVAILABLE HERE
-----------------------------------------
``hydroxy`` IS a preferred prefix on a parent whose own characteristic group outranks
the alcohol (a ketone, a carboxylic acid,...). With no senior group present the
alcohol IS the principal one and requires it as the suffix. The ``not
oxo``/``not amino``/``not suffix_groups`` guard is exactly what keeps the promotion
scoped to that case -- the same shape as the sibling amine fix
(``test_p41_fused_n_substituted_amine.py``).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound


class TestFusedHeteroarylHydroxyTakesSuffix:

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # the reported defect -- a verbatim Blue Book PIN (:3942)
        ("Oc1cccc2cccnc12", "quinolin-8-ol"),
        # isoquinoline / indole share the collector; hydroxy is their only PCG
        ("Oc1nccc2ccccc12", "isoquinolin-1-ol"),
        ("Oc1ccc2[nH]ccc2c1", "1H-indol-5-ol"),
    ])
    def test_fused_heteroaryl_hydroxy_is_a_suffix(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        # carbocyclic fused rings already emitted the -ol suffix (a different
        # producer) and must be byte-identical after the fix.
        ("Oc1ccc2ccccc2c1", "naphthalen-2-ol"),
        ("Oc1cccc2ccccc12", "naphthalen-1-ol"),
        # amine on the same fused parent must be unchanged (amino suffix path)
        ("Nc1cccc2cccnc12", "quinolin-8-amine"),
        # a plain ring substituent (methyl) on the fused parent is unchanged
        ("Cc1cccc2cccnc12", "8-methylquinoline"),
    ])
    def test_controls_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.unit
    def test_senior_ketone_keeps_hydroxy_a_prefix(self):
        """When a KETONE (class 16, senior to the alcohol) is on the fused parent,
        the hydroxy must stay a ``hydroxy-`` prefix and the ``-one`` hold the
        suffix. The ``not oxo_substituents`` guard is what enforces it -- the
        control that makes the promotion safe rather than indiscriminate.
        (7-hydroxyquinolin-2(1H)-one)."""
        name = name_compound("Oc1ccc2C=CC(=O)Nc2c1")
        assert name is not None
        assert "hydroxy" in name and name.endswith("one"), name

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", [
        "Oc1cccc2cccnc12",   # quinolin-8-ol
        "Oc1nccc2ccccc12",   # isoquinolin-1-ol
        "Oc1ccc2[nH]ccc2c1",  # 1H-indol-5-ol
    ])
    def test_promoted_names_roundtrip(self, smiles, opsin_to_smiles):
        """The emitted -ol name must parse back (OPSIN) to the input structure --
        a SPELLING fix must not change the molecule."""
        name = name_compound(smiles)
        parsed = opsin_to_smiles(name)
        assert parsed is not None, f"OPSIN could not parse {name!r}"
        assert Chem.CanonSmiles(parsed) == Chem.CanonSmiles(smiles)
