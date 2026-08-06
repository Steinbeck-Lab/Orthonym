"""v30 Slice C (recursive-complete substituent naming) — first atomic slice.

Gap (measured, ): the acylamino
substituent ``-NHC(=O)R`` declines when R is a substituted acetyl (phenylacetyl,
chloroacetyl, ...). Root cause: ``acid_name_to_amido_prefix`` covers ``-oic acid``,
``carboxylic acid`` and a two-entry retained table, but NOT the retained
``acetic acid`` FAMILY, so ``phenylacetic acid`` -> None. The acid names +
round-trips standalone (``2-phenylacetamidoacetic acid`` RT-verified), so the
substituent is reachable once the amido converter covers this family.

P-66.1.1.4.3 method (1): amido prefix = amide name with final 'e' -> 'o'.
``acetamide`` is a retained amide PIN, so a substituted acetamide's amido prefix
is ``<subst>acetamido`` (``phenylacetamido``, ``chloroacetamido``).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.substituent_naming import acid_name_to_amido_prefix


@pytest.mark.unit
class TestAceticAcidFamilyAmido:
    """acid_name_to_amido_prefix must cover the retained acetic-acid family."""

    @pytest.mark.parametrize("acid,expected", [
        ("acetic acid", "acetamido"),               # retained (regression guard)
        ("phenylacetic acid", "phenylacetamido"),   # THE gap
        ("2-phenylacetic acid", "2-phenylacetamido"),
        ("chloroacetic acid", "chloroacetamido"),
    ])
    def test_substituted_acetic_to_amido(self, acid, expected):
        assert acid_name_to_amido_prefix(acid) == expected

    @pytest.mark.parametrize("acid", [
        "pentanethioic acid",   # functional replacement -> still fail closed
        "pentanedioic acid",    # poly-acid -> still fail closed
        "",
        "not an acid",
    ])
    def test_still_fails_closed(self, acid):
        assert acid_name_to_amido_prefix(acid) is None


@pytest.mark.unit
class TestPhenylacetylAcylaminoRoundTrips:
    """Whole-molecule: N-(substituted-acetyl) amino substituents now name + RT.

    Asserts round-trip (name -> OPSIN -> InChIKey == input) rather than an exact
    string, because enclosing-mark placement is the assembly's call; the breadth
    contract is that the emitted name denotes the RIGHT molecule.
    """

    @pytest.mark.parametrize("smiles", [
        "OC(=O)CNC(=O)Cc1ccccc1",       # N-(phenylacetyl)glycine
        "CC(NC(=O)Cc1ccccc1)C(=O)O",    # N-(phenylacetyl)alanine
    ])
    def test_names_and_round_trips(self, smiles):
        from orthonym import name_compound
        from orthonym.validation.opsin_roundtrip import opsin_parse
        name = name_compound(smiles)
        assert name and "unknown" not in name.lower(), f"abstained/sentinel: {name!r}"
        got = opsin_parse(name)
        assert got, f"OPSIN could not parse {name!r}"
        want_ik = inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
        got_ik = inchi.MolToInchiKey(Chem.MolFromSmiles(got))
        assert got_ik == want_ik, f"{name!r} -> {got} (IK {got_ik} != {want_ik})"
