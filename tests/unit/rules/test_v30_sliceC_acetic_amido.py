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
        ("acetic acid", "acetamido"),                 # retained (regression guard)
        # P-66.1.1.4.3 method (1): the prefix is the AMIDE name e->o. Acetamide has
        # two substitutable sites (N, C2), so the C2 locant is REQUIRED
        # ('N-phenylacetamide' = acetanilide, a DIFFERENT molecule) -> '2-...amido'.
        ("phenylacetic acid", "2-phenylacetamido"),   # THE gap (locant required)
        ("2-phenylacetic acid", "2-phenylacetamido"), # explicit locant not doubled
        ("chloroacetic acid", "2-chloroacetamido"),
    ])
    def test_substituted_acetic_to_amido(self, acid, expected):
        assert acid_name_to_amido_prefix(acid) == expected

    @pytest.mark.parametrize("acid", [
        "pentanethioic acid",   # functional replacement -> still fail closed
        "pentanedioic acid",    # poly-acid -> still fail closed
        "peracetic acid",       # peroxy acid -> fail closed (not 'peracetamido')
        "peroxyacetic acid",    # peroxy acid -> fail closed
        "",
        "not an acid",
    ])
    def test_still_fails_closed(self, acid):
        assert acid_name_to_amido_prefix(acid) is None


@pytest.mark.unit
class TestPhenylacetylAcylaminoPIN:
    """Whole-molecule: N-(substituted-acetyl) amino substituents name to the exact PIN.

    Asserts the EXACT string (not just round-trip): the amido prefix carries its
    required C2 locant and is enclosed (P-66.1.1.4.3 method (1) + P-16.5.1.2), the
    spelling-layer contract the round-trip check alone cannot see.
    """

    @pytest.mark.parametrize("smiles,expected", [
        # acetic-acid parent: parent locant omitted (P-14.3.4.6, one substitutable C),
        # substituted amido enclosed with its own C2 locant.
        ("OC(=O)CNC(=O)Cc1ccccc1", "(2-phenylacetamido)acetic acid"),
        # propanoic/pentanoic parents: parent locant cited (P-16.5.1.2).
        ("CC(NC(=O)Cc1ccccc1)C(=O)O", "2-(2-phenylacetamido)propanoic acid"),
        ("CC(C)C[C@H](NC(=O)Cc1ccccc1)C(=O)O",
         "(2S)-4-methyl-2-(2-phenylacetamido)pentanoic acid"),
        # regression guard: the SIMPLE benzamido sibling is unchanged (no locant/marks).
        ("OC(=O)CNC(=O)c1ccccc1", "benzamidoacetic acid"),
    ])
    def test_exact_pin(self, smiles, expected):
        from orthonym import name_compound
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles", [
        "OC(=O)CNC(=O)Cc1ccccc1",
        "CC(C)C[C@H](NC(=O)Cc1ccccc1)C(=O)O",
    ])
    def test_round_trips(self, smiles):
        from orthonym import name_compound
        from orthonym.validation.opsin_roundtrip import opsin_parse
        name = name_compound(smiles)
        assert name and "unknown" not in name.lower(), f"abstained/sentinel: {name!r}"
        got = opsin_parse(name)
        assert got, f"OPSIN could not parse {name!r}"
        assert (inchi.MolToInchiKey(Chem.MolFromSmiles(got))
                == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))), f"{name!r} -> {got}"
