"""v30 Slice C slice-2 — amido prefix for a RING-heteroatom acyl on a chain parent.

Measured gap (Fable review + spy, ): the
acylamino path `-NHC(=O)R` where R is a ring bearing a heteroatom (proline's N-ring,
oxolane's O-ring) LINEARIZED the acyl (carbon-only BFS) to 'pentanoic acid' and emitted
the wrong '(pentanoylamino)' — suppressed to 'unknown' by SELF-01. Two root-cause fixes:

  (1) walk the WHOLE acyl fragment (ring heteroatoms included) so it is not linearized;
  (2) when the retained acid name does not convert to an amido ('proline' -> None), retry
      with the SYSTEMATIC acid name ('pyrrolidine-2-carboxylic acid' -> ...carboxamido),
      threaded via name_fragment_recursively(style='systematic'). P-66.1.1.4.3 method (1);
      the peptide 'prolyl' form implies L and breaks RT (BB:54717), so systematic is required.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.fragment_naming import name_fragment_recursively


@pytest.mark.unit
class TestSystematicStyleThreadedThroughFragmentNamer:
    """name_fragment_recursively must honour style='systematic' (was discarded)."""

    def test_default_and_systematic_differ_for_amino_acid(self):
        # The default (pin) path returns the retained name; systematic bypasses
        # it. Use DEFINED-stereo L-proline here: the v33 Phase-1 stereo-honesty
        # fix now makes the default path DECLINE the retained name for a
        # stereo-UNSPECIFIED proline fragment too (bare 'proline' implies L, a
        # different molecule -- BB:54717), so 'OC(=O)C1CCCN1' would return the
        # systematic acid in BOTH styles and the contract would be untestable.
        # With defined stereo the default legitimately returns the retained
        # 'L-proline', which differs from the systematic acid -- the actual
        # style-threading contract this test exists to pin.
        default = name_fragment_recursively("OC(=O)[C@@H]1CCCN1")
        systematic = name_fragment_recursively("OC(=O)[C@@H]1CCCN1", style="systematic")
        assert default != systematic
        assert default == "L-proline"
        assert systematic == "(2S)-pyrrolidine-2-carboxylic acid"

    def test_systematic_style_bypasses_retained(self):
        assert (name_fragment_recursively("OC(=O)C1CCCN1", style="systematic")
                == "pyrrolidine-2-carboxylic acid")

    def test_systematic_preserves_stereo(self):
        assert (name_fragment_recursively("OC(=O)[C@@H]1CCCN1", style="systematic")
                == "(2S)-pyrrolidine-2-carboxylic acid")


@pytest.mark.unit
class TestRingAcylAmidoPIN:
    """Whole-molecule exact PIN for ring-heteroatom acylamino on an acetic-acid parent."""

    @pytest.mark.parametrize("smiles,expected", [
        # oxolane O-ring: tests the ring-walk fix alone (acid already systematic)
        ("OC(=O)CNC(=O)C1CCCO1", "(oxolane-2-carboxamido)acetic acid"),
        # proline N-ring: tests ring-walk + systematic retry
        ("OC(=O)CNC(=O)C1CCCN1", "(pyrrolidine-2-carboxamido)acetic acid"),
        # P-16.5.4 nesting: a compound amido prefix that itself contains marks
        # escalates ()->[]  (Fable review of cbb28539).
        ("OC(=O)CNC(=O)Cc1ccncc1", "[2-(pyridin-4-yl)acetamido]acetic acid"),
        # regression: benzoyl (all-carbon ring) unchanged, bare (no marks)
        ("OC(=O)CNC(=O)c1ccccc1", "benzamidoacetic acid"),
        # regression: slice-1 mark-free prefix stays single parens (depth 0)
        ("OC(=O)CNC(=O)Cc1ccccc1", "(2-phenylacetamido)acetic acid"),
    ])
    def test_exact_pin(self, smiles, expected):
        from orthonym import name_compound
        assert name_compound(smiles) == expected

    def test_nested_stereo_uses_square_brackets_not_parens(self):
        # P-16.5.4: a stereodescriptor-bearing amido prefix nests to []
        from orthonym import name_compound
        name = name_compound("OC(=O)CNC(=O)[C@@H]1CCCN1")
        assert "[(2S)-pyrrolidine-2-carboxamido]" in name, name
        assert "((2S)-pyrrolidine-2-carboxamido)" not in name, name

    @pytest.mark.parametrize("smiles", [
        "OC(=O)CNC(=O)C1CCCN1",
        "OC(=O)CNC(=O)[C@@H]1CCCN1",   # stereo proline: RT (bracket nesting may vary)
        "OC(=O)CNC(=O)C1CCCO1",
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
