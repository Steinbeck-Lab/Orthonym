"""Leads program L7 / 40b: the 'L' omission of is a licence of the PEPTIDE.

 'Indication of configuration in peptides' (the Blue Book),:54717:
"The stereodescriptor 'L' is not indicated in the names nor in the symbolic
representation of peptides composed of amino acids listed in Table 10.4. In contrast,
the stereodescriptor 'D' is indicated at the front of the acyl group or name of each
component having that configuration." Example:54721: 'Leu-D-Glu-L-aThr-D-Val-Leu (the
symbol aThr is for allothreonine); L-leucyl-D-glutamyl-L-allothreonyl-D-valyl-L-leucine'.
Allothreonine is in Table 10.5 (:54220), not Table 10.4 (:54186), and the example cites
every 'L', the leucines included.

The engine dropped 'L-' on every residue, including residues outside Table 10.4
(ornithine, sarcosine, dopa,...), e.g. 'glutamylornithine' for L-Glu-L-Orn.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules.peptides import _assemble_peptide_name, name_peptide
from tests.support.rt_assert import _independent_parse


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


# L-Glu-L-Orn: independent of the code under test, the expected name must read back to
# the input's full InChIKey through a fresh OPSIN parse.
GLU_ORN = "NCCC[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)O"

# (smiles, name): a residue outside Table 10.4 (ornithine) voids the 'L' omission.
NON_TABLE = [
    (GLU_ORN, "L-glutamyl-L-ornithine"),
    ("N[C@@H](C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)O", "L-alanyl-L-ornithyl-L-alanine"),
    ("N[C@@H](C)C(=O)N[C@H](CCCN)C(=O)N[C@@H](C)C(=O)O", "L-alanyl-D-ornithyl-L-alanine"),
    ("NCC(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)O", "glycyl-L-ornithyl-L-alanine"),
]

# Capped termini (Lever C: an N-methyl N-terminus, a C-terminal carboxamide): the licence is
# about the AMINO ACIDS, so a 'valinamide' / 'N-methylisoleucyl' of Table 10.4 amino acids
# keeps the omission, and with a residue outside Table 10.4 the descriptors are cited and the
# 'N-methyl' prefix takes a hyphen before them.
CAPPED_TABLE_10_4 = (
    "CC[C@H](C)[C@H](NC)C(=O)N[C@@H](CO)C(=O)N1CCC[C@H]1C(=O)N[C@@H](C)C(=O)"
    "N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@@H](CO)C(=O)"
    "N[C@@H](CC(C)C)C(=O)N[C@H](C(N)=O)C(C)C",
    "N-methylisoleucylserylprolylalanylleucylleucylalanylserylleucylvalinamide",
)
CAPPED_NON_TABLE = [
    ("CN[C@@H](C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)O",
     "N-methyl-L-alanyl-L-ornithyl-L-alanine"),
    ("N[C@@H](C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(N)=O",
     "L-alanyl-L-ornithyl-L-alaninamide"),
]

# Table 10.4 residues only: the licence applies, 'L' stays omitted, 'D' is cited.
TABLE_10_4 = [
    ("N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O", "alanylalanylalanine"),
    ("N[C@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O", "D-alanylalanylalanine"),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "aspartylvalylphenylalanine",
    ),
]


@pytest.mark.unit
class TestLOmissionLicenceIsPerPeptide:
    @pytest.mark.parametrize("smiles,expected", NON_TABLE)
    def test_non_table_10_4_residue_cites_every_L(self, smiles, expected):
        assert name_peptide(Chem.MolFromSmiles(smiles)) == expected

    @pytest.mark.parametrize("smiles,expected", TABLE_10_4)
    def test_table_10_4_peptide_keeps_the_omission(self, smiles, expected):
        assert name_peptide(Chem.MolFromSmiles(smiles)) == expected

    @pytest.mark.parametrize("smiles,expected", NON_TABLE + TABLE_10_4)
    def test_name_reads_back_to_the_input(self, smiles, expected):
        # a fresh OPSIN 2.9.0 parse, not the engine's validity oracle
        parsed = _independent_parse(expected)
        assert parsed is not None, f"OPSIN rejects {expected!r}"
        assert _key(parsed) == _key(smiles), (expected, parsed)

    def test_capped_table_10_4_peptide_keeps_the_omission(self):
        smiles, expected = CAPPED_TABLE_10_4
        assert name_peptide(Chem.MolFromSmiles(smiles)) == expected
        parsed = _independent_parse(expected)
        assert parsed is not None and _key(parsed) == _key(smiles)

    @pytest.mark.parametrize("smiles,expected", CAPPED_NON_TABLE)
    def test_capped_peptide_with_a_non_table_residue_cites_the_L(self, smiles, expected):
        assert name_peptide(Chem.MolFromSmiles(smiles)) == expected
        parsed = _independent_parse(expected)
        assert parsed is not None and _key(parsed) == _key(smiles), (expected, parsed)

    def test_bb_example_spelling(self):
        # the Blue Book, assembled from its residues. allothreonine is a
        # Table 10.5 name, so the 'L' of every residue is cited.
        residues = [
            {"name": "leucine", "acyl": "leucyl", "stereo": "L-"},
            {"name": "glutamic acid", "acyl": "glutamyl", "stereo": "D-"},
            {"name": "allothreonine", "acyl": "allothreonyl", "stereo": "L-"},
            {"name": "valine", "acyl": "valyl", "stereo": "D-"},
            {"name": "leucine", "acyl": "leucyl", "stereo": "L-"},
        ]
        assert (
            _assemble_peptide_name(residues)
            == "L-leucyl-D-glutamyl-L-allothreonyl-D-valyl-L-leucine"
        )

    def test_bb_example_residues_without_the_table_10_5_residue(self):
        # the same chain with threonine (Table 10.4): only 'D' is cited
        residues = [
            {"name": "leucine", "acyl": "leucyl", "stereo": "L-"},
            {"name": "glutamic acid", "acyl": "glutamyl", "stereo": "D-"},
            {"name": "threonine", "acyl": "threonyl", "stereo": "L-"},
            {"name": "valine", "acyl": "valyl", "stereo": "D-"},
            {"name": "leucine", "acyl": "leucyl", "stereo": "L-"},
        ]
        assert _assemble_peptide_name(residues) == "leucyl-D-glutamylthreonyl-D-valylleucine"

    @pytest.mark.opsin_gate
    def test_pipeline_emits_the_corrected_spelling(self):
        r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(GLU_ORN)
        assert r["name"] == "L-glutamyl-L-ornithine", r
        assert r["tier"] == "systematic_verified", r  # no PIN claim: gives no PINs here
