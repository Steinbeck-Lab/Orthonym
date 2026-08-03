"""Tests for branched acid ester acyloxy prefix naming (IUPAC P-65.6.3.2.2).

Ensures that esters with branched acid fragments use the principal chain
length (not total carbon count) for the acid stem, and include branch
substituent prefixes in the acyloxy name.
"""
import pytest
from rdkit import Chem

from orthonym.rules.esters import name_ester_as_prefix
from orthonym.perception.functional_groups import detect_functional_groups


def _get_acyloxy_prefix(smiles: str) -> str:
    """Helper: get acyloxy prefix from first ester in molecule."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)
    esters = fgs.get("ester", [])
    assert esters, f"No ester found in {smiles}"
    return name_ester_as_prefix(mol, esters[0])


class TestBranchedAcidAcyloxy:
    """IUPAC P-65.6.3.2.2: branched acid -> branched acyloxy prefix."""

    def test_isobutyrate_uses_principal_chain(self):
        """Isobutyric acid: 3C principal chain + 1 methyl branch.

        CC(C)C(=O)O... has 4 total carbons but principal chain = 3C,
        so the acid stem should be "propanoic" (not "butanoic").
        """
        # Phenyl isobutyrate: CC(C)C(=O)Oc1ccccc1
        result = _get_acyloxy_prefix("CC(C)C(=O)Oc1ccccc1")
        assert result is not None
        # Should contain "propanoyl" (3C chain), not "butanoyl" (4C total)
        assert "propanoyl" in result, f"Expected propanoyl in {result}"
        assert "methyl" in result, f"Expected methyl branch in {result}"

    def test_simple_acetate_unchanged(self):
        """Acetic acid: 2C, no branch -> 'acetyloxy' (trivial)."""
        result = _get_acyloxy_prefix("CC(=O)Oc1ccccc1")
        assert result == "acetyloxy"

    def test_simple_propanoate_unchanged(self):
        """Propanoic acid: 3C, no branch -> 'propanoyloxy'."""
        result = _get_acyloxy_prefix("CCC(=O)Oc1ccccc1")
        assert result == "propanoyloxy"

    def test_linear_butanoate_unchanged(self):
        """Butanoic acid: 4C linear, no branch -> 'butanoyloxy'."""
        result = _get_acyloxy_prefix("CCCC(=O)Oc1ccccc1")
        assert result == "butanoyloxy"

    def test_straight_chain_fatty_acid_stem_preserved(self):
        """C16:0 linear ester -> 'hexadecanoyloxy'.

        Purpose of this test (unchanged): the branched naming path must NOT
        override the straight-chain stem -- a branch-decomposition bug would
        show up as a methyl/shorter-chain stem here.

        Asserted word changed from 'palmitoyloxy' to the PIN in Task J3.
        P-65.6.3.2.3 "Esters cited as prefixes" (BlueBookV2.md:31696) prints the
        trivial-derived acyloxy prefix as the NON-preferred alternative when the
        acid is retained for general nomenclature only -- ':31723
        3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)   3-(nicotinoyloxy)-
        propanoic acid'.  Palmitic acid is in that same general-only list
        (P-65.1.1.2.2 heading :29745; row :29787 'palmitic acid  hexadecanoic
        acid (PIN)').  Appendix 2, whose legend at :55416 reads "The symbol *
        designates the preferred prefix", prints 'hexadecanoyl* = palmitoyl'
        (:56482, :56511).  'palmitoyloxy' occurs 0 times in the Blue Book;
        'hexadecanoyloxy' occurs at :31846, :55170, :55199.
        """
        result = _get_acyloxy_prefix("CCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "hexadecanoyloxy" in result, f"Expected hexadecanoyloxy in {result}"
        assert "palmitoyloxy" not in result, (
            f"'palmitoyloxy' is non-PIN (Appendix 2 :56482 'hexadecanoyl* = "
            f"palmitoyl'); got {result}"
        )

    def test_isovalerate_branched(self):
        """Isovaleric acid: 3-methylbutanoic acid (4C chain + 1 branch).

        CC(C)CC(=O)O... has 5 total C but principal chain = 4C.
        """
        result = _get_acyloxy_prefix("CC(C)CC(=O)OC")
        assert result is not None
        assert "butanoyl" in result, f"Expected butanoyl (4C chain) in {result}"
