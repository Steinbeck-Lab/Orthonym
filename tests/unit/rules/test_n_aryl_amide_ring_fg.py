"""N-aryl amide with a JUNIOR characteristic group on the N-aryl ring.

When an amide is the principal characteristic group and the ONLY other
characteristic group (a phenol / hydroxy, an amine, ...) lives entirely on an
N-substituent, the amide stays PCG (P-41) and the junior group is a prefix on
that N-substituent: paracetamol CC(=O)Nc1ccc(O)cc1 -> N-(4-hydroxyphenyl)acetamide.

Regression: the polyfunctional handler used to delegate to rules.amides.name_amide
ONLY when the acyl carbon was OFF the principal chain; the acyl-ON-chain case
(the ordinary N-aryl acetamide) fell through to a chain-suffix path that
mis-rooted the N-aryl as a C1 substituent -> the malformed, OPSIN-unparseable
'1-(4-hydroxyanilino)ethanamide' -> abstain. name_amide names it correctly.
"""
from orthonym import name_compound


class TestNArylAmideJuniorRingFG:
    def test_paracetamol(self):
        assert (name_compound("CC(=O)Nc1ccc(O)cc1", style="pin")
                == "N-(4-hydroxyphenyl)acetamide")

    def test_ortho_hydroxy(self):
        assert (name_compound("CC(=O)Nc1ccccc1O", style="pin")
                == "N-(2-hydroxyphenyl)acetamide")

    def test_propanamide_analog(self):
        assert (name_compound("CCC(=O)Nc1ccc(O)cc1", style="pin")
                == "N-(4-hydroxyphenyl)propanamide")

    def test_plain_acetanilide_unchanged(self):
        """No junior FG: must stay correct (control)."""
        assert (name_compound("CC(=O)Nc1ccccc1", style="pin")
                == "N-phenylacetamide")

    def test_chloro_unchanged(self):
        """Halogen (always a prefix, not a competing PCG): control."""
        assert (name_compound("CC(=O)Nc1ccc(Cl)cc1", style="pin")
                == "N-(4-chlorophenyl)acetamide")
