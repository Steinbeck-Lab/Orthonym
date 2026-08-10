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
from rdkit import Chem

from orthonym import name_compound
from orthonym.namer import _validity_gate_name_to_smiles


def _canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def _no_parseable_wrong_molecule(smiles: str) -> bool:
    """0-wrong producer-honesty check: the emitted name must NOT parse (via OPSIN)
    to a DIFFERENT molecule. Abstaining is fine, and an UNPARSEABLE name is fine
    too — that is inert garbage the production validity gate suppresses (the
    acceptable pre-state). The honesty violation this guards against is a
    PLAUSIBLE, OPSIN-PARSEABLE, WRONG-constitution name (name_amide's straight-
    chain false friend for a branched/unsaturated acyl). Gate-independent."""
    nm = name_compound(smiles, style="pin")
    if nm is None or nm == "unknown organic compound":
        return True
    back = _validity_gate_name_to_smiles(nm)
    if back is None:
        return True  # unparseable -> caught by the validity gate, not a wrong molecule
    return _canon(back) == _canon(smiles)


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


class TestAcylBranchUnsatHonesty:
    """The amide-delegation must only hand a SIMPLE unbranched/saturated acyl to
    name_amide (whose count-based parent naming drops branches + unsaturation).
    A branched / unsaturated acyl must fall through (correct chain name) or
    abstain — NEVER the straight-chain saturated false friend. Fable review of
    5fbe266c found the broadened delegation both (1) pre-empted a correct
    chain-machinery name with a truthy-wrong name and (2) emitted 9 gate-off
    wrong constitutions; the acyl-simplicity guard fixes both.
    """

    def test_blocker1_unsat_acyl_falls_through_not_preempted(self):
        """C=CC(=O)NCCOC must keep its correct chain name, not be pre-empted into
        name_amide's C=C-dropping 'N-(2-methoxyethyl)propanamide' -> abstain."""
        assert (name_compound("C=CC(=O)NCCOC", style="pin")
                == "N-(2-methoxyethyl)prop-2-enamide")

    def test_branched_acyl_not_false_friend(self):
        """Isobutyramide N-aryl-phenol: must NOT drop the 2-methyl branch."""
        nm = name_compound("CC(C)C(=O)Nc1ccc(O)cc1", style="pin")
        assert nm != "N-(4-hydroxyphenyl)propanamide", (
            f"acyl 2-methyl branch dropped (false friend): {nm!r}")
        assert _no_parseable_wrong_molecule("CC(C)C(=O)Nc1ccc(O)cc1")

    def test_unsat_and_branched_acyls_honest(self):
        """A spread of branched / unsaturated acyls: every one either round-trips
        or abstains (0-wrong), independent of the SELF-01 gate."""
        for smi in (
            "C=CC(=O)Nc1ccc(O)cc1",   # acryloyl
            "CC(C)C(=O)NCCO",         # isobutyryl
            "C#CC(=O)NCCO",           # propioloyl
            "CC(C)(C)C(=O)NCCO",      # pivaloyl
            "CC=CC(=O)NCCO",          # but-2-enoyl
        ):
            assert _no_parseable_wrong_molecule(smi), f"wrong constitution for {smi}"
