"""W2F-P8 Task 6: substituted diarylamine parent choice (P-45.5) + evidence PIN.

The parent (aniline) ring is chosen among the two substituted rings by the
alphanumerical order of the complete candidate names (P-45.5.1, BlueBookV2.md:22236
/ :6273 'bromochloro' < 'dibromo') with the lower-locant criterion (P-45.2.2). The
choice runs through the single preferred-parent authority, so it is DETERMINISTIC
(name-string comparison, never atom-order-dependent).
"""
import orthonym
from rdkit import Chem

EV = "Brc1cc(Cl)ccc1Nc1ccc(Br)cc1Br"
TARGET = "2-bromo-4-chloro-N-(2,4-dibromophenyl)aniline"


def test_p455_evidence_pin():
    # parent = bromo/chloro ring; 'bromo...chloro...' precedes 'dibromo...' at the
    # first point of difference (P-45.5.1)
    assert orthonym.name_compound(EV, style="pin") == TARGET


def test_methyl_positional_choice():
    # 2-methyl (ortho) ring is the parent: ring-substituent locant 2 < 4 (P-45.2.2)
    assert orthonym.name_compound("Cc1ccccc1Nc1ccc(C)cc1", style="pin") == \
        "2-methyl-N-(4-methylphenyl)aniline"


def test_determinism_10_spellings():
    mol = Chem.MolFromSmiles(EV)
    names = {
        orthonym.name_compound(Chem.MolToSmiles(mol, doRandom=True), style="pin")
        for _ in range(10)
    }
    assert names == {TARGET}, names
