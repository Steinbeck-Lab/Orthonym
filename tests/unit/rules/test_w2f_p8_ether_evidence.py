"""W2F-P8 Task 4: the P-45.6 stereo-differing diaryl-ether evidence PIN.

Two constitutionally identical para-(1-chloroethyl)phenyl rings joined by O, one
arm R and one S. Multiplicative is disallowed (P-45.6.2); the substitutive PIN is
chosen with the R arm first-cited (P-45.6.3, 'R' precedes 'S'). The parent ring is
the one bearing R, so its own 1-chloroethyl (cited before the ...phenoxy complex
prefix) carries (1R). BlueBookV2.md:22585 (Example 2), :22603 (P-45.6.3).

Determinism: the parent choice is a name-string comparison, so 10 random SMILES
respellings must all yield one identical name.
"""
import orthonym
from rdkit import Chem

EV = "C[C@H](Cl)c1ccc(Oc2ccc(cc2)[C@@H](C)Cl)cc1"
TARGET = "1-[(1R)-1-chloroethyl]-4-{4-[(1S)-1-chloroethyl]phenoxy}benzene"


def test_p456_evidence_pin():
    assert orthonym.name_compound(EV, style="pin") == TARGET


def test_stereo_lossy_refused():
    # the non-stereo string describes a different (achiral) molecule -> the
    # emitted name must carry BOTH descriptors (never a stereo-lossy string).
    name = orthonym.name_compound(EV, style="pin")
    assert "(1R)" in name and "(1S)" in name


def test_determinism_10_spellings():
    mol = Chem.MolFromSmiles(EV)
    names = {
        orthonym.name_compound(
            Chem.MolToSmiles(mol, doRandom=True), style="pin")
        for _ in range(10)
    }
    assert names == {TARGET}, names
