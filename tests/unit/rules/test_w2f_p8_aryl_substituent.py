"""W2F-P8 Task 1: substituted-aryl complex-substituent core namer.

`decorated_ring_substituent_name` must produce the shared `...phenyl` core for a
substituted aryl ring — including a BRANCHED alkyl bearing a CIP stereodescriptor
(`4-[(1S)-1-chloroethyl]phenyl`), which the v1 simple-substituent table refused.
This core feeds the O-linked (phenoxy, Task 3/4) and N-linked (N-aryl, Task 5/6)
paths. BB (the Blue Book),.

Fail-closed: an un-nameable ring substituent (nested ring) must return None so no
partially-described / structure-dropping name can leak.
"""
from rdkit import Chem

from orthonym.rules.ring_substituents import decorated_ring_substituent_name


def _prep(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    Chem.rdCIPLabeler.AssignCIPLabels(mol)
    return mol


class TestArylSubstituentCore:
    # Evidence ether: ring2 (phenoxy side) atoms (8,9,10,11,12,13), attach C8
    # (bonded to the ether O), chloroethyl arm at C11 is CIP R.
    ETHER = "C[C@H](Cl)c1ccc(Oc2ccc(cc2)[C@@H](C)Cl)cc1"

    def test_para_chloroethyl_stereo_R(self):
        mol = _prep(self.ETHER)
        core = decorated_ring_substituent_name(
            mol, (8, 9, 10, 11, 12, 13), 8,
            expected_atoms={8, 9, 10, 11, 12, 13, 14, 15, 16},
        )
        assert core == "4-[(1R)-1-chloroethyl]phenyl"

    def test_para_chloroethyl_stereo_S(self):
        # ring1 (parent side) atoms (3,4,5,6,17,18), attach C6 (bonded to O),
        # chloroethyl arm at C3 is CIP S.
        mol = _prep(self.ETHER)
        core = decorated_ring_substituent_name(
            mol, (3, 4, 5, 6, 17, 18), 6,
            expected_atoms={0, 1, 2, 3, 4, 5, 6, 17, 18},
        )
        assert core == "4-[(1S)-1-chloroethyl]phenyl"

    def test_dibromo_simple(self):
        # regression: simple-halogen ring unchanged (byte-identical legacy path)
        mol = _prep("Brc1cc(Cl)ccc1Nc1ccc(Br)cc1Br")
        assert decorated_ring_substituent_name(
            mol, (9, 10, 11, 12, 14, 15), 9,
            expected_atoms={9, 10, 11, 12, 13, 14, 15, 16},
        ) == "2,4-dibromophenyl"

    def test_bromochloro_simple(self):
        mol = _prep("Brc1cc(Cl)ccc1Nc1ccc(Br)cc1Br")
        assert decorated_ring_substituent_name(
            mol, (1, 2, 3, 5, 6, 7), 7,
            expected_atoms={0, 1, 2, 3, 4, 5, 6, 7},
        ) == "2-bromo-4-chlorophenyl"

    def test_nested_ring_substituent_fails_closed(self):
        # a ring substituent whose decoration is itself a ring (biphenyl arm) is
        # out of the acyclic-branch scope -> None (never a partial name).
        mol = _prep("Ic1ccc(-c2ccccc2)cc1")
        # attach = the ring C bonded to I (atom 1); the para carbon bears a phenyl
        assert decorated_ring_substituent_name(mol, (1, 2, 3, 4, 11, 12), 1) is None
