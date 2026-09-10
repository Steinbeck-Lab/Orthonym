""" a phase (P-68.3.2.3.2.2 / P-67.1.4.1.1): a mononuclear P/As/Sb ACYL group
cited as a SUBSTITUENT prefix on a senior parent.

The senior characteristic group (a carboxylic acid, here) owns the suffix; the
=E'-bearing P/As/Sb group is demoted to the P-67.1.4.1.1 acyl PREFIX, consumed
from the shared ``functional_replacement.acyl_prefix_for`` table (never
re-spelled). BB L39255 verbatim: ``4-(dimethylphosphinothioyl)benzoic acid``
(PIN). The recognizer must NOT poach the trivalent ``phosphanyl``/``arsanyl``
substituents (no =E') nor the P-67.1.4.1.1.1 retained ``phosphono``/``arsono``
prefixes (skeletal-0, all -OH).
"""

from rdkit import Chem

import orthonym
from orthonym.rules.benzene import _bfs_substituent_atoms
from orthonym.rules.phosphorus import name_acyl_prefix_substituent


class TestAcylPrefixSubstituentIntegration:
    def test_dimethylphosphinothioyl_benzoic_acid(self):
        # P-68.3.2.3.2.2, BB L39255: (CH3)2P(=S)- is 'dimethylphosphinothioyl'
        # (phosphinothioic acid = skeletal-2 =S acid), NOT 'dimethylphosphanyl'
        # (which silently drops the =S).
        assert orthonym.name_compound(
            "CP(=S)(C1=CC=C(C(=O)O)C=C1)C", style="pin") == \
            "4-(dimethylphosphinothioyl)benzoic acid"

    def test_dimethylphosphinoyl_benzoic_acid(self):
        # The =O sibling: (CH3)2P(=O)- is 'dimethylphosphinoyl' (phosphinic acid =
        # skeletal-2 =O acid). Generalises from the shared table, same class.
        assert orthonym.name_compound(
            "OC(=O)c1ccc(P(=O)(C)C)cc1", style="pin") == \
            "4-(dimethylphosphinoyl)benzoic acid"

    def test_methoxyphosphoryl_dibenzoic_acid(self):
        # P-68.3.2.3.2.2, BB L39246: the multiplicative >P(=O)(OCH3)< bridge with
        # an -O-methyl (ester) residual -> 'methoxyphosphoryl' (skeletal-0
        # phosphoryl + methoxy). Extends the bridge residual beyond terminal -OH.
        assert orthonym.name_compound(
            "COP(=O)(C1=CC=C(C(=O)O)C=C1)C1=CC=C(C(=O)O)C=C1", style="pin") == \
            "4,4'-(methoxyphosphoryl)dibenzoic acid"


class TestAcylPrefixSubstituentUnit:
    def _frag(self, smiles, central_symbol):
        mol = Chem.MolFromSmiles(smiles)
        # the ring is the parent; find the P/As central and its substituent atoms
        ring_atoms = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
        central = next(a.GetIdx() for a in mol.GetAtoms()
                       if a.GetSymbol() == central_symbol
                       and a.GetIdx() not in ring_atoms)
        frag = _bfs_substituent_atoms(mol, central, ring_atoms)
        return mol, frag, central

    def test_recognizer_phosphinothioyl(self):
        mol, frag, c = self._frag("CP(=S)(c1ccc(C(=O)O)cc1)C", "P")
        assert name_acyl_prefix_substituent(mol, frag, c) == \
            "dimethylphosphinothioyl"

    def test_recognizer_declines_trivalent_phosphanyl(self):
        # -PPh2 has no =E' -> not an acyl group; recognizer declines (None) so the
        # trivalent 'phosphanyl' path keeps ownership.
        mol, frag, c = self._frag("c1ccc(P(c2ccccc2)c2ccccc2)cc1", "P")
        assert name_acyl_prefix_substituent(mol, frag, c) is None

    def test_recognizer_declines_retained_phosphono(self):
        # -P(=O)(OH)2 is the P-67.1.4.1.1.1 retained 'phosphono' prefix (skeletal-0,
        # all -OH); the recognizer must decline so the retained path owns it.
        mol, frag, c = self._frag("OC(=O)c1ccc(P(=O)(O)O)cc1", "P")
        assert name_acyl_prefix_substituent(mol, frag, c) is None


class TestAcylPrefixSubstituentRegression:
    def test_phosphono_retained_unchanged(self):
        # regression: the retained 'phosphono' PIN must survive (upstream FG path)
        assert orthonym.name_compound(
            "OC(=O)c1ccc(P(=O)(O)O)cc1", style="pin") == "4-phosphonobenzoic acid"

    def test_diphenylphosphanyl_unchanged(self):
        # regression: trivalent -PPh2 stays 'diphenylphosphanyl'
        assert orthonym.name_compound(
            "O=C(O)c1ccc(P(c2ccccc2)c2ccccc2)cc1", style="pin") == \
            "4-(diphenylphosphanyl)benzoic acid"

    def test_hydroxyarsoryl_multiplicative_unchanged(self):
        # regression: the multiplicative >As(=O)(OH)< bridge (named by
        # rules/multiplicative.py, 2 attachments) is NOT poached by this recognizer.
        assert orthonym.name_compound(
            "O[As](=O)(C1=CC=C(C(=O)O)C=C1)C1=CC=C(C(=O)O)C=C1", style="pin") == \
            "4,4'-(hydroxyarsoryl)dibenzoic acid"
