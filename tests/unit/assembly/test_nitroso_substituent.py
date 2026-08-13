"""v30 Phase-1 tail: the -N=O nitroso substituent morpheme (P-66.5).

Before this, name_substituent spelled -N=O via skeletal replacement
('2-oxa-1-azaeth-1-en-1-yl') -- RT-valid but non-PIN and ugly. The morpheme
emits the clean 'nitroso'. Guarded so a nitrite (-O-N=O) never matches.
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.namer import Orthonym

BE = dict(general_fallback=True, general_fallback_unverified=True,
          allow_aromatic_general=True)


def _sub(smi, attach_sym):
    """name_substituent on the whole non-attachment fragment, attach = the
    first atom of `attach_sym` bonded to the rest."""
    mol = Chem.MolFromSmiles(smi)
    return mol


class TestNitrosoMorpheme:
    def test_nitroso_fragment_token(self):
        """A bare -N=O fragment (attach = N) -> 'nitroso'."""
        mol = Chem.MolFromSmiles("O=NN1CCCC1")  # 1-nitrosopyrrolidine
        n_attach = None
        for a in mol.GetAtoms():
            if a.GetSymbol() == "N" and any(nb.GetSymbol() == "O" for nb in a.GetNeighbors()):
                n_attach = a.GetIdx()
                o = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetSymbol() == "O"][0]
        assert name_substituent(mol, frozenset([n_attach, o]), n_attach) == "nitroso"

    def test_end_to_end_1_nitrosopyrrolidine(self):
        """Ring + N-nitroso assembles to the clean PIN, not the oxa-aza chain."""
        r = Orthonym(**BE).name_tiered("O=NN1CCCC1")
        assert r.get("name") == "1-nitrosopyrrolidine"

    def test_nitrite_is_not_nitroso(self):
        """CCON=O (ethyl nitrite, -O-N=O) must NOT be spelled nitroso."""
        r = Orthonym(**BE).name_tiered("CCON=O")
        assert "nitroso" not in (r.get("name") or "")
