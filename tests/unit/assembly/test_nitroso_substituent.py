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


class TestNitrosoOnHeterocycleWithSuffix:
    """v30 tail #24: an N-nitroso ring that ALSO carries a principal-group
    suffix (acid / ester). The heterocycle handler's hetero-substituent
    identifier lacked a -N=O branch, so it flagged nitroso 'unnameable' and
    the whole heterocycle candidate declined -- even though bare
    1-nitrosopyrrolidine already named. Both now reach T1 (PIN)."""

    def test_nitrosoproline_acid_pin(self):
        # Default PIN tier: the acid is now a clean T1 PIN.
        r = Orthonym().name_tiered("O=NN1CCCC1C(=O)O")
        assert r.get("name") == "1-nitrosopyrrolidine-2-carboxylic acid"

    def test_methyl_nitrosoprolinate_pin(self):
        # The methyl ester (#24) — its acyl owner is the same ring acid.
        r = Orthonym().name_tiered("COC(=O)C1CCCN1N=O")
        assert r.get("name") == "methyl 1-nitrosopyrrolidine-2-carboxylate"

    def test_n_methyl_analog_unchanged(self):
        # Byte-identity guard: the N-methyl sibling is untouched.
        r = Orthonym().name_tiered("CN1CCCC1C(=O)O")
        assert r.get("name") == "1-methylpyrrolidine-2-carboxylic acid"
