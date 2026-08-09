"""v30 #41 — the middle carbocyclic ring of a linear ring assembly has TWO
inter-ring junctions; a substituent/PCG on it must get a DETERMINISTIC lowest
locant (2', not 3') independent of input SMILES spelling.

Pre-fix: ``_get_substituent_info`` numbered the middle ring from the FIRST
inter-system connection atom (SMILES-order dependent), so p-terphenyl-2'-
carboxylic acid emitted ``-2'-`` or ``-3'-`` depending on the spelling. Same
molecule (0-wrong, SELF-01 blind; the PIN oracle uses fixed spellings, so the
gate's determinism check is blind to it). Found by fable review a998bea712.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym

pytestmark = pytest.mark.unit


def _namer():
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


def test_terphenyl_middle_ring_carboxy_locant_is_deterministic():
    """12 random spellings of the SAME molecule -> one canonical name."""
    nm = _namer()
    base = "OC(=O)c1cc(-c2ccccc2)ccc1-c1ccccc1"
    m = Chem.MolFromSmiles(base)
    names = set()
    for i in range(12):
        smi = Chem.MolToSmiles(m, doRandom=True)
        names.add(nm.name(smi))
    assert len(names) == 1, f"nondeterministic: {names}"
    # lowest-locant rule: the junction adjacent to the substituent is 1',
    # so the carboxy sits at 2' (the two outer phenyls are identical, so
    # either junction may be numbered 1').
    assert names == {"[1,1':4',1''-terphenyl]-2'-carboxylic acid"}
