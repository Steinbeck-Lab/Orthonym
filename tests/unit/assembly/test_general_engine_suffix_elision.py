"""The general engine elides the terminal 'a' of a multiplier before a vowel-initial suffix.

 (c) (the Blue Book, under ELISION OF VOWELS:7591): vowels are elided in
"the terminal letter 'a' in the names of numerical multiplicative prefixes when followed by a
suffix beginning with 'a' or 'o'"; '[1,1'-biphenyl]-3,3',4,4'-tetramine (PIN, '
(:7623), 'benzenehexol (PIN, (not benzenehexaol)' (:7625). The other name builders
join through naming_utils._join_multiplied_suffix; the general engine's chain and ring suffix
helpers concatenated the multiplier and spelled 'hexane-1,2,3,4,5-pentaol' and
'cyclohexane-1,2,3,4-tetraol' at best-effort. OPSIN reads both spellings, so the spelling
rests on this test; each engine name is read back here by an independent OPSIN call."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly.general_engine import _ring_suffix_text, _suffix_block, name_general
from tests.support.pin_tiers import name_breadth
from tests.support.rt_assert import name_is_rt_exact


@pytest.mark.parametrize("core,locants,text", [
    ("ol", [1, 2, 3, 4], "-1,2,3,4-tetrol"),
    ("ol", [1, 2, 3, 4, 5], "-1,2,3,4,5-pentol"),
    ("amine", [1, 2, 3, 4], "-1,2,3,4-tetramine"),
    ("one", [2, 3, 4, 5], "-2,3,4,5-tetrone"),
    ("ol", [1, 2], "-1,2-diol"),
    ("ol", [1, 2, 3], "-1,2,3-triol"),
    ("thiol", [1, 2, 3, 4], "-1,2,3,4-tetrathiol"),
    ("ol", [3], "-3-ol"),
])
def test_the_multiplier_is_elided_before_a_vowel_initial_suffix(core, locants, text):
    assert _suffix_block("locant", core, locants) == text
    assert _ring_suffix_text(core, locants) == text


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,ending", [
    ("OC1C(O)C(O)C(O)C(=NOC)C1", "cyclohexane-1,2,3,4-tetrol"),
    ("OCC(O)C(O)C(O)C(=NOC)C", "hexane-1,2,3,4-tetrol"),
    ("OCC(O)C(O)C(O)C(O)C=NOC", "hexane-1,2,3,4,5-pentol"),
    ("NCC(N)C(N)C(N)C(=NOC)C", "hexane-1,2,3,4-tetramine"),
])
def test_the_general_engine_spells_the_elided_suffix(smiles, ending):
    # The engine itself, called the way the best-effort tier calls it (allow_aromatic_general).
    # The PIN path now names these witnesses first: since PIN class program batch 2
    # the nitrogen side of C=N-OCH3 under a senior suffix is one enclosed
    # '(methoxyimino)' prefix, the Blue Book, "The prefix 'imino' for =NH is
    # used in presence of characteristic groups having seniority over imines";,
    #:7232; 'methyl [(methylimino)silyl]acetate (PIN)',:26578). So the shipped row comes
    # from the PIN path and no longer from the engine, and the engine is asked directly to
    # keep its own suffix helpers under test.
    mol = Chem.MolFromSmiles(smiles)
    namer = Orthonym()
    feats = namer._perceive(mol, smiles, Chem.MolToSmiles(mol, canonical=True))
    namer._classify(feats)
    engine = name_general(mol, feats, allow_aromatic_general=True)
    assert engine is not None, smiles
    assert engine.name.endswith(ending), engine.name
    assert name_is_rt_exact(engine.name, smiles), engine.name
    # The shipped best-effort name, whichever producer builds it, spells the same suffix.
    row = name_breadth(smiles)
    name = row.get("name") or ""
    assert name.endswith(ending), row
    assert name_is_rt_exact(name, smiles), row
