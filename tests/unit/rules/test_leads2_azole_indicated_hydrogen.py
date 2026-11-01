"""LEADS2 topic 4: an azole whose substituted nitrogen is not the first of its ring ('4-methyl-4H-1,2,4-triazole').

The PIN tier built '1-methyl-1,2,4-triazole' for ``Cn1cnnc1`` and the final check withdrew it ('Can't kekulize' was the
earlier reading of the decline): the ring was numbered from the substituted nitrogen, so the heteroatom locants were
1,3,4 and the name cited the methyl at 1, and the parent had lost its indicated hydrogen because a substituent stands in
its place. Two causes, one class (N-substituted triazoles and tetrazoles whose substituted nitrogen is not at 1 in the
numbering that the heteroatoms fix):

1. ``### **** NUMBERING`` (the Blue Book): "(a) fixed numbering in chains, rings, or ring systems" and
   "(b) indicated hydrogen for unsubstituted compounds; a higher locant may be needed at another position to
   accommodate a substituent suffix". The numbering of a Hantzsch-Widman ring is fixed by its heteroatoms,
   and the indicated hydrogen takes the lowest locant among the numberings that keep them. The numbering function made the
   pyrrole-type atom the start of the numbering whatever the other heteroatom locants were
   (``heterocycles.orient_heterocycle_with_substituents``).
2. ``****`` (:4916), sentence:4918: "Type 1 retained names of parent hydrides described in Chapters and
    have unlimited substitution by substituent groups cited either as suffixes or prefixes", so the substituent that
   stands in the place of the indicated hydrogen leaves the parent '4H-1,2,4-triazole'; ``****`` (:24639)
   "it is necessary to specify hydrogen atoms of ring atoms that are attached only by single bonds to adjacent ring
   atoms" and the PIN prints the parent inside a name: 'N,1,4-triphenyl-1H-1,2,4-triazol-4-ium-3-aminide (PIN)' (:42460).
   The parent's name did not count a substituted nitrogen as the atom that holds the indicated hydrogen
   (``_monocycle_indicated_h_prefix``).

Every expected name is read back by a fresh OPSIN call (``assert_pin_at_both_tiers``); the book prints no triazole
with a substituent on N4, so the names are the rules above applied.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    ("Cn1cnnc1", "4-methyl-4H-1,2,4-triazole"),
    ("CCn1cnnc1", "4-ethyl-4H-1,2,4-triazole"),
    ("Cc1nnc[nH]1", "3-methyl-4H-1,2,4-triazole"),
    ("Cn1cnnc1C", "3,4-dimethyl-4H-1,2,4-triazole"),
    ("Cc1nnc(C)n1C", "3,4,5-trimethyl-4H-1,2,4-triazole"),
    ("Cn1cnnc1N", "4-methyl-4H-1,2,4-triazol-3-amine"),
    ("Cn1cnnc1Cl", "3-chloro-4-methyl-4H-1,2,4-triazole"),
    ("OC(=O)c1nncn1C", "4-methyl-4H-1,2,4-triazole-3-carboxylic acid"),
    ("Cn1nccn1", "2-methyl-2H-1,2,3-triazole"),
    ("Cn1ccnn1", "1-methyl-1H-1,2,3-triazole"),
    ("Cn1c(C(=O)O)cnn1", "1-methyl-1H-1,2,3-triazole-5-carboxylic acid"),
    ("Cn1nncn1", "2-methyl-2H-tetrazole"),
    ("Cn1nnc(C)n1", "2,5-dimethyl-2H-tetrazole"),
]


@pytest.mark.parametrize("smiles,pin", ROWS)
def test_the_substituted_nitrogen_keeps_its_place_in_the_numbering_the_heteroatoms_fix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


CONTROLS = [
    # the substituted nitrogen is the first of the ring in the numbering the heteroatoms fix: unchanged
    ("Cn1cncn1", "1-methyl-1H-1,2,4-triazole"),
    ("Cn1ccnc1", "1-methyl-1H-imidazole"),
    ("Cn1cnnn1", "1-methyl-1H-tetrazole"),
    ("Cc1ccn(C)n1", "1,3-dimethyl-1H-pyrazole"),
    ("Cn1nccc1C(=O)O", "1-methyl-1H-pyrazole-5-carboxylic acid"),
    ("c1nnc[nH]1", "4H-1,2,4-triazole"),
    ("Cn1cccc1", "1-methyl-1H-pyrrole"),
]


@pytest.mark.parametrize("smiles,pin", CONTROLS)
def test_a_ring_that_was_numbered_correctly_keeps_its_name(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_heteroatom_locants_win_over_the_indicated_hydrogen():
    """The mutation: numbering from the N-substituted nitrogen gives the heteroatom locants 1,3,4 for the 4-methyl
    triazole; the numbering that keeps 1,2,4 puts the N-substituted nitrogen at 4."""
    from rdkit import Chem

    from orthonym.rules.heterocycles import orient_heterocycle_with_substituents
    mol = Chem.MolFromSmiles("Cn1cnnc1")
    ring = mol.GetRingInfo().AtomRings()[0]
    _, locant = orient_heterocycle_with_substituents(mol, ring, {1}, mancude_hydrogen=True)
    hetero = sorted(locant[a.GetIdx()] for a in mol.GetAtoms() if a.GetIdx() in ring and a.GetSymbol() == "N")
    assert hetero == [1, 2, 4] and locant[1] == 4, locant


# --- the cations: the neutral producer's numbering is not the ions' ----------------------------------------------------
#
# Fix a performance pass (review C1): the azolium namers read the numbering of the BARE parent and name the neutral parent of a
# cation through ``Orthonym.name``; the numbering of this module reaches only the neutral producer
# (``orient_heterocycle_with_substituents(..., mancude_hydrogen=True)``, ``name_heterocycle(..., substituent_holds_h=True)``),
# so the cation names are the book's pattern: 'N,1,4-triphenyl-1H-1,2,4-triazol-4-ium-3-aminide (PIN)' (the Blue Book:
# the indicated hydrogen at 1, the 'ium' at 4) and '1H-imidazol-3-ium (PIN)' (:41396). (:41366,:41368): the
# 'ium' locant is the position of the added hydron.
CATIONS = [
    ("C[n+]1cn(C)nc1", "1,4-dimethyl-1H-1,2,4-triazol-4-ium"),
    ("Cn1cn[n+](C)c1", "1,4-dimethyl-1H-1,2,4-triazol-4-ium"),      # the same cation, drawn with the other charge
    ("C[n+]1cn(C)cn1", "1,4-dimethyl-1H-1,2,4-triazol-4-ium"),
    ("c1ccccc1[n+]1cn(C)nc1", "1-methyl-4-phenyl-1H-1,2,4-triazol-4-ium"),
    ("C[n+]1cn(C)nc1N", "3-amino-1,4-dimethyl-1H-1,2,4-triazol-4-ium"),
    ("C[n+]1ccn(C)c1", "1,3-dimethyl-1H-imidazol-3-ium"),
    ("c1c[nH]c[nH+]1", "1H-imidazol-3-ium"),
    ("C[n+]1cnc[nH]1", "1-methyl-1H-1,2,4-triazol-2-ium"),
    # the hydron on the end nitrogen that holds the hydrogen: '3-ium', not '2-ium' (the full key cannot tell them apart)
    ("C[n+]1cc[nH]n1", "1-methyl-1H-1,2,3-triazol-3-ium"),
    ("Cn1cc[nH+]n1", "1-methyl-1H-1,2,3-triazol-3-ium"),
]


@pytest.mark.parametrize("smiles,name", CATIONS)
def test_a_cation_keeps_the_names_of_its_azolium_namers(smiles, name):
    from tests.support.pin_tiers import name_breadth, name_default
    from tests.support.rt_assert import name_is_rt_exact
    for row in (name_default(smiles), name_breadth(smiles)):
        assert row["name"] == name, row
    assert name_is_rt_exact(name, smiles)


def test_the_hydron_of_an_azolium_name_is_checked_on_the_atom_that_holds_it():
    """Mutation of the check: '1-methyl-1H-1,2,3-triazol-2-ium' (the hydron on the middle nitrogen) has the full
    InChIKey of the input and was shipped as a PIN; read against the drawing that holds the hydron it is refused."""
    from rdkit import Chem

    from orthonym.rules.charged_router import _cation_name_rt_ok
    res = Chem.MolFromSmiles("Cn1cc[nH+]n1")
    assert _cation_name_rt_ok("1-methyl-1H-1,2,3-triazol-3-ium", res)
    assert not _cation_name_rt_ok("1-methyl-1H-1,2,3-triazol-2-ium", res)
