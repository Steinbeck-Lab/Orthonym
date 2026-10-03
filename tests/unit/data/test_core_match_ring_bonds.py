"""The catalogue core matcher accepts a match only when the matched ring system has
exactly the ring bonds of the catalogue pattern.

A catalogue name denotes its whole ring system, the fusion names; the existing
coverage guard already stops a smaller pattern inside a larger system). A substructure
match can also map every atom of a ring system and still miss a ring bond: mancude
cyclopenta[a]indene (12 atoms, 3 rings, 14 bonds) contains the 12-atom perimeter of
benzo[8]annulene (2 rings, 13 bonds) plus one transannular bond, and was matched as
'benzo[8]annulene' -- a name of a different molecule.
"""
from rdkit import Chem

from orthonym.data.fused_heterocycles import match_fused_heterocycle_core


def _match(smiles):
    r = match_fused_heterocycle_core(Chem.MolFromSmiles(smiles))
    return r[0] if r else None


def test_cyclopenta_a_indene_is_not_benzo_8_annulene():
    assert _match("C1=CC2=Cc3ccccc3C2=C1") is None
    assert _match("C1=CC=C2C1=CC=1C=CC=CC21") is None


def test_the_catalogue_systems_themselves_still_match():
    assert _match("Cc1ccc2c(c1)cccccc2") == "benzo[8]annulene"   # substituted benzo[8]annulene
    assert _match("CC1=CCc2ccccc21") == "1H-indene"                # 3-methyl-1H-indene
    assert _match("Cc1ccc2ncccc2c1") == "quinoline"
    assert _match("O=c1ccc2ccccc2o1") is not None                 # an entry with an exocyclic =O
