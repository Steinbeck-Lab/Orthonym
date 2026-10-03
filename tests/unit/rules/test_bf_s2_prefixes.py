"""Bridge prefixes slice S2 adds:14069-:14073,:14095-:14118,
 :14146-:14155) and the citation of an unsymmetric bridge's locants
 :14201). Every name was read back by OPSIN 2.9.0 to the input's full
InChIKey (S2 planning notes)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build, prefixes


def _name(smiles):
    res = build(Chem.MolFromSmiles(smiles))
    return res[0] if res else None


@pytest.mark.parametrize("smiles,name", [
    ("C1=CC23C=CC=CC2(C=C1)C=CC3", "4a,8a-prop[1]enonaphthalene"),        #:14071
    ("C1=CC23C=CC=CC2(C=C1)C=CC=C3", "4a,8a-buta[1,3]dienonaphthalene"),  #:14073
    ("C1=CC23C=CC=CC2(C=C1)CC=CC3", "4a,8a-but[2]enonaphthalene"),        #:14073
    ("C1=CC2OOC1c1ccccc12", "1,4-dihydro-1,4-epidioxynaphthalene"),       #:14101
    ("C1=CC2OCC1c1ccccc12", "1,4-dihydro-1,4-(epoxymethano)naphthalene"),  #:14155
    ("C1=CC2[SiH2]C1c1ccccc12", "1,4-dihydro-1,4-silanonaphthalene"),     #:14108
    ("C1=CC2[SnH2]C1c1ccccc12", "1,4-dihydro-1,4-stannanonaphthalene"),   #:14109
    ("C1=CC2BC1c1ccccc12", "1,4-dihydro-1,4-boranonaphthalene"),          #:14118
    ("C1=CC2N=NC1c1ccccc12", "1,4-dihydro-1,4-diazenonaphthalene"),       #:14112
    ("C1=CC23C=CC=CC2(C=C1)CCCCC3",
     "6,7,8,9-tetrahydro-5H-4a,9a-buta[1,3]dienobenzo[7]annulene"),
])
def test_bridge_prefixes(smiles, name):
    assert _name(smiles) == name


def test_an_unsymmetric_composite_bridge_cites_its_first_atom_first():
    # (:14201): locants "in the order expressed or implied by the name of the
    # bridge" -- the O of '(epoxymethano)' on the first locant (as in:14211
    # '7H-5,3-(epoxymethano)furo[2,3-c]pyran'); (b) (:14231) then prefers '1,4'
    # to '4,1', before (f) places the methyl. OPSIN reads '2-methyl-...-4,1-
    # (epoxymethano)...' to the same key: only this test holds the spelling.
    assert _name("CC1=CC2OCC1c1ccccc12") == "3-methyl-1,4-dihydro-1,4-(epoxymethano)naphthalene"


def test_two_identical_composite_bridges_are_not_spelled():
    # (:14173) 'bis' with composite bridges; OPSIN 2.9.0 cannot read
    # 'bis(epoxymethano)' (spec section 7)
    assert prefixes.bridge_text([("(epoxymethano)", (1, 4)), ("(epoxymethano)", (5, 8))]) is None


def test_composite_bridges_are_cited_alphabetically_without_parentheses():
    assert prefixes.citation_key("(epoxymethano)") == "epoxymethano"
    assert prefixes.bridge_text([("(epoxymethano)", (1, 4)), ("methano", (5, 8))]) == \
        "1,4-(epoxymethano)-5,8-methano"


@pytest.mark.parametrize("smiles", [
    "C1=CC2OOOC1c1ccccc12",   # -O-O-O-: the 1,2,3-benzotrioxepine reading has more atoms
                              # (b)) and no parent name in the tables
    "C1=CC2PC1c1ccccc12",     # -PH-: no prefix is spelled (and the phosphane is read as a
                              # principal characteristic group)
    "O=S1(=O)C2C=CC1c1ccccc12",  # -SO2-: a nonstandard bonding number:2744)
    "C12=CC=C(C3=CC=CC=C13)C#C2",  # a triple bond is not a bridge prefix
])
def test_declined(smiles):
    assert _name(smiles) is None
