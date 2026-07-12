"""W2F-P10 (P-25.5.3): polycomponent fusion + metheno-bridge parent ring systems.

These fused/bridged PARENT ring systems are cataloged in POLYCYCLIC_DATA with
OPSIN-authoritative numbering (the same curated-catalog mechanism as the sibling
dinaphtho[1,2-c:2',1'-m]picene entry; a general P-25.3/P-25.5 orientation engine
is not built). Bare compounds match by exact canonical SMILES; substituted
derivatives use the stored numbering. Also locks the P-16.3.3 fix that sets a
digit-initial parent stem off from a letter-ending substituent prefix with a
hyphen.

Fast, deterministic, OPSIN-free (production namer string assertions only). The
name+OPSIN round-trip envelope was verified separately via 
"""
import pytest
from rdkit import Chem
from orthonym.namer import name_compound

# Canonical SMILES -> expected PIN (BB:14527 for the full target)
BARE = {
    # single first-order attached component (P-25.3.1)
    "c1ccc2cc3c(ccc4cc5ccc6cc7ccccc7cc6c5cc43)cc2c1":
        "naphtho[2,3-a]pentaphene",
    # two identical attached components on pentaphene (P-25.3.4.1.2) — the core
    "c1ccc2cc3c(ccc4cc5ccc6cc7ccc8cc9ccccc9cc8c7cc6c5cc43)cc2c1":
        "dinaphtho[2,3-a:2',3'-o]pentaphene",
    # full P-25.5.3 target: di(metheno) bridges over the core
    "c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc1c1cc%10c9cc8c7cc6c5cc4c3cc21":
        "12,19:13,18-di(metheno)dinaphtho[2,3-a:2',3'-o]pentaphene",
}


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", list(BARE.items()))
def test_bare_parent_names(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", list(BARE.items()))
def test_determinism_random_spellings(smiles, expected):
    """Exact-canonical catalog match -> every random spelling yields one name."""
    m = Chem.MolFromSmiles(smiles)
    names = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(10)}
    assert names == {expected}


@pytest.mark.unit
def test_full_target_substituent_hyphen():
    """P-16.3.3: a substituent prefix before a locant-initial parent stem gets a
    hyphen (was 'N-methyl12,...' — a malformed PIN)."""
    # 1-methyl-<full target>  (OPSIN structure, RDKit-canonical)
    smi = ("CC1=CC=2C=C3C=CC=4C=C5C=CC6=C7C5=CC4C3=CC2C2=C1C=C1C=CC3="
           "C(C1=C2)C=C2C(C=CC(=C6)C2=C7)=C3")
    name = name_compound(smi)
    assert name == ("1-methyl-12,19:13,18-di(metheno)"
                    "dinaphtho[2,3-a:2',3'-o]pentaphene")


@pytest.mark.unit
def test_regression_digit_initial_parent_hyphen():
    """The same P-16.3.3 fix corrects the pre-existing '2-methyl9,10-...' bug."""
    # 2-methyl-9,10-dihydroanthracene
    assert name_compound("CC1=CC=2CC3=CC=CC=C3CC2C=C1") == \
        "2-methyl-9,10-dihydroanthracene"


@pytest.mark.unit
def test_letter_initial_parent_unchanged():
    """Letter-initial parents must NOT gain a spurious hyphen."""
    assert name_compound("Cc1ccc2ccccc2c1") == "2-methylnaphthalene"
    assert name_compound("CC=1C2=CC=CC=C2C=C2C=CC=CC12") == "9-methylanthracene"


# NB: the fail-closed boundary for uncataloged polycomponent systems (e.g.
# dibenzo[a,c]anthracene) is locked by the W2F-P10 `protect` gold in
#  — it relies on the SELF-01
# OPSIN round-trip gate and so is exercised in the OPSIN-backed pin_oracle eval,
# not in this OPSIN-free unit module.
