"""Task I: the systematic alpha-amino-acid namer must PROVE its structure.

`_name_amino_acid_systematic` emits `2-amino{stem}anoic acid`, a name that denotes
exactly one constitution:

    HOOC-CH(NH2)-(CH2)n-CH3

...an unbranched, fully saturated carbon backbone, one terminal NH2 on C2, and no
other heteroatom anywhere. Historically the stem came from a whole-molecule carbon
count (`sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')`) guarded by a
growing DENY-LIST of SMARTS. A deny-list can only ever exclude the holes somebody
already found, and every hole it misses emits a name for a DIFFERENT MOLECULE:

  * an in-chain secondary N   (`CNCC(=O)O`, sarcosine)  -> `2-aminopropanoic acid` (alanine)
  * an in-chain ether O       (`COCC(N)C(=O)O`)         -> `2-aminobutanoic acid`
  * a C=C / C#C in the chain  (`C=CCC(N)C(=O)O`)        -> `2-aminopentanoic acid`

Each of those heteroatoms/unsaturations is silently CONTRACTED: its flanking carbons
are counted into the backbone and the heteroatom vanishes from the name.

These tests pin the ALLOW-LIST invariant instead: the producer must return None for
anything it cannot prove is the structure its name denotes.
"""

import pytest
from rdkit import Chem

from orthonym.rules.amino_acids import _name_amino_acid_systematic


def _name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles}"
    return _name_amino_acid_systematic(mol)


# --------------------------------------------------------------------------
# The producer must REFUSE anything that is not HOOC-CH(NH2)-(CH2)n-CH3.
# Refusing routes the molecule to the general polyfunctional pipeline, which
# names these correctly (verified via the CLI in the Task I report).
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,why",
    [
        # --- in-chain nitrogen: the headline wrong-molecule defect -------------
        ("CNCC(=O)O", "sarcosine: N-methyl absorbed as C-methyl -> alanine"),
        ("CCNCC(=O)O", "N-ethylglycine -> 2-aminobutanoic acid"),
        ("CNCCNCCNCC(=O)O", "3 in-chain N contracted; 2 nitrogens silently dropped"),
        ("NCCNCC(=O)O", "in-chain N contracted, one N dropped"),
        ("CNC(C)C(=O)O", "N-methylalanine -> 2-aminobutanoic acid"),
        # --- in-chain ether oxygen: same shape as the patched thioether bug ----
        ("COCC(N)C(=O)O", "O-methylserine: ether O contracted and dropped"),
        # --- unsaturation in the backbone -------------------------------------
        ("C=CCC(N)C(=O)O", "allylglycine: C=C flattened to a saturated stem"),
        ("C#CCC(N)C(=O)O", "propargylglycine: C#C flattened to a saturated stem"),
    ],
)
def test_producer_refuses_structures_it_cannot_denote(smiles, why):
    assert _name(smiles) is None, f"must refuse ({why}), got {_name(smiles)!r}"


def test_distinct_molecules_never_collapse_onto_one_name():
    """Three constitutionally different acids must not all be `2-aminopentanoic acid`."""
    norvaline = _name("CCCC(N)C(=O)O")
    allylglycine = _name("C=CCC(N)C(=O)O")
    propargylglycine = _name("C#CCC(N)C(=O)O")
    assert norvaline == "2-aminopentanoic acid"
    # the two unsaturated ones must not share norvaline's name
    assert allylglycine != norvaline
    assert propargylglycine != norvaline


# --------------------------------------------------------------------------
# The producer must KEEP naming the straight-chain saturated 2-amino acids it
# is actually for. These are the regression anchors: the fix fails closed, so
# it must not fail closed on its own class.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CCC(N)C(=O)O", "2-aminobutanoic acid"),          # 2-aminobutyric acid (Abu)
        ("CCCC(N)C(=O)O", "2-aminopentanoic acid"),        # norvaline
        ("CCCCC(N)C(=O)O", "2-aminohexanoic acid"),        # norleucine
        ("CCCCCC(N)C(=O)O", "2-aminoheptanoic acid"),
        ("CC(N)C(=O)O", "2-aminopropanoic acid"),          # alanine (systematic)
        ("NCC(=O)O", "2-aminoethanoic acid"),              # glycine (systematic)
    ],
)
def test_producer_still_names_its_own_class(smiles, expected):
    assert _name(smiles) == expected


def test_stereodescriptor_still_injected():
    """The CIP prefix path must survive the allow-list."""
    name = _name("CC[C@H](N)C(=O)O")
    assert name is not None
    assert name.endswith("2-aminobutanoic acid")
    assert name.startswith("(")


# --------------------------------------------------------------------------
# The previously-patched deny-list holes must stay closed under the allow-list.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "smiles",
    [
        "CSCCC(N)C(=O)O",        # methionine: thioether (v23 Phase 12 F-THIOETHER-DROP)
        "C[Se]CC(N)C(=O)O",      # selenoether (v26 BP-3 C1)
        "CC[C@H](C)[C@@H](N)C(=O)O",  # isoleucine: branched (v23 Phase 12)
        "OCC(N)C(=O)O",          # serine: hydroxy (PEP-02)
        "OC(=O)CCC(N)C(=O)O",    # glutamic acid: dicarboxylic (AMAC-01)
        "NCCCCC(N)C(=O)O",       # lysine: two primary amines
        "OC(=O)C1CCCN1",         # proline: ring
    ],
)
def test_previously_patched_holes_stay_closed(smiles):
    assert _name(smiles) is None
