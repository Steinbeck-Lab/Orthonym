"""a phase (v52): fusion/retained ring parent must win over von Baeyer.
Rows from benchmarks/bb_conformance (def_id in comments). Each already
round-trips as a von-Baeyer degrade today (RIGHT_MOL_NONPIN); this pins the PIN."""
import pytest
from orthonym import name_compound

# --- Regression guards: these already MATCH and must stay MATCH ---
@pytest.mark.parametrize("smiles,expected", [
    ("c1ccccc1", "benzene"),
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("Oc1ccc2ccccc2c1", "naphthalen-2-ol"),
    ("OC(=O)c1ccc2ccccc2c1", "naphthalene-2-carboxylic acid"),
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),
])
def test_easy_cases_stay_match(smiles, expected):
    assert name_compound(smiles) == expected

# --- Sub-pattern A: ring assemblies of naphthalene, ---
@pytest.mark.parametrize("smiles,expected", [
    # base 2,2'-binaphthalene (unit regression, not a gold row)
    ("c1ccc2cc(-c3ccc4ccccc4c3)ccc2c1", "2,2'-binaphthalene"),
    # 63.1.2
    ("Oc1c(-c2ccc3ccccc3c2O)ccc2ccccc12", "[2,2'-binaphthalene]-1,1'-diol"),
    # 63.1.2
    ("Oc1cc(-c2ccc3cccc(O)c3c2)cc2ccccc12", "[2,2'-binaphthalene]-4,8'-diol"),
])
def test_naphthalene_ring_assembly(smiles, expected):
    assert name_compound(smiles) == expected

# --- Sub-pattern B: added-indicated-H / quinoid naphthalene ---
@pytest.mark.parametrize("smiles,expected", [
    ("OC12C=CC=CC1=CCC=C2", "naphthalen-4a(2H)-ol"),           # 63.1.2
    ("NC12C=CC=CC1=CCC=C2", "naphthalen-4a(2H)-amine"),        # 62.2.6.2
    ("NC1C=CC2(N)C=CC=CC2=C1", "naphthalene-2,4a(2H)-diamine"),# 62.2.6.2
    ("OC12C=CC=CC1(O)C=CC=C2", "naphthalene-4a,8a-diol"),      # 14.7.2
    ("CN=C1C=CC(=NC)c2ccccc21",
     "N1,N4-dimethylnaphthalene-1,4-diimine"),                 # 62.3.1.1
])
def test_added_hydrogen_naphthalene(smiles, expected):
    assert name_compound(smiles) == expected

# --- Task 6: selection guard -- the retained/fusion offer must
# never lose to a von-Baeyer degrade for any Phase-4 target. Covers every
# SMILES in the two groups above (3 ring-assembly + 5 added-H = 8). ---
_ALL_A_B_TARGETS = [
    "c1ccc2cc(-c3ccc4ccccc4c3)ccc2c1",
    "Oc1c(-c2ccc3ccccc3c2O)ccc2ccccc12",
    "Oc1cc(-c2ccc3cccc(O)c3c2)cc2ccccc12",
    "OC12C=CC=CC1=CCC=C2",
    "NC12C=CC=CC1=CCC=C2",
    "NC1C=CC2(N)C=CC=CC2=C1",
    "OC12C=CC=CC1(O)C=CC=C2",
    "CN=C1C=CC(=NC)c2ccccc21",
]

@pytest.mark.parametrize("smiles", _ALL_A_B_TARGETS)
def test_no_von_baeyer_degrade(smiles):
    name = name_compound(smiles)
    assert "bicyclo[4.4.0]deca" not in name
    assert "cyclohexa-1,3,5-trien" not in name


# --- a review defect D6: the added-H suffix producer builds ONLY the bare
# naphthalene parent + suffix; a ring substituent it cannot render was SILENTLY
# DROPPED (0-wrong rested only on). It must fail closed (return None) on
# any off-ring heavy atom that is not a suffix O/N. ---
def test_d6_ring_substituent_fails_closed():
    from rdkit import Chem
    from orthonym.rules.partial_saturation import (
        name_added_h_fused_carbocycle_suffix,
    )
    # OC12C=C(C)C=CC1=CCC=C2 = naphthalen-4a(2H)-ol WITH a ring methyl. At HEAD
    # the producer dropped the methyl and returned 'naphthalen-4a(2H)-ol'.
    mol = Chem.MolFromSmiles("OC12C=C(C)C=CC1=CCC=C2")
    assert name_added_h_fused_carbocycle_suffix(mol) is None


@pytest.mark.parametrize("smiles,expected", [
    ("OC12C=CC=CC1=CCC=C2", "naphthalen-4a(2H)-ol"),
    ("NC12C=CC=CC1=CCC=C2", "naphthalen-4a(2H)-amine"),
    ("OC12C=CC=CC1(O)C=CC=C2", "naphthalene-4a,8a-diol"),
])
def test_d6_guard_keeps_unsubstituted_targets(smiles, expected):
    # The D6 fail-closed guard must not over-reject the bare targets.
    from rdkit import Chem
    from orthonym.rules.partial_saturation import (
        name_added_h_fused_carbocycle_suffix,
    )
    assert name_added_h_fused_carbocycle_suffix(Chem.MolFromSmiles(smiles)) == expected


# --- a review defects D4/D5: per-COMPONENT numbering /. The
# T2/ per-atom automorphism minimisation gave a fused component with a
# non-trivial junction stabiliser (anthracene 9/10; a doubly-linked middle
# naphthalene) an inconsistent NON-numbering that voided -> abstain.
# One numbering per component (min over automorphisms of the junction-locant
# tuple, then the substituent-locant tuple) recovers the PIN. ---
def test_d4_bianthracene_diol_per_component_numbering():
    # OPSIN round-trips [9,9'-bianthracene]-1,5-diol to this SMILES; at HEAD the
    # engine built [9,9'-bianthracene]-1,4-diol (per-atom {1,4} not {1,5}).
    assert (name_compound("Oc1cccc2c(-c3c4ccccc4cc4ccccc34)c3c(O)cccc3cc12")
            == "[9,9'-bianthracene]-1,5-diol")


def test_d5_ternaphthalene_middle_junctions_distinct():
    # A linear ternaphthalene whose MIDDLE unit is linked at two positions. At
    # HEAD both middle junctions were minimised INDEPENDENTLY to 2'
    # (2,2':2',2'', a NON-numbering that voided -> abstain). The
    # per-component numbering now gives the middle unit's two junctions DISTINCT
    # locants {2',6'}, so a valid name for the CORRECT molecule is emitted
    # (breadth recovered; not a bb gold row).
    #
    # NOTE: the exact lowest-locant prime ORDERING is 2,2':6',2'' but the engine
    # emits the equivalent 2,6':2',2'' (both OPSIN-round-trip to the same
    # molecule). Choosing the unprimed terminal by lowest DESCRIPTOR for
    # symmetric MULTI-ring terminals is a separate assembly-ordering refinement
    # (follow-on) -- _order_systems_along_path picks min(terminals) by index and
    # the downstream compare_locant_sets only reorders SINGLE-ring components.
    out = name_compound("c1ccc2cc(-c3ccc4cc(-c5ccc6ccccc6c5)ccc4c3)ccc2c1")
    assert out == "2,6':2',2''-ternaphthalene"
    assert out != "2,2':2',2''-ternaphthalene"  # the fixed non-numbering
