"""v23 Phase 11 (CARB-01) — carbohydrate F-OXANE-DROP root-cause + determinism.

These lock in the root-cause fix (the PIN gate does NOT run the unit suite, so a
regression in the shared ring-substituent numbering path would otherwise slip
through). The SELF-01 self-consistency gate is OFF in the unit suite (conftest
autouse fixture), so the substitutive sugar names appear directly rather than
being suppressed to 'unknown'.

F-OXANE-DROP: a sugar ring that is NOT catalog-recognised (thio / anhydro /
mid-chain-deoxy sugar) falls through to the substitutive heterocycle namer. That
namer used to render an exocyclic -CH2OH as 'methyl' (oxygen dropped = structure
loss). It must now render it as '(hydroxymethyl)' (a valid, OPSIN-round-trippable
substitutive alternative to the carbohydrate-stem PIN), fail closed (decline,
never drop atoms) on heteroatom-anchored / unnameable exocyclic groups, AND
number the ring DETERMINISTICALLY (the -ol suffix gets the lowest locants,
independent of the input SMILES atom order).

WSD-08 glycoside-linkage is DEFERRED (its blocker is the principal_ring P-44.1
selection race, a separate high-blast-radius parent-selection fix); it stays a
documented known_protect_failure and is not asserted here.
"""

from rdkit import Chem
import pytest

from orthonym.namer import name_compound


# --- F-OXANE-DROP: exocyclic -CH2OH survives as (hydroxymethyl), lowest-locant -

@pytest.mark.parametrize("smiles, expected", [
    # 5-thio-beta-D-glucopyranose: -ol suffix at the lowest set {2,3,4,5}
    ("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](S1)CO",
     "(2R,3R,4S,5S,6R)-6-(hydroxymethyl)thiane-2,3,4,5-tetraol"),
    # 5-thio-alpha-D-glucopyranose
    ("O[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](S1)CO",
     "(2S,3R,4S,5S,6R)-6-(hydroxymethyl)thiane-2,3,4,5-tetraol"),
    # 5-thio-D-mannose
    ("OC[C@H]1S[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O",
     "(2R,3S,4S,5S,6R)-6-(hydroxymethyl)thiane-2,3,4,5-tetraol"),
    # 1,5-anhydro-D-glucitol (oxane ring)
    ("C1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO",
     "(2R,3S,4R,5S)-2-(hydroxymethyl)oxane-3,4,5-triol"),
    # 2-deoxy-beta-D-erythro-pentofuranose (oxolane ring): diol at {2,4}
    ("O[C@H]1C[C@H](O)[C@H](O1)CO",
     "(2R,4S,5R)-5-(hydroxymethyl)oxolane-2,4-diol"),
])
def test_foxane_drop_hydroxymethyl_lowest_locant(smiles, expected):
    assert name_compound(smiles) == expected


def test_foxane_drop_no_bare_methyl_for_ch2oh():
    """The -CH2OH must NOT collapse to a bare 'methyl' (the structure-loss bug)."""
    name = name_compound("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](S1)CO")
    assert "hydroxymethyl" in name
    assert "methylthiane" not in name  # the old F-OXANE-DROP output


@pytest.mark.parametrize("smiles", [
    "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](S1)CO",   # 5-thio-glucopyranose
    "C1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO",        # 1,5-anhydroglucitol
    "O[C@H]1C[C@H](O)[C@H](O1)CO",                   # 2-deoxypentofuranose
])
def test_foxane_drop_numbering_is_deterministic(smiles):
    """The ring numbering must not depend on the input SMILES atom order:
    every equivalent spelling of the molecule yields the SAME name."""
    mol = Chem.MolFromSmiles(smiles)
    names = {name_compound(Chem.MolToSmiles(mol, doRandom=True)) for _ in range(12)}
    assert len(names) == 1, f"non-deterministic numbering: {names}"


def test_foxane_drop_failclosed_drops_no_heteroatoms():
    """An unnameable exocyclic ester group fails closed (no structure-loss name).

    alpha-D-glucopyranose 2-sulfate: the -O-SO3H ester O has no nameable prefix,
    so the heterocycle candidate declines rather than emit a sulfate-dropping
    name like '6-methyloxane-2,4,5-triol'.
    """
    name = name_compound("S(=O)(=O)(O)O[C@H]1[C@@H](O)O[C@@H]([C@H]([C@@H]1O)O)CO")
    assert "methyloxane" not in name  # never the methyl-for-CH2OH structure loss


# --- Regression guards: catalog sugars + ordinary heterocycles unchanged ----

@pytest.mark.parametrize("smiles, expected", [
    ("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "beta-D-glucopyranose"),
    ("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "alpha-D-glucopyranose"),
    ("Cc1ccccn1", "2-methylpyridine"),
    ("CC1CCCCO1", "2-methyloxane"),
    ("OC1CCOCC1", "oxan-4-ol"),
    ("OC1CCNCC1", "piperidin-4-ol"),
])
def test_no_regression_catalog_and_ordinary(smiles, expected):
    assert name_compound(smiles) == expected
