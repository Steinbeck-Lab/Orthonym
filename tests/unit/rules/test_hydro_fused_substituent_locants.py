"""A substituent on a hydro-fused parent inherits the parent's numbering.

`rules.polycyclics.name_partially_saturated_carbocycle` numbers a partially
saturated fused carbocycle by matching its mancude parent skeleton, and
`detect_carbocyclic_partial_saturation` returns the winning `atom_to_locant`
map. For `CC1CCc2ccccc2C1` that map is::

    {10: 1, 1: 2, 2: 3, 3: 4, 4: '4a', 5: 5, 6: 6, 7: 7, 8: 8, 9: '8a'}

and atom 1 -- the methyl-bearing sp3 carbon -- is locant **2**. The producer
used to return only the name string, so the map died there; the composer's
enrichment pass then RE-DERIVED a numbering from `features` and placed the
methyl at locant 1, naming a different molecule (`CC1CCCc2ccccc21`) that
SELF-01 suppressed -- so a correct name was lost to an abstention.

The fix threads the producer's own map into enrichment. Two numberings for one
name is the defect; there is now exactly one, and the same map also spells the
principal-characteristic-group suffix.

Governing rules, quoted with their section headings:

* **P-58.2.5 "Nondetachable hydro prefixes *vs*. indicated hydrogen"**
  (`BlueBookV2/BlueBookV2.md:24890`), on the PIN example
  `5,8-dioxo-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid (PIN)`:
  "detachable but nonalphabetized hydro prefixes **do not have precedence over
  the principal characteristic group for low numbering**, but has precedence
  over other detachable prefixes." So low locants go to the suffix FIRST, then
  to hydro, then to the detachable prefixes.
* **P-63.1 "HYDROXY COMPOUNDS AND CHALCOGEN ANALOGUES"**
  (`BlueBookV2/BlueBookV2.md:26880`): "(1) 5,6,7,8-tetrahydronaphthalen-2-ol
  (PIN)" -- verbatim, and the load-bearing test in this file: the hydroxy
  suffix pulls the numbering onto the AROMATIC ring (2, not 6), which only
  happens if the suffix and the hydro prefix read the SAME map.
* **P-16.7.1(a)** terminal-'e' elision: `naphthalen-2-ol` but
  `naphthalene-1,3-diol` (the BB prints both forms at `:26866` /`:26880`).

Every expected name in this file was checked name -> OPSIN 2.9.0 -> canonical
SMILES against its input.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound

pytestmark = pytest.mark.opsin_gate


# --------------------------------------------------------------------------
# Acceptance: the two rows the re-derived locant cost us.
# --------------------------------------------------------------------------

ACCEPTANCE = [
    # (smiles, expected PIN)
    ("CC1CCc2ccccc2C1", "2-methyl-1,2,3,4-tetrahydronaphthalene"),
    ("OC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalen-2-ol"),
]

# Correct before the fix -- a change here is a regression, not progress.
UNCHANGED = [
    ("CC1CCCc2ccccc21", "1-methyl-1,2,3,4-tetrahydronaphthalene"),
    ("Cc1ccc2c(c1)CCCC2", "6-methyl-1,2,3,4-tetrahydronaphthalene"),
    ("C1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalene"),
    ("CC1Cc2ccccc2C1", "2-methyl-2,3-dihydro-1H-indene"),
    # the sp3-ring gem-free dimethyl case already numbered correctly
    ("CC1CC(C)c2ccccc2C1", "1,3-dimethyl-1,2,3,4-tetrahydronaphthalene"),
    # An -OH on an EXOCYCLIC carbon is NOT a ring '-ol': the carbinol carbon is
    # the parent, so this must keep its ring-as-substituent name.
    ("OCC1CCc2ccccc2C1", "(1,2,3,4-tetrahydronaphthalen-2-yl)methanol"),
]

# The rest of the class the same single map unlocks.
CLASS = [
    # P-63.1 (:26880) BB-verbatim PIN. The suffix pulls the numbering onto the
    # aromatic ring, so hydro becomes 5,6,7,8 -- one map, or this is impossible.
    ("Oc1ccc2c(c1)CCCC2", "5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("OC1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalen-1-ol"),
    # multiplied suffix -> 'di' is consonant-initial, so the terminal 'e' stays
    ("OC1CC(O)c2ccccc2C1", "1,2,3,4-tetrahydronaphthalene-1,3-diol"),
    # the pre-existing ring-carboxylic-acid PCG, whose suffix was built from the
    # same map all along but was vetoed downstream as an atom drop
    ("OC(=O)C1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalene-2-carboxylic acid"),
    ("OC(=O)C1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalene-1-carboxylic acid"),
    # prefix-only substituents on the sp3 ring
    ("CCC1CCc2ccccc2C1", "2-ethyl-1,2,3,4-tetrahydronaphthalene"),
    ("CC1CCc2cc(C)ccc2C1", "2,6-dimethyl-1,2,3,4-tetrahydronaphthalene"),
]


@pytest.mark.parametrize("smiles,expected", ACCEPTANCE)
def test_sp3_ring_substituent_inherits_the_parent_numbering(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_already_correct_rows_are_byte_identical(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", CLASS)
def test_rest_of_the_hydro_fused_substituent_class(smiles, expected):
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# The map is threaded, not recomputed.
# --------------------------------------------------------------------------


def test_producer_returns_the_map_it_spelled_the_name_from():
    """`name_partially_saturated_carbocycle_with_locants` yields the name
    together with the numbering it was spelled from, and that numbering is
    `detect_carbocyclic_partial_saturation`'s own -- not a second derivation."""
    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle,
        name_partially_saturated_carbocycle_with_locants,
    )
    from orthonym.rules.partial_saturation import (
        detect_carbocyclic_partial_saturation,
    )

    mol = Chem.MolFromSmiles("CC1CCc2ccccc2C1")
    result = name_partially_saturated_carbocycle_with_locants(mol)
    assert result is not None
    atom_to_locant = result.atom_to_locant

    # the string-returning function is a thin wrapper over the same call
    assert result.name == name_partially_saturated_carbocycle(mol)

    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    detected = detect_carbocyclic_partial_saturation(mol, ring_atoms)
    assert atom_to_locant == detected["atom_to_locant"]

    # the methyl-bearing sp3 carbon is locant 2
    assert atom_to_locant[1] == 2
    # fusion locants are strings and must survive the hand-off
    assert atom_to_locant[4] == "4a"
    assert atom_to_locant[9] == "8a"


def test_producer_wrapper_is_none_when_the_class_does_not_apply():
    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle,
        name_partially_saturated_carbocycle_with_locants,
    )

    mol = Chem.MolFromSmiles("CC1Cc2ccccc2C1")  # indane -- not this producer's
    assert name_partially_saturated_carbocycle_with_locants(mol) is None
    assert name_partially_saturated_carbocycle(mol) is None


def test_enrich_handler_name_supplied_map_overrides_the_features_fallback():
    """`_enrich_handler_name` takes an OPTIONAL `atom_to_locant`. Supplied, it
    decides the substituent locant; absent, behaviour is exactly as before.

    MEASURED SCOPE, so this test is not read as more than it is: on a 7,500-row
    corpus sweep the producer fires on 38 molecules, enrichment is entered 27
    times, and with `already_spelled_atoms` also supplied it changes the name
    **0** times -- the producer spells every ring substituent itself and fails
    closed otherwise, so there is nothing left to enrich. The override is
    therefore the CORRECTNESS PRECONDITION of the enrichment call that remains
    (if the producer's coverage ever has a hole, the substituent lands on the
    inherited locant rather than a re-derived one), not the mechanism that fixes
    the defect -- that is the single-speller producer. This test pins the
    precondition directly, by calling enrichment on the UNDECORATED parent.
    """
    from orthonym.assembly.composer import _enrich_handler_name
    from orthonym.namer import Orthonym

    smiles = "CC1CCc2ccccc2C1"
    engine = Orthonym()
    mol = Chem.MolFromSmiles(smiles)
    features = engine._perceive(mol, smiles, Chem.MolToSmiles(mol))

    base = "1,2,3,4-tetrahydronaphthalene"
    correct = "2-methyl-1,2,3,4-tetrahydronaphthalene"

    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle_with_locants,
    )
    produced = name_partially_saturated_carbocycle_with_locants(mol)

    # supplied map, nothing declared already-spelled -> the producer's locant
    assert _enrich_handler_name(
        features, base, "partial_sat", atom_to_locant=produced.atom_to_locant,
    ) == correct
    # no map -> the features-derived fallback, which is NOT it. That is the
    # second numbering this whole change exists to remove.
    assert _enrich_handler_name(features, base, "partial_sat") != correct
    # and with the producer's already-spelled set, enrichment adds nothing --
    # the producer's own name is complete.
    assert _enrich_handler_name(
        features, produced.name, "partial_sat",
        atom_to_locant=produced.atom_to_locant,
        already_spelled_atoms=produced.spelled_offring_atoms,
    ) == produced.name


def test_handler_publishes_the_map_as_its_locant_hint():
    from orthonym.assembly.handlers.partial_sat import name_partial_sat
    from orthonym.namer import Orthonym

    smiles = "CC1CCc2ccccc2C1"
    engine = Orthonym()
    mol = Chem.MolFromSmiles(smiles)
    features = engine._perceive(mol, smiles, Chem.MolToSmiles(mol))

    from orthonym.assembly.candidate_pool import pop_pool, push_pool

    push_pool(features)
    try:
        result = name_partial_sat(features)
    finally:
        pop_pool()
    assert result is not None
    assert result.name == "2-methyl-1,2,3,4-tetrahydronaphthalene"
    assert result.atom_to_locant_hint is not None
    assert result.atom_to_locant_hint[1] == 2


# --------------------------------------------------------------------------
# Fail-closed boundaries preserved.
# --------------------------------------------------------------------------


def test_ring_fusion_atom_substituent_still_declines():
    """P-58.2.2.3 / R12: neither the producer nor enrichment can place a
    substituent on a ring-FUSION atom, so the producer must keep failing
    closed there rather than dropping the atom."""
    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle_with_locants,
    )

    # naphthalene-4a,8a-diol's hydro relative: substituents on both fusion atoms
    mol = Chem.MolFromSmiles("OC12CCCCC1(O)c1ccccc12")
    assert name_partially_saturated_carbocycle_with_locants(mol) is None


def test_exocyclic_hydroxy_is_not_a_ring_ol_suffix():
    """The '-ol' PCG is scoped to an -OH on a ring atom of the fused system.
    An -OH on an exocyclic carbon makes that carbon the parent, so the ring
    must stay a substituent -- never `2-...-1,2,3,4-tetrahydronaphthalen-?-ol`."""
    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle_with_locants,
    )

    mol = Chem.MolFromSmiles("OCC1CCc2ccccc2C1")
    result = name_partially_saturated_carbocycle_with_locants(mol)
    # the producer may decline or name the bare parent, but it must never
    # claim an -ol suffix for an oxygen it does not carry
    if result is not None:
        assert not result.name.endswith("ol")


def test_ol_suffix_yields_to_the_senior_carboxylic_acid():
    """P-41 seniority: a ring -COOH outranks a ring -OH, so the acid keeps the
    suffix and the hydroxy stays a detachable prefix (never two suffixes)."""
    from orthonym.rules.polycyclics import (
        name_partially_saturated_carbocycle_with_locants,
    )

    mol = Chem.MolFromSmiles("OC(=O)C1CCc2ccccc2C1O")
    result = name_partially_saturated_carbocycle_with_locants(mol)
    if result is not None:
        name = result.name
        assert "carboxylic acid" in name
        assert not name.endswith("ol")
