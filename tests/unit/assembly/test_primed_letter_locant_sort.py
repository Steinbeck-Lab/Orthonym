"""Bug A (audit 2026-09-03): ``_enrich_complex_ring_with_subs`` sorted a
substituent group's locants with a key that assumed the base of a primed
locant is an int. A fused component's letter locant arrives as ``('8a', "'")``
and the key raised ``TypeError: '<' not supported between 'str' and 'int'``;
the molecule then abstained inside a broad except.

Blue Book (the Blue Book): "Primed locants are placed
immediately after the corresponding unprimed locants in a set arranged in
ascending order; locants consisting of a number and a lower-case letter with
or without primes as 4a and 4′a..." and (:3207) "4a is lower than 4′a".
The old key also put every unprimed locant before every primed one, which
that sentence forbids.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import composer

WITNESS = "C[C@@H]1CCC2C(C)(C)[C@H](O)CC[C@]2(C)[C@@]12Cc1c(O)cc3c(c1O2)CNC3=O"  # a dev split


def test_letter_locant_inside_a_primed_tuple_sorts_without_error():
    locs = [(2, "'"), (5, "'"), (5, "'"), ("8a", "'")]
    assert sorted(locs, key=composer._primed_locant_sort_key) == locs


def test_ascending_set_with_primed_right_after_its_unprimed_twin():
    #: one ascending set; a primed locant follows its unprimed twin.
    locs = [(6, "'"), 3, (2, "'"), 5, (5, "'"), ("8a", "'"), "8a", 8]
    assert sorted(locs, key=composer._primed_locant_sort_key) == [
        (2, "'"), 3, 5, (5, "'"), (6, "'"), 8, "8a", ("8a", "'")]


def test_render_keeps_letter_and_prime():
    assert composer._render_primed_locant(("8a", "'")) == "8a'"
    assert composer._render_primed_locant(3) == "3"


@pytest.mark.opsin_gate   # production config: the OPSIN validity gate is ON
def test_witness_no_longer_raises_and_never_ships_a_wrong_molecule(monkeypatch):
    from orthonym import Orthonym
    from orthonym.errors import is_failure_name
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from tests.support.jars import jar_or_skip
    monkeypatch.setenv("ORTHONYM_STRICT", "1")          # a TypeError would surface here
    name = Orthonym(style="pin").name(WITNESS)
    if is_failure_name(name):
        return                                             # an honest abstention is allowed
    jar_or_skip()
    back = opsin_parse(name)
    assert back, name
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
    assert key(back) == key(WITNESS), (name, back)
