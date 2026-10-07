"""Lane L2 proper fix (R2): the multiplier of a substituent prefix comes from how its
writer built it, never from its punctuation.

* (a) (the Blue Book): simple components -- unsubstituted parent hydrides,
  unsubstituted prefixes, functionalized parent hydrides, retained names -- take 'di',
  'tri'; (c) (:7035) "any component which is substituted automatically requires use of
  the multiplicative forms 'bis', 'tris', etc.".
* (f) (:7104) "simple components containing brackets": 'di(bicyclo[3.2.1]octan-
  3-yl) (preferred prefix; see '; (a) (:7104) 'bis' for compound or
  complex (substituted) prefixes: 'bis(2-chloropropan-2-yl)'.
"""
import copy
import pickle

import pytest
from rdkit import Chem

from orthonym.assembly.book_prefixes import methyl_group_name
from orthonym.assembly.composer import _assemble_decorated_amino_prefix
from orthonym.assembly.naming_utils import (
    format_substituent_prefix, get_multiplier_prefix, multiplied_component)
from orthonym.assembly.prefix_derivation import (
    PrefixDerivation, PrefixName, built, carried, derivation_of)
from orthonym.assembly.substituent_naming import (
    _add_substituent_stereo, _located_acyclic_alkyl_name)
from orthonym.rules.ring_substituents import (
    _ring_assembly_substituent_prefix, _spiro_substituent_name,
    _vonbaeyer_substituent_name)


def test_the_record_is_the_text_and_survives_pickle_and_copy():
    n = built("hydroxy(phenyl)methyl", substituted=True)
    assert isinstance(n, PrefixName) and isinstance(n, str)
    assert n == "hydroxy(phenyl)methyl" and hash(n) == hash("hydroxy(phenyl)methyl")
    assert derivation_of(n) == PrefixDerivation(substituted=True)
    for clone in (pickle.loads(pickle.dumps(n)), copy.deepcopy(n)):
        assert clone == n and derivation_of(clone) == derivation_of(n)
    assert derivation_of(n + "x") is None            # a string operation drops the record
    assert derivation_of("hydroxy(phenyl)methyl") is None
    assert built(None, substituted=False) is None


def test_carried_keeps_the_record_of_the_name_it_decorates():
    base = built("hydroxy(phenyl)methyl", substituted=True)
    out = carried("(S)-hydroxy(phenyl)methyl", like=base)
    assert out == "(S)-hydroxy(phenyl)methyl"
    assert derivation_of(out) == PrefixDerivation(substituted=True)
    assert derivation_of(carried("x", like="plain")) is None


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    return m


# (writer call, its text, substituted) -- each writer decides from what it attached
WRITERS = [
    (lambda: _vonbaeyer_substituent_name(_mol("C1CC2CCC(C1)C2"), 0),
     "bicyclo[3.2.1]octan-3-yl", False),
    (lambda: _spiro_substituent_name(_mol("C1CCC2(CC1)CCCC2"), 0),
     "spiro[4.5]decan-8-yl", False),
    (lambda: _ring_assembly_substituent_prefix(
        _mol("Cc1cccc(-c2ccccc2)c1"), tuple(range(1, 13)), 1),
     "[1,1'-biphenyl]-3-yl", False),
    (lambda: methyl_group_name(["hydroxy", "phenyl"]), "hydroxy(phenyl)methyl", True),
    (lambda: methyl_group_name(["bromo", "phenyl"]), "bromo(phenyl)methyl", True),
    (lambda: methyl_group_name(["trimethylsilyl"] * 2), "bis(trimethylsilyl)methyl", True),
    (lambda: _assemble_decorated_amino_prefix([("cyclohexyl", False), ("methyl", False)]),
     "[cyclohexyl(methyl)amino]", True),
    (lambda: _assemble_decorated_amino_prefix([("propan-2-yl", False)] * 2),
     "[di(propan-2-yl)amino]", True),
    (lambda: _assemble_decorated_amino_prefix([("methyl", False)] * 2),
     "(dimethylamino)", True),                      # 'bis(dimethylamino)',:7104
    (lambda: _located_acyclic_alkyl_name(_mol("CC(C)c1ccccc1"), [0, 1, 2], 1)[0],
     "propan-2-yl", False),                         # '1,4-di(propan-2-yl)...',:25719
]


@pytest.mark.parametrize("make,text,substituted", WRITERS, ids=[w[1] for w in WRITERS])
def test_each_writer_records_whether_it_attached_a_prefix(make, text, substituted):
    # compares the one field this task adds (Tasks 4 and 10 add fields with defaults)
    name = make()
    assert name == text
    assert derivation_of(name) is not None
    assert derivation_of(name).substituted is substituted


@pytest.mark.parametrize("make,text,substituted", WRITERS, ids=[w[1] for w in WRITERS])
def test_one_decision_for_every_consumer(make, text, substituted):
    name = make()
    want = "bis" if substituted else "di"
    assert get_multiplier_prefix(2, name) == want
    assert multiplied_component(2, name, name).startswith(want)
    assert format_substituent_prefix(name, [2, 4], 2).startswith("2,4-" + want)


def test_the_stereo_writer_keeps_the_record():
    m = _mol("O[C@@H](c1ccccc1)c1ccncc1")
    name = _add_substituent_stereo(m, list(range(8)), methyl_group_name(["hydroxy", "phenyl"]),
                                   attach_idx=1)
    assert name == "(S)-hydroxy(phenyl)methyl"
    assert derivation_of(name) is not None and derivation_of(name).substituted is True
    assert format_substituent_prefix(name, [2, 6], 2) == "2,6-bis[(S)-hydroxy(phenyl)methyl]"
