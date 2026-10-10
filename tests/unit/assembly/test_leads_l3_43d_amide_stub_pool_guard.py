"""Leads program L3, item 43d: an empty candidate pool is never dereferenced.

``_assemble_amide_name`` returns the stub ``'amide'`` when it cannot name a lactam or macrocyclic
amide; the inline amide branch of ``_assemble_name_impl`` offered it to the candidate pool, the pool
rejected it (the ratio floor of a large molecule), and ``return pool.best.name`` dereferenced
None: ``AttributeError: 'NoneType' object has no attribute 'name'`` at the pin tier (4 of 500 a dev split
rows, 157 of the 2,148 amide-like rows of the v1.0.6 evals; ``assemble_name`` and the namer's
catch-all swallowed it, and the namer's recovery then ran). No nomenclature rule decides this; it is
the engine contract that a producer exception must never decide an outcome.

The five ``pool.best.name`` sites of ``_assemble_name_impl`` now fall through when the pool is
empty. The stub itself stays the truthy string it was: where the pool ACCEPTS it, ``assemble_name``
returns the stub-only name and ``Orthonym._name_impl`` answers with the decomposition fallback, which
names two pin rows of those 2,148 (below); a falsy decline skips that fallback (measured: two pin
rows lost, one gained), so the producer contract is a separate item (needs ``_name_impl`` and
``handlers/amide.py``, outside lane L3).

The four a dev split rows and the tricyclic aza lactam of the user report are named below.
"""
import sys

import pytest
from rdkit import Chem

import orthonym.assembly.composer as composer
from orthonym import Orthonym
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

ROWS = [
    # the four a dev split rows whose amide branch raised at composer.py:1391
    "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN([C@@H]2CCCNC2=O)C3=O)C[C@@H]1O",
    "C=C1[C@@H](C)[C@H]2[C@H](Cc3c[nH]c4ccccc34)NC(=O)[C@]23C(=O)C[C@H]2C(=O)C(O)=C(C)"
    "[C@@H]2[C@@H](C)C/C=C/[C@H]3[C@@H]1O",
    "COc1ccc([C@@]23Oc4cc(OC)cc(OC)c4[C@]2(O)[C@H](O)[C@H](C(=O)N(C)C)[C@H]3c2ccccc2)cc1",
    "CC[C@@H]1C(C)=C[C@@H]2/C=C/C[C@H](C)/C=C(\\C)CC/C=C/C(=O)[C@]23C(=O)N"
    "[C@@H]([C@H](C)c2c[nH]c4ccccc24)[C@H]13",
    # the tricyclic aza SMILES of the user report
    "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O",
]

#: two rows the stub feeds to the decomposition fallback: pin_verified on main, and still
PIN_ROWS_THE_STUB_FEEDS = [
    ("CC(O)C(NC(=O)C1CCCN1C)C1OC(SCCOC(=O)c2ccccc2N)C(O)C(O)C1O",
     "2-({3,4,5-trihydroxy-6-[2-hydroxy-1-(1-methylpyrrolidine-2-carboxamido)propyl]oxan-2-yl}"
     "sulfanyl)ethyl 2-aminobenzoate"),
    ("CN[C@H](C(=O)O[C@@H]1CC[C@@](C)(/C=C/c2ccc3c(c2O)[C@@](O)(c2ccccc2)[C@H](OC)C(=O)N3)"
     "C[C@@H]1C)C(C)C",
     "(1R,2S,4R)-4-{(1E)-2-[(3S,4S)-4,5-dihydroxy-3-methoxy-2-oxo-4-phenyl-1,2,3,4-"
     "tetrahydroquinolin-6-yl]ethen-1-yl}-2,4-dimethylcyclohexyl (2S)-3-methyl-2-(methylamino)butanoate"),
]


def _features(smiles):
    eng = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(smiles)
    feats = eng._perceive(mol, smiles, Chem.MolToSmiles(mol))
    eng._classify(feats)
    return feats


@pytest.mark.parametrize("smiles", ROWS)
def test_assemble_name_does_not_raise_when_the_pool_rejects_the_amide_stub(smiles):
    out = composer.assemble_name(_features(smiles), style="pin")
    assert isinstance(out, str)


@pytest.mark.parametrize("smiles", ROWS[:3])
def test_empty_candidate_pool_falls_through_instead_of_dereferencing_none(smiles, monkeypatch):
    """The guard on ``pool.best`` itself, independent of the producer: a candidate the pool
    rejects leaves the pool empty, and the branch falls through to the next producer."""
    monkeypatch.setattr(composer, "_assemble_amide_name", lambda features, style: "amide")
    out = composer.assemble_name(_features(smiles), style="pin")
    assert isinstance(out, str)


def _raised_attribute_errors(smiles_list, tier_engine):
    """AttributeErrors raised anywhere in the package while naming ``smiles_list``."""
    tool = next((i for i in (3, 4) if sys.monitoring.get_tool(i) is None), None)
    assert tool is not None, "no free sys.monitoring tool id"
    seen = []

    def on_raise(code, offset, exc):
        if isinstance(exc, AttributeError) and "/orthonym/" in code.co_filename:
            seen.append((code.co_filename.rsplit("/orthonym/", 1)[-1], code.co_name, str(exc)[:60]))

    sys.monitoring.use_tool_id(tool, "l3-43d")
    sys.monitoring.register_callback(tool, sys.monitoring.events.RAISE, on_raise)
    sys.monitoring.set_events(tool, sys.monitoring.events.RAISE)
    try:
        for smi in smiles_list:
            tier_engine.name_tiered(smi)
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.RAISE, None)
        sys.monitoring.free_tool_id(tool)
    return seen


def test_no_attribute_error_is_raised_inside_the_pin_tier():
    seen = _raised_attribute_errors(ROWS, Orthonym(style="pin"))
    assert not seen, seen


@pytest.mark.parametrize("smiles", ROWS)
def test_best_effort_still_names_the_row_and_reads_back(smiles):
    # the wider tier never goes down
    res = name_best_effort(smiles)
    assert res["name"] and res["source"] != "abstain", res
    assert name_is_rt_exact(res["name"], smiles), res


@pytest.mark.parametrize("smiles, pin", PIN_ROWS_THE_STUB_FEEDS)
def test_the_pin_rows_the_stub_feeds_to_the_decomposition_fallback_keep_their_name(smiles, pin):
    """Why the stub stays a truthy string: a falsy decline loses these two pin rows (the first
    version of this fix did)."""
    assert name_is_rt_exact(pin, smiles), pin
    d = Orthonym(style="pin").name_tiered(smiles)
    assert (d["name"], d["tier"]) == (pin, "pin_verified"), d
