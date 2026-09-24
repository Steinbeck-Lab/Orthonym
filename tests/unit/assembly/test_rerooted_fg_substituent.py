""" FG-capable re-rooted substituent namer (invariant-18 lever).

Names a complex FG-bearing fragment as a substituent rooted at an ARBITRARY
attachment atom, with structural free-valence numbering / —
the proper form the parent_to_prefix band-aid cannot produce. Best-effort-gated,
so the default/PIN path is byte-identical. See
internal notes
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import (
    name_substituent_fragment, _located_acyclic_alkyl_name,
    _located_fg_hetero_root, _fg_enclose,
)
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.metrics.provenance import best_effort_ctx

pytestmark = pytest.mark.unit


@pytest.fixture
def _best_effort():
    tok = best_effort_ctx.set(True)
    yield
    best_effort_ctx.reset(tok)


def _mol(smi):
    m = Chem.MolFromSmiles(smi)
    assign_stereochemistry(m)
    return m


def test_fg_tier_off_on_default_path_byte_identical():
    # Without best_effort_ctx, the FG tier must NOT fire (default/PIN gold path
    # byte-identical). A carbonyl-bearing chain rooted at a terminal C declines.
    m = _mol('OCC(=O)CCCC')
    # attach at the terminal CH3 (last carbon)
    c = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'C'][-1]
    # default path: FG tier gated off -> may still name via other tiers, but the
    # allow_functional deriver itself is inert unless invoked.
    res = _located_acyclic_alkyl_name(m, list(range(m.GetNumHeavyAtoms())), 0,
                                      allow_functional=False)
    # allow_functional=False keeps the strict contract: an -OH/=O bearing chain
    # (non-simple) declines -> None (byte-identical strict behaviour).
    assert res is None


def test_hetero_root_sulfanyl(_best_effort):
    # -S-CH2CH3 rooted at S -> ethylsulfanyl
    m = _mol('SCC')
    s = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'S'][0]
    out = _located_fg_hetero_root(m, list(range(m.GetNumHeavyAtoms())), s)
    assert out is not None and out.endswith('sulfanyl')


def test_hetero_root_amino(_best_effort):
    # -NH-CH2CH3 rooted at N -> (ethyl)amino / ethylamino
    m = _mol('NCC')
    n = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'][0]
    out = _located_fg_hetero_root(m, list(range(m.GetNumHeavyAtoms())), n)
    assert out is not None and out.endswith('amino')


def test_fg_enclose():
    assert _fg_enclose('hydroxy') == 'hydroxy'          # simple stays bare
    assert _fg_enclose('2-hydroxyethyl') == '(2-hydroxyethyl)'   # locant -> enclosed
    assert _fg_enclose('[x]oxy') == '[x]oxy'            # already enclosed


@pytest.mark.slow
def test_pantetheine_fragment_rt_exact(_best_effort):
    # The pantetheine-acyl half named as a substituent rooted at the C14 terminus
    # must round-trip to the identical structure (constitution + stereo).
    from orthonym.validation.atom_coverage import _parse_name_with_opsin_uncached as opsin
    from tests.support.jars import jar_or_skip
    jar = jar_or_skip()
    frag = 'CCCCCCCCCCCCCC[C@@H](O)C(=O)SCCNC(=O)CCN'
    m = _mol(frag)
    name = name_substituent_fragment(m, list(range(m.GetNumHeavyAtoms())), 0, [])
    assert name is not None
    tn = ('[' + name + ']' if not name[0].isdigit() else '(' + name + ')') + 'benzene'
    s = opsin(tn, jar)
    assert s is not None
    tgt = Chem.MolToInchiKey(Chem.MolFromSmiles('c1ccccc1' + frag))
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(s)) == tgt
