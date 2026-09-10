""" P-67.2.6: recursive phosphoanhydride (P-O-P) substituent nomenclature.

The namer emits the Blue Book's method-(1) systematic form (recursive phosphoryl
nesting) for a diphosphate/triphosphate ester bridge cited as a substituent — the
class that blocked acyl-CoA (Orthonym named a terminal -O-P(=O)(OH)2 as
'phosphonooxy' but returned 'unknown' for any P-O-P bridge). Best-effort tier;
the PIN (diphosphoxane skeletal replacement) is a future PIN-tier build. See
internal notes
"""
import pytest
from rdkit import Chem

from orthonym.rules.phosphorus import name_phosphoanhydride_oxy_substituent as name_pa

pytestmark = pytest.mark.unit


def _o_and_parent(smi):
    """Return (mol, bridging_O_idx, parent_C_idx) for the C-O-P linkage."""
    m = Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O':
            nbrs = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
            if sorted(n.GetSymbol() for n in nbrs) == ['C', 'P']:
                c = [n for n in nbrs if n.GetSymbol() == 'C'][0]
                return m, a.GetIdx(), c.GetIdx()
    raise AssertionError("no C-O-P linkage found")


def test_terminal_monophosphate_contracts_to_phosphonooxy():
    m, o, c = _o_and_parent('OCCOP(=O)(O)O')
    assert name_pa(m, o, c) == 'phosphonooxy'


def test_diphosphate_bridge_recursive_phosphoryl():
    # -O-P(=O)(OH)-O-P(=O)(OH)2 -> [hydroxy(phosphonooxy)phosphoryl]oxy
    # (alphanumerical order: hydroxy before phosphonooxy, per P-14.5 / BB method 1).
    m, o, c = _o_and_parent('OCCOP(=O)(O)OP(=O)(O)O')
    assert name_pa(m, o, c) == '[hydroxy(phosphonooxy)phosphoryl]oxy'


def test_triphosphate_bridge_nests():
    m, o, c = _o_and_parent('OCCOP(=O)(O)OP(=O)(O)OP(=O)(O)O')
    out = name_pa(m, o, c)
    # two levels of nesting, terminal contracts to phosphonooxy
    assert out.endswith('phosphoryl]oxy')
    assert 'phosphonooxy' in out
    assert out.count('phosphoryl') == 2


def test_declines_phosphonate_p_c_bond():
    # A P-C bond (phosphonate) is a different nomenclature -> fail closed (None).
    m, o, c = _o_and_parent('OCCOP(=O)(O)C')   #...O-P(=O)(OH)-CH3: P has a C ligand
    assert name_pa(m, o, c) is None


def test_phosphoxane_pin_diphosphate():
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px
    m, o, c = _o_and_parent('OCCOP(=O)(O)OP(=O)(O)O')
    assert name_px(m, o, c) == '(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy'


def test_phosphoxane_pin_triphosphate():
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px
    m, o, c = _o_and_parent('OCCOP(=O)(O)OP(=O)(O)OP(=O)(O)O')
    assert name_px(m, o, c) == (
        '(1,3,5,5-tetrahydroxy-1,3,5-trioxo-1λ5,3λ5,5λ5-triphosphoxan-1-yl)oxy')


def test_phosphoxane_declines_single_p():
    # A single terminal phosphate is not a phosphoxane chain -> None (method-1
    # then contracts it to 'phosphonooxy').
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px
    m, o, c = _o_and_parent('OCCOP(=O)(O)O')
    assert name_px(m, o, c) is None


def test_phosphoxane_compound_ester_branch_is_enclosed():
    # P-16.3.3: a COMPOUND ester branch (its own leading locant, e.g. the
    # contracted '2-methylpropoxy') must be enclosed before the diphosphoxane
    # locant is prepended, or the two locants collide into an unparseable
    # '3-2-methylpropoxy'. Defect A,.
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px
    m, o, c = _o_and_parent('OCCOP(=O)(O)OP(=O)(O)OCC(C)C')
    out = name_px(m, o, c)
    assert out == (
        '(1,3-dihydroxy-3-(2-methylpropoxy)-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy')
    assert '3-2-methylpropoxy' not in out


def test_phosphoxane_compound_bracketed_ester_branch_is_enclosed():
    # A larger, bracket-carrying compound ester branch (an amino-acyl-decorated
    # alkoxy, as in the acyl-CoA pantetheine arm) must ALSO be enclosed; before
    # the fix this produced the OPSIN-unparseable '3-4-[...]butoxy' collision.
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px
    m, o, c = _o_and_parent(
        'OCCOP(=O)(O)OP(=O)(O)OCC(C)(C)C(O)C(=O)NCCC(=O)NCCSC(=O)C')
    out = name_px(m, o, c)
    assert '3-4-[' not in out
    assert out.startswith(
        '(3-(4-[(3-{[2-(acetylsulfanyl)ethyl]amino}-3-oxopropyl)amino]'
        '-3-hydroxy-2,2-dimethyl-4-oxobutoxy)-1,3-dihydroxy')


@pytest.mark.slow
def test_phosphoxane_compound_ester_branch_roundtrips():
    # GREEN end-to-end: the enclosed name is OPSIN-parseable and round-trips
    # to the exact input structure (was opsin_parse_failed before the fix).
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    from orthonym.rules.phosphorus import name_phosphoxane_oxy_substituent as name_px

    smi = 'OCCOP(=O)(O)OP(=O)(O)OCC(C)(C)C(O)C(=O)NCCC(=O)NCCSC(=O)C'
    m, o, c = _o_and_parent(smi)
    token = name_px(m, o, c)
    name = f"2-[{token}]ethan-1-ol"
    with jvm_slots(1, purpose="test_phosphoxane_compound_ester_branch_roundtrips"):
        result = opsin_roundtrip_check(smi, name)
    assert result['passed'] is True
    assert result['inchi_match'] is True


def test_declines_no_oxo_phosphite():
    # Trivalent P (no P=O) is a phosphite, not a phosphoryl -> None.
    m = Chem.MolFromSmiles('OCCOP(O)O')
    o = c = None
    for a in m.GetAtoms():
        if a.GetSymbol() == 'O':
            nbrs = [n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
            if sorted(n.GetSymbol() for n in nbrs) == ['C', 'P']:
                o = a.GetIdx(); c = [n for n in nbrs if n.GetSymbol() == 'C'][0].GetIdx()
                break
    assert o is not None
    assert name_pa(m, o, c) is None


@pytest.mark.slow
def test_end_to_end_diphosphate_names_on_best_effort():
    # The primary form is now the diphosphoxane PIN skeletal parent (method 2),
    # emitted on the best-effort tier; method-1 recursive-phosphoryl is the fallback.
    from orthonym import Orthonym
    eng = Orthonym(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True)
    out = eng.name('OCCOP(=O)(O)OP(=O)(O)O')
    assert out and 'unknown' not in out.lower()
    assert 'diphosphoxan-1-yl' in out and 'dioxo' in out and 'trihydroxy' in out
