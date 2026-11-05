"""lane/l17 item 13 (speed): the structural half of the ``name_substituent`` memo key is kept for the naming scope, and
the ambient state of the keys is read without an import statement per call. Names and tiers are unchanged; these
tests pin that a repeated question is served from the scope, that a changed state is not, and that the key and the
state are the values they were."""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_key, memo
from orthonym.assembly import substituent_enumerator as se
from orthonym.assembly.nested_memo import ambient_naming_state, renaming_state
from orthonym.perception.molcache import MolCacheMismatchError

pytestmark = pytest.mark.unit

SMILES = "CC[C@H](C)[C@@H](C(=O)NCC(=O)N[C@@H](CCCN=C(N)N)C(=O)O)NC(=O)[C@@H]1CCCN1C(=O)C"
FRAG = list(range(11, 21))           # holds the stereocentre 12
ATTACH = 12


def _mol():
    mol = Chem.MolFromSmiles(SMILES)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


@pytest.fixture
def scope():
    token = memo.push_scope()
    try:
        yield
    finally:
        memo.pop_scope(token)


@pytest.fixture
def smiles_calls(monkeypatch):
    """The calls of RDKit's fragment SMILES made by the key."""
    calls = []
    real = Chem.MolFragmentToSmiles

    def counting(*args, **kwargs):
        calls.append(kwargs.get("atomsToUse"))
        return real(*args, **kwargs)
    monkeypatch.setattr(fragment_key.Chem, "MolFragmentToSmiles", counting)
    return calls


def test_a_repeated_question_is_served_from_the_scope(scope, smiles_calls):
    mol = _mol()
    first = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    again = se._substituent_memo_key(mol, list(FRAG), ATTACH, False, False)
    assert again == first
    assert len(smiles_calls) == 1


def test_a_hit_leaves_the_output_order_properties_a_call_would_leave(scope, smiles_calls):
    mol = _mol()
    se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    left = {k: mol.GetProp(k) for k in ("_smilesAtomOutputOrder", "_smilesBondOutputOrder")}
    se._substituent_memo_key(mol, [0, 1, 2, 3], 1, False, False)           # another fragment overwrites them
    assert {k: mol.GetProp(k) for k in left} != left
    se._substituent_memo_key(mol, FRAG, ATTACH, False, False)               # a hit
    assert len(smiles_calls) == 2
    assert {k: mol.GetProp(k) for k in left} == left
    computed = mol.GetProp("__computedProps")
    assert "_smilesAtomOutputOrder" in computed and "_smilesBondOutputOrder" in computed


def test_a_changed_cip_label_is_not_served_the_earlier_key(scope, smiles_calls):
    mol = _mol()
    centre = next(a for a in mol.GetAtoms() if a.GetIdx() in FRAG and a.HasProp("_CIPCode"))
    before = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    centre.SetProp("_CIPCode", "S" if centre.GetProp("_CIPCode") == "R" else "R")
    after = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    assert len(smiles_calls) == 2
    assert after != before


def test_an_edit_of_the_graph_is_not_served_the_earlier_key(scope, smiles_calls):
    mol = _mol()
    before = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    atom = mol.GetAtomWithIdx(FRAG[-1])
    atom.SetIsotope(13)                                                    # a change of the graph in place
    after = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    assert len(smiles_calls) == 2
    assert after != before


def test_a_different_fragment_or_attachment_is_not_served_the_earlier_key(scope, smiles_calls):
    mol = _mol()
    keys = {se._substituent_memo_key(mol, FRAG, ATTACH, False, False),
            se._substituent_memo_key(mol, FRAG, ATTACH + 1, False, False),
            se._substituent_memo_key(mol, FRAG[:-1], ATTACH, False, False)}
    assert len(keys) == 3
    assert len(smiles_calls) == 3


def test_a_second_molecule_object_of_the_same_graph_has_its_own_entry(scope, smiles_calls):
    first, second = _mol(), _mol()
    a = se._substituent_memo_key(first, FRAG, ATTACH, False, False)
    b = se._substituent_memo_key(second, FRAG, ATTACH, False, False)
    assert a == b and len(smiles_calls) == 2
    assert second.GetProp("_smilesAtomOutputOrder") == first.GetProp("_smilesAtomOutputOrder")


def test_the_tier_flags_are_in_the_key_beside_the_cached_half(scope, smiles_calls):
    mol = _mol()
    assert se._substituent_memo_key(mol, FRAG, ATTACH, False, False) != se._substituent_memo_key(
        mol, FRAG, ATTACH, True, False)
    assert se._substituent_memo_key(mol, FRAG, ATTACH, False, False) != se._substituent_memo_key(
        mol, FRAG, ATTACH, False, True)
    assert len(smiles_calls) == 1


def test_outside_a_scope_nothing_is_kept(smiles_calls):
    mol = _mol()
    first = se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    assert se._substituent_memo_key(mol, FRAG, ATTACH, False, False) == first
    assert len(smiles_calls) == 2


def test_a_read_write_molecule_is_not_kept(scope, smiles_calls):
    mol = Chem.RWMol(_mol())
    se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    assert len(smiles_calls) == 2


def test_the_cached_key_is_the_unrestricted_key(scope):
    """Every question of a panel gives the key the code without the cache gives (and a mixture is keyed unrooted)."""
    for smi in (SMILES, "CC(=O)N[C@@H](C)C(=O)O.C/C=C/C(=O)[O-].[Na+]", "C/C=C\\C=C/CC(N)=O"):
        mol = Chem.MolFromSmiles(smi)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        n = mol.GetNumAtoms()
        for attach in range(min(n, 6)):
            others = [i for i in range(n) if i != attach]
            for frag in ([attach], [attach] + others[:3], others[: max(2, n // 2)], others):
                for att in (attach, n - 1):
                    cached = fragment_key.structural_parts(mol, frag, att)
                    plain = fragment_key.structural_parts_unrestricted(mol, frag, att)
                    assert cached == plain


def test_verify_mode_finds_a_stale_entry(scope, monkeypatch):
    """With ORTHONYM_MOLCACHE=verify a hit is recomputed and compared: an entry whose stamp did not see an edit is caught."""
    from orthonym.perception import molcache
    mol = _mol()
    se._substituent_memo_key(mol, FRAG, ATTACH, False, False)
    scope_dict = memo._cache_var.get()
    parts_key = next(k for k in scope_dict if isinstance(k, tuple) and k[0] == fragment_key._NS_PARTS)
    stored = scope_dict[parts_key]
    scope_dict[parts_key] = (stored[0], ("tampered",) + tuple(stored[1][1:]))
    monkeypatch.setattr(molcache, "_MODE", "verify")
    with pytest.raises(MolCacheMismatchError):
        se._substituent_memo_key(mol, FRAG, ATTACH, False, False)


# ---- the ambient state: the value of the function with an import statement per call ---------------------------------


def _reference_ambient():
    """``nested_memo.ambient_naming_state`` as it was before the modules were imported once (frozen reference)."""
    from orthonym.assembly import fragment_naming as _fn
    from orthonym.decomposition.engine import nacyl_float_refusing
    from orthonym.metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx,
                                             full_coverage_ctx, general_fallback_ctx)
    from orthonym.namer import _ALT_PARENT_RESCUE
    from orthonym.routing import dispatch_table as _dt
    from orthonym.assembly.group_splitting import _IN_SPLIT_PROBE
    from orthonym.assembly.locant_omission import (_ISOTOPE_PARENT_POSITIONAL, _ISOTOPIC_NAMING_SCOPE,
                                                   forced_locant_reason)
    from orthonym.assembly.memo import book_forms_var
    from orthonym.assembly.substituent_enumerator import _gate_is_reentrant
    guard = _fn._fragment_guard
    visited = getattr(guard, 'visited', None)
    return (frozenset(visited) if visited else frozenset(),
            best_effort_ctx.get(), general_fallback_ctx.get(),
            allow_aromatic_general_ctx.get(), full_coverage_ctx.get(),
            nacyl_float_refusing(), _dt._PEPTIDE_SUBST_ACTIVE, _dt._PEPTIDE_NAME_STANDS,
            _gate_is_reentrant(), getattr(_IN_SPLIT_PROBE, 'active', False),
            forced_locant_reason(), _ISOTOPIC_NAMING_SCOPE.get(), _ISOTOPE_PARENT_POSITIONAL.get(),
            book_forms_var.get(), getattr(_ALT_PARENT_RESCUE, 'active', False))


def test_the_ambient_state_is_the_reference_in_the_idle_state():
    assert ambient_naming_state() == _reference_ambient()


def test_the_ambient_state_follows_every_component_it_reads():
    from orthonym.assembly import fragment_naming as _fn
    from orthonym.metrics import provenance
    from orthonym.assembly.memo import book_forms_var
    from orthonym.routing import dispatch_table as dt
    seen = {ambient_naming_state()}
    for var in (provenance.best_effort_ctx, provenance.general_fallback_ctx, provenance.allow_aromatic_general_ctx,
                provenance.full_coverage_ctx, book_forms_var):
        token = var.set(not var.get())
        try:
            state = ambient_naming_state()
            assert state == _reference_ambient()
            seen.add(state)
        finally:
            var.reset(token)
    before = getattr(_fn._fragment_guard, "visited", None)
    _fn._fragment_guard.visited = {"CCO"}
    try:
        state = ambient_naming_state()
        assert state == _reference_ambient() and state[0] == frozenset({"CCO"})
        seen.add(state)
    finally:
        _fn._fragment_guard.visited = before
    saved = dt._PEPTIDE_SUBST_ACTIVE
    dt._PEPTIDE_SUBST_ACTIVE = not saved
    try:
        state = ambient_naming_state()
        assert state == _reference_ambient()
        seen.add(state)
    finally:
        dt._PEPTIDE_SUBST_ACTIVE = saved
    assert len(seen) == 8                    # every change of one component is a state of its own
    assert renaming_state() == (False,) + ambient_naming_state()[1:8]


def test_the_ambient_state_reads_a_name_on_its_module_at_every_call(monkeypatch):
    """The names are looked up on the module at every call (as the ``from`` imports it had looked them up)."""
    from orthonym.decomposition import engine
    monkeypatch.setattr(engine, "nacyl_float_refusing", lambda: "patched")
    assert ambient_naming_state()[5] == "patched"


def test_the_fragment_naming_key_is_unchanged_in_its_ambient_components():
    from orthonym.assembly import substituent_naming as sn
    mol = _mol()
    key = sn._nsf_memo_key(mol, FRAG, ATTACH, None)
    assert key[0] == id(mol) and key[3] == tuple(FRAG)
    assert key[10:] == _reference_ambient()[1:14]
