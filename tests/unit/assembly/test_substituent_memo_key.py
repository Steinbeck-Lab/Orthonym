"""The name_substituent memo key must distinguish a terminal formyl-on-N fragment
(formamido) from an acyl-bridge fragment whose carbon bonds the rest of the
molecule (carbamoyl); both render MolFragmentToSmiles == 'NC=O' rooted at N."""
import os
# The memo reads ORTHONYM_MEMO once, at import: set it BEFORE importing orthonym,
# and put the environment back right after. Under xdist every worker imports every
# test module at collection, so a write that stayed in os.environ ran every child
# process the rest of the suite spawned in memo-verify mode (the same leak class
# as TRIAGE C1, where two modules switched the OPSIN gates off in every child).
_ENV_BEFORE = {"ORTHONYM_MEMO": os.environ.get("ORTHONYM_MEMO")}
os.environ["ORTHONYM_MEMO"] = "verify"
from rdkit import Chem
from orthonym.assembly import memo
from orthonym import name_compound

for _k, _v in _ENV_BEFORE.items():   # restore: see the note above the import
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


def test_no_name_substituent_verify_mismatches_on_carbamoyl_witness():
    # NOTE: the plan brief mis-cited this as "ns_mols witness #1 (a cyanine-dye
    # macrocycle)" -- re-verified against ns_spy.py directly: witness #1 (0-indexed,
    # the cyanine dye) produces ZERO name_substituent verify-mismatch events; the
    # formamido/carbamoyl collision (18 events, matching the brief's exact key
    # smi='NC=O', ext_free_valence=1, atom_cips=(None,None,None),
    # bond_cips=(None,None)) is actually produced by witness #4 (0-indexed, a
    # disulfide-bridged peptide with an Fmoc-like carbamate). Using the CORRECT
    # witness here so the test genuinely fails pre-fix / passes post-fix.
    smi = ("C[C@@H](C(=O)N[C@@H]1CSSC([C@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)"
           "[C@@H](NC1=O)CCCNC(=N)N)CC2=CC3=CC=CC=C3C=C2)CC4=CNC=N4)C(=O)O)"
           "(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)CCCCCNC(=O)C5=CC=C(C=C5)"
           "CNC(=O)OCC6C7=CC=CC=C7C8=CC=CC=C68")
    memo.reset_verify_mismatches()
    try:
        name_compound(smi)
    except Exception:
        pass  # a naming failure is fine; we only assert on the verify counter
    ns = sum(1 for (n, k) in memo._VERIFY_MISMATCHES if n == "name_substituent")
    assert ns == 0, f"name_substituent memo key still collides ({ns} events)"


def test_substituent_memo_key_separates_breadth_flags():
    # a lever: a value cached at one breadth configuration must never be served to
    # another (the strict PIN twin runs with the four flags OFF). The key must carry
    # the three flags that are not already present (best_effort is best_effort_val).
    from orthonym.assembly.substituent_enumerator import _substituent_memo_key
    from orthonym.metrics.provenance import (
        general_fallback_ctx, allow_aromatic_general_ctx, full_coverage_ctx)
    mol = Chem.MolFromSmiles("CCc1ccccc1"); frag = {0, 1}; attach = 1
    keys = set()
    for gf, aag, fc in [(False, False, False), (True, False, False),
                        (True, True, False), (True, True, True)]:
        t1 = general_fallback_ctx.set(gf)
        t2 = allow_aromatic_general_ctx.set(aag)
        t3 = full_coverage_ctx.set(fc)
        try:
            keys.add(_substituent_memo_key(
                mol, frag, attach, allow_mancude=True, best_effort_val=False))
        finally:
            full_coverage_ctx.reset(t3)
            allow_aromatic_general_ctx.reset(t2)
            general_fallback_ctx.reset(t1)
    assert len(keys) == 4


def test_fused_core_memo_key_separates_breadth_flags():
    # a lever mirror: the fused_core memo key must carry the same three flags, for
    # the same reason -- a match's index-keyed atom_mapping can differ by tier when a
    # breadth flag enables a producer the strict path lacks (2026-09-12 a lever).
    from orthonym.data.fused_heterocycles import _fused_core_memo_key
    from orthonym.metrics.provenance import (
        general_fallback_ctx, allow_aromatic_general_ctx, full_coverage_ctx)
    mol = Chem.MolFromSmiles("c1ccc2[nH]ccc2c1")  # indole
    keys = set()
    for gf, aag, fc in [(False, False, False), (True, False, False),
                        (True, True, False), (True, True, True)]:
        t1 = general_fallback_ctx.set(gf)
        t2 = allow_aromatic_general_ctx.set(aag)
        t3 = full_coverage_ctx.set(fc)
        try:
            keys.add(_fused_core_memo_key(mol))
        finally:
            full_coverage_ctx.reset(t3)
            allow_aromatic_general_ctx.reset(t2)
            general_fallback_ctx.reset(t1)
    assert len(keys) == 4
