"""The name_substituent memo key must distinguish a terminal formyl-on-N fragment
(formamido) from an acyl-bridge fragment whose carbon bonds the rest of the
molecule (carbamoyl); both render MolFragmentToSmiles == 'NC=O' rooted at N."""
import os
os.environ["ORTHONYM_MEMO"] = "verify"
from rdkit import Chem
from orthonym.assembly import memo
from orthonym import name_compound


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
