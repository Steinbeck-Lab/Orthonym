"""P-66.1.6.1.1.3 - N-substituted urea substituent prefix (W2F p6 Task 1).

BB P-66.1.6.1.1.3 (BlueBookV2.md:33338,33354): urea substituent = (R-carbamoyl)amino;
'ureido'/'3-methylureido' NOT recommended. Distal-N substituents are cited inside the
carbamoyl acyl name, enclosed.

NOTE (re-anchor at HEAD 9a33eb63): the plan authored a helper importing
``get_fg_prefix_form`` from ``orthonym.assembly.substituent_prefix_forms`` with
arg order ``(mol, fg, atoms, None)``. The real dispatcher is
``get_substituent_prefix_form(fg_name, mol, atoms, principal_chain)`` in that module,
and FG detection is ``detect_functional_groups`` (not ``find_functional_groups``).
This test uses the real API.
"""
import pytest
from rdkit import Chem
from orthonym.assembly.substituent_prefix_forms import get_substituent_prefix_form
from orthonym.perception.functional_groups import detect_functional_groups


def _prefix(smiles, fg="urea"):
    mol = Chem.MolFromSmiles(smiles)
    fgs = detect_functional_groups(mol)
    atoms = (fgs.get(fg) or [None])[0]
    # prefix-only context (principal_chain=None) mirrors the ring-parent caller
    return get_substituent_prefix_form(fg, mol, atoms, None), mol


class TestUreaPrefixP661613:
    def test_unsubstituted_distal_n_unchanged(self):
        # regression: bare urea substituent stays 'carbamoylamino'
        name, _ = _prefix("NC(=O)NCCC(=O)O")
        assert name == "carbamoylamino"

    def test_n_methyl_distal(self):
        name, _ = _prefix("CNC(=O)NCCC(=O)O")
        assert name == "(methylcarbamoyl)amino"

    def test_n_n_dimethyl_distal(self):
        name, _ = _prefix("CN(C)C(=O)Nc1ccccc1C(=O)O")
        assert name == "(dimethylcarbamoyl)amino"

    def test_unnameable_distal_fails_closed(self):
        # distal N bearing an un-nameable fragment -> None (never a truncated prefix)
        name, _ = _prefix("O=C(O)CCNC(=O)N[Si](C)(C)C")  # N-silyl distal
        assert name is None
