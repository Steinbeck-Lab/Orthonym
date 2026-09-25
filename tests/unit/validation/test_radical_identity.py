"""validation/radical_identity.py -- the radical-identity rule every gate shares.

Each pair below shares ONE full standard InChIKey (verified in the test), yet
the two structures are different molecules: the key encodes neither radical
electrons nor bond order. A gate that accepts on key equality alone would pass
a wrong name for any of them.
"""
import pytest
from rdkit import Chem

from orthonym.namer import _self_consistency_verdict
from orthonym.validation.radical_identity import radical_identity_verdict, radical_profile

KEY_TWINS = [
    ("[CH2][CH2]", "C=C"),                          # ethane-1,2-diyl vs ethene
    ("[O]c1ccc([O])cc1", "O=C1C=CC(=O)C=C1"),       # bis(oxyl) vs p-benzoquinone
    ("C[N][O]", "CN=O"),                            # vs nitrosomethane
    ("CN(C)[O]", "C[N+](C)[O-]"),                   # aminoxyl vs its charge form
    ("CC(=O)[NH]", "CC([O])=N"),                    # acetamidyl vs ethanimidoyloxyl
    ("[CH2]C=CC", "C=C[CH]C"),                      # but-2-en-1-yl vs but-3-en-2-yl
    ("C=C", "[CH2][CH2]"),                          # closed-shell input, radical parse
]


@pytest.mark.parametrize("inp,parsed", KEY_TWINS)
def test_key_twins_share_a_key_but_are_rejected(inp, parsed):
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(inp)) == Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert radical_identity_verdict(inp, parsed) == "mismatch"


@pytest.mark.parametrize("inp,parsed", KEY_TWINS)
def test_self01_rejects_key_twins(inp, parsed):
    assert _self_consistency_verdict(inp, parsed) == "mismatch"


@pytest.mark.parametrize("inp,parsed", [
    ("C[O]", "[O]C"),                               # same radical, other spelling
    ("CC1(C)CCCC(C)(C)N1[O]", "CC1(N(C(CCC1)(C)C)[O])C"),   # TEMPO as OPSIN writes it
    ("[O][O]", "O=O"),                              # dioxygen: named exemption
])
def test_same_radical_accepted(inp, parsed):
    assert radical_identity_verdict(inp, parsed) == "ok"


@pytest.mark.parametrize("inp,parsed", [
    ("CCO", "OCC"),                                 # closed shell both sides
    ("O=[As]([O-])([O-])O.[Pb+2]", "[As](O)([O-])([O-])=O.[Pb+2]"),  # metal cation, not a radical
    ("Cl[Sn]Cl", "[Sn+2].[Cl-].[Cl-]"),            # metal valence artefact
])
def test_not_applicable_without_a_chemical_radical(inp, parsed):
    assert radical_identity_verdict(inp, parsed) == "n/a"


def test_unreadable_parse_of_a_radical_fails_closed():
    assert radical_identity_verdict("C[O]", "not a smiles") == "mismatch"


def test_profile_ignores_metals_keeps_metalloids():
    assert radical_profile(Chem.MolFromSmiles("[Pb+2]")) == ()
    assert radical_profile(Chem.MolFromSmiles("[SiH3]")) == (("Si", 1),)
