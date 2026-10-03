"""Unit tests for the substituted mono-6-oxo purine (hypoxanthine/guanine
family) naming engine.

Both RT-exact targets below were confirmed via OPSIN 2.9.0 round-trip
(name -> SMILES -> InChIKey, identical to the input SMILES's InChIKey) before
this engine was written:

    9-methylguanine Cn1cnc2c1nc(N)[nH]c2=O
        -> 2-amino-9-methyl-1,9-dihydro-6H-purin-6-one
    9-methylhypoxanthine Cn1cnc2c1[nH]cnc2=O (also O=c1[nH]cnc2n(C)cnc12)
        -> 9-methyl-1,9-dihydro-6H-purin-6-one

The C2-amino (guanine family) is placed as an ordinary `2-amino` PREFIX --
never as the shared fused-ring assembler's amino SUFFIX, which would silently
drop the C6 oxo (a wrong-molecule defect; see module docstring in
rules/purine.py::name_oxo_purine). The engine declines the bare parent
(hypoxanthine or guanine with no OTHER substituent), a 2,6-dione (caffeine
family; purine_oxo.py owns that), any extra ring oxo (8-oxo/trione), and any
substituent outside plain alkyl/halogen/bare-amino (fail-closed).
"""
import pytest
from rdkit import Chem

from orthonym.rules.purine import name_oxo_purine

pytestmark = pytest.mark.unit


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def test_9_methylguanine():
    mol = _mol("Cn1cnc2c1nc(N)[nH]c2=O")
    assert Chem.MolToInchiKey(mol) == "UUWJNBOCAPUTBK-UHFFFAOYSA-N"  # 9-methylguanine
    assert name_oxo_purine(mol) == "2-amino-9-methyl-1,9-dihydro-6H-purin-6-one"


@pytest.mark.parametrize("smiles,expected", [
    # N3-H tautomer: the 1,9-dihydro spelling (asserted here before) is the N1-H
    # tautomer -- same standard InChIKey, different fixed-H InChI (OPSIN 2.9.0).
    ("Cn1cnc2c1[nH]cnc2=O", "9-methyl-3,9-dihydro-6H-purin-6-one"),
    ("O=c1[nH]cnc2n(C)cnc12", "9-methyl-1,9-dihydro-6H-purin-6-one"),
])
def test_9_methylhypoxanthine(smiles, expected):
    mol = _mol(smiles)
    assert Chem.MolToInchiKey(mol) == "PESGUQRDJASXOR-UHFFFAOYSA-N"  # 9-methylhypoxanthine
    assert name_oxo_purine(mol) == expected


def test_bare_hypoxanthine_is_named_on_purine():
    # 'hypoxanthine' does not occur in the Blue Book (0 hits); "the PIN is 7H-purine"
    # (the Blue Book): the bare base is named here (it used to decline)
    mol = _mol("O=c1[nH]cnc2[nH]cnc12")
    assert Chem.MolToInchiKey(mol) == "FDGQSTZJBFJUBT-UHFFFAOYSA-N"
    assert name_oxo_purine(mol) == "1,9-dihydro-6H-purin-6-one"


def test_bare_guanine_is_named_on_purine():
    # 'guanine' does not occur in the Blue Book (0 hits); the C2-amino is an ordinary
    # prefix on the purine parent
    mol = _mol("Nc1nc2[nH]cnc2c(=O)[nH]1")
    assert Chem.MolToInchiKey(mol) == "UYTPUPDQBNUYGX-UHFFFAOYSA-N"
    assert name_oxo_purine(mol) == "2-amino-1,9-dihydro-6H-purin-6-one"


@pytest.mark.parametrize("smiles,expected", [
    # the saturated six-ring N is N3, not N1: the 1,7-/1,9-dihydro spelling would be
    # another tautomer (OPSIN 2.9.0 parse: same standard InChIKey, different fixed-H
    # InChI). indicated/added hydrogen at the atoms that carry it.
    ("Nc1nc(=O)c2[nH]cnc2[nH]1", "2-amino-3,7-dihydro-6H-purin-6-one"),
    ("O=c1nc[nH]c2nc[nH]c12", "3,7-dihydro-6H-purin-6-one"),
    ("Nc1nc(=O)c2[nH]c(C)nc2[nH]1", "2-amino-8-methyl-3,7-dihydro-6H-purin-6-one"),
])
def test_n3_hydrogen_tautomer(smiles, expected):
    assert name_oxo_purine(_mol(smiles)) == expected


def test_caffeine_declines():
    # a 2,6-dione is a DIFFERENT parent (purine_oxo.py owns it); this engine's
    # core pins exactly one ring C=O (C6), so a second ring =O at C2 is typed
    # 'oxo' by the shared identifier -- not an accepted type -> decline
    mol = _mol("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
    assert Chem.MolToInchiKey(mol) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"  # caffeine
    assert name_oxo_purine(mol) is None


def test_2_chloro_9_methylhypoxanthine():
    # halogen-substituted mono-6-oxo purine (RT-verified via OPSIN 2.9.0:
    # name -> SMILES -> InChIKey, identical to the input's; the input carries H on N3,
    # so the fixed-H InChI matches only the 3,9-dihydro spelling, not 1,9-dihydro)
    mol = _mol("Cn1cnc2c1[nH]c(Cl)nc2=O")
    assert Chem.MolToInchiKey(mol) == "GFDFINQSCGJTCD-UHFFFAOYSA-N"
    assert name_oxo_purine(mol) == "2-chloro-9-methyl-3,9-dihydro-6H-purin-6-one"


def test_uric_acid_like_trione_declines():
    # extra C8=O (uric acid / trione family) -> not this engine's scope
    mol = _mol("Cn1c(=O)c2[nH]c(=O)[nH]c2n(C)c1=O")
    assert Chem.MolToInchiKey(mol) == "OTSBKHHWSQYEHK-UHFFFAOYSA-N"
    assert name_oxo_purine(mol) is None


def test_non_purine_declines():
    assert name_oxo_purine(_mol("c1ccccc1")) is None
    assert name_oxo_purine(_mol("CCO")) is None


def test_none_input():
    assert name_oxo_purine(None) is None


def test_unclassifiable_substituent_fails_closed():
    # a boronic-acid ring substituent is NOT typed by _identify_fused_substituent
    # -> fail closed rather than silently omit it and name a different molecule
    mol = _mol("O=c1[nH]cnc2n(B(O)O)cnc12")
    assert name_oxo_purine(mol) is None


@pytest.mark.opsin_gate
def test_9_methylguanine_end_to_end():
    from orthonym import Orthonym
    smi = "Cn1cnc2c1nc(N)[nH]c2=O"
    assert Orthonym().name(smi) == "2-amino-9-methyl-1,9-dihydro-6H-purin-6-one"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # 9-methylhypoxanthine: H on N-1
    ("Cn1cnc2c1nc[nH]c2=O", "9-methyl-1,9-dihydro-6H-purin-6-one"),
    # the N-3 tautomer: its indicated hydrogen at the ketone and hydro prefixes at
    # the saturated positions, the Blue Book), so '3,9-dihydro'.
    # This row used to expect the N-1 name: same standard InChIKey (mobile H), but
    # OPSIN's structure for it is the N-1 tautomer (canonical SMILES differ), not
    # the drawn one (TRIAGE j12 finding 5)
    ("Cn1cnc2c1[nH]cnc2=O", "9-methyl-3,9-dihydro-6H-purin-6-one"),
])
def test_9_methylhypoxanthine_end_to_end(smi, expected):
    from orthonym import Orthonym
    assert Orthonym().name(smi) == expected


@pytest.mark.opsin_gate
def test_bare_guanine_unchanged_end_to_end():
    from orthonym import Orthonym
    assert Orthonym().name("Nc1nc2[nH]cnc2c(=O)[nH]1") == "2-amino-1,9-dihydro-6H-purin-6-one"


@pytest.mark.opsin_gate
def test_bare_hypoxanthine_unchanged_end_to_end():
    from orthonym import Orthonym
    assert Orthonym().name("O=c1[nH]cnc2[nH]cnc12") == "1,9-dihydro-6H-purin-6-one"


@pytest.mark.opsin_gate
def test_caffeine_unchanged_end_to_end():
    from orthonym import Orthonym
    assert Orthonym().name("Cn1c(=O)c2c(ncn2C)n(C)c1=O") == \
        "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"
